"""The guardrail gateway: every tool call the model asks for goes through here before it reaches the
MCP server, and every tool result goes through here before it reaches the model.

Checks on a call, in order (the first that fails decides):
  1. budget      total tool calls, calls per tool, tokens and wall-clock time for the episode
  2. allowlist   only the tools this deployment allows
  3. schema      arguments must match the tool's JSON schema from the MCP server
  4. policy      rules in code: email only to addresses the user typed
  5. approval    a person approves any tool whose MCP annotations say it is not read-only

Checks on a result:
  6. filter      lines an injection classifier flags are removed from untrusted tool output
  7. spotlight   untrusted output is wrapped in markers the system prompt explains

Checks on the final answer:
  8. links       links only to allow-listed domains

`enforce=False` is shadow mode: every check runs and is logged with what it would have done, but
nothing is blocked. That is how the "no guardrails" baseline is measured, and how a team would trial
new rules on live traffic before switching them on.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import jsonschema

from .approval import EMAIL, ApprovalRequest

URL = re.compile(r"https?://[^\s)\]>\"']+|\b[\w-]+\.(?:com|net|org|io)/[^\s)\]>\"']*", re.I)


@dataclass
class Limits:
    max_tool_calls: int = 10
    per_tool: dict = field(default_factory=lambda: {"send_email": 2, "search_filings": 4, "read_page": 5})
    max_tokens: int = 40_000
    max_seconds: float = 600


@dataclass
class GuardConfig:
    name: str = "guarded"
    enforce: bool = True
    allowed_tools: set | None = None            # None = every tool the server offers
    recipient_rule: bool = True                 # email only to addresses the user typed
    approval: object | None = None              # an Approver; None = no human step
    injection_filter: object | None = None      # object with .clean(text) -> (text, n_removed)
    spotlight: bool = True
    link_allowlist: tuple | None = ("sec.gov",)
    limits: Limits = field(default_factory=Limits)


@dataclass
class CallDecision:
    tool: str
    args: dict
    allowed: bool
    layer: str = ""            # which check decided, "" if all passed
    reason: str = ""
    approved_by: str = ""
    would_block: bool = False  # shadow mode: a check failed but the call ran anyway
    ms: float = 0.0


def allowed_url(u: str, domains: tuple) -> bool:
    host = re.sub(r"^https?://", "", u.lower()).split("/")[0]
    return any(host == d or host.endswith("." + d) for d in domains)


class Gateway:
    def __init__(self, tools: list, cfg: GuardConfig, user_message: str, call_tool):
        """tools: MCP Tool objects from list_tools(); call_tool: async (name, args) -> str."""
        self.tools = {t.name: t for t in tools}
        self.cfg, self.user_message, self._call = cfg, user_message, call_tool
        self.t0 = time.time()
        self.tokens = 0
        self.calls: dict[str, int] = {}
        self.log: list[CallDecision] = []

    def visible_tools(self) -> list:
        allow = self.cfg.allowed_tools
        return [t for n, t in self.tools.items() if allow is None or n in allow]

    def side_effect(self, name: str) -> bool:
        a = getattr(self.tools.get(name), "annotations", None)
        return not (a and a.readOnlyHint)       # unknown = treat as side effect

    def untrusted(self, name: str) -> bool:
        a = getattr(self.tools.get(name), "annotations", None)
        return bool(a and a.openWorldHint)

    def _check(self, name: str, args: dict) -> tuple[str, str]:
        c, lim = self.cfg, self.cfg.limits
        if sum(self.calls.values()) >= lim.max_tool_calls:
            return "budget", f"more than {lim.max_tool_calls} tool calls"
        if self.calls.get(name, 0) >= lim.per_tool.get(name, lim.max_tool_calls):
            return "budget", f"more than {lim.per_tool.get(name)} calls to {name}"
        if self.tokens > lim.max_tokens:
            return "budget", f"over {lim.max_tokens} tokens"
        if time.time() - self.t0 > lim.max_seconds:
            return "budget", f"over {lim.max_seconds:.0f}s"
        if name not in self.tools or (c.allowed_tools is not None and name not in c.allowed_tools):
            return "allowlist", f"tool {name!r} is not allowed"
        try:
            jsonschema.validate(args, self.tools[name].inputSchema)
        except jsonschema.ValidationError as e:
            return "schema", e.message[:200]
        if name == "send_email" and c.recipient_rule:
            to = EMAIL.findall(str(args.get("to", "")))
            typed = {e.lower() for e in EMAIL.findall(self.user_message)}
            if not to or any(t.lower() not in typed for t in to):
                return "policy", "email recipient was not given by the user"
        return "", ""

    async def call(self, name: str, args: dict) -> tuple[str, CallDecision]:
        t = time.perf_counter()
        args = args if isinstance(args, dict) else {}
        layer, reason = self._check(name, args)
        d = CallDecision(name, args, allowed=not layer, layer=layer, reason=reason)
        if not layer and self.side_effect(name) and self.cfg.approval is not None:
            dec = await self.cfg.approval.decide(ApprovalRequest(name, args, self.user_message))
            d.approved_by = dec.by
            if not dec.approved:
                d.allowed, d.layer, d.reason = False, "approval", f"denied by {dec.by}: {dec.note}"
        if not d.allowed and not self.cfg.enforce:
            d.allowed, d.would_block = True, True
        self.calls[name] = self.calls.get(name, 0) + 1
        if d.allowed:
            try:
                out = await self._call(name, args)
            except Exception as e:  # noqa: BLE001 - a failing tool is reported to the model, not raised
                out = f"tool error: {type(e).__name__}: {e}"[:300]
            out = self._clean_output(name, out)
        else:
            out = f"BLOCKED ({d.layer}): {d.reason}. Do not retry this call."
        d.ms = round((time.perf_counter() - t) * 1000, 1)
        self.log.append(d)
        return out, d

    def _clean_output(self, name: str, out: str) -> str:
        if not self.untrusted(name):
            return out
        f = self.cfg.injection_filter
        if f is not None and self.cfg.enforce:
            out, _ = f.clean(out)
        if self.cfg.spotlight and self.cfg.enforce:
            out = f"<<<UNTRUSTED_DOCUMENT\n{out}\nUNTRUSTED_DOCUMENT>>>"
        return out

    def check_answer(self, text: str) -> tuple[str, list[str]]:
        """Final answer: remove links outside the allow list. Returns (text shown to the user, events)."""
        doms = self.cfg.link_allowlist
        if not doms:
            return text, []
        bad = [u for u in URL.findall(text) if not allowed_url(u, doms)]
        if not bad:
            return text, []
        if not self.cfg.enforce:
            return text, ["would remove link to a domain not on the allow list"]
        for u in bad:
            text = text.replace(u, "[link removed]")
        return text, ["removed link to a domain not on the allow list"]

    def audit(self) -> list[dict]:
        return [json.loads(json.dumps(d.__dict__, default=str)) for d in self.log]
