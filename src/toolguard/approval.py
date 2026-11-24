"""Human approval for tool calls that change something outside the agent (side effects).

The gateway asks an Approver before it runs any tool whose MCP annotations say it is not read-only.
Three approvers:

  ConsoleApprover   asks in the terminal (demo: python -m toolguard.cli)
  QueueApprover     keeps a queue of pending requests that a person resolves over HTTP
                    (toolguard.app: GET /approvals, POST /approvals/{id}); no answer in time = deny
  RuleApprover      a scripted reviewer for the evaluation. It approves an email only if every recipient
                    is an address the user typed, which is what a careful person checking the request does
"""
from __future__ import annotations

import asyncio
import itertools
import re
import time
from dataclasses import dataclass, field

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


@dataclass
class ApprovalRequest:
    tool: str
    args: dict
    user_message: str
    reason: str = "tool has side effects"
    id: int = 0
    created: float = field(default_factory=time.time)


@dataclass
class Decision:
    approved: bool
    by: str
    note: str = ""
    wait_s: float = 0.0


class RuleApprover:
    name = "scripted-reviewer"

    async def decide(self, req: ApprovalRequest) -> Decision:
        to = EMAIL.findall(str(req.args.get("to", "")))
        typed = {e.lower() for e in EMAIL.findall(req.user_message)}
        ok = bool(to) and all(t.lower() in typed for t in to)
        return Decision(ok, self.name, "recipient typed by the user" if ok else "recipient not from the user")


class ConsoleApprover:
    name = "console"

    async def decide(self, req: ApprovalRequest) -> Decision:
        t0 = time.time()
        print(f"\n--- approval needed: {req.tool} ---")
        for k, v in req.args.items():
            print(f"  {k}: {str(v)[:500]}")
        ans = await asyncio.to_thread(input, "approve? [y/N] ")
        return Decision(ans.strip().lower() in {"y", "yes"}, "console", wait_s=time.time() - t0)


class QueueApprover:
    """Pending requests wait for a person. Unanswered requests are denied after `timeout_s`."""
    name = "inbox"

    def __init__(self, timeout_s: float = 300):
        self.timeout_s = timeout_s
        self.pending: dict[int, tuple[ApprovalRequest, asyncio.Future]] = {}
        self.history: list[dict] = []
        self._ids = itertools.count(1)

    async def decide(self, req: ApprovalRequest) -> Decision:
        req.id = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self.pending[req.id] = (req, fut)
        try:
            d = await asyncio.wait_for(fut, self.timeout_s)
        except TimeoutError:
            d = Decision(False, "timeout", f"no answer in {self.timeout_s:.0f}s")
        finally:
            self.pending.pop(req.id, None)
        d.wait_s = time.time() - req.created
        self.history.append({"id": req.id, "tool": req.tool, "args": req.args, "approved": d.approved, "by": d.by,
                             "note": d.note, "wait_s": round(d.wait_s, 1)})
        return d

    def resolve(self, req_id: int, approved: bool, by: str = "reviewer", note: str = "") -> bool:
        item = self.pending.get(req_id)
        if not item or item[1].done():
            return False
        item[1].set_result(Decision(approved, by, note))
        return True
