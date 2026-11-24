"""Approval inbox: run the agent behind an HTTP API, with a person approving side effects in a browser.

    uvicorn toolguard.app:app --port 8100        (PYTHONPATH=src, Ollama running, pipeline data built)
    open http://localhost:8100                   the inbox page

    POST /ask              {"question": "...", "model": "qwen2.5:3b"}  -> {"run_id": 1}
    GET  /runs/{id}        status, answer, and the audit log of every tool call and decision
    GET  /approvals        requests waiting for a person
    POST /approvals/{id}   {"approve": true, "by": "dana", "note": "..."}
    GET  /metrics          counts of calls, blocks by layer, approvals, and tokens (Prometheus format)

A request nobody answers within TOOLGUARD_APPROVAL_TIMEOUT seconds (default 300) is denied.
"""
from __future__ import annotations

import asyncio
import itertools
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field

from .agent import mcp_session, run_episode
from .approval import QueueApprover
from .gateway import GuardConfig

DEFAULT_MODEL = os.environ.get("TOOLGUARD_MODEL", "qwen2.5:3b")
approver = QueueApprover(timeout_s=float(os.environ.get("TOOLGUARD_APPROVAL_TIMEOUT", "300")))
runs: dict[int, dict] = {}
_ids = itertools.count(1)
state: dict = {}

CALLS = Counter("toolguard_tool_calls_total", "Tool calls the model asked for", ["tool", "decision"])
BLOCKS = Counter("toolguard_blocks_total", "Tool calls blocked, by the check that blocked them", ["layer"])
TOKENS = Counter("toolguard_tokens_total", "Model tokens", ["kind"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with mcp_session() as (session, tools):
        state["session"], state["tools"] = session, tools
        state["lock"] = asyncio.Lock()          # one stdio session: run one episode at a time
        yield


app = FastAPI(title="Agent approval inbox", lifespan=lifespan)


class Ask(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    model: str = DEFAULT_MODEL


class Verdict(BaseModel):
    approve: bool
    by: str = "reviewer"
    note: str = ""


def _config() -> GuardConfig:
    filt = state.get("filter")
    if filt is None and os.environ.get("TOOLGUARD_FILTER", "1") == "1":
        try:
            from run_eval import trained_filter
            filt = state["filter"] = trained_filter()
        except Exception:  # noqa: BLE001 - the filter needs the mirrored training data
            filt = None
    return GuardConfig("inbox", approval=approver, injection_filter=filt)


async def _run(run_id: int, q: Ask) -> None:
    async with state["lock"]:
        runs[run_id]["status"] = "running"
        try:
            ep, gw = await run_episode(state["session"], state["tools"], q.question, q.model, _config())
            runs[run_id].update(status="done", answer=ep.answer_shown, answer_events=ep.answer_events,
                                audit=gw.audit(), tokens=ep.input_tokens + ep.output_tokens,
                                seconds=round(ep.wall_s, 1), stopped=ep.stopped)
            for c in gw.log:
                CALLS.labels(tool=c.tool, decision="allowed" if c.allowed else "blocked").inc()
                if c.layer and not c.allowed:
                    BLOCKS.labels(layer=c.layer).inc()
            TOKENS.labels(kind="input").inc(ep.input_tokens)
            TOKENS.labels(kind="output").inc(ep.output_tokens)
        except Exception as e:  # noqa: BLE001
            runs[run_id].update(status="error", error=f"{type(e).__name__}: {e}"[:300])


@app.post("/ask")
async def ask(q: Ask) -> dict:
    run_id = next(_ids)
    runs[run_id] = {"id": run_id, "question": q.question, "model": q.model, "status": "queued"}
    asyncio.create_task(_run(run_id, q))
    return {"run_id": run_id}


@app.get("/runs/{run_id}")
def get_run(run_id: int) -> dict:
    if run_id not in runs:
        raise HTTPException(404, "no such run")
    return runs[run_id]


@app.get("/approvals")
def pending() -> list[dict]:
    return [{"id": r.id, "tool": r.tool, "args": r.args, "user_message": r.user_message, "reason": r.reason,
             "waiting_s": round(time.time() - r.created, 1)}
            for r, _ in approver.pending.values()]


@app.post("/approvals/{req_id}")
def decide(req_id: int, v: Verdict) -> dict:
    if not approver.resolve(req_id, v.approve, v.by, v.note):
        raise HTTPException(404, "no such pending request")
    return {"ok": True}


@app.get("/approvals/history")
def history() -> list[dict]:
    return approver.history


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


PAGE = """<!doctype html><meta charset=utf-8><title>Approval inbox</title>
<style>body{font:15px system-ui;margin:24px;max-width:900px}textarea{width:100%;height:60px}
.card{border:1px solid #ccc;border-radius:8px;padding:12px;margin:10px 0}pre{white-space:pre-wrap;background:#f6f6f6;padding:8px}
button{margin-right:8px;padding:6px 14px}</style>
<h1>Agent approval inbox</h1>
<p>Ask the filing agent a question. Any tool call with side effects waits here for you.</p>
<textarea id=q>What is 3M's FY2018 capital expenditure in USD millions? Then email the answer to me@example.com.</textarea>
<button onclick=ask()>Ask</button><span id=run></span>
<h2>Waiting for approval</h2><div id=pending>none</div>
<h2>Run</h2><pre id=out></pre>
<script>
let rid=null;
async function ask(){const r=await fetch('/ask',{method:'POST',headers:{'content-type':'application/json'},
 body:JSON.stringify({question:document.getElementById('q').value})});rid=(await r.json()).run_id;
 document.getElementById('run').textContent=' run '+rid}
async function decide(id,ok){await fetch('/approvals/'+id,{method:'POST',headers:{'content-type':'application/json'},
 body:JSON.stringify({approve:ok,by:'inbox user'})})}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function tick(){const p=await (await fetch('/approvals')).json();
 document.getElementById('pending').innerHTML=p.length?p.map(x=>`<div class=card><b>${esc(x.tool)}</b> (waiting ${x.waiting_s}s)
 <pre>${esc(JSON.stringify(x.args,null,2))}</pre><button onclick="decide(${x.id},true)">Approve</button>
 <button onclick="decide(${x.id},false)">Deny</button></div>`).join(''):'none';
 if(rid){const r=await (await fetch('/runs/'+rid)).json();document.getElementById('out').textContent=JSON.stringify(r,null,2)}}
setInterval(tick,1500);tick();
</script>"""


@app.get("/", response_class=HTMLResponse)
def inbox() -> str:
    return PAGE
