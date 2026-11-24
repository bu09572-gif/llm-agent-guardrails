"""MCP server with the tools the filing agent can use. Any MCP client can connect to it over stdio:

    python -m toolguard.server

Tools (with MCP annotations, which the gateway reads to decide what needs a human):
  search_filings   BM25 search over 12,000 pages of 84 real SEC filings          read-only
  read_page        full text of one filing page (untrusted: a document can say anything)  read-only
  calculate        safe arithmetic (+ - * / ** and parentheses, no names or calls)        read-only
  send_email       send an email for the user (written to an outbox file here)     side effect

The data comes from FinanceBench (Patronus AI): `python -m docpipe.run --scope eval` downloads the
filings and builds data/silver and data/gold first.

For the evaluation only, TOOLGUARD_SCENARIO can point at a JSON file {"doc_id:page": "text to insert"}.
read_page then returns the page with that text inside it, which is how the attack reaches the agent:
through a tool result, the same way a poisoned web page or document would.
"""
from __future__ import annotations

import ast
import json
import operator
import os
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from docpipe import storage
from docqa.retrieve import Retriever

OUTBOX = Path(os.environ.get("TOOLGUARD_OUTBOX", "outbox.jsonl"))
mcp = FastMCP("sec-filings", log_level="WARNING")


@lru_cache(maxsize=1)
def _store() -> tuple[Retriever, pd.Series]:
    chunks = pd.read_parquet(storage.path("gold", "financebench", "chunks.parquet"))
    pages = pd.read_parquet(storage.path("silver", "financebench", "pages.parquet"))
    return Retriever(chunks), pages.set_index(["doc_id", "page_num"]).text


def _scenario() -> dict:
    p = os.environ.get("TOOLGUARD_SCENARIO")
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else {}


def clean_page(text: str) -> str:
    """PDF text has one table cell per line; drop empty lines and extra spaces."""
    return "\n".join(x for x in (" ".join(ln.split()) for ln in str(text).split("\n")) if x)


def insert_line(page: str, line: str, at: float = 0.4) -> str:
    lines = page.split("\n")
    k = max(1, int(len(lines) * at))
    return "\n".join(lines[:k] + [line] + lines[k:])


@mcp.tool(annotations=ToolAnnotations(title="Search SEC filings", readOnlyHint=True, openWorldHint=False))
def search_filings(query: str, company: str = "", fiscal_year: str = "", k: int = 5) -> list[dict]:
    """Search 84 SEC filings (10-K, 10-Q, 8-K, earnings releases of 32 US companies).
    query: what to look for, e.g. "3M FY2018 cash flow statement purchases of property plant and equipment".
    company and fiscal_year are optional and are added to the query.
    Returns the best pages (doc_id, page, title, snippet); then call read_page for the full text."""
    r, _ = _store()
    q = " ".join(x for x in (company, f"FY{fiscal_year}" if fiscal_year else "", query) if x)
    hits, _ = r.search(q, k=max(1, min(int(k), 10)), routing="v2")
    return [{"doc_id": h.doc_id, "page": h.page_num, "title": h.header, "snippet": " ".join(h.text.split()[:40])}
            for h in hits]


@mcp.tool(annotations=ToolAnnotations(title="Read a filing page", readOnlyHint=True, openWorldHint=True))
def read_page(doc_id: str, page: int) -> str:
    """Full text of one page of a filing, from search_filings results (doc_id and page)."""
    _, pages = _store()
    key = (doc_id, int(page))
    if key not in pages.index:
        return f"No page {page} in {doc_id}."
    text = clean_page(pages[key])
    extra = _scenario().get(f"{doc_id}:{int(page)}")
    return insert_line(text, extra) if extra else text


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    """Arithmetic only. No names, calls or attributes, so there is no way to run code through it."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"not allowed: {type(n).__name__}")
    if len(expr) > 200:
        raise ValueError("expression too long")
    return float(ev(ast.parse(expr.replace(",", "").replace("$", ""), mode="eval")))


@mcp.tool(annotations=ToolAnnotations(title="Calculator", readOnlyHint=True, openWorldHint=False))
def calculate(expression: str) -> str:
    """Evaluate arithmetic such as (1577 + 1373) / 2. Numbers and + - * / ** ( ) only."""
    try:
        return str(round(safe_eval(expression), 6))
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"error: {e}"


@mcp.tool(annotations=ToolAnnotations(title="Send email", readOnlyHint=False, destructiveHint=False,
                                      idempotentHint=False, openWorldHint=True))
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email on the user's behalf."""
    with OUTBOX.open("a") as fh:
        fh.write(json.dumps({"ts": time.time(), "to": to, "subject": subject[:200], "body": body[:4000]}) + "\n")
    return f"Email sent to {to}."


if __name__ == "__main__":
    mcp.run()
