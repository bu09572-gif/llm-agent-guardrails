"""Ask the agent one question in the terminal; you approve side effects at the prompt.

    python -m toolguard.cli "What is 3M's FY2018 capex? Then email it to me@example.com." --model qwen2.5:3b
"""
from __future__ import annotations

import argparse
import asyncio
import json

from .agent import mcp_session, run_episode
from .approval import ConsoleApprover
from .gateway import GuardConfig


async def main(question: str, model: str) -> None:
    async with mcp_session() as (session, tools):
        ep, gw = await run_episode(session, tools, question, model, GuardConfig("cli", approval=ConsoleApprover()))
    print("\nanswer:", ep.answer_shown or "(none)", ep.answer_events or "")
    print("audit:", json.dumps(gw.audit(), indent=1)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="qwen2.5:3b")
    a = ap.parse_args()
    asyncio.run(main(a.question, a.model))
