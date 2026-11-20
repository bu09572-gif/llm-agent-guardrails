"""Metrics. Each one is a small pure function so it can be unit-tested and reused.

Retrieval (no LLM needed):
  hit@k      did any of the top-k results land on a gold evidence page?
  mrr        1 / rank of the first gold page (0 if none in the list)
Answer:
  numeric_match   does the answer contain the gold number (1% tolerance, handles $, %, commas, "million")?
  judge           an LLM grader compares the answer to the gold answer: correct / incorrect / refused
  citation_page_precision   share of the answer's citations that point at a gold evidence page
"""
from __future__ import annotations

import json
import re

NUM = re.compile(r"\(?-?\$?\s?\d[\d,]*\.?\d*\)?\s?%?")


def page_key(doc_id: str, page_num: int) -> str:
    return f"{doc_id}:{page_num}"


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    return float(any(r in gold for r in retrieved[:k]))


def mrr(retrieved: list[str], gold: set[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


CITE_TAG = re.compile(r"\[S\d+\]")
YEAR_LIKE = re.compile(r"\b(?:FY|Q[1-4]\s?(?:of\s)?(?:FY)?)?\s?(?:19|20)\d\d\b")


def numbers(text: str) -> list[float]:
    """Numbers in a text, ignoring citation tags like [S2] and fiscal-year labels like FY2023."""
    text = YEAR_LIKE.sub(" ", CITE_TAG.sub(" ", text))
    text = re.sub(r"\bQ[1-4]\b", " ", text)
    out = []
    for m in NUM.findall(text):
        neg = m.strip().startswith("(") and m.strip().endswith(")") or "-" in m
        v = re.sub(r"[^\d.]", "", m)
        if v and v != ".":
            try:
                out.append(-float(v) if neg else float(v))
            except ValueError:
                pass
    return out


def numeric_match(answer: str, gold: str, rel_tol: float = 0.01) -> float | None:
    """None when the gold answer has no number (then only the judge applies)."""
    g = numbers(gold)
    if not g:
        return None
    target = g[0]
    for a in numbers(answer):
        for scale in (1, 1e3, 1e-3, 1e6, 1e-6, 100, 0.01):   # "1.6 billion" vs "1577 million", "15%" vs 0.15
            if abs(a * scale - target) <= rel_tol * max(abs(target), 1e-9):
                return 1.0
    return 0.0


def final_grade(judge_grade: str, refused: bool, numeric: float | None) -> str:
    """Combine the LLM judge with rule-based checks; the rules win.
    The 3B judge graded most "not enough information" replies as correct, so refusals
    are detected by rule, and a numeric answer that misses the gold number can't be correct."""
    if refused:
        return "refused"
    if numeric == 0.0:
        return "incorrect"
    return judge_grade


def citation_page_precision(cited_pages: list[str], gold: set[str]) -> float | None:
    if not cited_pages:
        return None
    return sum(p in gold for p in cited_pages) / len(cited_pages)


JUDGE_PROMPT = """You are grading an answer to a question about a company's SEC filing.

Question: {question}
Gold answer (written by a finance expert): {gold}
Candidate answer: {answer}

Grade the candidate:
- "correct": it reaches the same conclusion/number as the gold answer (small rounding differences are fine)
- "incorrect": it gives a different conclusion or number, or includes a material error
- "refused": it says it cannot answer or lacks information

Reply with JSON only: {{"grade": "correct|incorrect|refused", "reason": "<one sentence>"}}"""


def judge(question: str, gold: str, answer: str, provider: str | None = None) -> dict:
    """LLM grader. Same free backend as answering unless EVAL_JUDGE_PROVIDER / EVAL_JUDGE_MODEL are set.
    A model grading itself can be lenient, so numeric_match is reported alongside as a
    deterministic cross-check."""
    import os

    from docqa import llm
    c = llm.chat("You are a strict grader. Reply with JSON only.",
                 JUDGE_PROMPT.format(question=question, gold=gold, answer=answer),
                 provider=provider, model=os.environ.get("EVAL_JUDGE_MODEL") or None, max_tokens=256)
    m = re.search(r"\{.*\}", c.text, re.DOTALL)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    grade = str(out.get("grade", "unparsed")).lower()
    return {"grade": grade if grade in {"correct", "incorrect", "refused"} else "unparsed",
            "reason": out.get("reason", c.text[:200]), "judge_model": c.model}
