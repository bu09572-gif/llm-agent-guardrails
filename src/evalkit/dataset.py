"""A common eval-set format, so every project in the portfolio is scored the same way.

One JSON line per item:
  {"id": "...", "question": "...", "gold_answer": "...",
   "gold_evidence": [{"doc_id": "...", "page_num": 12}],
   "tags": {"question_type": "...", "reasoning": "...", "sector": "..."},
   "doc_id": "..."}          # the filing the question is about (used for "single-document" mode)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EvalItem:
    id: str
    question: str
    gold_answer: str
    gold_evidence: list[dict] = field(default_factory=list)
    tags: dict = field(default_factory=dict)
    doc_id: str | None = None
    raw_evidence: list[dict] = field(default_factory=list)   # evidence text, for data checks


def from_financebench() -> list[EvalItem]:
    from docpipe.sources.financebench import load_questions
    q = load_questions()
    return [EvalItem(
        id=r.financebench_id, question=r.question, gold_answer=str(r.answer),
        gold_evidence=[{"doc_id": e["doc_name"], "page_num": int(e["evidence_page_num"])} for e in r.evidence],
        tags={"question_type": r.question_type, "reasoning": r.question_reasoning, "sector": r.gics_sector,
              "company": r.company},
        doc_id=r.doc_name,
        raw_evidence=[{"doc_id": e["doc_name"], "page_num": int(e["evidence_page_num"]), "text": e["evidence_text"]}
                      for e in r.evidence]) for r in q.itertuples()]


def save(items: list[EvalItem], path: str | Path) -> None:
    Path(path).write_text("\n".join(json.dumps(asdict(i)) for i in items) + "\n")


def load(path: str | Path) -> list[EvalItem]:
    return [EvalItem(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


DATASETS = {"financebench": from_financebench}
