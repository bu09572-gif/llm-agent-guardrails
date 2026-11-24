"""Source: FinanceBench (Patronus AI, 2023): real SEC filings + expert Q&A.

368 public-company filings (10-K, 10-Q, 8-K, earnings releases) and an open set
of 150 questions written by finance analysts, each with a gold answer and the
page number of the evidence. Paper: https://arxiv.org/abs/2311.11944
Repo:  https://github.com/patronus-ai/financebench

We download files at run time and never commit them to git.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
import requests

RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
NAME = "financebench"


@dataclass
class Fetched:
    doc_id: str
    url: str
    content: bytes
    seconds: float


def _get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    """GET with exponential backoff; raises after the last attempt."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def load_questions() -> pd.DataFrame:
    q = pd.read_json(f"{RAW}/data/financebench_open_source.jsonl", lines=True)
    meta = pd.read_json(f"{RAW}/data/financebench_document_information.jsonl", lines=True)
    return q.merge(meta[["doc_name", "doc_type", "doc_period", "gics_sector"]], on="doc_name", how="left")


def load_document_info() -> pd.DataFrame:
    return pd.read_json(f"{RAW}/data/financebench_document_information.jsonl", lines=True)


def select_docs(scope: str, limit: int | None = None) -> pd.DataFrame:
    """scope='eval' -> only the 84 filings the 150 questions refer to; 'all' -> all 368."""
    info = load_document_info()
    if scope == "eval":
        info = info[info.doc_name.isin(load_questions().doc_name.unique())]
    info = info.sort_values("doc_name").reset_index(drop=True)
    return info.head(limit) if limit else info


def fetch(doc_id: str) -> Fetched:
    url = f"{RAW}/pdfs/{doc_id}.pdf"
    t0 = time.perf_counter()
    content = _get(url)
    return Fetched(doc_id, url, content, time.perf_counter() - t0)


def manifest_row(f: Fetched, meta: dict) -> dict:
    return {
        "doc_id": f.doc_id,
        "source": NAME,
        "source_url": f.url,
        "sha256": hashlib.sha256(f.content).hexdigest(),
        "bytes": len(f.content),
        "company": meta.get("company"),
        "doc_type": str(meta.get("doc_type", "")).upper(),
        "period": int(meta["doc_period"]) if pd.notna(meta.get("doc_period")) else None,
        "sector": meta.get("gics_sector"),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
