"""Silver pages -> gold chunks.

Chunks never cross a page boundary, so every answer can cite an exact page.
Long pages are split into overlapping word windows. Each chunk gets a short
header (company, filing, period, page) so a paragraph like "Net sales increased 9%"
can still be matched to the right company and year.
"""
from __future__ import annotations

import hashlib

import pandas as pd

WORDS = 220
OVERLAP = 40
MIN_WORDS = 15


def chunk_page(text: str, size: int = WORDS, overlap: int = OVERLAP) -> list[str]:
    words = text.split()
    if len(words) < MIN_WORDS:
        return []
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def build_chunks(pages: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    meta = manifest.set_index("doc_id")
    rows = []
    for p in pages.itertuples(index=False):
        m = meta.loc[p.doc_id]
        header = f"{m.company} | {m.doc_type} {m.period} | page {p.page_num}"
        for j, body in enumerate(chunk_page(p.text)):
            cid = hashlib.md5(f"{p.doc_id}:{p.page_num}:{j}".encode()).hexdigest()[:12]
            rows.append({"chunk_id": cid, "doc_id": p.doc_id, "page_num": p.page_num, "part": j,
                         "company": m.company, "doc_type": m.doc_type, "period": m.period,
                         "sector": m.sector, "header": header, "text": body,
                         "n_words": len(body.split())})
    return pd.DataFrame(rows)
