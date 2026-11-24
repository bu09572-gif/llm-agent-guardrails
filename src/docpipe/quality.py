"""Data-quality checks run on every pipeline run.

Each check returns pass/fail plus the observed value, so the run report shows
*why* something failed. `critical` checks stop the pipeline; warnings don't.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Check:
    name: str
    passed: bool
    observed: object
    expected: str
    critical: bool = True


def run_checks(manifest: pd.DataFrame, pages: pd.DataFrame, chunks: pd.DataFrame,
               max_empty_page_rate: float = 0.10) -> list[Check]:
    empty_rate = float((pages.n_chars < 50).mean()) if len(pages) else 1.0
    docs_without_pages = sorted(set(manifest.doc_id) - set(pages.doc_id))
    docs_without_chunks = sorted(set(manifest.doc_id) - set(chunks.doc_id))
    checks = [
        Check("manifest_doc_id_unique", manifest.doc_id.is_unique, int(manifest.doc_id.duplicated().sum()), "0 duplicates"),
        Check("manifest_sha256_not_null", manifest.sha256.notna().all(), int(manifest.sha256.isna().sum()), "0 nulls"),
        Check("pages_key_unique", not pages.duplicated(["doc_id", "page_num"]).any(),
              int(pages.duplicated(["doc_id", "page_num"]).sum()), "0 duplicate (doc_id, page_num)"),
        Check("every_doc_has_pages", not docs_without_pages, docs_without_pages[:10], "[]"),
        Check("empty_page_rate", empty_rate <= max_empty_page_rate, round(empty_rate, 4),
              f"<= {max_empty_page_rate} (scanned/image-only pages have no text layer)", critical=False),
        Check("every_doc_has_chunks", not docs_without_chunks, docs_without_chunks[:10], "[]"),
        Check("chunk_id_unique", chunks.chunk_id.is_unique, int(chunks.chunk_id.duplicated().sum()), "0 duplicates"),
        Check("chunk_text_not_empty", bool((chunks.n_words > 0).all()), int((chunks.n_words == 0).sum()), "0 empty chunks"),
        Check("company_metadata_present", chunks.company.notna().all(), int(chunks.company.isna().sum()), "0 nulls"),
    ]
    return checks


def as_dicts(checks: list[Check]) -> list[dict]:
    return [{**asdict(c), "passed": bool(c.passed)} for c in checks]
