"""Run the document pipeline end to end: fetch -> bronze -> silver -> gold -> checks.

    python -m docpipe.run --scope eval          # the 84 filings the eval questions use
    python -m docpipe.run --scope all           # all 368 filings
    python -m docpipe.run --scope eval --limit 5

Files already in bronze are not downloaded again, so a re-run only fetches new ones.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import pandas as pd

from . import storage
from .chunk import build_chunks
from .extract import pdf_pages
from .quality import as_dicts, run_checks
from .sources import financebench as src


def _bronze_pdf(doc_id: str) -> str:
    return storage.path("bronze", src.NAME, "pdfs", f"{doc_id}.pdf")


def _read_parquet(p: str) -> pd.DataFrame | None:
    return pd.read_parquet(p) if storage.exists(p) else None


def ingest(docs: pd.DataFrame, workers: int = 8) -> tuple[pd.DataFrame, dict]:
    """Download missing PDFs into bronze and return the manifest."""
    storage.ensure_dir(storage.path("bronze", src.NAME, "pdfs"))
    man_path = storage.path("bronze", src.NAME, "manifest.parquet")
    old = _read_parquet(man_path)
    have = set(old.doc_id) if old is not None else set()
    todo = [r for r in docs.to_dict("records") if r["doc_name"] not in have or not storage.exists(_bronze_pdf(r["doc_name"]))]
    rows, failures, secs = [], [], 0.0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(src.fetch, r["doc_name"]): r for r in todo}
        for fut in as_completed(futures):
            meta = futures[fut]
            try:
                f = fut.result()
            except Exception as e:  # record the failure and continue
                failures.append({"doc_id": meta["doc_name"], "error": repr(e)[:200]})
                continue
            with storage.fs_for(_bronze_pdf(f.doc_id)).open(_bronze_pdf(f.doc_id), "wb") as fh:
                fh.write(f.content)
            rows.append(src.manifest_row(f, meta))
            secs += f.seconds
    new = pd.DataFrame(rows)
    manifest = pd.concat([d for d in (old, new) if d is not None and len(d)], ignore_index=True) if (old is not None or len(new)) else new
    manifest = manifest.drop_duplicates("doc_id", keep="last")
    manifest = manifest[manifest.doc_id.isin(docs.doc_name)].reset_index(drop=True)
    with storage.fs_for(man_path).open(man_path, "wb") as fh:
        manifest.to_parquet(fh, index=False)
    return manifest, {"downloaded": len(rows), "skipped_existing": len(docs) - len(todo),
                      "failed": failures, "download_seconds_total": round(secs, 1)}


def extract(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc_id in manifest.doc_id:
        with storage.fs_for(_bronze_pdf(doc_id)).open(_bronze_pdf(doc_id), "rb") as fh:
            pages = pdf_pages(fh.read())
        rows += [{"doc_id": doc_id, "page_num": i, "text": t, "n_chars": len(t)} for i, t in enumerate(pages)]
    return pd.DataFrame(rows)


def write(df: pd.DataFrame, layer: str, name: str) -> str:
    storage.ensure_dir(storage.path(layer, src.NAME))
    p = storage.path(layer, src.NAME, f"{name}.parquet")
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    with storage.fs_for(p).open(p, "wb") as fh:
        fh.write(buf.getvalue())
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["eval", "all"], default="eval")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    t = {}
    t0 = time.perf_counter()
    docs = src.select_docs(a.scope, a.limit)
    manifest, ingest_stats = ingest(docs)
    t["ingest_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    pages = extract(manifest)
    t["extract_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    chunks = build_chunks(pages, manifest)
    t["chunk_s"] = time.perf_counter() - t2

    checks = run_checks(manifest, pages, chunks)
    critical_failed = [c.name for c in checks if c.critical and not c.passed]
    if not critical_failed:
        write(pages, "silver", "pages")
        write(chunks, "gold", "chunks")

    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": src.NAME, "scope": a.scope, "limit": a.limit,
        "counts": {"documents": len(manifest), "pages": len(pages), "chunks": len(chunks),
                   "companies": int(manifest.company.nunique()), "bronze_mb": round(manifest.bytes.sum() / 1e6, 1)},
        "ingest": ingest_stats,
        "timings_s": {k: round(v, 1) for k, v in t.items()},
        "quality_checks": as_dicts(checks),
        "status": "failed" if critical_failed else "success",
    }
    storage.ensure_dir(storage.path("reports", src.NAME))
    rp = storage.path("reports", src.NAME, f"pipeline_{a.scope}.json")
    with storage.fs_for(rp).open(rp, "w") as fh:
        fh.write(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ["status", "counts", "timings_s"]}, indent=2))
    for c in checks:
        print(f"  [{'PASS' if c.passed else ('FAIL' if c.critical else 'WARN')}] {c.name}: {c.observed}")
    return 1 if critical_failed else 0


if __name__ == "__main__":
    sys.exit(main())
