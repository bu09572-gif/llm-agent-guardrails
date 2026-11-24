"""Where pipeline data lives.

Locally (or in Codespaces) data goes under ./data. Set DATA_ROOT=s3://bucket/prefix
to write the same layout to S3 instead (needs `pip install s3fs`); Athena can then
query the parquet files directly.

Layout (medallion style):
    bronze/<source>/raw files exactly as downloaded + manifest.parquet
    silver/<source>/pages.parquet        one row per PDF page of extracted text
    gold/<source>/chunks.parquet         retrieval-ready chunks with metadata
    reports/<source>/run_<ts>.json       run report (counts, timings, quality checks)
"""
from __future__ import annotations

import os
from pathlib import Path

import fsspec

DATA_ROOT = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data"))


def path(*parts: str) -> str:
    return "/".join([DATA_ROOT.rstrip("/"), *parts])


def fs_for(p: str) -> fsspec.AbstractFileSystem:
    return fsspec.core.url_to_fs(p)[0]


def ensure_dir(p: str) -> None:
    fs_for(p).makedirs(p, exist_ok=True)


def exists(p: str) -> bool:
    return fs_for(p).exists(p)
