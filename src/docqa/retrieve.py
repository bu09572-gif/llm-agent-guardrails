"""Retrieval over the gold chunks: BM25 (sparse), optional dense embeddings, hybrid fusion.

BM25 is implemented directly on a scipy sparse matrix so it is fast on ~40k
chunks and easy to explain:  score(q, d) = sum over query terms t of
    idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len(d) / avg_len))

Metadata filters (company, period, doc type) are applied before ranking. In
finance, "revenue" without the right company and year is the wrong answer, so
we detect the company and fiscal year in the question and filter on them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

TOKEN = r"(?u)\b[\w$%.]*\w\b"
STOP = "english"


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    page_num: int
    score: float
    text: str
    header: str


class BM25Index:
    def __init__(self, chunks: pd.DataFrame, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks.reset_index(drop=True)
        self.vec = CountVectorizer(token_pattern=TOKEN, stop_words=STOP, lowercase=True, dtype=np.float32)
        docs = (self.chunks.header + " " + self.chunks.text).tolist()
        tf = self.vec.fit_transform(docs).tocsc()
        n = tf.shape[0]
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        lens = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * lens / lens.mean())
        tf = tf.tocsr()
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + norm[rows])
        self.weights = tf.tocsc()

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        cols = q.indices
        if len(cols) == 0:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.asarray(self.weights[:, cols] @ (self.idf[cols] * q.data)).ravel()


# Names people use for a company that differ from the name in the filing metadata.
ALIASES = {
    "amex": "American Express", "jpm": "JPMorgan", "jp morgan": "JPMorgan", "jnj": "Johnson & Johnson",
    "j&j": "Johnson & Johnson", "johnson and johnson": "Johnson & Johnson", "mgm": "MGM Resorts",
    "coca cola": "Coca-Cola", "coke": "Coca-Cola", "ulta": "Ulta Beauty", "aes": "AES Corporation",
    "activision": "Activision Blizzard", "square": "Block", "footlocker": "Foot Locker", "cvs": "CVS Health",
    "lockheed": "Lockheed Martin", "kraft": "Kraft Heinz", "american water": "American Water Works",
}
QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def find_company(question: str, companies: list[str]) -> str | None:
    ql = question.lower()
    for c in sorted(companies, key=len, reverse=True):          # longest name first: "American Express" before "Express"
        if c.lower() in ql:
            return c
    for alias, c in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        if c in companies and re.search(rf"(?<![\w&]){re.escape(alias)}(?![\w&])", ql):
            return c
    return None


def find_years(question: str) -> list[int]:
    """FY2022, 2022, FY22, FY 22 -> 2022."""
    ys = [int(y) for y in re.findall(r"(?:(?<!\d)|(?<=Q\d))(20\d\d)(?!\d)", question)]
    ys += [2000 + int(y) for y in re.findall(r"\bFY\s?'?(\d\d)\b", question, re.I)]
    return ys


def find_quarter(question: str) -> int | None:
    m = re.search(r"\bQ([1-4])(?:\b|(?=20\d\d))", question) or re.search(r"(?<=\d)Q([1-4])\b", question)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(first|second|third|fourth)\s+(?:fiscal\s+)?quarter", question, re.I)
    if m:
        return QUARTER_WORDS[m.group(1).lower()]
    if re.search(r"\bH1\b|first half", question, re.I):
        return 2
    return None


def detect_filters(question: str, companies: list[str]) -> dict:
    """v1 rules: exact company name and the first year mentioned. Kept for the ablation table."""
    out = {}
    ql = question.lower()
    for c in sorted(companies, key=len, reverse=True):
        if c.lower() in ql:
            out["company"] = c
            break
    years = re.findall(r"\b(?:FY\s?)?(20\d\d)\b", question)
    if years:
        out["period"] = int(years[0])
    return out


def route(question: str, docs: pd.DataFrame) -> tuple[list[str] | None, dict]:
    """Pick the filings a question can be about (v2 rules), from a table with one row per filing
    (doc_id, company, period). Returns (doc_ids or None for "all", what was detected).

    - company: name or a common alias (AMEX, JnJ, JPM ...)
    - year: the latest year mentioned that this company has a filing for. A FY2020 10-K also
      reports FY2018 and FY2019, so "FY2018 - FY2020 average" belongs in the 2020 filing.
      A year with no filing ("forecast for FY2023") is ignored instead of filtering to nothing.
    - quarter: "Q2", "Q22023", "second quarter", "H1" prefer that quarter's 10-Q or earnings release.
    """
    info: dict = {}
    company = find_company(question, sorted(docs.company.dropna().unique()))
    if not company:
        return None, info
    info["company"] = company
    cand = docs[docs.company == company]
    have = set(cand.period.dropna().astype(int))
    years = [y for y in find_years(question) if y in have]
    if years:
        info["period"] = max(years)
        cand = cand[cand.period == info["period"]]
    q = find_quarter(question)
    if q:
        qd = cand[cand.doc_id.str.upper().str.contains(f"Q{q}_")]
        if len(qd):
            info["quarter"] = q
            cand = qd
    return cand.doc_id.tolist(), info


# Analysts ask for metric names; filings use line-item names. A small, auditable
# glossary bridges that vocabulary gap (query expansion).
FINANCE_SYNONYMS = {
    "capital expenditure": "purchases of property plant and equipment capex additions to property",
    "capex": "purchases of property plant and equipment capital expenditures",
    "revenue": "net sales total revenues net revenues",
    "net income": "net earnings attributable",
    "cogs": "cost of goods sold cost of sales cost of revenue",
    "cost of goods sold": "cost of sales cost of revenue",
    "operating income": "operating profit income from operations",
    "cash from operations": "net cash provided by operating activities",
    "operating cash flow": "net cash provided by operating activities",
    "free cash flow": "net cash provided by operating activities purchases of property plant and equipment",
    "ebitda": "operating income depreciation and amortization",
    "dividend": "dividends declared per share cash dividends paid",
    "inventory": "inventories",
    "accounts payable": "trade payables",
    "accounts receivable": "trade receivables receivables net",
    "total assets": "balance sheet total assets",
    "working capital": "total current assets total current liabilities",
    "debt": "long-term debt borrowings notes payable",
    "eps": "earnings per share diluted basic",
    "margin": "gross profit net sales",
    "effective tax rate": "provision for income taxes effective tax rate",
    "segment": "segment reporting operating segments",
}


def expand_query(q: str) -> str:
    ql = q.lower()
    extra = [v for k, v in FINANCE_SYNONYMS.items() if k in ql]
    return q + (" " + " ".join(extra) if extra else "")


def rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion: combine several rankings without tuning score scales."""
    n = len(rankings[0])
    fused = np.zeros(n)
    for ranking in rankings:
        ranks = np.empty(n)
        ranks[ranking] = np.arange(1, n + 1)
        fused += 1.0 / (k + ranks)
    return fused


class Retriever:
    def __init__(self, chunks: pd.DataFrame, encoder=None, reranker=None, embeddings: np.ndarray | None = None):
        """encoder: optional object with .encode(list[str], query=bool) -> np.ndarray (see docqa.models).
        reranker: optional object with .score(question, list[str]) -> np.ndarray (a cross-encoder).
        embeddings: precomputed, L2-normalised chunk vectors in the same row order (saves re-encoding)."""
        self.chunks = chunks.reset_index(drop=True)
        self.bm25 = BM25Index(self.chunks)
        self.companies = sorted(self.chunks.company.dropna().unique())
        self.docs = self.chunks.drop_duplicates("doc_id")[["doc_id", "company", "period", "doc_type"]]
        self.encoder, self.reranker = encoder, reranker
        self.emb = embeddings
        if encoder is not None and embeddings is None:
            e = encoder.encode((self.chunks.header + " " + self.chunks.text).tolist())
            self.emb = e / np.linalg.norm(e, axis=1, keepdims=True)

    def _mask(self, question: str, use_filters: bool, routing: str, doc_id: str | None) -> tuple[np.ndarray, dict]:
        mask = np.ones(len(self.chunks), dtype=bool)
        if doc_id:
            return (self.chunks.doc_id == doc_id).to_numpy(), {"doc_id": doc_id}
        if not use_filters:
            return mask, {}
        if routing == "v2":
            ids, info = route(question, self.docs)
            if ids:
                mask = self.chunks.doc_id.isin(ids).to_numpy()
            return mask, info
        filters = detect_filters(question, self.companies)
        for col, val in filters.items():
            m = (self.chunks[col] == val).to_numpy()
            if m.any():               # never filter down to nothing because of a bad guess
                mask &= m
        return mask, filters

    def search(self, question: str, k: int = 5, use_filters: bool = True, doc_id: str | None = None,
               mode: str = "bm25", expand: bool = True, dedupe_pages: bool = True, routing: str = "v1",
               rerank: int = 0) -> tuple[list[Hit], dict]:
        """mode: bm25 | dense | hybrid (reciprocal rank fusion of the two).
        rerank: if > 0 and a reranker is loaded, re-score the top `rerank` pages with the cross-encoder."""
        mask, filters = self._mask(question, use_filters, routing, doc_id)
        s = self.bm25.scores(expand_query(question) if expand else question)
        if mode in ("dense", "hybrid") and self.emb is not None:
            qv = self.encoder.encode([question], query=True)[0]
            d = self.emb @ (qv / np.linalg.norm(qv))
            if mode == "dense":
                s = d
            else:                       # fuse ranks inside the routed filings only
                idx = np.flatnonzero(mask)
                fused = np.full(len(s), -np.inf)
                fused[idx] = rrf([np.argsort(-s[idx]), np.argsort(-d[idx])])
                s = fused
        s = np.where(mask, s, -np.inf)
        order = np.argsort(-s)
        depth = max(k, rerank if self.reranker is not None else 0)
        if dedupe_pages:   # overlapping windows of one page shouldn't use up several of the k slots
            pages = self.chunks.doc_id.to_numpy()[order] + ":" + self.chunks.page_num.astype(str).to_numpy()[order]
            _, first = np.unique(pages[: depth * 20], return_index=True)
            order = order[np.sort(first)]
        order = order[np.isfinite(s[order])]
        top = order[:depth]
        if rerank and self.reranker is not None and len(top):
            texts = (self.chunks.header.iloc[top] + "\n" + self.chunks.text.iloc[top]).tolist()
            rs = self.reranker.score(question, texts)
            top, scores = top[np.argsort(-rs)], np.sort(rs)[::-1]
        else:
            scores = s[top]
        top, scores = top[:k], scores[:k]
        hits = [Hit(r.chunk_id, r.doc_id, int(r.page_num), float(sc), r.text, r.header)
                for sc, r in zip(scores, self.chunks.iloc[top].itertuples(index=False))]
        return hits, filters
