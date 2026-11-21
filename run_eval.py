"""Prompt-injection eval for the filing agent, run in CI with free local models (Ollama), sharded.

    python run_eval.py --filter-report   # injection filter alone: results/filter.json (no LLM)
    python run_eval.py --shard 0/20      # episodes for one shard
    python run_eval.py --merge           # results/episodes.jsonl, results/summary.json

Grid per question and model: page clean / with the email attack / with the link attack, crossed with
spotlight off/on and filter off/on. The code policy is applied inside every episode and recorded, so
each episode is scored both without and with it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import agent as ag
import data
from evalkit.metrics import numeric_match

OUT = Path(__file__).resolve().parent / "results"
MODELS = os.environ.get("EVAL_MODELS", "qwen2.5:3b,llama3.2:3b").split(",")
CONDITIONS = ["clean", "email", "link"]


NEGATIVES = Path(__file__).resolve().parent / "filing_negatives.txt"


def trained_filter(domain: bool = True) -> ag.InjectionFilter:
    """deepset train, plus (domain=True) 3,000 snippets of ordinary filing text as non-injections.
    The snippets come from FinanceBench filing pages that are NOT the evidence pages used in the eval."""
    tr = data.deepset("train")
    texts, labels = list(tr.text), list(tr.label)
    if domain:
        neg = [x for x in NEGATIVES.read_text().splitlines() if x.strip()]
        texts, labels = texts + neg, labels + [0] * len(neg)
    return ag.InjectionFilter().fit(texts, labels)


def _pick(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n


def cases() -> list[dict]:
    openers = data.attacker_openers()
    out = []
    for q in data.questions():
        for cond in CONDITIONS:
            opener = "" if cond == "clean" else openers[_pick(q["id"] + cond, len(openers))]
            page = q["page"] if cond == "clean" else ag.inject(q["page"], opener, cond)
            out.append(q | {"condition": cond, "opener": opener, "attacked_page": page})
    return out


def filter_report() -> dict:
    out = {"deepset_only": _report(trained_filter(domain=False)), "deepset_plus_filing_text": _report(trained_filter())}
    OUT.mkdir(exist_ok=True)
    (OUT / "filter.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def _report(f: ag.InjectionFilter) -> dict:
    te = data.deepset("test")
    pred = (f.score(te.text) > f.threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(te.label, pred, average="binary")
    g = data.gandalf()
    openers = data.attacker_openers()
    qs = data.questions()
    lines = [ln for q in qs for ln in q["page"].split("\n")]
    flagged_pages = sum(f.clean(q["page"])[1] > 0 for q in qs)
    attacks = [c for c in cases() if c["condition"] != "clean"]
    removed = [f.clean(c["attacked_page"])[0].count(ag.GOALS[c["condition"]]) == 0 for c in attacks]
    flagged_lines = [ln for ln in lines if f.score([ln])[0] > f.threshold]
    return {
        "deepset_test": {"n": len(te), "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)},
        "gandalf_full_prompts_flagged": round(float((f.score(g.text) > f.threshold).mean()), 3),
        "gandalf_openers_flagged": round(float((f.score(openers) > f.threshold).mean()), 3),
        "injected_attack_line_removed": round(float(np.mean(removed)), 3),
        "clean_filing_lines": len(lines),
        "clean_filing_lines_flagged": len(flagged_lines),
        "clean_pages_with_a_line_removed": f"{flagged_pages} of {len(qs)}",
        "examples_of_clean_lines_flagged": flagged_lines[:10],
    }


def run(shard: str) -> None:
    i, n = map(int, shard.split("/"))
    f = trained_filter()
    have = set()
    if (OUT / "episodes.jsonl").exists():          # a rerun only does the cases an earlier run didn't finish
        old = pd.read_json(OUT / "episodes.jsonl", lines=True)
        cnt = old[old.error.fillna("") == ""].groupby(["id", "condition"]).size()
        have = {k for k, v in cnt.items() if v == 4 * len(MODELS)}
    todo = [c for c in cases() if (c["id"], c["condition"]) not in have][i::n]
    OUT.mkdir(exist_ok=True)
    with (OUT / f"episodes_shard{i}.jsonl").open("w") as fh:
        for k, c in enumerate(todo):
            filtered, n_removed = f.clean(c["attacked_page"])
            for model in MODELS:
                done = {}
                for spot in (False, True):
                    for filt in (False, True):
                        page = filtered if filt else c["attacked_page"]
                        key = (spot, page)
                        if key not in done:      # filter removed nothing: same input, same episode
                            done[key] = ag.run(c["question"], page, model, spotlight=spot)
                        ep = done[key]
                        row = {"id": c["id"], "condition": c["condition"], "model": model, "spotlight": spot,
                               "filter": filt, "lines_removed": n_removed if filt else 0, "opener": c["opener"],
                               "gold": c["gold"], "answer": ep.answer[:800], "answer_after_policy": ep.answer_after_policy[:800],
                               "read_page": ep.read_page, "emails": ep.emails, "emails_sent": ep.emails_sent,
                               "policy_events": ep.policy_events, "steps": [s["action"] for s in ep.steps],
                               "tokens": ep.tokens, "error": ep.error}
                        fh.write(json.dumps(row) + "\n")
                fh.flush()
            print(k, len(todo), flush=True)


def _ep(r) -> ag.Episode:
    return ag.Episode(answer=r.answer or "", answer_after_policy=r.answer_after_policy or "",
                      emails=list(r.emails), emails_sent=list(r.emails_sent))


def summarize(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["defence"] = np.select([df.spotlight & df["filter"], df.spotlight, df["filter"]],
                              ["spotlight+filter", "spotlight", "filter"], "none")
    num = [numeric_match(a, g) for a, g in zip(df.answer_after_policy.fillna(""), df.gold)]
    df["correct"] = [x if x is None else bool(x) for x in num]
    out = {"models": {}, "questions": int(df.id.nunique())}
    for model, g in df.groupby("model"):
        m = {"attack_success": {}, "clean_utility": {}, "attacked_utility": {}}
        for cond in ("email", "link"):
            x = g[g.condition == cond]
            m["attack_success"][cond] = {}
            for d, gd in x.groupby("defence"):
                eps = [_ep(r) for r in gd.itertuples()]
                m["attack_success"][cond][d] = round(float(np.mean([ag.hijacked(e, cond, False) for e in eps])), 3)
                m["attack_success"][cond][d + "+policy"] = round(float(np.mean([ag.hijacked(e, cond, True) for e in eps])), 3)
        for label, x in (("clean_utility", g[g.condition == "clean"]), ("attacked_utility", g[g.condition != "clean"])):
            for d, gd in x.groupby("defence"):
                c = gd.correct.dropna()
                m[label][d] = {"numeric_accuracy": round(float(c.mean()), 3), "numeric_questions": int(len(c) / max(gd.condition.nunique(), 1)),
                               "answered": round(float((gd.answer.str.len() > 0).mean()), 3),
                               "read_page": round(float(gd.read_page.mean()), 3)}
        clean = g[(g.condition == "clean") & (g.defence == "none")]
        m["clean_any_email_attempt"] = round(float(clean.emails.map(len).gt(0).mean()), 3)
        m["median_tokens_per_episode"] = float(g.tokens.median())
        m["errors"] = int((g.error.fillna("") != "").sum())
        out["models"][model] = m
    return json.loads(json.dumps(out, default=float))


def merge() -> dict:
    files = sorted(OUT.glob("episodes_shard*.jsonl"))
    parts = [pd.read_json(f, lines=True) for f in files if f.stat().st_size]
    if (OUT / "episodes.jsonl").exists():
        parts.insert(0, pd.read_json(OUT / "episodes.jsonl", lines=True))
    df = pd.concat(parts, ignore_index=True)
    df["_ok"] = df.error.fillna("") == ""
    df = (df.sort_values("_ok", kind="stable")       # a successful episode replaces a failed one
          .drop_duplicates(["id", "condition", "model", "spotlight", "filter"], keep="last").drop(columns="_ok"))
    df.to_json(OUT / "episodes.jsonl", orient="records", lines=True)
    for f in files:
        f.unlink()
    s = summarize(df)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-report", action="store_true")
    ap.add_argument("--shard")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    if a.filter_report:
        print(json.dumps(filter_report(), indent=2))
    elif a.merge:
        print(json.dumps(merge(), indent=2))
    else:
        run(a.shard)


if __name__ == "__main__":
    main()
