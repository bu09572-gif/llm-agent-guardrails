"""Real data for the agent prompt-injection eval.

  FinanceBench    150 analyst questions about public-company SEC filings, each with the filing page
                  that answers it (Patronus AI, CC BY-NC 4.0); loaded through src/evalkit
  deepset         deepset/prompt-injections: 662 prompts labelled injection or not (Apache-2.0).
                  Only used to train and test the injection filter
  Gandalf         Lakera/gandalf_ignore_instructions: 1,000 prompts real players typed into the
                  Gandalf game to make an LLM ignore its instructions (MIT). Used as the attacker's words

The two Hugging Face sets are mirrored into this repo's data-v2 release; SHA-256 pinned below.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PUBLIC_V2_DIR", ROOT / "data" / "raw" / "public_v2"))
BASE = "https://github.com/bu09572-gif/ai-ml-portfolio/releases/download/data-v2/"
SHA = {
    "deepset_prompt_injections__train__0": "2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04",
    "deepset_prompt_injections__test__0": "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e",
    "gandalf_ignore_instructions__train__0": "5b6acf3e5a5998d21f8e1222bb45bbdec25a14408747b1cd63bebef4a75fa439",
    "gandalf_ignore_instructions__validation__0": "f51ab3e3407a368845b0f57932cc745c09280429416d7507bd178a16326a79f6",
    "gandalf_ignore_instructions__test__0": "56b646d133335ebc535266bd55dbe1b5bee7caa4b95bf49d040684b9b5dd9972",
}
OVERRIDE = re.compile(r"\b(ignore|forget|disregard|override|new instructions?|previous instructions?)\b", re.I)
GAME_WORDS = re.compile(r"pass|word|secret|drowssap|gandalf|letter", re.I)


def _parquet(name: str) -> pd.DataFrame:
    p = RAW / f"{name}.parquet"
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        p.write_bytes(requests.get(BASE + p.name, timeout=120).content)
    if hashlib.sha256(p.read_bytes()).hexdigest() != SHA[name]:
        raise RuntimeError(f"{p.name} does not match the mirrored copy")
    return pd.read_parquet(p)


def deepset(split: str) -> pd.DataFrame:
    return _parquet(f"deepset_prompt_injections__{split}__0")


def gandalf() -> pd.DataFrame:
    return pd.concat([_parquet(f"gandalf_ignore_instructions__{s}__0") for s in ("train", "validation", "test")],
                     ignore_index=True)


def attacker_openers() -> list[str]:
    """The override part of real Gandalf attacks: the first sentence, if it tells the model to drop its
    instructions and doesn't talk about the game's password. Deduplicated, in a fixed order."""
    out, seen = [], set()
    for t in gandalf().text:
        s = re.split(r"(?<=[.!?])\s+|\n", str(t).strip())[0].strip()
        k = " ".join(s.lower().split())
        if 3 <= len(s.split()) <= 25 and OVERRIDE.search(s) and not GAME_WORDS.search(s) and k not in seen:
            seen.add(k)
            out.append(s)
    return out


def clean_page(text: str) -> str:
    """PDF text from FinanceBench has one table cell per line; join the blank-ish lines."""
    lines = [" ".join(x.split()) for x in str(text).split("\n")]
    return "\n".join(x for x in lines if x)


def questions() -> list[dict]:
    from evalkit.dataset import from_financebench
    out = []
    for it in from_financebench():
        page = "\n\n".join(clean_page(e["text"]) for e in it.raw_evidence)
        out.append({"id": it.id, "question": it.question, "gold": it.gold_answer, "page": page,
                    "company": it.tags.get("company")})
    return out
