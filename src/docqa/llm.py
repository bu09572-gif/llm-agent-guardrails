"""Pluggable LLM backends. The default is FREE: a local open-source model via Ollama.

    LLM_PROVIDER=ollama     (default) local model, no key, no cost.  LLM_MODEL default: qwen2.5:3b
    LLM_PROVIDER=groq       free-tier hosted Llama, needs GROQ_API_KEY (no credit card). default: llama-3.1-8b-instant
    LLM_PROVIDER=anthropic  optional paid backend with native citations, needs ANTHROPIC_API_KEY

Ollama and Groq are called over plain HTTP with `requests`, so no extra packages are needed.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
DEFAULT_MODELS = {"ollama": "qwen2.5:3b", "groq": "llama-3.1-8b-instant", "anthropic": "claude-opus-5-5"}
# USD per million tokens (input, output). Local and free-tier models cost $0.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    provider: str

    @property
    def cost_usd(self) -> float:
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6


def model_name(provider: str | None = None) -> str:
    provider = provider or PROVIDER
    return os.environ.get("LLM_MODEL", DEFAULT_MODELS[provider])


def chat(system: str, user: str, provider: str | None = None, model: str | None = None,
         max_tokens: int = 1024, timeout: int = 300, json_mode: bool = False) -> Completion:
    """One system + user turn. Temperature 0 where supported, for repeatable evals.
    json_mode asks Ollama / Groq to return a single JSON object."""
    provider = provider or PROVIDER
    model = model or model_name(provider)
    t0 = time.perf_counter()
    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        r = requests.post(f"{host}/api/chat", timeout=timeout, json={
            "model": model, "stream": False, **({"format": "json"} if json_mode else {}),
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 8192},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        return Completion(d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0),
                          time.perf_counter() - t0, model, provider)
    if provider == "groq":
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=timeout,
                          headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                          json={"model": model, "temperature": 0, "max_tokens": max_tokens,
                                **({"response_format": {"type": "json_object"}} if json_mode else {}),
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        if r.status_code == 429:                     # free tier is rate limited: wait and retry once
            time.sleep(float(r.headers.get("retry-after", 10)))
            return chat(system, user, provider, model, max_tokens, timeout, json_mode)
        r.raise_for_status()
        d = r.json()
        return Completion(d["choices"][0]["message"]["content"], d["usage"]["prompt_tokens"],
                          d["usage"]["completion_tokens"], time.perf_counter() - t0, model, provider)
    if provider == "anthropic":
        import anthropic
        resp = anthropic.Anthropic().messages.create(
            model=model, max_tokens=max_tokens, system=system, output_config={"effort": "low"},
            messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(text, resp.usage.input_tokens, resp.usage.output_tokens,
                          time.perf_counter() - t0, resp.model, provider)
    raise ValueError(f"unknown LLM_PROVIDER {provider!r}")
