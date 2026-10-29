"""LLM client for OpenRouter (OpenAI-compatible) with caching and retries.

Every response is saved to results/llm_cache/<experiment>.jsonl, keyed by a hash of
(model, provider, temperature, prompt, run index). Re-running an experiment never
pays twice for the same call, and crashed runs resume where they stopped.
"""
import hashlib
import json
import re
import threading
import time
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (CACHE_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
                    LLM_PROVIDER, LLM_REASONING_EFFORT)

_client = None
_lock = threading.Lock()


def client() -> OpenAI:
    global _client
    if _client is None:
        if not LLM_API_KEY or "your-key" in LLM_API_KEY:
            raise RuntimeError("Set LLM_API_KEY in .env (copy .env.example to .env).")
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


def _key(prompt: str, temperature: float, run: int) -> str:
    raw = json.dumps([LLM_MODEL, LLM_PROVIDER, LLM_REASONING_EFFORT,
                      temperature, prompt, run])
    return hashlib.sha256(raw.encode()).hexdigest()


class Cache:
    def __init__(self, experiment: str):
        self.path = Path(CACHE_DIR) / f"{experiment}.jsonl"
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec

    def get(self, key):
        return self.data.get(key)

    def put(self, rec):
        with _lock:
            self.data[rec["key"]] = rec
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


def _extra_body():
    body = {}
    if LLM_PROVIDER:
        # Same backend for every call; fail loudly instead of silently switching.
        body["provider"] = {"order": [LLM_PROVIDER], "allow_fallbacks": False}
    if LLM_REASONING_EFFORT:
        body["reasoning"] = {"effort": LLM_REASONING_EFFORT}
    return body


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, max=60))
def _call(prompt: str, temperature: float, max_tokens: int):
    resp = client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=_extra_body(),
    )
    msg = resp.choices[0].message
    content = msg.content or ""
    if not content.strip():
        raise RuntimeError("Empty response (often: max_tokens used up by reasoning)")
    usage = resp.usage.model_dump() if resp.usage else {}
    provider = getattr(resp, "provider", None) or (resp.model_extra or {}).get("provider")
    return content, usage, provider


def ask(prompt: str, cache: Cache, sample_id: str, temperature: float = 0.0,
        run: int = 0, max_tokens: int = 4000) -> dict:
    """Return a cached or fresh record: {key, sample_id, run, content, label, ...}."""
    key = _key(prompt, temperature, run)
    hit = cache.get(key)
    if hit:
        return hit
    t0 = time.time()
    content, usage, provider = _call(prompt, temperature, max_tokens)
    rec = {
        "key": key, "sample_id": sample_id, "run": run,
        "model": LLM_MODEL, "provider_requested": LLM_PROVIDER,
        "provider_served": provider, "temperature": temperature,
        "reasoning_effort": LLM_REASONING_EFFORT,
        "content": content, "label": parse_label(content),
        "usage": usage, "latency_s": round(time.time() - t0, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    cache.put(rec)
    return rec


def parse_label(text: str):
    """Extract 0/1. Prefers <ANSWER>..</ANSWER>, else the last standalone 0 or 1."""
    m = re.search(r"<ANSWER>\s*([01])\s*</ANSWER>", text, re.I)
    if m:
        return int(m.group(1))
    hits = re.findall(r"(?<![\w.])([01])(?![\w.])", text)
    return int(hits[-1]) if hits else None
