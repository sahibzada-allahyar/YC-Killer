"""
Basic LLM client for local inference.
"""
from __future__ import annotations

import os
import requests
from typing import List

_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

_TIMEOUT = 300


def _post_json(obj):
    """Make a POST request to the LLM API."""
    r = requests.post(f"{_BASE}/api/generate", json=obj, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()["response"]


def chat(system: str, user: str) -> str:
    """Simple single‑turn chat with the LLM."""
    prompt = f"[SYSTEM]\n{system}\n[USER]\n{user}"
    return _post_json(
        {
            "model": _MODEL,
            "prompt": prompt,
            "temperature": 0.7,
            "stream": False,
        }
    )
