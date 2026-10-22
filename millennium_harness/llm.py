"""Minimal LLM backends. Benchmarks use the Anthropic API (sync or Batch); the product can borrow the host CLI."""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# USD per MTok for claude-haiku-5-5 (<=100K prompt / >100K prompt)
PRICE = {"in": (0.10, 0.50), "out": (0.50, 2.50), "cache_read": (0.01, 0.05), "cache_write": (0.125, 0.625)}


def cost(usage: dict, batch: bool = False) -> float:
    i = usage.get("input_tokens", 0) or 0
    cr = usage.get("cache_read_input_tokens", 0) or 0
    cw = usage.get("cache_creation_input_tokens", 0) or 0
    o = usage.get("output_tokens", 0) or 0
    tier = 1 if (i + cr + cw) > 100_000 else 0
    c = (i * PRICE["in"][tier] + cr * PRICE["cache_read"][tier] + cw * PRICE["cache_write"][tier]
         + o * PRICE["out"][tier]) / 1e6
    return c * (0.5 if batch else 1.0)


def text_of(content) -> str:
    return "".join(getattr(b, "text", "") if not isinstance(b, dict) else b.get("text", "")
                   for b in content if (getattr(b, "type", None) or b.get("type")) == "text")


def anthropic_call(params: dict) -> tuple[str, dict]:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(**params)
    return text_of(msg.content), msg.usage.model_dump()


def anthropic_parallel(param_list: list[dict], workers: int = 8) -> list[tuple[str, dict]]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(anthropic_call, param_list))


def host_cli_call(params: dict, host: str) -> str:
    """Product path: reuse the user's harness auth. Runs in an empty dir so it never sees the code."""
    prompt = params["system"] + "\n\n" + params["messages"][0]["content"]
    tmp = os.path.join(os.path.expanduser("~"), ".cache", "millennium-harness", "empty")
    os.makedirs(tmp, exist_ok=True)
    if host == "claude":
        cmd = ["claude", "-p", "--model", params["model"], "--output-format", "json", prompt]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, timeout=600).stdout
        return json.loads(out).get("result", "")
    if host == "opencode":
        cmd = ["opencode", "--pure", "run", "-m", f"anthropic/{params['model']}", prompt]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, timeout=600).stdout
    raise ValueError(host)
