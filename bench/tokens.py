"""Uniform token accounting across harnesses (Harbor's per-agent cost fields are not comparable / sometimes missing).

extract(trial_dir) -> {"in": uncached_input, "cache": cache_read, "out": output, "src": where it came from}
Cost is then priced with one Haiku 5.5 table (≤100K tier) for every harness.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

P_IN, P_CACHE, P_OUT = 0.10, 0.01, 0.50


def _hermes(d: Path):
    f = d / "agent" / "hermes-session.jsonl"
    if not f.exists():
        return None
    txt = f.read_text()
    g = lambda k: max([int(x) for x in re.findall(rf'"{k}"\s*:\s*(\d+)', txt)] or [0])
    return {"in": g("input_tokens"), "cache": g("cache_read_tokens"), "out": g("output_tokens"), "src": "hermes-session"}


def _openclaw(d: Path):
    f = d / "agent" / "openclaw.txt"
    if not f.exists():
        return None
    txt, dec, best = f.read_text(), json.JSONDecoder(), None
    for m in re.finditer(r'"usage"\s*:\s*', txt):
        try:
            u, _ = dec.raw_decode(txt, m.end())
        except json.JSONDecodeError:
            continue
        if isinstance(u, dict) and "cacheRead" in u and (best is None or u.get("total", 0) > best.get("total", 0)):
            best = u
    if not best:
        return None
    # cache writes are billed at 1.25x input; fold them into uncached input at that rate
    return {"in": best.get("input", 0) + 1.25 * best.get("cacheWrite", 0), "cache": best.get("cacheRead", 0),
            "out": best.get("output", 0), "src": "openclaw"}


def extract(trial_dir: str | Path) -> dict:
    d = Path(trial_dir)
    ar = (json.loads((d / "result.json").read_text()).get("agent_result") or {}) if (d / "result.json").exists() else {}
    i, c, o = ar.get("n_input_tokens") or 0, ar.get("n_cache_tokens") or 0, ar.get("n_output_tokens") or 0
    if i or o:
        return {"in": max(0, i - c), "cache": c, "out": o, "src": "harbor"}
    for f in (_hermes, _openclaw):
        r = f(d)
        if r:
            return r
    return {"in": 0, "cache": 0, "out": 0, "src": "none"}


def usd(t: dict) -> float:
    return (t["in"] * P_IN + t["cache"] * P_CACHE + t["out"] * P_OUT) / 1e6


if __name__ == "__main__":
    import sys
    for j in sys.argv[1:]:
        rows = [extract(p.parent) for p in Path(j).glob("*/result.json")]
        print(j, len(rows), "trials", f"${sum(usd(r) for r in rows):.4f}", {r["src"] for r in rows})
