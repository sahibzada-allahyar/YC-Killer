"""Spend so far: Batch/generator ledger + every Harbor job (recomputed from tokens with one Haiku 5.5 price table).

  python bench/spend.py [--cap 120]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P_IN, P_CACHE, P_OUT = 0.10, 0.01, 0.50


def job_cost(job: Path) -> tuple[float, int]:
    usd, n = 0.0, 0
    for rj in job.glob("*/result.json"):
        ar = json.loads(rj.read_text()).get("agent_result") or {}
        i, c, o = ar.get("n_input_tokens") or 0, ar.get("n_cache_tokens") or 0, ar.get("n_output_tokens") or 0
        rep = ar.get("cost_usd")
        est = (max(0, i - c) * P_IN + c * P_CACHE + o * P_OUT) / 1e6
        usd += rep if rep is not None else est
        n += 1
    return usd, n


def main():
    cap = float(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 120.0
    led = [json.loads(l) for l in open(ROOT / "runs" / "ledger.jsonl")] if (ROOT / "runs" / "ledger.jsonl").exists() else []
    batch = sum(x["usd"] for x in led)
    jobs = 0.0
    rows = []
    for j in sorted((ROOT / "runs" / "jobs").glob("*")):
        if j.is_dir():
            c, n = job_cost(j)
            jobs += c
            rows.append((j.name, n, c))
    for name, n, c in rows:
        print(f"  {name:28s} {n:4d} trials  ${c:7.3f}")
    print(f"batch/generators ${batch:.2f} + agent jobs ${jobs:.2f} = ${batch + jobs:.2f} of ${cap:.0f} "
          f"({100 * (batch + jobs) / cap:.1f}%)")


if __name__ == "__main__":
    main()
