"""Paired analysis of Harbor jobs (arms = job names). Task-level bootstrap CIs + sign-flip permutation p-values.

  python bench/analyze.py runs/jobs A=e2dev-none B=e2dev-sorryless C=e2dev-selfstress [--vs A]
Seeds: pass several job names for one arm as A=job1,job2 (seed-mean per task).
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PRICE = {"in": 0.10, "cache": 0.01, "out": 0.50}


def load_job(root: Path, job: str) -> dict:
    out = {}
    for rj in (root / job).glob("*/result.json"):
        d = json.loads(rj.read_text())
        task = d.get("task_name")
        exc = (d.get("exception_info") or {}).get("exception_type")
        vr = (d.get("verifier_result") or {}).get("rewards") or {}
        if exc == "RewardFileNotFoundError":  # agent finished but produced no solution -> failure, not infra
            vr = {"reward": 0.0}
        ar = d.get("agent_result") or {}

        def dur(k):
            t = d.get(k) or {}
            try:
                return (datetime.fromisoformat(t["finished_at"]) - datetime.fromisoformat(t["started_at"])).total_seconds()
            except Exception:  # noqa: BLE001
                return None
        gate = []
        gl = rj.parent / "agent" / "gate.jsonl"
        if gl.exists():
            gate = [json.loads(l) for l in gl.read_text().splitlines() if l.strip()]
        out.setdefault(task, []).append({
            "reward": vr.get("reward"), "exc": exc, "cost": ar.get("cost_usd"),
            "in": ar.get("n_input_tokens") or 0, "cache": ar.get("n_cache_tokens") or 0,
            "out": ar.get("n_output_tokens") or 0, "secs": dur("agent_execution"), "gate": gate})
    return out


def infra_fail(r):
    return r["reward"] is None


def boot_ci(xs, n=10000, seed=0):
    rng = random.Random(seed)
    k = len(xs)
    ms = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return ms[int(.025 * n)], ms[int(.975 * n)]


def perm_p(xs, n=20000, seed=1):
    rng = random.Random(seed)
    obs = abs(sum(xs))
    hits = sum(abs(sum(x if rng.random() < .5 else -x for x in xs)) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def main():
    root = Path(sys.argv[1])
    arms = {}
    for a in sys.argv[2:]:
        if "=" in a:
            k, v = a.split("=", 1)
            arms[k] = v.split(",")
    data = {}
    for arm, jobs in arms.items():
        merged = defaultdict(list)
        for j in jobs:
            for t, rs in load_job(root, j).items():
                merged[t] += rs
        data[arm] = merged
    tasks = sorted(set.intersection(*[set(d) for d in data.values()]))
    # exclude a task from ALL arms if any arm has an infra failure on all its runs
    bad = {t for t in tasks if any(all(infra_fail(r) for r in data[a][t]) for a in data)}
    tasks = [t for t in tasks if t not in bad]
    print(f"{len(tasks)} paired tasks (excluded {len(bad)} with infra failures)\n")
    score = {a: {t: sum((r["reward"] or 0) for r in data[a][t] if not infra_fail(r)) /
                 max(1, sum(not infra_fail(r) for r in data[a][t])) for t in tasks} for a in data}
    print(f"{'arm':4s} {'pass@1':>7s} {'$/task':>8s} {'$/solve':>8s} {'p50 s':>6s} {'tok_out':>8s}")
    for a in data:
        rs = [r for t in tasks for r in data[a][t] if not infra_fail(r)]
        cost = [(r["in"] - r["cache"]) * PRICE["in"] / 1e6 + r["cache"] * PRICE["cache"] / 1e6 + r["out"] * PRICE["out"] / 1e6
                for r in rs]
        secs = sorted(r["secs"] for r in rs if r["secs"] is not None)
        p = sum(score[a].values()) / len(tasks)
        solves = sum(r["reward"] or 0 for r in rs)
        print(f"{a:4s} {100*p:6.1f}% {sum(cost)/len(rs):8.4f} {sum(cost)/max(1,solves):8.4f} "
              f"{secs[len(secs)//2] if secs else float('nan'):6.0f} {sum(r['out'] for r in rs)/len(rs):8.0f}")
    base = "A" if "A" in data else list(data)[0]
    for a in data:
        if a == base:
            continue
        d = [score[a][t] - score[base][t] for t in tasks]
        lo, hi = boot_ci(d)
        win = sum(x > 0 for x in d); loss = sum(x < 0 for x in d)
        print(f"\n{a} − {base}: {100*sum(d)/len(d):+.1f}pp  95% CI [{100*lo:+.1f}, {100*hi:+.1f}]  "
              f"perm p={perm_p(d):.3f}  wins={win} losses={loss}")
    for a in data:
        g = [r["gate"] for t in tasks for r in data[a][t] if r["gate"]]
        if not g:
            continue
        fired = sum(any(x.get("fired") for x in gg) for gg in g)
        statuses = defaultdict(int)
        for gg in g:
            statuses[gg[0].get("status")] += 1
        print(f"\n[{a}] gate: fired on {fired}/{len(g)} trials; first-check statuses {dict(statuses)}")


if __name__ == "__main__":
    main()
