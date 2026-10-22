"""Seeded DEV/TEST split of the FV pool by component (about 30% DEV, 70% TEST).

  .venv/bin/python -m bench.fv.split       # reads runs/fv/pool.json, writes runs/fv/split.json

A component (key = its smallest member id, from pool.py) is DEV if int(sha256("fv-split-v2|" + key)[:8], 16) % 10 < 3,
otherwise TEST; components holding an E4b task are forced to DEV. Per cell, the DEV and the TEST component lists are
each shuffled with random.Random(SEED). TEST task content is not read before the tag; TEST tasks come from pick_test().
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "runs" / "fv" / "pool.json"
OUT = ROOT / "runs" / "fv" / "split.json"
SEED = 20261012
SALT = "fv-split-v2|"


def is_dev(comp: str, forced=frozenset()) -> bool:
    return comp in forced or int(hashlib.sha256((SALT + comp).encode()).hexdigest()[:8], 16) % 10 < 3


def build(pool: dict, pool_sha256: str) -> dict:
    forced = frozenset(pool["e4b"]["components"])
    out = {"salt": SALT, "seed": SEED, "pool_sha256": pool_sha256, "counts": {}, "dev_ids": {}, "dev": {}, "test": {}}
    for cell, ids in pool["cells"].items():
        comps = defaultdict(list)
        for i in ids:
            comps[pool["component"][i]].append(i)
        for part in ("dev", "test"):
            keys = sorted(c for c in comps if is_dev(c, forced) == (part == "dev"))
            random.Random(SEED).shuffle(keys)
            out[part][cell] = [{"comp": c, "ids": sorted(comps[c])} for c in keys]
        out["dev_ids"][cell] = [i for c in out["dev"][cell] for i in c["ids"]]
        out["counts"][cell] = {
            "dev_components": len(out["dev"][cell]), "dev_tasks": len(out["dev_ids"][cell]),
            "test_components": len(out["test"][cell]), "test_tasks": sum(len(c["ids"]) for c in out["test"][cell]),
            "forced_dev_components": sum(c in forced for c in comps)}
    tests = {c["comp"] for cell in out["test"].values() for c in cell}
    out["counts"]["all"] = {"test_components": len(tests),
                            "dev_components": len({c["comp"] for cell in out["dev"].values() for c in cell})}
    return out


def pick_test(split: dict, cells, seed: int = SEED) -> dict[str, list[str]]:
    """At most 1 task per TEST component: a seeded draw among `cells` holding it, then a seeded task within that cell.
    Returns {cell: [ids]} in each cell's seeded TEST order."""
    rng = random.Random(seed)
    holders = defaultdict(list)
    for cell in cells:
        for c in split["test"][cell]:
            holders[c["comp"]].append(cell)
    owner = {k: rng.choice(sorted(v)) for k, v in sorted(holders.items())}
    picked = {cell: [] for cell in cells}
    for cell in cells:
        for c in split["test"][cell]:
            if owner[c["comp"]] == cell:
                picked[cell].append(rng.choice(c["ids"]))
    return picked


def main():
    raw = POOL.read_bytes()
    split = build(json.loads(raw), hashlib.sha256(raw).hexdigest())
    OUT.write_text(json.dumps(split, indent=1) + "\n")
    for cell, c in split["counts"].items():
        print(f"{cell:7s} {c}")
    print(f"pool sha256 {split['pool_sha256']} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
