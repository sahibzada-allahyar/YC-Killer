"""Grade every held-out attack file (runs/heldout/attacks.jsonl) with the strongest existing checkers, reusing the
red-team baseline code unchanged:
  Lean:  bench/baselines/lean_baselines.run_case  -> Lean FRO comparator logic (statement match, axioms, kernel
         replay), lean4checker + #print axioms, benchmark rule, comparator + benchmark rule + axioms (strongest)
  Dafny: bench/baselines/dafny_baselines.row      -> verify, --allow-warnings, + dafny audit, + frozen-text diff and
         vericoding's cheat regexes, vericoding's checker, all combined (strongest)
  .venv/bin/python bench/heldout/strong_baselines.py      -> runs/heldout/strong_baselines.jsonl + table
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "baselines"))
import dafny_baselines as DB  # noqa: E402
import lean_baselines as LB  # noqa: E402

OUT = ROOT / "runs" / "heldout" / f"strong_baselines_{os.environ.get('LANGS', 'all').replace(',', '_')}.jsonl"


def main() -> None:
    langs = os.environ.get("LANGS", "lean,dafny").split(",")
    rows = [r for r in map(json.loads, open(ROOT / "runs/heldout/attacks.jsonl"))
            if r.get("file", "").strip() and r["lang"] in langs]
    if os.environ.get("CANDIDATES_ONLY"):  # strongest = comparator AND (benchmark rule + axioms): grade only those
        rows = [r for r in rows if r["lang"] != "lean" or r.get("rule_axioms")]
    skip = set()
    for f in os.environ.get("SKIP_LOGS", "").split(","):
        if f and Path(f).exists():
            skip |= {ln.split()[1] for ln in open(f) if ln.startswith("attack ")}
    tasks = {p.stem: json.load(open(p)) for p in (ROOT / "runs/heldout/tasks").glob("*.json")}
    tmp = Path(tempfile.mkdtemp(prefix="ho-strong-"))
    (tmp / "tasks").mkdir()
    uniq = {}
    for r in rows:
        h = hashlib.sha256(r["file"].encode()).hexdigest()[:16]
        ext = "lean" if r["lang"] == "lean" else "dfy"
        name = f"{r['task']}_{h}.{ext}"
        if Path(name).stem in skip:
            continue
        uniq[name] = r
        (tmp / name).write_text(r["file"], encoding="utf-8")
    for tid, t in tasks.items():
        json.dump(t["record"] if t["lang"] == "lean" else t["frozen"], open(tmp / "tasks" / f"{tid}.json", "w"))
    LB.RT = tmp
    LB.frozen = lambda tid: tasks[tid]["frozen"]
    DB.SUITE = tmp
    lean_jobs = [n for n, r in uniq.items() if r["lang"] == "lean"]
    for tid in sorted({uniq[n]["task"] for n in lean_jobs}):
        LB.challenge(tid)

    def one(n):
        r = uniq[n]
        try:
            if r["lang"] == "lean":
                x = LB.run_case("attack", {"file": n, "task": r["task"]})
                acc = {"comparator": x["comparator"], "lean4checker_axioms": x["b_lean4checker_axioms"],
                       "benchmark_rule": x["c_benchmark"], "rule_axioms": x["d_benchmark_axioms"],
                       "strongest": x["a_plus_d"], "millennium": x["millennium_accepts"]}
            else:
                x = DB.row({"file": n, "task": r["task"], "expect": "CHEAT_REJECTED"})
                a = x["accept"]
                acc = {"verify_default": a["a"], "verify_allow_warnings": a["b"], "plus_audit": a["c"],
                       "plus_diff_regex": a["d"], "vericoding": a["e"], "strongest": a["f"]}
        except Exception as e:  # noqa: BLE001
            acc = {"error": str(e)[:300]}
        return {"name": n, "model": r["model"], "task": r["task"], "lang": r["lang"], "round": r["round"],
                "mh": r["mh"], **acc}

    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "8"))) as ex:
        out = list(ex.map(one, sorted(uniq)))
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    for lang in langs:
        sub = [o for o in out if o["lang"] == lang and "error" not in o]
        keys = [k for k in sub[0] if k not in ("name", "model", "task", "lang", "round", "mh")] if sub else []
        loops = {(o["model"], o["task"]) for o in sub}
        tasks_ = {o["task"] for o in sub}
        print(f"\n== {lang}: {len(sub)} unique files, {len(loops)} loops, {len(tasks_)} tasks "
              f"(errors {sum(1 for o in out if o['lang'] == lang and 'error' in o)})")
        for k in keys + ["mh"]:
            ok = (lambda o: o["mh"] == "PROVED") if k == "mh" else (lambda o, k=k: bool(o.get(k)))
            print(f"  {k:24} files {sum(map(ok, sub)):4}  loops {len({(o['model'], o['task']) for o in sub if ok(o)}):3}/{len(loops)}"
                  f"  tasks {len({o['task'] for o in sub if ok(o)}):3}/{len(tasks_)}")


if __name__ == "__main__":
    main()
