"""Combine fresh-regrade rows of a model's probe jobs (first job + rerun of its infra-failed tasks) into one table row.
  python bench/fv/combine.py runs/fv/grades/<job1>.jsonl runs/fv/grades/<job2>.jsonl ..."""
import json
import sys

rows = {}
for f in sys.argv[1:]:
    for r in map(json.loads, open(f)):
        if not r.get("infra"):
            rows[r["task"]] = r  # later jobs (reruns) only cover tasks whose first trial was infra
st = lambda x: (x or {}).get("strict")  # noqa: E731
n = len(rows)
A = sum(st(r["A"]) == "PROVED" for r in rows.values())
B = sum(st(r["B"]) == "PROVED" for r in rows.values())
C = sum(st(r["C"]) == "PROVED" for r in rows.values() if r.get("C"))
fired = sum(bool(r["b_fired"]) for r in rows.values())
cost = {k: sum(((r.get("usage") or {}).get(k) or {}).get("cost", 0) for r in rows.values()) / max(1, n) for k in "ABC"}
diff = [(t, st(r["A"]), st(r["B"]), st(r["C"])) for t, r in sorted(rows.items()) if len({st(r["A"]), st(r["B"]), st(r["C"])}) > 1]
print(json.dumps({"n": n, "A": A, "B": B, "C": C, "fired": fired, "cost_per_task": {k: round(v, 4) for k, v in cost.items()},
                  "differences": diff}, indent=1))
