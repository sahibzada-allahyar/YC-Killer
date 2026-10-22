"""Final held-out table: tasks / attack loops / submitted files accepted by each checker (submission level).
Strongest existing: Lean = comparator logic AND benchmark rule + #print axioms (comparator graded on every file that
passes rule+axioms; see strong_baselines.py, logs runs/heldout/strong_lean_part1.log + strong_baselines_lean.jsonl);
Dafny = runs/heldout/strong_baselines_dafny.jsonl 'strongest'.
  .venv/bin/python bench/heldout/table.py   -> runs/heldout/table.json"""
import hashlib
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "runs" / "heldout"
rows = [r for r in map(json.loads, open(R / "attacks.jsonl")) if "mh" in r]
name = lambda r: f"{r['task']}_{hashlib.sha256(r['file'].encode()).hexdigest()[:16]}"  # noqa: E731
comp = {}
for ln in open(R / "strong_lean_part1.log"):
    m = re.match(r"attack (\S+)\s+comp=(\w+).* d=(\w+)", ln)
    if m:
        comp[m.group(1)] = m.group(2) == "True" and m.group(3) == "True"
for o in map(json.loads, open(R / "strong_baselines_lean.jsonl")):
    comp[Path(o["name"]).stem] = bool(o["strongest"])
dstrong = {Path(o["name"]).stem: bool(o["strongest"]) for o in map(json.loads, open(R / "strong_baselines_dafny.jsonl"))}


def strongest(r):
    if not r["file"].strip():
        return False
    if r["lang"] == "lean":
        return bool(r.get("rule_axioms")) and comp.get(name(r), None) is True
    return dstrong.get(name(r), False)


crit = {"attacked": lambda r: True, "exit0": lambda r: bool(r.get("exit0")),
        "benchmark_rule": lambda r: bool(r.get("benchmark_rule")), "rule_axioms": lambda r: bool(r.get("rule_axioms")),
        "strongest": strongest, "mh": lambda r: r["mh"] == "PROVED"}
missing = sum(1 for r in rows if r["lang"] == "lean" and r.get("rule_axioms") and name(r) not in comp)
out = {}
for lang in ("lean", "dafny"):
    sub = [r for r in rows if r["lang"] == lang]
    out[lang] = {k: {"tasks": len({r["task"] for r in sub if f(r)}),
                     "loops": len({(r["model"], r["task"]) for r in sub if f(r)}),
                     "files": sum(1 for r in sub if f(r))} for k, f in crit.items()}
json.dump(out, open(R / "table.json", "w"), indent=1)
for lang, d in out.items():
    for k, v in d.items():
        print(f"{lang:6} {k:15} {v}")
print("lean rule+axioms files without comparator grade:", missing)
