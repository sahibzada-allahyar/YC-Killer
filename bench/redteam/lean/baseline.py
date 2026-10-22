"""Lean red team against three plain-Lean acceptance criteria and the Millennium Harness rung (strict and
lenient).

  python bench/redteam/lean/baseline.py        # needs Lean v4.23.0-rc2 (MILLENNIUM_LEAN or elan); prints a table + JSON

For every case in manifest.json (honest solutions and cheats), with the pinned toolchain:
  L0  `lean Solution.lean` reports no error (exit 0): what "it compiles" means;
  L1  L0 and no `declaration uses 'sorry'` warning: the usual benchmark criterion;
  L2  L1 and `#print axioms T` (appended after the file) lists only propext / Classical.choice / Quot.sound for every
      frozen theorem T of the task: the first rung of the Lean reference's proof-validation ladder;
  SL  millennium_harness.lean.check (strict: text ban scan on), and SL-k (lenient: kernel-side checks only).
Per-case rows go to runs/redteam/lean_baseline.jsonl.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from millennium_harness import lean  # noqa: E402

LEAN = lean.find_lean()
AX = re.compile(r"'([^']+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)")


@cache
def frozen(task: str) -> dict:
    return lean.lock(json.loads((HERE / "tasks" / f"{task}.json").read_text(encoding="utf-8")), LEAN)


def plain(text: str, theorems: list[str]) -> dict:
    with tempfile.TemporaryDirectory(prefix="sl-base-") as d:
        ok, out, timed_out = lean._compile(text, d, LEAN, 300)
    msgs = lean._messages(out)
    l0 = ok and not timed_out and not any(m["severity"] == "error" for m in msgs)
    l1 = l0 and not any("declaration uses 'sorry'" in m["text"] for m in msgs)
    tail = "".join(f"\n#print axioms {t}" for t in theorems)
    with tempfile.TemporaryDirectory(prefix="sl-base-") as d:
        ok2, out2, to2 = lean._compile(text.rstrip("\n") + "\n" + tail + "\n", d, LEAN, 300)
    msgs2 = lean._messages(out2)
    seen = {}
    for name, axs in AX.findall(out2):  # `lean FILE` prints info output bare (no `file:line:col: info:` prefix)
        seen[name] = {a.strip() for a in axs.split(",") if a.strip()}
    ax_ok = (ok2 and not to2 and not any(m["severity"] == "error" for m in msgs2)
             and all(t in seen and seen[t] <= lean.STD_AXIOMS for t in theorems))
    return {"L0": l0, "L1": l1, "L2": l1 and ax_ok, "axioms": {k: sorted(v) for k, v in seen.items()}}


def run_case(kind: str, c: dict) -> dict:
    text = (HERE / c["file"]).read_text(encoding="utf-8")
    fz = frozen(c["task"])
    theorems = [n for n, x in fz["constants"].items() if x["role"] == "theorem"]
    p = plain(text, theorems)
    strict = lean.check(text, fz, LEAN)["status"]
    lenient = lean.check(text, fz, LEAN, bans=False)["status"]
    exp_ok = (strict == "PROVED") if kind == "honest" else (
        strict == c["strict"] and (c["lenient"] is None or lenient == c["lenient"]))
    return {"kind": kind, "file": c["file"], "task": c["task"], "what": c.get("what", ""), **p,
            "SL": strict, "SL_lenient": lenient, "as_expected": exp_ok}


def main():
    m = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    for t in {c["task"] for c in m["honest"] + m["cheats"]}:
        frozen(t)  # lock serially (cache), then grade in parallel
    jobs = [("honest", c) for c in m["honest"]] + [("cheat", c) for c in m["cheats"]]
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as ex:
        rows = list(ex.map(lambda j: run_case(*j), jobs))
    out = ROOT / "runs" / "redteam"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "lean_baseline.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("| case | L0 compiles | L1 +no sorry | L2 +axioms | Millennium Harness | lenient | ok |\n"
          "|---|---|---|---|---|---|---|")
    yn = {True: "accepts", False: "rejects"}
    for r in rows:
        print(f"| {r['kind']}:{Path(r['file']).stem} | {yn[r['L0']]} | {yn[r['L1']]} | {yn[r['L2']]} | {r['SL']} | "
              f"{r['SL_lenient']} | {'yes' if r['as_expected'] else 'NO'} |")
    summary = {}
    for kind in ("honest", "cheat"):
        rs = [r for r in rows if r["kind"] == kind]
        summary[kind] = {"n": len(rs), **{k: sum(r[k] for r in rs) for k in ("L0", "L1", "L2")},
                         "millennium_PROVED": sum(r["SL"] == "PROVED" for r in rs),
                         "lenient_PROVED": sum(r["SL_lenient"] == "PROVED" for r in rs),
                         "as_expected": sum(r["as_expected"] for r in rs)}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
