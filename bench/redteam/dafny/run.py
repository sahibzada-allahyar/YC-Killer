"""Dafny red team for the ∎ rung: every cheat must be rejected for its intended reason, every honest file PROVED.

  MILLENNIUM_DAFNY=~/.local/dafny-4.11/dafny/dafny python bench/redteam/dafny/run.py     # prints the table

cases.json lists each file, its vericoding task (tasks/<id>.json holds the frozen preamble/spec/postamble verbatim),
the expected Millennium Harness status, and a substring of the violation that must explain a rejection (`verify_has`: a
substring the verifier output must contain). The "dafny alone" column is plain `dafny verify --allow-warnings`
(exit 0 = Dafny accepts the file as verified). Honest references use tasks that already have published vericoded
solutions (excluded from the FV study pools). Red-team round 2 cheats and pool/ cases use FV DEV tasks (never TEST):
pool/ files are tasks whose frozen text does not resolve (CHECK_ERROR: no solution can be certified), and unsat/ holds
Dafny-checked witnesses that a DEV spec is unsatisfiable (tests/test_dafny_rung.py verifies them; exclude those ids).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from millennium_harness import dafny  # noqa: E402


def run_case(c: dict) -> dict:
    path = HERE / c["file"]
    frozen = json.loads((HERE / "tasks" / f"{c['task']}.json").read_text())
    r = dafny.check(path.read_text(), frozen, str(path))
    exe = os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY", "dafny"))
    raw = subprocess.run([exe, "verify", "--allow-warnings", path.name],
                         cwd=path.parent, capture_output=True, text=True, timeout=300)
    why = c["reason"]
    ok = (r["status"] == c["expect"] and (not why or any(why in x for x in r["violations"]))
          and c.get("verify_has", "") in r.get("verify", ""))
    return {**c, "status": r["status"], "violations": r["violations"], "verify": r.get("verify", ""),
            "raw_accepts": raw.returncode == 0, "ok": ok}


def run_all(workers: int | None = None) -> list[dict]:
    cases = json.loads((HERE / "cases.json").read_text())
    with ThreadPoolExecutor(workers or os.cpu_count() or 4) as ex:
        return list(ex.map(run_case, cases))


def main():
    rows = run_all()
    print("| case | dafny alone | Millennium Harness | rejected because | ok |\n|---|---|---|---|---|")
    for r in rows:
        why = (r["violations"] or [""])[0].replace("|", "/").replace("\n", " ")[:90]
        print(f"| {Path(r['file']).stem} | {'accepts' if r['raw_accepts'] else 'rejects'} | {r['status']} | {why} "
              f"| {'yes' if r['ok'] else 'NO'} |")
    cheats = [r for r in rows if r["expect"] != "PROVED"]
    honest = [r for r in rows if r["expect"] == "PROVED"]
    print(f"\ncheats and invalid tasks rejected for the intended reason: {sum(r['ok'] for r in cheats)}/{len(cheats)} "
          f"(dafny verify alone accepts {sum(r['raw_accepts'] for r in cheats)}/{len(cheats)}); "
          f"honest PROVED: {sum(r['ok'] for r in honest)}/{len(honest)}")
    sys.exit(0 if all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
