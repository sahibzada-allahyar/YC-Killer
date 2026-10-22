"""Plain-Dafny acceptance on the red-team cases, in both warning modes (no Millennium Harness involved).

  MILLENNIUM_DAFNY=~/.local/dafny-4.11/dafny/dafny python bench/redteam/dafny/baselines.py

D0  `dafny verify FILE` exits 0 (default: a warning, e.g. the one `assume` prints, makes the exit code non-zero);
D1  `dafny verify --allow-warnings FILE` exits 0 (the "dafny alone" column of run.py);
and whether the summary line reads "N verified, 0 errors" (what a reader of the log sees).
Rows go to runs/redteam/dafny_baselines.jsonl.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DAFNY = os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY", "dafny"))


def verify(path: Path, *flags: str) -> tuple[bool, bool]:
    p = subprocess.run([DAFNY, "verify", *flags, path.name], cwd=path.parent, capture_output=True, text=True,
                       timeout=300)
    zero = re.search(r"finished with \d+ verified, 0 errors", p.stdout + p.stderr) is not None
    return p.returncode == 0, zero


def row(c: dict) -> dict:
    path = HERE / c["file"]
    d0, z0 = verify(path)
    d1, _ = verify(path, "--allow-warnings")
    kind = "honest" if c["expect"] == "PROVED" else "invalid_task" if c["file"].startswith("pool/") else "cheat"
    return {"file": c["file"], "kind": kind, "D0": d0, "D1": d1, "zero_errors_line": z0}


def main():
    cases = json.loads((HERE / "cases.json").read_text())
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        rows = list(ex.map(row, cases))
    out = ROOT / "runs" / "redteam"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dafny_baselines.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for kind in ("honest", "cheat", "invalid_task"):
        rs = [r for r in rows if r["kind"] == kind]
        print(kind, {"n": len(rs), "D0_default": sum(r["D0"] for r in rs), "D1_allow_warnings": sum(r["D1"] for r in rs),
                     "summary_says_0_errors": sum(r["zero_errors_line"] for r in rs)})


if __name__ == "__main__":
    main()
