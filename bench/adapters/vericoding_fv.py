"""vericoding -> Harbor tasks for the Millennium Harness FV study (v1.0; docs/research/win-plan-synthesis.md §2-§3).

  .venv/bin/python bench/adapters/vericoding_fv.py build OUT_DIR --cell L-apps --ids-from runs/fv/split.json \
      --part dev --n 30 [--skip 0]
  --part test takes --cells (the GO cells; one task per TEST component via bench.fv.split.pick_test) and refuses to
  run until the v1.0-fv-frozen tag exists. Lean cells need --frozen-dir/<ID>.json (`python -m
  millennium_harness.lean lock`, which compiles with the pinned Lean) unless that Lean is installed locally.

Lean cells write /app/Solution.lean (preamble, HELPERS markers, then the definitions and theorems with their `sorry`
holes, then the postamble). Dafny cells write /app/solution.dfy the same way bench/adapters/vericoding.py does.
There is one instruction template per language, identical across arms. task.toml sets a non-root [agent] user,
allowlist egress to the model API only, and an agent timeout of 4500 s (A 1200 + B 600 + C 600 + rung checks). tests/test.sh grades with the same rung
command as the in-sandbox gate (bench.agents_fv.rung_cmd). It writes grade.json and reward 1 only for PROVED.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.agents_fv import SOLUTION, rung_cmd  # noqa: E402
from bench.fv.pool import CELLS  # noqa: E402
from millennium_harness import lean as sl_lean  # noqa: E402
from millennium_harness.dafny import HELPERS_BEGIN, HELPERS_END  # noqa: E402

LEAN_BEGIN = "-- ===== HELPERS: you may add definitions and lemmas here ====="
LEAN_END = "-- ===== END HELPERS ====="
DATA = ROOT / "runs" / "dafny" / "vericoding.jsonl"
FROZEN_DIR = ROOT / "runs" / "fv" / "frozen"  # <ID>.json = `python -m millennium_harness.lean lock` output (pinned Lean)
IMAGES = ROOT / "bench" / "fv" / "images"
API_HOSTS = ["api.anthropic.com"]
FROZEN_TAG = "v1.0-fv-frozen"

LEAN_INSTRUCTION = """=== TASK ===
Complete the Lean 4 file /app/Solution.lean (toolchain leanprover/lean4:v4.23.0-rc2; core and Std only, no Mathlib)
so that

    lean /app/Solution.lean

reports no errors and no `declaration uses 'sorry'` warning.
{description}
Rules:
- Replace every `sorry` with an implementation or a proof. You may add helper definitions and lemmas between the
  HELPERS markers or between top-level declarations.
- Everything else is frozen and must stay byte-identical: the preamble, each declaration header up to and including
  its `:=` (or `:= by`), every definition without a `sorry`, and everything after the last theorem.
- Implementations must be computable (they are compiled and may be executed).
- No escape hatches in /app/Solution.lean: `sorry`, `admit`, `axiom`, `native_decide` or any other native evaluation
  (`Lean.ofReduceBool`, `Lean.trustCompiler`), `unsafe`, `noncomputable`, `@[implemented_by]`, `@[extern]`,
  `@[export]`, `#exit`, `#eval`, `run_cmd`/`run_tac`/`run_elab`/`run_meta`, new `macro`/`syntax`/`notation`/`elab`
  declarations, `initialize`, `import`, `set_option debug.*` or `set_option compiler.*`. A file that uses any of them
  does not count.

You run autonomously: nobody will answer questions. When you stop, the current /app/Solution.lean is your answer.
"""

DAFNY_INSTRUCTION = """=== TASK ===
Complete the Dafny file /app/solution.dfy (Dafny 4.11) so that

    dafny verify /app/solution.dfy

reports 0 errors.
{description}
Rules:
- Edit ONLY the method body and the region between the HELPERS markers (where you may add lemmas, functions and
  predicates). The preamble, the signature with its requires/ensures, and anything after the body are frozen and must
  stay byte-identical.
- The code must also compile, not just verify (for example, no non-compilable `:|` in methods).
- No escape hatches: `assume`, `{{:axiom}}`, `{{:verify false}}`, `{{:extern}}`, `{{:compile false}}`,
  `decreases *`, `expect`, `module`, `import`, `include`, `refines`, and lemmas/functions/methods with `ensures` but
  no body. A file that uses any of them does not count.

You run autonomously: nobody will answer questions. When you stop, the current /app/solution.dfy is your answer.
"""


def instruction(lang: str, description: str) -> str:
    d = description.strip()[:4000]
    return (LEAN_INSTRUCTION if lang == "lean" else DAFNY_INSTRUCTION).format(
        description=f"\nTask description:\n{d}\n" if d else "")


def task_toml(lang: str, cell: str, rid: str, part: str) -> str:
    hosts = json.dumps(API_HOSTS)
    # The environment baseline uses the same allowlist as the agent phase. Then Modal needs no runtime policy switch,
    # setup installs nothing, and the verifier never uses the network.
    return f"""version = "1.0"

[metadata]
author_name = "vericoding (Beneficial AI Foundation, MIT) via Millennium Harness adapter"
category = "{lang}"
cell = "{cell}"
vericoding_id = "{rid}"
part = "{part}"

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 4500.0
user = "agent"
network_mode = "allowlist"
allowed_hosts = {hosts}

[environment]
build_timeout_sec = 1800.0
cpus = 1
memory_mb = 4096
storage_mb = 8192
network_mode = "allowlist"
allowed_hosts = {hosts}
"""


def test_sh(lang: str) -> str:
    sol = SOLUTION[lang]
    name, g = sol.rsplit("/", 1)[1], "/tmp/sl-grade"
    rung = rung_cmd(lang, f"{g}/{name}", f"{g}/frozen.json", f"{g}/w")
    return f"""#!/bin/bash
# Grade {sol} with the Millennium Harness rung. It runs as the unprivileged agent user on a root-owned copy, with
# agent processes killed before and after. Reward is 1 only for PROVED; grade.json is the rung's JSON verdict.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
G={g}
mkdir -p /logs/verifier
pkill -9 -u agent 2>/dev/null
rm -rf $G && mkdir -p $G/w && chmod 755 $G && chown agent:agent $G/w && chmod 700 $G/w
runuser -u agent -- cat {sol} > $G/{name}; cp /tests/frozen.json $G/frozen.json; chmod 644 $G/{name} $G/frozen.json
runuser -u agent -- bash -c {shlex.quote(rung)} > /logs/verifier/rung.out 2> /logs/verifier/rung.err
pkill -9 -u agent 2>/dev/null
tail -n 1 /logs/verifier/rung.out > /logs/verifier/grade.json
python3 -I -c "import json; print(1 if json.load(open('/logs/verifier/grade.json')).get('status') == 'PROVED' else 0)" \\
  > /logs/verifier/reward.txt 2>/dev/null || echo 0 > /logs/verifier/reward.txt
"""


def lean_files(r: dict, frozen_dir: str | Path = FROZEN_DIR) -> tuple[str, dict]:
    """The skeleton is millennium_harness.lean.skeleton() with HELPERS markers after the preamble. frozen.json is
    the millennium_harness.lean.lock() output: read from <frozen_dir>/<ID>.json (the lock compiles the skeleton with
    the pinned Lean, e.g. in the Lean image), or computed here if that Lean is installed. The starting file must pass
    its own chunk lock, and its holes may hold nothing banned except the `sorry`s the agent is asked to replace."""
    parts = [(r.get(k) or "").strip() for k in ("vc-preamble", "vc-definitions", "vc-theorems", "vc-postamble")]
    region = "\n".join((LEAN_BEGIN, (r.get("vc-helpers") or "").strip(), LEAN_END))
    sol = "\n\n".join(x for x in (parts[0], region, *parts[1:]) if x) + "\n"
    f = Path(frozen_dir) / f"{r['id']}.json"
    try:
        frozen = json.loads(f.read_text()) if f.exists() else sl_lean.lock(r)
    except sl_lean.LockError as e:
        raise SystemExit(f"{r['id']}: no {f} and it cannot be locked here ({e})")
    hs, v = sl_lean.holes(sol, frozen)
    v = v or [x for x in sl_lean.ban_scan(hs) if not x.startswith("banned construct: sorry ")]
    if v:
        raise SystemExit(f"{r['id']}: the starting file fails its own lock: {v}")
    return sol, frozen


def dafny_files(r: dict) -> tuple[str, dict]:
    pre, spec, post = r.get("vc-preamble") or "", r["vc-spec"], r.get("vc-postamble") or ""
    sol = f"{pre}\n\n{HELPERS_BEGIN}\n\n{HELPERS_END}\n\n{spec}\n{{\n  // implement\n}}\n\n{post}\n"
    return sol, {"preamble": pre, "spec": spec, "postamble": post}


def select_ids(split: dict, cell: str, part: str, n: int, skip: int = 0, cells: list[str] | None = None) -> list[str]:
    """DEV: the first id of each DEV component, in seeded order. TEST: bench.fv.split.pick_test over `cells`."""
    if part == "dev":
        ids = [c["ids"][0] for c in split["dev"][cell]]
    else:
        from bench.fv.split import pick_test
        ids = pick_test(split, cells or [cell])[cell]
    return ids[skip:skip + n]


def write_task(out: str | Path, r: dict, cell: str, part: str, frozen_dir: str | Path = FROZEN_DIR) -> Path:
    lang = CELLS[cell][0]
    sol, frozen = lean_files(r, frozen_dir) if lang == "lean" else dafny_files(r)
    td = Path(out) / r["id"].lower()
    if td.exists():
        shutil.rmtree(td)
    env = td / "environment"
    shutil.copytree(ROOT / "millennium_harness", env / "sl" / "millennium_harness",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(IMAGES / f"{lang}.Dockerfile", env / "Dockerfile")
    (env / SOLUTION[lang].rsplit("/", 1)[1]).write_text(sol)
    (td / "tests").mkdir()
    (td / "tests" / "frozen.json").write_text(json.dumps(frozen))
    (td / "tests" / "test.sh").write_text(test_sh(lang))
    os.chmod(td / "tests" / "test.sh", 0o755)
    (td / "task.toml").write_text(task_toml(lang, cell, r["id"], part))
    (td / "instruction.md").write_text(instruction(lang, r.get("vc-description") or ""))
    return td


def _tagged(tag: str = FROZEN_TAG) -> bool:
    out = subprocess.run(["git", "-C", str(ROOT), "tag", "-l", tag], capture_output=True, text=True).stdout
    return out.strip() == tag


def build(out: str, cell: str, ids_from: str, part: str, n: int, skip: int = 0, cells: list[str] | None = None,
          frozen_dir: str | Path = FROZEN_DIR, data: Path = DATA) -> list[Path]:
    if part == "test" and not _tagged():
        raise SystemExit(f"refusing: TEST tasks are built only after the {FROZEN_TAG} tag (plan §6)")
    ids = select_ids(json.loads(Path(ids_from).read_text()), cell, part, n, skip, cells)
    want, recs = set(ids), {}
    with open(data) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                recs[r["id"]] = r
    lang, sources = CELLS[cell]
    dirs = []
    for i in ids:
        r = recs[i]
        assert r["language"] == lang and r["source"] in sources, (i, r["language"], r["source"])
        dirs.append(write_task(out, r, cell, part, frozen_dir))
    print(f"built {len(dirs)} {cell} {part} tasks into {out} (skip {skip})")
    return dirs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("out")
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--ids-from", default=str(ROOT / "runs" / "fv" / "split.json"))
    ap.add_argument("--part", choices=["dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--skip", type=int, default=0, help="skip the first K components (e.g. L2 = the next 20)")
    ap.add_argument("--cells", default=None, help="TEST only: comma-separated GO cells for the component draw")
    ap.add_argument("--frozen-dir", default=str(FROZEN_DIR),
                    help="Lean only: <ID>.json from `millennium_harness.lean lock`")
    a = ap.parse_args()
    build(a.out, a.cell, a.ids_from, a.part, a.n, a.skip, a.cells.split(",") if a.cells else None, a.frozen_dir)


if __name__ == "__main__":
    main()
