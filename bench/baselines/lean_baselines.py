"""Existing Lean proof checkers vs the Lean red team (bench/redteam/lean/manifest.json): every cheat and honest solution.

  python bench/baselines/lean_baselines.py      # Lean v4.23.0-rc2 (MILLENNIUM_LEAN or elan); WORKERS=6, FRESH=1 default

Writes runs/redteam/lean_baselines.jsonl (one row per case) and prints a markdown summary. Configurations, each set up
the way its documentation says to check an untrusted solution:

  (a) comparator   Lean FRO's judge (github.com/leanprover/comparator, HEAD ca04cfc), ported to v4.23.0-rc2 in
                   fro_tools_v423.lean. Challenge.lean = the frozen task as the agent received it minus the postamble
                   (vc-preamble, vc-helpers, vc-definitions and vc-theorems with their `sorry`s; the postamble's #eval
                   tests cannot run on a `sorry` implementation, so a Challenge that kept them would not build), built
                   first; Solution.lean = the submitted file. Config per its README: theorem_names = the frozen theorems,
                   definition_names = the frozen implementation holes (the `sorry` defs), permitted_axioms = propext,
                   Quot.sound, Classical.choice. Both modules must build (lake treats any error as a failed build).
  (b) lean4checker the file compiles (no error), `lean4checker Solution` (kernel replay of the module's .olean on top of
                   its imports; lean4checker tag v4.23.0-rc2, replay' embedded verbatim in fro_tools_v423.lean) succeeds,
                   and `#print axioms T` lists only the three standard axioms for every frozen theorem T.
                   (b+) is the same with `lean4checker --fresh` (every constant replayed into an empty environment).
  (c) benchmark    `lean Solution.lean` exits 0 with no error and no "declaration uses 'sorry'" (L1 in
                   bench/redteam/lean/baseline.py, reused here).
  (d) (c)+axioms   (c) and the #print axioms whitelist for every frozen theorem (L2 in baseline.py).
  (a)+(d)          both: the strongest combination of existing tools (comparator does not run the task's tests and
                   leaves definition holes to "an additional verifier"; the compiled file does run them).
  millennium       millennium_harness.lean.check (strict), for reference.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RT = ROOT / "bench" / "redteam" / "lean"
TOOLS = HERE / "fro_tools_v423.lean"
sys.path.insert(0, str(ROOT))
from millennium_harness import lean  # noqa: E402

_spec = importlib.util.spec_from_file_location("lean_redteam_baseline", RT / "baseline.py")
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)  # plain(): L0/L1/L2 (+ #print axioms) on the pinned toolchain

LEAN = lean.find_lean()
FRESH = os.environ.get("FRESH", "1") == "1"
AXIOMS = ["propext", "Quot.sound", "Classical.choice"]
HARMLESS = {"cheats/skip_kernel_tc.lean", "cheats/skip_kernel_tc_guillemet.lean"}  # manifest: "the proof is real"
CHALLENGES = Path(tempfile.mkdtemp(prefix="sl-comparator-challenges-"))
_lock = threading.Lock()


def _env(d) -> dict:
    return {"PATH": "/usr/bin:/bin", "HOME": str(d), "LANG": "C.UTF-8", "LEAN_ABORT_ON_PANIC": "1"}


def build(d: Path, module: str, timeout: int = 300) -> tuple[bool, str]:
    """`lake build MODULE` for a one-file library: compile to MODULE.olean; any error fails the build."""
    try:
        p = subprocess.run([LEAN, "-o", f"{module}.olean", "-i", f"{module}.ilean", f"{module}.lean"], cwd=d,
                           env=_env(d), capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = p.stdout + p.stderr
    ok = p.returncode == 0 and (d / f"{module}.olean").exists() and not any(
        m["severity"] == "error" for m in lean._messages(out))
    return ok, out


def tool(*args: str, d: Path, timeout: int = 900) -> dict:
    try:
        p = subprocess.run([LEAN, "--run", str(TOOLS), *args], cwd=d, env=_env(d), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "timeout", "error": "timeout"}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if not lines:  # a crash (e.g. a panic under LEAN_ABORT_ON_PANIC) is a rejection, as in comparator
        return {"ok": False, "stage": "crash", "error": (p.stdout + p.stderr)[-800:]}
    return json.loads(lines[-1])


@cache
def frozen(task: str) -> dict:
    return baseline.frozen(task)


@cache
def challenge(task: str) -> Path:
    """Challenge.lean/.olean for a task, built once and before any solution (comparator's assumption 2)."""
    rec = json.loads((RT / "tasks" / f"{task}.json").read_text(encoding="utf-8"))
    d = CHALLENGES / task
    d.mkdir(parents=True, exist_ok=True)
    (d / "Challenge.lean").write_text(lean.skeleton({**rec, "vc-postamble": ""}), encoding="utf-8")
    ok, out = build(d, "Challenge")
    if not ok:
        raise RuntimeError(f"challenge {task} does not build: {out[-800:]}")
    return d


def run_case(kind: str, c: dict) -> dict:
    text = (RT / c["file"]).read_text(encoding="utf-8")
    fz = frozen(c["task"])
    theorems = [n for n, x in fz["constants"].items() if x["role"] == "theorem"]
    impls = [n for n, x in fz["constants"].items() if x["role"] == "impl"]
    p = baseline.plain(text, theorems)
    ax_ok = all(t in p["axioms"] and set(p["axioms"][t]) <= lean.STD_AXIOMS for t in theorems)
    with tempfile.TemporaryDirectory(prefix="sl-fro-") as td:
        d = Path(td)
        ch = challenge(c["task"])
        for f in ("Challenge.lean", "Challenge.olean", "Challenge.ilean"):
            shutil.copy(ch / f, d / f)
        (d / "Solution.lean").write_text(text, encoding="utf-8")
        built, out = build(d, "Solution")
        cfg = {"challenge_module": "Challenge", "solution_module": "Solution", "theorem_names": theorems,
               "definition_names": impls, "permitted_axioms": AXIOMS}
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        skip = {"ok": False, "stage": "build", "error": "Solution does not build"}
        comp = tool("comparator", str(d), str(d / "config.json"), d=d) if built else skip
        l4c = tool("lean4checker", str(d), "Solution", d=d) if built else skip
        l4f = (tool("lean4checker", "--fresh", str(d), "Solution", d=d) if built else skip) if FRESH else None
    sl = lean.check(text, fz, LEAN)["status"]
    row = {
        "kind": kind, "file": c["file"], "task": c["task"], "what": c.get("what", ""), "harmless": c["file"] in HARMLESS,
        "builds": built, "axioms": p["axioms"], "axioms_ok": ax_ok,
        "comparator": comp["ok"], "comparator_stage": comp.get("stage"), "comparator_error": comp["error"][:400],
        "lean4checker": l4c["ok"], "lean4checker_error": l4c["error"][:400],
        "b_lean4checker_axioms": built and l4c["ok"] and ax_ok,
        "c_benchmark": p["L1"], "d_benchmark_axioms": p["L2"],
        "a_plus_d": comp["ok"] and p["L2"], "millennium": sl, "millennium_accepts": sl == "PROVED",
    }
    if l4f is not None:
        row["lean4checker_fresh"] = l4f["ok"]
        row["b_fresh_axioms"] = built and l4f["ok"] and ax_ok
    with _lock:
        print(f"{kind:6} {Path(c['file']).stem:40} comp={comp['ok']!s:5} ({comp.get('stage')}) b={row['b_lean4checker_axioms']!s:5} "
              f"c={p['L1']!s:5} d={p['L2']!s:5} SL={sl}", file=sys.stderr, flush=True)
    return row


CONFIGS = [("(a) comparator (Lean FRO)", "comparator"),
           ("(b) lean4checker + #print axioms", "b_lean4checker_axioms"),
           ("(b+) lean4checker --fresh + #print axioms", "b_fresh_axioms"),
           ("(c) benchmark rule: exit 0, no sorry warning", "c_benchmark"),
           ("(d) (c) + #print axioms whitelist", "d_benchmark_axioms"),
           ("(a)+(d) comparator and (d)", "a_plus_d"),
           ("Millennium Harness (ours)", "millennium_accepts")]


def summary(rows: list[dict]) -> str:
    ch = [r for r in rows if r["kind"] == "cheat"]
    harm = [r for r in ch if not r["harmless"]]
    ho = [r for r in rows if r["kind"] == "honest"]
    out = [f"| checker configuration | cheats accepted (n/{len(ch)}) | harmful cheats accepted (n/{len(harm)}) | "
           f"honest accepted (n/{len(ho)}) |", "|---|---|---|---|"]
    for label, key in CONFIGS:
        if key not in rows[0]:
            continue
        out.append(f"| {label} | {sum(r[key] for r in ch)}/{len(ch)} | {sum(r[key] for r in harm)}/{len(harm)} | "
                   f"{sum(r[key] for r in ho)}/{len(ho)} |")
    return "\n".join(out)


def main():
    m = json.loads((RT / "manifest.json").read_text(encoding="utf-8"))
    cases = [("honest", c) for c in m["honest"]] + [("cheat", c) for c in m["cheats"]]
    for t in sorted({c["task"] for _, c in cases}):  # lock and build every Challenge serially, before any solution
        frozen(t)
        challenge(t)
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        rows = list(ex.map(lambda j: run_case(*j), cases))
    out = ROOT / "runs" / "redteam"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "lean_baselines.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.rmtree(CHALLENGES, ignore_errors=True)
    print(summary(rows))
    for key in ("comparator", "a_plus_d"):
        acc = [Path(r["file"]).stem for r in rows if r["kind"] == "cheat" and r[key]]
        rej = [Path(r["file"]).stem for r in rows if r["kind"] == "honest" and not r[key]]
        print(f"\n{key}: cheats accepted: {acc}\n{key}: honest rejected: {rej}")


if __name__ == "__main__":
    main()
