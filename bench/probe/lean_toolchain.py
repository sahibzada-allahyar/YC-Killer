"""Infra probe (no model calls): which Lean toolchain elaborates vericoding Lean specs (bodies/proofs left as `sorry`)?
  modal run bench/probe/lean_toolchain.py
"""
import json, random
from pathlib import Path
import modal

app = modal.App("millennium-harness-probe-lean")
TOOLCHAINS = ["v4.22.0", "v4.24.0", "stable"]
image = (modal.Image.debian_slim(python_version="3.11").apt_install("curl", "git", "ca-certificates")
         .run_commands("curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none",
                       *[f"/root/.elan/bin/elan toolchain install leanprover/lean4:{t}" for t in TOOLCHAINS]))


def assemble(r):
    keys = ["vc-preamble", "vc-helpers", "vc-definitions", "vc-theorems", "vc-postamble"]
    return "\n\n".join((r.get(k) or "") for k in keys)


@app.function(image=image, cpu=2, memory=4096, timeout=1800)
def check(srcs: list) -> dict:
    import subprocess, tempfile, os, time
    out = {}
    for t in TOOLCHAINS:
        ok, errs, secs = 0, [], 0.0
        for i, s in enumerate(srcs):
            f = tempfile.mktemp(suffix=".lean"); open(f, "w").write(s)
            t0 = time.time()
            p = subprocess.run(["/root/.elan/bin/elan", "run", f"leanprover/lean4:{t}", "lean", f], capture_output=True, text=True, timeout=300)
            secs += time.time() - t0
            txt = p.stdout + p.stderr
            bad = [l for l in txt.splitlines() if "error" in l]
            if not bad: ok += 1
            elif len(errs) < 3: errs.append(bad[0][:200])
        ver = subprocess.run(["/root/.elan/bin/elan", "run", f"leanprover/lean4:{t}", "lean", "--version"], capture_output=True, text=True).stdout.strip()
        out[t] = {"version": ver, "ok": ok, "n": len(srcs), "avg_s": round(secs / len(srcs), 1), "errors": errs}
    return out


@app.local_entrypoint()
def main():
    rows = [json.loads(l) for l in open("runs/dafny/vericoding.jsonl")]
    lean = [r for r in rows if r["language"] == "lean" and str(r.get("qa-issue")) == "0"
            and "Mathlib" not in "".join((r.get(k) or "") for k in ("vc-preamble", "vc-definitions", "vc-theorems"))]
    random.Random(7).shuffle(lean)
    by = {}
    for r in lean:
        by.setdefault(r["source"], []).append(r)
    sample = [r for s in sorted(by) for r in by[s][:3]]
    res = check.remote([assemble(r) for r in sample])
    print(json.dumps(res, indent=1))
