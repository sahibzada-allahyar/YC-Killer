"""Fresh-context executable specs: prompt, parse, adequacy filter, k-reference majority suite, hash-lock.

A spec is a Python module (written by a model that never sees any solution) defining gen / validate / ref
[/ same]. A Suite combines k specs: only inputs >=2 validators accept and outputs >=2 references agree on are
ever used to refute code. A spec that cannot reproduce the examples, kill trivial mutants, or agree with its
peers is discarded. Fewer than `min_refs` survivors => SPEC_INVALID and the gate stays silent.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .problem import Problem, same_output
from . import run as runner

MODEL = "claude-haiku-5-5"


# ---------- prompt / parse ----------

def spec_request(p: Problem, model: str = MODEL, effort: str = "medium", max_tokens: int = 16000) -> dict:
    prompt = (Path(__file__).parent / "prompts" / "spec.md").read_text()
    user = (f"Task statement:\n{p.statement}\n\n{p.format_hint()}\n\n{p.examples_block()}\n\n"
            "Write the specification module now.")
    return {"model": model, "max_tokens": max_tokens, "system": prompt,
            "output_config": {"effort": effort}, "messages": [{"role": "user", "content": user}]}


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return max(blocks, key=len) if blocks else None


# ---------- worker plumbing ----------

def _worker(spec_path: str, payload: dict, timeout: float = 60) -> dict:
    try:
        p = subprocess.run([sys.executable, "-m", "millennium_harness._specworker", os.path.abspath(spec_path)],
                           input=json.dumps(payload), capture_output=True, text=True, timeout=timeout,
                           cwd=str(Path(__file__).parent.parent))
        return json.loads(p.stdout) if p.stdout.strip() else {"error": p.stderr.strip()[-400:]}
    except subprocess.TimeoutExpired:
        return {"error": "worker timeout"}
    except json.JSONDecodeError:
        return {"error": "bad worker output"}


# ---------- adequacy ----------

@dataclass
class Adequacy:
    ok: bool
    reason: str
    inputs: list[str] = field(default_factory=list)  # valid sampled inputs


def adequacy(p: Problem, spec_path: str, seed: int = 0, n: int = 150, min_inputs: int = 20) -> Adequacy:
    info = _worker(spec_path, {"cmd": "info"})
    if "load_error" in info or "error" in info:
        return Adequacy(False, f"load: {info.get('load_error') or info.get('error')}")
    if not info.get("applicable"):
        return Adequacy(False, "not applicable")
    if not (info["has_gen"] and info["has_ref"] and info["has_validate"]):
        return Adequacy(False, "missing gen/ref/validate")
    ex_in = [i for i, _ in p.examples]
    if ex_in:  # no examples (free-form product tasks): rely on k-reference agreement instead
        v = _worker(spec_path, {"cmd": "validate", "inputs": ex_in}).get("ok") or []
        if not v or not all(v):
            return Adequacy(False, "validate rejects an example")
        r = _worker(spec_path, {"cmd": "ref", "inputs": ex_in, "timeout": 5.0})
        for (inp, exp), out in zip(p.examples, r.get("out") or [None] * len(ex_in)):
            if not same_output(p.kind, exp, out):
                return Adequacy(False, "ref fails an example")
    s = _worker(spec_path, {"cmd": "sample", "n": n, "seed": seed, "timeout": 8.0}, timeout=40)
    edge = _worker(spec_path, {"cmd": "edge"}).get("inputs") or []
    cand = list(dict.fromkeys(edge + (s.get("inputs") or [])))
    ok = _worker(spec_path, {"cmd": "validate", "inputs": cand}).get("ok") or []
    valid = [x for x, k in zip(cand, ok) if k]
    if len(valid) < min_inputs:
        return Adequacy(False, f"only {len(valid)} valid generated inputs ({s.get('error', '')})")
    rr = _worker(spec_path, {"cmd": "ref", "inputs": valid, "timeout": 2.0}, timeout=2.5 * len(valid) + 30)
    outs = rr.get("out") or []
    done = [o is not None for o in outs]
    if sum(done) < 0.95 * len(valid):
        return Adequacy(False, f"ref failed/timed out on {len(valid) - sum(done)}/{len(valid)} inputs")
    pairs = [(x, o) for x, o in zip(valid, outs) if o is not None]
    const = p.examples[0][1] if p.examples else (pairs[0][1] if pairs else "")
    mutants = {"constant": lambda x: const, "empty": lambda x: ""}
    if p.kind == "stdin":
        mutants["echo"] = lambda x: x
    for name, m in mutants.items():
        if all(same_output(p.kind, o, m(x)) for x, o in pairs):
            return Adequacy(False, f"cannot distinguish trivial mutant '{name}'")
    return Adequacy(True, "ok", [x for x, _ in pairs])


def _unfiltered(spec_path: str, seed: int, n: int = 150) -> Adequacy:
    """Ablation: trust the spec blindly (no example/mutant/coverage checks)."""
    s = _worker(spec_path, {"cmd": "sample", "n": n, "seed": seed, "timeout": 8.0}, timeout=40)
    xs = s.get("inputs") or []
    rr = _worker(spec_path, {"cmd": "ref", "inputs": xs, "timeout": 2.0}, timeout=2.5 * len(xs) + 30)
    xs = [x for x, o in zip(xs, rr.get("out") or []) if o is not None]
    return Adequacy(bool(xs), "unfiltered" if xs else "no inputs", xs)


# ---------- suite ----------

@dataclass
class Disagreement:
    input: str
    expected: str
    got: str | None
    err: str | None


@dataclass
class CheckResult:
    status: str  # REFUTED | BOUNDED_OK | SPEC_INVALID
    n_checked: int = 0
    n_disagree: int = 0
    smallest: Disagreement | None = None
    agree_refs: int = 0
    outputs: list | None = None


class Suite:
    def __init__(self, p: Problem, spec_paths: list[str], min_refs: int = 2, max_inputs: int = 200,
                 seed: int = 0, outlier_frac: float = 0.05, filter: bool = True):
        self.p, self.min_refs = p, min_refs
        self.report: dict = {"specs": {}}
        adeq = {}
        for i, sp in enumerate(spec_paths):
            a = adequacy(p, sp, seed=seed + i) if filter else _unfiltered(sp, seed + i)
            self.report["specs"][os.path.basename(sp)] = a.reason
            if a.ok:
                adeq[sp] = a
        self.survivors = list(adeq)
        self.inputs: list[str] = []
        self.expected: list[str] = []
        self.sha = ""
        if len(self.survivors) < min_refs:
            self.status = "SPEC_INVALID"
            return
        # pool inputs round-robin across specs (diversity), then smallest first
        pool, seen = [], set()
        lists = [adeq[s].inputs for s in self.survivors]
        for row in range(max(len(l) for l in lists)):
            for l in lists:
                if row < len(l) and l[row] not in seen:
                    seen.add(l[row]); pool.append(l[row])
        pool.sort(key=lambda x: (len(x), x))
        need = min(2, len(self.survivors))
        votes = [_worker(s, {"cmd": "validate", "inputs": pool}).get("ok") or [False] * len(pool)
                 for s in self.survivors]
        pool = [x for j, x in enumerate(pool) if sum(v[j] for v in votes) >= need][: max_inputs * 2]
        outs = {s: (_worker(s, {"cmd": "ref", "inputs": pool, "timeout": 2.0}, timeout=2.5 * len(pool) + 30)
                    .get("out") or [None] * len(pool)) for s in self.survivors}
        # drop outlier references (only meaningful with >=3)
        if len(self.survivors) >= 3:
            keep = []
            for s in self.survivors:
                others = [t for t in self.survivors if t != s]
                dis = tot = 0
                for j in range(len(pool)):
                    o = outs[s][j]
                    ag = [outs[t][j] for t in others if outs[t][j] is not None]
                    if o is None or len(ag) < 2 or not same_output(p.kind, ag[0], ag[1]):
                        continue
                    tot += 1
                    dis += not same_output(p.kind, ag[0], o)
                if tot == 0 or dis / tot <= outlier_frac:
                    keep.append(s)
                self.report["specs"][os.path.basename(s)] += f" | outlier-dis={dis}/{tot}"
            self.survivors = keep
        if len(self.survivors) < min_refs:
            self.status = "SPEC_INVALID"
            return
        for j, x in enumerate(pool):
            vals = [outs[s][j] for s in self.survivors if outs[s][j] is not None]
            maj = None
            if len(self.survivors) == 1:
                maj = vals[0] if vals else None
            else:
                for a in range(len(vals)):
                    if sum(same_output(p.kind, vals[a], b) for b in vals) >= 2:
                        maj = vals[a]; break
            if maj is not None:
                self.inputs.append(x); self.expected.append(maj)
            if len(self.inputs) >= max_inputs:
                break
        self.status = "OK" if self.inputs else "SPEC_INVALID"
        h = hashlib.sha256()
        for s in sorted(self.survivors):
            h.update(Path(s).read_bytes())
        self.sha = h.hexdigest()[:12]
        self.report.update(n_inputs=len(self.inputs), survivors=[os.path.basename(s) for s in self.survivors])

    def check_code(self, code: str, workdir: str, timeout: float = 2.0) -> CheckResult:
        if self.status != "OK":
            return CheckResult("SPEC_INVALID")
        os.makedirs(workdir, exist_ok=True)
        path = runner.prepare(code, workdir)
        res = runner.run(self.p.kind, path, self.inputs, self.p.func_name, timeout=timeout)
        dis = [Disagreement(x, e, o, err) for x, e, (o, err) in zip(self.inputs, self.expected, res)
               if not same_output(self.p.kind, e, o)]
        return CheckResult("REFUTED" if dis else "BOUNDED_OK", len(self.inputs), len(dis),
                           dis[0] if dis else None, len(self.survivors), [o for o, _ in res])

    def save(self, path: str):
        Path(path).write_text(json.dumps({"pid": self.p.pid, "kind": self.p.kind, "func": self.p.func_name,
                                          "status": self.status, "sha": self.sha, "n_refs": len(self.survivors),
                                          "inputs": self.inputs, "expected": self.expected,
                                          "report": self.report}))
