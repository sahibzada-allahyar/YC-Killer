"""The gate: run code against a locked suite, decide, and render the frozen feedback block + receipt line.

Benchmarks and the product call exactly this. CLI (used inside benchmark sandboxes, suite uploaded only for
the duration of the check):
    python -m millennium_harness.gate SUITE.json SOLUTION.py --round 1 --max-rounds 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

from .problem import same_output
from . import run as runner

MAX_FIELD = 1200


def _clip(s: str | None) -> str:
    if s is None:
        return "(no output)"
    s = s.rstrip("\n")
    return s if len(s) <= MAX_FIELD else s[:MAX_FIELD] + f"\n… ({len(s) - MAX_FIELD} more chars)"


def check_suite(suite: dict, code: str, timeout: float = 2.0) -> dict:
    t0 = time.time()
    if suite.get("status") != "OK" or not suite.get("inputs"):
        return {"status": "SPEC_INVALID", "secs": 0.0}
    work = tempfile.mkdtemp(prefix="sl-")
    path = runner.prepare(code, work)
    res = runner.run(suite["kind"], path, suite["inputs"], suite.get("func"), timeout=timeout)
    dis = [(x, e, o, err) for x, e, (o, err) in zip(suite["inputs"], suite["expected"], res)
           if not same_output(suite["kind"], e, o)]
    out = {"status": "REFUTED" if dis else "BOUNDED_OK", "n": len(suite["inputs"]), "n_dis": len(dis),
           "n_refs": suite.get("n_refs", 0), "sha": suite.get("sha", ""), "secs": round(time.time() - t0, 1),
           "max_len": max(len(x) for x in suite["inputs"])}
    if dis:
        x, e, o, err = dis[0]  # inputs are sorted smallest-first
        out["cex"] = {"input": x, "expected": e, "got": o, "err": err}
    return out


def feedback(r: dict, rnd: int, max_rounds: int) -> str:
    if r["status"] != "REFUTED":
        return ""
    c = r["cex"]
    got = _clip(c["got"]) if c["got"] is not None else f"(crashed: {(c['err'] or '')[:300]})"
    return (
        f"⟦Millennium Harness⟧ ✗ REFUTED · differential · {r['n_refs']} independent references agree · "
        f"round {rnd}/{max_rounds}\n"
        f"Your solution disagrees with an executable specification on {r['n_dis']} of {r['n']} small inputs.\n"
        f"Smallest counterexample input:\n```\n{_clip(c['input'])}\n```\n"
        f"Expected output (the references agree):\n```\n{_clip(c['expected'])}\n```\n"
        f"Your output:\n```\n{got}\n```\n"
        f"spec {r['sha']} locked · written from the task statement only, without seeing your code.\n"
        "A reference can be wrong: if you are certain your output is correct, say why and keep your solution. "
        "Do not special-case this input; find and fix the underlying bug, then re-check your solution."
    )


def receipt_line(r: dict) -> str:
    s = r["status"]
    if s == "BOUNDED_OK":
        return f"▢ agrees with {r['n_refs']} references on {r['n']} inputs · spec {r['sha']} · {r['secs']}s"
    if s == "REFUTED":
        return f"✗ refuted on {r['n_dis']}/{r['n']} inputs · spec {r['sha']}"
    return "◧ no checkable spec (gate silent)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("solution")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=2)
    a = ap.parse_args(argv)
    suite = json.load(open(a.suite))
    if not os.path.exists(a.solution):
        # the deliverable itself is missing: the task is not done (a structural gate never lets that through)
        print(json.dumps({"status": "REFUTED", "missing": True, "n_refs": suite.get("n_refs", 0), "sha": suite.get("sha", ""),
                          "feedback": f"⟦Millennium Harness⟧ ✗ NOT DONE · round {a.round}/{a.max_rounds}\n"
                                      f"There is no solution at {a.solution}. The task is not complete: "
                                      "write the full solution to that file, then check it against the examples."}))
        return
    r = check_suite(suite, open(a.solution).read())
    r["feedback"] = feedback(r, a.round, a.max_rounds)
    r["receipt"] = receipt_line(r) if r["status"] != "NO_SOLUTION" else ""
    sys.stdout.write(json.dumps(r))


if __name__ == "__main__":
    main()
