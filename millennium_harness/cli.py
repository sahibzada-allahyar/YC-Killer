"""Millennium Harness CLI — review the spec, not the code.

  millennium-harness spec TASK [--target solution.py | solution.py::func] [-k 3] [--host anthropic|claude|opencode]
      Fresh-context spec from the task text ONLY (written before / without seeing code), checked, locked.
  millennium-harness check [TARGET]   Run the locked suite against the target; print receipt or counterexample.
  millennium-harness gate --hook claude|opencode
      Stop-hook entry: blocks "done" (with the frozen feedback block) while a counterexample exists, <= 2 rounds.
  millennium-harness show            Print the locked spec — the thing to review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from . import gate as G
from .problem import Problem

STATE = Path(".millennium-harness")
MAX_ROUNDS = 2
GLYPH = {"BOUNDED_OK": "▢", "REFUTED": "✗", "SPEC_INVALID": "◧", "PROVED": "∎"}


def _store(sha: str) -> Path:
    repo = hashlib.sha1(str(Path.cwd().resolve()).encode()).hexdigest()[:10]
    return Path.home() / ".cache" / "millennium-harness" / repo / sha


def _examples(text: str) -> list[tuple[str, str]]:
    """Pull 'Example/Sample Input/Output' pairs out of a task statement (best effort)."""
    blocks = re.findall(r"(?:Sample|Example)\s*Input\s*\d*\s*:?\s*\n```?\n?(.*?)```?\s*\n\s*(?:Sample|Example)\s*Output"
                        r"\s*\d*\s*:?\s*\n```?\n?(.*?)(?:```|\n\n)", text, re.S | re.I)
    return [(i.strip("\n") + "\n", o.strip()) for i, o in blocks]


def _target(t: str | None) -> tuple[str, str, str | None]:
    t = t or "solution.py"
    if "::" in t:
        path, fn = t.split("::", 1)
        return "functional", path, fn
    return "stdin", t, None


def cmd_spec(a):
    from .llm import anthropic_parallel, host_cli_call
    from .spec import Suite, extract_code, spec_request
    text = Path(a.task).read_text() if os.path.exists(a.task) else a.task
    kind, path, fn = _target(a.target)
    p = Problem(pid="task", statement=text, kind=kind, examples=_examples(text), func_name=fn)
    reqs = [spec_request(p) for _ in range(a.k)]
    if a.host == "anthropic":
        texts = [t for t, _ in anthropic_parallel(reqs, workers=a.k)]
    else:
        texts = [host_cli_call(r, a.host) for r in reqs]
    tmp = STATE / "drafts"
    tmp.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(texts):
        f = tmp / f"spec_{i}.py"
        f.write_text(extract_code(t) or "APPLICABLE = False\n")
        paths.append(str(f))
    s = Suite(p, paths, min_refs=min(2, a.k))
    if s.status != "OK":
        print(f"◧ no checkable spec ({json.dumps(s.report.get('specs'))[:300]}) — gate stays silent")
        (STATE / "target.json").write_text(json.dumps({"status": "SPEC_INVALID"}))
        return
    store = _store(s.sha)
    store.mkdir(parents=True, exist_ok=True)
    s.save(str(store / "suite.json"))
    for sp in s.survivors:
        (store / Path(sp).name).write_text(Path(sp).read_text())
    os.chmod(store / "suite.json", 0o444)
    (STATE / "target.json").write_text(json.dumps({"sha": s.sha, "kind": kind, "path": path, "func": fn,
                                                   "store": str(store), "round": 0}))
    print(f"spec {s.sha} locked · {len(s.survivors)} independent references agree on {len(s.inputs)} inputs · "
          f"written from the task text only\nreview it: millennium-harness show")


def _load_target():
    t = json.loads((STATE / "target.json").read_text())
    if t.get("status") == "SPEC_INVALID":
        return t, None
    suite_path = Path(t["store"]) / "suite.json"
    suite = json.loads(suite_path.read_text())
    if suite.get("sha") != t["sha"]:
        return t, {"status": "SPEC_TAMPERED"}
    if t["kind"] == "functional":
        suite["kind"], suite["func"] = "functional", t["func"]
    return t, suite


def cmd_check(a):
    t, suite = _load_target()
    if suite is None or suite.get("status") != "OK":
        print("◧", (suite or t).get("status")); return 0
    code = Path(a.target or t["path"]).read_text()
    if t["kind"] == "functional" and "class Solution" not in code:
        code += f"\n\nclass Solution:\n    def {t['func']}(self, *a, **k):\n        return {t['func']}(*a, **k)\n"
    r = G.check_suite(suite, code)
    print(G.receipt_line(r) if r["status"] != "REFUTED" else G.feedback(r, 1, 1))
    return 1 if r["status"] == "REFUTED" else 0


def cmd_gate(a):
    hook = json.loads(sys.stdin.read() or "{}")
    if not (STATE / "target.json").exists():
        return 0
    t, suite = _load_target()
    if suite is None or suite.get("status") != "OK" or not Path(t["path"]).exists():
        return 0
    if t.get("round", 0) >= MAX_ROUNDS:
        return 0
    code = Path(t["path"]).read_text()
    if t["kind"] == "functional" and "class Solution" not in code:
        code += f"\n\nclass Solution:\n    def {t['func']}(self, *a, **k):\n        return {t['func']}(*a, **k)\n"
    r = G.check_suite(suite, code)
    (STATE / "receipt.md").write_text(G.receipt_line(r) + "\n")
    if r["status"] != "REFUTED":
        return 0
    t["round"] = t.get("round", 0) + 1
    (STATE / "target.json").write_text(json.dumps(t))
    msg = G.feedback(r, t["round"], MAX_ROUNDS)
    if a.hook == "claude":  # Claude Code Stop hook protocol: block with a reason
        print(json.dumps({"decision": "block", "reason": msg}))
        return 0
    print(msg)
    return 2


def cmd_show(a):
    t, suite = _load_target()
    if suite is None:
        print("◧ no spec"); return
    store = Path(t["store"])
    print(f"spec {t['sha']} · target {t['path']}{'::' + t['func'] if t.get('func') else ''}\n")
    first = sorted(store.glob("spec_*.py"))[0]
    print(first.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="millennium-harness", description="Review the spec, not the code.")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("spec"); s.add_argument("task"); s.add_argument("--target"); s.add_argument("-k", type=int, default=3)
    s.add_argument("--host", default=os.environ.get("MILLENNIUM_HOST", os.environ.get("SORRYLESS_HOST", "anthropic")))
    c = sp.add_parser("check"); c.add_argument("target", nargs="?")
    g = sp.add_parser("gate"); g.add_argument("--hook", default="cli")
    sp.add_parser("show")
    a = ap.parse_args(argv)
    sys.exit({"spec": cmd_spec, "check": cmd_check, "gate": cmd_gate, "show": cmd_show}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
