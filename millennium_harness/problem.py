"""Task model + output comparison. Comparison semantics mirror LiveCodeBench's grader exactly."""
from __future__ import annotations

import base64
import json
import pickle
import zlib
from dataclasses import dataclass, field


@dataclass
class Problem:
    pid: str
    statement: str
    kind: str  # "stdin" | "functional"
    examples: list[tuple[str, str]]  # (input, output) — public only
    func_name: str | None = None
    starter_code: str = ""
    meta: dict = field(default_factory=dict)

    def examples_block(self) -> str:
        out = []
        for i, (inp, exp) in enumerate(self.examples, 1):
            out.append(f"Example {i} input:\n```\n{inp}\n```\nExample {i} output:\n```\n{exp}\n```")
        return "\n".join(out)

    def format_hint(self) -> str:
        if self.kind == "functional":
            return (f"This is a FUNCTION task: the solution is method `{self.func_name}` of `class Solution`.\n"
                    f"Starter code:\n```python\n{self.starter_code}\n```\n"
                    "Inputs are the arguments, one JSON value per line; the output is the JSON return value.")
        return "This is a STANDARD INPUT task: the program reads stdin and writes stdout."


def _lcb_stdin_decode(s: str) -> str:
    # LCB's grader feeds stdin as input.encode().decode('unicode_escape'); mirror it for dataset tests.
    try:
        return s.encode().decode("unicode_escape")
    except Exception:
        return s


def _json_or_raw(s):
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


def _norm(v):
    if isinstance(v, tuple):
        return [_norm(x) for x in v]
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def same_output(kind: str, expected: str, got: str) -> bool:
    """LCB semantics. stdin: non-empty stripped lines equal, or float within 1e-9. functional: JSON ==."""
    if got is None or expected is None:
        return False
    if kind == "functional":
        return _norm(_json_or_raw(expected)) == _norm(_json_or_raw(got))
    a = [ln.strip() for ln in got.strip().splitlines() if ln.strip()]
    e = [ln.strip() for ln in expected.strip().splitlines() if ln.strip()]
    if len(a) != len(e):
        return False
    for x, y in zip(a, e):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) < 1e-9:
                continue
        except ValueError:
            pass
        return False
    return True


def decode_private_tests(value) -> list[dict]:
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        pass
    return json.loads(pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8")))))


def from_lcb_record(rec: dict, with_private: bool = False) -> tuple[Problem, list[tuple[str, str]]]:
    """Build a Problem from a LiveCodeBench record (HF code_generation_lite / Harbor config.json)."""
    pub = rec["public_test_cases"]
    pub = json.loads(pub) if isinstance(pub, str) else pub
    meta = rec.get("metadata") or "{}"
    meta = json.loads(meta) if isinstance(meta, str) else meta
    # no examples (free-form tasks): a function name / starter code still means a function task
    functional = (pub[0].get("testtype") == "functional") if pub else bool(meta.get("func_name"))
    kind = "functional" if functional else "stdin"
    dec = (lambda s: s) if functional else _lcb_stdin_decode
    examples = [(dec(t["input"]), t["output"]) for t in pub]
    prob = Problem(
        pid=str(rec["question_id"]),
        statement=rec["question_content"],
        kind=kind,
        examples=examples,
        func_name=meta.get("func_name"),
        starter_code=rec.get("starter_code") or "",
        meta={"platform": rec.get("platform"), "difficulty": rec.get("difficulty"),
              "contest_date": rec.get("contest_date"), "title": rec.get("question_title")},
    )
    hidden: list[tuple[str, str]] = []
    if with_private:
        hidden = [(dec(t["input"]), t["output"]) for t in decode_private_tests(rec["private_test_cases"])]
    return prob, hidden
