"""Hermetic tests for the spec suite + gate (no API calls)."""
import json
from pathlib import Path

from millennium_harness.gate import check_suite, feedback
from millennium_harness.problem import Problem, same_output
from millennium_harness.spec import Suite, adequacy

SPEC = '''
APPLICABLE = True
from hypothesis import strategies as st

@st.composite
def gen(draw):
    n = draw(st.integers(1, 6))
    a = draw(st.lists(st.integers(-5, 5), min_size=n, max_size=n))
    return f"{n}\\n{' '.join(map(str, a))}\\n"

def validate(inp):
    t = inp.split()
    n = int(t[0]); a = t[1:]
    return 1 <= n <= 100000 and len(a) == n

def ref(inp):
    t = inp.split(); a = list(map(int, t[1:]))
    best = a[0]
    for i in range(len(a)):
        for j in range(i, len(a)):
            best = max(best, sum(a[i:j + 1]))
    return str(best)
'''
BAD_SPEC = SPEC.replace("best = max(best, sum(a[i:j + 1]))", "best = max(best, a[j])")  # wrong reference

GOOD = '''
n = int(input()); a = list(map(int, input().split()))
best = cur = a[0]
for x in a[1:]:
    cur = max(x, cur + x); best = max(best, cur)
print(best)
'''
BUGGY = GOOD.replace("cur = max(x, cur + x)", "cur = max(0, cur + x)")  # wrong when all negative

P = Problem(pid="maxsub", kind="stdin", statement="max subarray sum",
            examples=[("3\n1 -2 3\n", "3"), ("4\n-1 2 3 -1\n", "5")])


def _write(tmp_path, name, code):
    f = tmp_path / name
    f.write_text(code)
    return str(f)


def test_same_output_semantics():
    assert same_output("stdin", "3\n", "3")
    assert same_output("stdin", "0.1000000000", "0.1")
    assert not same_output("stdin", "3", "4")
    assert same_output("functional", "[1, 2]", "[1,2]")


def test_adequacy_rejects_wrong_reference(tmp_path):
    assert adequacy(P, _write(tmp_path, "good.py", SPEC)).ok
    assert not adequacy(P, _write(tmp_path, "bad.py", BAD_SPEC)).ok


def test_suite_refutes_bug_and_passes_correct(tmp_path):
    specs = [_write(tmp_path, f"s{i}.py", SPEC) for i in range(3)]
    s = Suite(P, specs)
    assert s.status == "OK" and len(s.inputs) >= 20
    assert s.check_code(GOOD, str(tmp_path / "w1")).status == "BOUNDED_OK"
    r = s.check_code(BUGGY, str(tmp_path / "w2"))
    assert r.status == "REFUTED" and r.smallest is not None
    s.save(str(tmp_path / "suite.json"))
    suite = json.loads(Path(tmp_path / "suite.json").read_text())
    g = check_suite(suite, BUGGY)
    assert g["status"] == "REFUTED"
    fb = feedback(g, 1, 2)
    assert "REFUTED" in fb and len(fb) < 6000
    assert check_suite(suite, GOOD)["status"] == "BOUNDED_OK"


def test_outlier_reference_dropped(tmp_path):
    specs = [_write(tmp_path, f"s{i}.py", SPEC) for i in range(2)] + [_write(tmp_path, "bad.py", BAD_SPEC)]
    s = Suite(P, specs)
    assert s.status == "OK" and len(s.survivors) == 2


def test_function_task_without_examples_stays_functional():
    from millennium_harness.problem import from_lcb_record
    rec = {"question_id": "x", "question_content": "Return x+y.", "public_test_cases": "[]",
           "starter_code": "class Solution:\n    def add(self, x: int, y: int) -> int:\n", "metadata": '{"func_name": "add"}'}
    p, _ = from_lcb_record(rec)
    assert p.kind == "functional" and "one JSON value per line" in p.format_hint()
    rec2 = dict(rec, starter_code="", metadata="{}")
    assert from_lcb_record(rec2)[0].kind == "stdin"
