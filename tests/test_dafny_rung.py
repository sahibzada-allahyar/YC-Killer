"""Dafny ∎ rung: lexer/layout/ban unit tests (no Dafny needed) and the bench/redteam/dafny red team (needs Dafny
4.11: MILLENNIUM_DAFNY (or the older SORRYLESS_DAFNY), `dafny` on PATH, or ~/.local/dafny-4.11/dafny/dafny; skipped
otherwise)."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from millennium_harness import dafny

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "bench" / "redteam" / "dafny"
CASES = json.loads((RT / "cases.json").read_text())


def _task(tid: str) -> dict:
    return json.loads((RT / "tasks" / f"{tid}.json").read_text())


def _file(tid: str, helpers: str = "", body: str = "{\n}") -> str:
    t = _task(tid)
    return f"{t['preamble']}\n\n{dafny.HELPERS_BEGIN}\n{helpers}\n{dafny.HELPERS_END}\n\n{t['spec']}\n{body}\n"


def _static(text: str, tid: str) -> list[str]:
    """Everything the rung decides about the solution before running Dafny."""
    spans, v = dafny._layout(text, _task(tid))
    code = dafny.strip_comments(text)
    v = v or dafny._seams(text, code, spans) + dafny._captures(code, spans)
    return v or dafny._bans([code[s:e] for s, e in dafny._agent_spans(text, spans)])


# ---- no Dafny needed ----

def test_strip_comments_keeps_offsets_and_handles_nesting_strings_chars():
    src = 'a /* x /* y */ assume */ b // assume\nc := "// not a comment"; d := \'"\'; e\' := @"q""uote"; f'
    out = dafny.strip_comments(src)
    assert len(out) == len(src) and "assume" not in out and "not a comment" not in out
    assert out.split() == ["a", "b", "c", ":=", '"', '";', "d", ":=", '"', '";', "e'", ":=", '"', '";', "f"]


def test_honest_layouts_have_no_static_violations():
    for tid in ("DA0545", "DT0254", "DT0076"):  # DT0254's preamble is a lone `//` comment line
        assert _static(_file(tid, helpers="lemma L()\n  ensures true\n{\n}", body="{\n  // ok\n}"), tid) == []


@pytest.mark.parametrize("helpers,body,needle", [
    ("", "  requires false\n{\n}", "followed directly by the method body"),
    ("", "  || true\n{\n}", "followed directly by the method body"),
    ("ghost", "{\n}", "right before the frozen spec"),
    ("class C {", "{\n}\n}", "right before the frozen spec"),
    ("class Shadow {\n  predicate P() { false }", "{\n}\n}", "must stay at the top level"),
    ("datatype D = D {\n  predicate P() { true }", "{\n}\n}", "must stay at the top level"),
    ("const k := (1 +\n  2", "{\n}\n)", "must stay at the top level"),
    ("/*", "*/ method solve(a: seq<int>) returns (result: seq<int>) {}", "starts inside an agent comment or string"),
    ('const s := @"', '"; method solve(a: seq<int>) returns (result: seq<int>) {}', "starts inside"),
])
def test_agent_text_cannot_glue_onto_the_frozen_spec(helpers, body, needle):
    v = _static(_file("DA0545", helpers, body), "DA0545")
    assert any(needle in x for x in v), v


def test_agent_text_after_preamble_must_start_a_declaration():
    text = _file("DA0545").replace(dafny.HELPERS_BEGIN, "&& true " + dafny.HELPERS_BEGIN)
    assert any("must start a new declaration" in x for x in _static(text, "DA0545"))


@pytest.mark.parametrize("helpers,needle", [
    ("method {:selective_checking} H() {}", "attribute {:selective_checking} is not allowed"),
    ("method {:verify (false)} H() {}", "banned construct: verify false"),
    ("method {:compile false} H() {}", "banned construct: compile false"),
    ("trait T { method {:termination false} M() }", "attribute {:termination} is not allowed"),
    ("method {:rlimit 1000} H() {}", "attribute {:rlimit} is not allowed"),
    ("@IsolateAssertions\nlemma H() {}", "banned construct: @-attribute"),
    ("@ Axiom\nlemma H() ensures false {}", "banned construct: @-attribute"),
    ("@/**/Axiom\nlemma H() ensures false {}", "banned construct: @-attribute"),
    ("const k: int@Axiom\nlemma H() ensures false {}", "banned construct: @-attribute"),
    ('@ResourceLimit("1e9")', "banned construct: @-attribute"),
    ("class {:autocontracts} B { ghost predicate Valid() { true } }", "attribute {:autocontracts} is not allowed"),
    ("module M {}", "banned construct: module"),
    ("lemma H()\n  ensures false\n", "bodiless declaration with ensures"),
])
def test_bans_in_agent_text(helpers, needle):
    v = _static(_file("DA0545", helpers), "DA0545")
    assert any(needle in x for x in v), v


def test_allowed_constructs_are_not_banned():
    helpers = ('lemma {:induction false} H(s: seq<int>)\n  ensures forall i {:trigger s[i]} :: 0 <= i < |s| ==> s[i] == s[i]\n'
               '{\n  var msg := "assume expect module {:axiom}";\n}\ntwostate lemma T(c: C) ensures old@L(true) {}\n'
               'method M(c: C) modifies c { label L: c.f := 1; assert old @L(c.f) == 0 || unchanged@L(c) || true; }')
    assert _static(_file("DA0545", helpers), "DA0545") == []


@pytest.mark.parametrize("tid,helpers,needle", [
    ("DT0137", "function Some(s: string): StringOption\n{\n  StringOption.None\n}", "declares `Some`"),
    ("DT0137", "function {:opaque} Some(s: string): StringOption { None }", "declares `Some`"),
    ("DT0137", 'const None: StringOption := StringOption.Some("")', "declares `None`"),
    ("DT0137", "datatype Mine = Some(x: int) | Other", "declares `Some`"),
    ("DT0137", "copredicate Some(s: string) { true }", "declares `Some`"),
    ("DT0137", "datatype Mine = | A(x: int)\n  | ghost None {\n  predicate P() { true }\n}", "declares `None`"),
    # DA0433's frozen text uses a helper it never defines: caught as a task error (frozen must resolve on its own),
    # not by the capture check, which only guards names the frozen text itself declares
    ("DA0433", "function computeFinalBoundsHelper(lo: int, hi: int, ops: seq<(int, int)>, k: int): (int, int)\n"
               "{\n  (lo, hi)\n}", None),
    ("DT0056", "function flattened(): int { 0 }", None),  # a frozen local binder, not a frozen declaration
    ("DA0545", "lemma solve'() {}\nghost predicate ValidTransformation'() { true }", None),
    ("DT0137", "class C {\n  function Some(): int { 0 }\n}", None),  # members do not shadow top-level names
    ("DT0137", "datatype D = A | B {\n  predicate None() { true }\n}\nfunction Codes(c: string): StringOption "
               "{ Some(c) }", None),
])
def test_agent_cannot_declare_a_name_the_frozen_text_uses(tid, helpers, needle):
    v = _static(_file(tid, helpers, "{\n  result := None;\n}" if tid == "DT0137" else "{\n}"), tid)
    if needle is None:
        assert v == []
    else:
        assert any(needle in x for x in v), v


def test_check_error_feedback_does_not_ask_for_a_fix():
    fb = dafny.feedback({"status": "CHECK_ERROR", "violations": ["task error: x"]}, 1, 2)
    assert "task error: x" in fb and "cannot be checked" in fb and "byte-identical" not in fb


def test_audit_keeps_soundness_findings_in_agent_text_only(monkeypatch):
    text = _file("DA0545", helpers="lemma L()\n  ensures true\n{\n}")
    spans, _ = dafny._layout(text, _task("DA0545"))
    ln = text[:text.index("lemma L")].count("\n") + 1
    hint = "Warning: Triggers were added to this quantifier that may introduce matching loops, which may cause brittle"
    canned = "\n".join([
        f"solution.dfy({ln},2): {hint} verification.",
        f"solution.dfy({ln},2): Warning: Could not find a trigger for this quantifier.",
        f"solution.dfy({ln},6): Warning: L: Declaration has explicit `{{:axiom}}` attribute. Possible mitigation: x",
        "solution.dfy(1,9): Warning: TransformElement: Definition has `assume` statement in body. Possible mitigation: x",
        "lib.dfy(1,0): Warning: M: Ghost declaration has no body and has an ensures clause. Possible mitigation: x",
    ])
    monkeypatch.setattr(dafny, "_dafny", lambda args, cwd, timeout: (0, canned))
    found = dafny._audit(text, "solution.dfy", "/nonexistent", dafny._agent_spans(text, spans))
    assert [x.split(":")[1].strip() for x in found] == ["L", "M"]  # agent-line axiom + other-file finding


def test_cli_ignores_other_argv_for_embedded_graders(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["grade.py", "/app/solution.dfy", "/tests/frozen.json"])
    dafny.main()  # adapters append their own __main__ block after the module source
    assert capsys.readouterr().out == ""


def test_red_team_inventory():
    cheats = [c for c in CASES if c["expect"] != "PROVED"]
    honest = [c for c in CASES if c["expect"] == "PROVED"]
    assert len(cheats) >= 12 and all(c["reason"] for c in cheats)
    assert len(honest) >= 4 and any(c.get("verify_has") == "Could not find a trigger" for c in honest)
    for c in CASES:
        frozen = _task(c["task"])
        assert (RT / c["file"]).exists() and frozen["spec"].strip()



def test_dafny_binary_env_var_and_legacy_fallback(tmp_path, monkeypatch):
    """MILLENNIUM_DAFNY picks the binary; the pre-rename SORRYLESS_DAFNY is still read when it is unset."""
    def fake(name):
        p = tmp_path / name
        p.write_text(f"#!/bin/sh\necho {name}\n")
        p.chmod(0o755)
        return str(p)
    new, old = fake("new"), fake("old")
    monkeypatch.delenv("MILLENNIUM_DAFNY", raising=False)
    monkeypatch.setenv("SORRYLESS_DAFNY", old)
    assert dafny._dafny(["--version"], str(tmp_path), 30) == (0, "old\n")
    monkeypatch.setenv("MILLENNIUM_DAFNY", new)
    assert dafny._dafny(["--version"], str(tmp_path), 30) == (0, "new\n")


# ---- needs Dafny 4.11 ----

def _find_dafny() -> str | None:
    for exe in (os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY")), shutil.which("dafny"),
                str(Path.home() / ".local" / "dafny-4.11" / "dafny" / "dafny")):
        if exe and os.path.exists(exe):
            if subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.startswith("4.11"):
                return exe
    return None


DAFNY = _find_dafny()
needs_dafny = pytest.mark.skipif(DAFNY is None, reason="Dafny 4.11 not installed (set MILLENNIUM_DAFNY)")


@pytest.fixture(scope="module")
def red_team():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MILLENNIUM_DAFNY", DAFNY)
        spec = importlib.util.spec_from_file_location("dafny_redteam", RT / "run.py")
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        yield {r["file"]: r for r in run.run_all()}


@needs_dafny
@pytest.mark.parametrize("case", CASES, ids=[Path(c["file"]).stem for c in CASES])
def test_red_team_case(red_team, case):
    r = red_team[case["file"]]
    assert r["ok"], f"{r['status']} (expected {case['expect']}: {case['reason']!r}) {r['violations']}\n{r['verify'][-800:]}"


@needs_dafny
@pytest.mark.parametrize("tid,needle", [
    ("DA0433", "unresolved identifier: computeFinalBoundsHelper"),  # frozen text calls an undefined helper
    ("DA0149", "unresolved identifier: splitLines"),
    ("DT0131", "this symbol not expected in Dafny"),  # leftover LLM prose in the preamble
    ("DT0643", "this symbol not expected in Dafny"),  # malformed split
    ("DA0545", None), ("DT0137", None), ("DA0192", None), ("DT0072", None), ("DA0384", None), ("DT0606", None),
])
def test_task_problems_needs_frozen_text_that_resolves_alone(monkeypatch, tid, needle):
    monkeypatch.setenv("MILLENNIUM_DAFNY", DAFNY)
    v = dafny.task_problems(_task(tid))
    assert (v == []) if needle is None else (len(v) == 1 and needle in v[0]), v


@needs_dafny
@pytest.mark.parametrize("witness", sorted((RT / "unsat").glob("*.dfy")), ids=lambda p: p.stem)
def test_unsatisfiable_spec_witnesses_verify(witness):
    """unsat/<ID>.dfy holds the task's frozen preamble and spec verbatim plus a Witness method that derives `false`
    from the spec's contract at one input meeting its precondition: no honest solution exists, so the pool drops it."""
    text, t = witness.read_text(), _task(witness.stem)
    assert t["preamble"] in text and t["spec"] in text and "ensures false" in text
    rest = text.replace(t["preamble"], "").replace(t["spec"], "")
    assert dafny._bans([dafny.strip_comments(rest)]) == []
    out = subprocess.run([DAFNY, "verify", "--allow-warnings", witness.name], cwd=witness.parent, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0 and " 0 errors" in out.stdout, out.stdout[-1500:]


@needs_dafny
def test_cli_lint():
    out = subprocess.run([sys.executable, "-m", "millennium_harness.dafny", "lint", str(RT / "tasks" / "DA0433.json")],
                         capture_output=True, text=True, cwd=ROOT, env={**os.environ, "MILLENNIUM_DAFNY": DAFNY},
                         timeout=300)
    assert "computeFinalBoundsHelper" in json.loads(out.stdout)["problems"][0]


@needs_dafny
def test_cli_end_to_end():
    c = CASES[0]
    out = subprocess.run([sys.executable, "-m", "millennium_harness.dafny", "check", str(RT / c["file"]),
                          str(RT / "tasks" / f"{c['task']}.json"), "1", "2"], capture_output=True, text=True,
                         cwd=ROOT, env={**os.environ, "MILLENNIUM_DAFNY": DAFNY}, timeout=600)
    r = json.loads(out.stdout)
    assert r["status"] == "PROVED" and r["feedback"] == "" and r["verified"] >= 1
