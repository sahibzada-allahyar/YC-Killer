"""FV Stage 0 (bench/fv/stage0.py): renamer, prompts and go/no-go rules run hermetically; the DEV lock-rate check
and grading need the pinned toolchains (Lean v4.23.0-rc2, Dafny 4.11) and runs/ data, and skip without them. No API
calls."""
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import types

import pytest

from bench.adapters import vericoding_fv as ad
from bench.fv import stage0 as S
from millennium_harness import dafny as sl_dafny
from millennium_harness import lean as sl_lean

try:
    LEAN = sl_lean.find_lean()
except (sl_lean.ToolchainError, OSError, subprocess.SubprocessError):
    LEAN = None
DAFNY = next((d for d in (os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY")), shutil.which("dafny"),
                          str(S.LOCAL_DAFNY))
              if d and os.path.isfile(d)), None)
needs_lean = pytest.mark.skipif(LEAN is None, reason=f"Lean {sl_lean.TOOLCHAIN} not installed")
needs_dafny = pytest.mark.skipif(DAFNY is None, reason="Dafny 4.11 not installed (set MILLENNIUM_DAFNY)")
has_data = pytest.mark.skipif(not (S.SPLIT.exists() and S.DATA.exists()), reason="runs/fv/split.json or data absent")

LEAN_REC = {
    "id": "LA9999", "language": "lean", "source": "apps",
    "vc-description": "Compute solve(xs): see `ValidInput` and `repeat`. Use repeat to solve it.",
    "vc-preamble": ("-- ValidInput says solve gets a list\n@[reducible, simp]\n"
                    "def ValidInput (xs : List Nat) : Prop :=\n  xs.length > 0\n\n"
                    "structure Shelf where\n  head : Nat\n  tail : Nat\n\n"
                    "def Shelf.width (s : Shelf) : Nat := s.tail - s.head\n\ndef length (xs : List Nat) : Nat := "
                    "xs.length + List.length xs\n\ndef «repeat» (n : Nat) : Nat := n\n\n"
                    "def f1 (c : Char) : Bool := c = 'f'"),
    "vc-helpers": "",
    "vc-definitions": "def solve (xs : List Nat) (h : ValidInput xs) : Nat :=\n  sorry",
    "vc-theorems": ("theorem solve_spec (xs : List Nat) (h : ValidInput xs) :\n"
                    "    solve xs h ≥ length xs - length xs + (Shelf.mk 1 2).width * «repeat» 0 := by\n  sorry"),
    "vc-postamble": '/--\ninfo: "solve"\n-/\n#guard_msgs in\n#eval toString "solve"\n\ndef main : IO Unit := pure ()',
}
DAFNY_REC = {
    "id": "DT9999", "language": "dafny", "source": "numpy_triple",
    "vc-description": "numpy.abs via `Abs`, Valid(r) and Res.Ok; Valid results are Ok",
    "vc-preamble": ('datatype Res = Ok(v: int) | Err\n\n// Valid is true for Ok: Valid(r) iff r.Ok?, never "Valid"\n'
                    'predicate Valid(r: Res) { r.Ok? && r != Res.Err }\n\nfunction Twice(x: int): int { 2 * x }\n\n'
                    'lemma TwiceOk(x: int)\n  ensures Twice(x) == x + x\n{\n}\n\nconst Name := "Valid"\n\n'
                    'method Main() { }'),
    "vc-spec": "method Abs(x: int) returns (r: int)\n  ensures r >= 0 && Valid(Ok(r))",
    "vc-code": "{\n  assume {:axiom} false;\n}", "vc-postamble": "",
}
# A #guard_msgs expected output that prints a renamed constructor (must follow the rename) next to one that prints a
# string literal equal to a declared name (must not): the builder's own LEAN_REC postamble showed the second bug.
GUARD_REC = {
    "id": "LA9997", "language": "lean", "source": "apps", "vc-description": "",
    "vc-preamble": "inductive Color where\n  | red\n  | green\nderiving Repr",
    "vc-helpers": "", "vc-definitions": "def pick (n : Nat) : Color :=\n  sorry",
    "vc-theorems": "theorem pick_spec (n : Nat) : pick n = pick n := by\n  sorry",
    "vc-postamble": ('/--\ninfo: Color.red\n-/\n#guard_msgs in\n#eval Color.red\n\n'
                     '/--\ninfo: "pick"\n-/\n#guard_msgs in\n#eval "pick"\n\n'
                     '/-- info: Color.green -/\n#guard_msgs in\n#eval pick 1'),
}


# ---------------------------------------------------------------------------------------------------- renamer
def test_lean_rename_is_consistent_and_keeps_library_names_and_literals():
    r2, names = S.rename(LEAN_REC, "lean")
    assert names == {"ValidInput": "P1", "Shelf": "T1", "width": "f2", "length": "f3", "«repeat»": "f4",
                     "f1": "f5", "solve": "f6", "solve_spec": "t1"}  # f1 is taken by the task: fresh names skip it
    pre, defs, thm, post = (r2[k] for k in ("vc-preamble", "vc-definitions", "vc-theorems", "vc-postamble"))
    assert "-- P1 says solve gets a list" in pre and "def P1 (xs : List Nat) : Prop" in pre  # prose `solve` kept
    assert "structure T1 where\n  head : Nat\n  tail : Nat" in pre and "s.tail - s.head" in pre  # fields stay
    assert "def T1.f2 (s : T1)" in pre and "(T1.mk 1 2).f2" in thm  # qualified name renamed after its dot
    assert "def f3 (xs : List Nat) : Nat := xs.length + List.length xs" in pre  # library `.length` untouched
    assert "def f4 (n : Nat)" in pre and "* f4 0" in thm and "c = 'f'" in pre and "def f5 (c : Char)" in pre
    assert defs.startswith("def f6 (xs : List Nat) (h : P1 xs) : Nat :=") and "f6 xs h ≥ f3 xs - f3 xs" in thm
    assert '/--\ninfo: "solve"\n-/' in post  # regression: the expected output of #eval "solve" is not renamed
    assert '#eval toString "solve"' in post and "def main" in post  # string literal and `main` kept
    assert r2["vc-description"] == "Compute f6(xs): see `P1` and `f4`. Use repeat to solve it."
    assert r2["vc-helpers"] == "" and r2["id"] == "LA9999"


def test_dafny_rename_constructors_discriminators_and_literals():
    r2, names = S.rename(DAFNY_REC, "dafny")
    assert names == {"Res": "T1", "Valid": "P1", "Twice": "f1", "TwiceOk": "t1", "Name": "k1", "Abs": "m1",
                     "Ok": "c1", "Err": "c2"}
    pre, spec = r2["vc-preamble"], r2["vc-spec"]
    assert pre.startswith("datatype T1 = c1(v: int) | c2") and "r.c1? && r != T1.c2" in pre
    assert '// Valid is true for Ok: P1(r) iff r.c1?, never "Valid"' in pre  # prose words and quotes kept
    assert 'const k1 := "Valid"' in pre and "method Main()" in pre
    assert "lemma t1(x: int)\n  ensures f1(x) == x + x" in pre
    assert spec == "method m1(x: int) returns (r: int)\n  ensures r >= 0 && P1(c1(r))"
    assert r2["vc-description"] == "numpy.abs via `m1`, P1(r) and T1.c1; Valid results are Ok"


def test_prose_keeps_plain_words_and_renames_identifiers():
    """Regression (review bug 1): DA0626 "the maximum possible f1 of a subset", LF3947 "how many f1 does Andrzej
    need", LT0606 "byte arrays f1 to empty strings" / "bytes.f1", DT0268 "c2 ≠ c2": plain words that equal a declared
    name stay; identifier-like mentions follow the rename."""
    rec = {"vc-preamble": ("def sum (xs : List Nat) : Nat := xs.foldl (· + ·) 0\n\n"
                           "-- Empty byte arrays decode to empty strings; decode(b) is total, as is bytes.decode\n"
                           "def decode (n : Nat) : Nat := n\n\ninductive Kind where\n  | NaN\n  | Finite\n\n"
                           "-- NaN ≠ NaN; Finite values give false; Kind.NaN is a Kind. "
                           "Calls best_sum, not \"best_sum\"\n"
                           "/-- `sum xs` adds up xs -/\ndef friends (n : Nat) : Nat := n"),
           "vc-helpers": "", "vc-definitions": "def best_sum (xs : List Nat) : Nat :=\n  sorry",
           "vc-theorems": "theorem best_sum_spec (xs : List Nat) : best_sum xs ≥ 0 := by\n  sorry",
           "vc-postamble": "",
           "vc-description": ("Find the maximum possible sum of a subset such that the sum is NOT a multiple of 10. "
                              "How many friends does Andrzej need? See `sum xs`, sum(xs), best_sum and Kind.Finite; "
                              "print \"sum\" or 'decode'.\n```python\nfriends 3 == 3\n```")}
    r2, names = S.rename(rec, "lean")
    assert names == {"sum": "f1", "decode": "f2", "Kind": "T1", "NaN": "c1", "Finite": "c2", "friends": "f3",
                     "best_sum": "f4", "best_sum_spec": "t1"}
    assert r2["vc-description"] == (
        "Find the maximum possible sum of a subset such that the sum is NOT a multiple of 10. How many friends does "
        "Andrzej need? See `f1 xs`, f1(xs), f4 and T1.c2; print \"sum\" or 'decode'.\n```python\nf3 3 == 3\n```")
    pre = r2["vc-preamble"]
    assert "-- Empty byte arrays decode to empty strings; f2(b) is total, as is bytes.decode\n" in pre
    assert "-- NaN ≠ NaN; Finite values give false; T1.c1 is a Kind. Calls f4, not \"best_sum\"\n" in pre
    assert "/-- `f1 xs` adds up xs -/\ndef f3 (n : Nat)" in pre and "def f1 (xs : List Nat)" in pre


def test_guard_msgs_expected_output_follows_code_not_prose():
    """#eval prints renamed constructors (`T1.c1`) but never a renamed string literal (`"pick"` stays)."""
    r2, names = S.rename(GUARD_REC, "lean")
    assert names == {"Color": "T1", "red": "c1", "green": "c2", "pick": "f1", "pick_spec": "t1"}
    assert r2["vc-postamble"] == ('/--\ninfo: T1.c1\n-/\n#guard_msgs in\n#eval T1.c1\n\n/--\ninfo: "pick"\n-/\n'
                                  '#guard_msgs in\n#eval "pick"\n\n/-- info: T1.c2 -/\n#guard_msgs in\n#eval f1 1')


def test_names_inside_bodies_and_fields_are_not_top_level():
    rec = {"vc-preamble": "datatype P = P(x: int)\nfunction F(p: P): int { var y := p.x; y }", "vc-spec": "",
           "vc-postamble": ""}
    assert S.rename(rec, "dafny")[1] == {"P": "T1", "F": "f1"}


@needs_lean
@needs_dafny
def test_synthetic_renamed_tasks_lock_like_the_originals(dafny_env):
    """«keyword» names, qualified names, structures, constructors and discriminators: the renamed copies lock, and
    their #guard_msgs tests still pass (review bug 3: LEAN_REC's renamed postamble used to expect "f6")."""
    for lang, rec in (("lean", LEAN_REC), ("lean", GUARD_REC), ("dafny", DAFNY_REC)):
        r2, names = S.rename(rec, lang)
        (_, f0), (sol2, f1) = S.files(lang, rec), S.files(lang, r2)
        S.same_task(f0, f1, names)
        assert S.postamble(lang, rec) == [] and S.postamble(lang, r2) == [], rec["id"]
        assert "f1" in sol2 if rec is GUARD_REC else "f4" in sol2 if lang == "lean" else "c1?" in sol2


@needs_lean
def test_prepare_drops_a_renamed_form_whose_guard_msgs_cannot_pass(monkeypatch):
    """The lock alone does not compile the postamble: prepare also requires skeleton + postamble to give only sorry
    aborts, so a renamed #guard_msgs expectation that can never pass leaves only the original."""
    good = S.rename(GUARD_REC, "lean")
    bad = ({**good[0], "vc-postamble": good[0]["vc-postamble"].replace('info: "pick"', 'info: "f1"')}, good[1])
    curated = {"LA9997": {"status": "OK", "reason": "test"}}
    for ren, ok in ((good, True), (bad, False)):
        monkeypatch.setattr(S, "rename", lambda rec, lang, ren=ren: ren)
        p = S.prepare("L-apps", {"LA9997": GUARD_REC}, ["LA9997"], curated, n=1)
        assert [t["id"] for t in p["tasks"]] == ["LA9997"] and ("ren" in p["tasks"][0]["forms"]) == ok
        assert ok or "skeleton + postamble" in p["renamed_failed"]["LA9997"]


# ---------------------------------------------------------------------------------------------------- curation
def _fake_files(lang, rec, frozen_dir=None):
    return f"SOL {rec['id']} {rec.get('vc-spec', '')}\n", {"preamble": rec.get("vc-preamble", ""),
                                                          "spec": rec.get("vc-spec", ""), "postamble": ""}


def _drec(i):
    return {"id": i, "language": "dafny", "source": "numpy_triple", "vc-description": f"task {i}",
            "vc-preamble": "", "vc-spec": f"method M{i}(x: int) returns (r: int)", "vc-postamble": ""}


def test_prepare_selects_only_curated_ok_and_lists_uncurated(monkeypatch):
    """Regression (review bug 2): DA0473 is VACUOUS (its default body is PROVED) and used to be selected."""
    monkeypatch.setattr(S, "files", _fake_files)
    monkeypatch.setattr(S, "postamble", lambda lang, rec: [])
    ids = ["D1", "D2", "D3", "D4", "D5", "D6"]
    curated = {"D1": {"status": "OK"}, "D2": {"status": "VACUOUS", "reason": "default body PROVED"},
               "D3": {"status": "TRIVIAL_THEOREMS", "reason": "battery"}, "D5": {"status": "OK"},
               "D6": {"status": "OK"}}
    p = S.prepare("D-np", {i: _drec(i) for i in ids}, ids, curated, n=2)
    assert [t["id"] for t in p["tasks"]] == ["D1"] and p["uncurated"] == ["D4"]  # D4 holds the second slot
    assert p["skipped"] == {"D2": "curated VACUOUS: default body PROVED", "D3": "curated TRIVIAL_THEOREMS: battery"}
    p = S.prepare("D-np", {i: _drec(i) for i in ids}, ids, {**curated, "D4": {"status": "BROKEN"}}, n=2)
    assert [t["id"] for t in p["tasks"]] == ["D1", "D5"] and p["uncurated"] == [] and "D4" in p["skipped"]


def test_prepare_flags_frozen_dir_locks_that_differ(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "files", lambda lang, rec, frozen_dir=None: ("SOL\n", {"constants": {}, "chunks": [],
                                                                                    "preamble": rec["id"]}))
    monkeypatch.setattr(S, "postamble", lambda lang, rec: [])
    monkeypatch.setattr(S, "FROZEN_DIR", tmp_path)
    (tmp_path / "L1.json").write_text(json.dumps({"constants": {}, "chunks": [], "preamble": "L1"}))
    (tmp_path / "L2.json").write_text(json.dumps({"constants": {}, "chunks": [], "preamble": "older rung"}))
    recs = {i: {**LEAN_REC, "id": i} for i in ("L1", "L2", "L3")}
    p = S.prepare("L-apps", recs, ["L1", "L2", "L3"], {i: {"status": "OK"} for i in recs}, n=3)
    assert p["frozen_dir_differs"] == ["L2"] and len(p["tasks"]) == 3


@pytest.fixture
def fake_build(tmp_path, monkeypatch):
    """build() on a tiny split with fake locks: D-np has D1..D4 in seeded order, the other cells are empty."""
    split = {"dev": {c: [] for c in S.STAGE0_CELLS}, "test": {c: [] for c in S.STAGE0_CELLS}}
    split["dev"]["D-np"] = [{"ids": [i]} for i in ("D1", "D2", "D3", "D4")]
    (tmp_path / "split.json").write_text(json.dumps(split))
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(_drec(i)) + "\n" for i in ("D1", "D2", "D3", "D4")))
    out = tmp_path / "stage0"
    for k, v in {"SPLIT": tmp_path / "split.json", "DATA": tmp_path / "data.jsonl", "CURATED": tmp_path / "cur.json",
                 "OUT": out, "REQUESTS": out / "requests.jsonl", "TASKS": out / "tasks.json",
                 "BATCH": out / "batch.json", "files": _fake_files, "postamble": lambda lang, rec: []}.items():
        monkeypatch.setattr(S, k, v)
    sha = hashlib.sha256((tmp_path / "split.json").read_bytes()).hexdigest()

    def curate(tasks, split_sha=sha):
        S.CURATED.write_text(json.dumps({"split_sha256": split_sha, "tasks": tasks}))
    return curate


def test_build_refuses_uncurated_candidates_and_records_curation(fake_build, capsys):
    with pytest.raises(SystemExit, match="is missing"):  # no curated.json at all
        S.build(2)
    fake_build({"D1": {"status": "OK"}}, split_sha="other split")
    with pytest.raises(SystemExit, match="another split"):
        S.build(2)
    fake_build({"D1": {"status": "OK"}, "D2": {"status": "VACUOUS", "reason": "default body PROVED"}})
    with pytest.raises(SystemExit, match=r'no curated.json status.*"D-np": \["D3"\]'):
        S.build(2)
    assert not S.OUT.exists()  # nothing written
    fake_build({"D1": {"status": "OK"}, "D2": {"status": "VACUOUS", "reason": "default body PROVED"},
                "D3": {"status": "OK"}})
    summary = S.build(2)
    assert summary["D-np"]["ids"] == ["D1", "D3"] and summary["D-np"]["skipped"] == {
        "D2": "curated VACUOUS: default body PROVED"}
    tasks = json.loads(S.TASKS.read_text())
    assert tasks["curated_sha256"] == hashlib.sha256(S.CURATED.read_bytes()).hexdigest()
    assert tasks["rungs"] == S.rung_hashes() and set(tasks["rungs"]) == set(S.RUNGS)
    assert [json.loads(x)["custom_id"] for x in S.REQUESTS.read_text().splitlines()] == [
        "s0__D-np__D1__orig", "s0__D-np__D1__ren", "s0__D-np__D3__orig", "s0__D-np__D3__ren"]


def test_build_after_submit_keeps_requests_or_refuses(fake_build):
    """Stale-lock note: a rebuild after a rung change refreshes the locks but may not change a submitted batch."""
    fake_build({i: {"status": "OK"} for i in ("D1", "D2", "D3")})
    S.build(2)
    before = S.REQUESTS.read_text()
    S.BATCH.write_text('{"id": "b"}')
    S.build(2)  # same inputs: same requests, locks refreshed
    assert S.REQUESTS.read_text() == before
    fake_build({"D1": {"status": "VACUOUS", "reason": "x"}, "D2": {"status": "OK"}, "D3": {"status": "OK"}})
    with pytest.raises(SystemExit, match="would change"):
        S.build(2)
    assert S.REQUESTS.read_text() == before


def test_submit_refuses_a_build_that_is_not_current(fake_build, monkeypatch):
    fake_build({i: {"status": "OK"} for i in ("D1", "D2", "D3")})
    S.build(2)
    tasks = json.loads(S.TASKS.read_text())
    S.TASKS.write_text(json.dumps({**tasks, "rungs": {**tasks["rungs"], S.RUNGS[0]: "0" * 64}}))
    with pytest.raises(SystemExit, match="changed since build"):
        S.submit()
    S.TASKS.write_text(json.dumps({k: v for k, v in tasks.items() if k != "curated_sha256"}))  # pre-curation build
    with pytest.raises(SystemExit, match="predates curation"):
        S.submit()


# ---------------------------------------------------------------------------------------------------- requests
@pytest.mark.parametrize("lang,rec", [("lean", LEAN_REC), ("dafny", DAFNY_REC)])
def test_prompt_keeps_the_agent_rules_and_asks_for_the_whole_file(lang, rec):
    sol = "SKELETON\n"
    text = S.prompt(lang, rec["vc-description"], sol)
    rules = ad.instruction(lang, rec["vc-description"]).split("\nYou run autonomously")[0]
    fence = "lean" if lang == "lean" else "dafny"
    assert text.startswith(rules) and "Rules:\n- " in text and "You run autonomously" not in text
    assert f"complete file {ad.SOLUTION[lang]} in one ```{fence} code block" in text
    assert text.endswith(f"```{fence}\nSKELETON\n```\n")
    req = S.request(f"s0__L-apps__{rec['id']}__ren", text)
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", req["custom_id"])
    assert req["params"] == {"model": "claude-haiku-5-5", "max_tokens": 16000, "output_config": {"effort": "medium"},
                             "messages": [{"role": "user", "content": text}]}


def test_extract_takes_the_longest_block():
    assert S.extract("plan\n```lean\ndef a := 1\ndef b := 2\n```\nthen\n```\nx\n```") == "def a := 1\ndef b := 2\n"
    assert S.extract("no code") == "" and S.extract(None) == ""


def test_stage0_cells_include_l_np_only_with_enough_dev():
    dev = {c: [{}] * 20 for c in S.STAGE0_CELLS}
    assert S.stage0_cells({"dev": {**dev, "L-np": [{}] * 5}}) == list(S.STAGE0_CELLS)
    assert "L-np" not in S.stage0_cells({"dev": {**dev, "L-np": [{}] * 4}})


# ---------------------------------------------------------------------------------------------------- rules
def _grades(n, orig, ren=None, comp=0, pairs=None):
    ids = [f"T{i}" for i in range(n)]
    g = {(i, "orig"): {"status": "PROVED" if k < orig else "UNPROVED", "compiles": k < max(orig, comp)}
         for k, i in enumerate(ids)}
    for k, i in enumerate(ids[:n if pairs is None else pairs]):
        if ren is not None:
            g[i, "ren"] = {"status": "PROVED" if k < ren else "UNPROVED", "compiles": True}
    return ids, g


@pytest.mark.parametrize("args,flags", [
    ((20, 16, 16), ["CEILING"]),                        # >= 80%
    ((20, 15, 15), []),
    ((20, 0, 0, 4), ["FLOOR-RISK"]),                    # 0 verified and 4/20 = 20% compile
    ((20, 0, 0, 5), []),                                # 25% compile is not < 25%
    ((20, 8, 4), ["MEMO-SUSPECT"]),                     # -20pp and 4 tasks
    ((20, 7, 4), []),                                   # 3 tasks
    ((7, 3, 0), []),                                    # 43pp but only 3 tasks
    ((20, 16, 10), ["CEILING", "MEMO-SUSPECT"]),
    ((20, 8, 4, 0, 6), []),                             # paired tasks only: 6 vs 4 (not 8 vs 4)
])
def test_decide_thresholds(args, flags):
    ids, g = _grades(*args)
    assert S.decide(ids, g, {i for i, f in g if f == "ren"})["flags"] == flags


def test_decide_refuses_missing_rows():
    """Regression (review bug 4): a missing original grade used to count as a failure (and a missing renamed one
    silently dropped the pair); both are now an error, since collect writes a row for every item."""
    ids, g = _grades(20, 20, 20)
    for k in (("T0", "orig"), ("T0", "ren")):
        with pytest.raises(ValueError, match="rerun collect"):
            S.decide(ids, {x: r for x, r in g.items() if x != k}, set(ids))
    assert S.decide(ids, {x: r for x, r in g.items() if x[1] == "orig"}, set())["pairs"] == 0  # no renamed form


def test_decide_excludes_infrastructure_failures_from_both_forms_alike():
    """Review bug 4's numbers: dropping 4 originals of proved tasks gave orig-ren = -4, dropping 4 renamed ones gave
    pairs = 16 with no flag. Now either way the 4 tasks leave the comparison and the cell is INCOMPLETE."""
    ids, g = _grades(20, 20, 20)
    for form in ("orig", "ren"):
        for status in S.EXCLUDED:
            h = {**g, **{(i, form): {"status": status, "compiles": False} for i in ids[:4]}}
            out = S.decide(ids, h, set(ids))
            assert (out["pairs"], out["orig_minus_ren"], out["excluded"]) == (16, 0, ids[:4])
            assert out["n"] == (16 if form == "orig" else 20) and out["flags"] == ["CEILING", "INCOMPLETE"]
    h = {**g, ("T0", "orig"): {"status": "GRADE_ERROR"}, ("T1", "ren"): {"status": "NO_RESPONSE"}}
    assert S.decide(ids, h, set(ids))["flags"] == ["CEILING"]  # 2/20 = 10% lost: not INCOMPLETE
    h = {**g, ("T0", "orig"): {"status": "NO_CODE", "compiles": False}}  # the model's failure, not excluded
    out = S.decide(ids, h, set(ids))
    assert (out["n"], out["orig_proved"], out["pairs"], out["orig_minus_ren"]) == (20, 19, 20, -1)


# ---------------------------------------------------------------------------------------------------- toolchains
@pytest.fixture
def dafny_env(monkeypatch):
    monkeypatch.setenv("MILLENNIUM_DAFNY", DAFNY)


def _needs(cell):
    lang = S.CELLS[cell][0]
    return pytest.mark.skipif((LEAN if lang == "lean" else DAFNY) is None, reason=f"no {lang} toolchain")


@has_data
@pytest.mark.parametrize("cell", [pytest.param(c, marks=_needs(c)) for c in S.STAGE0_CELLS])
def test_renamed_dev_skeletons_lock(cell, monkeypatch, capsys):
    """Plan §4 Stage 0 on the real DEV order: >= 90% of the selected tasks must lock after renaming."""
    if DAFNY:
        monkeypatch.setenv("MILLENNIUM_DAFNY", DAFNY)
    split = json.loads(S.SPLIT.read_text())
    if cell not in S.stage0_cells(split):
        pytest.skip(f"{cell} has fewer than {S.MIN_DEV} DEV components")
    ids = ad.select_ids(split, cell, "dev", len(split["dev"][cell]))
    ok = {i: {"status": "OK"} for i in ids}  # the renamer over every candidate, not a curated selection
    p = S.prepare(cell, S.load_records(set(ids)), ids, ok)
    with capsys.disabled():
        print(f"\n{cell}: {len(p['tasks'])} tasks, {len(p['skipped'])} skipped (original does not lock), renamed lock "
              f"{p['renamed_ok']}/{len(p['tasks'])} = {p['renamed_rate']:.0%}")
    assert len(p["tasks"]) == min(S.N, len(ids) - len(p["skipped"])) and p["renamed_rate"] >= 0.9, p["renamed_failed"]
    for t in p["tasks"]:
        if "ren" in t["forms"]:
            sol = t["forms"]["ren"]["sol"]  # no old name left in code (prose may keep plain words)
            code = sl_lean.mask(sol)[0] if S.CELLS[cell][0] == "lean" else sl_dafny.strip_comments(sol)
            left = [o for o in t["names"] if re.search(rf"(?<![\w'.])\b{re.escape(o)}(?![\w'!?])", code)]
            assert t["names"] and sol != t["forms"]["orig"]["sol"] and not left, (t["id"], left)


@needs_lean
def test_grade_lean():
    rec = {"id": "LA9998", "vc-preamble": "@[reducible, simp]\ndef pre (n : Nat) : Prop := n > 0", "vc-helpers": "",
           "vc-definitions": "def solve (n : Nat) (h : pre n) : Nat :=\n  sorry",
           "vc-theorems": "theorem solve_spec (n : Nat) (h : pre n) : solve n h ≥ 1 := by\n  sorry",
           "vc-postamble": ""}
    sol, frozen = S.files("lean", rec)
    honest = sol.replace(":=\n  sorry", ":=\n  n", 1).replace("by\n  sorry", "by\n  exact h")
    job = lambda text: S.grade({"custom_id": "x", "lang": "lean", "text": text, "frozen": frozen})  # noqa: E731
    assert (job(honest)["status"], job(honest)["compiles"]) == ("PROVED", True)
    r = job(sol)
    assert r["status"] == "CHEAT_REJECTED" and r["compiles"]  # `sorry` compiles but never counts
    r = job(honest.replace("exact h", "exact Nat.le_refl 1"))
    assert r["status"] == "UNPROVED" and not r["compiles"]
    assert job("  ")["status"] == "NO_CODE"
    r = job(honest.split("theorem solve_spec")[0])  # review bug 5: deleting the theorem compiles but is no attempt
    assert r["status"] == "SPEC_TAMPERED" and not r["compiles"]


@needs_dafny
def test_grade_dafny(dafny_env):
    rec = {"id": "DT9998", "vc-preamble": "", "vc-spec": "method Abs(x: int) returns (r: int)\n  ensures r >= 0",
           "vc-postamble": ""}
    sol, frozen = S.files("dafny", rec)
    job = lambda text: S.grade({"custom_id": "x", "lang": "dafny", "text": text, "frozen": frozen})  # noqa: E731
    r = job(sol.replace("// implement", "r := if x < 0 then -x else x;"))
    assert (r["status"], r["compiles"]) == ("PROVED", True)
    r = job(sol.replace("// implement", "r := x;"))
    assert r["status"] == "UNPROVED" and r["compiles"]  # resolves, does not verify
    r = job(sol.replace("// implement", "r := y;"))
    assert r["status"] == "UNPROVED" and not r["compiles"]
    r = job(sol.replace("ensures r >= 0", "ensures true").replace("// implement", "r := x;"))  # review bug 5
    assert r["status"] == "SPEC_TAMPERED" and not r["compiles"]


def test_grade_never_counts_spec_tampering_as_compiling(monkeypatch):
    """Regression (review bug 5): `compiles` was `PROVED or compiles(text)`, true for a reply that rewrites the spec."""
    monkeypatch.setattr(S, "compiles", lambda lang, text: True)
    for status, ok in (("SPEC_TAMPERED", False), ("UNPROVED", True), ("CHEAT_REJECTED", True), ("PROVED", True)):
        monkeypatch.setattr(S.sl_dafny, "check", lambda text, frozen, path, status=status: {"status": status})
        assert S.grade({"custom_id": "x", "lang": "dafny", "text": "x", "frozen": {}})["compiles"] is ok, status


@pytest.fixture
def fake_collect(tmp_path, monkeypatch):
    """tasks.json with 3 items, all answered; grade() is replaced per test."""
    for k in ("TASKS", "ANSWERS", "GRADES", "REPORT"):
        monkeypatch.setattr(S, k, tmp_path / getattr(S, k).name)
    items = {f"s0__D-np__D{k}__{f}": {"cell": "D-np", "id": f"D{k}", "form": f, "lang": "dafny", "frozen": {}}
             for k, f in ((1, "orig"), (1, "ren"), (2, "orig"))}
    S.TASKS.write_text(json.dumps({"rungs": S.rung_hashes(), "cells": {"D-np": {"ids": ["D1", "D2"]}},
                                   "items": items}))
    S.ANSWERS.write_text("".join(json.dumps({"custom_id": c, "result": "succeeded", "text": "```\nx\n```"}) + "\n"
                                 for c in items))
    return items


def _ok(job):
    return {"custom_id": job["custom_id"], "status": "PROVED", "violations": [], "compiles": True}


def test_collect_writes_grade_error_rows_locally(fake_collect, monkeypatch):
    """Regression (review bug 4): an answer the grader cannot finish is a GRADE_ERROR row, not a missing row."""
    def grade(job):
        if job["custom_id"].endswith("D1__ren"):
            raise RuntimeError("grader crashed")
        return _ok(job)
    monkeypatch.setattr(S, "grade", grade)
    assert S.collect(workers=2) == 1
    g = {r["custom_id"]: r for r in map(json.loads, S.GRADES.read_text().splitlines())}
    assert set(g) == set(fake_collect) and g["s0__D-np__D1__ren"]["status"] == "GRADE_ERROR"
    assert "grader crashed" in g["s0__D-np__D1__ren"]["violations"][0]
    out = S.report()["D-np"]
    assert (out["n"], out["pairs"], out["grade_errors"], out["excluded"]) == (2, 0, 1, ["D1"])
    assert "INCOMPLETE" in out["flags"]
    with pytest.raises(SystemExit) as e:
        S.main(["collect"])
    assert e.value.code == 1


def test_collect_with_modal_failures_writes_grade_error_rows(fake_collect, monkeypatch):
    """Regression (review bug 4): with --modal, exceptions were printed and dropped while grades.jsonl was written."""
    class Remote:
        @staticmethod
        def map(jobs, return_exceptions, order_outputs):
            assert return_exceptions
            for j in reversed(jobs):
                yield RuntimeError("container died") if j["custom_id"].endswith("D2__orig") else _ok(j)
    monkeypatch.setattr(S, "modal", types.SimpleNamespace())
    monkeypatch.setattr(S, "app", types.SimpleNamespace(run=contextlib.nullcontext), raising=False)
    monkeypatch.setattr(S, "grade_remote", Remote, raising=False)
    assert S.collect(use_modal=True) == 1
    g = {r["custom_id"]: r for r in map(json.loads, S.GRADES.read_text().splitlines())}
    assert set(g) == set(fake_collect) and g["s0__D-np__D2__orig"]["status"] == "GRADE_ERROR"
    assert "container died" in g["s0__D-np__D2__orig"]["violations"][0]
    out = S.report()["D-np"]
    assert (out["n"], out["orig_proved"], out["pairs"]) == (1, 1, 1)  # D2 is excluded, not a failure


def test_collect_refuses_after_a_rung_change(fake_collect, monkeypatch):
    """Stale-lock note: tasks.json stores the rung hashes its locks were made with."""
    monkeypatch.setattr(S, "grade", _ok)
    tasks = json.loads(S.TASKS.read_text())
    for rungs in ({**tasks["rungs"], "millennium_harness/lean_check/Check.lean": "0" * 64}, None):
        S.TASKS.write_text(json.dumps({**tasks, "rungs": rungs}))
        with pytest.raises(SystemExit, match="changed since build"):
            S.collect()
        assert not S.GRADES.exists()


@needs_lean
@needs_dafny
def test_collect_and_report_offline(tmp_path, monkeypatch, dafny_env, capsys):
    """collect + report on a fake answers file (no Batch API, no Modal): answers are graded with the local rungs."""
    for k in ("TASKS", "ANSWERS", "GRADES", "REPORT"):
        monkeypatch.setattr(S, k, tmp_path / getattr(S, k).name)
    lrec = {"id": "LA9998", "vc-preamble": "", "vc-helpers": "", "vc-postamble": "",
            "vc-definitions": "def solve (n : Nat) : Nat :=\n  sorry",
            "vc-theorems": "theorem solve_spec (n : Nat) : solve n = n := by\n  sorry"}
    drec = {"id": "DT9998", "vc-preamble": "", "vc-spec": "method Abs(x: int) returns (r: int)\n  ensures r >= 0",
            "vc-postamble": ""}
    items, answers = {}, []
    for cell, rec, lang, good in (("L-apps", lrec, "lean", ("  n", "rfl")), ("D-np", drec, "dafny", None)):
        for form in ("orig", "ren"):
            r = rec if form == "orig" else S.rename(rec, lang)[0]
            sol, frozen = S.files(lang, r)
            cid = f"s0__{cell}__{rec['id']}__{form}"
            items[cid] = {"cell": cell, "id": rec["id"], "form": form, "lang": lang, "frozen": frozen, "names": {}}
            if lang == "lean":
                ans = sol.replace("  sorry", good[0], 1).replace("  sorry", "  " + good[1]) if form == "orig" else None
            else:
                ans = sol.replace("// implement", "r := if x < 0 then -x else x;" if form == "ren" else "r := x;")
            answers.append({"custom_id": cid, "result": "succeeded", "text": f"Here:\n```{lang}\n{ans}```", "stop":
                            "end_turn"} if ans else {"custom_id": cid, "result": "errored"})
    S.TASKS.write_text(json.dumps({"cells": {"L-apps": {"ids": ["LA9998"]}, "D-np": {"ids": ["DT9998"]}},
                                   "items": items, "rungs": S.rung_hashes()}))
    S.ANSWERS.write_text("".join(json.dumps(a) + "\n" for a in answers))
    assert S.collect(workers=2) == 0
    g = {(r["cell"], r["form"]): r["status"] for r in map(json.loads, S.GRADES.read_text().splitlines())}
    assert g == {("L-apps", "orig"): "PROVED", ("L-apps", "ren"): "NO_RESPONSE", ("D-np", "orig"): "UNPROVED",
                 ("D-np", "ren"): "PROVED"}
    out = S.report()
    assert out["L-apps"]["flags"] == ["CEILING", "INCOMPLETE"] and out["L-apps"]["no_response"] == 1
    assert out["L-apps"]["pairs"] == 0 and out["L-apps"]["excluded"] == ["LA9998"]
    assert out["D-np"]["flags"] == [] and out["D-np"]["ren_only"] == ["DT9998"]
    assert json.loads(S.REPORT.read_text())["D-np"]["compile_rate"] == 1.0
