"""bench.fv.curate: pure helpers always run; synthetic Lean/Dafny controls (every status, every vacuity variant, the
TRIVIAL_THEOREMS rule) need the pinned toolchains and skip otherwise; runs/fv/curated.json invariants skip until a
curation run exists. Each review finding (wf_a66d7273 review 6, bugs 1-8) has a regression test named after it."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from bench.fv import curate as C
from millennium_harness import dafny as sl_dafny
from millennium_harness import lean as sl_lean

try:
    LEAN = sl_lean.find_lean()
except (sl_lean.ToolchainError, OSError, subprocess.SubprocessError):
    LEAN = None
DAFNY = next((d for d in (os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY")), str(C.LOCAL_DAFNY),
                          shutil.which("dafny"))
              if d and os.path.exists(d)), None)
needs_lean = pytest.mark.skipif(LEAN is None, reason=f"Lean {sl_lean.TOOLCHAIN} not installed")
needs_dafny = pytest.mark.skipif(DAFNY is None, reason="Dafny 4.11 not installed (set MILLENNIUM_DAFNY)")
STD_DO = "import Std.Do.Triple\nimport Std.Tactic.Do\nopen Std.Do"


def lean_rec(defs="def f (n : Nat) : Nat :=\n  sorry", thms="", post="", pre="", rid="T"):
    return {"id": rid, "language": "lean", "vc-preamble": pre, "vc-helpers": "", "vc-definitions": defs,
            "vc-theorems": thms, "vc-postamble": post}


def guard(expr, out):
    return f"/--\ninfo: {out}\n-/\n#guard_msgs in\n#eval {expr}"


# ------------------------------------------------------------------------------------------------ pure Python
def test_fill_replaces_only_agent_sorries_and_keeps_the_lock():
    rec = lean_rec(defs="def f (xs : List Nat) : List Nat :=\n  sorry\n\ndef g : Nat → Nat\n  | 0 => 1\n  | _ => sorry\n\n"
                        "def h (n : Nat) : Nat := by\n  sorry\n\ndef k : Nat → Nat :=\n  fun _ => sorry",
                   thms="theorem t1 (xs : List Nat) : f xs = xs := sorry\n\ntheorem t2 : g 0 = 1 := by\n  sorry",
                   post="-- sorry in a comment stays")
    text = C.fill(rec, ["exact xs", "exact default", "exact n", "exact pure default"], ["P1", "P2"])
    assert "def f (xs : List Nat) : List Nat :=\n  (by exact xs)" in text
    assert "| _ => (by exact default)" in text and "fun _ => (by exact pure default)" in text
    assert "def h (n : Nat) : Nat := by\n  exact n" in text
    assert ":= (by P1)" in text and ":= by\n  P2" in text
    assert text.count("sorry") == 1 and text.rstrip().endswith("-- sorry in a comment stays")
    assert "-- sorry in a comment stays" not in C.fill(rec, "exact default", "sorry", postamble=False)
    fz = {"preamble": "", "postamble": rec["vc-postamble"], "chunks": sl_lean._chunks(rec)}
    hs, v = sl_lean.holes(C.fill(rec, "exact default", "(simp_all [f, g]; done)"), fz)
    assert not v and sl_lean.ban_scan(hs) == []
    assert [(h["kind"], h["name"], h["args"]) for h in C.lean_holes(rec)] == [
        ("impl", "f", ["xs"]), ("impl", "g", []), ("impl", "h", ["n"]), ("impl", "k", []),
        ("proof", "t1", ["xs"]), ("proof", "t2", [])]


def test_fill_renames_examples_only_when_asked():
    rec = lean_rec(thms="example (n : Nat) : f n ≥ 0 := sorry\n\ntheorem t : f 0 = 0 := sorry\n\nexample : True := sorry")
    text = C.fill(rec, "exact default", "P", rename_examples=True)
    assert "theorem _sl_example_0 (n : Nat) : f n ≥ 0 := (by P)" in text and "theorem _sl_example_2 : True" in text
    assert "example " not in text and "example (n : Nat)" in C.fill(rec, "exact default", "P")


def test_bug2_explicit_args_lists_every_explicit_binder():
    assert C._explicit_args("def f {α : Type} [BEq α] (xs ys : List α) (n : Nat := 3) : Nat :=") == ["xs", "ys", "n"]
    assert C._explicit_args("def f : (Nat × Nat) → Nat :=") == []
    assert C._explicit_args("def f (_ : Nat) (h : (a : Nat) → a = a) : Nat :=") == ["h"]
    assert C._explicit_args("def f(a b : Nat) (c : Nat) : Nat :=") == ["a", "b", "c"]
    assert C._explicit_args("theorem t (h₁ : a > 0) (x' : Nat) : x' = x' :=") == ["h₁", "x'"]


def test_tactics_unfold_impls_and_gate_mvcgen():
    plain = C.tactics(lean_rec(), ["f", "g"])
    assert plain[0] == "rfl" and "simp_all [f, g]" in plain and not any("mvcgen" in t for t in plain)
    assert "mvcgen [f] <;> simp_all [f]" in C.tactics(lean_rec(pre=STD_DO), ["f"])
    assert C.tactics(lean_rec(), []).count("simp_all") == 1 and len(C.tactics(lean_rec(), [])) == 6
    assert not C.std_do(lean_rec(pre="-- import Std.Do\nimport Std.Data.HashMap"))


def test_bug2_variants_combine_defs_independently_and_cap():
    vs, capped = C.variants([["d", "p"], ["d", "a", "b"]])
    assert vs[0] == ("d", "d") and len(vs) == 6 and ("p", "b") in vs and not capped
    vs, capped = C.variants([["d"] + [f"x{k}" for k in range(9)]] * 3, cap=64)
    assert capped and len(vs) <= 64 and vs[0] == ("d", "d", "d")
    assert C.variants([]) == ([()], False)


def test_axioms_lines_parse():
    out = ("S.lean:3:8: error: Tactic `rfl` failed\n't'' depends on axioms: [sorryAx]\n'Foo.t' does not depend on "
           "any axioms\n'«my thm»' depends on axioms: [propext,\n Classical.choice, Quot.sound]\n")
    ax = {m.group(1): {x.strip() for x in (m.group(2) or "").split(",") if x.strip()} for m in C.AXIOMS_LINE.finditer(out)}
    assert ax == {"t'": {"sorryAx"}, "Foo.t": set(), "«my thm»": {"propext", "Classical.choice", "Quot.sound"}}
    assert C._clean(ax, "Foo.t") and C._clean(ax, "«my thm»") and not C._clean(ax, "t'") and not C._clean(ax, "u")


def test_lean_axioms_reads_each_answer_from_its_own_slice(monkeypatch):
    seen = {}

    def fake(text, lean, timeout):
        seen["text"] = text
        return False, ("S.lean:2:0: error: Tactic `rfl` failed\nsl_curate_axioms_0\n'_private.S.0.pp' does not depend on "
                       "any axioms\nsl_curate_axioms_1\nS.lean:9:14: error: Unknown constant `t`\nsl_curate_axioms_2\n"
                       "'repeat' depends on axioms: [sorryAx]\nsl_curate_end_of_file\n"), False
    monkeypatch.setattr(C, "_compile", fake)
    assert C.lean_axioms("X", ["pp", "t", "«repeat»"], "lean", 10) == {"pp": set(), "«repeat»": {"sorryAx"}}
    assert '#print "sl_curate_axioms_1"\n#print axioms t\n' in seen["text"]


def test_bug3_lean_dying_before_the_end_is_an_infra_error(monkeypatch):
    monkeypatch.setattr(C, "_compile", lambda text, lean, timeout: (False, "Segmentation fault", False))
    with pytest.raises(C.InfraError):
        C.lean_axioms("theorem t : True := trivial", ["t"], "lean", 10)
    monkeypatch.setattr(C, "_compile", lambda text, lean, timeout: (False, "", True))
    assert C.lean_axioms("theorem t : True := trivial", ["t"], "lean", 10) is None
    with pytest.raises(C.InfraError):
        C.postamble_problems(lean_rec(), "lean")
    monkeypatch.setattr(C, "_compile", lambda text, lean, timeout: (False, "Killed", False))
    with pytest.raises(C.InfraError):  # no error message: a crash or an OOM kill, not a BROKEN postamble
        C.postamble_problems(lean_rec(), "lean")


ABORT_MSG = ("error: aborting evaluation since the expression depends on the 'sorry' axiom, which can lead to runtime "
             "instability and crashes.")


@pytest.mark.parametrize("severity,text,allowed", [
    ("warning", "S.lean:1:4: warning: declaration uses 'sorry'", True),
    ("warning", "S.lean:1:4: warning: unused variable `h`", True),
    ("error", f"S.lean:9:0: {ABORT_MSG}\n\nTo attempt to evaluate anyway despite the risks, use the '#eval!' command.",
     True),
    ("error", "S.lean:8:0: error: ❌️ Docstring on `#guard_msgs` does not match generated message:\n\n- info: [2]\n"
              f"+ {ABORT_MSG}\n+\n+ To attempt to evaluate anyway despite the risks, use the '#eval!' command.", True),
    ("error", "S.lean:8:0: error: ❌️ Docstring on `#guard_msgs` does not match generated message:\n\n- info: 3\n"
              "+ info: 2", False),
    ("error", "S.lean:6:6: error(lean.unknownIdentifier): Unknown identifier `len`", False),
    ("error", "S.lean:6:11: error: unexpected token '('; expected command", False),
])
def test_sorry_only(severity, text, allowed):
    assert C._sorry_only({"severity": severity, "text": text}) is allowed


def test_dafny_defaults_and_bodies():
    assert C.dafny_default("map<int, seq<int>>") == "map[]" and C.dafny_default("array<real>") == "new real[0]"
    assert C.dafny_default("(int, (bool, string))") == '(0, (false, ""))' and C.dafny_default("bv8") == "0"
    assert C.dafny_default("Color") is None and C.dafny_default("int -> int") is None
    assert C._split_top("a: int -> int, b: map<int,int>") == ["a: int -> int", " b: map<int,int>"]
    assert C.dafny_body("method M(x: int) returns (r: int, ghost s: seq<D>, d: D)\n  ensures r >= 0") == \
        "{\n  r := 0;\n  s := [];\n  d := *;\n}"
    assert C.dafny_body("method M(a: array<int>) // returns (r: int)\n  modifies a") == "{\n}"
    assert C.dafny_body("function F<T>(s: seq<T>): (r: nat)\n  ensures r <= |s|") == "{\n  0\n}"
    assert C.dafny_body("predicate P(x: int)\n  ensures P(x) ==> x > 0") == "{\n  false\n}"
    assert C.dafny_body("function F(x: int): Color") is None


SPEC = "method M(x: int) returns (r: int)\n  ensures r == x + 1"


def test_bug7_dafny_frozen_axioms():
    fz = lambda pre="", post="", spec=SPEC: {"preamble": pre, "spec": spec, "postamble": post}  # noqa: E731
    assert C.dafny_frozen_axioms(fz()) == []  # the bodiless spec is the agent's hole, not an axiom
    assert C.dafny_frozen_axioms(fz(pre="function F(x: int): int\npredicate P(x: int) { x > 0 }")) == []
    assert C.dafny_frozen_axioms(fz(pre="lemma {:axiom} Bad()\n  ensures false")) == ["{:axiom} in the frozen preamble"]
    assert C.dafny_frozen_axioms(fz(pre="lemma Bad()\n  ensures false\n\nfunction G(): int { 0 }")) == [
        "bodiless lemma Bad with ensures in the frozen preamble"]
    assert C.dafny_frozen_axioms(fz(post="lemma L() ensures false { assume false; }")) == [
        "assume in the frozen postamble"]
    assert C.dafny_frozen_axioms(fz(pre="// we assume {:axiom} nothing\nconst s := \"assume\"")) == []
    assert "@Axiom in the frozen preamble" in C.dafny_frozen_axioms(fz(pre="@Axiom\nlemma A() ensures false"))


def test_bug8_find_dafny_pins_the_version(tmp_path, monkeypatch):
    def fake(name, version):
        p = tmp_path / name / "dafny"
        p.parent.mkdir()
        p.write_text(f"#!/bin/sh\necho '{version}'\n")
        p.chmod(0o755)
        return p
    good, old, path_one = fake("good", "4.11.0+fcb2042d"), fake("old", "4.10.0+abc"), fake("path", "4.9.1")
    monkeypatch.setenv("PATH", f"{path_one.parent}:/usr/bin:/bin")
    monkeypatch.delenv("SORRYLESS_DAFNY", raising=False)  # the pre-rename name is only a fallback
    monkeypatch.setenv("MILLENNIUM_DAFNY", str(old))
    with pytest.raises(C.InfraError, match="is not Dafny 4.11.0"):
        C.find_dafny()
    monkeypatch.setenv("MILLENNIUM_DAFNY", str(good))
    assert C.find_dafny() == str(good)
    monkeypatch.delenv("MILLENNIUM_DAFNY")
    monkeypatch.setenv("SORRYLESS_DAFNY", str(old))  # the pre-rename name is still read
    with pytest.raises(C.InfraError, match="is not Dafny 4.11.0"):
        C.find_dafny()
    monkeypatch.delenv("SORRYLESS_DAFNY")
    monkeypatch.setattr(C, "LOCAL_DAFNY", good)  # the pinned local install wins over a PATH dafny
    assert C.find_dafny() == str(good)
    monkeypatch.setattr(C, "LOCAL_DAFNY", tmp_path / "none")
    with pytest.raises(C.InfraError, match="is not Dafny 4.11.0"):  # PATH's 4.9.1 is refused, not used
        C.find_dafny()


def test_bug3_dafny_infra_failures_are_errors_not_statuses(monkeypatch):
    monkeypatch.setenv("MILLENNIUM_DAFNY", "dafny")  # curate_dafny writes it; restored after the test
    monkeypatch.setattr(C, "_dafny_ready", lambda env: "dafny")
    rec = {"id": "D", "language": "dafny", "vc-preamble": "", "vc-spec": SPEC, "vc-postamble": ""}
    for rc, out in [(None, "dafny resolve timed out after 300s"), (134, "Unhandled exception. Out of memory.")]:
        monkeypatch.setattr(sl_dafny, "_dafny", lambda args, cwd, timeout, rc=rc, out=out: (rc, out))
        r = C.curate(rec)
        assert "status" not in r and "InfraError" in r["error"], r
    monkeypatch.setattr(sl_dafny, "_dafny", lambda args, cwd, timeout: (2, "frozen.dfy(2,12): Error: unresolved Foo"))
    assert C.curate(rec)["status"] == "BROKEN"
    monkeypatch.setattr(sl_dafny, "_dafny", lambda args, cwd, timeout: (0, "resolved"))
    for r_check in [{"status": "UNPROVED", "violations": [], "verify": "dafny verify timed out after 300s"},
                    {"status": "CHECK_ERROR", "violations": ["task error"]},
                    {"status": "CHEAT_REJECTED", "verify_ok": True, "violations": ["frozen method `M` was not verified"]}]:
        monkeypatch.setattr(sl_dafny, "check", lambda *a, r_check=r_check, **k: r_check)
        r = C.curate(rec)
        assert "status" not in r and "InfraError" in r["error"], (r_check, r)
    monkeypatch.setattr(sl_dafny, "check", lambda *a, **k: {"status": "UNPROVED", "violations": [], "verify": "1 error"})
    assert "without an Error" in C.curate(rec)["error"]  # no Error line: undecided, not OK
    # DT0056's shape: one batch timed out, another postcondition is refuted -> the default body is refuted
    verify = ("solution.dfy(10,7): Error: Verification of 'hsplit' timed out after 265 seconds.\n"
              "solution.dfy(26,0): Error: a postcondition could not be proved on this return path\n"
              "Dafny program verifier finished with 0 verified, 1 error, 1 time out")
    monkeypatch.setattr(sl_dafny, "check", lambda *a, **k: {"status": "UNPROVED", "violations": [], "verify": verify})
    assert C.curate(rec) == {"id": "D", "status": "OK", "reason": "default body UNPROVED"}


def _split():
    return {"seed": 1, "dev": {"L-fv": [{"comp": "LF0002", "ids": ["LF0002"]}],
                               "D-np": [{"comp": "DT0001", "ids": ["DT0001", "DT0007"]}]},
            "test": {"L-fv": [{"comp": "LF0001", "ids": ["LF0001"]}, {"comp": "LF0003", "ids": ["LF0003"]},
                              {"comp": "C9", "ids": ["LF0009"]}],
                     "D-np": [{"comp": "DT0002", "ids": ["DT0002"]}, {"comp": "C9", "ids": ["DT0009", "DT0010"]}]}}


def test_select():
    s = _split()
    assert C.select(s, "dev") == ["LF0002", "DT0001", "DT0007"]
    assert C.select(s, "test", "L-fv=1") == ["LF0001"]
    assert C.select(s, "all", "D-np=1,L-fv") == ["DT0001", "LF0002", "DT0002", "LF0001", "LF0003", "LF0009"]


def _files(tmp_path):
    pool, split, out, fdir = tmp_path / "pool.json", tmp_path / "split.json", tmp_path / "curated.json", tmp_path / "fz"
    pool.write_text("{}")
    split.write_text(json.dumps(_split()))
    return out, fdir, pool, split


def test_save_merges_counts_and_hides_test_detail(tmp_path):
    out, fdir, pool, split = _files(tmp_path)
    fz = {"id": "x", "chunks": []}
    res = [{"id": "LF0002", "status": "OK", "reason": "a theorem has no trivial proof", "detail": "dev text",
            "frozen": fz},
           {"id": "LF0001", "status": "BROKEN", "reason": "skeleton+postamble: error other than a sorry abort",
            "detail": "TEST TEXT", "frozen": fz},
           {"id": "LF0003", "status": "TRIVIAL_THEOREMS", "reason": "no theorem to prove", "detail": "TEST2",
            "frozen": fz},
           {"id": "DT0002", "status": "VACUOUS", "reason": "default body PROVED", "detail": "{ }"}]
    doc = C.save(res, out, fdir, pool, split)
    assert "TEST TEXT" not in out.read_text() and "TEST2" not in out.read_text()
    assert "detail" not in doc["tasks"]["LF0001"] and doc["tasks"]["LF0002"]["detail"] == "dev text"
    assert doc["tasks"]["LF0001"]["part"] == "test" and doc["curator"]["lean"]["stamp"] == doc["tasks"]["LF0001"]["stamp"]
    assert sorted(p.name for p in fdir.iterdir()) == ["LF0001.json", "LF0002.json", "LF0003.json"]
    t = doc["counts"]["test"]
    assert (t["L-fv"]["pool"], t["L-fv"]["curated"], t["L-fv"]["BROKEN"], t["L-fv"]["TRIVIAL_THEOREMS"]) == (3, 2, 1, 1)
    assert t["D-np"]["VACUOUS"] == 1 and t["D-np"]["ok_components"] == 0 and doc["counts"]["dev"]["L-fv"]["OK"] == 1
    assert not t["L-fv"]["complete"] and doc["counts"]["dev"]["L-fv"]["complete"]
    doc = C.save([{"id": "DT0007", "status": "OK", "reason": "default body UNPROVED"}], out, fdir, pool, split)
    assert len(doc["tasks"]) == 5 and doc["counts"]["dev"]["D-np"]["ok_components"] == 1  # rerun merges by id


def test_bug5_save_never_mixes_curator_stamps_and_removes_stale_locks(tmp_path):
    out, fdir, pool, split = _files(tmp_path)
    new = C.curator()
    old = {lang: {**c, "stamp": "old"} for lang, c in new.items()}
    C.save([{"id": "LF0002", "status": "OK", "reason": "r", "frozen": {"id": "LF0002"}},
            {"id": "LF0001", "status": "OK", "reason": "r", "frozen": {"id": "LF0001"}}], out, fdir, pool, split, old)
    doc = C.save([{"id": "LF0003", "status": "OK", "reason": "r", "frozen": {"id": "LF0003"}}], out, fdir, pool, split,
                 new)
    assert list(doc["tasks"]) == ["LF0003"] and doc["curator"] == new  # old-battery labels are gone, not relabelled
    assert sorted(p.name for p in fdir.iterdir()) == ["LF0003.json"]
    doc = C.save([{"id": "LF0003", "status": "LOCK_FAIL", "reason": "skeleton does not compile"}], out, fdir, pool,
                 split, new)
    assert doc["tasks"]["LF0003"]["status"] == "LOCK_FAIL" and not (fdir / "LF0003.json").exists()
    assert not list(tmp_path.glob("*.tmp"))  # atomic replace leaves no temp file


def test_bug5_stamps_are_per_language(tmp_path):
    out, fdir, pool, split = _files(tmp_path)
    cur = C.curator()
    assert cur["lean"]["stamp"] != cur["dafny"]["stamp"]
    assert set(cur["lean"]["sources_sha256"]) == {"bench/fv/curate.py", "millennium_harness/lean.py",
                                                  "millennium_harness/lean_check/Check.lean"}
    C.save([{"id": "LF0002", "status": "OK", "reason": "r", "frozen": {"id": "LF0002"}},
            {"id": "DT0001", "status": "OK", "reason": "r"}], out, fdir, pool, split, cur)
    edited = {**cur, "lean": {**cur["lean"], "stamp": "lean.py edited"}}
    # a Dafny-only save keeps the Lean entries and the header they were made under
    doc = C.save([{"id": "DT0007", "status": "OK", "reason": "r"}], out, fdir, pool, split, edited)
    assert sorted(doc["tasks"]) == ["DT0001", "DT0007", "LF0002"] and doc["curator"] == cur
    # a Lean save under the edited rung drops the old Lean entries and their locks, never relabels them
    doc = C.save([{"id": "LF0001", "status": "OK", "reason": "r", "frozen": {"id": "LF0001"}}], out, fdir, pool, split,
                 edited)
    assert sorted(doc["tasks"]) == ["DT0001", "DT0007", "LF0001"] and doc["curator"] == edited
    assert sorted(p.name for p in fdir.iterdir()) == ["LF0001.json"]
    assert doc["tasks"]["LF0001"]["stamp"] == "lean.py edited"


def test_bug5_concurrent_saves_merge(tmp_path):
    import multiprocessing
    out, fdir, pool, split = _files(tmp_path)
    ids = ["LF0001", "LF0003", "LF0009", "DT0002", "DT0009", "DT0010"]
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_save_one, args=(i, out, fdir, pool, split)) for i in ids]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert sorted(json.loads(out.read_text())["tasks"]) == sorted(ids)


def _save_one(i, out, fdir, pool, split):
    C.save([{"id": i, "status": "OK", "reason": "r"}], out, fdir, pool, split)


def test_bug4_ok_ids_counts_and_curated_split_follow_the_draw(tmp_path):
    out, fdir, pool, split_path = _files(tmp_path)
    split = _split()
    st = {"LF0002": "OK", "DT0001": "BROKEN", "DT0007": "OK", "LF0001": "OK", "LF0003": "VACUOUS", "LF0009": "OK",
          "DT0002": "TRIVIAL_THEOREMS", "DT0009": "LOCK_FAIL", "DT0010": "OK"}
    doc = C.save([{"id": i, "status": s, "reason": "r"} for i, s in st.items()], out, fdir, pool, split_path)
    assert doc["ok_ids"] == {"dev": {"L-fv": ["LF0002"], "D-np": ["DT0007"]},
                             "test": {"L-fv": ["LF0001", "LF0009"], "D-np": ["DT0010"]}}
    t = doc["counts"]["test"]
    # component C9 is usable in both TEST cells; the draw gives it to one, so it is flagged in each
    assert (t["L-fv"]["ok_components"], t["L-fv"]["ok_components_multicell"]) == (2, 1)
    assert (t["D-np"]["ok_components"], t["D-np"]["ok_components_multicell"]) == (1, 1)
    cs = C.curated_split(split, doc)
    assert cs["dev"]["D-np"] == [{"comp": "DT0001", "ids": ["DT0007"]}]  # DEV's ids[0] is now the usable member
    assert cs["test"]["L-fv"] == [{"comp": "LF0001", "ids": ["LF0001"]}, {"comp": "C9", "ids": ["LF0009"]}]
    assert cs["test"]["D-np"] == [{"comp": "C9", "ids": ["DT0010"]}] and cs["seed"] == 1
    assert cs["dev_ids"] == {"L-fv": ["LF0002"], "D-np": ["DT0007"]}
    from bench.fv.split import pick_test
    drawn = pick_test(cs, ["L-fv", "D-np"])
    assert all(st[i] == "OK" for ids in drawn.values() for i in ids)
    new = {lang: {**c, "stamp": "new"} for lang, c in C.curator().items()}  # a Lean result under another stamp
    doc = C.save([{"id": "LF0002", "status": "OK", "reason": "r"}], out, fdir, pool, split_path, new)
    with pytest.raises(ValueError, match="curated"):
        C.curated_split(split, doc)


def test_bug6_modal_image_installs_the_toolchain_like_the_dockerfile():
    cmds = "\n".join(C.IMAGE_CMDS)
    assert "--default-toolchain none" in cmds and f"toolchain install leanprover/lean4:v{sl_lean.TOOLCHAIN}" in cmds
    assert f"elan default leanprover/lean4:v{sl_lean.TOOLCHAIN}" in cmds and f"version {sl_lean.TOOLCHAIN}," in cmds
    assert 'test -x "$L"' in cmds and f"dafny --version | grep -F '{C.DAFNY_VERSION}'" in cmds
    docker = (Path(C.ROOT) / "bench" / "fv" / "images" / "lean.Dockerfile").read_text()
    assert "--default-toolchain none" in docker and "elan toolchain install" in docker


# ------------------------------------------------------------------------------------------------ Lean controls
TT = "TRIVIAL_THEOREMS"
ONE, MANY, VAC = "under one variant", "under different variants", "(VACUOUS)"
LEAN_CASES = {  # name -> (record, status, reason substring)
    "vacuous default (omega)": (lean_rec(thms="theorem t (n : Nat) : f n ≥ 0 := by\n  sorry", post=guard("f 3", 0)),
                                TT, VAC),
    "vacuous passthrough": (lean_rec(thms="theorem t (n : Nat) : f n = n := sorry", post=guard("f 3", 3)), TT, VAC),
    "vacuous pure default": (lean_rec(defs="def h (n : Nat) : Option Nat := sorry",
                                      thms="theorem t (n : Nat) : (h n).isSome := sorry"), TT, VAC),
    "vacuous via mvcgen [f]": (lean_rec(defs="def f (n : Nat) : Id Nat :=\n  sorry", pre=STD_DO,
                                        thms="theorem t (n : Nat) :\n    ⦃⌜True⌝⦄ f n ⦃⇓r => ⌜r ≥ 0⌝⦄ := by\n  sorry"),
                               TT, VAC),
    # bug 1: LF0382's shape. The theorem holds for any f; only the postamble constrains f -> excluded.
    "bug1 LF0382: trivial theorem, postamble rejects default": (
        lean_rec(thms="theorem t (n : Nat) : f n ≥ 0 := by\n  sorry", post=guard("f 3", 5)), TT, ONE),
    # bug 1: per theorem across variants (each lower bound is a passthrough; no single trivial body proves both)
    "bug1 per theorem across variants": (
        lean_rec(defs="def f (a b : Nat) : Nat :=\n  sorry",
                 thms="theorem t1 (a b : Nat) : f a b ≥ a := sorry\n\ntheorem t2 (a b : Nat) : f a b ≥ b := sorry",
                 post=guard("f 3 5", 5)), TT, MANY),
    # bug 2: `rfl` throws maxRecDepth (a runtime exception `first` does not catch); omega alone proves it
    "bug2 rfl recursion does not mask omega": (
        lean_rec(thms="theorem t (n : Nat) : f n + (List.replicate 100000 1).length ≥ 0 := sorry",
                 post=guard("f 3", 5)), TT, ONE),
    "bug2 passthrough of the second explicit argument": (
        lean_rec(defs="def f (a b : Nat) : Nat :=\n  sorry", thms="theorem t (a b : Nat) : f a b = b := sorry",
                 post=guard("f 1 2", 2)), TT, VAC),
    "bug2 two implementation defs need different bodies": (
        lean_rec(defs="def f (n : Nat) : Nat :=\n  sorry\n\ndef g (n : Nat) : Nat :=\n  sorry",
                 thms="theorem t (n : Nat) : f n = n ∧ g n = 0 := sorry"), TT, VAC),
    "no theorem hole": (lean_rec(post=guard("f 3", 3)), TT, "no theorem to prove"),
    # hole names Lean prints differently from the source, or that the lock does not record (LF0364, LT0063)
    "private implementation and theorem holes": (
        lean_rec(thms="private def g (n : Nat) : Nat :=\n  sorry\n\nprivate theorem pt (n : Nat) : g n ≥ 0 := sorry",
                 post=guard("f 3", 4)), TT, ONE),
    "escaped name «repeat»": (lean_rec(defs="def «repeat» (n : Nat) : Nat :=\n  sorry",
                                      thms="theorem t (n : Nat) : «repeat» n ≥ 0 := sorry", post=guard("«repeat» 3", 4)),
                             TT, ONE),
    "theorem in a namespace closed before the end": (
        lean_rec(thms="namespace Foo\n\ntheorem t (n : Nat) : f n ≥ 0 := sorry\n\nend Foo", post=guard("f 3", 4)),
        TT, ONE),
    "example hole": (lean_rec(thms="example (n : Nat) : f n ≥ 0 := sorry", post=guard("f 3", 4)), TT, ONE),
    "ok: real spec, unused hypothesis": (lean_rec(defs="def f (n : Nat) (h : n > 0) : Nat :=\n  sorry",
                                               thms="theorem t (n : Nat) (h : n > 0) : f n h = n * n + 1 := sorry"),
                                      "OK", "no trivial proof"),
    "ok: one real theorem among trivial ones": (
        lean_rec(thms="theorem t1 (n : Nat) : f n ≥ 0 := sorry\n\ntheorem t2 (n : Nat) : f n = n * n + 1 := sorry",
                 post=guard("f 3", 10)), "OK", "no trivial proof"),
    "ok (conservative): no trivial body typechecks": (
        lean_rec(pre="structure S where\n  x : Nat", defs="def f (n : Nat) : S :=\n  sorry",
                 thms="theorem t (n : Nat) : (f n).x ≥ 0 := sorry"), "OK", "no trivial proof"),
    "broken: python postamble": (lean_rec(thms="theorem t (n : Nat) : f n ≥ 0 := sorry", post="#eval len f(3)"),
                                 "BROKEN", "skeleton+postamble"),
    "broken: unknown name in a test": (lean_rec(thms="", post=guard("g 3", 0)), "BROKEN", "skeleton+postamble"),
    "broken: test independent of f": (lean_rec(thms="", post=guard("1 + 1", 3)), "BROKEN", "skeleton+postamble"),
    "lock fail: ill-typed statement": (lean_rec(thms='theorem t (n : Nat) : f n ≥ "x" := sorry'),
                                       "LOCK_FAIL", "skeleton does not compile"),
    "lock fail: two sorries": (lean_rec(defs="def f (n : Nat) : Nat :=\n  sorry + sorry"),
                               "LOCK_FAIL", "sorry outside a declaration body"),
}


@needs_lean
@pytest.mark.parametrize("name", list(LEAN_CASES))
def test_lean_controls(name):
    rec, status, why = LEAN_CASES[name]
    r = C.curate(rec)
    assert (r.get("status"), r.get("status") in C.STATUSES) == (status, True), r
    assert why in r["reason"], r
    assert ("frozen" in r) == (status != "LOCK_FAIL")
    if status in ("OK", TT):
        assert r["frozen"] == sl_lean.lock(rec, LEAN)
    if VAC not in why and status == TT:
        assert "VACUOUS" not in r["reason"]
    assert ("no trivial body typechecks" in r.get("detail", "")) == ("conservative" in name)


@needs_lean
def test_bug2_old_first_battery_missed_the_recursion_case():
    """The pre-fix `first | (rfl; done) | (omega; done) ...` really is aborted by rfl's maxRecDepth: keep the repro."""
    rec = LEAN_CASES["bug2 rfl recursion does not mask omega"][0]
    frozen = sl_lean.lock(rec, LEAN)
    first = "first " + " ".join(f"| ({t}; done)" for t in C.tactics(rec, ["f"]))
    ax = C.lean_axioms(C.fill(rec, "exact default", first, postamble=False), ["t"], LEAN, 120)
    assert not C._clean(ax, "t")
    ax = C.lean_axioms(C.fill(rec, "exact default", "(omega; done)", postamble=False), ["t"], LEAN, 120)
    assert C._clean(ax, "t") and frozen["constants"]["t"]["role"] == "theorem"


@needs_lean
def test_bug3_lock_timeout_is_an_error_not_lock_fail(monkeypatch):
    monkeypatch.setattr(C, "LEAN_TIMEOUT", 0.001)
    r = C.curate(LEAN_CASES["ok: real spec, unused hypothesis"][0])
    assert "status" not in r and "InfraError" in r["error"] and "timed out" in r["error"], r


@needs_lean
def test_bug3_check_lean_crash_is_an_error_not_lock_fail(monkeypatch):
    def crash(*a, **k):
        raise RuntimeError("Check.lean: out of memory")
    monkeypatch.setattr(sl_lean, "_check_lean", crash)
    r = C.curate(LEAN_CASES["ok: real spec, unused hypothesis"][0])
    assert "status" not in r and "out of memory" in r["error"], r


@needs_lean
def test_bug3_postamble_timeout_is_an_error_not_broken():
    with pytest.raises(C.InfraError, match="timed out"):
        C.postamble_problems(LEAN_CASES["vacuous default (omega)"][0], LEAN, timeout=0.001)


@needs_lean
def test_bug3_battery_timeout_is_an_error_unless_another_tactic_wins(monkeypatch):
    real = C.lean_axioms

    def slow(tactic):
        def la(text, names, lean, timeout):
            return None if f"({tactic}; done)" in text else real(text, names, lean, timeout)
        return la
    monkeypatch.setattr(C, "lean_axioms", slow("decide"))  # the OK task: undecided -> uncurated, never OK
    r = C.curate(LEAN_CASES["ok: real spec, unused hypothesis"][0])
    assert "status" not in r and "timed out" in r["error"], r
    monkeypatch.setattr(C, "lean_axioms", slow("rfl"))  # the trivial task: omega still wins, so the status is sure
    assert C.curate(LEAN_CASES["bug1 LF0382: trivial theorem, postamble rejects default"][0])["status"] == TT


@needs_lean
def test_saved_lock_is_what_the_adapter_reads(tmp_path):
    from bench.adapters import vericoding_fv as A
    rec = lean_rec(thms="theorem t (n : Nat) : f n = n * n := sorry", rid="LF9999")
    r = C.curate(rec)
    assert r["status"] == "OK", r
    (tmp_path / "LF9999.json").write_text(json.dumps(r["frozen"], indent=1, ensure_ascii=False))
    sol, frozen = A.lean_files(rec, tmp_path)
    assert frozen == r["frozen"] and sl_lean.holes(sol, frozen)[1] == []


# ------------------------------------------------------------------------------------------------ Dafny controls
DAFNY_CASES = {
    "vacuous default": ("", "method M(x: int) returns (r: int, s: seq<int>)\n  ensures r >= 0 && |s| <= 3", "VACUOUS"),
    "vacuous havoc of a datatype": ("datatype D = A | B", "method M() returns (d: D)\n  ensures d == A || d == B",
                                    "VACUOUS"),
    "vacuous function": ("", "function F(x: int): (r: int)\n  ensures r >= 0", "VACUOUS"),
    "ok": ("", SPEC, "OK"),
    "broken: undefined name": ("", "method M(x: int) returns (r: int)\n  ensures r == Foo(x)", "BROKEN"),
    "broken: prose": ("Looking at the error, here is the program:", "method M() returns (r: int)", "BROKEN"),
    "bug7 lock fail: frozen axiom lemma": ("lemma {:axiom} Bad()\n  ensures false", SPEC, "LOCK_FAIL"),
    "bug7 lock fail: frozen bodiless lemma": ("lemma Bad()\n  ensures false\n\nfunction G(): int { 0 }", SPEC,
                                              "LOCK_FAIL"),
}


@needs_dafny
@pytest.mark.parametrize("name", list(DAFNY_CASES))
def test_dafny_controls(name, monkeypatch):
    monkeypatch.setenv("MILLENNIUM_DAFNY", DAFNY)
    pre, spec, status = DAFNY_CASES[name]
    r = C.curate({"id": "D", "language": "dafny", "vc-preamble": pre, "vc-spec": spec, "vc-postamble": ""})
    assert r.get("status") == status and "frozen" not in r, r


@needs_dafny
def test_bug3_dafny_resolve_timeout_is_an_error(monkeypatch):
    monkeypatch.setenv("MILLENNIUM_DAFNY", DAFNY)
    monkeypatch.setattr(C, "DAFNY_TIMEOUT", 0.05)
    r = C.curate({"id": "D", "language": "dafny", "vc-preamble": "", "vc-spec": SPEC, "vc-postamble": ""})
    assert "status" not in r and "timed out" in r["error"], r


# ------------------------------------------------------------------------------------------------ curated.json
@pytest.mark.skipif(not (C.OUT.exists() and C.SPLIT.exists()), reason="run bench.fv.curate first")
def test_curated_json_invariants():
    doc, split = json.loads(C.OUT.read_text()), json.loads(C.SPLIT.read_text())
    where = {i: (p, cell) for p in ("dev", "test") for cell, comps in split[p].items() for c in comps for i in c["ids"]}
    for i, t in doc["tasks"].items():
        assert t["status"] in C.STATUSES and (t["part"], t["cell"]) == where[i]
        assert t["stamp"] == doc["curator"][C._lang(t["cell"])]["stamp"]  # one battery/rung version per language
        assert t["part"] == "dev" or "detail" not in t  # TEST: ids and content-free reasons only
        frozen = C.FROZEN_DIR / f"{i}.json"
        if t["cell"].startswith("L-") and t["status"] != "LOCK_FAIL":
            assert json.loads(frozen.read_text())["id"] == i
        else:
            assert not frozen.exists()
    assert doc["counts"] == C.counts(doc["tasks"], split) and doc["ok_ids"] == C.ok_ids(doc["tasks"], split)
