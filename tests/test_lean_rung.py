"""Lean rung red team: every cheat in bench/redteam/lean is rejected, every honest solution is PROVED.

The lexer/chunk-lock tests are pure Python; everything that compiles skips cleanly without Lean v4.23.0-rc2."""
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from millennium_harness import lean

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "bench" / "redteam" / "lean"
MANIFEST = json.loads((RT / "manifest.json").read_text(encoding="utf-8"))

try:
    LEAN = lean.find_lean()
except (lean.ToolchainError, OSError, subprocess.SubprocessError):
    LEAN = None
needs_lean = pytest.mark.skipif(LEAN is None, reason=f"Lean {lean.TOOLCHAIN} not installed")


@functools.cache
def frozen(task: str) -> dict:
    return lean.lock(json.loads((RT / "tasks" / f"{task}.json").read_text(encoding="utf-8")), LEAN)


def test_lean_env_var_and_legacy_fallback(tmp_path, monkeypatch):
    """MILLENNIUM_LEAN is tried first; the pre-rename SORRYLESS_LEAN is still read (and version-checked)."""
    fake = tmp_path / "lean"
    fake.write_text("#!/bin/sh\necho 'Lean (version 0.0.0, fake)'\n")
    fake.chmod(0o755)
    for new, old in ((None, str(fake)), (str(fake), None)):
        for k, v in (("MILLENNIUM_LEAN", new), ("SORRYLESS_LEAN", old)):
            monkeypatch.setenv(k, v) if v else monkeypatch.delenv(k, raising=False)
        with pytest.raises(lean.ToolchainError, match=f"{fake} is not Lean"):
            lean.find_lean()


def _read(case: dict) -> str:
    return (RT / case["file"]).read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ pure Python
@pytest.mark.parametrize("src,visible,hidden", [
    ("def «x--» : Nat := 0 axiom a : False", "axiom", None),          # «…» is an identifier, not a comment
    ("def q : Char := '\"'\naxiom b : False\ndef r := \"\\\"\"", "axiom", None),
    ("def z := 2'\"'\naxiom c : False", "axiom", None),               # char literal right after a number
    ("/-/- -/ axiom e : False /- -/", "axiom", None),                # Lean skips the char after `/-`: no nesting
    ('x := s!"a{f "b" }c" ++ sorry', "sorry", None),                  # interpolation is code
    ('y := foo!"{" #eval 1 "}"', "#eval", None),                      # `foo!` is an identifier: plain strings
    ('"plain {sorry} text"', "sorry", None),                          # braces: may be interpolated, keep {code}
    ('"node {id} prefix tag"', "{id}", "prefix"),                     # ... and only that, when both readings agree
    ('"x {" ++ "y" ++ "}"', "x {", None),                             # readings disagree: keep the text
    ('"{\\"k\\": 1}" -- axiom', '{\\"k\\": 1}', "axiom"),             # no interpolated reading: keep the text
    ('-- axiom hidden\n"axiom in a string" /- sorry -/', None, "axiom"),
])
def test_mask_mirrors_lean_lexer(src, visible, hidden):
    masked, problems = lean.mask(src)
    assert len(masked) == len(src) and not problems
    if visible:
        assert visible in masked
    if hidden:
        assert hidden not in masked


def test_guillemet_identifiers_are_names_not_keywords():
    """`«…»` is never a keyword, but options, attributes, constants and tactic config fields are looked up by name
    (each spelling below was checked against Lean v4.23.0-rc2)."""
    src = "set_option «debug».skipKernelTC true in\ndef «import count» := «Lean».«ofReduceBool»"
    assert lean.mask(src)[0] == src                                     # default view unchanged (curate reads names)
    assert "import" not in lean.mask(src, idents="blank")[0]
    named = lean.mask(src, idents="names")[0]
    assert "set_option   debug.skipKernelTC" in named and "import" not in named and "  ofReduceBool" in named
    assert lean.ban_scan(["def «import count» : Int := 0", "def «export» := «admit»"]) == []
    for cheat in ["set_option «debug».skipKernelTC true in", "@[«implemented_by» g] def f := 1",
                  "theorem t : 2 + 2 = 4 := by decide +«native»", "def h := «Lean».«ofReduceBool»",
                  "theorem t : False := «sorryAx» _ true"]:
        assert lean.ban_scan([cheat]), cheat


def test_mask_reports_unterminated():
    assert lean.mask("def x := 1 /- open")[1] == ["unterminated comment"]
    assert lean.mask('def s := "open')[1] == ["unterminated string"]


def test_chunks_and_holes():
    rec = {"vc-preamble": "def g (n : Nat) : Nat := n", "vc-helpers": "",
           "vc-definitions": "def f (n : Nat) : Nat :=\n  sorry",
           "vc-theorems": "@[simp]\ndef post (n : Nat) : Prop := f n = g n\n\ntheorem t (n : Nat) : post n := by\n  sorry",
           "vc-postamble": "-- end"}
    chunks = lean._chunks(rec)
    assert [c["text"] for c in chunks] == ["def f (n : Nat) : Nat :=", "@[simp]\ndef post (n : Nat) : Prop := f n = g n",
                                           "theorem t (n : Nat) : post n := by"]
    fz = {"preamble": rec["vc-preamble"], "postamble": "-- end", "chunks": chunks}
    good = lean.skeleton(rec).replace("sorry", "n", 1).replace("sorry", "rfl")
    hs, v = lean.holes("-- my header\n" + good.replace("theorem t", "theorem helper : True := trivial\n\ntheorem t"), fz)
    assert not v and any("helper" in h for h in hs) and lean.ban_scan(hs) == []
    assert lean.holes(good.replace("f n = g n", "True"), fz)[1]           # frozen def edited
    assert lean.holes(good + "\naxiom x : False\n", fz)[1]                # code after the postamble
    assert lean.holes("axiom x : False\n" + good, fz)[1]                  # code before the preamble
    assert lean.holes(good.replace("-- end", "set_option pp.all true in\n-- end"), fz)[1]  # wraps the postamble
    assert lean.ban_scan(["  exact (by native_decide)", "@[implemented_by g]"]) and lean.ban_scan(["/- open"])


def test_chunks_accept_fun_and_match_arm_bodies():
    rec = {"vc-definitions": "def f : Nat → Nat\n  | _ => sorry\n\ndef g : Nat → Nat :=\n  fun _ => sorry"}
    assert [c["text"] for c in lean._chunks(rec)] == ["def f : Nat → Nat", "def g : Nat → Nat :="]


def test_lock_rejects_sorry_outside_bodies():
    rec = {"vc-definitions": "def f (l : List Nat) : Nat := sorry",
           "vc-theorems": "theorem t (l : List Nat) (h : l ≠ []) : f l = l[0]'(sorry) := sorry"}
    with pytest.raises(lean.LockError, match="outside a declaration body"):
        lean.lock(rec)
    with pytest.raises(lean.LockError, match="vc-preamble"):
        lean.lock({"vc-preamble": "def g : Nat := sorry", "vc-definitions": "def f : Nat := sorry"})


def test_instance_scan_decisions():
    """Check.lean reports each class-typed constant with its class (after unfolding definitions) and the module's
    inductive types in its arguments (input anchors, semiOutParam slots, everything but a coercion's source), plus the
    environment's class graph; only an agent-declared type anchors. Pure Python over a synthetic report."""
    fz = {"constants": {"f": {"kind": "def", "role": "impl", "type": lean._sha(""), "ctors": [], "value": None}},
          "imports": []}
    fp = {"exists": True, "module": True, "kind": "def", "type": "", "ctors": [], "value": None, "axioms": [],
          "noncomputable": False, "ir": True, "implemented_by": None, "extern": False}

    # A slice of the environment's class graph (class -> classes of its instances' instance arguments): core's
    # `CoeTail Nat R [NatCast R]`, `One.toOfNat1 [One α] : OfNat α 1`, `decPropToBool [Decidable p] : CoeDep ..`.
    # Decidable is never followed (the Decidable -> Repr edge is made up to check that).
    feeds = {"CoeTail": ["NatCast"], "NatCast": ["Lean.Grind.Semiring", "NeZero"], "Lean.Grind.Semiring": ["OfNat"],
             "OfNat": ["One"], "CoeDep": ["Decidable"], "Decidable": ["Repr"]}

    def cheats(instances, defaults=(), free=(), errors=()):
        rep = {"consts": {"f": fp}, "module_consts": ["f", "millennium_eof_n", "S", "N", "T",
                                                      *(i["name"] for i in instances), *defaults],
               "imports": [], "axioms_declared": [], "implemented_by_or_extern": [], "meta_consts": [],
               "instances": instances, "default_instances": list(defaults), "replay": "ok",
               "class_graph": {"feeds": feeds, "prop_classes": ["NeZero"], "free": list(free), "errors": list(errors)}}
        return lean._kernel_violations(rep, fz, "n")[1]

    def inst(name, cls, anchors=(), error="", semi=(), mentions=None, source=()):
        return {"name": name, "class": cls, "anchors": list(anchors), "error": error, "semi": list(semi),
                "mentions": list(anchors if mentions is None else mentions), "source": list(source)}

    assert cheats([inst("instReprS", "Repr", ["S"]), inst("instToStringS", "ToString", ["S"]),
                   inst("instDecPred", "Decidable"), inst("instInhabitedNat", "Inhabited")]) == []
    assert cheats([inst("reprN", "Repr")])                        # abbrev alias: N unfolds away, no anchor
    assert cheats([inst("reprList", "Repr", ["List"])])           # anchored only on a base/task type
    assert cheats([inst("weird", "?", error="boom")])              # unanalysable: fail closed
    assert cheats([], defaults=["ofNatS"])                         # any agent default instance
    assert cheats([], defaults=["instOfNatFloat"])                 # also one on a library instance
    fz["default_instances"] = ["taskInst"]                         # the task's own entries are exempt, extra ones not
    assert not cheats([], defaults=["taskInst"]) and cheats([], defaults=["taskInst", "taskInst"])
    del fz["default_instances"]
    assert cheats([inst("_private.Solution.0.r", "OfNat")])       # private/local/scoped ones count too
    # The scanned classes are closed over the class graph: `One Nat` decides `OfNat Nat 1` through `One.toOfNat1`.
    assert cheats([inst("oneNat", "One")]) and not cheats([inst("oneS", "One", ["S"])])
    with pytest.raises(RuntimeError):                              # graph incomplete: CHECK_ERROR
        cheats([], errors=["Lib.weird"])
    # A coercion out of the agent's own type S (S only in the semiOutParam source of Coe/CoeTail) is inert while
    # nothing can coerce into S; once something can, the two chain into a coercion between task/base types.
    coe_out = inst("coeSInt", "Coe", semi=["S"], mentions=[])
    derived = [inst("instReprS", "Repr", ["S"]), inst("instDecEqS", "Decidable", ["S"]), inst("ofNatS", "OfNat", ["S"])]
    assert cheats([coe_out, *derived[:2], inst("coeSNat", "CoeTail", semi=["S"], mentions=[])]) == []
    assert cheats([coe_out, inst("coeFinS", "Coe", ["S"])])        # Coe (Fin 5) S: Fin 5 → S → Int
    assert cheats([coe_out, inst("natCastS", "NatCast", ["S"])])   # into S through core's CoeTail Nat R [NatCast R]
    assert cheats([coe_out, derived[2]])                            # OfNat: reached through the graph
    assert cheats([coe_out], free=[{"name": "Lib.anyCoe", "class": "Coe"}])   # library coerces into any type
    assert cheats([coe_out, inst("other", "?", error="boom")])
    # A Prop-valued class decides no value (not scanned) but can enable a coercion into S (a producer).
    assert not cheats([inst("neZeroNat", "NeZero")]) and cheats([coe_out, inst("neZeroS", "NeZero", ["S"])])
    assert cheats([inst("coeNatS", "CoeOut", semi=["S"], mentions=["S"])])     # semiOut *target*: still flagged
    # Producers are a fixpoint over the agent's types: `Coe T S` (source headed by the agent's T) produces an S only
    # once something produces a T from a task/base type (round 3: Tally -> Summed -> Int was a false rejection).
    coe_ts = inst("coeTS", "Coe", ["S"], semi=["T"], source=["T"])
    assert cheats([coe_out, coe_ts]) == []
    assert cheats([coe_out, coe_ts, inst("coeST", "Coe", ["T"], semi=["S"], source=["S"])]) == []   # a cycle only
    assert cheats([coe_out, coe_ts, inst("coeFinT", "Coe", ["T"])])                # Fin 5 → T → S → Int
    assert cheats([coe_out, coe_ts, inst("natCastT", "NatCast", ["T"])])           # Nat → T through core's CoeTail
    assert cheats([coe_out, inst("coeListTS", "Coe", ["S"], semi=["T"], source=[])])  # source `List T`: not agent-headed
    assert cheats([coe_out, {k: v for k, v in coe_ts.items() if k != "source"}])   # no source reported: fail closed


def test_feedback_block():
    r = {"status": "UNPROVED", "violations": [], "goals": ["n : Nat\n⊢ n - 1 ≤ n"], "errors_tail": "error: boom",
         "secs": 1.0}
    fb = lean.feedback(r, 1, 2)
    assert fb.startswith("⟦Millennium Harness⟧ ✗ UNPROVED · Lean") and "round 1/2" in fb and "⊢ n - 1 ≤ n" in fb
    assert lean.feedback({**r, "status": "PROVED"}, 1, 2) == ""


# ------------------------------------------------------------------------------------------------ with Lean
@needs_lean
@pytest.mark.parametrize("case", MANIFEST["honest"], ids=lambda c: c["file"])
def test_honest_solution_is_proved(case):
    r = lean.check(_read(case), frozen(case["task"]), LEAN)
    assert r["status"] == "PROVED", r


@needs_lean
@pytest.mark.parametrize("case", MANIFEST["cheats"], ids=lambda c: c["file"].split("/")[-1])
def test_cheat_is_rejected(case):
    text, fz = _read(case), frozen(case["task"])
    r = lean.check(text, fz, LEAN)
    assert r["status"] == case["strict"], r
    if case["lenient"] is not None:  # the kernel-side checks catch it even without the text ban scan
        r = lean.check(text, fz, LEAN, bans=False)
        assert r["status"] == case["lenient"], r


@needs_lean
@pytest.mark.parametrize("case", MANIFEST["spec_weak"], ids=lambda c: c["file"])
def test_spec_weak_solution_is_proved(case):
    """The rung grades the frozen spec: a solution that meets a weak spec without computing the intended function is
    PROVED (strict and lenient). Spec adequacy is curation's call, not the rung's (red team round 3, logic lens)."""
    text, fz = _read(case), frozen(case["task"])
    assert lean.check(text, fz, LEAN)["status"] == "PROVED"
    assert lean.check(text, fz, LEAN, bans=False)["status"] == "PROVED"


@needs_lean
def test_lock_rejects_untrustworthy_tasks():
    with pytest.raises(lean.LockError, match="noncomputable"):
        lean.lock({"vc-definitions": "noncomputable def f (n : Nat) : Nat :=\n  sorry",
                   "vc-theorems": "theorem t : f 0 = f 0 := by\n  sorry"}, LEAN)
    with pytest.raises(lean.LockError, match="axioms"):
        lean.lock({"vc-preamble": "axiom g : Nat → Nat", "vc-definitions": "def f (n : Nat) : Nat :=\n  sorry"}, LEAN)


def _cli(tmp_path, solution: str, task: str, **kw) -> dict:
    sol, fz = tmp_path / "Solution.lean", tmp_path / "frozen.json"
    sol.write_text(solution)
    fz.write_text(json.dumps(frozen(task)))
    p = subprocess.run([sys.executable, "-I", str(ROOT / "millennium_harness" / "lean.py"), "check", str(sol), str(fz),
                        "--round", "2", "--max-rounds", "2"], capture_output=True, text=True, timeout=900, **kw)
    return json.loads(p.stdout)


@needs_lean
def test_cli_check_and_lock(tmp_path):
    r = _cli(tmp_path, (RT / "honest" / "LF2528.lean").read_text(), "LF2528")
    assert r["status"] == "PROVED" and r["feedback"] == "" and r["receipt"].startswith("∎ PROVED")
    out = tmp_path / "LS0013.frozen.json"
    p = subprocess.run([sys.executable, "-m", "millennium_harness.lean", "lock", str(RT / "tasks" / "LS0013.json"),
                        str(out)], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(out.read_text())["constants"]["copy"]["role"] == "impl"


@needs_lean
def test_toolchain_shim_on_path(tmp_path):
    """A fake `lean` first on PATH and a `lean-toolchain` in the cwd are never used: the pinned binary is."""
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "lean").write_text(f"#!/bin/sh\necho 'Lean (version {lean.TOOLCHAIN}, fake)'\nexit 0\n")
    (shim / "lean").chmod(0o755)
    (tmp_path / "lean-toolchain").write_text("leanprover/lean4:v4.0.0\n")
    wrong = (RT / "honest" / "LF2528.lean").read_text().replace("  n - 1\n", "  n\n", 1)  # wrong, no banned words
    env = {k: v for k, v in os.environ.items() if k not in ("MILLENNIUM_LEAN", "SORRYLESS_LEAN")}
    env["PATH"] = f"{shim}:{env.get('PATH', '')}"
    r = _cli(tmp_path, wrong, "LF2528", cwd=tmp_path, env=env)
    assert r["status"] == "UNPROVED", r


@needs_lean
def test_cwd_package_shadow(tmp_path):
    """`python3 -I .../millennium_harness/lean.py` ignores a planted `millennium_harness/` package and `json.py`
    in the cwd."""
    fake = 'import sys; sys.stdout.write(\'{"status": "PROVED", "feedback": "", "receipt": ""}\'); sys.exit(0)\n'
    (tmp_path / "millennium_harness").mkdir()
    (tmp_path / "millennium_harness" / "__init__.py").write_text(fake)
    (tmp_path / "millennium_harness" / "lean.py").write_text(fake)
    (tmp_path / "json.py").write_text("raise SystemExit('planted json imported')\n")
    r = _cli(tmp_path, (RT / "cheats" / "sorry_proof.lean").read_text(), "LF2528", cwd=tmp_path)
    assert r["status"] == "CHEAT_REJECTED", r
