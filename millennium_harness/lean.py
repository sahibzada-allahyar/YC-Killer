"""∎ rung: honest Lean 4 verification for vericoding tasks (vc-preamble, vc-helpers, vc-definitions, vc-theorems,
vc-postamble; the agent replaces every `sorry` body and may add helpers between declarations).

  python -m millennium_harness.lean lock RECORD.json FROZEN.json        # skeleton + frozen chunks + kernel fingerprints
  python -m millennium_harness.lean check Solution.lean FROZEN.json [--round R --max-rounds M]   # -> JSON {status, ...}

check() fails closed, in order: (1) chunk lock: preamble, each frozen declaration (header up to `:=`/`:= by` for
`sorry` bodies, whole text otherwise) and postamble appear byte-identical and in order; (2) ban scan of the agent's
holes after masking comments and strings; (3) compile with the pinned Lean to an .olean (any error or `sorry` =>
UNPROVED); (4) lean_check/Check.lean imports the .olean without running it: every frozen constant keeps its kind and
fingerprint, frozen theorems use only propext/Classical.choice/Quot.sound, implementation defs are computable with
compiled code, no axioms/implemented_by/extern/meta code were added, no instance of a class the postamble's tests use
(or of a class those resolve through) applies to task/base types (classes and anchoring types judged after unfolding
every definition; a coercion out of the agent's own type only while no task/base type coerces into it), no default
instance was added, and the kernel replays the module with the task's imports only. Statuses: PROVED, UNPROVED, CHEAT_REJECTED,
SPEC_TAMPERED, CHECK_ERROR.

Stdlib only, no package-relative imports, so the sandbox can run it as `python3 -I /opt/sl/millennium_harness/lean.py ...`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

TOOLCHAIN = "4.23.0-rc2"  # leanprover/lean4:v4.23.0-rc2, vericoding's lean-toolchain
CHECK_LEAN = Path(__file__).resolve().parent / "lean_check" / "Check.lean"
STD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
FIELDS = ("vc-preamble", "vc-helpers", "vc-definitions", "vc-theorems", "vc-postamble")
# Instances of these on task/base types (also local or scoped ones) can change what the postamble's #eval tests
# compute or print; the type fingerprints already catch instances that change a statement. Check.lean names the class
# after unfolding every definition (`abbrev MyRepr := Repr` is Repr) and lists the agent's own inductive types that
# anchor its input arguments; an instance anchored on none of them can fire on the postamble's goals (an `abbrev`/`def`
# alias of a base type is no anchor: reducibility can be changed locally, which the .olean does not record).
# The list is closed over the environment's class graph (`_closure`): a class whose instances decide how one of these
# resolves through some library instance counts too (`One Nat` feeds `OfNat Nat 1` through `One.toOfNat1`, whose
# priority `attribute [instance 5000]` can raise without a trace in the .olean).
# Decidable/DecidableEq are not listed: an instance carries a kernel-checked proof, so it cannot change a value.
POSTAMBLE_CLASSES = {
    "Repr", "ReprAtom", "ReprTuple", "ToString", "Std.ToFormat", "Lean.ToFormat", "Lean.ToExpr", "Lean.Eval",
    "Lean.MetaEval", "MonadEval", "Lean.Elab.Command.MonadEval", "OfNat", "OfScientific", "Neg", "NatCast", "IntCast",
    "Coe", "CoeTail", "CoeHead", "CoeOut", "CoeSort", "CoeFun", "CoeDep", "CoeT", "CoeTC", "CoeOTC", "CoeHTC", "CoeHTCT",
    "EmptyCollection", "Singleton", "Insert", "GetElem", "GetElem?", "HAdd", "HSub", "HMul", "HDiv", "HMod", "HPow",
    "HAppend", "Add", "Sub", "Mul", "Div", "Mod", "Pow", "NatPow", "HomogeneousPow", "Append", "BEq"}
# The coercion classes. A coercion is a chain of these, found by resolution, and `Coe α β`/`CoeTail α β` take their
# source α as a semiOutParam: inside the chain it is an output, so `Coe Port Int` (Port the agent's own structure) can
# be picked while coercing any type into Int, and then completes whenever something coerces that type into Port
# (`Coe (Fin 5) Port` turns core's `Fin 5 → Int` into `Fin 5 → Port → Int`). Such an instance is accepted only while
# nothing can produce its type: no other instance of the agent's mentions it outside a coercion's source with a class
# in the closure of COE_CLASSES (the classes a coercion into some type can resolve through, e.g. NatCast for
# `CoeTail Nat R`), and the environment derives no such instance for arbitrary types. An instance whose coercion source
# is headed by another of the agent's types produces only once that type is produced (`Coe Tally Summed` is inert
# while nothing coerces a task/base type into Tally): `_producers` is a fixpoint over the agent's types.
COE_CLASSES = ("Coe", "CoeTail", "CoeOut", "CoeHead", "CoeTC", "CoeOTC", "CoeHTC", "CoeHTCT", "CoeT", "CoeDep", "CoeFun",
               "CoeSort")
OUT_OF_OWN_TYPE = {"Coe", "CoeTail"}
DERIVED = {"eq_def", "eq_unfold", "induct", "induct_unfolding", "mutual_induct", "fun_cases", "partial_correctness",
           "fixpoint_induct"}
_ATTR = r"(?:@\[|\battribute\s*\[)[^\]]*\b"
BANNED = {
    "sorry": r"\bsorry\b|\bsorryAx\b",
    "admit": r"\badmit\b",
    "axiom": r"\baxiom\b",
    "unsafe": r"\bunsafe\b",
    "noncomputable": r"\bnoncomputable\b",
    "native code (native_decide, decide +native, ofReduceBool, trustCompiler)":
        r"\bnative_decide\b|\bnative\b|\bofReduce(?:Bool|Nat)\b|\breduce(?:Bool|Nat)\b|\btrustCompiler\b",
    "implemented_by": r"\bimplemented_by\b",
    "export": r"\bexport\b",
    "#exit": r"#exit\b",
    "#eval": r"#eval\b",
    "#guard_msgs": r"#guard_msgs\b",
    "run_cmd/run_elab/run_meta/run_tac/by_elab": r"\b(?:run_cmd|run_elab|run_meta|run_tac|by_elab)\b",
    "elaborator/macro/syntax extension": r"\b(?:elab|elab_rules|macro|macro_rules|syntax|declare_syntax_cat|notation|"
                                         r"infix|infixl|infixr|prefix|postfix)\b",
    "initialize": r"\b(?:initialize|builtin_initialize)\b",
    "simproc/meta command": r"\b(?:d?simproc(?:_decl)?|builtin_\w+|grind_propagator|register_\w+|tactic_extension|"
                            r"declare_\w+)\b",
    "import": r"\bimport\b",
    "module system": r"(?m)^\s*module\b",
    "set_option debug./compiler./interpreter.": r"\bset_option\s+(?:debug|compiler|interpreter)\.",
    "extern / meta attribute": _ATTR + r"(?:extern|init|builtin_\w+|command_elab|term_elab|tactic|delab|app_unexpander|cpass|"
                              r"env_extension)\b",
}
# The bans that also reach a name Lean looks up (a constant, attribute, option or tactic configuration field), which a
# «guillemet» identifier can spell: `«Lean».«ofReduceBool»`, `@[«implemented_by» g]`, `set_option «debug».x`,
# `decide +«native»` all work. They are matched with such identifiers shown as names; every other ban is a keyword,
# which a «guillemet» identifier never is (`def «import count»` is a plain name), so it is matched with them blanked.
NAME_BANS = {"sorry", "native code (native_decide, decide +native, ofReduceBool, trustCompiler)", "implemented_by",
             "simproc/meta command", "set_option debug./compiler./interpreter.", "extern / meta attribute"}
_DECL_START = re.compile(
    r"^(?:@\[|(?:private|protected|noncomputable|partial|unsafe|nonrec|scoped|local)\s|"
    r"(?:def|theorem|lemma|abbrev|instance|structure|inductive|class|opaque|axiom|example|namespace|section|end|open|"
    r"variable|universe|set_option|attribute|mutual|export|deriving|termination_by|decreasing_by)\b|#)", re.M)
# A `sorry` body the agent replaces wholesale; the frozen header ends where the body starts.
_SORRY_BODIES = (re.compile(r":=(?:\s*by)?()\s*sorry\s*\Z"),                 # `:= sorry`, `:= by sorry`
                 re.compile(r":=()\s*fun\b[^\n]*?=>\s*sorry\s*\Z"),             # `:= fun _ => sorry`
                 re.compile(r"()\n[ \t]*\|[^|\n]*=>\s*sorry\s*\Z"))             # a single `| _ => sorry` arm


class LockError(ValueError):
    """The task cannot be locked (sorry outside a body, skeleton does not compile, axioms in the preamble...)."""


class ToolchainError(RuntimeError):
    """The pinned Lean toolchain is missing or is another version."""


# ---------------------------------------------------------------------------------------------------- lexing
# Lean syntaxes whose string argument is interpolated (`{term}` is code); any other string is plain.
_INTERPOLATED = re.compile(r"(?:(?<![\w'!?.])(?:s|f|m|v|println|reportIssue)!|\bdbg_trace|\bthrowError(?:At\s+\S+)?|"
                           r"\btrace\[[^\]]*\]|\b(?:log|throw)Named\w*\s+\S+)\s*$")


def _id_rest(ch: str) -> bool:
    return ch.isalnum() or ch in "_'!?" or 0x2100 <= ord(ch) <= 0x214F or 0x1D49C <= ord(ch) <= 0x1D59F


def _in_ident(src: str, i: int) -> bool:
    """Is position i (a `'`) inside an identifier? Not after a number literal, where Lean starts a char literal."""
    j = i
    while j > 0 and _id_rest(src[j - 1]):
        j -= 1
    return j < i and not src[j].isdigit()


def _plain_ident(s: str) -> bool:
    return bool(s) and not s[0].isdigit() and all(_id_rest(ch) for ch in s)


def mask(src: str, idents: str = "keep") -> tuple[str, list[str]]:
    """Blank out comments and literal contents (newlines kept, same length), mirroring Lean's lexer: nested `/- -/`
    with the character after a plain `/-` skipped, `--` to end of line, strings with escapes, raw strings, char
    literals, «escaped identifiers» and `{code}` inside interpolated strings (`s!"…"`, `dbg_trace "…"`...). A plain
    string that contains a brace may still be read as interpolated by some library syntax: when that reading ends it at
    the same quote its `{code}` stays (and the rest is blanked), otherwise its whole text stays. The text of a
    «guillemet» identifier is kept with idents="keep"; "blank" blanks it (it is a name, never a keyword); "names" shows
    it as the name component it spells, when it spells a plain one, right-aligned (`set_option «debug».skipKernelTC`
    reads `set_option   debug.skipKernelTC`), and blanks it otherwise: options, attributes, constants and tactic
    configuration fields are looked up by name, and Lean never splits a «…» at its dots. Returns (masked, problems);
    an unterminated comment, string or identifier is a problem."""
    return _Masker(src, idents, {}, 0).run()


class _Masker:
    """mask()'s lexer state. `memo` (shared with the speculative lexers) holds the interpolated reading of each plain
    string with a brace, by the position after its opening quote."""

    DEPTH = 16  # nested speculative readings; deeper ones keep the plain string's text

    def __init__(self, src: str, idents: str, memo: dict, depth: int):
        self.src, self.n, self.idents, self.memo, self.depth = src, len(src), idents, memo, depth
        self.out, self.problems = list(src), []

    def run(self) -> tuple[str, list[str]]:
        self.code(0)
        return "".join(self.out), self.problems

    def blank(self, a, b):
        for k in range(a, b):
            if self.out[k] != "\n":
                self.out[k] = " "

    def block_comment(self, i):  # i is just past the opener's third character; Lean's finishCommentBlock
        src, n, depth = self.src, self.n, 1
        while i < n:
            c, i = src[i], i + 1
            if c == "-" and i < n and src[i] == "/":
                i, depth = i + 1, depth - 1
                if depth == 0:
                    return i
            elif c == "/" and i < n and src[i] == "-":
                i, depth = i + 1, depth + 1
        self.problems.append("unterminated comment")
        return n

    def string(self, i, interpolated):  # i is just past the opening quote; returns index past the closing quote
        src, n = self.src, self.n
        start, braced = i, False
        while i < n:
            c = src[i]
            if c == "\\":
                i += 2
            elif c == '"':
                if not braced:
                    self.blank(start, i)
                else:
                    spec = self.interpolated_reading(start)
                    if spec is not None and spec[0] == i + 1:
                        self.out[start:i] = spec[1]
                return i + 1
            elif c == "{" and interpolated:
                self.blank(start, i)
                i = self.code(i + 1, until_brace=True)
                start = i
            else:
                braced = braced or (not interpolated and c in "{}")
                i += 1
        self.problems.append("unterminated string")
        return n

    def interpolated_reading(self, start):
        """The plain string whose text starts at `start`, read as interpolated: (index past its closing quote, masked
        text), or None when that reading does not end (Lean could not parse it so either) or nests too deep."""
        if start not in self.memo:
            self.memo[start] = None
            if self.depth < self.DEPTH:
                spec = _Masker(self.src, self.idents, self.memo, self.depth + 1)
                end = spec.string(start, True)
                if not spec.problems:
                    self.memo[start] = (end, spec.out[start:end - 1])
        return self.memo[start]

    def code(self, i, until_brace=False):
        src, n, depth = self.src, self.n, 0
        while i < n:
            c, nxt = src[i], src[i + 1] if i + 1 < n else ""
            if c == "-" and nxt == "-":
                j = src.find("\n", i)
                j = n if j < 0 else j
                self.blank(i, j)
                i = j
            elif c == "/" and nxt == "-":
                j = self.block_comment(i + 3)
                self.blank(i, j)
                i = j
            elif c == "r" and (i == 0 or not _id_rest(src[i - 1])) and re.match(r'r#*"', src[i:i + 64]):
                hashes = re.match(r'r(#*)"', src[i:]).group(1)
                close = src.find('"' + hashes, i + 2 + len(hashes))
                if close < 0:
                    self.problems.append("unterminated raw string")
                    return n
                self.blank(i + 2 + len(hashes), close)
                i = close + 1 + len(hashes)
            elif c == '"':
                i = self.string(i + 1, bool(_INTERPOLATED.search(src[max(0, i - 60):i])))
            elif c == "'" and nxt and nxt != "'" and not _in_ident(src, i):
                m = re.match(r"'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^\\'\n])'", src[i:i + 12])
                if m:
                    self.blank(i + 1, i + len(m.group(0)) - 1)
                    i += len(m.group(0))
                else:
                    i += 1
            elif c == "«":
                j = src.find("»", i)
                if j < 0:
                    self.problems.append("unterminated «identifier»")
                    return n
                name = src[i + 1:j]
                if self.idents == "names" and _plain_ident(name):
                    self.out[i:j + 1] = [" ", " ", *name]
                elif self.idents != "keep":
                    self.blank(i + 1, j)
                i = j + 1
            elif until_brace and c == "{":
                depth, i = depth + 1, i + 1
            elif until_brace and c == "}":
                if depth == 0:
                    return i + 1
                depth, i = depth - 1, i + 1
            else:
                i += 1
        if until_brace:
            self.problems.append("unterminated string interpolation")
        return n


# ---------------------------------------------------------------------------------------------------- lock
def skeleton(rec: dict) -> str:
    """The file the agent starts from: the five fields in order, blank-line separated."""
    return "\n\n".join(p for p in (rec.get(k, "").strip() for k in FIELDS) if p) + "\n"


def _decls(text: str) -> list[tuple[int, int]]:
    """Spans of top-level declarations/commands (start at a column-0 command keyword; trailing comments excluded)."""
    masked, problems = mask(text)
    if problems:
        raise LockError(f"cannot lex task text: {problems}")
    starts = [m.start() for m in _DECL_START.finditer(masked)
              if not re.match(r"(?:termination_by|decreasing_by|deriving)\b", masked[m.start():])]
    spans = []
    for a, b in zip(starts, starts[1:] + [len(text)]):
        if spans and re.fullmatch(r"@\[[^\]]*\]\s*", masked[spans[-1][0]:a]):
            a = spans.pop()[0]  # an attribute line belongs to the declaration below it
        spans.append((a, a + len(masked[a:b].rstrip())))
    if masked[:starts[0] if starts else len(text)].strip():
        raise LockError("task text does not start with a declaration")
    return spans


def _chunks(rec: dict) -> list[dict]:
    """Frozen declaration chunks of vc-definitions + vc-theorems: `sorry` bodies become holes."""
    out = []
    for field in ("vc-definitions", "vc-theorems"):
        text = rec.get(field, "").strip()
        if not text:
            continue
        for a, b in _decls(text):
            decl, mdecl = text[a:b], mask(text[a:b])[0]
            n_sorry = len(re.findall(r"\bsorry\b", mdecl))
            if n_sorry == 0:
                out.append({"text": decl, "hole_after": False})
                continue
            m = next((m for m in (p.search(mdecl) for p in _SORRY_BODIES) if m), None)
            if n_sorry > 1 or not m:
                raise LockError(f"`sorry` outside a declaration body in {field}: {decl[:120]!r}")
            out.append({"text": decl[:m.start(1)], "hole_after": True})
    return out


def lock(rec: dict, lean: str | None = None, timeout: int = 300) -> dict:
    """Build frozen.json for one vericoding Lean record. Raises LockError if the task cannot be locked honestly."""
    for field in ("vc-preamble", "vc-helpers", "vc-postamble"):
        if re.search(r"\bsorry\b", mask(rec.get(field, ""))[0]):
            raise LockError(f"`sorry` in {field}")
    chunks = _chunks(rec)
    lean = find_lean(lean)
    body = skeleton({**rec, "vc-postamble": ""})  # sorry implementations cannot run the postamble's #eval tests
    with tempfile.TemporaryDirectory(prefix="sl-lock-") as d:
        ok, out, _ = _compile(body + _sentinel("lock"), d, lean, timeout)
        msgs = [m for m in _messages(out) if m["severity"] == "error"]
        if not ok or msgs:
            raise LockError("skeleton does not compile: " + "\n".join(m["text"] for m in msgs)[-1500:])
        rep = _check_lean(d, lean, None)
    if rep["axioms_declared"]:
        raise LockError(f"task declares axioms: {rep['axioms_declared']}")
    consts, impl = {}, []
    for name, fp in rep["consts"].items():
        if fp["type_sorry"]:
            raise LockError(f"`sorry` inside the statement of {name}")
        if fp["kind"] == "def" and fp["value_sorry"]:
            impl.append(name)
    for name, fp in rep["consts"].items():
        if any(name.startswith(p + ".") for p in impl) or name.rsplit(".", 1)[-1] in DERIVED:
            continue  # equation/induction lemmas: realized on demand and determined by their (locked) definition
        role = "impl" if name in impl else "theorem" if fp["kind"] == "theorem" else "frozen"
        if role == "impl" and fp["noncomputable"]:
            raise LockError(f"implementation def {name} is noncomputable in the task")
        if role == "frozen" and fp["value_sorry"]:
            raise LockError(f"`sorry` inside frozen {name}")
        consts[name] = {"kind": fp["kind"], "role": role, "type": _sha(fp["type"]), "ctors": fp["ctors"],
                        "value": _sha(fp["value"]) if role == "frozen" and fp["value"] is not None else None}
    # Python-style expected outputs ('str', True, None) can never match Lean's #eval: such tasks cannot be PROVED.
    suspect = re.findall(r"(?m)^info: (?:'.*'|True|False|None|\[.*'.*\])$", rec.get("vc-postamble", ""))
    frozen = {"id": rec.get("id", ""), "toolchain": TOOLCHAIN, "skeleton": skeleton(rec),
              "warnings": [f"postamble expects a non-Lean output: {x}" for x in suspect],
              "preamble": rec.get("vc-preamble", "").strip(), "postamble": rec.get("vc-postamble", "").strip(),
              "chunks": chunks, "imports": rep["imports"], "constants": consts,
              "default_instances": rep["default_instances"]}
    frozen["sha"] = _sha(json.dumps([frozen["preamble"], chunks, frozen["postamble"], consts], sort_keys=True))[:12]
    return frozen


# ---------------------------------------------------------------------------------------------------- check
def _blank(s: str) -> bool:
    masked, problems = mask(s)
    return not masked.strip() and not problems


def _norm(s: str) -> str:
    """Trailing whitespace and CRLF are not part of the lock (editors strip them; Lean ignores them)."""
    return "\n".join(ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n"))


def holes(text: str, frozen: dict) -> tuple[list[str], list[str]]:
    """Split the solution into agent-written holes around the frozen chunks; only comments may precede the preamble
    or follow the postamble. Returns (holes, violations)."""
    text, pre, post = _norm(text), _norm(frozen["preamble"]), _norm(frozen["postamble"])
    i = text.find(pre) if pre else 0
    if i < 0 or (pre and not _blank(text[:i])):
        return [], ["the preamble (vc-preamble) was modified or is not at the start of the file"]
    pos = i + len(pre)
    end = text.rfind(post) if post else len(text)
    if end < pos or (post and not _blank(text[end + len(post):])):
        return [], ["the postamble (vc-postamble, with its #guard_msgs tests) was modified or is not at the end"]
    out = [text[:i]]
    for c in frozen["chunks"]:
        chunk = _norm(c["text"])
        j = text.find(chunk, pos, end)
        if j < 0:
            return [], [f"frozen declaration modified, moved or missing: `{chunk.splitlines()[0][:100]}`"]
        out.append(text[pos:j])
        pos = j + len(chunk)
    out.append(text[pos:end])
    if post and re.search(r"(?<![\w'!?.])in\s*\Z", mask(out[-1])[0]):
        return [], ["a command ending in `in` right before the postamble would wrap its first #guard_msgs test"]
    return out, []


def ban_scan(hole_texts: list[str]) -> list[str]:
    v = []
    for h in hole_texts:
        masked, problems = mask(h, idents="blank")
        named = mask(h, idents="names")[0]
        v += [f"{p} in your code (a comment or string may not hide or span frozen text)" for p in problems]
        for name, pat in BANNED.items():
            m = re.search(pat, named if name in NAME_BANS else masked)
            if m:
                line = h[:m.start()].count("\n")
                v.append(f"banned construct: {name} (`{h.splitlines()[line].strip()[:80] if h.splitlines() else ''}`)")
    return sorted(set(v))


def check(text: str, frozen: dict, lean: str | None = None, timeout: int = 300, bans: bool = True) -> dict:
    """Grade one solution. bans=False is the grader's lenient variant (no text ban scan; kernel-side checks stay)."""
    t0 = time.time()
    hs, v = holes(text, frozen)
    if v:
        return _result("SPEC_TAMPERED", t0, violations=v)
    if bans:
        v = ban_scan(hs)
        if v:
            return _result("CHEAT_REJECTED", t0, violations=v)
    try:
        lean = find_lean(lean)
    except (ToolchainError, OSError, subprocess.SubprocessError) as e:
        return _result("CHECK_ERROR", t0, violations=[str(e)])
    nonce = secrets.token_hex(6)
    with tempfile.TemporaryDirectory(prefix="sl-lean-") as d:
        ok, out, timed_out = _compile(text + _sentinel(nonce), d, lean, timeout)
        msgs = _messages(out)
        errs = [m for m in msgs if m["severity"] == "error"]
        sorry = [m for m in msgs if "declaration uses 'sorry'" in m["text"]]
        if timed_out or not ok or errs or sorry:
            tail = "\n\n".join(m["text"] for m in errs + sorry) or out
            goals = [m["text"] for m in errs if "unsolved goals" in m["text"]][:3]
            return _result("UNPROVED", t0, violations=[f"lean timed out after {timeout} s"] if timed_out else [],
                           errors_tail=_clip(tail, 3000, tail=True), goals=[_clip(g, 1500) for g in goals])
        try:
            with tempfile.TemporaryDirectory(prefix="sl-check-") as d2:  # only the .olean crosses over
                shutil.copyfile(Path(d, "Solution.olean"), Path(d2, "Solution.olean"))
                rep = _check_lean(d2, lean, list(frozen["constants"]))
            tampered, cheats = _kernel_violations(rep, frozen, nonce)
        except Exception as e:  # noqa: BLE001 - any checker failure fails closed
            return _result("CHECK_ERROR", t0, violations=[f"Check.lean failed: {_clip(repr(e), 800)}"])
    if tampered:
        return _result("SPEC_TAMPERED", t0, violations=tampered + cheats)
    if cheats:
        return _result("CHEAT_REJECTED", t0, violations=cheats)
    return _result("PROVED", t0)


def _kernel_violations(rep: dict, frozen: dict, nonce: str) -> tuple[list[str], list[str]]:
    tampered, cheats = [], []
    if f"millennium_eof_{nonce}" not in rep["module_consts"]:
        cheats.append("the file did not elaborate to its end (#exit, an open namespace or a swallowed postamble)")
    for name, ref in frozen["constants"].items():
        fp = rep["consts"].get(name, {"exists": False})
        if not fp["exists"] or not fp["module"]:
            tampered.append(f"frozen `{name}` is missing (renamed, namespaced, private or commented out)")
            continue
        if fp["kind"] != ref["kind"]:
            tampered.append(f"frozen `{name}` changed kind: {ref['kind']} -> {fp['kind']} (partial/opaque/axiom?)")
            continue
        if _sha(fp["type"]) != ref["type"] or fp["ctors"] != ref["ctors"]:
            tampered.append(f"the statement/type of `{name}` elaborates differently from the task's (same text, "
                            "different meaning: a shadowing instance, open, variable or notation?)")
        if ref["value"] is not None and _sha(fp["value"]) != ref["value"]:
            tampered.append(f"the frozen definition `{name}` elaborates differently from the task's")
        bad = sorted(set(fp["axioms"]) - STD_AXIOMS)
        if bad:
            cheats.append(f"`{name}` depends on {', '.join(bad)} (only propext, Classical.choice, Quot.sound allowed)")
        if ref["role"] == "impl":
            if fp["noncomputable"] or not fp["ir"]:
                cheats.append(f"implementation `{name}` is noncomputable / has no compiled code")
            if fp["implemented_by"] or fp["extern"]:
                cheats.append(f"implementation `{name}` swaps its compiled code (implemented_by/extern)")
    if rep["imports"] != frozen["imports"]:
        cheats.append(f"imports changed: {rep['imports']} (the task's are {frozen['imports']})")
    if rep["axioms_declared"]:
        cheats.append(f"the file declares axioms: {', '.join(rep['axioms_declared'])}")
    if rep["implemented_by_or_extern"]:
        cheats.append(f"implemented_by/extern on: {', '.join(rep['implemented_by_or_extern'][:5])}")
    agent = set(rep["module_consts"]) - set(frozen["constants"])
    meta = [c for c in rep["meta_consts"] if c in agent]
    if meta:
        cheats.append(f"meta-programming or IO declarations: {', '.join(meta[:5])}")
    graph = rep["class_graph"]
    if graph["errors"]:
        raise RuntimeError(f"Check.lean could not analyse {graph['errors'][:5]}")
    # Proofs decide no value (Prop-valued classes, and Decidable), but they can enable a coercion into some type.
    scanned = _closure(POSTAMBLE_CLASSES, graph["feeds"], {"Decidable", *graph["prop_classes"]})
    into = _closure(COE_CLASSES, graph["feeds"], {"Decidable"})
    for inst in rep["instances"]:
        if inst["name"] not in agent:
            continue
        if inst["class"] == "?":
            cheats.append(f"`{inst['name']}` could not be analysed as a possible instance: {_clip(inst['error'], 300)}")
        elif inst["class"] in scanned and not set(inst["anchors"]) & agent:
            own = sorted(set(inst["semi"]) & agent) if inst["class"] in OUT_OF_OWN_TYPE else []
            producers = _producers(own, inst, rep, graph, into, agent)
            if own and not producers:
                continue  # a coercion out of the agent's own type that nothing can coerce into
            if own:
                cheats.append(f"instance `{inst['name']}` coerces your type {', '.join(own)} into a task/base type, and "
                              f"{', '.join(producers[:3])} can produce a {own[0]} from other types: chained by "
                              "coercion resolution they form a new coercion between task/base types that could change "
                              "how the #guard_msgs tests elaborate (keep one direction, or call a conversion function)")
            else:
                cheats.append(f"instance `{inst['name']}` of {inst['class']} applies to task/base types and could "
                              "change what the #guard_msgs tests compute or print (an abbrev/def alias is not a new "
                              "type: declare your own `structure` and put the instance on it)")
    # Every default-instance entry the module added beyond the task's own (also one on a library instance).
    defaults = list((Counter(rep["default_instances"]) - Counter(frozen.get("default_instances", []))).elements())
    if defaults:
        cheats.append(f"@[default_instance] on {', '.join(defaults[:5])} could change how the #guard_msgs tests elaborate")
    if rep["replay"] != "ok":
        cheats.append(f"kernel replay rejected the module: {_clip(rep['replay'], 400)}")
    return tampered, cheats


def _closure(seeds, feeds: dict, skip: set) -> set:
    """Classes whose instances can decide how a goal of a seed class resolves: closed over Check.lean's class graph
    (class -> classes of its instances' instance arguments), never following the classes in `skip`."""
    out, todo = set(seeds), list(seeds)
    while todo:
        for h in feeds.get(todo.pop(), ()):
            if h not in out and h not in skip:
                out.add(h)
                todo.append(h)
    return out


def _producers(own: list, inst: dict, rep: dict, graph: dict, into: set, agent: set) -> list:
    """What could coerce some task/base type into one of the agent's types `own` (see COE_CLASSES), possibly through
    other agent types. Anything, for an instance the environment derives for arbitrary types in a class of `into`, or
    a class-typed constant that could not be analysed. Otherwise a least fixpoint over the module's other class-typed
    constants with a class in `into`: such a constant produces the agent types it mentions outside a coercion's source,
    unless that source is headed by an agent type nothing produces (`Coe Tally Summed` lets a Summed into a chain only
    once something can coerce a task/base type into Tally; a source that is a variable or a task/base type, and any
    class without a source such as NatCast, always counts)."""
    if not own:
        return []
    out = [f"`{f['name']}`" for f in graph["free"] if f["class"] in into][:3]
    out += [f"`{o['name']}`" for o in rep["instances"] if o["name"] != inst["name"] and o["class"] == "?"]
    if out:
        return out
    links = [o for o in rep["instances"] if o["name"] != inst["name"] and o["class"] in into]
    made: dict[str, str] = {}  # agent type -> a constant that produces it
    grew = True
    while grew:
        grew = False
        for o in links:
            src = set(o.get("source", ())) & agent
            if src and not src & made.keys():
                continue
            for t in o["mentions"]:
                if t in agent and t not in made:
                    made[t], grew = o["name"], True
    return sorted({f"`{made[t]}`" for t in own if t in made})


# ---------------------------------------------------------------------------------------------------- toolchain
def find_lean(lean: str | None = None) -> str:
    """The pinned toolchain's own `lean` binary: never PATH or the elan proxy (a lean-toolchain file could redirect it)."""
    cands = [lean, os.environ.get("MILLENNIUM_LEAN"), os.environ.get("SORRYLESS_LEAN"), "/opt/lean/bin/lean",
             str(Path.home() / ".elan" / "toolchains" / f"leanprover--lean4---v{TOOLCHAIN}" / "bin" / "lean")]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            v = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=60).stdout
            if f"version {TOOLCHAIN}," not in v:
                raise ToolchainError(f"{c} is not Lean {TOOLCHAIN}: {v.strip()[:120]}")
            return c
    raise ToolchainError(f"Lean {TOOLCHAIN} not found (set MILLENNIUM_LEAN or install leanprover/lean4:v{TOOLCHAIN})")


def _env(d: str) -> dict:
    return {"PATH": "/usr/bin:/bin", "HOME": d, "LANG": "C.UTF-8"}


def _compile(text: str, d: str, lean: str, timeout: int) -> tuple[bool, str, bool]:
    Path(d, "Solution.lean").write_text(text, encoding="utf-8")
    try:
        p = subprocess.run([lean, "-o", "Solution.olean", "Solution.lean"], cwd=d, env=_env(d), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "", True
    return p.returncode == 0 and Path(d, "Solution.olean").exists(), p.stdout + p.stderr, False


def _check_lean(d: str, lean: str, names: list[str] | None) -> dict:
    args = [lean, "--run", str(CHECK_LEAN), d]
    if names is not None:
        Path(d, "names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        args.append(str(Path(d, "names.txt")))
    p = subprocess.run(args, cwd=d, env=_env(d), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=600)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError((p.stdout + p.stderr)[-1500:])
    return json.loads(lines[-1])


def _sentinel(nonce: str) -> str:
    return f"\n\ntheorem millennium_eof_{nonce} : True := trivial\n"


def _messages(out: str) -> list[dict]:
    msgs = []
    for block in re.split(r"(?m)^(?=\S+:\d+:\d+: )", out):
        m = re.match(r"\S+:\d+:\d+: (error|warning|info)\b", block)
        if m:
            msgs.append({"severity": m.group(1), "text": block.strip()})
    return msgs


def _sha(s) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()


def _clip(s: str, n: int, tail: bool = False) -> str:
    if len(s) <= n:
        return s
    return "… " + s[-n:] if tail else s[:n] + " …"


def _result(status: str, t0: float, violations=(), errors_tail: str = "", goals=()) -> dict:
    return {"status": status, "violations": list(violations), "errors_tail": errors_tail, "goals": list(goals),
            "secs": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------------- feedback
def feedback(r: dict, rnd: int, max_rounds: int) -> str:
    if r["status"] == "PROVED":
        return ""
    head = f"⟦Millennium Harness⟧ ✗ {r['status']} · Lean {TOOLCHAIN} · round {rnd}/{max_rounds}\n"
    if r.get("violations"):
        head += "Rule violations (these never count as verified):\n" + "\n".join(f"- {x}" for x in r["violations"]) + "\n"
    if r.get("goals"):
        head += "Unsolved goals:\n" + "\n".join(f"```\n{g}\n```" for g in r["goals"]) + "\n"
    if r.get("errors_tail"):
        head += f"`lean Solution.lean` errors (tail):\n```\n{_clip(r['errors_tail'], 2000, tail=True)}\n```\n"
    return head + ("Keep the preamble, every declaration header up to its `:=` (and every frozen definition) and the "
                   "postamble with its #guard_msgs tests byte-identical; replace each `sorry` with a real "
                   "implementation or proof, adding helper defs and lemmas between declarations if useful. sorry, "
                   "admit, axioms, native_decide, implemented_by, macros/elaborators and set_option debug.* never "
                   "count. Make `lean Solution.lean` report no errors and no `sorry` honestly.")


def receipt_line(r: dict, frozen: dict) -> str:
    if r["status"] != "PROVED":
        return f"✗ {r['status']} · Lean {TOOLCHAIN}"
    return (f"∎ PROVED · Lean {TOOLCHAIN} · 0 sorry · axioms ⊆ std · statements locked {frozen.get('sha', '')} · "
            f"{r['secs']}s")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m millennium_harness.lean")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("solution")
    c.add_argument("frozen")
    c.add_argument("--round", type=int, default=1)
    c.add_argument("--max-rounds", type=int, default=2)
    c.add_argument("--lean")
    c.add_argument("--lenient", action="store_true", help="grader variant without the text ban scan")
    lk = sub.add_parser("lock")
    lk.add_argument("record")
    lk.add_argument("out")
    lk.add_argument("--lean")
    a = ap.parse_args(argv)
    if a.cmd == "lock":
        try:
            frozen = lock(json.load(open(a.record)), a.lean)
        except LockError as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(2)
        Path(a.out).write_text(json.dumps(frozen, indent=1, ensure_ascii=False))
        print(json.dumps({"id": frozen["id"], "constants": len(frozen["constants"]), "sha": frozen["sha"]}))
        return
    frozen = json.load(open(a.frozen))
    if not os.path.exists(a.solution):
        r = _result("UNPROVED", time.time(), violations=[f"there is no solution at {a.solution}"])
    else:
        try:
            r = check(open(a.solution, encoding="utf-8").read(), frozen, a.lean, bans=not a.lenient)
        except Exception as e:  # noqa: BLE001 - fail closed
            r = _result("CHECK_ERROR", time.time(), violations=[f"{type(e).__name__}: {_clip(str(e), 500)}"])
    r["feedback"] = feedback(r, a.round, a.max_rounds)
    r["receipt"] = receipt_line(r, frozen)
    sys.stdout.write(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
