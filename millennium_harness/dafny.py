"""∎ rung: honest Dafny verification. The frozen parts (preamble, spec, postamble) must be byte-identical and mean
what they mean on their own, agent-written regions must not use escape hatches, and `dafny verify` must report zero
errors.

  python -m millennium_harness.dafny check SOLUTION.dfy FROZEN.json [ROUND MAX_ROUNDS]   # -> JSON {status, violations, verify}
  python -m millennium_harness.dafny lint FROZEN.json                                     # -> JSON {problems} (pool filter)

Statuses: PROVED, UNPROVED, CHEAT_REJECTED, SPEC_TAMPERED, CHECK_ERROR. Everything outside the frozen parts is agent
text, and:
0. The task is checkable (`task_problems`, else CHECK_ERROR): the frozen parts alone, with a bodiless spec, pass
   `dafny resolve`. Otherwise the frozen text uses a name that only agent code defines, and the agent picks its meaning.
1. Frozen parts appear in order, each at the top level (bracket depth 0: never inside an agent class, datatype or trait,
   whose members would shadow frozen names and whose {:autocontracts} would add contracts), and agent text never glues
   onto them: no frozen part starts inside an agent comment or string, no modifier (`ghost`, `static`, ...) or open
   expression right before one, agent text after the preamble or postamble starts a new declaration, and the spec is
   followed directly by the method body `{`. Agent text declares no top-level name (declaration or datatype
   constructor) that the frozen text uses, since a top-level function or const shadows a frozen constructor. With 0,
   every frozen name therefore binds exactly as in the frozen text alone.
2. Agent text (comments and strings blanked) has no BANNED construct (incl. any `@` other than an `old@L`-style label
   reference: Dafny accepts `@ Axiom`, `@/**/Axiom` and `int@Axiom`), only SAFE_ATTRIBUTES, and no bodiless
   lemma/function/method with `ensures`.
3. `dafny verify` with VERIFY_FLAGS reports 0 errors, and every frozen method/lemma has a passed correctness result
   (so the verified count is at least the number of frozen methods; `{:only}` elsewhere cannot skip them).
4. `dafny audit` has no soundness finding in agent text; quantifier-trigger hints are ignored (brittleness, not
   soundness).
5. `dafny translate py --no-verify` succeeds: a non-compilable `:|`, bodiless loop or bodiless method is no
   implementation.
Dafny runs on a private copy of the text in a fresh temp dir, so stray files (dfyconfig.toml, includes) cannot
interfere. MILLENNIUM_DAFNY (or the older SORRYLESS_DAFNY) overrides the `dafny` binary.

VERIFY_FLAGS: `--cores 1` and a fixed Z3 resource limit of 5e8 units per assertion batch, which is deterministic
(unlike Dafny's default 30 s time limit). Calibration: the heaviest of 612 passing batches in the 92 E4b agent
solutions used 1.96e8 units (2.5x margin); one core does 1.6-18M units/s depending on load, so 5e8 is about 30 s on a
fast idle core, i.e. Dafny's default budget. The 180 s per-batch time limit is only a backstop. Agents cannot override
these: `{:rlimit}`/`{:timeLimit}` are not in SAFE_ATTRIBUTES and `@ResourceLimit`/`@TimeLimit` are banned.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile

HELPERS_BEGIN = "// ===== HELPERS: you may add lemmas, functions and predicates here ====="
HELPERS_END = "// ===== END HELPERS ====="
VERIFY_FLAGS = ["--allow-warnings", "--cores", "1", "--resource-limit", "500000000", "--verification-time-limit", "180"]
BANNED = {
    "assume": r"\bassume\b",
    "axiom attribute": r"\{:\s*axiom\b",
    "verify false": r"\{:\s*verify\b",
    "extern": r"\{:\s*extern\b",
    "decreases *": r"\bdecreases\s+\*",
    "expect": r"\bexpect\b",
    "trusted body attribute": r"\{:\s*(?:opaque_reveal|only|focus)\b",
    "compile false": r"\{:\s*compile\b",
    "module": r"\bmodule\b",
    "import": r"\bimport\b",
    "include": r"\binclude\b",
    "refines": r"\brefines\b",
    "@-attribute": r"@",  # after LABEL_REF is blanked
}
LABEL_REF = re.compile(r"\b(?:old|unchanged|fresh|allocated)\s*@")
# Sound and common in honest proofs. Anything else (e.g. {:selective_checking}, {:termination false}, {:rlimit}) is
# rejected: Dafny passes unknown attributes on to Boogie, and some of them silently drop proof obligations.
# {:autocontracts} is out: it adds contracts (`requires Valid()`) to methods.
SAFE_ATTRIBUTES = {"trigger", "induction", "fuel", "opaque", "transparent", "vcs_split_on_every_assert", "split_here",
                   "isolate_assertions", "nowarn", "tailrecursion", "nativeType", "abstemious", "print", "error",
                   "disableNonlinearArithmetic"}
MODIFIERS = r"ghost|static|twostate|abstract|opaque|least|greatest|inductive|export|replaceable"
IDENT = r"[A-Za-z_][\w'?]*"
_DECL_KW = (r"function|predicate|copredicate|method|lemma|colemma|const|var|type|newtype|datatype|codatatype|class"
            r"|trait|iterator")
TOP_DECL = re.compile(rf"(?=\b(?:{_DECL_KW})\s+({IDENT}))")  # lookahead: `function method F` yields both names
# Dafny has no module-level `var`: a top-level-looking `var x :=` in frozen text is a let-binding inside an expression
FROZEN_DECL = re.compile(rf"(?=\b(?:{_DECL_KW.replace('|var', '')})\s+({IDENT}))")
CTORS = re.compile(rf"\b(?:co)?datatype\b[^=]*=(.*?)(?=\b(?:{_DECL_KW})\b|\Z)", re.S)
DECL_START = re.compile(r"\s*(?:$|@|(?:lemma|function|predicate|method|ghost|const|datatype|codatatype|type|newtype|class"
                        r"|trait|static|opaque|twostate|least|greatest|inductive|copredicate|colemma|iterator|abstract|module"
                        r"|import|include)\b)")
UNSOUND = re.compile(r"Possible mitigation|assum|axiom|no body|bodyless|verify false|extern|decreases \*|VerifyOnly",
                     re.I)
TRIGGER_HINT = re.compile(r"trigger", re.I)
_CHAR = re.compile(r"'(?:[^'\\\n]|\\(?:u\{[0-9a-fA-F_]+\}|U\{[0-9a-fA-F_]+\}|u[0-9a-fA-F]{4}|.))'")


def _literals(text: str) -> list[tuple[int, int, bool]]:
    """(start, end, is_comment) of comments (`//`, nested `/* */`) and string/char literals; unterminated -> to end."""
    out, i, n = [], 0, len(text)
    while i < n:
        char = text[i] == "'" and not (i and (text[i - 1].isalnum() or text[i - 1] in "_'?")) and _CHAR.match(text, i)
        if text.startswith("//", i):
            j, com = text.find("\n", i), True
            j = n if j < 0 else j
        elif text.startswith("/*", i):
            depth, j, com = 1, i + 2, True
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
        elif text[i] == '"' or text.startswith('@"', i):
            verbatim, com = text[i] == "@", False
            j = i + 2 if verbatim else i + 1
            while j < n:
                if verbatim and text.startswith('""', j) or not verbatim and text[j] == "\\":
                    j += 2
                elif text[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
        elif char:
            j, com = char.end(), False
        else:
            i += 1
            continue
        out.append((i, min(j, n), com))
        i = min(j, n)
    return out


def strip_comments(code: str) -> str:
    """Blank out comments and string/char literals (keeping offsets and newlines; literals keep `"` delimiters)."""
    out = list(code)
    for s, e, com in _literals(code):
        out[s:e] = [c if c == "\n" else " " for c in code[s:e]]
        if not com:
            out[s] = out[e - 1] = '"'
    return "".join(out)


def _layout(text: str, frozen: dict) -> tuple[list[tuple[int, int]] | None, list[str]]:
    """Spans of the frozen preamble, spec and postamble, in this order (or violations)."""
    pre, spec, post = (frozen.get(k, "").strip() for k in ("preamble", "spec", "postamble"))
    v = [f"frozen {k} was modified" for k, s in zip(("preamble", "spec", "postamble"), (pre, spec, post))
         if s and s not in text]
    if v:
        return None, v
    a = text.find(pre) if pre else 0
    b = text.find(spec, a + len(pre))
    c = text.rfind(post) if post else len(text)
    if b < 0 or c < b + len(spec):
        return None, ["frozen parts are out of order (preamble, helpers, spec, method body, postamble)"]
    return [(a, a + len(pre)), (b, b + len(spec)), (c, c + len(post))], []


def _depths(code: str) -> list[int]:
    """Bracket depth ({ ( [) of each character of comment-stripped code; a bracket has its outer depth."""
    out, d = [], 0
    for ch in code:
        d -= ch in ")]}"
        out.append(d)
        d += ch in "([{"
    return out


def _seams(text: str, code: str, spans: list[tuple[int, int]]) -> list[str]:
    """Agent text may add whole declarations and the method body, but never glue onto or swallow a frozen part."""
    v, cuts, depth = [], [0] + [x for s in spans for x in s] + [len(text)], _depths(code)
    lits = [(s, e) for s, e, _ in _literals(text)]
    for k, name in enumerate(("preamble", "spec", "postamble")):
        (s, e), before, after = spans[k], code[cuts[2 * k]:spans[k][0]], code[spans[k][1]:cuts[2 * k + 3]]
        if s == e:
            continue
        if any(a < s < b for a, b in lits):
            v.append(f"the frozen {name} starts inside an agent comment or string")
        else:
            if before.strip() and (not re.search(r"[\w)\]}>'?;\"]\s*$", before)
                                   or re.search(rf"\b(?:{MODIFIERS})\s*$", before)):
                v.append(f"agent text right before the frozen {name} changes it: ...{' '.join(before.split()[-4:])!r}")
            if depth[s]:
                v.append(f"the frozen {name} must stay at the top level, but agent text before it leaves {depth[s]} "
                         "bracket(s) open (e.g. a class, datatype or trait body around it)")
        if name == "spec" and not re.match(r"\s*\{", after):
            v.append("the frozen spec must be followed directly by the method body `{ ... }` "
                     f"(no extra requires/ensures/modifies or expression): {after.lstrip()[:40]!r}...")
        elif name != "spec" and not DECL_START.match(after):
            v.append(f"agent text right after the frozen {name} must start a new declaration: "
                     f"{after.lstrip()[:40]!r}...")
    return v


def _agent_spans(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cuts = [0] + [x for s in spans for x in s] + [len(text)]
    return [(cuts[i], cuts[i + 1]) for i in range(0, len(cuts), 2) if cuts[i] < cuts[i + 1]]


def _captures(code: str, spans: list[tuple[int, int]]) -> list[str]:
    """Top-level names (declarations, datatype constructors) declared by agent text that the frozen text also declares.

    The frozen text must resolve on its own (task_problems), so every name it uses is a frozen top-level declaration,
    a constructor, a builtin or a local binder; locals shadow globals, so only re-declaring a frozen top-level name or
    constructor (e.g. a function named like a frozen constructor) can change what the frozen parts mean."""
    depth = _depths(code)

    def top_names(regions, decl=TOP_DECL):
        names = set()
        for s, e in regions:
            flat = "".join(c if depth[i] == 0 and c not in "([{}])" else " " for i, c in enumerate(code[s:e], s))
            names |= {m.group(1) for m in decl.finditer(flat)}
            names |= {n for m in CTORS.finditer(flat) for n in re.findall(IDENT, m.group(1))}
        return names
    clash = (top_names(_agent_spans(code, spans)) & top_names(spans, FROZEN_DECL)) - set(MODIFIERS.split("|"))
    return [f"agent text declares `{n}`, a name the frozen text declares: the frozen parts must mean what they mean "
            "on their own (rename it)" for n in sorted(clash)]


def agent_regions(text: str, frozen: dict) -> tuple[str | None, list[str]]:
    """Return (agent-written code, violations about frozen parts)."""
    spans, v = _layout(text, frozen)
    return (None, v) if spans is None else ("\n".join(text[s:e] for s, e in _agent_spans(text, spans)), v)


def _bodiless_with_ensures(code: str) -> list[str]:
    """lemma/function/method declarations that have `ensures` but no body (= unproven assumptions)."""
    # A header ends at a blank line, the next declaration, or EOF; `;` is not a terminator (it occurs inside
    # `ensures var x := e; P` let-expressions), and reaching `{` first means the declaration has a body.
    nxt = r"\n\s*(?:lemma|method|function|predicate|ghost|datatype|codatatype|const|class|trait|type|newtype|module|" \
          r"import|include|twostate|opaque|static|least|greatest|iterator)\b"
    pat = rf"\b(lemma|method|function|predicate)\b(?:(?!{nxt})[^{{])*?\bensures\b(?:(?!{nxt})[^{{])*?(?=\n\s*\n|{nxt}|\Z)"
    out = []
    for m in re.finditer(pat, code, re.S):
        if not code[m.end():].lstrip().startswith("{"):
            out.append(m.group(0).split("(")[0].strip())
    return out


def _bans(segments: list[str]) -> list[str]:
    code = LABEL_REF.sub(lambda m: m.group(0)[:-1] + " ", "\n".join(segments))
    v = [f"banned construct: {name}" for name, pat in BANNED.items() if re.search(pat, code)]
    for a in sorted(set(re.findall(r"\{:\s*(\w+)", code)) - SAFE_ATTRIBUTES):
        if not any(re.search(p, "{:" + a + " ") for p in BANNED.values()):
            v.append(f"attribute {{:{a}}} is not allowed (allowed: {', '.join(sorted(SAFE_ATTRIBUTES))})")
    return v + [f"bodiless declaration with ensures: {d}" for seg in segments for d in _bodiless_with_ensures(seg)]


def _dafny(args: list[str], cwd: str, timeout: int) -> tuple[int | None, str]:
    try:
        p = subprocess.run([os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY", "dafny")), *args],
                           cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return None, f"dafny {args[0]} timed out after {timeout}s"


def _audit(text: str, name: str, tmp: str, agent: list[tuple[int, int]]) -> list[str]:
    """`dafny audit` exits 0 even with findings: parse them, keep soundness findings located in agent text."""
    rc, out = _dafny(["audit", name], tmp, 120)
    starts = [0] + [m.end() for m in re.finditer(r"\n", text)]
    found = set()
    for m in re.finditer(r"([^\s()]+)\((\d+),(\d+)\): Warning: (.+)", out):
        f, ln, col, msg = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if TRIGGER_HINT.search(msg) or not UNSOUND.search(msg):  # ignore trigger/brittleness hints
            continue
        off = starts[min(ln, len(starts)) - 1] + col
        if os.path.basename(f) != name or any(a <= off < b for a, b in agent):
            found.add(f"audit: {msg[:160]}")
    return sorted(found)


def _unverified(tmp: str, spec: str) -> list[str]:
    """Frozen methods/lemmas without a passed `(correctness)` result in the verifier's CSV log."""
    try:
        with open(os.path.join(tmp, "verify.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        rows = []
    passed = {r.get("TestResult.DisplayName", "") for r in rows if r.get("TestResult.Outcome") == "Passed"}
    names = re.findall(r"\b(?:method|lemma)\s+(?:\{:[^}]*\}\s*)*([\w'?]+)", strip_comments(spec))
    return [n for n in names if f"{n} (correctness)" not in passed]


def task_problems(frozen: dict, timeout: int = 120) -> list[str]:
    """Model-free task check (pool filter; check() runs it too): the frozen parts alone, with the spec left bodiless,
    must pass `dafny resolve`. Catches non-Dafny text (leftover LLM prose), malformed splits and frozen text that
    calls functions it never defines."""
    skeleton = "\n\n".join(frozen.get(k, "").strip() for k in ("preamble", "spec", "postamble")) + "\n"
    with tempfile.TemporaryDirectory(prefix="sl-dafny-") as tmp:
        with open(os.path.join(tmp, "frozen.dfy"), "w", encoding="utf-8") as f:
            f.write(skeleton)
        rc, out = _dafny(["resolve", "--allow-warnings", "frozen.dfy"], tmp, timeout)
    if rc == 0:
        return []
    errs = list(dict.fromkeys(re.findall(r"Error: (.+)", out))) or [out.strip()[-300:]]
    return ["task error: the frozen preamble/spec/postamble do not resolve on their own, so no solution can be "
            "certified: " + " | ".join(errs[:3])]


def check(text: str, frozen: dict, path: str, timeout: int = 300) -> dict:
    problems = task_problems(frozen)
    if problems:
        return {"status": "CHECK_ERROR", "violations": problems}
    spans, v = _layout(text, frozen)
    code = strip_comments(text)
    v = v or _seams(text, code, spans) + _captures(code, spans)
    if v:
        return {"status": "SPEC_TAMPERED", "violations": v}
    agent = _agent_spans(text, spans)
    v = _bans([code[s:e] for s, e in agent])
    with tempfile.TemporaryDirectory(prefix="sl-dafny-") as tmp:
        name = os.path.basename(path) if path.endswith(".dfy") else "solution.dfy"
        with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
            f.write(text)
        log = f"csv;LogFileName={os.path.join(tmp, 'verify.csv')}"
        rc, out = _dafny(["verify", *VERIFY_FLAGS, "--log-format", log, name], tmp, timeout)
        m = re.search(r"finished with (\d+) verified, (\d+) errors?", out)
        ok = rc == 0 and m is not None and m.group(2) == "0"
        verified = int(m.group(1)) if m else 0
        out = re.sub(r"(?m)^Results File: .*$", "", out).strip()[-6000:]
        v += _audit(text, name, tmp, agent)
        if ok:
            v += [f"frozen method/lemma `{n}` was not verified (verified count {verified})"
                  for n in _unverified(tmp, frozen["spec"])]
        if ok and not v:
            rc, tr = _dafny(["translate", "py", "--no-verify", "--allow-warnings", "--output",
                             os.path.join(tmp, "py", "out"), name], tmp, timeout)
            if rc != 0:
                errs = re.findall(r"Error: (.+)", tr) or [tr.strip()[-300:]]
                v.append("not compilable (`dafny translate py --no-verify` failed): " + " | ".join(errs[:3]))
    r = {"verify_ok": ok, "verified": verified, "verify": out[-3000:]}
    if v:
        return {"status": "CHEAT_REJECTED", "violations": v, **r}
    return {"status": "PROVED" if ok else "UNPROVED", "violations": [], **r}


def feedback(r: dict, rnd: int, max_rounds: int) -> str:
    if r["status"] == "PROVED":
        return ""
    head = f"⟦Millennium Harness⟧ ✗ {r['status']} · Dafny · round {rnd}/{max_rounds}\n"
    if r.get("violations"):
        head += "Rule violations (these never count as verified):\n" + "\n".join(f"- {x}" for x in r["violations"]) + "\n"
    if r["status"] == "CHECK_ERROR":
        return head + "This task cannot be checked; nothing in the solution can change that."
    if r.get("verify") and not r.get("verify_ok"):
        head += f"`dafny verify` output (tail):\n```\n{r['verify'][-2000:]}\n```\n"
    return head + ("Keep the preamble/spec/postamble byte-identical, add only helper declarations and the method body "
                   "`{ ... }` right after the spec, keep the code compilable, and make `dafny verify /app/solution.dfy` "
                   "report 0 errors honestly.")


def main():
    if len(sys.argv) > 3 and sys.argv[1] == "check":
        path, frozen = sys.argv[2], json.load(open(sys.argv[3]))
        r = check(open(path).read(), frozen, path)
        rnd = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        r["feedback"] = feedback(r, rnd, int(sys.argv[5]) if len(sys.argv) > 5 else 2)
        sys.stdout.write(json.dumps(r))
    elif len(sys.argv) == 3 and sys.argv[1] == "lint":
        sys.stdout.write(json.dumps({"problems": task_problems(json.load(open(sys.argv[2])))}))


if __name__ == "__main__":
    main()
