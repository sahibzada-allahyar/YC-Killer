"""Model-free curation of the FV pool (docs/research/win-plan-synthesis.md §2, "Exclusions" and "Vacuity battery", and
the TRIVIAL_THEOREMS decision fixed before Stage 0).

  .venv/bin/python -m bench.fv.curate --local --cells L-fv=20,D-np=10 [--part dev] [--jobs 8] [--missing]
  . scripts/env.sh && modal run bench/fv/curate.py [--part all] [--cells L-fv]          # full pool, cpu=1/task

Every task in the runs/fv/pool.json cells gets one status, decided by the Millennium Harness rungs themselves, in this
order:
  LOCK_FAIL  Lean: millennium_harness.lean.lock() refuses it with a real Lean or lock error (skeleton does not
             compile, `sorry` outside a body, axioms...). Dafny (after BROKEN): the frozen text declares an axiom the
             agent could call ({:axiom}, {:verify false}, {:extern}, @Axiom, @Verify, `assume`, or a bodiless
             lemma/function/method with `ensures` in the preamble or postamble); the rung only bans these in agent
             text.
  BROKEN     Lean: the starting file fails its own lock (banned text outside the holes), or compiling skeleton +
             postamble gives an error other than "aborting evaluation ... 'sorry'" (also as the generated side of a
             #guard_msgs diff): such a postamble can never pass. Warnings are ignored, as the rung ignores them.
             Dafny: the frozen preamble/spec/postamble do not resolve on their own (`dafny resolve` reports an Error).
  TRIVIAL_THEOREMS  Lean (FINDINGS, decided before Stage 0): the task counts only if at least one frozen theorem is not
             provable by the battery once every implementation def is replaced by a trivial body. With the postamble
             removed, a theorem is trivial if SOME implementation variant and SOME tactic prove it (no sorryAx, only
             propext/Classical.choice/Quot.sound, read with `#print axioms`); results combine per theorem across
             variants, so different theorems may need different variants. A variant gives every implementation def,
             independently, one of: `default`, `pure default`, or any one of its explicit arguments (only the choices
             that typecheck). Each (variant, tactic) pair is its own compile, so a runtime exception (maxRecDepth,
             heartbeats) in one tactic cannot hide another and every theorem gets the full heartbeat budget. Tactics:
             BATTERY (+ MVCGEN if Std.Do is imported). A task with no theorem hole is TRIVIAL_THEOREMS too.
             The plan's Lean VACUOUS (one trivial solution PROVED by the rung, postamble included) implies this
             status; such tasks say so in their reason (confirmed with millennium_harness.lean.check), so that count
             is kept.
  VACUOUS    Dafny: the body assigning each output its type's default (`*` when the type has no literal default: if
             any value verifies, the default does), `{ default }` for a function spec, verifies.
  OK         none of the above.
Infrastructure failures (wall-clock timeouts, crashes, OOM, a wrong toolchain) are never task statuses: the task is
printed as ERROR and left uncurated, and `--missing` retries exactly the uncurated ones.

Writes runs/fv/curated.json and runs/fv/frozen/<ID>.json (the Lean lock, read by bench/adapters/vericoding_fv.py):
  counts[part][cell]  pool, curated, complete, one count per status, ok_components (components with an OK id in
                      this cell), ok_components_multicell (of those, how many also have an OK id in another cell of
                      the part: the TEST draw gives such a component to one cell only), why (reason counts)
  ok_ids[part][cell]  OK ids in seeded split order. Consumers apply curation BEFORE the seeded draws:
                      curated_split() restricts split.json to these ids for select_ids()/pick_test().
  tasks[id]           cell, part, status, reason (content-free category), stamp; "detail" for DEV tasks only, so
                      TEST shows up as ids and counts.
curator[lean|dafny] records what made that language's entries: rules, battery, toolchains, sha256 of this file and of
that language's rung (lean.py + lean_check/Check.lean, or dafny.py), and their hash, the stamp every task carries. A
save that brings results of a language drops that language's entries under another stamp, with their frozen locks;
a save without results of a language keeps its entries and header as they are. So the entries of one language always
come from one battery and rung version, and a Dafny retry does not wipe Lean results made under an older Lean rung.
Saves take a file lock and replace the file atomically after every task, so concurrent runs merge.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from millennium_harness import dafny as sl_dafny  # noqa: E402
from millennium_harness import lean as sl_lean  # noqa: E402

DATA = ROOT / "runs" / "dafny" / "vericoding.jsonl"
POOL = ROOT / "runs" / "fv" / "pool.json"
SPLIT = ROOT / "runs" / "fv" / "split.json"
OUT = ROOT / "runs" / "fv" / "curated.json"
FROZEN_DIR = ROOT / "runs" / "fv" / "frozen"
LOCAL_DAFNY = Path.home() / ".local" / "dafny-4.11" / "dafny" / "dafny"
DAFNY_VERSION = "4.11.0"
RULES = "fv-curate-v2: TRIVIAL_THEOREMS per theorem; one compile per (variant, tactic); infra errors uncurated"
STATUSES = ("OK", "BROKEN", "VACUOUS", "TRIVIAL_THEOREMS", "LOCK_FAIL")
LEAN_TIMEOUT, VACUITY_TIMEOUT, DAFNY_TIMEOUT, DAFNY_VERIFY_TIMEOUT = 300, 300, 300, 600
MAX_VARIANTS = 64


class InfraError(RuntimeError):
    """The battery cannot decide this task here (a wall-clock timeout, a crash, a toolchain problem, a hole it cannot
    name): reported, and the task stays uncurated, never given a status."""


# ---------------------------------------------------------------------------------------------------- Lean
ABORT = "aborting evaluation since the expression depends on the 'sorry' axiom"
# Plan §2's proofs, plus simp_all/mvcgen that may unfold the implementation defs ({u} = ` [f, g]`): without that a
# trivial body is opaque to simp and mvcgen. Each one runs alone (`(t; done)`), never inside `first`.
BATTERY = ["rfl", "trivial", "omega", "simp_all", "intros; simp_all", "simp_all{u}", "intros; simp_all{u}", "decide"]
MVCGEN = ["mvcgen <;> simp_all", "mvcgen{u} <;> simp_all{u}"]  # only if Std.Do is imported
IMPL = {"default": "exact default", "pure default": "exact pure default", "passthrough": "exact {arg}"}
PROOF_DECL = re.compile(r"(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|nonrec)\s+)*(?:theorem|lemma|example)\b")
_HEAD = re.compile(r"(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|partial|nonrec|unsafe)\s+)*"
                   r"(def|abbrev|theorem|lemma|instance|example|opaque)\b[ \t]*([^\s(){}\[\]:⦃]*)")
AXIOMS_LINE = re.compile(r"(?m)^'(.+?)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)")
END_MARK, AX_MARK = "sl_curate_end_of_file", "sl_curate_axioms_"
NO_THEOREM = "no theorem hole"
NO_BODY = "exact default"  # what an implementation hole keeps when no trivial body typechecks (it then elaborates to
#                            sorry, so no theorem about it counts as trivial: conservative, and listed in the detail)
LOCK_REASONS = [("skeleton does not compile", "skeleton does not compile"),
                ("outside a declaration body", "sorry outside a declaration body"),
                ("`sorry` in vc-", "sorry in preamble/helpers/postamble"),
                ("task declares axioms", "task declares axioms"),
                ("inside the statement", "sorry inside a statement"),
                ("inside frozen", "sorry inside a frozen definition"),
                ("noncomputable", "noncomputable implementation"),
                ("cannot lex", "task text does not lex"),
                ("does not start with a declaration", "task text does not start with a declaration")]


def std_do(rec: dict) -> bool:
    mods = re.findall(r"(?m)^\s*import\s+(\S+)", sl_lean.mask(rec.get("vc-preamble") or "")[0])
    return any(m in ("Std", "Std.Do", "Std.Tactic.Do") or m.startswith("Std.Do.") for m in mods)


def tactics(rec: dict, impls: list[str]) -> list[str]:
    """The battery for this task, `{u}` filled with the implementation defs (deduplicated, in order)."""
    u = f" [{', '.join(impls)}]" if impls else ""
    return list(dict.fromkeys(t.format(u=u) for t in BATTERY + (MVCGEN if std_do(rec) else [])))


def _explicit_args(head: str) -> list[str]:
    """Names bound by the explicit `( ... )` binders of a masked declaration head, in order (`_` skipped)."""
    m = _HEAD.match(head)
    if not m:
        return []
    out, depth, start, opener = [], 0, 0, ""
    for k in range(m.end(), len(head)):
        ch = head[k]
        if depth == 0 and ch == ":":
            break
        if ch in "([{⦃":
            if depth == 0:
                start, opener = k, ch
            depth += 1
        elif ch in ")]}⦄":
            depth -= 1
            if depth == 0 and opener == "(" and ":" in head[start + 1:k]:
                names = head[start + 1:k].split(":", 1)[0].split()
                out += [n for n in names if n != "_" and re.fullmatch(r"[^\W\d][\w'!?]*", n)]
    return list(dict.fromkeys(out))


def _sorry_sites(text: str) -> list[tuple[int, int, str, bool]]:
    """(sorry offset, declaration offset, masked head up to the sorry, is a proof) per agent hole, in file order."""
    masked, out = sl_lean.mask(text)[0], []
    for a, b in sl_lean._decls(text):
        m = re.search(r"\bsorry\b", masked[a:b])
        if m:
            out.append((a + m.start(), a, masked[a:a + m.start()], bool(PROOF_DECL.match(masked, a))))
    return out


def lean_holes(rec: dict) -> list[dict]:
    """Every agent hole of vc-definitions + vc-theorems in file order: {kind: impl|proof, keyword, name, args}."""
    out = []
    for field in ("vc-definitions", "vc-theorems"):
        for _, _, head, proof in _sorry_sites((rec.get(field) or "").strip()):
            m = _HEAD.match(head)
            out.append({"kind": "proof" if proof else "impl", "keyword": m.group(1) if m else "",
                        "name": (m.group(2) if m else "") or None, "args": _explicit_args(head)})
    return out


def fill(rec: dict, impls, proofs, postamble: bool = True, rename_examples: bool = False) -> str:
    """The starting file with every agent `sorry` replaced: implementation holes by `impls`, theorem holes by `proofs`
    (each a tactic for every hole, or a list with one tactic per hole of that kind, in file order), wrapped in
    `(by ...)` in term position. rename_examples turns proof hole p's `example` into `theorem _sl_example_p`, so that
    `#print axioms` can read it. Only for records that lock (one `sorry` per body)."""
    tacs = {"impl": iter(impls) if isinstance(impls, list) else itertools.repeat(impls),
            "proof": iter(proofs) if isinstance(proofs, list) else itertools.repeat(proofs)}
    n_proof = 0

    def field(text: str) -> str:
        nonlocal n_proof
        masked, out, last = sl_lean.mask(text)[0], [], 0
        for s, a, head, proof in _sorry_sites(text):
            tac = next(tacs["proof" if proof else "impl"])
            if proof:
                ex = re.compile(r"(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|nonrec)\s+)*(example)\b")
                m = ex.match(masked, a)
                if rename_examples and m:
                    out += [text[last:m.start(1)], f"theorem _sl_example_{n_proof}"]
                    last = m.end(1)
                n_proof += 1
            out += [text[last:s], tac if re.search(r"\bby\s*$", head) else f"(by {tac})"]
            last = s + len("sorry")
        return "".join(out) + text[last:]

    fields = {k: field((rec.get(k) or "").strip()) for k in ("vc-definitions", "vc-theorems")}
    return sl_lean.skeleton({**rec, **fields, **({} if postamble else {"vc-postamble": ""})})


def _sorry_only(msg: dict) -> bool:
    """A message the sorry skeleton may produce: any warning or info, or an error that is an #eval abort because of
    sorry (also as the generated side of a #guard_msgs diff)."""
    text = msg["text"]
    if msg["severity"] != "error":
        return True
    if "Docstring on `#guard_msgs` does not match generated message" in text:
        gen = [ln[1:].strip() for ln in text.splitlines()[1:] if ln.startswith(("+", " "))]
        return any(ABORT in ln for ln in gen) and all(
            not ln or ABORT in ln or ln.startswith("To attempt to evaluate anyway") for ln in gen)
    return ABORT in text.splitlines()[0]


def _compile(text: str, lean: str, timeout: float) -> tuple[bool, str, bool]:
    with tempfile.TemporaryDirectory(prefix="sl-curate-") as d:
        return sl_lean._compile(text, d, lean, timeout)


def postamble_problems(rec: dict, lean: str, timeout: float | None = None) -> list[dict]:
    """Messages from compiling skeleton + postamble that are not `_sorry_only`. A timeout or a failure without any
    error message (crash, OOM kill) is an InfraError, never a BROKEN task."""
    timeout = timeout or LEAN_TIMEOUT
    ok, out, timed_out = _compile(sl_lean.skeleton(rec), lean, timeout)
    if timed_out:
        raise InfraError(f"skeleton+postamble: lean timed out after {timeout} s")
    msgs = sl_lean._messages(out)
    if not ok and not any(m["severity"] == "error" for m in msgs):
        raise InfraError(f"skeleton+postamble: lean failed without an error message: {out[-300:]!r}")
    return [m for m in msgs if not _sorry_only(m)]


def lean_axioms(text: str, names: list[str], lean: str, timeout: float) -> dict[str, set[str]] | None:
    """Compile `text` + `#print axioms` for each name (as written in the source: `«repeat»`, a private name...):
    {requested name: axioms} for the names that resolve; None on a wall-clock timeout. Each request is preceded by a
    `#print` marker, so its answer is read from its own slice of the output whatever name Lean prints (private names
    print mangled, escaped ones unescaped). Lean dying before the end of the file is an InfraError."""
    marks = [f"{AX_MARK}{k}" for k in range(len(names))] + [END_MARK]
    probe = text.rstrip() + "\n\n" + "".join(f'#print "{m}"\n#print axioms {n}\n' for m, n in zip(marks, names)) + \
        f'#print "{END_MARK}"\n'
    ok, out, timed_out = _compile(probe, lean, timeout)
    if timed_out:
        return None
    pos = [re.search(rf"(?m)^{m}$", out) for m in marks]
    if not all(pos):
        raise InfraError(f"lean stopped before the end of the file: {out[-300:]!r}")
    res = {}
    for k, n in enumerate(names):
        m = AXIOMS_LINE.search(out[pos[k].end():pos[k + 1].start()])
        if m:
            res[n] = {x.strip() for x in (m.group(2) or "").split(",") if x.strip()}
    return res


def _clean(ax: dict, name: str) -> bool:
    return name in ax and ax[name] <= sl_lean.STD_AXIOMS


def _escape(name: str) -> str:
    return ".".join(f"«{c}»" for c in name.split("."))


def hole_names(rec: dict, frozen: dict, lean: str) -> list[str | None]:
    """Per hole (lean_holes order), a name `#print axioms` resolves at the end of the file (postamble removed): the
    name as written, else the frozen constant's full name (a theorem declared in a namespace closed before the end),
    escaped. Examples get `_sl_example_p` (fill renames them). One compile. A named hole that resolves to nothing is
    an InfraError: an unreadable theorem would otherwise count as non-trivial (and its task as OK) unnoticed. None =
    an anonymous implementation hole (`instance : C := sorry`), which keeps `default` only."""
    holes, cands, p = lean_holes(rec), [], 0
    for h in holes:
        if h["kind"] == "proof" and h["keyword"] == "example":
            c = [f"_sl_example_{p}"]
        elif h["name"]:
            role = "theorem" if h["kind"] == "proof" else "impl"
            hits = [k for k, v in frozen["constants"].items()
                    if v["role"] == role and (k == h["name"] or k.endswith("." + h["name"].strip("«»")))]
            c = [h["name"]] + ([_escape(hits[0])] if len(hits) == 1 else [])
        else:
            c = []
        p += h["kind"] == "proof"
        cands.append(c)
    ax = lean_axioms(fill(rec, IMPL["default"], "sorry", postamble=False, rename_examples=True),
                     list(dict.fromkeys(n for c in cands for n in c)), lean, VACUITY_TIMEOUT)
    if ax is None:
        raise InfraError(f"name probe timed out after {VACUITY_TIMEOUT} s")
    out = [next((n for n in c if n in ax), None) for c in cands]
    bad = [c[0] for c, n in zip(cands, out) if c and n is None]
    if bad:
        raise InfraError(f"cannot resolve hole name(s) {bad} with #print axioms")
    return out


def impl_choices(rec: dict, names: list[str | None], lean: str) -> tuple[list[str], list[list[str]], list[str]]:
    """(implementation hole labels, per hole the trivial bodies that typecheck: default, pure default, each explicit
    argument; the labels of holes where none does). One probe compile per candidate position. `names` are the
    implementation holes' hole_names (None = anonymous: `default` only)."""
    impl = [h for h in lean_holes(rec) if h["kind"] == "impl"]
    cands = [[IMPL["default"], IMPL["pure default"]] + [IMPL["passthrough"].format(arg=a) for a in h["args"]]
             for h in impl]
    valid = [[] for _ in impl]
    for k in range(max(map(len, cands), default=0)):
        tacs = [c[k] if k < len(c) else IMPL["default"] for c in cands]
        ax = lean_axioms(fill(rec, tacs, "sorry", postamble=False), [n for n in names if n], lean, VACUITY_TIMEOUT)
        if ax is None:
            raise InfraError(f"implementation probe timed out after {VACUITY_TIMEOUT} s")
        for j, n in enumerate(names):
            if k < len(cands[j]) and n and _clean(ax, n):
                valid[j].append(cands[j][k])
    labels = [h["name"] or f"#{j}" for j, h in enumerate(impl)]
    return labels, [v or [NO_BODY] for v in valid], [n for n, v in zip(labels, valid) if not v]


def variants(choices: list[list[str]], cap: int = MAX_VARIANTS) -> tuple[list[tuple[str, ...]], bool]:
    """Every combination of per-def choices (all-default first). Over `cap`, the defs with the most choices lose their
    last ones until the product fits; the bool says whether that happened."""
    choices, capped = [list(c) for c in choices], False
    while len(choices) and _prod(len(c) for c in choices) > cap:
        j = max(range(len(choices)), key=lambda i: len(choices[i]))
        choices[j].pop()
        capped = True
    return list(itertools.product(*choices)), capped


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def _label(labels: list[str], v: tuple[str, ...]) -> str:
    def one(t):
        return "default" if t == IMPL["default"] else "pure default" if t == IMPL["pure default"] else \
            "passthrough " + t.removeprefix("exact ")
    return ", ".join(f"{n}={one(t)}" for n, t in zip(labels, v)) or "no implementation hole"


def lean_triviality(rec: dict, frozen: dict, lean: str) -> dict:
    """{trivial, vacuous, detail}: trivial = every theorem hole is proved by some (variant, tactic), postamble removed;
    vacuous = one variant proves every theorem AND millennium_harness.lean.check PROVES that solution with the
    postamble.
    Variants that pass the postamble's tests come first, so the first variant proving everything is a passing one
    whenever one exists."""
    proofs = [h for h in lean_holes(rec) if h["kind"] == "proof"]
    if not proofs:
        return {"trivial": True, "one_variant": True, "vacuous": None, "detail": NO_THEOREM}
    every = hole_names(rec, frozen, lean)
    kinds = [h["kind"] for h in lean_holes(rec)]
    names = [n for n, k in zip(every, kinds) if k == "proof"]
    impl_names = [n for n, k in zip(every, kinds) if k == "impl"]
    tacs = tactics(rec, [n for n in impl_names if n])
    labels, choices, untypable = impl_choices(rec, impl_names, lean)
    vs, capped = variants(choices)
    passes = {v: _passes_postamble(rec, v, lean) for v in vs}
    vs.sort(key=lambda v: not passes[v])
    won = [{} for _ in proofs]  # theorem -> {variant index: tactic}
    timeouts, full = set(), None
    for vi, v in enumerate(vs):
        for t in tacs:
            todo = [p for p in range(len(proofs)) if vi not in won[p]]
            if not todo:
                break
            for p, res in _try(rec, list(v), t, todo, [w.get(vi) for w in won], names, lean).items():
                if res:
                    won[p][vi] = t
                elif res is None:
                    timeouts.add(p)
        if all(vi in w for w in won):
            full = vi
            break
    lines = []
    for p, w in enumerate(won):
        first = next(iter(w.items()), None)
        lines.append(f"{names[p]}: " + (f"{first[1]} with {_label(labels, vs[first[0]])}" if first else "no trivial proof"))
    lines.append(f"{len(vs)} variant(s){' (capped)' if capped else ''} x {len(tacs)} tactics")
    lines += [f"{n}: no trivial body typechecks" for n in untypable]
    if not all(won) and any(not won[p] for p in timeouts):
        raise InfraError(f"vacuity battery timed out ({VACUITY_TIMEOUT} s) on a theorem with no trivial proof yet")
    vacuous = None
    if full is not None and passes[vs[full]]:
        text = fill(rec, list(vs[full]), [f"({won[p][full]}; done)" for p in range(len(proofs))])
        r = sl_lean.check(text, frozen, lean, timeout=VACUITY_TIMEOUT)
        if r["status"] == "CHECK_ERROR" or any("timed out" in x for x in r["violations"]):
            raise InfraError(f"VACUOUS confirmation: {r['status']} {r['violations'][:2]}")
        vacuous = r["status"] == "PROVED"
        lines.append(f"trivial solution ({_label(labels, vs[full])}) with the postamble: rung {r['status']}")
    return {"trivial": all(won), "one_variant": full is not None, "vacuous": vacuous, "detail": "\n".join(lines)}


def _passes_postamble(rec: dict, v: tuple[str, ...], lean: str) -> bool:
    if not (rec.get("vc-postamble") or "").strip():
        return True
    ok, out, timed_out = _compile(fill(rec, list(v), "sorry"), lean, VACUITY_TIMEOUT)
    if timed_out:
        raise InfraError(f"postamble run of a trivial variant timed out after {VACUITY_TIMEOUT} s")
    return not any(m["severity"] == "error" for m in sl_lean._messages(out))


def _try(rec, v, t, todo, winners, names, lean) -> dict[int, bool | None]:
    """Theorems in `todo` proved by tactic t under variant v (theorems already won under v keep their tactic, the rest
    of `todo` gets t too). After a wall-clock timeout each theorem is retried alone; None = still timed out."""
    def proofs(active):
        return [f"({t}; done)" if p in active else f"({w}; done)" if w else "sorry" for p, w in enumerate(winners)]

    ax = lean_axioms(fill(rec, v, proofs(set(todo)), postamble=False, rename_examples=True),
                     [names[p] for p in todo], lean, VACUITY_TIMEOUT)
    if ax is not None:
        return {p: _clean(ax, names[p]) for p in todo}
    out = {}
    for p in todo:
        ax = lean_axioms(fill(rec, v, proofs({p}), postamble=False, rename_examples=True), [names[p]], lean,
                         VACUITY_TIMEOUT)
        out[p] = None if ax is None else _clean(ax, names[p])
    return out


def curate_lean(rec: dict) -> dict:
    lean = sl_lean.find_lean()  # a missing or wrong toolchain raises: it is not a task status
    try:
        frozen = sl_lean.lock(rec, lean, LEAN_TIMEOUT)
    except sl_lean.LockError as e:
        if re.fullmatch(r"skeleton does not compile:\s*", str(e)):  # lean.lock() does not report its own timeout
            raise InfraError("lock: the skeleton compile timed out or crashed without an error message") from e
        why = next((r for k, r in LOCK_REASONS if k in str(e)), "lock refused")
        return {"status": "LOCK_FAIL", "reason": why, "detail": str(e)[-2000:]}
    # any other exception (Check.lean crash, OOM, timeout) propagates: an infrastructure error, not LOCK_FAIL
    hs, v = sl_lean.holes(sl_lean.skeleton(rec), frozen)
    v = v or [x for x in sl_lean.ban_scan(hs) if not x.startswith("banned construct: sorry ")]
    if v:
        return {"status": "BROKEN", "reason": "starting file fails its own lock", "detail": "\n".join(v)[-2000:],
                "frozen": frozen}
    bad = postamble_problems(rec, lean)
    if bad:
        return {"status": "BROKEN", "reason": "skeleton+postamble: error other than a sorry abort",
                "detail": "\n\n".join(m["text"] for m in bad)[-3000:], "frozen": frozen}
    t = lean_triviality(rec, frozen, lean)
    if not t["trivial"]:
        return {"status": "OK", "reason": "a theorem has no trivial proof", "detail": t["detail"], "frozen": frozen}
    why = ("no theorem to prove" if t["detail"] == NO_THEOREM else
           "every theorem trivial, under different variants" if not t["one_variant"] else
           "every theorem trivial under one variant, which passes the postamble (VACUOUS)" if t["vacuous"] else
           "every theorem trivial under one variant")
    return {"status": "TRIVIAL_THEOREMS", "reason": why, "detail": t["detail"], "frozen": frozen}


# ---------------------------------------------------------------------------------------------------- Dafny
DAFNY_DEFAULTS = {"int": "0", "nat": "0", "real": "0.0", "bool": "false", "char": "'D'", "string": '""',
                  "seq": "[]", "set": "{}", "iset": "iset{}", "multiset": "multiset{}", "map": "map[]", "imap": "imap[]"}
# Frozen-text declarations an agent could call to prove anything (the rung bans these only in agent text).
DAFNY_AXIOMS = {"{:axiom}": r"\{:\s*axiom\b", "{:verify false}": r"\{:\s*verify\s+false\b", "{:extern}": r"\{:\s*extern\b",
                "@Axiom": r"@\s*Axiom\b", "@Verify": r"@\s*Verify\b", "assume": r"\bassume\b"}


def find_dafny() -> str:
    """MILLENNIUM_DAFNY (or the older SORRYLESS_DAFNY), else ~/.local/dafny-4.11, else PATH; it must report version
    4.11.0 (as find_lean pins Lean)."""
    for c in (os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY")),
              str(LOCAL_DAFNY) if LOCAL_DAFNY.exists() else None, shutil.which("dafny")):
        if c:
            try:
                v = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=120).stdout.strip()
            except (OSError, subprocess.SubprocessError) as e:
                raise InfraError(f"{c} --version failed: {e}") from e
            if not v.startswith(DAFNY_VERSION):
                raise InfraError(f"{c} is not Dafny {DAFNY_VERSION}: {v[:80]!r}")
            return c
    raise InfraError(f"Dafny {DAFNY_VERSION} not found (set MILLENNIUM_DAFNY)")


@functools.lru_cache(maxsize=None)
def _dafny_ready(env_value: str | None) -> str:
    return find_dafny()


def _split_top(s: str) -> list[str]:
    """Split on commas outside (), <>, [] (arrows are not brackets)."""
    parts, depth, start = [], 0, 0
    for k, ch in enumerate(re.sub(r"-+>|~>", lambda m: " " * len(m.group(0)), s)):
        depth += (ch in "(<[") - (ch in ")>]")
        if ch == "," and depth == 0:
            parts, start = parts + [s[start:k]], k + 1
    return [p for p in parts + [s[start:]] if p.strip()]


def dafny_default(t: str) -> str | None:
    """A literal default value of a Dafny type, or None (user types, type parameters, function types...)."""
    t = t.strip()
    if t in DAFNY_DEFAULTS:
        return DAFNY_DEFAULTS[t]
    if re.fullmatch(r"bv\d+", t):
        return "0"
    m = re.fullmatch(r"(seq|set|iset|multiset|map|imap)\s*<.*>", t, re.S)
    if m:
        return DAFNY_DEFAULTS[m.group(1)]
    m = re.fullmatch(r"array\s*<(.*)>", t, re.S)
    if m:
        return f"new {m.group(1).strip()}[0]"
    if t.startswith("(") and t.endswith(")"):
        ds = [dafny_default(p) for p in _split_top(t[1:-1])]
        return None if None in ds else ds[0] if len(ds) == 1 else "(" + ", ".join(ds) + ")"
    return None


def _balanced(code: str, i: int) -> int:
    """Index just past the bracket group that opens at code[i]."""
    depth = 0
    for j in range(i, len(code)):
        depth += (code[j] == "(") - (code[j] == ")")
        if depth == 0:
            return j + 1
    return len(code)


def dafny_body(spec: str) -> str | None:
    """The default body for a frozen spec: outputs := defaults (`*` if unknown), `{ default }` for a function."""
    code = sl_dafny.strip_comments(spec)
    m = re.search(r"\b(method|lemma|function|predicate|constructor)\b", code)
    if not m:
        return None
    if m.group(1) in ("function", "predicate"):
        p = code.find("(", m.end())
        r = p >= 0 and re.match(r"\s*:\s*(.*?)\s*(?=\b(?:requires|ensures|reads|decreases)\b|$)",
                                code[_balanced(code, p):], re.S)
        typ = "bool" if m.group(1) == "predicate" else r.group(1) if r else ""
        named = re.fullmatch(r"\(\s*[\w'?]+\s*:(.*)\)", typ, re.S)  # function F(...): (r: T)
        d = dafny_default(named.group(1) if named else typ)
        return f"{{\n  {d}\n}}" if d else None
    r = re.search(r"\breturns\s*\(", code)
    outs = _split_top(code[r.end():_balanced(code, r.end() - 1) - 1]) if r else []
    lines = []
    for o in outs:
        name, _, typ = o.partition(":")
        lines.append(f"  {name.split()[-1]} := {dafny_default(typ) or '*'};")
    return "{\n" + "\n".join(lines) + ("\n" if lines else "") + "}"


def dafny_frozen(rec: dict) -> dict:
    return {"preamble": rec.get("vc-preamble") or "", "spec": rec["vc-spec"], "postamble": rec.get("vc-postamble") or ""}


def dafny_text(frozen: dict, body: str) -> str:
    """bench/adapters/vericoding_fv.py's starting file with `body` in place of the empty method body."""
    return (f"{frozen['preamble']}\n\n{sl_dafny.HELPERS_BEGIN}\n\n{sl_dafny.HELPERS_END}\n\n{frozen['spec']}\n{body}\n\n"
            f"{frozen['postamble']}\n")


def dafny_resolve_errors(frozen: dict, timeout: float | None = None) -> list[str]:
    """millennium_harness.dafny.task_problems() with its failure modes told apart: the `Error:` lines of `dafny
    resolve` on the frozen parts alone (bodiless spec); a timeout, or a failure without an `Error:` line, is an InfraError."""
    timeout = timeout or DAFNY_TIMEOUT
    skeleton = "\n\n".join(frozen.get(k, "").strip() for k in ("preamble", "spec", "postamble")) + "\n"
    with tempfile.TemporaryDirectory(prefix="sl-curate-") as tmp:
        Path(tmp, "frozen.dfy").write_text(skeleton, encoding="utf-8")
        rc, out = sl_dafny._dafny(["resolve", "--allow-warnings", "frozen.dfy"], tmp, timeout)
    if rc is None:
        raise InfraError(f"dafny resolve timed out after {timeout} s")
    errs = list(dict.fromkeys(re.findall(r"Error: (.+)", out)))
    if rc != 0 and not errs:
        raise InfraError(f"dafny resolve failed (rc {rc}) without an Error line: {out.strip()[-300:]!r}")
    return errs if rc != 0 else []


def dafny_frozen_axioms(frozen: dict) -> list[str]:
    found = []
    for part in ("preamble", "spec", "postamble"):
        code = sl_dafny.strip_comments(frozen.get(part) or "")
        found += [f"{name} in the frozen {part}" for name, pat in DAFNY_AXIOMS.items() if re.search(pat, code)]
        if part != "spec":  # the spec is bodiless on purpose: it is the agent's hole
            found += [f"bodiless {d} with ensures in the frozen {part}" for d in sl_dafny._bodiless_with_ensures(code)]
    return found


def curate_dafny(rec: dict) -> dict:
    os.environ["MILLENNIUM_DAFNY"] = _dafny_ready(os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY")))
    frozen = dafny_frozen(rec)
    errs = dafny_resolve_errors(frozen)
    if errs:
        return {"status": "BROKEN", "reason": "frozen parts do not resolve", "detail": " | ".join(errs[:3])[-2000:]}
    ax = dafny_frozen_axioms(frozen)
    if ax:
        return {"status": "LOCK_FAIL", "reason": "task declares axioms", "detail": "\n".join(ax)}
    body = dafny_body(frozen["spec"])
    if body is None:
        return {"status": "OK", "reason": "no default body for this spec"}
    r = sl_dafny.check(dafny_text(frozen, body), frozen, "solution.dfy", timeout=DAFNY_VERIFY_TIMEOUT)
    timed_out = re.search(r"timed out", (r.get("verify") or "") + " ".join(r.get("violations", [])), re.I)
    # a verification error that is not a time-out refutes the default body, whatever else timed out
    definite = [e for e in re.findall(r"Error: (.+)", r.get("verify") or "") if not re.search(r"timed out", e, re.I)]
    if r["status"] == "CHECK_ERROR":
        raise InfraError(f"Dafny vacuity check: CHECK_ERROR {r.get('violations', [])[:2]}")
    if r["status"] == "UNPROVED" and definite:
        return {"status": "OK", "reason": "default body UNPROVED"}
    if timed_out or r["status"] == "UNPROVED":  # only time-outs, or a failure without an Error line: undecided
        raise InfraError(f"Dafny vacuity check: {r['status']}{' (timed out)' if timed_out else ' without an Error'}")
    if r["status"] == "PROVED":
        return {"status": "VACUOUS", "reason": "default body PROVED", "detail": body}
    if r["status"] == "CHEAT_REJECTED" and r.get("verify_ok") and all(
            x.startswith(("not compilable", "audit:")) for x in r["violations"]):
        return {"status": "VACUOUS", "reason": "default body verifies (rejected only as not compilable/audit)",
                "detail": body + "\n" + "\n".join(r["violations"])}
    raise InfraError(f"Dafny vacuity check inconclusive: {r['status']} {r.get('violations', [])[:2]}")


# ---------------------------------------------------------------------------------------------------- driver
def curate(rec: dict) -> dict:
    """One task -> {id, status, reason[, detail][, frozen]}; {id, error} on an infrastructure failure."""
    try:
        r = curate_lean(rec) if rec["language"] == "lean" else curate_dafny(rec)
    except Exception as e:  # noqa: BLE001 - reported and left uncurated, never turned into a task status
        return {"id": rec["id"], "error": f"{type(e).__name__}: {str(e)[-500:]}"}
    return {"id": rec["id"], **r}


def select(split: dict, part: str, cells: str = "") -> list[str]:
    """Ids in seeded split order. cells: "" = every cell, or "L-fv=20,D-np" (=N keeps the first N per cell)."""
    want = dict((c.split("=") + [None])[:2] for c in cells.split(",") if c) or dict.fromkeys(split["dev"])
    ids = []
    for p in ("dev", "test") if part == "all" else (part,):
        for cell, n in want.items():
            got = [i for c in split[p][cell] for i in c["ids"]]
            ids += got[:int(n)] if n else got
    return ids


def load_records(ids, data: Path = DATA) -> list[dict]:
    want, recs = set(ids), {}
    with open(data) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                recs[r["id"]] = r
    return [recs[i] for i in ids]


def _is_ok(tasks: dict, i: str) -> bool:
    return tasks.get(i, {}).get("status") == "OK"


def counts(tasks: dict, split: dict) -> dict:
    out = {}
    for p in ("dev", "test"):
        ok_cells = defaultdict(set)  # component -> cells of this part where it has an OK id
        for cell, comps in split[p].items():
            for c in comps:
                if any(_is_ok(tasks, i) for i in c["ids"]):
                    ok_cells[c["comp"]].add(cell)
        out[p] = {}
        for cell, comps in split[p].items():
            ids = [i for c in comps for i in c["ids"]]
            st = Counter(tasks[i]["status"] for i in ids if i in tasks)
            okc = [c["comp"] for c in comps if cell in ok_cells[c["comp"]]]
            out[p][cell] = {"pool": len(ids), "curated": sum(st.values()), "complete": sum(st.values()) == len(ids),
                            **{s: st[s] for s in STATUSES}, "ok_components": len(okc),
                            "ok_components_multicell": sum(len(ok_cells[k]) > 1 for k in okc),
                            "why": dict(sorted(Counter(tasks[i]["reason"] for i in ids
                                                       if i in tasks and tasks[i]["status"] != "OK").items()))}
    return out


def ok_ids(tasks: dict, split: dict) -> dict:
    return {p: {cell: [i for c in comps for i in c["ids"] if _is_ok(tasks, i)] for cell, comps in split[p].items()}
            for p in ("dev", "test")}


def curated_split(split: dict, doc: dict, parts=("dev", "test")) -> dict:
    """split.json restricted to curated-OK ids in the same seeded order (components left with none are dropped), for
    bench.adapters.vericoding_fv.select_ids() and bench.fv.split.pick_test(): curation applies BEFORE the seeded draws,
    so DEV's ids[0] and TEST's per-component draws only ever see usable tasks. Raises if a cell of `parts` is not fully
    curated or the file mixes curator stamps."""
    if any(t.get("stamp") != doc["curator"][_lang(t["cell"])]["stamp"] for t in doc["tasks"].values()):
        raise ValueError("curated.json mixes curator stamps")
    out = {k: v for k, v in split.items() if k not in ("dev", "test", "dev_ids")}
    for p in ("dev", "test"):
        out[p] = {}
        for cell, comps in split[p].items():
            c = doc["counts"][p][cell]
            if p in parts and not c["complete"]:
                raise ValueError(f"{p} {cell}: only {c['curated']}/{c['pool']} tasks curated")
            out[p][cell] = [{**comp, "ids": ok} for comp in comps
                            if (ok := [i for i in comp["ids"] if _is_ok(doc["tasks"], i)])]
    out["dev_ids"] = {cell: [i for c in comps for i in c["ids"]] for cell, comps in out["dev"].items()}
    return out


def _lang(cell: str) -> str:
    return "lean" if cell.startswith("L-") else "dafny"


RUNG_SOURCES = {"lean": ("millennium_harness/lean.py", "millennium_harness/lean_check/Check.lean"),
                "dafny": ("millennium_harness/dafny.py",)}


def curator() -> dict:
    """{lean|dafny: {rules, toolchains, battery, sources_sha256 (this file + that language's rung), stamp}}."""
    srcs = {"bench/fv/curate.py": Path(__file__), "millennium_harness/lean.py": Path(sl_lean.__file__),
            "millennium_harness/lean_check/Check.lean": sl_lean.CHECK_LEAN,
            "millennium_harness/dafny.py": Path(sl_dafny.__file__)}
    base = {"rules": RULES, "toolchains": {"lean": sl_lean.TOOLCHAIN, "dafny": DAFNY_VERSION},
            "battery": {"impl": IMPL, "proof": BATTERY, "std_do": MVCGEN, "max_variants": MAX_VARIANTS}}
    out = {}
    for lang, files in RUNG_SOURCES.items():
        c = {**base, "sources_sha256": {f: hashlib.sha256(srcs[f].read_bytes()).hexdigest()
                                        for f in ("bench/fv/curate.py", *files)}}
        out[lang] = {**c, "stamp": hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]}
    return out


@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{path}.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def save(results: list[dict], out: Path = OUT, frozen_dir: Path = FROZEN_DIR, pool_path: Path = POOL,
         split_path: Path = SPLIT, cur: dict | None = None) -> dict:
    """Merge results by id under a file lock. For each language these results bring, entries under another curator
    stamp are dropped with their frozen locks; other languages keep their entries and header. A result without a lock
    (LOCK_FAIL, Dafny) removes any stale lock file of that id."""
    cur = cur or curator()
    split = json.loads(split_path.read_text())
    where = {i: (p, cell) for p in ("dev", "test") for cell, comps in split[p].items() for c in comps for i in c["ids"]}
    with _locked(out):
        old_doc = json.loads(out.read_text()) if out.exists() else {}
        old, old_cur = old_doc.get("tasks", {}), old_doc.get("curator", {})
        touched = {_lang(where[r["id"]][1]) for r in results}
        kept = {_lang(t["cell"]) for t in old.values()} - touched  # languages this save leaves as they are
        hdr = {lang: old_cur[lang] if lang in kept and isinstance(old_cur.get(lang), dict) else cur[lang]
               for lang in cur}
        tasks = {i: t for i, t in old.items() if t.get("stamp") == hdr[_lang(t["cell"])]["stamp"]}
        dropped = sorted(set(old) - set(tasks))
        for i in dropped:
            (frozen_dir / f"{i}.json").unlink(missing_ok=True)
        for r in results:
            f = frozen_dir / f"{r['id']}.json"
            if r.get("frozen"):
                frozen_dir.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(r["frozen"], indent=1, ensure_ascii=False))
            else:
                f.unlink(missing_ok=True)
            p, cell = where[r["id"]]
            tasks[r["id"]] = {"cell": cell, "part": p, "status": r["status"], "reason": r["reason"],
                              "stamp": hdr[_lang(cell)]["stamp"], **({"detail": r["detail"]} if p == "dev" and r.get("detail") else {})}
        doc = {"pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
               "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(), "curator": hdr,
               "counts": counts(tasks, split), "ok_ids": ok_ids(tasks, split), "tasks": dict(sorted(tasks.items()))}
        tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
    if dropped:
        print(f"dropped {len(dropped)} curated entries (and their locks) made under another curator stamp", flush=True)
    return doc


def run(ids: list[str], results) -> dict:
    """Consume (id, result) pairs from a local pool or a Modal map; print DEV lines only, save after every task, print
    counts."""
    split, cur = json.loads(SPLIT.read_text()), curator()
    dev = {i for comps in split["dev"].values() for c in comps for i in c["ids"]}
    done, errors, doc = 0, 0, None
    for i, r in results:
        if isinstance(r, BaseException) or "error" in r:
            errors += 1
            print(f"{i} ERROR {r if i in dev else '(TEST: hidden)'}", flush=True)
            continue
        done += 1
        doc = save([r], cur=cur)
        if i in dev:
            print(f"{i} {r['status']:16s} {r['reason']}", flush=True)
    doc = doc or save([], cur=cur)
    for p, cells in doc["counts"].items():
        for cell, c in cells.items():
            if c["curated"]:
                print(f"{p:4s} {cell:6s} " + " ".join(f"{k}={v}" for k, v in c.items() if k != "why") + f"  {c['why']}")
    print(f"{done} curated, {errors} errors (left uncurated; rerun with --missing) -> {OUT.relative_to(ROOT)}")
    if {k: v["stamp"] for k, v in curator().items()} != {k: v["stamp"] for k, v in cur.items()}:
        print("WARNING: curate.py or the rung sources changed during this run; its results carry the stamp of the "
              "start and the next save under the new sources drops them", flush=True)
    return doc


def missing(ids: list[str], out: Path = OUT) -> list[str]:
    """The ids not curated under the current curator stamp (errors, or never run)."""
    tasks = json.loads(out.read_text())["tasks"] if out.exists() else {}
    cur = curator()
    return [i for i in ids if i not in tasks or tasks[i].get("stamp") != cur[_lang(tasks[i]["cell"])]["stamp"]]


# ---------------------------------------------------------------------------------------------------- Modal
# The toolchains of bench/fv/images/{lean,dafny}.Dockerfile (elan installs no default toolchain eagerly any more).
ELAN = "ELAN_HOME=/opt/elan /opt/elan/bin/elan"
IMAGE_CMDS = [
    "curl -fsSL -o /tmp/elan-init.sh https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh",
    "ELAN_HOME=/opt/elan sh /tmp/elan-init.sh -y --no-modify-path --default-toolchain none && rm /tmp/elan-init.sh",
    f"{ELAN} toolchain install leanprover/lean4:v{sl_lean.TOOLCHAIN}",
    f"{ELAN} default leanprover/lean4:v{sl_lean.TOOLCHAIN}",
    f'L="$({ELAN} which lean)" && test -x "$L" && ln -s "$(dirname "$(dirname "$L")")" /opt/lean',
    f"/opt/lean/bin/lean --version | grep -F 'version {sl_lean.TOOLCHAIN},'",
    "curl -fsSL -o /tmp/d.zip "
    f"https://github.com/dafny-lang/dafny/releases/download/v{DAFNY_VERSION}/dafny-{DAFNY_VERSION}-x64-ubuntu-22.04.zip",
    f"unzip -q /tmp/d.zip -d /opt && rm /tmp/d.zip && /opt/dafny/dafny --version | grep -F '{DAFNY_VERSION}'"]
app = modal.App("millennium-harness-fv-curate")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("curl", "ca-certificates", "xz-utils", "unzip", "libicu72")
         .run_commands(*IMAGE_CMDS)
         .env({"MILLENNIUM_LEAN": "/opt/lean/bin/lean", "MILLENNIUM_DAFNY": "/opt/dafny/dafny",
               "DOTNET_CLI_TELEMETRY_OPTOUT": "1"})
         .add_local_python_source("millennium_harness", ignore=[]))  # ignore=[]: keep lean_check/Check.lean


@app.function(image=image, cpu=1, memory=4096, timeout=3600, max_containers=64)  # Check.lean peaks near 1.5 GB
def curate_remote(rec: dict) -> dict:
    return curate(rec)


@app.local_entrypoint()
def modal_main(part: str = "all", cells: str = "", only_missing: bool = False):
    ids = select(json.loads(SPLIT.read_text()), part, cells)
    ids = missing(ids) if only_missing else ids
    recs = load_records(ids)
    run(ids, zip(ids, curate_remote.map(recs, return_exceptions=True)))


def default_jobs() -> int:
    """One job per core, at most one per 2 GB of RAM (Check.lean peaks near 1.5 GB)."""
    try:
        gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        gb = 16
    return max(1, min(os.cpu_count() or 1, int(gb // 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", required=True, help="run here (the full pool runs on Modal)")
    ap.add_argument("--part", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--cells", default="", help='e.g. "L-fv=20,D-np=10" (first N per cell in seeded order)')
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--missing", action="store_true", help="only ids not curated under the current curator stamp")
    a = ap.parse_args()
    ids = select(json.loads(SPLIT.read_text()), a.part, a.cells)
    ids = missing(ids) if a.missing else ids
    recs = load_records(ids)
    if any(r["language"] == "dafny" for r in recs):
        os.environ["MILLENNIUM_DAFNY"] = find_dafny()  # fail fast on a wrong version; workers inherit it
    if any(r["language"] == "lean" for r in recs):
        sl_lean.find_lean()
    with multiprocessing.get_context("spawn").Pool(a.jobs) as pool:
        run(ids, zip(ids, pool.imap(curate, recs)))


if __name__ == "__main__":
    main()
