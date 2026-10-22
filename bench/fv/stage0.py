"""Stage 0 of Millennium Harness FV (docs/research/win-plan-synthesis.md §4): single-shot Batch probe plus
alpha-rename memorization probe on DEV tasks. No agent, no Harbor.

  .venv/bin/python bench/fv/stage0.py build              # -> runs/fv/stage0/{requests.jsonl,tasks.json}; no API call
  .venv/bin/python bench/fv/stage0.py submit             # Batch API (claude-haiku-5-5, effort medium, 16000 tokens)
  .venv/bin/python bench/fv/stage0.py collect [--modal]  # download answers once, grade with the rungs -> grades.jsonl
  .venv/bin/python bench/fv/stage0.py report             # CEILING / FLOOR-RISK / MEMO-SUSPECT per cell -> report.json

Tasks: per cell (L-fv, L-apps, D-np, D-apps; L-np if it has >= 5 DEV components) the first 20 DEV components in the
seeded order of runs/fv/split.json (first id of each, as bench.adapters.vericoding_fv.select_ids) whose
runs/fv/curated.json status is OK (plan §2/§4: VACUOUS, BROKEN, LOCK_FAIL, TRIVIAL_THEOREMS... are excluded before any
look, reasons recorded). build refuses while a candidate it would need has no curated status. Each task is asked
twice, as the original and as an alpha-renamed copy: every name the task declares (Lean
def/theorem/abbrev/structure/inductive/class/instance and inductive constructors; Dafny top-level
function/predicate/method/lemma/type/datatype/const and datatype constructors) becomes a neutral name (f1, P1 for Prop
defs/predicates, t1 theorems/lemmas, T1 types, c1 constructors, m1 Dafny methods, k1 consts), consistently in code;
never inside string or char literals; in Lean #guard_msgs expected-output docstrings as in code (outside quotes); in
other comments and the description only where a token reads as an identifier (in `code` or a ``` block, called as
`name(`, snake_case/digits/camelCase, part of a dotted name), so plain words such as "sum" or "product" keep their
meaning. Keywords, library names, parameters, structure/datatype fields and `main`/`Main` keep their names. Both forms
are locked with the current rung. A renamed copy is used only if it locks like the original (Lean: lock() passes, same
constant kinds/roles, every renamed declaration present, and skeleton + postamble give only sorry aborts, as
bench.fv.curate.postamble_problems; Dafny: the frozen parts resolve), otherwise only the original is asked.

The prompt is the agent instruction of bench/adapters/vericoding_fv.py (same rules text) with its last paragraph
replaced by a single-shot answer format, followed by the starting file the agent would get. The answer is the longest
fenced code block. Grades: the strict rung (millennium_harness.lean.check / millennium_harness.dafny.check), plus
`compiles` (Lean: 0 errors, `sorry` warnings allowed; Dafny: `dafny resolve` passes; never for SPEC_TAMPERED) for the
FLOOR-RISK rule.
An answer the grader could not finish is a GRADE_ERROR row (collect exits 1); report treats GRADE_ERROR and
NO_RESPONSE alike in both forms. tasks.json records the rung's sha256: submit and collect refuse after a rung change
(rerun build: model-free, and it refuses to change requests.jsonl once a batch was submitted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from bench.adapters.vericoding_fv import (DATA, FROZEN_DIR, dafny_files, instruction, lean_files,  # noqa: E402
                                         select_ids)
from bench.agents_fv import SOLUTION  # noqa: E402
from bench.fv.pool import CELLS  # noqa: E402
from bench.fv.split import OUT as SPLIT  # noqa: E402
from millennium_harness import dafny as sl_dafny  # noqa: E402
from millennium_harness import lean as sl_lean  # noqa: E402

OUT = ROOT / "runs" / "fv" / "stage0"
CURATED = ROOT / "runs" / "fv" / "curated.json"  # bench/fv/curate.py
REQUESTS, TASKS, BATCH = OUT / "requests.jsonl", OUT / "tasks.json", OUT / "batch.json"
ANSWERS, GRADES, REPORT = OUT / "answers.jsonl", OUT / "grades.jsonl", OUT / "report.json"
RUNGS = ("millennium_harness/lean.py", "millennium_harness/lean_check/Check.lean",
         "millennium_harness/dafny.py")  # what the locks depend on
STAGE0_CELLS, N, MIN_DEV = ("L-fv", "L-apps", "L-np", "D-np", "D-apps"), 20, 5
MODEL, EFFORT, MAX_TOKENS = "claude-haiku-5-5", "medium", 16000
CEILING, FLOOR_COMPILE, MEMO_PP, MEMO_TASKS = 0.80, 0.25, 0.20, 4  # plan §4 Stage 0
EXCLUDED = ("NO_RESPONSE", "GRADE_ERROR")  # infrastructure, not the model: excluded from both forms alike
MAX_INFRA = 0.10  # more than this share of a cell's tasks excluded -> INCOMPLETE
LOCAL_DAFNY = Path.home() / ".local" / "dafny-4.11" / "dafny" / "dafny"
FIELDS = {"lean": ("vc-preamble", "vc-helpers", "vc-definitions", "vc-theorems", "vc-postamble"),
          "dafny": ("vc-preamble", "vc-helpers", "vc-spec", "vc-code", "vc-postamble")}
FENCE = {"lean": "lean", "dafny": "dafny"}
KEEP = {"main", "Main"}

# ---------------------------------------------------------------------------------------------------- renaming
IDENT = {"lean": re.compile(r"«[^»\n]*»|(?<![\w'])[^\W\d][\w'!?]*"),
         "dafny": re.compile(rf"(?<![\w'?]){sl_dafny.IDENT}")}
_LEAN_DECL = re.compile(r"(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|partial|unsafe|nonrec)\s+)*"
                        r"(?:class\s+)?(def|theorem|lemma|abbrev|instance|structure|inductive|class|opaque|axiom)\s+"
                        r"(«[^»\n]*»|[^\s:(){}\[\]⦃⟨«]+)")
_LEAN_KIND = {"theorem": "t", "lemma": "t", "structure": "T", "inductive": "T", "class": "T"}
_DAFNY_DECL = re.compile(rf"\b({sl_dafny._DECL_KW})\s+({sl_dafny.IDENT})")
_DAFNY_KIND = {"predicate": "P", "copredicate": "P", "function": "f", "method": "m", "lemma": "t", "colemma": "t",
               "const": "k", "var": "k"}
_DAFNY_CTORS = re.compile(rf"\b(?:co)?datatype\s+{sl_dafny.IDENT}[^=]*=(.*?)(?=\b(?:{sl_dafny._DECL_KW})\b|\Z)", re.S)


def lean_names(rec: dict) -> tuple[dict[str, str], set[str]]:
    """{declared name (last component): kind prefix} in order, and the names that may follow a `.` (declared
    qualified, or inductive constructors)."""
    kinds, dotable = {}, set()
    for f in FIELDS["lean"]:
        masked = sl_lean.mask(rec.get(f) or "")[0]
        starts = [m.start() for m in re.finditer(r"(?m)^(?=\S)", masked)]  # top-level commands start in column 0
        for a, b in zip(starts, starts[1:] + [len(masked)]):
            decl = masked[a:b]
            m = _LEAN_DECL.match(decl)
            if not m:
                continue
            kw, (qual, _, name) = m.group(1), m.group(2).rpartition(".")
            if not name:
                continue
            if qual:
                dotable.add(name)
            head = re.split(r":=|\bwhere\b|\n\s*\|", decl, maxsplit=1)[0]
            kinds.setdefault(name, _LEAN_KIND.get(kw) or ("P" if re.search(r":\s*Prop\s*$", head) else "f"))
            if kw == "inductive":
                for c in re.findall(r"(?m)^\s*\|\s*([^\s:(){}\[\]]+)", decl):
                    kinds.setdefault(c, "c")
                    dotable.add(c)
    return kinds, dotable


def dafny_names(rec: dict) -> tuple[dict[str, str], set[str]]:
    """Top-level (bracket depth 0) declarations and datatype constructors; constructors may follow a `.`."""
    kinds, ctors = {}, set()
    for f in FIELDS["dafny"]:
        code = sl_dafny.strip_comments(rec.get(f) or "")
        depth = sl_dafny._depths(code)
        flat = "".join(c if depth[i] == 0 and c not in "([{}])" else " " for i, c in enumerate(code))
        for m in _DAFNY_DECL.finditer(flat):
            if not re.fullmatch(sl_dafny._DECL_KW, m.group(2)):  # `function method F`: F comes next
                kinds.setdefault(m.group(2), _DAFNY_KIND.get(m.group(1), "T"))
        for m in _DAFNY_CTORS.finditer(flat):
            for c in re.findall(rf"(?:^|\|)\s*(?:ghost\s+)?({sl_dafny.IDENT})", m.group(1).strip()):
                kinds.setdefault(c, "c")
                ctors.add(c)
    return kinds, ctors


_GUARD_DOC = re.compile(r"/--(?:(?!-/).)*-/(?=\s*#guard_msgs\b)", re.S)  # a #guard_msgs test's expected output
_CODE_SPAN = re.compile(r"```.*?```|`[^`\n]+`", re.S)  # ``` blocks and `inline code` in prose
_CODE_LIKE = re.compile(r"[_\d]|[a-z][A-Z][a-z]")  # snake_case, digits, camelCase (not NaN, HermiteE, Finite)


def _literal(masked: str, a: int, b: int) -> bool:
    """A blanked token whose nearest non-blank neighbours are quotes sits inside a string or char literal."""
    left, right = masked[:a].rstrip()[-1:], masked[b:].lstrip()[:1]
    return bool(left and right) and left in "\"'" and right in "\"'"


def _quoted(text: str, a: int, b: int) -> bool:
    """In a comment or description: inside "..." or “...” on the token's line, or directly in '...' (a string the
    code or an #eval output spells literally; renaming it would change its meaning)."""
    end = text.find("\n", b)
    before = re.sub(r"\\.", "", text[text.rfind("\n", 0, a) + 1:a])
    after = re.sub(r"\\.", "", text[b:end if end >= 0 else len(text)])
    return (before.count('"') % 2 == 1 and '"' in after or before.count("“") > before.count("”") and "”" in after
            or text[a - 1:a] == "'" == text[b:b + 1])


def _code_like(text: str, a: int, b: int, tok: str, spans: list[tuple[int, int]]) -> bool:
    """A prose token that reads as an identifier: in `code` or a ``` block, called as `tok(`, a «name», snake_case /
    digits / camelCase, or part of a dotted name (`tok.x`, `x.tok`). Plain words ("sum", "decode", "Finite") are not."""
    return (any(s <= a and b <= e for s, e in spans) or text[b:b + 1] == "(" or tok.startswith("«")
            or bool(_CODE_LIKE.search(tok)) or text[a - 1:a] == "."
            or text[b:b + 1] == "." and bool(re.match(r"[^\W\d]", text[b + 1:b + 2])))


def _sub(text: str, masked: str, names: dict, dotable: set, lang: str, prose: bool = False) -> str:
    """Rename `names` in one field. Code: every whole token (after a `.` only qualified names and constructors).
    String and char literals: never. Lean #guard_msgs expected-output docstrings: as code except inside quotes (#eval
    prints renamed constructors and #check renamed names, never a renamed string literal). Other comments and the
    description (prose=True, the whole text is prose): only _code_like tokens outside quotes."""
    spans = [m.span() for m in _CODE_SPAN.finditer(text)]
    expected = [] if prose or lang != "lean" else [
        m.span() for m in _GUARD_DOC.finditer(text)
        if not masked[m.start()].strip() and masked[m.end():].lstrip().startswith("#guard_msgs")]

    def repl(m):
        tok, a, b = m.group(0), m.start(), m.end()
        code, new, base = not prose and masked[a:b] == tok, names.get(tok), tok
        if new is None and not code:
            new = names.get(f"«{tok}»")  # a Lean «keyword» name mentioned in prose
        if new is None and lang == "dafny" and tok.endswith("?") and tok[:-1] in dotable and tok[:-1] in names:
            new, base = names[tok[:-1]] + "?", tok[:-1]  # datatype discriminator `x.Ctor?`
        if new is None or text[a - 1:a] == "." and base not in dotable:
            return tok  # after a `.` only qualified names and constructors resolve to the task's (not `bytes.decode`)
        if code:
            return new
        if not prose and _literal(masked, a, b) or _quoted(text, a, b):
            return tok  # string/char literal, or a quoted string in a comment / the description
        if any(s <= a < e for s, e in expected) or _code_like(text, a, b, tok, spans):
            return new
        return tok
    return IDENT[lang].sub(repl, text)


def rename(rec: dict, lang: str) -> tuple[dict, dict[str, str]]:
    """Alpha-renamed copy of a vericoding record and the {old: new} map."""
    kinds, dotable = (lean_names if lang == "lean" else dafny_names)(rec)
    taken = {t for f in (*FIELDS[lang], "vc-description") for t in IDENT[lang].findall(rec.get(f) or "")}
    names, count = {}, Counter()
    for old, p in kinds.items():
        if old in KEEP:
            continue
        while True:
            count[p] += 1
            if (new := f"{p}{count[p]}") not in taken:
                break
        names[old] = new
    out = dict(rec)
    for f in (f for f in FIELDS[lang] if f in rec):
        text = rec[f] or ""
        masked = sl_lean.mask(text)[0] if lang == "lean" else sl_dafny.strip_comments(text)
        out[f] = _sub(text, masked, names, dotable, lang)
    out["vc-description"] = _sub(rec.get("vc-description") or "", "", names, dotable, lang, prose=True)
    return out, names


# ---------------------------------------------------------------------------------------------------- tasks
def use_local_dafny() -> None:
    if not (os.environ.get("MILLENNIUM_DAFNY") or os.environ.get("SORRYLESS_DAFNY")) and LOCAL_DAFNY.exists():
        os.environ["MILLENNIUM_DAFNY"] = str(LOCAL_DAFNY)


def files(lang: str, rec: dict, frozen_dir: Path | None = None) -> tuple[str, dict]:
    """The adapter's starting file and frozen.json; raises LockError if the task cannot be locked. Lean locks with the
    local pinned Lean unless frozen_dir holds <ID>.json; Dafny's frozen parts must resolve on their own."""
    if lang == "dafny":
        sol, frozen = dafny_files(rec)
        if p := sl_dafny.task_problems(frozen):
            raise sl_lean.LockError("; ".join(p))
        return sol, frozen
    with tempfile.TemporaryDirectory(prefix="sl-s0-") as d:
        try:
            return lean_files(rec, frozen_dir if frozen_dir and (frozen_dir / f"{rec['id']}.json").exists() else d)
        except SystemExit as e:  # lean_files reports lock failures this way
            raise sl_lean.LockError(str(e)) from None


def postamble(lang: str, rec: dict) -> list[dict]:
    """Lean: messages from compiling skeleton + postamble other than sorry aborts (bench.fv.curate's BROKEN rule), e.g.
    a #guard_msgs test whose expected output no longer matches; Dafny: [] (task_problems covers the postamble)."""
    if lang != "lean":
        return []
    from bench.fv.curate import postamble_problems  # imports modal: build-time only, never in the grading image
    return postamble_problems(rec, sl_lean.find_lean())


def same_task(frozen: dict, frozen2: dict, names: dict) -> None:
    """Lean: the renamed lock has the same constants (kinds/roles) and holes, and every renamed declaration."""
    if "constants" not in frozen:
        return
    kr = lambda f: Counter((c["kind"], c["role"]) for c in f["constants"].values())  # noqa: E731
    holes = lambda f: [c["hole_after"] for c in f["chunks"]]  # noqa: E731
    missing = [new for old, new in names.items() if old in frozen["constants"] and new not in frozen2["constants"]]
    if kr(frozen) != kr(frozen2) or holes(frozen) != holes(frozen2) or missing:
        raise sl_lean.LockError(f"renamed lock differs from the original (missing {missing[:5]})")


def prompt(lang: str, description: str, skeleton: str) -> str:
    """The agent instruction (same rules text) with a single-shot answer format instead of its last paragraph."""
    head, sep, _ = instruction(lang, description).partition("\nYou run autonomously")
    assert sep, "the adapter instruction changed: update stage0.prompt"
    name, fence = SOLUTION[lang], FENCE[lang]
    return (f"{head}\nThis is a single reply: you cannot run tools or the toolchain. Reply with the complete file "
            f"{name} in one ```{fence} code block; that block is your answer.\n\nCurrent {name}:\n```{fence}\n"
            f"{skeleton}```\n")


def request(custom_id: str, text: str) -> dict:
    return {"custom_id": custom_id, "params": {"model": MODEL, "max_tokens": MAX_TOKENS,
                                               "output_config": {"effort": EFFORT},
                                               "messages": [{"role": "user", "content": text}]}}


def _problems(msgs: list[dict]) -> str:
    return "; ".join(m["text"].strip().replace("\n", " ")[:200] for m in msgs[:3])


def prepare(cell: str, recs: dict, ids: list[str], curated: dict, n: int = N, workers: int = 8) -> dict:
    """The first n ids of `ids` (seeded DEV order) that `curated` (runs/fv/curated.json "tasks") marks OK and that lock
    with the current rung, with their original and, if it locks like the original, renamed forms. Any other status is
    skipped with its reason; an id `curated` lacks is listed under "uncurated" and holds its slot (the walk stops once
    done + uncurated reach n, so the list names the next ids to curate; build refuses while it is not empty).
    "frozen_dir_differs": Lean ids whose runs/fv/frozen lock (made at curation) differs from the current lock."""
    lang = CELLS[cell][0]

    def one(i):
        st = curated.get(i)
        if st is None:
            return {"id": i, "uncurated": True}
        if st.get("status") != "OK":
            return {"id": i, "skip": f"curated {st.get('status')}: {st.get('reason', '')}"[:300]}
        r = recs[i]
        try:
            sol, frozen = files(lang, r)
            if p := postamble(lang, r):
                raise sl_lean.LockError(f"skeleton + postamble: {_problems(p)}")
        except sl_lean.LockError as e:
            return {"id": i, "skip": f"original does not lock: {str(e)[:300]}"}
        f = FROZEN_DIR / f"{i}.json"
        stale = lang == "lean" and f.exists() and json.loads(f.read_text()) != json.loads(json.dumps(frozen))
        forms = {"orig": {"rec": r, "sol": sol, "frozen": frozen, "names": {}}}
        r2, names = rename(r, lang)
        try:
            sol2, frozen2 = files(lang, r2)
            same_task(frozen, frozen2, names)
            if p := postamble(lang, r2):
                raise sl_lean.LockError(f"skeleton + postamble: {_problems(p)}")
            forms["ren"] = {"rec": r2, "sol": sol2, "frozen": frozen2, "names": names}
            err = None
        except sl_lean.LockError as e:
            err = f"renamed does not lock: {str(e)[:300]}"
        return {"id": i, "forms": forms, "ren_error": err, "names": names, "stale": stale}

    done, skipped, uncurated, pos = [], {}, [], 0
    with ThreadPoolExecutor(workers) as ex:
        while (k := n - len(done) - len(uncurated)) > 0 and pos < len(ids):
            batch, pos = ids[pos:pos + k], pos + k
            for res in ex.map(one, batch):
                if res.get("uncurated"):
                    uncurated.append(res["id"])
                elif "skip" in res:
                    skipped[res["id"]] = res["skip"]
                else:
                    done.append(res)
    ok = sum("ren" in t["forms"] for t in done)
    return {"cell": cell, "lang": lang, "tasks": done, "skipped": skipped, "uncurated": uncurated, "renamed_ok": ok,
            "renamed_rate": ok / len(done) if done else 0.0,
            "renamed_failed": {t["id"]: t["ren_error"] for t in done if t["ren_error"]},
            "frozen_dir_differs": [t["id"] for t in done if t["stale"]]}


def load_records(ids: set[str], data: Path = DATA) -> dict:
    recs = {}
    with open(data) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in ids:
                recs[r["id"]] = r
    return recs


def stage0_cells(split: dict) -> list[str]:
    return [c for c in STAGE0_CELLS if c != "L-np" or len(split["dev"][c]) >= MIN_DEV]


def rung_hashes() -> dict[str, str]:
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in RUNGS}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_curated(split_sha: str) -> tuple[dict, str]:
    """runs/fv/curated.json "tasks" and the file's sha256; refuses a missing file or one made for another split."""
    if not CURATED.exists():
        sys.exit(f"refusing: {CURATED} is missing: curate the DEV candidates first (bench/fv/curate.py)")
    raw = CURATED.read_bytes()
    doc = json.loads(raw)
    if doc.get("split_sha256") != split_sha:
        sys.exit(f"refusing: {CURATED} was made for another split.json (rerun bench/fv/curate.py)")
    return doc["tasks"], hashlib.sha256(raw).hexdigest()


def build(n: int = N) -> dict:
    """Model-free. Refuses (writing nothing) while a needed candidate is uncurated, and, once a batch was submitted,
    whenever requests.jsonl would change (a rebuild after a rung change then only refreshes the locks)."""
    use_local_dafny()
    rungs = rung_hashes()
    raw = SPLIT.read_bytes()
    split = json.loads(raw)
    curated, curated_sha = load_curated(hashlib.sha256(raw).hexdigest())
    cells = stage0_cells(split)
    cand = {c: select_ids(split, c, "dev", len(split["dev"][c])) for c in cells}
    recs = load_records({i for ids in cand.values() for i in ids}, DATA)
    items, reqs, summary, uncurated = {}, [], {}, {}
    for cell in cells:
        p = prepare(cell, recs, cand[cell], curated, n)
        if p["uncurated"]:
            uncurated[cell] = p["uncurated"]
        for t in p["tasks"]:
            for form, x in t["forms"].items():
                cid = f"s0__{cell}__{t['id']}__{form}"
                text = prompt(p["lang"], x["rec"].get("vc-description") or "", x["sol"])
                reqs.append(request(cid, text))
                items[cid] = {"cell": cell, "id": t["id"], "form": form, "lang": p["lang"], "frozen": x["frozen"],
                              "names": x["names"], "prompt_chars": len(text)}
        summary[cell] = {"ids": [t["id"] for t in p["tasks"]], "skipped": p["skipped"],
                         "renamed_ok": p["renamed_ok"], "renamed_rate": round(p["renamed_rate"], 3),
                         "renamed_failed": p["renamed_failed"], "frozen_dir_differs": p["frozen_dir_differs"]}
        print(f"{cell:7s} {len(p['tasks']):2d} tasks (skipped {len(p['skipped'])}, uncurated {len(p['uncurated'])}), "
              f"renamed lock {p['renamed_ok']}/{len(p['tasks'])} = {p['renamed_rate']:.0%}"
              + ("" if p["renamed_rate"] >= 0.9 else "  < 90%: fix the renamer before submitting")
              + (f"  ({len(p['frozen_dir_differs'])} runs/fv/frozen locks differ from the current rung: "
                 "curation predates it)" if p["frozen_dir_differs"] else ""), flush=True)
    if uncurated:
        sys.exit("refusing: no curated.json status for these candidates (curate them first; any status but OK is "
                 f"excluded): {json.dumps(uncurated)}")
    if rung_hashes() != rungs:
        sys.exit(f"refusing: {', '.join(RUNGS)} changed during build: rerun it")
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs)
    if BATCH.exists() and (not REQUESTS.exists() or REQUESTS.read_text() != body):
        sys.exit(f"refusing: a batch was submitted ({BATCH}) from a different requests.jsonl; this build would change "
                 "the requests, so its answers would not match")
    OUT.mkdir(parents=True, exist_ok=True)
    REQUESTS.write_text(body)
    TASKS.write_text(json.dumps({"model": MODEL, "effort": EFFORT, "max_tokens": MAX_TOKENS,
                                 "split_sha256": hashlib.sha256(raw).hexdigest(), "curated_sha256": curated_sha,
                                 "rungs": rungs, "cells": summary, "items": items}, ensure_ascii=False))
    print(f"{len(reqs)} requests -> {REQUESTS} (not submitted)")
    return summary


def same_rungs(tasks: dict) -> None:
    """The stored locks were made by this rung; else every answer could grade SPEC_TAMPERED against a stale lock."""
    if tasks.get("rungs") != rung_hashes():
        sys.exit(f"refusing: {', '.join(RUNGS)} changed since build (or tasks.json predates rung hashes), so its "
                 "locks may be stale: rerun build (model-free; with a submitted batch it keeps requests.jsonl or "
                 "refuses)")


# ---------------------------------------------------------------------------------------------------- batch
def submit() -> None:
    """Send requests.jsonl as one Batch (refuses a second submit, a build that is not current, and an estimate above
    STAGE0_MAX_USD, default $1)."""
    if BATCH.exists():
        sys.exit(f"refusing: already submitted ({BATCH.read_text().strip()})")
    tasks = json.loads(TASKS.read_text())
    same_rungs(tasks)
    if not CURATED.exists() or tasks.get("curated_sha256") != _sha(CURATED):
        sys.exit(f"refusing: {CURATED} changed since build (or the build predates curation): rerun build")
    import anthropic
    reqs = [json.loads(line) for line in REQUESTS.read_text().splitlines()]
    est = sum(len(r["params"]["messages"][0]["content"]) / 3 * 0.10 + 8000 * 0.50 for r in reqs) / 1e6 * 0.5
    print(f"{len(reqs)} requests; estimate ${est:.2f} (8K output tokens each, batch price)")
    if est > float(os.environ.get("STAGE0_MAX_USD", 1.0)):
        sys.exit("refusing: estimate above STAGE0_MAX_USD")
    b = anthropic.Anthropic().messages.batches.create(requests=reqs)
    BATCH.write_text(json.dumps({"id": b.id, "n": len(reqs), "t": time.time()}))
    print("batch", b.id, b.processing_status)


def download() -> None:
    """Wait for the batch, then write answers.jsonl (once) and the cost to runs/ledger.jsonl."""
    import anthropic
    from millennium_harness.llm import cost, text_of
    bid, c = json.loads(BATCH.read_text())["id"], anthropic.Anthropic()
    while (b := c.messages.batches.retrieve(bid)).processing_status != "ended":
        print(time.strftime("%H:%M:%S"), b.request_counts, flush=True)
        time.sleep(60)
    rows, usd = [], 0.0
    for r in c.messages.batches.results(bid):
        row = {"custom_id": r.custom_id, "result": r.result.type}
        if r.result.type == "succeeded":
            m = r.result.message
            u = m.usage.model_dump()
            usd += cost(u, batch=True)
            row.update(text=text_of(m.content), stop=m.stop_reason, tokens_in=u.get("input_tokens"),
                       tokens_out=u.get("output_tokens"))
        rows.append(row)
    ANSWERS.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    with open(ROOT / "runs" / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "item": "fv-stage0", "usd": round(usd, 4)}) + "\n")
    print(f"{len(rows)} answers ({sum(x['result'] == 'succeeded' for x in rows)} succeeded); cost ${usd:.3f}")


def extract(text: str | None) -> str:
    """The longest fenced code block (the whole file), or '' if there is none."""
    blocks = re.findall(r"```[^\n`]*\n(.*?)```", text or "", re.S)
    return max(blocks, key=len) if blocks else ""


def compiles(lang: str, text: str) -> bool:
    """Lean: 0 errors (`sorry` warnings allowed). Dafny: `dafny resolve` passes (parse + types, no verification)."""
    with tempfile.TemporaryDirectory(prefix="sl-s0-") as d:
        if lang == "lean":
            ok, out, _ = sl_lean._compile(text, d, sl_lean.find_lean(), 300)
            return ok and not any(m["severity"] == "error" for m in sl_lean._messages(out))
        Path(d, "solution.dfy").write_text(text, encoding="utf-8")
        return sl_dafny._dafny(["resolve", "--allow-warnings", "solution.dfy"], d, 120)[0] == 0


def grade(job: dict) -> dict:
    """job = {custom_id, lang, text, frozen} -> the strict rung's verdict plus `compiles`. A SPEC_TAMPERED answer never
    compiles: deleting or rewriting the theorems (Lean) or editing the spec (Dafny) would otherwise count."""
    text, lang = job["text"], job["lang"]
    if not text.strip():
        return {"custom_id": job["custom_id"], "status": "NO_CODE", "violations": [], "compiles": False}
    try:
        r = (sl_lean.check(text, job["frozen"]) if lang == "lean"
             else sl_dafny.check(text, job["frozen"], "solution.dfy"))
        ok = r["status"] == "PROVED" or r["status"] != "SPEC_TAMPERED" and compiles(lang, text)
    except Exception as e:  # noqa: BLE001 - recorded, audited by hand
        r, ok = {"status": "CHECK_ERROR", "violations": [f"{type(e).__name__}: {str(e)[:300]}"]}, False
    tail = r.get("errors_tail") or r.get("verify") or ""
    return {"custom_id": job["custom_id"], "status": r["status"], "violations": r.get("violations", []),
            "compiles": ok, "secs": r.get("secs"), "tail": tail[-1500:]}


try:
    import modal
except ImportError:  # unit tests and local grading need no Modal
    modal = None

if modal is not None:
    app = modal.App("millennium-harness-fv-stage0")
    _LEAN_BIN = "/root/.elan/toolchains/leanprover--lean4---v4.23.0-rc2/bin/lean"
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ca-certificates", "curl", "unzip", "libicu72")
             .run_commands("curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh "
                           "| sh -s -- -y --no-modify-path --default-toolchain leanprover/lean4:v4.23.0-rc2",
                           f"{_LEAN_BIN} --version",
                           "curl -fsSL -o /tmp/d.zip https://github.com/dafny-lang/dafny/releases/download/v4.11.0/"
                           "dafny-4.11.0-x64-ubuntu-22.04.zip && unzip -q /tmp/d.zip -d /opt && rm /tmp/d.zip "
                           "&& /opt/dafny/dafny --version")
             .env({"MILLENNIUM_LEAN": _LEAN_BIN, "MILLENNIUM_DAFNY": "/opt/dafny/dafny",
                   "DOTNET_CLI_TELEMETRY_OPTOUT": "1"})
             .add_local_python_source("bench")
             .add_local_python_source("millennium_harness", ignore=lambda p: p.suffix == ".pyc"))  # keeps lean_check/Check.lean

    @app.function(image=image, cpu=1, memory=4096, timeout=1800, max_containers=32)
    def grade_remote(job: dict) -> dict:
        return grade(job)


def _grade_error(custom_id: str, why: str) -> dict:
    return {"custom_id": custom_id, "status": "GRADE_ERROR", "violations": [why[:500]], "compiles": False}


def _grade_safe(job: dict) -> dict:
    try:
        return grade(job)
    except Exception as e:  # noqa: BLE001 - becomes a GRADE_ERROR row, never a model failure
        return _grade_error(job["custom_id"], f"{type(e).__name__}: {str(e)[:300]}")


def collect(use_modal: bool = False, workers: int = 4) -> int:
    """Download the answers (first call only), then grade every answer; rerun after a checker fix to regrade. Every
    item gets a row: an answer the grader did not finish (Modal call failed, grader crashed) is GRADE_ERROR. Returns
    the number of GRADE_ERROR rows (main exits 1 if any: rerun collect)."""
    tasks = json.loads(TASKS.read_text())
    same_rungs(tasks)
    if not ANSWERS.exists():
        download()
    items = tasks["items"]
    answers = {a["custom_id"]: a for a in map(json.loads, ANSWERS.read_text().splitlines())}
    jobs = [{"custom_id": cid, "lang": it["lang"], "text": extract(answers.get(cid, {}).get("text")),
             "frozen": it["frozen"]} for cid, it in items.items()]
    errors = []
    if use_modal:
        if modal is None:
            sys.exit("modal is not installed")
        res = []
        with app.run():
            for r in grade_remote.map(jobs, return_exceptions=True, order_outputs=False):
                if isinstance(r, BaseException):
                    errors.append(repr(r)[:200])  # unordered outputs: the job is found below by its missing row
                    print("ERR", errors[-1])
                else:
                    res.append(r)
    else:
        use_local_dafny()
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(_grade_safe, jobs))
    got = {r["custom_id"] for r in res}
    why = "not graded: " + ("; ".join(dict.fromkeys(errors)) or "no result")
    res += [_grade_error(j["custom_id"], why) for j in jobs if j["custom_id"] not in got]
    rows = []
    for r in res:
        it, a = items[r["custom_id"]], answers.get(r["custom_id"], {})
        if a.get("result") != "succeeded":
            r = {**r, "status": "NO_RESPONSE"}
        rows.append({"cell": it["cell"], "id": it["id"], "form": it["form"], "stop": a.get("stop"),
                     "tokens_out": a.get("tokens_out"), **r})
    GRADES.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    bad = sum(x["status"] == "GRADE_ERROR" for x in rows)
    print(f"graded {len(rows) - bad}/{len(jobs)}: {dict(Counter(x['status'] for x in rows))} -> {GRADES}")
    if bad:
        print(f"{bad} answers not graded (GRADE_ERROR): rerun collect; report excludes them from both forms")
    return bad


# ---------------------------------------------------------------------------------------------------- report
def decide(ids: list[str], grades: dict, ren_ids: set[str]) -> dict:
    """Plan §4 rules for one cell. grades = {(id, form): row}; ren_ids = the tasks that have a renamed form
    (tasks.json items). Every expected row must exist (ValueError otherwise: rerun collect). NO_RESPONSE (the batch
    request got no answer) and GRADE_ERROR (the grader did not finish) are infrastructure exclusions, treated alike in
    both forms: original rates are over the tasks whose original was graded, the rename comparison over the tasks whose
    two forms were both graded. NO_CODE is the model's failure. INCOMPLETE: more than MAX_INFRA of the tasks lost a
    form to an exclusion (the rules then rest on fewer tasks than planned)."""
    missing = [(i, f) for i in ids for f in ("orig", "ren") if (f == "orig" or i in ren_ids) and (i, f) not in grades]
    if missing:
        raise ValueError(f"grades.jsonl lacks {len(missing)} expected rows, e.g. {missing[:3]}: rerun collect")
    proved = lambda i, f: grades[i, f]["status"] == "PROVED"  # noqa: E731
    graded = lambda i, f: grades[i, f]["status"] not in EXCLUDED  # noqa: E731
    orig = [i for i in ids if graded(i, "orig")]
    n = len(orig)
    po = sum(proved(i, "orig") for i in orig)
    comp = sum(bool(grades[i, "orig"].get("compiles")) for i in orig)
    pairs = [i for i in orig if i in ren_ids and graded(i, "ren")]
    d = sum(proved(i, "orig") for i in pairs) - sum(proved(i, "ren") for i in pairs)
    lost = [i for i in ids if not graded(i, "orig") or i in ren_ids and not graded(i, "ren")]
    out = {"n": n, "orig_proved": po, "orig_rate": po / n if n else 0.0, "compile_rate": comp / n if n else 0.0,
           "pairs": len(pairs), "ren_proved": sum(proved(i, "ren") for i in pairs),
           "orig_minus_ren": d, "orig_minus_ren_pp": d / len(pairs) if pairs else 0.0,
           "orig_only": [i for i in pairs if proved(i, "orig") and not proved(i, "ren")],
           "ren_only": [i for i in pairs if proved(i, "ren") and not proved(i, "orig")], "excluded": lost}
    out["flags"] = [f for f, on in (
        ("CEILING", n and out["orig_rate"] >= CEILING),
        ("FLOOR-RISK", n and po == 0 and out["compile_rate"] < FLOOR_COMPILE),
        ("MEMO-SUSPECT", pairs and out["orig_minus_ren_pp"] >= MEMO_PP and d >= MEMO_TASKS),
        ("INCOMPLETE", len(lost) > MAX_INFRA * len(ids))) if on]
    return out


def report() -> dict:
    tasks = json.loads(TASKS.read_text())
    rows = [json.loads(line) for line in GRADES.read_text().splitlines()]
    out = {}
    for cell, s in tasks["cells"].items():
        g = {(r["id"], r["form"]): r for r in rows if r["cell"] == cell}
        ren = {it["id"] for it in tasks["items"].values() if it["cell"] == cell and it["form"] == "ren"}
        out[cell] = decide(s["ids"], g, ren)
        out[cell]["audit"] = [{"id": r["id"], "form": r["form"], "status": r["status"],
                               "violations": r["violations"][:3]} for r in g.values()
                              if r["status"] in ("CHEAT_REJECTED", "SPEC_TAMPERED", "CHECK_ERROR", "GRADE_ERROR")]
        out[cell]["no_response"] = sum(r["status"] in ("NO_RESPONSE", "NO_CODE") for r in g.values())
        out[cell]["grade_errors"] = sum(r["status"] == "GRADE_ERROR" for r in g.values())
        out[cell]["length_stops"] = sum(r.get("stop") == "max_tokens" for r in g.values())
        out[cell]["renamed_lock_rate"] = s.get("renamed_rate")
        c = out[cell]
        print(f"{cell:7s} orig {c['orig_proved']:2d}/{c['n']} ({c['orig_rate']:.0%}) compile {c['compile_rate']:.0%} "
              f"| renamed {c['ren_proved']}/{c['pairs']} (orig-ren {c['orig_minus_ren_pp']:+.0%}, "
              f"{c['orig_minus_ren']:+d}) | no answer {c['no_response']}, grade errors {c['grade_errors']}, "
              f"length stops {c['length_stops']}, to audit {len(c['audit'])} | {' '.join(c['flags']) or 'ok'}")
    if any("INCOMPLETE" in c["flags"] for c in out.values()):
        print("INCOMPLETE: over 10% of a cell's tasks lack a graded form; rerun collect (GRADE_ERROR) or resubmit the "
              "missing requests (NO_RESPONSE) before deciding.")
    if out and all({"CEILING", "MEMO-SUSPECT"} & set(c["flags"]) for c in out.values()):
        print("K6: every cell is CEILING or MEMO-SUSPECT -> STOP")
    print("Audit every CHEAT_REJECTED / SPEC_TAMPERED / CHECK_ERROR by hand (plan §4: 0 false rejections).")
    REPORT.write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "submit", "collect", "report"])
    ap.add_argument("--n", type=int, default=N, help="build: tasks per cell")
    ap.add_argument("--modal", action="store_true", help="collect: grade with the Modal map instead of locally")
    ap.add_argument("--workers", type=int, default=4, help="collect: local grading threads")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.n)
    elif a.cmd == "submit":
        submit()
    elif a.cmd == "collect":
        sys.exit(1 if collect(a.modal, a.workers) else 0)
    else:
        report()


if __name__ == "__main__":
    main()
