r"""Compiled audit of vericoding's published 'vericoded' (LLM-verified) solutions: for every Lean and Dafny solution,
vericoding's own acceptance criterion (BASELINE), the plain tool criterion (PLAIN) and the Millennium Harness rung
(MILLENNIUM), on the same file. Verus has the text audit only (bench/audit/vericoded_text.py). No LLM calls.

  python bench/audit/vericoded_rung.py run [--langs lean,dafny] [--lean-tasks N] [--only ID,...] [--missing]  # Modal
  (. scripts/env.sh first; env VC_LOCK_CACHE=<jsonl> reuses Lean locks across runs, VC_MAX_CONTAINERS caps containers.
   The 2026-10-11 audit: `run --lean-tasks 80`, then `run --missing --lean-tasks 170`, plus a rung-version probe.)
  python bench/audit/vericoded_rung.py run --local --langs dafny [--sample N]        # Dafny 4.11 on this machine
  python bench/audit/vericoded_rung.py summary                                       # table from the JSONL

Inputs: VERICODED (default /tmp/vc/vericoding/vericoded: lean/<source>/<model>/<ID>.lean, dafny/<ID>_vericoded.dfy),
records runs/dafny/vericoding.jsonl, text audit runs/audit/vericoded_text.jsonl (joined for reference).
Output: runs/audit/vericoded_rung.jsonl, one row per solution file.

BASELINE, replicated from github.com/Beneficial-AI-Foundation/vericoding @ b265a3b (the published `vericoded/` files
are the final `current_code` of src/vericoding/processing/file_processor.py:process_spec_file):
  success  = verify_file(...).success and not has_final_failure_cheats(current_code)      (file_processor.py:370-400)
  verify_file (src/vericoding/core/language_tools.py:132-252): run compile_check_command with a 60 s timeout
    (l.204), fail on returncode != 0 (l.206); then verify_command with timeout getattr(config, "timeout", 120) = 120 s
    (l.226-231; LanguageConfig has no `timeout` field, src/vericoding/core/config.py:27-40); success = returncode == 0
    (l.234). Both commands are the same (config/language_config.toml:12-13 Dafny, :58-59 Lean), so this replication
    runs the command once with the binding 60 s timeout (a second identical run is redundant up to nondeterminism).
    Lean: `lake build <file>` inside the vericoding Lake package (Mathlib dependency at lake-manifest.json rev
      90c0e186fd333cbdddbe0e5148aa3cd2d24dcf9a, toolchain lean-toolchain v4.23.0-rc2; generated files are copied into
      benchmarks/lean, language_tools.py:141-187). Lake compiles the module with the package's leanOptions
      (lakefile.lean:9-14: autoImplicit=true, relaxedAutoImplicit=false, pp.unicode.fun=true, linter.missingDocs=false)
      and fails only on errors: `sorry` is a warning. Replicated as `lean -D<those options> -o -i -c <file>` with the
      same Mathlib build on the toolchain's search path, returncode == 0 within 60 s.
    Dafny: `dafny verify <file>` with no options (Dafny 4.11.0, the version their CI installs:
      .github/workflows/dafny-validation.yml:23, vericoder-regression.yml:143); returncode == 0 within 60 s. Without
      --allow-warnings, Dafny 4.11 fails on warnings too.
  has_final_failure_cheats (src/vericoding/processing/cheat_checker.py:7-88, 144-197): a regex hit, on the raw text
    (comments included), inside the editable sections only. Lean: \bsorry\b, \badmit\b, \baxiom\b, \bunsafe\b,
    \bUnchecked\.cast\b, @\[extern inside <vc-definitions>, <vc-theorems>, <vc-helpers> (the tag-to-tag spans).
    Dafny: \{:axiom\}, \bassume\b, assume\s*\{:axiom\} inside <vc-code>, <vc-helpers>. Copied verbatim below.
  Spec preservation is structural in their pipeline: the model returns a JSON array of replacements that
    apply_json_replacements (src/vericoding/processing/code_fixer.py:73-348) writes only over `sorry` tokens inside
    editable sections and over <vc-helpers>/<vc-code> bodies; nothing re-checks the final file against the task
    (config spec_patterns, language_config.toml:41-44/87-90, are never read by src/). The text audit compares frozen
    text; it is joined here as `text_tamper`.
  Lean files: every task file gets `import Mathlib` prepended unless it already contains that string
    (src/vericoding/preprocessing.py:4-16, file_processor.py:174-175), so all 1503 published Lean files import
    Mathlib. The image therefore builds Mathlib at vericoding's manifest rev and links its .olean trees into the pinned
    toolchain's own lib/lean (no LEAN_PATH: the Millennium Harness rung runs `lean` with a minimal
    environment).
  Timeouts are wall-clock on their machine; a BASELINE timeout here is reported as REJECT:timeout, and the file is
  re-run with the same options and a 300 s timeout (`baseline.no_timeout`), so machine speed is visible. A Dafny
  BASELINE reject is re-run once with the same command (`baseline_retry`; the status stays the first run's).

PLAIN: Lean `lean <file>` (default options) with no error message; Dafny `dafny verify --allow-warnings <file>` with
returncode 0 and "0 errors". Implied PASS when BASELINE's compile passed (relaxedAutoImplicit=false only rejects more;
--allow-warnings only accepts more); otherwise run with a 300 s timeout.

MILLENNIUM: the rung in the working tree's millennium_harness/ at launch (never modified here; each row's `rung_sha` is the
sha256[:16] of the rung sources it ran with), strict mode (ban scan on). Lean:
millennium_harness.lean.lock(record) then millennium_harness.lean.check(file, frozen); a LockError is LOCK_FAIL (the
rung refuses to certify anything for that task, e.g. axioms or `sorry` in the preamble). Dafny: frozen = the record's
vc-preamble/vc-spec/vc-postamble (bench.fv.curate.dafny_frozen), then millennium_harness.dafny.check(file, frozen).
Framing adaptations (recorded per row in `adaptations`; the solution text is never changed):
  Lean: when the file starts with vericoding's injected `import Mathlib` line (and only comments/blank lines, i.e. the
    `-- <vc-preamble>` marker) before the record's preamble, and the record's preamble does not contain
    "import Mathlib", that exact prefix becomes part of the frozen preamble (the task as vericoding presented it); with
    an empty record preamble the injected line alone is the frozen preamble. Task text after the last closing vc
    marker (outside every section, so not model-written, e.g. verified-cogen's `def main : IO Unit := return ()`) is
    appended to the frozen vc-theorems when the record has no postamble.
  Dafny: when a frozen part is not found verbatim but the file's own marked region (<vc-preamble>, or the text before
    the first <vc-...> marker; <vc-spec>) equals the record's part after removing comments and collapsing whitespace
    outside string/char literals, the file's region is the frozen part (the jsonl records drop comments; an empty
    record part matches a comment-only region).
`record_mismatch` (informative): the file does not contain the record's frozen text even after these adaptations,
i.e. the solution was made from another version of the task (bignum Lean files wrap the task in a namespace).
Lean MILLENNIUM is budget-limited (each Lean lock/check loads Mathlib twice in Check.lean, ~75 CPU-s): BASELINE and
PLAIN run on every file; MILLENNIUM runs on every file of the CERTAIN tasks (text-audit flag, record mismatch,
`axiom`/`sorry` in the frozen text) and of the first --lean-tasks tasks of a seeded permutation of the others (RANDOM);
the rest are NOT_SAMPLED (`lean_strata`). Dafny MILLENNIUM runs on every file.
NOT_AUDITABLE: a solution whose Mathlib-importing task cannot be compiled in this image (counted; with the Mathlib
build in the image this should be none). `mathlib_task` marks tasks whose own preamble imports Mathlib.
Infra failures (container crash, wall-clock timeout of the whole job) are INFRA_ERROR rows; `run --missing` retries
exactly those and the never-run files.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from bench.fv.curate import IMAGE_CMDS, LOCAL_DAFNY, dafny_frozen  # noqa: E402
from millennium_harness import dafny as sl_dafny  # noqa: E402
from millennium_harness import lean as sl_lean  # noqa: E402

VERICODED = Path(os.environ.get("VERICODED", "/tmp/vc/vericoding/vericoded"))
DATA = ROOT / "runs" / "dafny" / "vericoding.jsonl"
TEXT_AUDIT = ROOT / "runs" / "audit" / "vericoded_text.jsonl"
OUT = ROOT / "runs" / "audit" / "vericoded_rung.jsonl"
APP_NAME = "millennium-harness-audit-vericoded"
MATHLIB_REV = "90c0e186fd333cbdddbe0e5148aa3cd2d24dcf9a"  # vericoding lake-manifest.json, inputRev v4.23.0-rc2
LAKE_OPTIONS = ["-DautoImplicit=true", "-DrelaxedAutoImplicit=false", "-Dpp.unicode.fun=true",
                "-Dlinter.missingDocs=false"]  # vericoding lakefile.lean leanOptions
BASELINE_TIMEOUT, RERUN_TIMEOUT = 60, 300
LEAN_SEED = 20261011  # seeded permutation of Lean tasks for the budget-limited MILLENNIUM sample
RUNG_SOURCES = {"lean": ("millennium_harness/lean.py", "millennium_harness/lean_check/Check.lean"),
                "dafny": ("millennium_harness/dafny.py",)}


# ---------------------------------------------------------------------------------------------------- vericoding code
# Verbatim from vericoding src/vericoding/processing/cheat_checker.py:7-88 (MIT, Beneficial AI Foundation), Lean and
# Dafny branches only.
def check_for_cheats(code: str, language: str) -> list[tuple[str, str]]:
    if language.lower() == "lean":
        cheat_patterns = [
            (r"\bsorry\b", "uses 'sorry' to bypass verification"),
            (r"\badmit\b", "uses 'admit' to bypass verification"),
            (r"\baxiom\b", "introduces axioms bypassing verification"),
            (r"\bunsafe\b", "uses unsafe operations"),
            (r"\bUnchecked\.cast\b", "bypasses type checking"),
            (r"@\[extern", "uses extern functions bypassing verification"),
        ]
        sections = ["vc-definitions", "vc-theorems", "vc-helpers"]
    elif language.lower() == "dafny":
        cheat_patterns = [
            (r"\{:axiom\}", "uses '{:axiom}' to bypass verification"),
            (r"\bassume\b", "uses 'assume' to bypass verification"),
            (r"assume\s*\{:axiom\}", "uses 'assume {:axiom}' to bypass verification"),
        ]
        sections = ["vc-code", "vc-helpers"]
    else:
        raise ValueError(language)
    detected_cheats = []
    editable_sections = []
    for section_name in sections:
        pattern = rf"<{section_name}>(.*?)</{section_name}>"
        matches = list(re.finditer(pattern, code, re.DOTALL))
        editable_sections.extend([(match.start(), match.end()) for match in matches])
    for pattern, description in cheat_patterns:
        for match in re.finditer(pattern, code):
            match_pos = match.start()
            in_editable_section = any(start <= match_pos < end for start, end in editable_sections)
            if in_editable_section:
                detected_cheats.append((pattern, description))
                break
    return detected_cheats


# ---------------------------------------------------------------------------------------------------- jobs (local)
def solution_files(vdir: Path = VERICODED) -> list[dict]:
    """Every published Lean/Dafny solution with its record (same matching as bench/audit/vericoded_text.py)."""
    recs = {r["id"]: r for r in map(json.loads, open(DATA))}
    text = {r["file"]: r for r in map(json.loads, open(TEXT_AUDIT))} if TEXT_AUDIT.exists() else {}
    out = []
    for p in sorted(vdir.rglob("*")):
        if p.suffix not in (".lean", ".dfy"):
            continue
        m = re.match(r"([A-Z]{2}\d{4})", p.name)
        if not m or m.group(1) not in recs:
            continue
        rec, rel = recs[m.group(1)], str(p.relative_to(vdir))
        ta = text.get(rel, {})
        out.append({"key": rel, "id": rec["id"], "lang": rec["language"], "source": rec["source"],
                    "model": p.parent.name if rec["language"] == "lean" else None, "file": rel,
                    "text": p.read_text(encoding="utf-8"), "rec": rec, "clean_task": str(rec.get("qa-issue")) == "0",
                    "text_hatches": ta.get("hatches"), "text_tamper": ta.get("tamper")})
    return out


def mathlib_task(rec: dict) -> bool:
    imps = re.findall(r"(?m)^\s*import\s+(\S+)", sl_lean.mask(rec.get("vc-preamble") or "")[0])
    return any(i == "Mathlib" or i.startswith("Mathlib.") for i in imps)


def lean_adapt(rec: dict, text: str) -> tuple[dict, list[str]]:
    """The record as vericoding presented it to the model: its injected `import Mathlib` (plus the comment-only lines
    up to the preamble, i.e. the `-- <vc-preamble>` marker) becomes part of the frozen preamble; task text after the
    last closing vc marker (outside every section, so not model-written: code_fixer.py:120-275 writes only inside
    editable sections; e.g. verified-cogen's `def main : IO Unit := return ()`) is appended to the frozen
    vc-theorems as a whole frozen declaration when the record has no postamble."""
    rec, notes = _lean_adapt_preamble(rec, text)
    marks = list(re.finditer(r"--\s*</vc-[\w-]+>[^\n]*\n?", text))
    tail = text[marks[-1].end():].strip() if marks else ""
    if tail and not sl_lean._blank(tail) and not (rec.get("vc-postamble") or "").strip() \
            and not re.match(r"end\b", sl_lean.mask(tail)[0].strip()):
        rec = {**rec, "vc-theorems": ((rec.get("vc-theorems") or "").rstrip() + "\n\n" + tail).strip()}
        notes.append(f"lean: task text after the last vc section appended to frozen vc-theorems: {tail[:60]!r}")
    return rec, notes


def _lean_adapt_preamble(rec: dict, text: str) -> tuple[dict, list[str]]:
    pre = (rec.get("vc-preamble") or "").strip()
    if "import Mathlib" in pre:
        return rec, []
    t, npre = sl_lean._norm(text), sl_lean._norm(pre)
    if not pre:  # empty preamble: the injected line alone is the frozen preamble
        return ({**rec, "vc-preamble": "import Mathlib"}, ["lean: frozen preamble = vericoding's injected "
                "'import Mathlib' (empty record preamble; preprocessing.py:4-16)"]) if t.startswith("import Mathlib\n") \
            else (rec, [])
    i = t.find(npre)
    head = t[:i] if i > 0 else ""
    if head.startswith("import Mathlib\n") and sl_lean._blank(head[len("import Mathlib\n"):]):
        return {**rec, "vc-preamble": head + pre}, [
            f"lean: frozen preamble prefixed with vericoding's injected framing {head!r} (preprocessing.py:4-16)"]
    return rec, []


def _tokens(s: str) -> list[str]:
    """Dafny text without comments, whitespace-insensitive outside string/char literals (kept verbatim)."""
    out, last = [], 0
    for a, b, com in sl_dafny._literals(s) + [(len(s), len(s), True)]:
        out += s[last:a].split()
        if not com:
            out.append(s[a:b])
        last = b
    return out


def _dafny_region(text: str, tag: str) -> str | None:
    m = re.search(rf"//\s*<{tag}>[^\n]*\n(.*?)//\s*</{tag}>", text, re.S)
    if m:
        return m.group(1)
    if tag == "vc-preamble":
        m = re.search(r"//\s*<vc-", text)
        return text[:m.start()] if m else None
    return None


def dafny_adapt(rec: dict, text: str) -> tuple[dict, list[str]]:
    frozen, notes = dafny_frozen(rec), []
    for part, tag in (("preamble", "vc-preamble"), ("spec", "vc-spec")):
        s = (frozen.get(part) or "").strip()
        if s and s in text:
            continue
        region = _dafny_region(text, tag)  # an empty record part matches a comment-only region (records drop comments)
        if region is not None and region.strip() and _tokens(region) == _tokens(s):
            frozen[part] = region.strip()
            notes.append(f"dafny: frozen {part} taken from the file's <{tag}> region (= record modulo comments/"
                         "whitespace)")
    return frozen, notes


def record_mismatch(job: dict) -> list[str]:
    """Informative only (the rung decides): does the file contain the record's frozen text (after the framing
    adaptation)? A miss means the solution was made from another version of the task than the benchmark record."""
    if job["lang"] == "dafny":
        return sl_dafny._layout(job["text"], job["frozen"])[1]
    rec = job["rec"]
    try:
        chunks = sl_lean._chunks(rec)
    except sl_lean.LockError as e:
        return [f"record does not lock: {str(e)[:200]}"]
    return sl_lean.holes(job["text"], {"preamble": (rec.get("vc-preamble") or "").strip(), "chunks": chunks,
                                       "postamble": (rec.get("vc-postamble") or "").strip()})[1]


def jobs(langs: set, only: set, sample: int, seed: int = 0) -> list[dict]:
    js = [j for j in solution_files() if j["lang"] in langs and (not only or j["id"] in only or j["key"] in only)]
    if sample:
        rnd = random.Random(seed)
        by = collections.defaultdict(list)
        for j in js:
            by[j["lang"]].append(j)
        js = [j for lang in sorted(by) for j in rnd.sample(by[lang], min(sample, len(by[lang])))]
    for j in js:
        if j["lang"] == "lean":
            j["mathlib_task"] = mathlib_task(j["rec"])  # the task's own preamble, before any adaptation
            rec, j["adaptations"] = lean_adapt(j["rec"], j["text"])
            j["rec"] = {k: rec.get(k) for k in ("id",) + sl_lean.FIELDS}
            j["lock_key"] = hashlib.sha256(json.dumps(j["rec"], sort_keys=True).encode()).hexdigest()[:16]
        else:
            j["frozen"], j["adaptations"] = dafny_adapt(j["rec"], j["text"])
            j["rec"] = None
            j["mathlib_task"] = False
        j["record_mismatch"] = record_mismatch(j)
    return js


def lean_strata(js: list[dict], n_random: int, seed: int = LEAN_SEED) -> dict[str, str]:
    """Lean task id -> stratum for the (budget-limited) MILLENNIUM part. CERTAIN: every task with a text-audit flag
    (escape hatch or changed frozen text), a record mismatch, or `axiom`/`sorry` in its frozen text (the rung refuses
    to lock those). RANDOM: the first n_random of the other tasks in a fixed seeded permutation (a larger n_random
    extends the same sample). Every other task: NOT_SAMPLED (BASELINE and PLAIN only)."""
    tasks = sorted({j["id"] for j in js if j["lang"] == "lean"})
    certain = set()
    for j in js:
        if j["lang"] != "lean":
            continue
        rec = j["rec"]
        frozen = sl_lean.mask("\n".join(rec.get(k) or "" for k in sl_lean.FIELDS))[0]
        if j.get("text_hatches") or j.get("text_tamper") or j["record_mismatch"] or \
                re.search(r"(?m)^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*axiom\b", frozen) or \
                re.search(r"\bsorry\b", sl_lean.mask("\n".join(rec.get(k) or "" for k in
                                                              ("vc-preamble", "vc-helpers", "vc-postamble")))[0]):
            certain.add(j["id"])
    rest = [t for t in tasks if t not in certain]
    random.Random(seed).shuffle(rest)
    picked = set(rest[:n_random])
    return {t: "CERTAIN" if t in certain else "RANDOM" if t in picked else "NOT_SAMPLED" for t in tasks}


# ---------------------------------------------------------------------------------------------------- remote work
def _run(cmd: list[str], cwd: str, timeout: float, env: dict | None = None) -> tuple[int | None, str, float]:
    """(returncode or None on timeout, output, seconds); the whole process group dies on a timeout (Z3 children)."""
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out, round(time.time() - t0, 1)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return None, out or "", round(time.time() - t0, 1)


def _tail(s: str, n: int = 1500) -> str:
    return s if len(s) <= n else "… " + s[-n:]


def _peak_mb() -> int:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024  # Linux: KiB


def _cpu() -> float:
    """CPU seconds (user + system) of all finished child processes so far (cost accounting)."""
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def lean_compile(text: str, options: list[str], timeout: float) -> dict:
    lean = sl_lean.find_lean()
    with tempfile.TemporaryDirectory(prefix="vc-base-") as d:
        Path(d, "Solution.lean").write_text(text, encoding="utf-8")
        rc, out, secs = _run([lean, *options, "-o", "Solution.olean", "-i", "Solution.ilean", "-c", "Solution.c",
                              "Solution.lean"], d, timeout, sl_lean._env(d))
    errs = [m for m in sl_lean._messages(out) if m["severity"] == "error"]
    status = "timeout" if rc is None else "ok" if rc == 0 and not errs else "error"
    return {"status": status, "secs": secs, "tail": _tail("\n\n".join(m["text"] for m in errs) or out)}


def dafny_verify(text: str, name: str, flags: list[str], timeout: float) -> dict:
    dafny = os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY", "dafny"))
    with tempfile.TemporaryDirectory(prefix="vc-base-") as d:
        Path(d, name).write_text(text, encoding="utf-8")
        rc, out, secs = _run([dafny, "verify", *flags, name], d, timeout)
    m = re.search(r"finished with (\d+) verified, (\d+) errors?", out)
    # BASELINE: returncode 0 (language_tools.py:234). PLAIN: returncode 0 and a "0 errors" verifier summary.
    ok = rc == 0 if not flags else rc == 0 and m is not None and m.group(2) == "0"
    return {"status": "timeout" if rc is None else "ok" if ok else "error", "rc": rc, "secs": secs,
            "summary": m.group(0) if m else None, "tail": _tail(out)}


def baseline(job: dict, compile_fn) -> tuple[dict, str]:
    """BASELINE status and PLAIN status for one solution."""
    lang, text = job["lang"], job["text"]
    b = compile_fn("baseline", BASELINE_TIMEOUT)
    cheats = [d for _, d in check_for_cheats(text, lang)]
    res = {"compile": b["status"], "secs": b["secs"], "cheats": cheats, "summary": b.get("summary")}
    if b["status"] != "ok":
        res["tail"] = b["tail"]
    if b["status"] == "timeout":
        r = compile_fn("baseline", RERUN_TIMEOUT)
        res["no_timeout"] = {"compile": r["status"], "secs": r["secs"]}
    status = ("ACCEPT" if not cheats else "REJECT:cheat-pattern") if b["status"] == "ok" else \
        f"REJECT:{'timeout' if b['status'] == 'timeout' else 'compile'}" + ("+cheat-pattern" if cheats else "")
    if b["status"] == "ok" or res.get("no_timeout", {}).get("compile") == "ok":
        plain = "PASS"  # implied: see module docstring
    else:
        p = compile_fn("plain", RERUN_TIMEOUT)
        plain = {"ok": "PASS", "error": "FAIL", "timeout": "TIMEOUT"}[p["status"]]
    res["status"] = status
    return res, plain


def lean_lock(job: dict) -> dict:
    """{lock_key, frozen} or {lock_key, lock_error} or {lock_key, infra_error}."""
    t0, c0 = time.time(), _cpu()
    try:
        frozen = sl_lean.lock(job["rec"])
        return {"lock_key": job["lock_key"], "frozen": frozen, "secs": round(time.time() - t0, 1),
                "cpu": round(_cpu() - c0, 1), "peak_mb": _peak_mb()}
    except sl_lean.LockError as e:
        if re.fullmatch(r"skeleton does not compile:\s*", str(e)):  # lock() does not report its own timeout
            return {"lock_key": job["lock_key"], "infra_error": "lock: skeleton compile timed out or crashed"}
        return {"lock_key": job["lock_key"], "lock_error": str(e)[-2000:], "secs": round(time.time() - t0, 1),
                "cpu": round(_cpu() - c0, 1)}
    except Exception as e:  # noqa: BLE001 - Check.lean crash, OOM...: infrastructure, never a task status
        return {"lock_key": job["lock_key"], "infra_error": f"{type(e).__name__}: {str(e)[-1500:]}"}


def _row(job: dict) -> dict:
    return {k: job.get(k) for k in ("key", "id", "lang", "source", "model", "file", "clean_task", "mathlib_task",
                                    "adaptations", "record_mismatch", "text_hatches", "text_tamper", "rung_sha")}


def lean_audit(job: dict) -> dict:
    row, t0, c0 = _row(job), time.time(), _cpu()
    lock = job.get("lock") or {}

    def compile_fn(kind, timeout):
        return lean_compile(job["text"], LAKE_OPTIONS if kind == "baseline" else [], timeout)
    try:
        if job.get("prev_baseline"):  # extending the sample: BASELINE/PLAIN already ran for this file
            row.update(job["prev_baseline"])
            b = row["baseline"]
        else:
            b, plain = baseline(job, compile_fn)
            row.update(baseline_status=b["status"], plain_status=plain, baseline=b,
                       baseline_cpu=round(_cpu() - c0, 1))
        row.update(stratum=job["stratum"], lock_cpu=lock.get("cpu"), lock_secs=lock.get("secs"))
        if job["stratum"] == "NOT_SAMPLED":
            row.update(millennium_status="NOT_SAMPLED", violations=[], errors_tail="")
        elif "frozen" in lock:
            r = sl_lean.check(job["text"], lock["frozen"])
            row.update(millennium_status=r["status"], violations=r["violations"], errors_tail=r["errors_tail"],
                       millennium_secs=r["secs"])
        elif "lock_error" in lock:
            row.update(millennium_status="LOCK_FAIL", violations=[lock["lock_error"]], errors_tail="")
        else:
            row.update(millennium_status="INFRA_ERROR", violations=[lock.get("infra_error", "no lock result")],
                       errors_tail="")
        if b["compile"] == "error" and re.search(r"unknown module prefix '(?:Mathlib|Batteries)'", b.get("tail", "")):
            row["millennium_status"] = row["baseline_status"] = "NOT_AUDITABLE"  # no Mathlib in this image
    except Exception as e:  # noqa: BLE001
        row.update(baseline_status=row.get("baseline_status", "INFRA_ERROR"), millennium_status="INFRA_ERROR",
                   violations=[f"{type(e).__name__}: {str(e)[-1000:]}"], errors_tail="")
    row.update(secs=round(time.time() - t0, 1), cpu=round(_cpu() - c0, 1), peak_mb=_peak_mb())
    return row


def dafny_audit(job: dict) -> dict:
    row, t0, c0 = _row(job), time.time(), _cpu()
    name = Path(job["file"]).name

    def compile_fn(kind, timeout):
        return dafny_verify(job["text"], name, [] if kind == "baseline" else ["--allow-warnings"], timeout)
    try:
        if job.get("prev_baseline") and job["prev_baseline"]["baseline_status"] == "ACCEPT":
            row.update(job["prev_baseline"])
        else:
            b, plain = baseline(job, compile_fn)
            row.update(baseline_status=b["status"], plain_status=plain, baseline=b)
            if b["status"] != "ACCEPT":  # flakiness probe: the same command once more (status stays the first run's)
                r2 = compile_fn("baseline", BASELINE_TIMEOUT)
                row["baseline_retry"] = {"compile": r2["status"], "secs": r2["secs"], "summary": r2.get("summary")}
        r = sl_dafny.check(job["text"], job["frozen"], name)
        row.update(millennium_status=r["status"], violations=r.get("violations", []),
                   errors_tail="" if r["status"] == "PROVED" else _tail(r.get("verify") or "", 1500))
    except Exception as e:  # noqa: BLE001
        row.update(baseline_status=row.get("baseline_status", "INFRA_ERROR"), millennium_status="INFRA_ERROR",
                   violations=[f"{type(e).__name__}: {str(e)[-1000:]}"], errors_tail="")
    row.update(secs=round(time.time() - t0, 1), cpu=round(_cpu() - c0, 1), peak_mb=_peak_mb())
    return row


# ---------------------------------------------------------------------------------------------------- Modal
# bench/fv/curate.py's recipe (elan + Lean v4.23.0-rc2 at /opt/lean, Dafny 4.11.0 at /opt/dafny) plus vericoding's
# Mathlib, prebuilt from the Mathlib cache and linked into the toolchain's lib/lean (Lean's built-in search path).
MATHLIB_CMDS = [
    "git clone --depth 1 --branch v4.23.0-rc2 https://github.com/leanprover-community/mathlib4.git /opt/mathlib",
    f'test "$(git -C /opt/mathlib rev-parse HEAD)" = {MATHLIB_REV}',
    "grep -qF 'v4.23.0-rc2' /opt/mathlib/lean-toolchain",
    "cd /opt/mathlib && PATH=/opt/lean/bin:/usr/bin:/bin ELAN_HOME=/opt/elan lake exe cache get",
    "test -f /opt/mathlib/.lake/build/lib/lean/Mathlib.olean",
    'for d in /opt/mathlib/.lake/build/lib/lean /opt/mathlib/.lake/packages/*/.lake/build/lib/lean; do '
    'for f in "$d"/*; do ln -s "$f" "/opt/lean/lib/lean/$(basename "$f")"; done; done',
    "printf 'import Mathlib\\nexample : (2 : Real) + 2 = 4 := by norm_num\\nlemma l : True := trivial\\n' > /tmp/t.lean"
    " && cd /tmp && env -i PATH=/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 /opt/lean/bin/lean /tmp/t.lean && rm /tmp/t.lean",
]

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App(APP_NAME)
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("curl", "ca-certificates", "xz-utils", "unzip", "libicu72", "git")
             .run_commands(*IMAGE_CMDS)
             .run_commands(*MATHLIB_CMDS)
             .env({"MILLENNIUM_LEAN": "/opt/lean/bin/lean", "MILLENNIUM_DAFNY": "/opt/dafny/dafny",
                   "DOTNET_CLI_TELEMETRY_OPTOUT": "1"})
             .add_local_python_source("bench")
             .add_local_python_source("millennium_harness", ignore=lambda p: p.suffix == ".pyc"))  # keeps Check.lean

    MAXC = int(os.environ.get("VC_MAX_CONTAINERS", "48"))

    @app.function(image=image, cpu=1, memory=3072, timeout=3600, max_containers=MAXC)
    def lean_lock_remote(job: dict) -> dict:
        return lean_lock(job)

    @app.function(image=image, cpu=1, memory=3072, timeout=3600, max_containers=MAXC)
    def lean_audit_remote(job: dict) -> dict:
        return lean_audit(job)

    @app.function(image=image, cpu=1, memory=1024, timeout=3600, max_containers=MAXC)
    def dafny_audit_remote(job: dict) -> dict:
        return dafny_audit(job)


# ---------------------------------------------------------------------------------------------------- driver
_write_lock = threading.Lock()


_pending: list[str] = []


def _emit(row: dict, out: Path) -> None:
    """Append a row; on a write error (disk full) keep it in memory and retry with the next row and at the end."""
    with _write_lock:
        _pending.append(json.dumps(row, ensure_ascii=False) + "\n")
        try:
            with open(out, "a", encoding="utf-8") as f:
                f.write("".join(_pending))
            _pending.clear()
        except OSError as e:
            print(f"WRITE FAILED ({e}); {len(_pending)} rows pending", flush=True)
    print(f"{row['lang']:5s} {row['file']:52s} base={row.get('baseline_status')!s:24s} "
          f"plain={row.get('plain_status')!s:7s} sl={row.get('millennium_status')!s:15s} {row.get('secs')}s "
          f"cpu={row.get('cpu')}s {row.get('peak_mb')}MB", flush=True)


def _cache_lock(cache: str | None, r: dict) -> None:
    if cache and "infra_error" not in r:
        try:
            with open(cache, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except OSError:
            pass


def run_lean(js: list[dict], out: Path, use_modal: bool) -> None:
    if not js:
        return
    keys = {}
    for j in js:
        if j["stratum"] != "NOT_SAMPLED":
            keys.setdefault(j["lock_key"], {"lock_key": j["lock_key"], "rec": j["rec"]})
    print(f"lean: {len(js)} solutions, {len(keys)} locks", flush=True)
    locks, cache = {}, os.environ.get("VC_LOCK_CACHE")  # optional JSONL cache of lock results across runs
    if cache and os.path.exists(cache):
        for line in open(cache, encoding="utf-8"):
            r = json.loads(line)
            if r["lock_key"] in keys and "infra_error" not in r:
                locks[r["lock_key"]] = r
        print(f"lean: {len(locks)} locks from {cache}", flush=True)
        keys = {k: v for k, v in keys.items() if k not in locks}
    if use_modal and keys:
        for r in lean_lock_remote.map(list(keys.values()), return_exceptions=True, order_outputs=False):
            if isinstance(r, BaseException):
                print("lock ERR", repr(r)[:200], flush=True)
                continue
            locks[r["lock_key"]] = r
            _cache_lock(cache, r)
    else:
        with ThreadPoolExecutor(4) as ex:
            for r in ex.map(lean_lock, keys.values()):
                locks[r["lock_key"]] = r
                _cache_lock(cache, r)
    st = collections.Counter("frozen" if "frozen" in r else "lock_error" if "lock_error" in r else "infra"
                             for r in locks.values())
    print(f"lean locks: {dict(st)}", flush=True)
    work = [j | {"lock": locks.get(j["lock_key"], {} if j["stratum"] == "NOT_SAMPLED" else
                                    {"infra_error": "lock call failed"})} for j in js]
    _consume(work, lean_audit_remote if use_modal else None, lean_audit, out)


def _consume(work: list[dict], remote, local, out: Path) -> None:
    done = set()
    if remote is not None:
        for r in remote.map(work, return_exceptions=True, order_outputs=False):
            if isinstance(r, BaseException):
                print("ERR", repr(r)[:200], flush=True)
                continue
            done.add(r["key"])
            _emit(r, out)
    else:
        with ThreadPoolExecutor(4) as ex:
            for r in ex.map(local, work):
                done.add(r["key"])
                _emit(r, out)
    for j in work:
        if j["key"] not in done:
            _emit(_row(j) | {"baseline_status": "INFRA_ERROR", "plain_status": None, "millennium_status": "INFRA_ERROR",
                             "violations": ["the remote call failed"], "errors_tail": ""}, out)


def run(langs: set, only: set, sample: int, missing: bool, use_modal: bool, out: Path, lean_tasks: int) -> None:
    js = jobs(langs, only, sample)
    strata, sha = lean_strata(js, lean_tasks), _rung_sha()  # the rung sources shipped with this run
    for j in js:
        j["stratum"] = strata.get(j["id"]) if j["lang"] == "lean" else "ALL"
        j["rung_sha"] = None if j["stratum"] == "NOT_SAMPLED" else sha[j["lang"]]
    print("lean strata (tasks):", dict(collections.Counter(strata.values())), flush=True)
    if missing and out.exists():
        old = {r["key"]: r for r in load(out)}
        keep = []
        for j in js:
            r = old.get(j["key"])
            prev = {k: r.get(k) for k in ("baseline_status", "plain_status", "baseline", "baseline_cpu",
                                          "baseline_retry") if k in r} if r else None
            if r is None or "INFRA_ERROR" in (r.get("baseline_status"), r.get("millennium_status")):
                keep.append(j)
            elif j["lang"] == "dafny" and r["baseline_status"] != "ACCEPT" and "baseline_retry" not in r:
                keep.append(j)  # rerun once with the flakiness probe
            elif r.get("adaptations") != j["adaptations"]:
                j["prev_baseline"] = prev  # framing changed: rerun the MILLENNIUM part only
                keep.append(j)
            elif r.get("millennium_status") == "NOT_SAMPLED" and j["stratum"] != "NOT_SAMPLED":
                j["prev_baseline"] = prev
                keep.append(j)
        js = keep
    print(f"{len(js)} solutions to audit: {dict(collections.Counter(j['lang'] for j in js))}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    lean_js = [j for j in js if j["lang"] == "lean"]
    dafny_js = [j for j in js if j["lang"] == "dafny"]
    t0 = time.time()
    if use_modal:
        if modal is None:
            sys.exit("modal is not installed")
        with modal.enable_output(), app.run():
            with ThreadPoolExecutor(2) as ex:
                fs = [ex.submit(run_lean, lean_js, out, True),
                      ex.submit(_consume, dafny_js, dafny_audit_remote if dafny_js else None, dafny_audit, out)]
                for f in fs:
                    f.result()
    else:
        if dafny_js:
            os.environ.setdefault("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY", str(LOCAL_DAFNY)))
        run_lean(lean_js, out, False)
        _consume(dafny_js, None, dafny_audit, out)
    while _pending:  # disk was full: retry until the rows are written
        try:
            with open(out, "a", encoding="utf-8") as f:
                f.write("".join(_pending))
            _pending.clear()
        except OSError as e:
            print(f"WRITE FAILED ({e}); retrying in 30 s", flush=True)
            time.sleep(30)
    print(f"wall {time.time() - t0:.0f}s", flush=True)
    consolidate(out)
    summary(load(out))


# ---------------------------------------------------------------------------------------------------- results
def load(out: Path = OUT) -> list[dict]:
    rows = {}
    for line in open(out, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["key"]] = r  # the last row of a file wins (reruns)
    return list(rows.values())


def consolidate(out: Path = OUT) -> None:
    rows = sorted(load(out), key=lambda r: (r["lang"], r["id"], r.get("model") or ""))
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _rung_sha() -> dict:
    return {lang: hashlib.sha256(b"".join((ROOT / f).read_bytes() for f in fs)).hexdigest()[:16]
            for lang, fs in RUNG_SOURCES.items()}


def cells(rows: list[dict]) -> dict:
    c = collections.Counter()
    for r in rows:
        if "NOT_AUDITABLE" in (r["baseline_status"], r["millennium_status"]):
            c["not_auditable"] += 1
        elif "INFRA_ERROR" in (r["baseline_status"], r["millennium_status"]):
            c["infra_error"] += 1
        elif r["millennium_status"] == "NOT_SAMPLED":  # BASELINE only
            c["not_sampled"] += 1
            c["B+" if r["baseline_status"] == "ACCEPT" else "B-"] += 1
        else:
            c[("B+" if r["baseline_status"] == "ACCEPT" else "B-") + ("S+" if r["millennium_status"] == "PROVED"
                                                                         else "S-")] += 1
    return c


def reason(r: dict) -> str:
    """Why Millennium Harness rejected a row: task-level (no solution of the task can be certified as posed) or
    solution-level (this file). Read from the rung's own violation text."""
    st, v = r["millennium_status"], " ".join(r.get("violations") or [])
    if r.get("record_mismatch"):
        return "task: file made from another version of the task than the benchmark record"
    if st == "LOCK_FAIL":
        return "task: declares axioms (rung refuses to lock)" if "axioms" in v else "task: `sorry` in frozen task text"
    if st == "CHECK_ERROR":
        return ("task: spec uses a name only the solution defines" if "unresolved identifier" in v
                else "task: frozen parts do not resolve")
    if "elaborates differently" in v:
        return ("solution: adds an instance that changes what a frozen statement means (e.g. `local instance : LE Int "
                ":= ⟨fun _ _ => True⟩`)" if "instance `" in v else
                "solution: rewrites a spec-relevant definition the task put in vc-helpers (vericoding lets the model "
                "replace vc-helpers)")
    if "was not verified" in v:
        return "task: frozen spec contains bodiless (assumed) lemmas/methods"
    if "native" in v:
        return "solution: native_decide (trusts the compiler)"
    if "bodiless declaration with ensures" in v:
        return "solution: bodiless lemma/function with ensures (an assumption)"
    if "decreases *" in v:
        return "solution: decreases * (no termination proof)"
    if "inside an agent comment" in v:
        head = re.split(r"//\s*<vc-helpers>", (VERICODED / r["file"]).read_text(errors="replace"))[0] \
            if (VERICODED / r["file"]).exists() else ""
        if head.count("/*") > head.count("*/"):
            return "task: its preamble opens a comment that never closes (the whole file is a comment: 0 verified)"
        return "solution: an unclosed comment swallows frozen spec text"
    if "must start a new declaration" in v:
        return "solution: gives a body to a frozen bodiless declaration"
    m = re.search(r"followed directly by the method body.*?: '([^']*)", v)
    if m:
        nxt = m.group(1).split()
        return f"solution: {'no body at all' if not nxt else 'adds `' + nxt[0] + '` clause'} between spec and body"
    if "meta-programming or IO" in v:
        return "solution: meta-programming/IO declaration"
    if "noncomputable" in v:
        return "solution: noncomputable instance/def (e.g. a classical `Coe Prop Bool` that re-elaborates the spec)"
    if "not compilable" in v:
        return "solution: not compilable (the code calls or declares a bodiless function)"
    if st == "UNPROVED":
        return "solution: does not verify under the rung's settings"
    return f"{st}: {v[:80]}"


def summary(rows: list[dict]) -> dict:
    """Cells (B = vericoding BASELINE ACCEPT, S = Millennium Harness PROVED) per language, model, source and Lean stratum."""
    def line(label, rs):
        c = cells(rs)
        print(f"  {label:30s} n={len(rs):5d}  B+S+={c['B+S+']:5d}  B+S-={c['B+S-']:4d}  B-S+={c['B-S+']:3d}  "
              f"B-S-={c['B-S-']:3d}  | not_sampled={c['not_sampled']:4d} (B+ {c['B+']:4d}, B- {c['B-']:3d})  "
              f"not_auditable={c['not_auditable']}  infra={c['infra_error']}")
        return dict(c)
    excluded = ("PROVED", "NOT_AUDITABLE", "INFRA_ERROR", "NOT_SAMPLED")
    out = {"rung_sha": _rung_sha(), "cells": {}, "by_model": {}, "by_stratum": {}, "baseline": {}, "plain": {},
           "millennium": {}, "baseline_accept_millennium_reject": {}}
    print(f"\nrung sources sha256[:16]: {out['rung_sha']}   (B = vericoding baseline ACCEPT, S = Millennium Harness PROVED)")
    for lang in ("lean", "dafny"):
        rs = [r for r in rows if r["lang"] == lang]
        if not rs:
            continue
        print(f"\n== {lang}")
        out["cells"][lang] = line("ALL", rs)
        if lang == "lean":
            for st in ("CERTAIN", "RANDOM", "NOT_SAMPLED"):
                sub = [r for r in rs if r.get("stratum") == st]
                if sub:
                    out["by_stratum"][st] = line(f"stratum={st} ({len({r['id'] for r in sub})} tasks)", sub)
            for m in sorted({r["model"] for r in rs}):
                out["by_model"][m] = line(m, [r for r in rs if r["model"] == m])
        for src in sorted({r["source"] for r in rs}):
            line(f"source={src}", [r for r in rs if r["source"] == src])
        for k in ("baseline", "plain", "millennium"):
            key = "plain_status" if k == "plain" else f"{k}_status"
            out[k][lang] = dict(collections.Counter(r.get(key) for r in rs).most_common())
            print(f"  {k:10s}", out[k][lang])
        vac = [r for r in rs if r["baseline_status"] == "ACCEPT" and
               re.search(r"finished with 0 verified", r["baseline"].get("summary") or "")]
        if lang == "dafny":
            print(f"  BASELINE ACCEPT with 'finished with 0 verified' (nothing checked): {len(vac)} "
                  f"{sorted(r['id'] for r in vac)} -> MILLENNIUM "
                  f"{dict(collections.Counter(r['millennium_status'] for r in vac))}")
        retry = [r for r in rs if r.get("baseline_retry")]
        if retry:
            print(f"  BASELINE rejects re-run once: {dict(collections.Counter(r['baseline_retry']['compile'] for r in retry))}")
        to = [r for r in rs if r["baseline_status"].startswith("REJECT:timeout")]
        if to:
            print(f"  baseline timeouts that pass with {RERUN_TIMEOUT}s:",
                  sum((r["baseline"].get("no_timeout") or {}).get("compile") == "ok" for r in to), "of", len(to))
        ar = [r for r in rs if r["baseline_status"] == "ACCEPT" and r["millennium_status"] not in excluded]
        print("  rung versions (sha256[:16] of the rung sources) behind MILLENNIUM:",
              dict(collections.Counter(r.get("rung_sha") for r in rs if r["millennium_status"] != "NOT_SAMPLED")))
        print("  B+S- by MILLENNIUM status:", dict(collections.Counter(r["millennium_status"] for r in ar).most_common()),
              f"(of which the file does not contain the record's frozen text: {sum(bool(r['record_mismatch']) for r in ar)})")
        why = collections.Counter(reason(r) for r in ar)
        print("  B+S- reasons:")
        for k, n in sorted(why.items(), key=lambda kv: (not kv[0].startswith("task"), -kv[1])):
            print(f"    {n:4d}  {k}")
        out["baseline_accept_millennium_reject"][lang] = [
            {"id": r["id"], "model": r["model"], "file": r["file"], "stratum": r.get("stratum"), "reason": reason(r),
             "millennium_status": r["millennium_status"], "record_mismatch": bool(r["record_mismatch"]),
             "why": [v[:300] for v in (r.get("violations") or [(r.get("errors_tail") or "")[-300:]])[:2]]} for r in ar]
        rnd = [r for r in rs if r.get("stratum") == "RANDOM"]
        if rnd:
            c, pop = cells(rnd), sum(r.get("stratum") in ("RANDOM", "NOT_SAMPLED") for r in rs)
            n = sum(c[k] for k in ("B+S+", "B+S-", "B-S+", "B-S-"))
            if n:
                print(f"  RANDOM stratum B+S- rate {c['B+S-']}/{n} = {100 * c['B+S-'] / n:.1f}% of solutions, "
                      f"~{c['B+S-'] / n * pop:.0f} expected among the {pop} non-CERTAIN solutions")
        mt = [r for r in rs if r.get("mathlib_task")]
        if mt:
            print(f"  tasks whose own preamble imports Mathlib: {len({r['id'] for r in mt})} tasks, {len(mt)} solutions "
                  f"(NOT_AUDITABLE: {sum(r['millennium_status'] == 'NOT_AUDITABLE' for r in mt)})")
        ad = collections.Counter(a.split(":")[0] + ":" + a.split(":")[1][:40] for r in rs for a in r.get("adaptations") or [])
        if ad:
            print("  framing adaptations:", dict(ad))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--langs", default="lean,dafny")
    r.add_argument("--only", default="", help="comma-separated ids or files")
    r.add_argument("--sample", type=int, default=0, help="N random solutions per language (seed 0)")
    r.add_argument("--missing", action="store_true", help="skip files that already have a non-INFRA row")
    r.add_argument("--local", action="store_true")
    r.add_argument("--lean-tasks", type=int, default=10**6,
                   help="Lean tasks in the RANDOM MILLENNIUM stratum (seeded prefix; default all)")
    r.add_argument("--out", default=str(OUT))
    s = sub.add_parser("summary")
    s.add_argument("--out", default=str(OUT))
    s.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run(set(a.langs.split(",")), set(x for x in a.only.split(",") if x), a.sample, a.missing, not a.local,
            Path(a.out), a.lean_tasks)
    else:
        res = summary(load(Path(a.out)))
        if a.json:
            print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
