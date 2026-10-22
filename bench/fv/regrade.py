"""Fresh-container regrade of FV trial snapshots (Millennium Harness FV v1.0; docs/research/win-plan-synthesis.md
§6 Grading).

  . scripts/env.sh && .venv/bin/python bench/fv/regrade.py runs/jobs/<job>     # Modal: a fresh container per file
  .venv/bin/python bench/fv/regrade.py runs/jobs/<job> --local                  # local pinned toolchains (dev, tests)
  -> runs/fv/grades/<job>.jsonl, one row per trial dir, in trial start order (result.json started_at)

Input: what bench/agents_fv.py leaves in each trial dir: the events in agent/fv.jsonl and the snapshots
agent/fv/snap/<label>.tar.gz (+ <label>.json manifest) for labels S0, B1.., C1... Only the solution
(app/Solution.lean or app/solution.dfy) is read, never extracted. It is resolved inside the tar as the sandbox's `cat`
would see it: hard links and symlinks (of the file or of its directories) are followed within the archive. A symlink
that leaves the snapshotted roots cannot be graded (a snapshot problem); one that dangles inside them, a directory or
another non-file means there is no solution (UNPROVED). The tar must match the sha256 taken in the sandbox and the file
the manifest's sha256; otherwise that snapshot is not graded. A = S0. B = the last B round's snapshot (S0 if the gate
never resumed the session). C = the last C round's snapshot (null for A/B-only agents).

`infra` (the trial is excluded from every branch) lists everything that makes the trial's A/B/C not a valid outcome:
  - Harbor did not finish the trial (no result.json) or recorded an agent-phase exception (agent timeout, cancel...);
    verifier-phase exceptions (VerifierTimeoutError, reward file...) come after the snapshots and only warn;
  - the flow was interrupted: no summary event, or the summary has no A stop, no B final gate status, or (when the
    agent runs C: FVBranch, or the trial's nudge kwarg) no C end time; a C on an agent configured without one;
  - an in-sandbox gate status outside the rung's five (GATE_ERROR, NO_TASK, missing): B never had a working gate;
  - C did not start from S0 (the restore/verify before C failed);
  - the egress probe got an HTTP answer (any code but 000) from a non-model host (plan K2);
  - a problem with a snapshot that A, B or C uses: missing, failing an integrity check, ungraded, a grader error, or
    a grader status outside PROVED/UNPROVED/CHEAT_REJECTED/SPEC_TAMPERED (e.g. CHECK_ERROR: a task that cannot be
    certified or a crashing grader). The raw status is kept; such a branch never counts as a valid zero.
Problems with intermediate snapshots (B1 when B2 exists, C1 when C2 exists) go to `warnings` and do not exclude the
trial, so trials where B fired (more snapshots) are not excluded more often.

Each unique (file sha256, frozen.json sha256) is graded once by two graders, each in its own fresh process with a clean
environment and an empty private HOME. Strict runs first: lenient compiles agent code that strict's ban scan refuses.
  strict   the full rung: millennium_harness.lean.check / millennium_harness.dafny.check.
  lenient  Lean: the rung without the text ban scan (`bans=False`; chunk lock, compile and Check.lean with the axiom
           whitelist stay). Dafny: millennium_harness.dafny.check itself, run with its ban scan, `dafny audit` and the
           `dafny translate` compilability check switched off, so every frozen-part check the rung has (or gains) and
           `dafny verify` with the rung's flags apply unchanged. Unlike Lean lenient it has no axiom check (`assume
           false` and {:axiom} pass): more permissive, so it can only make the lenient claim condition harder.
A Dafny verdict that ran `dafny verify` is computed twice, and a third time if the two differ; the majority wins (a
three-way split counts as the first non-PROVED status). Lean is deterministic and runs once. Each worker runs in its
own process group: whatever it leaves running (a z3 that outlived a `dafny` timeout) is killed between vote runs and
when the worker ends, so re-runs stay independent and do not slow other files.

Modal (grade_remote, defined here, launched only by the CLI without --local): one single-use container per unique
file, no network, no access to other Modal resources. The toolchain binaries (Lean: lean and lib/lean shared
libraries; Dafny: apphost, .dll assemblies, bundled z3) are hashed and compared with the record taken at image build,
and both graders run as the non-root user `grader`. Each result carries the toolchain and checker digests; the host
refuses results from an image whose checker (Millennium Harness lean/dafny rung + this file) differs from this
checkout.
--local runs agent code on this machine under your user: use it for tests and DEV debugging, not for TEST grading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # inside the grading container this is "/", and unused
if (ROOT / "bench" / "agents_fv.py").exists() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "runs" / "fv" / "grades"
SPLIT = ROOT / "runs" / "fv" / "split.json"
GRADERS = ("strict", "lenient")  # in this order
VERDICTS = ("PROVED", "UNPROVED", "CHEAT_REJECTED", "SPEC_TAMPERED")  # a grader status that is a valid outcome
GATE_OK = (*VERDICTS, "CHECK_ERROR")  # the rung's five statuses; the gate also logs GATE_ERROR and NO_TASK
NUDGE_DEFAULT = {"FVBranch": True, "ClaudeCodeFV": False}  # bench/agents_fv.py FVOptions / FVClaudeOptions .nudge
VERIFIER_EXC = {"VerifierTimeoutError", "RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError",
                "DownloadVerifierDirError", "AddTestsDirError"}  # Harbor's verifier phase: after the agent finished
EGRESS = re.compile(r"^\s*(\S+)\s+(\d{3})\s*$")  # a probe line "<host> <http code>"; 000 = no connection
LABEL = re.compile(r"[SBC]\d{1,2}")
MAX_BYTES = 4 << 20
MAX_LINKS = 40  # symlink hops, as Linux's ELOOP limit
WORKER_TIMEOUT = 3000
USAGE_KEYS = ("in", "out", "reasoning", "cache_read", "cache_write", "cost", "steps")
CHECKER = ("lean.py", "dafny.py", "lean_check/Check.lean")  # under <sl_root>/millennium_harness, plus this file
# The grading image: the same pins as bench/fv/images/*.Dockerfile, without harnesses. elan no longer installs the
# default toolchain eagerly, so the toolchain is installed explicitly (as in bench/fv/images/lean.Dockerfile).
LEAN_TOOLCHAIN = "leanprover/lean4:v4.23.0-rc2"
ELAN_INIT = "https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh"
DAFNY_ZIP = "https://github.com/dafny-lang/dafny/releases/download/v4.11.0/dafny-4.11.0-x64-ubuntu-22.04.zip"
REMOTE = {"lean": "/opt/lean/bin/lean", "dafny": "/opt/dafny/dafny", "sl_root": "/opt/sl/src",
          "script": "/opt/sl/regrade.py", "record": "/opt/sl/toolchain.json", "user": "grader"}
IMAGE_CMDS = [
    f"curl -fsSL -o /tmp/elan-init.sh {ELAN_INIT} && sh /tmp/elan-init.sh -y --no-modify-path "
    "--default-toolchain none && rm /tmp/elan-init.sh",
    f"/opt/elan/bin/elan toolchain install {LEAN_TOOLCHAIN} && /opt/elan/bin/elan default {LEAN_TOOLCHAIN}",
    'ln -s "$(dirname "$(dirname "$(/opt/elan/bin/elan which lean)")")" /opt/lean && /opt/lean/bin/lean --version',
    f"curl -fsSL -o /tmp/d.zip {DAFNY_ZIP} && unzip -q /tmp/d.zip -d /opt && rm /tmp/d.zip",
    "chmod -R a+rX,go-w /opt/elan /opt/dafny && useradd -m -u 1000 -s /bin/bash grader",
]


# ---------------------------------------------------------------- process groups

_REAP = False  # set in a worker that run_worker started as the root of its own process group


def _group_pids(pgid: int) -> list[int]:
    """Live (non-zombie) processes in process group `pgid`."""
    if os.path.isdir("/proc/self"):  # Linux
        out = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat") as f:
                    s = f.read()
            except OSError:
                continue
            fields = s[s.rfind(")") + 2:].split()  # state, ppid, pgrp, ...
            if len(fields) > 2 and fields[2] == str(pgid) and fields[0] != "Z":
                out.append(int(name))
        return out
    try:  # macOS (local tests and DEV runs); `ps` itself is in the group, and left out
        ps = subprocess.Popen(["ps", "-A", "-o", "pid=", "-o", "pgid=", "-o", "stat="], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True)
        out = ps.communicate(timeout=60)[0]
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(f[0]) for f in (line.split() for line in out.splitlines())
            if len(f) >= 2 and f[1] == str(pgid) and int(f[0]) != ps.pid and not (len(f) > 2 and f[2].startswith("Z"))]


def kill_strays() -> int:
    """In a worker that run_worker started in its own process group: SIGKILL every other process of the group (not
    the worker, not its parent: runuser on Modal), e.g. a z3 that `dafny` left running after a timeout. No-op
    elsewhere. Returns the number of kill signals sent."""
    if not _REAP:
        return 0
    keep, pg, n = {os.getpid(), os.getppid()}, os.getpgrp(), 0
    for _ in range(50):
        pids = [p for p in _group_pids(pg) if p not in keep]
        if not pids:
            break
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
                n += 1
            except OSError:
                pass
        time.sleep(0.05)
    return n


def kill_group(pgid: int) -> None:
    """SIGKILL a whole process group (a finished or timed-out worker and everything it left running)."""
    for _ in range(3):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.05)


# ---------------------------------------------------------------- graders (run inside the worker process)

def _safe(run) -> dict:
    try:
        r = run()
    except Exception as e:  # noqa: BLE001 - any grader failure fails closed
        return {"status": "CHECK_ERROR", "violations": [f"{type(e).__name__}: {str(e)[:300]}"]}
    if not isinstance(r, dict) or not r.get("status"):
        return {"status": "CHECK_ERROR", "violations": [f"grader returned {str(r)[:200]}"]}
    return r


def vote(run, repeat: bool, between=None) -> dict:
    """One grader verdict. With `repeat`, a verdict that ran `dafny verify` is computed twice and a third time if the
    two differ; the majority wins, and a three-way split resolves to the first non-PROVED status. `between` (default
    kill_strays) runs before each re-run."""
    between = between or kill_strays
    t0 = time.time()
    runs = [_safe(run)]
    if repeat and "verify_ok" in runs[0]:
        between()
        runs.append(_safe(run))
        if runs[1]["status"] != runs[0]["status"]:
            between()
            runs.append(_safe(run))
    sts = [r["status"] for r in runs]
    top = max(sts, key=lambda s: (sts.count(s), s != "PROVED"))
    r = next(r for r in runs if r["status"] == top)
    return {"status": top, "runs": sts, "violations": [str(v)[:300] for v in (r.get("violations") or [])[:5]],
            "secs": round(time.time() - t0, 1)}


def dafny_lenient(text: str, frozen: dict, timeout: int = 300) -> dict:
    """Lenient Dafny grader: millennium_harness.dafny.check with its ban scan, `dafny audit` and `dafny translate`
    (the compilability check) switched off. Everything else is the rung's own code, so its frozen-part checks cannot drift
    from it. A renamed rung helper raises AttributeError here: CHECK_ERROR, never a silent pass."""
    from unittest import mock

    from millennium_harness import dafny as D
    real = D._dafny

    def dafny(args, cwd, t):
        return (0, "") if args[:1] in (["audit"], ["translate"]) else real(args, cwd, t)

    with mock.patch.object(D, "_bans", lambda segments: []), mock.patch.object(D, "_dafny", dafny):
        return D.check(text, frozen, "solution.dfy", timeout)


def grade_text(lang: str, text: str, frozen: dict, grader: str) -> dict:
    """Toolchains come from MILLENNIUM_LEAN / MILLENNIUM_DAFNY (set by run_worker)."""
    if lang == "lean":
        from millennium_harness import lean
        return vote(lambda: lean.check(text, frozen, bans=grader == "strict"), repeat=False)
    from millennium_harness import dafny
    if grader == "strict":
        return vote(lambda: dafny.check(text, frozen, "solution.dfy"), repeat=True)
    return vote(lambda: dafny_lenient(text, frozen), repeat=True)


def init_worker(job: dict) -> None:
    global _REAP
    _REAP = bool(job.get("reap"))


def _worker(sl_root: str, job_path: str) -> None:
    sys.path.insert(0, sl_root)
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    init_worker(job)
    sys.stdout.write(json.dumps(grade_text(job["lang"], job["text"], job["frozen"], job["grader"])) + "\n")


# ---------------------------------------------------------------- processes, toolchains (host or container)

def run_worker(job: dict, tools: dict, python: str = sys.executable, script: str = str(Path(__file__).resolve()),
               sl_root: str = str(ROOT), user: str | None = None, timeout: int = WORKER_TIMEOUT) -> dict:
    """One grader on one file in a fresh `python -I` process: clean environment, a private empty HOME/TMPDIR, as
    `user` if given (the container runs as root), as the root of a new process group that is killed when the worker
    ends or times out. Its stdout must be exactly one JSON verdict line."""
    d = tempfile.mkdtemp(prefix="sl-regrade-")
    io = tempfile.mkdtemp(prefix="sl-regrade-io-")  # outputs go to files: a leftover process cannot block a pipe
    try:
        Path(d, "job.json").write_text(json.dumps({**job, "reap": True}), encoding="utf-8")
        if user:
            for p in (d, f"{d}/job.json"):
                shutil.chown(p, user, user)
        env = ["/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", f"HOME={d}", f"TMPDIR={d}", "LANG=C.UTF-8",
               "DOTNET_CLI_TELEMETRY_OPTOUT=1", *(f"MILLENNIUM_{k.upper()}={v}" for k, v in sorted(tools.items()))]
        cmd = [*(["/usr/sbin/runuser", "-u", user, "--"] if user else []), *env, python, "-I", script, "_worker",
               sl_root, f"{d}/job.json"]
        with open(f"{io}/out", "wb") as out, open(f"{io}/err", "wb") as err:
            p = subprocess.Popen(cmd, cwd=d, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rc = None
            finally:
                kill_group(p.pid)
                if p.poll() is None:
                    p.kill()
                p.wait()
        if rc is None:
            return {"error": f"worker timed out after {timeout} s"}
        stdout, stderr = (Path(io, n).read_bytes()[:1 << 20].decode("utf-8", "replace") for n in ("out", "err"))
        lines = stdout.splitlines()
        try:
            r = json.loads(lines[0]) if rc == 0 and len(lines) == 1 else None
        except ValueError:
            r = None
        if not isinstance(r, dict) or not r.get("status"):
            return {"error": f"worker rc={rc}, {len(lines)} stdout lines: {(stderr or stdout)[-400:]}"}
        return r
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(io, ignore_errors=True)


def _sha_path(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def toolchain_hash(lang: str, exe: str) -> dict:
    """sha256 over the binaries a toolchain runs, by path relative to the toolchain root, plus its --version."""
    exe = Path(exe).resolve()
    if lang == "lean":
        root = exe.parent.parent
        files = [exe, *(p for p in sorted((root / "lib" / "lean").glob("*")) if p.suffix in (".so", ".dylib"))]
    else:
        root = exe.parent
        files = [exe, *sorted(root.glob("*.dll")), *(p for p in sorted((root / "z3").rglob("*")) if p.is_file())]
    lines = "".join(f"{_sha_path(p)}  {p.relative_to(root)}\n" for p in files)
    ver = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=120).stdout.strip()
    return {"version": ver.splitlines()[0][:120] if ver else "", "files": len(files),
            "digest": hashlib.sha256(lines.encode()).hexdigest()}


def checker_digest(sl_root: str | Path, script: str | Path) -> str:
    h = hashlib.sha256()
    for p in [*(Path(sl_root) / "millennium_harness" / f for f in CHECKER), Path(script)]:
        h.update(f"{p.name}\0{_sha_path(p)}\n".encode())
    return h.hexdigest()


def find_dafny() -> str | None:
    for exe in (os.environ.get("MILLENNIUM_DAFNY", os.environ.get("SORRYLESS_DAFNY")),
                str(Path.home() / ".local" / "dafny-4.11" / "dafny" / "dafny"), shutil.which("dafny")):
        if exe and os.path.isfile(exe):
            out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=120).stdout
            if out.startswith("4.11"):
                return exe
    return None


def grade_in_container(job: dict) -> dict:
    """The body of grade_remote: root in a fresh container. Checks the toolchain against the image build record,
    then runs each grader as the non-root user and kills whatever that user left running."""
    lang = job["lang"]
    got = toolchain_hash(lang, REMOTE[lang])
    out = {"toolchain": got["digest"], "checker": checker_digest(REMOTE["sl_root"], REMOTE["script"])}
    want = json.loads(Path(REMOTE["record"]).read_text())[lang]
    if got != want:
        return {**out, "error": f"{lang} toolchain differs from the image build record: {got} != {want}"}
    tools = {"lean": REMOTE["lean"], "dafny": REMOTE["dafny"]}
    for g in GRADERS:
        out[g] = run_worker({**job, "grader": g}, tools, sys.executable, REMOTE["script"], REMOTE["sl_root"],
                            REMOTE["user"])
        subprocess.run(["/usr/bin/pkill", "-9", "-u", REMOTE["user"]], capture_output=True)
    return out


def local_grade(jobs: list[dict], workers: int = 4) -> list[dict]:
    """Grade with this machine's pinned toolchains (Lean v4.23.0-rc2 via millennium_harness.lean.find_lean,
    Dafny 4.11)."""
    from millennium_harness import lean
    langs, tools = {j["lang"] for j in jobs}, {}
    if "lean" in langs:
        tools["lean"] = lean.find_lean()
    if "dafny" in langs:
        tools["dafny"] = find_dafny() or sys.exit("Dafny 4.11 not found (set MILLENNIUM_DAFNY)")
    tc = {lang: toolchain_hash(lang, exe)["digest"] for lang, exe in tools.items()}
    checker = checker_digest(ROOT, __file__)

    def one(job):
        return {"toolchain": tc[job["lang"]], "checker": checker,
                **{g: run_worker({**job, "grader": g}, tools) for g in GRADERS}}

    with ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(one, jobs))


def modal_grade(jobs: list[dict]) -> list[dict]:
    with modal.enable_output(), app.run():
        res = list(grade_remote.map(jobs, order_outputs=True, return_exceptions=True))
    return [r if isinstance(r, dict) else {"error": f"{type(r).__name__}: {str(r)[:300]}"} for r in res]


modal = None
if sys.argv[1:2] not in (["_worker"], ["_hashes"]):  # the worker processes never need Modal
    try:
        import modal
    except ImportError:  # --local does not either
        pass

if modal is not None:
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ca-certificates", "curl", "unzip", "xz-utils", "libicu72", "procps")
             .env({"ELAN_HOME": "/opt/elan", "DOTNET_CLI_TELEMETRY_OPTOUT": "1"})
             .run_commands(*IMAGE_CMDS)
             .add_local_file(Path(__file__).resolve(), REMOTE["script"], copy=True)
             .add_local_dir(ROOT / "millennium_harness", f"{REMOTE['sl_root']}/millennium_harness", copy=True,
                            ignore=["**/__pycache__", "**/*.pyc"])
             .run_commands(f"chmod -R a+rX,go-w /opt/sl && python3 -I {REMOTE['script']} _hashes > {REMOTE['record']}"
                           f" && chmod 444 {REMOTE['record']}"))
    app = modal.App("millennium-harness-fv-regrade", image=image)

    @app.function(cpu=1.0, memory=4096, timeout=7200, block_network=True, restrict_modal_access=True,
                  single_use_containers=True, max_containers=32)
    def grade_remote(job: dict) -> dict:
        """One unique file, strict then lenient, in a fresh single-use container without network."""
        return grade_in_container(job)


# ---------------------------------------------------------------- trials (host)

def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _flag(v) -> bool | None:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def read_events(p: Path) -> list[dict]:
    out = []
    for line in (p.read_text(errors="replace") if p.is_file() else "").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict) and "ev" in e:
            out.append(e)
    return out


def split_index(path: Path = SPLIT) -> dict:
    """{task id: (cell, part)} from split.json (ids only)."""
    s = _json(Path(path)) or {}
    return {i: (cell, part) for part in ("dev", "test") for cell, comps in (s.get(part) or {}).items()
            for c in comps for i in c["ids"]}


def egress_hosts(out: str | None) -> list[str]:
    """Probe lines where a non-model host answered: '<host> <code>' with any HTTP code but 000 (a TLS connection to
    that host was made, whatever the code)."""
    return [f"{m.group(1)} {m.group(2)}" for line in str(out or "").splitlines()
            if (m := EGRESS.match(line)) and m.group(2) != "000"]


def _norm(path: str) -> list[str]:
    """Archive path components ('/' = the archive root), with '.' and '..' applied ('..' stops at the root)."""
    out = []
    for c in path.split("/"):
        if c in ("", "."):
            continue
        if c == "..":
            if out:
                out.pop()
        else:
            out.append(c)
    return out


def resolve_member(members: dict, name: str) -> tuple[tarfile.TarInfo | None, str, int]:
    """Follow symlinks of `name` and of its parent directories inside the archive, as open() in the sandbox would.
    Returns (member or None, resolved archive path, hops); hops = -1 for a symlink loop."""
    parts, i, hops = _norm(name), 0, 0
    while i < len(parts):
        m = members.get("/".join(parts[:i + 1]))
        if m is not None and m.issym():
            hops += 1
            if hops > MAX_LINKS:
                return None, "/".join(parts), -1
            base = [] if m.linkname.startswith("/") else parts[:i]
            parts, i = _norm("/".join([*base, m.linkname, *parts[i + 1:]])), 0
            continue
        i += 1
    path = "/".join(parts)
    return members.get(path), path, hops


def read_snapshot(snapdir: Path, label: str, sol: str, ev: dict | None) -> dict:
    """The solution file in one downloaded snapshot: {sha, text} or {sha: None, text: None, note} (no usable file:
    UNPROVED), or {problems} (snapshot missing, failing its integrity checks, or a solution that cannot be read back
    from it: not graded)."""
    tar = snapdir / f"{label}.tar.gz"
    if not tar.is_file():
        return {"problems": [f"{label}: snapshot not downloaded"]}
    if not (ev or {}).get("tar_sha256"):
        return {"problems": [f"{label}: no in-sandbox sha256 for this snapshot"]}
    man = _json(snapdir / f"{label}.json")
    # Solution-only snapshots (bench/fv/shrink_snaps.py, or the lite download in agents_fv) no longer match the full
    # in-sandbox tar sha; the per-file sha256 in the in-sandbox manifest (checked below) then guards the graded file.
    if _sha_path(tar) != ev["tar_sha256"] and not (man or {}).get("manifest"):
        return {"problems": [f"{label}: snapshot sha256 differs from the in-sandbox one"]}
    roots = [_norm(r) for r in (man or {}).get("roots") or []]
    none = {"sha": None, "text": None}
    try:
        with tarfile.open(tar, "r:gz") as t:
            members = {}
            for m in t.getmembers():  # the last entry of a name wins, as extraction would leave it
                members["/".join(_norm(m.name))] = m
            m, path, hops = resolve_member(members, sol)
            if hops < 0:
                return {**none, "note": "the solution is a symlink loop"}
            if m is None:
                if not hops:
                    return {**none, "note": "no solution file in the snapshot"}
                if any(_norm(path)[:len(r)] == r for r in roots):  # dangles inside the snapshotted roots
                    return {**none, "note": f"the solution is a dangling symlink (to /{path})"}
                return {"problems": [f"{label}: the solution is a symlink to /{path}, outside the snapshot"]}
            if not (m.isreg() or m.islnk()):
                return {**none, "note": "the solution is not a regular file"}
            data = t.extractfile(m).read(MAX_BYTES + 1)  # a hard link reads its target's data
            if len(data) > MAX_BYTES:
                return {**none, "note": f"the solution is larger than {MAX_BYTES} bytes"}
    except (tarfile.TarError, OSError, EOFError, zlib.error, KeyError) as e:
        return {"problems": [f"{label}: unreadable snapshot ({type(e).__name__})"]}
    sha = hashlib.sha256(data).hexdigest()
    if man:
        entry = ((man.get("manifest") or {}).get("/" + path) or "")
        if not entry.startswith("f:"):
            return {"problems": [f"{label}: /{path} is not a regular file in the snapshot manifest"]}
        want = entry.rsplit(":", 1)[-1]
    else:
        want = ev.get("sol_sha") if not hops else None
    if want and want != sha:
        return {"problems": [f"{label}: solution sha256 differs from the snapshot manifest"]}
    try:  # read like the rung CLI does: UTF-8, universal newlines
        text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        return {"sha": sha, "text": None, "note": "the solution is not valid UTF-8"}
    return {"sha": sha, "text": text, **({"via": f"/{path}"} if hops else {})}


def _usage(events: list[dict]) -> dict:
    acc = dict.fromkeys(USAGE_KEYS, 0)
    for e in events:
        for k in USAGE_KEYS:
            v = (e.get("usage") or {}).get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                acc[k] += v
    acc["cost"] = round(acc["cost"], 6)
    return acc


def load_trial(d: Path, tasks_dir: str | None = None, split: dict | None = None, app_dir: str = "/app") -> dict:
    """Everything the regrade needs from one Harbor trial dir. `app_dir` is the in-sandbox /app (tests move it)."""
    from bench.agents_fv import SOLUTION
    cfg, res = _json(d / "config.json") or {}, _json(d / "result.json")
    ev = read_events(d / "agent" / "fv.jsonl")
    start = next((e for e in ev if e["ev"] == "start"), {})
    a = next((e for e in ev if e["ev"] == "A"), {})
    summary = next((e for e in reversed(ev) if e["ev"] == "summary"), None)
    tp = (cfg.get("task") or {}).get("path")
    name = Path(tp).name if tp else start.get("task")
    task = Path(tasks_dir) / name if tasks_dir and name else Path(tp) if tp else None
    meta = {}
    if task and (task / "task.toml").is_file():
        meta = tomllib.loads((task / "task.toml").read_text()).get("metadata", {})
    tid = meta.get("vericoding_id") or str(start.get("task") or d.name.split("__")[0]).upper()
    cell, part = meta.get("cell"), meta.get("part")
    if not cell:
        cell, part = (split or {}).get(tid, (None, None))
    lang = start.get("lang") or meta.get("category")
    agent_cfg = cfg.get("agent") or {}
    agent = start.get("agent") or (str(agent_cfg.get("name") or "").rsplit(":", 1)[-1] or None)
    kw = agent_cfg.get("kwargs") or {}
    want_c = _flag(kw["nudge"]) if "nudge" in kw else NUDGE_DEFAULT.get(agent)  # None: unknown agent, infer
    raw = (task / "tests" / "frozen.json").read_bytes() if task and (task / "tests" / "frozen.json").is_file() else None
    exc = ((res or {}).get("exception_info") or {}).get("exception_type")
    infra, warn = [], []
    if not ev:
        infra.append(f"no agent/fv.jsonl events{f' (Harbor: {exc})' if exc else ''}")
    if res is None:
        infra.append("no result.json: Harbor did not finish this trial")
    elif ev and exc:
        if exc in VERIFIER_EXC:
            warn.append(f"Harbor verifier-phase exception {exc} (after the snapshots; regrade does not use it)")
        else:
            infra.append(f"Harbor exception {exc}: the agent phase did not end normally")
    if ev and summary is None:
        infra.append("the trial did not finish (no summary event)")
    if ev and lang not in SOLUTION:
        infra.append(f"unknown language {lang!r}")
    if raw is None:
        infra.append(f"no tests/frozen.json under {task}")
    t = {"trial": d.name, "task": tid, "cell": cell, "part": part, "lang": lang, "agent": agent,
         "started_at": (res or {}).get("started_at"), "infra": infra, "warnings": warn, "exception": exc,
         "frozen": json.loads(raw) if raw else None, "frozen_sha": hashlib.sha256(raw).hexdigest() if raw else None,
         "snaps": {}, "branches": {}}
    if not ev:
        return t
    S = summary or {}
    sa, sb, sc = (S.get(x) if isinstance(S.get(x), dict) else {} for x in "ABC")
    if summary is not None:  # the summary is written in a `finally`: also when Harbor cancelled the run mid-branch
        if not sa.get("stop"):
            infra.append("A did not finish (no stop in the summary)")
        if "final" not in sb:
            infra.append("B did not finish (no final gate status in the summary): interrupted")
        if want_c is False and sc:
            infra.append(f"a C branch ran, but {agent} is configured without one")
        elif (want_c or sc) and "secs" not in sc:
            infra.append("C did not finish (no end time in the summary): interrupted")
        elif (want_c or sc) and not any(e["ev"] == "round" and str(e.get("label")).startswith("C") for e in ev):
            infra.append("C ran no nudge round: C would be S0, not the control")
    snap_ev = {e.get("label"): e for e in ev if e["ev"] == "snap"}
    sd = d / "agent" / "fv" / "snap"
    on_disk = {p.name[:-7] for p in sd.glob("*.tar.gz")} if sd.is_dir() else set()
    labels = sorted((x for x in set(snap_ev) | on_disk if x and LABEL.fullmatch(x)),
                    key=lambda x: ("SBC".index(x[0]), int(x[1:])))
    if lang in SOLUTION:
        sol = f"{app_dir}/{SOLUTION[lang].rsplit('/', 1)[1]}"
        t["snaps"] = {x: read_snapshot(sd, x, sol, snap_ev.get(x)) for x in labels}
    rounds = [e for e in ev if e["ev"] == "round"]
    br = [e for e in rounds if str(e.get("label")).startswith("B")]
    cr = [e for e in rounds if str(e.get("label")).startswith("C")]
    rv = [e for e in ev if e["ev"] in ("restore", "verify")]  # C's return to S0
    checks = [e for e in ev if e["ev"] == "check"]
    c_ran = bool(want_c or sc or cr or rv or any(e.get("branch") == "C" for e in ev))
    c_restore_ok = None  # no evidence (C never started, or no C): not a restore failure
    if c_ran and ("restore_ok" in sc or rv):
        c_restore_ok = sc.get("restore_ok", bool(rv and rv[-1].get("ok")))
    if c_restore_ok is not None and c_restore_ok is not True:
        infra.append("C did not start from S0: the restore/verify of S0 before C failed")
    if summary is not None and "final" in sb and not checks:
        infra.append("no gate check event")
    for e in checks:
        st, why = e.get("status"), e.get("error") or "; ".join(map(str, e.get("violations") or []))
        msg = (f"gate check {e.get('round')}{' (final)' if e.get('final') else ''}: {st}"
               + (f" ({why[:200]})" if why else ""))
        if st not in GATE_OK:
            infra.append(msg + ": B had no working gate")
        elif st == "CHECK_ERROR":  # the rung's own verdict: by design B does not resume on it
            warn.append(msg)
    eg = next((e for e in ev if e["ev"] == "egress"), None)
    if eg is None:
        warn.append("no egress probe in this trial")
    elif egress_hosts(eg.get("out")):
        infra.append(f"egress: non-model host reachable from the agent ({', '.join(egress_hosts(eg.get('out')))}; K2)")
    elif not str(eg.get("out") or "").strip():
        warn.append("the egress probe printed nothing")

    def secs(events):
        return round(sum(e.get("secs") or 0 for e in events), 1)

    t.update(
        s0_stop=a.get("stop"), gate=[e.get("status") for e in checks], egress=eg.get("out") if eg else None,
        b_fired=any(e.get("fired") for e in checks), b_rounds=len(br),
        c_rounds=len(cr) if c_ran else None, c_restore_ok=c_restore_ok,
        branches={"A": "S0", "B": br[-1]["label"] if br else "S0",
                  "C": (cr[-1]["label"] if cr else "S0") if c_ran else None},
        usage={"A": _usage([a]), "B": _usage(br), "C": _usage(cr) if c_ran else None},
        # wall time of each branch, gate checks / restore and snapshots included; agent_secs: model rounds only
        secs={"A": a.get("secs"), "B": sb.get("secs", secs(br + checks)),
              "C": sc.get("secs", secs(cr + rv)) if c_ran else None},
        agent_secs={"B": sb.get("agent_secs", secs(br)), "C": sc.get("agent_secs", secs(cr)) if c_ran else None})
    return t


def trial_row(job: str, t: dict, results: dict, grader: dict) -> dict:
    snaps, problems, tc = {}, {}, None
    for label, s in t["snaps"].items():
        e, p = {"sha": s.get("sha")}, list(s.get("problems") or [])
        if p:
            e.update(strict=None, lenient=None, problems=p)
        elif s.get("text") is None:  # a sound snapshot without a usable solution file: nothing verifies
            e.update(strict="UNPROVED", lenient="UNPROVED", note=s["note"])
        elif t["frozen"] is None:  # already in infra
            e.update(strict=None, lenient=None)
        else:
            if s.get("via"):
                e["via"] = s["via"]
            g = results.get(s["key"]) or {"error": "not graded"}
            tc = tc or g.get("toolchain")
            for name in GRADERS:
                r = g.get(name) or {}
                err = g.get("error") or r.get("error") or (None if r else "not graded")
                e[name] = None if err else r["status"]
                e.setdefault("runs", {})[name] = r.get("runs")
                if r.get("violations"):
                    e.setdefault("violations", {})[name] = r["violations"]
                if not err and r["status"] not in VERDICTS:  # CHECK_ERROR: the task or the grader failed
                    why = "; ".join(map(str, r.get("violations") or []))
                    err = f"grader status {r['status']}" + (f": {why[:300]}" if why else "")
                if err:
                    e.setdefault("errors", {})[name] = err
                    p.append(f"{label} {name}: {err}")
        snaps[label] = e
        problems[label] = p
    infra, warnings = list(t["infra"]), list(t.get("warnings") or [])
    used = {x for x in t["branches"].values() if x}
    for label, p in problems.items():  # only the snapshots A, B and C use can exclude the trial
        (infra if label in used else warnings).extend(p)

    def branch(label):
        if label is None:
            return None
        if label not in snaps:
            infra.append(f"no {label} snapshot")
            return {"label": label, "sha": None, "strict": None, "lenient": None}
        return {"label": label, **{k: snaps[label][k] for k in ("sha", "strict", "lenient")}}

    br = {k: branch(t["branches"].get(k)) for k in ("A", "B", "C")}
    return {"job": job, "trial": t["trial"], "task": t["task"], "cell": t["cell"], "part": t["part"],
            "lang": t["lang"], "agent": t["agent"], "started_at": t.get("started_at"), "s0_stop": t.get("s0_stop"),
            "b_fired": t.get("b_fired"), "b_rounds": t.get("b_rounds"), "c_rounds": t.get("c_rounds"),
            "gate": t.get("gate"), "c_restore_ok": t.get("c_restore_ok"), "egress": t.get("egress"), **br,
            "snaps": snaps, "usage": t.get("usage"), "secs": t.get("secs"), "agent_secs": t.get("agent_secs"),
            "infra": list(dict.fromkeys(infra)), "warnings": list(dict.fromkeys(warnings)),
            "exception": t["exception"], "grader": {**grader, "toolchain": tc}}


def _start_key(t: dict) -> tuple:
    """Trials in start order (a Harbor retry after its original); no start time last; then by name."""
    return (t.get("started_at") is None, str(t.get("started_at") or ""), t["trial"])


def regrade(job_dir: str | Path, grade=local_grade, out: str | Path | None = None, tasks_dir: str | None = None,
            mode: str = "local", app_dir: str = "/app") -> list[dict]:
    """Grade every unique file of a job once and write one row per trial, in start order. `grade(jobs) -> results`
    (same order)."""
    job_dir = Path(job_dir)
    split = split_index()
    trials = sorted((load_trial(d, tasks_dir, split, app_dir) for d in job_dir.iterdir()
                     if d.is_dir() and (d / "config.json").is_file()), key=_start_key)
    jobs = {}
    for t in trials:
        for s in t["snaps"].values():
            if s.get("text") is not None and t["frozen"] is not None:
                s["key"] = (s["sha"], t["frozen_sha"])
                jobs.setdefault(s["key"], {"lang": t["lang"], "text": s["text"], "frozen": t["frozen"]})
    results = dict(zip(jobs, grade(list(jobs.values())), strict=True)) if jobs else {}
    checker = checker_digest(ROOT, __file__)
    for r in results.values():
        if "error" not in r and r.get("checker") != checker:
            r["error"] = "graded by another checker version than this checkout (stale image?)"
    rows = [trial_row(job_dir.name, t, results, {"mode": mode, "checker": checker}) for t in trials]
    out = Path(out) if out else OUT_DIR / f"{job_dir.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} trials, {len(jobs)} unique files graded ({mode}) -> {out}")
    return rows


def summary(rows: list[dict]) -> str:
    """Per cell: trials, infra-flagged trials, B fired, and PROVED counts over the trials without infra."""
    def ok(rs, br, g):
        return sum(bool(r[br]) and r[br][g] == "PROVED" for r in rs if not r["infra"])

    lines = []
    for cell in sorted({r["cell"] or "?" for r in rows}):
        rs = [r for r in rows if (r["cell"] or "?") == cell]
        lines.append(f"{cell:7s} n={len(rs):3d} infra={sum(bool(r['infra']) for r in rs):2d} "
                     f"fired={sum(bool(r['b_fired']) for r in rs):3d}  "
                     + " | ".join(f"{g} A {ok(rs, 'A', g)} B {ok(rs, 'B', g)} C {ok(rs, 'C', g)}" for g in GRADERS))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir")
    ap.add_argument("--local", action="store_true", help="grade with this machine's toolchains instead of Modal")
    ap.add_argument("--out", default=None, help="default: runs/fv/grades/<job>.jsonl")
    ap.add_argument("--tasks-dir", default=None, help="where the task dirs are now (default: each trial's task path)")
    ap.add_argument("--workers", type=int, default=4, help="--local: files graded in parallel")
    a = ap.parse_args(argv)
    if a.local:
        grade, mode = (lambda jobs: local_grade(jobs, a.workers)), "local"
    elif modal is None:
        sys.exit("modal is not installed (or pass --local)")
    else:
        grade, mode = modal_grade, "modal"
    print(summary(regrade(a.job_dir, grade, a.out, a.tasks_dir, mode)))


if __name__ == "__main__":
    if sys.argv[1:2] == ["_worker"]:
        _worker(*sys.argv[2:4])
    elif sys.argv[1:2] == ["_hashes"]:
        print(json.dumps({lang: toolchain_hash(lang, REMOTE[lang]) for lang in ("lean", "dafny")}))
    else:
        main()
