"""bench/fv/regrade.py on synthetic Harbor trial dirs (no Modal, no Harbor, no API calls).

Trials come from the real FVMixin flow (fv.jsonl events, snapshot tarballs and manifests made by SNAP_SCRIPT) with a
scripted gate and a shell 'agent' that writes one file per round; /app-layout snapshots are built like fv_snap builds
them for root /app. Grading uses the local pinned toolchains (Lean v4.23.0-rc2, Dafny 4.11) and skips without them."""
import asyncio
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.adapters import vericoding_fv as ad
from bench.agents_fv import SOLUTION, FVMixin, _sha_file, oc_usage
from bench.fv import regrade as rg
from millennium_harness import dafny as sl_dafny
from millennium_harness import lean as sl_lean

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "bench" / "redteam"
try:
    LEAN = sl_lean.find_lean()
except (sl_lean.ToolchainError, OSError, subprocess.SubprocessError):
    LEAN = None
DAFNY = rg.find_dafny()
needs_lean = pytest.mark.skipif(LEAN is None, reason=f"Lean {sl_lean.TOOLCHAIN} not installed")
needs_dafny = pytest.mark.skipif(DAFNY is None, reason="Dafny 4.11 not installed (set MILLENNIUM_DAFNY)")
BLOCKED = "\n".join(f"{h} 000\n{h} blocked" for h in ("github.com", "huggingface.co", "atcoder.jp"))  # probe output


# ---------------------------------------------------------------- synthetic trials via the real FVMixin flow

def _ev(n):
    return json.dumps({"type": "step_finish", "sessionID": "ses_A",
                       "part": {"reason": "stop", "cost": n / 1000, "tokens": {"input": n, "output": 2 * n}}})


class Env:
    """Runs commands locally with bash; downloads are copies. User switches are ignored (user '-' has no home)."""
    default_user = "-"

    def __init__(self, name):
        self.environment_name = name

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        p = subprocess.run(["bash", "-c", command], capture_output=True, text=True, env={**os.environ, **(env or {})})
        return SimpleNamespace(return_code=p.returncode, stdout=p.stdout, stderr=p.stderr)

    async def download_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, target_path)


class Agent(FVMixin):
    """A -> S0 -> B -> C with the real snapshots and events; round <label> writes files[label] to the solution.
    cancel_at: the round label at which Harbor cancels the run (asyncio.wait_for); restore_fails: SNAP_SCRIPT's
    restore errors; egress: what the egress probe prints."""
    _usage = staticmethod(oc_usage)

    def __init__(self, box, trial, tasks, lang, files, statuses, nudge, cancel_at=None, restore_fails=False,
                 egress=BLOCKED):
        self.box, self.files, self.statuses = box, files, list(statuses)
        self.cancel_at, self.restore_fails, self.egress = cancel_at, restore_fails, egress
        self.sol = box / "app" / SOLUTION[lang].rsplit("/", 1)[1]
        self.logs_dir, self.environment_logs_dir = trial / "agent", box / "logs"
        self.fv_python, self.fv_snapdir = sys.executable, str(box / "snapdir")
        self.options = SimpleNamespace(max_rounds=2, initial_timeout_sec=60, branch_timeout_sec=600,
                                       rung_timeout_sec=60, nudge=nudge, tasks_dir=str(tasks))

    def _fv_fixed_roots(self):
        return [str(self.sol.parent)]

    def _fv_main_log(self):
        return f"{self.environment_logs_dir}/main.txt"

    def _fv_sol(self, lang):
        return str(self.sol)

    async def _fv_run_harness(self, instruction, environment, context):
        self.sol.parent.mkdir(parents=True)
        self.sol.write_text(self.files["S0"])
        Path(self._fv_main_log()).write_text(_ev(10) + "\n")

    async def _root(self, environment, cmd):
        if "curl" in cmd:  # the egress probe
            return SimpleNamespace(return_code=0, stdout=self.egress, stderr="")
        return await super()._root(environment, cmd)

    async def _sh(self, environment, op, label="-", roots=None):
        if op == "restore" and self.restore_fails:
            return {"error": "tar: Cannot open: No such file or directory"}
        return await super()._sh(environment, op, label, roots)

    async def _rung(self, environment, rnd):
        return {"status": self.statuses.pop(0), "feedback": "fix it"}

    async def _round(self, environment, message, label, budget):
        if label == self.cancel_at:
            raise asyncio.CancelledError  # what Harbor's asyncio.wait_for does at the agent timeout
        return await super()._round(environment, message, label, budget)

    def _round_cmd(self, message, out, budget):
        label = Path(out).stem
        (self.box / f"{label}.in").write_text(self.files[label])
        return (f"cp {self.box}/{label}.in {self.sol} && echo {shlex.quote(_ev(5))} | tee -a {self._fv_main_log()} "
                f"{out}", {})


def _result_json(trial: Path, started_at: str | None, exception: str | None = None) -> None:
    (trial / "result.json").write_text(json.dumps({
        "trial_name": trial.name, "started_at": started_at, "finished_at": started_at,
        "exception_info": {"exception_type": exception, "exception_message": "x"} if exception else None}))


def make_trial(tmp, task_id, lang, cell, frozen, files, statuses, nudge=True, skeleton=None, exception=None,
               started_at="2026-10-12T10:00:00.000000Z", **agent_kw):
    """A Harbor-style trial dir (config.json -> task dir with task.toml and tests/frozen.json, result.json) run
    through FVMixin."""
    name, sol = task_id.lower(), SOLUTION[lang].rsplit("/", 1)[1]
    td = tmp / "tasks" / name
    (td / "tests").mkdir(parents=True, exist_ok=True)
    (td / "environment").mkdir(exist_ok=True)
    (td / "environment" / sol).write_text(skeleton or files["S0"])
    (td / "tests" / "frozen.json").write_text(json.dumps(frozen))
    (td / "task.toml").write_text(ad.task_toml(lang, cell, task_id, "dev"))
    trial = tmp / "job" / f"{name}__T{len(list((tmp / 'job').glob('*'))) if (tmp / 'job').exists() else 0}"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"task": {"path": str(td)}, "trial_name": trial.name,
                                                   "agent": {"name": "tests:Agent", "kwargs": {"nudge": nudge}}}))
    box = tmp / "box"
    shutil.rmtree(box, ignore_errors=True)
    (box / "logs").mkdir(parents=True)
    try:
        asyncio.run(Agent(box, trial, tmp / "tasks", lang, files, statuses, nudge, **agent_kw)._fv_flow(
            "do it", Env(name), None))
    except asyncio.CancelledError:
        pass
    _result_json(trial, started_at, exception)
    return trial


class Counting:
    """Wraps a grade function and records how many unique files it was asked to grade."""

    def __init__(self, fn):
        self.fn, self.jobs = fn, []

    def __call__(self, jobs):
        self.jobs += jobs
        return self.fn(jobs)


def _verdict(status):
    return {"status": status, "runs": [status],
            **({"violations": ["task error: the frozen parts do not resolve"]} if status == "CHECK_ERROR" else {})}


def stub(status="PROVED", lenient=None, by_text=None):
    """A grader stub: (strict, lenient) = by_text[text] if given, else (status, lenient or status)."""
    checker = rg.checker_digest(rg.ROOT, rg.__file__)

    def one(job):
        s, le = (by_text or {}).get(job["text"], (status, lenient or status))
        return {"toolchain": "tc", "checker": checker, "strict": _verdict(s), "lenient": _verdict(le)}

    return Counting(lambda jobs: [one(j) for j in jobs])


def _regrade(tmp, grade, **kw):
    rows = rg.regrade(tmp / "job", grade, out=tmp / "grades.jsonl", app_dir=str(tmp / "box" / "app"), **kw)
    assert [json.loads(x) for x in (tmp / "grades.jsonl").read_text().splitlines()] == rows
    return rows


def _summary(trial: Path) -> dict:
    return next(e for e in reversed(rg.read_events(trial / "agent" / "fv.jsonl")) if e["ev"] == "summary")


# ---------------------------------------------------------------- end to end with the local toolchains

@needs_lean
def test_lean_trial_b_fires_twice_c_runs_dedupe_and_strict_vs_lenient(tmp_path):
    rec = json.loads((RT / "lean" / "tasks" / "LT0070.json").read_text())
    (tmp_path / "fz").mkdir()
    (tmp_path / "fz" / "LT0070.json").write_text(json.dumps(sl_lean.lock(rec, LEAN)))
    s0, frozen = ad.lean_files(rec, tmp_path / "fz")
    cheat = (RT / "lean" / "cheats" / "skip_kernel_tc.lean").read_text()  # strict CHEAT_REJECTED, lenient PROVED
    honest = (RT / "lean" / "honest" / "LT0070.lean").read_text()
    make_trial(tmp_path, "LT0070", "lean", "L-np", frozen,
               {"S0": s0, "B1": cheat, "B2": honest, "C1": s0, "C2": honest},
               ["UNPROVED", "CHEAT_REJECTED", "PROVED"])
    grade = Counting(lambda jobs: rg.local_grade(jobs, workers=3))
    [r] = _regrade(tmp_path, grade)
    assert len(grade.jobs) == 3  # S0 (= C1), B1, B2 (= C2): each unique file once
    assert (r["task"], r["cell"], r["part"], r["lang"], r["agent"]) == ("LT0070", "L-np", "dev", "lean", "Agent")
    assert r["s0_stop"] == "end_turn" and r["b_fired"] and r["b_rounds"] == 2 and r["c_rounds"] == 2
    assert r["gate"] == ["UNPROVED", "CHEAT_REJECTED", "PROVED"] and r["c_restore_ok"] and r["infra"] == []
    assert r["warnings"] == [] and r["egress"] == BLOCKED and r["started_at"] == "2026-10-12T10:00:00.000000Z"
    # the untouched skeleton: its `sorry`s are banned (strict) and compile to a `sorry` warning (lenient)
    assert r["A"] == {"label": "S0", "sha": hashlib.sha256(s0.encode()).hexdigest(), "strict": "CHEAT_REJECTED",
                      "lenient": "UNPROVED"}
    assert (r["B"]["label"], r["B"]["strict"], r["B"]["lenient"]) == ("B2", "PROVED", "PROVED")
    assert (r["C"]["label"], r["C"]["strict"], r["C"]["lenient"]) == ("C2", "PROVED", "PROVED")
    s = r["snaps"]
    assert list(s) == ["S0", "B1", "B2", "C1", "C2"]
    assert (s["B1"]["strict"], s["B1"]["lenient"]) == ("CHEAT_REJECTED", "PROVED") and s["B1"]["violations"]["strict"]
    assert s["C1"]["sha"] == s["S0"]["sha"] and s["C2"]["sha"] == s["B2"]["sha"] == r["B"]["sha"]
    assert s["S0"]["runs"] == {"strict": ["CHEAT_REJECTED"], "lenient": ["UNPROVED"]}  # Lean runs once
    assert r["usage"]["A"]["in"] == 10 and r["usage"]["B"]["in"] == 10 and r["usage"]["C"]["out"] == 20
    assert r["usage"]["B"]["cost"] == pytest.approx(0.01) and r["usage"]["B"]["steps"] == 2
    assert r["grader"]["mode"] == "local" and len(r["grader"]["toolchain"]) == 64


@needs_dafny
def test_dafny_trial_gate_silent_b_is_s0_and_verify_runs_twice(tmp_path):
    fz = json.loads((RT / "dafny" / "tasks" / "DA0545.json").read_text())
    s0, frozen = ad.dafny_files({"vc-preamble": fz["preamble"], "vc-spec": fz["spec"], "vc-postamble": fz["postamble"]})
    cheat = (RT / "dafny" / "cheats" / "assume_false.dfy").read_text()  # strict CHEAT_REJECTED, lenient PROVED
    honest = (RT / "dafny" / "honest" / "DA0545_transform_loop.dfy").read_text()
    make_trial(tmp_path, "DA0545", "dafny", "D-apps", frozen, {"S0": s0, "C1": cheat, "C2": honest}, ["PROVED"])
    grade = Counting(lambda jobs: rg.local_grade(jobs, workers=3))
    [r] = _regrade(tmp_path, grade)
    assert len(grade.jobs) == 3 and r["infra"] == [] and r["warnings"] == []
    assert not r["b_fired"] and r["b_rounds"] == 0 and r["gate"] == ["PROVED"] and r["c_rounds"] == 2
    assert r["B"] == {**r["A"], "label": "S0"} and (r["A"]["strict"], r["A"]["lenient"]) == ("UNPROVED", "UNPROVED")
    assert (r["C"]["label"], r["C"]["strict"], r["C"]["lenient"]) == ("C2", "PROVED", "PROVED")
    s = r["snaps"]
    assert (s["C1"]["strict"], s["C1"]["lenient"]) == ("CHEAT_REJECTED", "PROVED")
    assert "banned construct: assume" in s["C1"]["violations"]["strict"]
    for label in ("S0", "C1", "C2"):  # dafny verify ran and agreed twice: no third run
        assert [len(x) for x in s[label]["runs"].values()] == [2, 2], s[label]
    assert r["usage"]["B"]["in"] == 0 and r["usage"]["C"]["in"] == 10


# ---------------------------------------------------------------- flow parsing (stub grader, no toolchain)

def test_ab_only_agent_has_no_c_and_gate_check_error_does_not_resume(tmp_path):
    trial = make_trial(tmp_path, "LF0001", "lean", "L-fv", {"preamble": "", "postamble": "", "chunks": []},
                       {"S0": "theorem t : True := trivial\n"}, ["CHECK_ERROR"], nudge=False)
    grade = stub()
    [r] = _regrade(tmp_path, grade)
    assert len(grade.jobs) == 1 and r["infra"] == [] and r["C"] is None and r["c_rounds"] is None
    assert r["warnings"] == ["gate check 1: CHECK_ERROR"]  # the rung's own verdict: logged, by design no resume
    assert not r["b_fired"] and r["b_rounds"] == 0 and r["B"]["label"] == "S0" and r["A"]["strict"] == "PROVED"
    assert r["usage"]["C"] is None and r["secs"]["C"] is None and r["usage"]["A"]["in"] == 10
    sb = _summary(trial)["B"]  # B's time is the whole branch (the gate check included), not only model rounds
    assert r["secs"]["B"] == sb["secs"] and r["agent_secs"] == {"B": sb["agent_secs"], "C": None}


def test_one_task_many_trials_dedupe_across_trials(tmp_path):
    files = {"S0": "x", "B1": "y", "C1": "x", "C2": "y"}
    for _ in range(2):
        make_trial(tmp_path, "DT0001", "dafny", "D-np", {"preamble": "", "spec": "s", "postamble": ""}, files,
                   ["UNPROVED", "PROVED"])
    grade = stub("UNPROVED", "PROVED")
    rows = _regrade(tmp_path, grade)
    assert len(rows) == 2 and len(grade.jobs) == 2  # "x" and "y" for the same frozen.json
    assert all(r["B"]["label"] == "B1" and r["b_rounds"] == 1 and r["C"]["label"] == "C2" for r in rows)
    assert all((r["B"]["strict"], r["B"]["lenient"]) == ("UNPROVED", "PROVED") for r in rows)
    assert all(r["infra"] == [] and r["secs"]["C"] >= r["agent_secs"]["C"] for r in rows)


LF = {"preamble": "", "postamble": "", "chunks": []}
LFILES = {"S0": "a", "B1": "b", "B2": "c", "C1": "d", "C2": "e"}


# bug 1: Harbor's agent timeout cancels _fv_flow; its `finally` still writes a summary, so a summary is no proof
@pytest.mark.parametrize("exception", [None, "AgentTimeoutError"])
def test_trial_cancelled_during_c_is_infra(tmp_path, exception):
    trial = make_trial(tmp_path, "LF0002", "lean", "L-fv", LF, LFILES, ["UNPROVED", "PROVED"], cancel_at="C2",
                       exception=exception)
    assert "secs" not in _summary(trial)["C"]  # the summary exists, but C never ended
    [r] = _regrade(tmp_path, stub())
    assert r["C"]["label"] == "C1" and r["c_rounds"] == 1  # what the old regrade counted as a complete C
    want = ["C did not finish (no end time in the summary): interrupted"]
    if exception:
        want.insert(0, "Harbor exception AgentTimeoutError: the agent phase did not end normally")
    assert r["infra"] == want and r["exception"] == exception


def test_trial_cancelled_during_b_is_infra_for_ab_only_agents_too(tmp_path):
    make_trial(tmp_path, "LF0003", "lean", "L-fv", LF, LFILES, ["UNPROVED"], nudge=False, cancel_at="B1",
               exception="AgentTimeoutError")
    [r] = _regrade(tmp_path, stub())
    assert r["B"]["label"] == "S0" and r["C"] is None  # B = S0 and d = 0, as a valid trial, before the fix
    assert r["infra"] == ["Harbor exception AgentTimeoutError: the agent phase did not end normally",
                          "B did not finish (no final gate status in the summary): interrupted"]


def test_c_branch_that_never_started_is_infra_when_the_agent_runs_c(tmp_path):
    job, tasks = tmp_path / "job", tmp_path / "tasks"
    _app_trial(job, tasks, "DT0020", DFZ, {"S0": {"solution.dfy": GOOD}}, name="dt0020__a", agent="FVBranch",
               summary={"A": {"stop": "end_turn"}, "B": {"final": "PROVED", "secs": 1.0}, "C": {}})
    _app_trial(job, tasks, "DT0021", DFZ, {"S0": {"solution.dfy": GOOD}}, name="dt0021__a", agent="ClaudeCodeFV",
               summary={"A": {"stop": "end_turn"}, "B": {"final": "PROVED", "secs": 1.0}, "C": {}})
    _app_trial(job, tasks, "DT0022", DFZ, {"S0": {"solution.dfy": GOOD}}, name="dt0022__a", agent="ClaudeCodeFV",
               summary={"A": {}, "B": {"final": "PROVED", "secs": 1.0}, "C": {"rounds": 0, "secs": 2.0}})
    _app_trial(job, tasks, "DT0025", DFZ, {"S0": {"solution.dfy": GOOD}}, name="dt0025__a", agent="FVBranch")
    fv, cc, odd, none = rg.regrade(job, stub(), out=tmp_path / "g.jsonl")
    assert fv["infra"] == ["C did not finish (no end time in the summary): interrupted"] and fv["C"]["label"] == "S0"
    assert none["infra"] == ["C ran no nudge round: C would be S0, not the control"] and none["c_rounds"] == 0
    assert cc["infra"] == [] and cc["C"] is None and cc["c_rounds"] is None
    assert odd["infra"] == ["A did not finish (no stop in the summary)",
                            "a C branch ran, but ClaudeCodeFV is configured without one"]


def test_harbor_verifier_phase_exception_only_warns_and_missing_result_json_is_infra(tmp_path):
    job, tasks = tmp_path / "job", tmp_path / "tasks"
    _app_trial(job, tasks, "DT0023", DFZ, {"S0": {"solution.dfy": GOOD}}, name="dt0023__a",
               exception="VerifierTimeoutError")
    gone = _app_trial(job, tasks, "DT0024", DFZ, {"S0": {"solution.dfy": GOOD}}, name="dt0024__a")
    (gone / "result.json").unlink()
    ver, nores = rg.regrade(job, stub(), out=tmp_path / "g.jsonl")
    assert ver["infra"] == [] and ver["exception"] == "VerifierTimeoutError"
    assert ver["warnings"] == ["Harbor verifier-phase exception VerifierTimeoutError (after the snapshots; regrade "
                               "does not use it)"]
    assert nores["infra"] == ["no result.json: Harbor did not finish this trial"] and nores["started_at"] is None


# bug 3: C's nudges ran from whatever state the failed restore left
def test_failed_restore_before_c_is_infra(tmp_path):
    make_trial(tmp_path, "LF0004", "lean", "L-fv", LF, LFILES, ["UNPROVED", "PROVED"], restore_fails=True)
    [r] = _regrade(tmp_path, stub())
    assert r["c_rounds"] == 2 and r["c_restore_ok"] is False and r["C"]["strict"] == "PROVED"
    assert r["infra"] == ["C did not start from S0: the restore/verify of S0 before C failed"]


# bug 4: the gate never produced a verdict, so B = S0 and d = 0 would count as "gate silent"
@pytest.mark.parametrize("statuses,msg", [
    (["GATE_ERROR"], "gate check 1: GATE_ERROR: B had no working gate"),
    (["NO_TASK"], "gate check 1: NO_TASK: B had no working gate"),
    (["UNPROVED", "UNPROVED", "GATE_ERROR"], "gate check 3 (final): GATE_ERROR: B had no working gate"),
])
def test_gate_errors_are_infra(tmp_path, statuses, msg):
    make_trial(tmp_path, "LF0005", "lean", "L-fv", LF, LFILES, statuses)
    [r] = _regrade(tmp_path, stub())
    assert r["gate"] == statuses and r["infra"] == [msg]


# ---------------------------------------------------------------- snapshot integrity (/app layout, as in the sandbox)

def _app_snap(sd: Path, label: str, members: dict, roots=("/app",)) -> dict:
    """<label>.tar.gz and <label>.json as fv_snap writes them: a dir member per root, then `members` in order. Keys are
    absolute paths or names under /app; values are str/bytes contents, ("link", target) symlinks or ("hard", key)
    hard links to an earlier member. Returns the snap event."""
    sd.mkdir(parents=True, exist_ok=True)
    man, data_of = {}, {}
    with tarfile.open(sd / f"{label}.tar.gz", "w:gz") as t:
        for r in roots:
            d = tarfile.TarInfo(r.lstrip("/"))
            d.type, d.mode = tarfile.DIRTYPE, 0o755
            t.addfile(d)
            man[r] = "d:755:1000:1000"
        for name, body in members.items():
            path = name if name.startswith("/") else f"/app/{name}"
            ti = tarfile.TarInfo(path.lstrip("/"))
            if isinstance(body, tuple) and body[0] == "link":
                ti.type, ti.linkname = tarfile.SYMTYPE, body[1]
                t.addfile(ti)
                man[path] = f"l:1000:1000:{body[1]}"
                continue
            if isinstance(body, tuple):  # ("hard", key): same inode as an earlier member
                target = body[1] if body[1].startswith("/") else f"/app/{body[1]}"
                ti.type, ti.linkname = tarfile.LNKTYPE, target.lstrip("/")
                t.addfile(ti)
                data_of[path] = data_of[target]
            else:
                data_of[path] = body.encode() if isinstance(body, str) else body
                ti.size, ti.mode = len(data_of[path]), 0o644
                t.addfile(ti, io.BytesIO(data_of[path]))
            man[path] = f"f:644:1000:1000:{hashlib.sha256(data_of[path]).hexdigest()}"
    (sd / f"{label}.json").write_text(json.dumps({"label": label, "roots": list(roots), "manifest": man}))
    sol = next((v for k, v in man.items() if k in SOLUTION.values()), "")
    return {"ev": "snap", "label": label, "tar_sha256": _sha_file(sd / f"{label}.tar.gz"),
            "sol_sha": sol.rsplit(":", 1)[-1] if sol.startswith("f:") else None, "download_ok": True}


def _app_trial(job: Path, tasks: Path, tid: str, frozen: dict, snaps: dict, rounds=(), finished=True,
               name=None, started_at="2026-10-12T10:00:00.000000Z", exception=None, agent="ClaudeCodeFV",
               egress=BLOCKED, summary=None, roots=("/app",)) -> Path:
    """A Dafny trial whose fv.jsonl is written by hand around /app-layout snapshots. The default agent runs A and B
    only; agent="FVBranch" runs C too."""
    td = tasks / tid.lower()
    (td / "tests").mkdir(parents=True, exist_ok=True)
    (td / "tests" / "frozen.json").write_text(json.dumps(frozen))
    (td / "task.toml").write_text(ad.task_toml("dafny", "D-np", tid, "dev"))
    trial = job / (name or f"{tid.lower()}__x")
    sd = trial / "agent" / "fv" / "snap"
    ev = [{"ev": "start", "agent": agent, "lang": "dafny", "task": tid.lower()}]
    ev += [{"ev": "egress", "out": egress}] if egress is not None else []
    ev += [{"ev": "A", "stop": "length", "usage": {"in": 7, "cost": 0.002}, "secs": 3.0}]
    ev += [_app_snap(sd, label, m, roots) for label, m in snaps.items()]
    ev += [{"ev": "check", "branch": "B", "round": 1, "status": "UNPROVED", "fired": bool(rounds)}]
    ev += [{"ev": "round", "label": x, "stop": "end_turn", "secs": 2.0, "usage": {"in": 1}} for x in rounds]
    if finished:
        ev.append({"ev": "summary", **(summary or {
            "A": {"stop": "length"}, "B": {"final": "UNPROVED", "secs": 9.0, "agent_secs": 4.0},
            "C": {} if agent == "ClaudeCodeFV" else {"restore_ok": True, "secs": 5.0, "agent_secs": 4.0}})})
    (trial / "agent" / "fv.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev) + "not json\n")
    (trial / "config.json").write_text(json.dumps({"task": {"path": str(td)}}))
    _result_json(trial, started_at, exception)
    return trial


DFZ = {"preamble": "", "spec": "method M() returns (r: int)\n  ensures r >= 0", "postamble": ""}
GOOD = DFZ["spec"] + "\n{\n  r := 1;\n}\n"


def test_snapshot_integrity_failures_and_unusable_files(tmp_path):
    job = tmp_path / "job"
    trial = _app_trial(job, tmp_path / "tasks", "DT0002", DFZ,
                       {"S0": {"solution.dfy": GOOD}, "B1": {"solution.dfy": GOOD + "// edit\n"},
                        "B2": {"solution.dfy": ("link", "/etc/passwd")}, "C1": {"notes.txt": "x"},
                        "C2": {"solution.dfy": b"\xff\xfe"}}, rounds=["B1", "B2", "C1", "C2"], finished=False,
                       agent="FVBranch")
    with open(trial / "agent/fv/snap/B1.tar.gz", "ab") as f:  # the download differs from the in-sandbox tar
        f.write(b"\0")
    grade = stub()
    [r] = rg.regrade(job, grade, out=tmp_path / "g.jsonl")
    s = r["snaps"]
    assert len(grade.jobs) == 1 and s["S0"]["strict"] == "PROVED" and r["s0_stop"] == "length"
    assert s["B1"] == {"sha": None, "strict": None, "lenient": None,
                       "problems": ["B1: snapshot sha256 differs from the in-sandbox one"]}
    # a symlink out of the snapshot cannot be graded: the sandbox's `cat` read a file the regrade does not have
    assert s["B2"] == {"sha": None, "strict": None, "lenient": None,
                       "problems": ["B2: the solution is a symlink to /etc/passwd, outside the snapshot"]}
    assert (s["C1"]["lenient"], s["C1"]["note"]) == ("UNPROVED", "no solution file in the snapshot")
    assert (s["C2"]["strict"], s["C2"]["note"]) == ("UNPROVED", "the solution is not valid UTF-8")
    assert r["B"]["label"] == "B2" and r["C"]["label"] == "C2" and r["b_fired"]
    assert r["infra"] == ["the trial did not finish (no summary event)",
                          "B2: the solution is a symlink to /etc/passwd, outside the snapshot"]
    assert r["warnings"] == ["B1: snapshot sha256 differs from the in-sandbox one"]  # B1 is not a branch outcome


def test_manifest_mismatch_missing_download_and_harbor_failure(tmp_path):
    job = tmp_path / "job"
    trial = _app_trial(job, tmp_path / "tasks", "DT0003", DFZ, {"S0": {"solution.dfy": GOOD}})
    man = json.loads((trial / "agent/fv/snap/S0.json").read_text())
    man["manifest"]["/app/solution.dfy"] = "f:644:1000:1000:" + "0" * 64
    (trial / "agent/fv/snap/S0.json").write_text(json.dumps(man))
    ev = (trial / "agent/fv.jsonl").read_text()
    (trial / "agent/fv.jsonl").write_text(ev + json.dumps({"ev": "snap", "label": "B1", "tar_sha256": "ab"}) + "\n"
                                          + json.dumps({"ev": "round", "label": "B1", "usage": {}}) + "\n")
    failed = job / "dt0004__y"
    failed.mkdir()
    (failed / "config.json").write_text(json.dumps({"task": {"path": str(tmp_path / "tasks" / "gone")}}))
    (failed / "result.json").write_text(json.dumps({"exception_info": {"exception_type": "ImageBuildError"},
                                                    "started_at": "2026-10-12T11:00:00Z"}))
    grade = stub()
    a, b = rg.regrade(job, grade, out=tmp_path / "g.jsonl")
    assert grade.jobs == [] and a["A"]["strict"] is None and a["B"]["label"] == "B1" and a["B"]["strict"] is None
    assert a["infra"] == ["S0: solution sha256 differs from the snapshot manifest", "B1: snapshot not downloaded"]
    assert b["task"] == "DT0004" and b["exception"] == "ImageBuildError" and b["A"] is None and b["C"] is None
    assert b["infra"][0] == "no agent/fv.jsonl events (Harbor: ImageBuildError)"
    assert b["infra"][1].startswith("no tests/frozen.json")


# bug 6: the sandbox's `cat` follows links, so the regrade must read the same bytes
def test_hard_and_symbolic_links_resolve_inside_the_snapshot(tmp_path):
    job = tmp_path / "job"
    g1, g2, g3 = GOOD, GOOD + "// 2\n", GOOD + "// 3\n"
    _app_trial(job, tmp_path / "tasks", "DT0006", DFZ, {
        "S0": {"backup.dfy": g1, "solution.dfy": ("hard", "backup.dfy")},  # tar stores the 2nd name as LNKTYPE
        "B1": {"real.dfy": g2, "solution.dfy": ("link", "real.dfy")},
        "B2": {"/tmp/w/s.dfy": g3, "solution.dfy": ("link", "/tmp/w/s.dfy")},
        "C1": {"solution.dfy": ("link", "gone.dfy")},
        "C2": {"solution.dfy": ("link", "a.dfy"), "a.dfy": ("link", "solution.dfy")},
    }, rounds=["B1", "B2", "C1", "C2"], roots=("/app", "/tmp/w"), agent="FVBranch")
    grade = stub(by_text={g1: ("PROVED", "PROVED"), g2: ("UNPROVED", "PROVED"), g3: ("PROVED", "PROVED")})
    [r] = rg.regrade(job, grade, out=tmp_path / "g.jsonl")
    s = r["snaps"]
    assert sorted(j["text"] for j in grade.jobs) == sorted([g1, g2, g3])
    assert s["S0"]["sha"] == hashlib.sha256(g1.encode()).hexdigest() and s["S0"]["strict"] == "PROVED"
    assert (s["B1"]["strict"], s["B1"]["lenient"], s["B1"]["via"]) == ("UNPROVED", "PROVED", "/app/real.dfy")
    assert (s["B2"]["strict"], s["B2"]["via"]) == ("PROVED", "/tmp/w/s.dfy")
    assert (s["C1"]["strict"], s["C1"]["note"]) == ("UNPROVED", "the solution is a dangling symlink (to /app/gone.dfy)")
    assert (s["C2"]["strict"], s["C2"]["note"]) == ("UNPROVED", "the solution is a symlink loop")
    assert r["infra"] == [] and r["warnings"] == []


def test_symlinked_directory_and_hardlink_size_cap(tmp_path, monkeypatch):
    sd = tmp_path / "snap"
    ev = _app_snap(sd, "S0", {"/tmp/w/solution.dfy": GOOD, "/app": ("link", "/tmp/w")}, roots=("/tmp/w",))
    s = rg.read_snapshot(sd, "S0", "/app/solution.dfy", ev)  # /app itself is a symlink into the snapshot
    assert s["text"] == GOOD and s["via"] == "/tmp/w/solution.dfy"
    monkeypatch.setattr(rg, "MAX_BYTES", 10)  # a hard link has size 0 in the tar: the cap must use its data
    ev = _app_snap(sd, "B1", {"x.dfy": GOOD, "solution.dfy": ("hard", "x.dfy")})
    assert rg.read_snapshot(sd, "B1", "/app/solution.dfy", ev)["note"] == "the solution is larger than 10 bytes"


# bug 7: B1/C1 are not outcomes; only problems with the snapshots A, B and C use may exclude the trial
def test_problems_on_intermediate_snapshots_only_warn(tmp_path):
    job = tmp_path / "job"
    b1, b2, c1, c2 = (GOOD + f"// {x}\n" for x in ("b1", "b2", "c1", "c2"))
    trial = _app_trial(job, tmp_path / "tasks", "DT0007", DFZ,
                       {"S0": {"solution.dfy": GOOD}, "B1": {"solution.dfy": b1}, "B2": {"solution.dfy": b2},
                        "C1": {"solution.dfy": c1}, "C2": {"solution.dfy": c2}}, rounds=["B1", "B2", "C1", "C2"],
                       agent="FVBranch")
    (trial / "agent/fv/snap/B1.tar.gz").unlink()
    grade = stub(by_text={c1: ("CHECK_ERROR", "CHECK_ERROR")})
    [r] = rg.regrade(job, grade, out=tmp_path / "g.jsonl")
    assert r["infra"] == [] and (r["B"]["label"], r["C"]["label"]) == ("B2", "C2")
    assert r["warnings"] == ["B1: snapshot not downloaded",
                             "C1 strict: grader status CHECK_ERROR: task error: the frozen parts do not resolve",
                             "C1 lenient: grader status CHECK_ERROR: task error: the frozen parts do not resolve"]


# ---------------------------------------------------------------- grader results

# bug 2: CHECK_ERROR (a task that cannot be certified, or a crashing grader) is not a valid zero
def test_grader_check_error_is_infra_never_a_zero(tmp_path):
    _app_trial(tmp_path / "job", tmp_path / "tasks", "DT0008", DFZ, {"S0": {"solution.dfy": GOOD}})
    [r] = rg.regrade(tmp_path / "job", stub("CHECK_ERROR"), out=tmp_path / "g.jsonl")
    assert r["A"]["strict"] == r["B"]["lenient"] == "CHECK_ERROR" and r["C"] is None  # kept, and flagged
    assert r["infra"] == ["S0 strict: grader status CHECK_ERROR: task error: the frozen parts do not resolve",
                          "S0 lenient: grader status CHECK_ERROR: task error: the frozen parts do not resolve"]
    assert r["snaps"]["S0"]["errors"]["strict"].startswith("grader status CHECK_ERROR")
    assert "strict A 0 B 0 C 0" in rg.summary([r])


def test_rung_helper_failures_are_check_error(monkeypatch):
    monkeypatch.setattr(sl_dafny, "task_problems", lambda frozen, timeout=120: [])
    monkeypatch.setattr(sl_dafny, "_layout", lambda text, frozen: 1 / 0)  # e.g. a rung refactor broke a helper
    assert rg.grade_text("dafny", GOOD, DFZ, "strict")["status"] == "CHECK_ERROR"
    assert rg.grade_text("dafny", GOOD, DFZ, "lenient")["status"] == "CHECK_ERROR"
    monkeypatch.delattr(sl_dafny, "_bans")  # lenient switches the ban scan off by name: a rename fails closed
    r = rg.grade_text("dafny", GOOD, DFZ, "lenient")
    assert r["status"] == "CHECK_ERROR" and "AttributeError" in r["violations"][0]


def test_dafny_lenient_is_the_rung_without_bans_audit_and_translate(monkeypatch):
    seen = {}

    def check(text, frozen, path, timeout=300):
        seen.update(bans=sl_dafny._bans(["assume false;"]), audit=sl_dafny._dafny(["audit", "x"], ".", 1),
                    translate=sl_dafny._dafny(["translate", "py"], ".", 1), args=(text, frozen, path, timeout))
        return {"status": "PROVED", "verify_ok": True}

    monkeypatch.setattr(sl_dafny, "check", check)
    monkeypatch.setattr(sl_dafny, "_dafny", lambda args, cwd, t: (7, "real dafny"))
    assert rg.dafny_lenient("txt", DFZ) == {"status": "PROVED", "verify_ok": True}
    assert seen == {"bans": [], "audit": (0, ""), "translate": (0, ""), "args": ("txt", DFZ, "solution.dfy", 300)}
    assert sl_dafny._dafny(["verify"], ".", 1) == (7, "real dafny")  # verify still runs; patches are undone
    assert sl_dafny._bans(["assume false;"])  # the ban scan is back after the call


def test_grader_errors_and_stale_checker_never_count(tmp_path):
    _app_trial(tmp_path / "job", tmp_path / "tasks", "DT0005", DFZ, {"S0": {"solution.dfy": GOOD}})
    [r] = rg.regrade(tmp_path / "job", lambda jobs: [{"error": "boom"}], out=tmp_path / "g.jsonl")
    assert r["A"]["strict"] is None and r["snaps"]["S0"]["errors"] == {"strict": "boom", "lenient": "boom"}
    stale = [{"checker": "old", "strict": {"status": "PROVED"}, "lenient": {"status": "PROVED"}}]
    [r] = rg.regrade(tmp_path / "job", lambda jobs: stale, out=tmp_path / "g.jsonl")
    assert r["A"]["strict"] is None and "another checker version" in r["infra"][0]


# ---------------------------------------------------------------- trial order, egress

# bug 8: the first usable trial per id must be the earliest, not the one with the smallest random name suffix
def test_rows_are_in_trial_start_order(tmp_path):
    job, tasks = tmp_path / "job", tmp_path / "tasks"
    _app_trial(job, tasks, "DT0009", DFZ, {"S0": {"solution.dfy": GOOD}}, name="dt0009__aaa",
               started_at="2026-10-12T12:00:00.000000Z")  # the retry
    _app_trial(job, tasks, "DT0009", DFZ, {"S0": {"solution.dfy": GOOD}}, name="dt0009__zzz",
               started_at="2026-10-12T09:00:00.000000Z")  # the original
    gone = _app_trial(job, tasks, "DT0009", DFZ, {"S0": {"solution.dfy": GOOD}}, name="dt0009__000")
    (gone / "result.json").unlink()
    rows = rg.regrade(job, stub(), out=tmp_path / "g.jsonl")
    assert [r["trial"] for r in rows] == ["dt0009__zzz", "dt0009__aaa", "dt0009__000"]
    assert [r["started_at"] for r in rows] == ["2026-10-12T09:00:00.000000Z", "2026-10-12T12:00:00.000000Z", None]


@pytest.mark.parametrize("egress,infra,warnings", [
    (BLOCKED, [], []),
    ("github.com 000\ngithub.com blocked\nhuggingface.co 200\natcoder.jp 403",
     ["egress: non-model host reachable from the agent (huggingface.co 200, atcoder.jp 403; K2)"], []),
    (None, [], ["no egress probe in this trial"]),
    ("", [], ["the egress probe printed nothing"]),
])
def test_egress_probe_answers_from_non_model_hosts_are_infra(tmp_path, egress, infra, warnings):
    _app_trial(tmp_path / "job", tmp_path / "tasks", "DT0010", DFZ, {"S0": {"solution.dfy": GOOD}}, egress=egress)
    [r] = rg.regrade(tmp_path / "job", stub(), out=tmp_path / "g.jsonl")
    assert (r["infra"], r["warnings"], r["egress"]) == (infra, warnings, egress)
    assert ("strict A 0" if infra else "strict A 1") in rg.summary([r])  # summary counts trials without infra only


# ---------------------------------------------------------------- pieces

@pytest.mark.parametrize("seq,status,n", [
    (["PROVED", "PROVED"], "PROVED", 2), (["PROVED", "UNPROVED", "UNPROVED"], "UNPROVED", 3),
    (["UNPROVED", "PROVED", "PROVED"], "PROVED", 3), (["PROVED", "UNPROVED", "CHECK_ERROR"], "UNPROVED", 3),
])
def test_vote_verifies_twice_and_a_third_time_on_disagreement(seq, status, n):
    it, calls = iter(seq), []
    r = rg.vote(lambda: {"status": next(it), "verify_ok": True}, repeat=True, between=lambda: calls.append(1))
    assert (r["status"], len(r["runs"])) == (status, n) and r["runs"] == seq[:n] and len(calls) == n - 1


def test_vote_without_verify_or_for_lean_runs_once_and_fails_closed():
    assert rg.vote(lambda: {"status": "SPEC_TAMPERED"}, repeat=True)["runs"] == ["SPEC_TAMPERED"]
    assert rg.vote(lambda: {"status": "PROVED", "verify_ok": True}, repeat=False)["runs"] == ["PROVED"]
    assert rg.vote(lambda: 1 / 0, repeat=True)["status"] == "CHECK_ERROR"
    assert rg.vote(lambda: None, repeat=True)["status"] == "CHECK_ERROR"
    assert rg.kill_strays() == 0  # outside a worker started by run_worker, reaping is a no-op


def test_worker_output_must_be_exactly_one_verdict_line(tmp_path, monkeypatch):
    bad = tmp_path / "w.py"
    bad.write_text("print('{\"status\": \"PROVED\"}')\nprint('{\"status\": \"PROVED\"}')\n")
    assert "2 stdout lines" in rg.run_worker({"lang": "lean"}, {}, script=str(bad))["error"]
    bad.write_text("import time; time.sleep(5)\n")
    assert "timed out" in rg.run_worker({"lang": "lean"}, {}, script=str(bad), timeout=1)["error"]


def _dead(pid: int, wait: float = 10.0) -> bool:
    end = time.time() + wait
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


LEAVER = textwrap.dedent("""\
    import json, subprocess, sys, time
    job = json.load(open(sys.argv[3]))
    p = subprocess.Popen(["sleep", "300"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    open(job["pidfile"], "w").write(str(p.pid))
    if job.get("hang"):
        time.sleep(60)
    print(json.dumps({"status": "PROVED"}))
""")


# bug 5: what a worker leaves running (z3 after a dafny timeout) must not outlive it
@pytest.mark.parametrize("hang", [False, True])
def test_worker_process_group_is_killed_when_it_ends_or_times_out(tmp_path, hang):
    (tmp_path / "w.py").write_text(LEAVER)
    pidfile = tmp_path / "pid"
    r = rg.run_worker({"lang": "dafny", "pidfile": str(pidfile), "hang": hang}, {}, script=str(tmp_path / "w.py"),
                      timeout=3)
    assert ("timed out" in r["error"]) if hang else r["status"] == "PROVED"
    assert _dead(int(pidfile.read_text()))


def test_vote_reruns_start_after_leftovers_are_killed(tmp_path):
    (tmp_path / "w.py").write_text(textwrap.dedent(f"""\
        import json, subprocess, sys
        sys.path.insert(0, {str(ROOT)!r})
        from bench.fv import regrade as rg
        rg.init_worker(json.load(open(sys.argv[3])))
        left = []

        def run():  # each run leaves a process behind, like `dafny` leaves z3 after its timeout
            alive = [p.pid for p in left if p.poll() is None]
            left.append(subprocess.Popen(["sleep", "300"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            return {{"status": "UNPROVED" if alive else "PROVED", "verify_ok": True, "violations": alive}}

        print(json.dumps({{**rg.vote(run, repeat=True), "last": left[-1].pid}}))
    """))
    r = rg.run_worker({"lang": "dafny"}, {}, script=str(tmp_path / "w.py"), timeout=60)
    assert r["runs"] == ["PROVED", "PROVED"], r  # run 2 found run 1's leftover gone
    assert _dead(r["last"])  # and run 2's was killed with the worker's group


@needs_dafny
def test_z3_left_by_a_dafny_timeout_dies_with_the_worker(tmp_path):
    slow = textwrap.dedent("""\
        function F(n: nat): nat { if n == 0 then 0 else F(n-1) + n }
        lemma L(a: nat, b: nat, c: nat)
          ensures F(a) * F(b) * F(c) == F(c) * F(b) * F(a) + F(a*b*c) - F(c*b*a)
          ensures (a*a*b + c) * (b*c*a + a) == (a*b*a + c) * (a*b*c + a)
          ensures F(a+b+c) * F(a) >= F(a) * F(b+c+a) || (a*b*c*a*b*c) % 7 == ((c*b*a*c*b*a) % 7)
        {}
    """)
    (tmp_path / "w.py").write_text(textwrap.dedent(f"""\
        import json, os, sys, tempfile
        sys.path.insert(0, {str(ROOT)!r})
        from bench.fv import regrade as rg
        from millennium_harness import dafny as D
        d = tempfile.mkdtemp()
        open(os.path.join(d, "slow.dfy"), "w").write({slow!r})
        rc, out = D._dafny(["verify", "--cores", "1", "slow.dfy"], d, 8)  # the rung's helper: kills dafny only
        z3 = [p for p in rg._group_pids(os.getpgrp()) if p != os.getpid()]
        print(json.dumps({{"status": "UNPROVED", "rc": rc, "z3": z3}}))
    """))
    r = rg.run_worker({"lang": "dafny"}, {"dafny": DAFNY}, script=str(tmp_path / "w.py"), timeout=120)
    if r.get("rc") is not None or not r.get("z3"):
        pytest.skip(f"dafny did not time out leaving z3 behind on this machine: {r}")
    assert all(_dead(p) for p in r["z3"])


def test_toolchain_hash_and_container_refuses_a_changed_toolchain(tmp_path, monkeypatch):
    tc = tmp_path / "lean"
    (tc / "bin").mkdir(parents=True)
    (tc / "lib" / "lean").mkdir(parents=True)
    (tc / "bin" / "lean").write_text("#!/bin/sh\necho 'Lean (version 4.23.0-rc2, fake)'\n")
    os.chmod(tc / "bin" / "lean", 0o755)
    (tc / "lib" / "lean" / "libleanshared.so").write_text("a")
    (tc / "lib" / "lean" / "Init.olean").write_text("not hashed")
    h = rg.toolchain_hash("lean", str(tc / "bin" / "lean"))
    assert h["files"] == 2 and h["version"].startswith("Lean (version 4.23.0-rc2")
    (tc / "lib" / "lean" / "Init.olean").write_text("changed")
    assert rg.toolchain_hash("lean", str(tc / "bin" / "lean")) == h
    (tc / "lib" / "lean" / "libleanshared.so").write_text("b")
    assert rg.toolchain_hash("lean", str(tc / "bin" / "lean"))["digest"] != h["digest"]
    (tmp_path / "record.json").write_text(json.dumps({"lean": h}))
    monkeypatch.setitem(rg.REMOTE, "lean", str(tc / "bin" / "lean"))
    monkeypatch.setitem(rg.REMOTE, "record", str(tmp_path / "record.json"))
    monkeypatch.setitem(rg.REMOTE, "sl_root", str(ROOT))
    monkeypatch.setitem(rg.REMOTE, "script", rg.__file__)
    r = rg.grade_in_container({"lang": "lean", "text": "", "frozen": {}})
    assert "toolchain differs from the image build record" in r["error"] and "strict" not in r


def test_modal_function_is_defined_not_run():
    modal = pytest.importorskip("modal")
    assert isinstance(rg.app, modal.App) and rg.app.name == "millennium-harness-fv-regrade"
    assert isinstance(rg.grade_remote, modal.Function)


def test_grading_image_installs_lean_like_the_fixed_agent_image():
    """The smoke found that elan no longer installs the default toolchain eagerly (FINDINGS, FV smoke)."""
    docker = (ROOT / "bench" / "fv" / "images" / "lean.Dockerfile").read_text()
    for part in ("--default-toolchain none", f"elan toolchain install {rg.LEAN_TOOLCHAIN}",
                 f"elan default {rg.LEAN_TOOLCHAIN}"):
        assert part in docker and any(part in c for c in rg.IMAGE_CMDS), part
    assert not any(f"--default-toolchain {rg.LEAN_TOOLCHAIN}" in c for c in rg.IMAGE_CMDS)


@needs_dafny
def test_cli_local_on_app_layout_snapshots(tmp_path):
    fz = json.loads((RT / "dafny" / "tasks" / "DT0254.json").read_text())
    honest = (RT / "dafny" / "honest" / "DT0254_array_equal_loop.dfy").read_text()
    _app_trial(tmp_path / "job", tmp_path / "tasks", "DT0254", fz, {"S0": {"solution.dfy": honest}})
    out = subprocess.run([sys.executable, str(ROOT / "bench/fv/regrade.py"), str(tmp_path / "job"), "--local",
                          "--out", str(tmp_path / "g.jsonl")], capture_output=True, text=True, cwd=tmp_path,
                         env={**os.environ, "MILLENNIUM_DAFNY": DAFNY}, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    [r] = [json.loads(x) for x in (tmp_path / "g.jsonl").read_text().splitlines()]
    assert (r["A"]["strict"], r["A"]["lenient"], r["infra"]) == ("PROVED", "PROVED", [])
    assert r["usage"]["A"] == {"in": 7, "out": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0, "cost": 0.002,
                               "steps": 0}
    assert "D-np" in out.stdout and "strict A 1 B 1 C 0" in out.stdout


def test_real_smoke_snapshot_layout_if_present():
    """The /app layout assumption against a real downloaded S0 (gitignored run data; skipped when absent)."""
    trials = sorted(ROOT.glob("runs/jobs/fvsmoke*/*/agent/fv/snap/S0.tar.gz"))
    if not trials:
        pytest.skip("no FV smoke snapshots on disk")
    for tar in trials[:3]:
        ev = {e["label"]: e for e in rg.read_events(tar.parents[2] / "fv.jsonl") if e["ev"] == "snap"}
        lang = next(e["lang"] for e in rg.read_events(tar.parents[2] / "fv.jsonl") if e["ev"] == "start")
        s = rg.read_snapshot(tar.parent, "S0", SOLUTION[lang], ev.get("S0"))
        assert "problems" not in s and s["sha"] == ev["S0"]["sol_sha"], s.get("problems")


def test_real_smoke_trial_cut_by_the_agent_timeout_is_infra_if_present(tmp_path):
    """fvsmoke1 dt0056: AgentTimeoutError at 2700 s during C (gitignored run data; skipped when absent)."""
    trial = next(iter(sorted(ROOT.glob("runs/jobs/fvsmoke1-oc/dt0056__*"))), None)
    if trial is None:
        pytest.skip("no FV smoke trial dt0056 on disk")
    t = rg.load_trial(trial)
    assert "Harbor exception AgentTimeoutError: the agent phase did not end normally" in t["infra"]
    assert "C did not finish (no end time in the summary): interrupted" in t["infra"]
