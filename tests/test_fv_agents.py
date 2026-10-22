"""Hermetic tests for the FV adapter and the same-trajectory agents (no Modal, no Harbor, no API calls)."""
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.adapters import vericoding_fv as ad
from bench.agents_fv import (NUDGE, SNAP_SCRIPT, FVMixin, cc_round_cmd, cc_usage, fs_manifest, fv_restore, fv_snap,
                             fv_verify, oc_round_cmd, oc_usage, parse_rung, rung_cmd, stop_reason)
from millennium_harness import lean as sl_lean

ROOT = Path(__file__).resolve().parent.parent
LEAN_REC = {"id": "LA9999", "language": "lean", "source": "apps", "vc-description": "",
            "vc-preamble": "@[reducible, simp]\ndef pre (n : Nat) : Prop := n > 0", "vc-helpers": "",
            "vc-definitions": "def solve (n : Nat) (h : pre n) : Nat :=\n  sorry",
            "vc-theorems": "theorem solve_spec (n : Nat) (h : pre n) : solve n h ≥ 0 := by\n  sorry",
            "vc-postamble": ""}
DAFNY_REC = {"id": "DT9999", "language": "dafny", "source": "numpy_triple", "vc-description": "numpy.abs",
             "vc-preamble": "", "vc-spec": "method Abs(x: int) returns (r: int)\n  ensures r >= 0",
             "vc-code": "{\n  assume {:axiom} false;\n}", "vc-postamble": ""}


# ---------------------------------------------------------------- adapter

def test_instructions_name_the_plain_toolchain_and_rules_but_not_the_gate():
    lean, dafny = ad.instruction("lean", ""), ad.instruction("dafny", "numpy.abs")
    assert "    lean /app/Solution.lean\n" in lean and "    dafny verify /app/solution.dfy\n" in dafny
    for text in (lean, dafny):
        assert "You run autonomously: nobody will answer questions." in text
        assert "millennium" not in text.lower() and "sorryless" not in text.lower() and "gate" not in text.lower()
    assert all(w in lean for w in ("`sorry`", "`axiom`", "`native_decide`", "`@[implemented_by]`", "frozen"))
    assert all(w in dafny for w in ("`assume`", "`{:axiom}`", "`{:verify false}`", "`{:compile false}`", "`refines`"))
    assert "Task description:\nnumpy.abs" in dafny and "Task description" not in lean


def test_task_toml_nonroot_allowlist_timeouts():
    cfg = tomllib.loads(ad.task_toml("lean", "L-apps", "LA9999", "dev"))
    assert cfg["agent"] == {"timeout_sec": 4500.0, "user": "agent", "network_mode": "allowlist",
                            "allowed_hosts": ["api.anthropic.com"]}
    assert cfg["environment"]["network_mode"] == "allowlist"
    assert cfg["environment"]["allowed_hosts"] == ["api.anthropic.com"]
    assert cfg["verifier"]["timeout_sec"] == 900.0
    assert cfg["metadata"]["cell"] == "L-apps" and cfg["metadata"]["vericoding_id"] == "LA9999"


def test_test_sh_grades_with_the_shared_rung_command():
    sh = ad.test_sh("lean")
    assert rung_cmd("lean", "/tmp/sl-grade/Solution.lean", "/tmp/sl-grade/frozen.json", "/tmp/sl-grade/w") in \
        shlex.split(sh.split("runuser -u agent -- bash -c ", 1)[1].split(" > /logs", 1)[0])[0]
    assert "runuser -u agent -- cat /app/Solution.lean > $G/Solution.lean" in sh
    assert "/logs/verifier/grade.json" in sh and "== 'PROVED' else 0" in sh and "reward.txt" in sh
    assert "millennium_harness.dafny check /tmp/sl-grade/solution.dfy" in ad.test_sh("dafny")


def test_rung_cmd_is_isolated_and_matches_each_cli(tmp_path):
    c = rung_cmd("lean", "/w/Solution.lean", "/w/frozen.json", "/w/x", 1, 2)
    assert c.startswith("cd /w/x && env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/w/x TMPDIR=/w/x")
    assert "MILLENNIUM_LEAN=/opt/lean/bin/lean MILLENNIUM_DAFNY=/opt/dafny/dafny" in c
    assert c.endswith("/usr/local/bin/python3 -I -m millennium_harness.lean check /w/Solution.lean /w/frozen.json "
                      "--round 1 --max-rounds 2")
    assert rung_cmd("dafny", "/w/s.dfy", "/w/f.json", "/w/x", 2, 2).endswith("check /w/s.dfy /w/f.json 2 2")
    assert rung_cmd("dafny", "/w/s.dfy", "/w/f.json", "/w/x").endswith("check /w/s.dfy /w/f.json")
    # the real Lean CLI accepts exactly these arguments (a missing solution is UNPROVED without running Lean)
    (tmp_path / "frozen.json").write_text("{}")
    argv = c.split(" -m millennium_harness.lean ", 1)[1].replace("/w/frozen.json", str(tmp_path / "frozen.json")).split()
    out = subprocess.run([sys.executable, "-m", "millennium_harness.lean", *argv], capture_output=True, text=True, cwd=ROOT,
                         env={"PYTHONPATH": str(ROOT)})
    r = json.loads(out.stdout.splitlines()[-1]) if out.stdout else {"stderr": out.stderr}
    assert r.get("status") == "UNPROVED" and "round 1/2" in r["feedback"], r


def _lock_dir(tmp_path, *recs):
    """Model-free stand-in for `millennium_harness.lean lock` (chunks only; the real lock also needs the pinned Lean)."""
    for r in recs:
        (tmp_path / f"{r['id']}.json").write_text(json.dumps({
            "id": r["id"], "preamble": r["vc-preamble"].strip(), "postamble": r["vc-postamble"].strip(),
            "chunks": sl_lean._chunks(r), "constants": {}}))
    return tmp_path


def test_lean_skeleton_order_and_holes(tmp_path):
    sol, frozen = ad.lean_files(LEAN_REC, _lock_dir(tmp_path, LEAN_REC))
    order = [sol.index(x) for x in (LEAN_REC["vc-preamble"], ad.LEAN_BEGIN, ad.LEAN_END,
                                    LEAN_REC["vc-definitions"], LEAN_REC["vc-theorems"])]
    assert order == sorted(order) and sol.count("sorry") == 2 and frozen["id"] == "LA9999"
    hs, v = sl_lean.holes(sol, frozen)
    assert v == [] and all(x.startswith("banned construct: sorry ") for x in sl_lean.ban_scan(hs))  # markers: clean


def test_lean_needs_a_lock_that_matches_the_skeleton(tmp_path, monkeypatch):
    def no_lean(rec, *a, **k):
        raise sl_lean.LockError("Lean 4.23.0-rc2 not found")
    monkeypatch.setattr(sl_lean, "lock", no_lean)
    with pytest.raises(SystemExit, match="cannot be locked here"):
        ad.lean_files(LEAN_REC, tmp_path)
    _lock_dir(tmp_path, {**LEAN_REC, "vc-preamble": "def other : Nat := 1"})
    with pytest.raises(SystemExit, match="fails its own lock"):
        ad.lean_files(LEAN_REC, tmp_path)


def test_dafny_skeleton_matches_e4b_layout():
    sol, frozen = ad.dafny_files(DAFNY_REC)
    assert ad.HELPERS_BEGIN in sol and sol.index(DAFNY_REC["vc-spec"]) > sol.index(ad.HELPERS_END)
    assert "assume" not in sol and frozen == {"preamble": "", "spec": DAFNY_REC["vc-spec"], "postamble": ""}


def test_write_task_layout(tmp_path):
    td = ad.write_task(tmp_path / "out", LEAN_REC, "L-apps", "dev", _lock_dir(tmp_path, LEAN_REC))
    env = td / "environment"
    assert (env / "Dockerfile").read_text() == (ROOT / "bench/fv/images/lean.Dockerfile").read_text()
    assert (env / "Solution.lean").exists() and (env / "sl/millennium_harness/dafny.py").exists()
    assert not list(env.rglob("__pycache__"))
    assert os.access(td / "tests/test.sh", os.X_OK) and json.loads((td / "tests/frozen.json").read_text())
    assert "COPY --chown=agent:agent Solution.lean /app/Solution.lean" in (env / "Dockerfile").read_text()


def test_select_ids_dev_seeded_order_and_skip():
    split = {"dev": {"L-apps": [{"comp": "c1", "ids": ["LA1", "LA2"]}, {"comp": "c2", "ids": ["LA3"]},
                                {"comp": "c3", "ids": ["LA4"]}]}}
    assert ad.select_ids(split, "L-apps", "dev", 2) == ["LA1", "LA3"]
    assert ad.select_ids(split, "L-apps", "dev", 5, skip=1) == ["LA3", "LA4"]


def test_build_dev_from_committed_split(tmp_path):
    split = ROOT / "runs/fv/split.json"
    if not split.exists() or not ad.DATA.exists():
        pytest.skip("split.json / vericoding.jsonl not present")
    comps = json.loads(split.read_text())["dev"]
    for cell in ("D-np", "L-apps"):
        ids = [c["ids"][0] for c in comps[cell][:2]]
        recs = [json.loads(line) for line in open(ad.DATA) if json.loads(line)["id"] in ids]
        dirs = ad.build(str(tmp_path / cell), cell, str(split), "dev", 2, frozen_dir=_lock_dir(tmp_path, *recs))
        assert [d.name for d in dirs] == [i.lower() for i in ids]
        cfg = tomllib.loads((dirs[0] / "task.toml").read_text())
        assert cfg["agent"]["user"] == "agent" and cfg["metadata"]["cell"] == cell


def test_build_test_refuses_before_the_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "_tagged", lambda tag=ad.FROZEN_TAG: False)
    with pytest.raises(SystemExit):
        ad.build(str(tmp_path), "L-apps", str(ROOT / "runs/fv/split.json"), "test", 2)


# ---------------------------------------------------------------- parsing

OC_LOG = "\n".join(json.dumps(e) for e in [
    {"type": "step_start", "sessionID": "ses_A"},
    {"type": "step_finish", "sessionID": "ses_A", "part": {"reason": "tool-calls", "cost": 0.01,
                                                           "tokens": {"input": 10, "output": 5, "reasoning": 2,
                                                                      "cache": {"read": 100, "write": 7}}}},
    {"type": "step_finish", "sessionID": "ses_A", "part": {"reason": "stop", "cost": 0.02,
                                                           "tokens": {"input": 1, "output": 3, "cache": {"read": 50}}}},
]) + "\nnot json\n"


def test_oc_usage_and_stop_reason():
    u = oc_usage(OC_LOG)
    assert (u["in"], u["out"], u["reasoning"], u["cache_read"], u["cache_write"], u["steps"]) == (11, 8, 2, 150, 7, 2)
    assert u["cost"] == pytest.approx(0.03) and u["session"] == "ses_A" and stop_reason(u) == "end_turn"
    assert stop_reason(oc_usage(OC_LOG.replace('"stop"', '"length"'))) == "length"
    assert stop_reason(oc_usage(OC_LOG + json.dumps({"type": "error", "error": {}}))) == "error"
    assert stop_reason(oc_usage("")) == "none"


def test_cc_usage():
    log = "\n".join(json.dumps(e) for e in [
        {"type": "assistant", "session_id": "s1", "message": {"stop_reason": None}},
        {"type": "result", "subtype": "success", "is_error": False, "num_turns": 4, "total_cost_usd": 0.05,
         "usage": {"input_tokens": 9, "output_tokens": 30, "cache_read_input_tokens": 400,
                   "cache_creation_input_tokens": 20}}])
    u = cc_usage(log)
    assert (u["in"], u["out"], u["cache_read"], u["cache_write"], u["steps"], u["session"]) == (9, 30, 400, 20, 4, "s1")
    assert stop_reason(u) == "end_turn"
    assert stop_reason(cc_usage(log.replace('"success", "is_error": false', '"error_max_turns", "is_error": true'))) \
        == "error"


def test_parse_rung_maps_the_private_copy_back_to_app():
    out = "noise\n" + json.dumps({"status": "UNPROVED", "feedback": "/tmp/.sl-1/Solution.lean:3:2: error: x"})
    r = parse_rung(out, "/tmp/.sl-1/Solution.lean", "/app/Solution.lean")
    assert r["status"] == "UNPROVED" and r["feedback"].startswith("/app/Solution.lean:3:2")
    assert parse_rung("Traceback ...", "a", "b")["status"] == "GATE_ERROR"
    assert parse_rung("", "a", "b")["status"] == "GATE_ERROR"


def test_round_commands():
    c = oc_round_cmd("anthropic/claude-haiku-5-5", "ses_A", "--variant medium", "fix it's", ["/l/opencode.txt", "/l/fv/B1.txt"], 600)
    assert "timeout -k 15 600 opencode --model=anthropic/claude-haiku-5-5 run --format=json --session ses_A" in c
    assert "--variant medium --thinking --dangerously-skip-permissions -- 'fix it'\"'\"'s'" in c
    assert c.endswith("| stdbuf -oL tee -a /l/opencode.txt /l/fv/B1.txt")
    assert "--continue" in oc_round_cmd("m", None, "", "x", ["/o"], 5)
    c = cc_round_cmd("SL_MSG_AB", "--permission-mode=bypassPermissions", ["/l/claude-code.txt", "/l/fv/B1.txt"], 600)
    assert 'printf "%s" "$SL_MSG_AB" | timeout -k 15 600 claude' in c and "--continue --print" in c


# ---------------------------------------------------------------- snapshots

def _tree(root: Path):
    (root / "app/sub").mkdir(parents=True)
    (root / "app/Solution.lean").write_text("theorem t : True := sorry\n")
    (root / "app/sub/notes.txt").write_text("n")
    os.symlink("Solution.lean", root / "app/link")
    (root / "xdg/opencode/storage").mkdir(parents=True)
    (root / "xdg/opencode/storage/msg.json").write_text('{"m": 1}')
    os.chmod(root / "app/sub/notes.txt", 0o600)
    return [str(root / "app"), str(root / "xdg"), str(root / "missing")]


def test_snapshot_manifest_restore_roundtrip(tmp_path):
    roots, snap = _tree(tmp_path / "box"), str(tmp_path / "snap")
    s0 = fv_snap("S0", snap, roots)
    assert s0["n"] == len(fs_manifest(roots)) and oct(os.stat(snap).st_mode & 0o777) == "0o700"
    assert fv_verify("S0", snap, roots)["ok"]
    app = Path(roots[0])
    (app / "Solution.lean").write_text("theorem t : True := trivial\n")
    (app / "new.lean").write_text("x")
    os.chmod(app / "sub/notes.txt", 0o644)
    shutil.rmtree(Path(roots[1]) / "opencode")
    v = fv_verify("S0", snap, roots)
    assert not v["ok"] and v["n_diff"] >= 4 and v["digest"] != v["expect"]
    fv_restore("S0", snap, roots)
    after = fv_verify("S0", snap, roots)
    assert after["ok"] and after["digest"] == s0["digest"]
    assert (app / "Solution.lean").read_text().endswith("sorry\n") and not (app / "new.lean").exists()
    assert os.readlink(app / "link") == "Solution.lean" and oct(os.stat(app / "sub/notes.txt").st_mode & 0o777) == "0o600"


def test_snap_script_runs_standalone_and_isolated(tmp_path):
    roots, snap = _tree(tmp_path / "box"), str(tmp_path / "snap")
    run = lambda *a: json.loads(subprocess.run([sys.executable, "-I", "-c", SNAP_SCRIPT, *a], capture_output=True,
                                               text=True, check=True, cwd=tmp_path).stdout.splitlines()[-1])
    s = run("snap", "S0", snap, "-", *roots)
    (Path(roots[0]) / "Solution.lean").write_text("changed")
    assert run("verify", "S0", snap, "-", *roots)["ok"] is False
    r = run("restore", "S0", snap, "-", *roots)
    assert r["ok"] and r["digest"] == s["digest"]
    assert run("kill", "-", "-", "no-such-user-xyz") == {"killed": 0, "left": 0}
    assert run("digest", "-", "-", "-", roots[0])["n"] == 5


# ---------------------------------------------------------------- round control flow (fake environment)

class FakeEnv:
    """Runs every command locally with bash; downloads are local copies. User switches are ignored."""
    environment_name = "la9999"
    default_user = "-"  # no such user: the snapshot script skips home, /tmp and process kills

    def __init__(self):
        self.cmds = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.cmds.append((user, command))
        p = subprocess.run(["bash", "-c", command], capture_output=True, text=True, env={**os.environ, **(env or {})})
        return SimpleNamespace(return_code=p.returncode, stdout=p.stdout, stderr=p.stderr)

    async def download_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, target_path)


def _ev(sid="ses_A", reason="stop", n=10):
    return json.dumps({"type": "step_finish", "sessionID": sid,
                       "part": {"reason": reason, "cost": 0.001, "tokens": {"input": n, "output": 1, "cache": {}}}})


class FakeFV(FVMixin):
    """FVMixin with a scripted rung and a shell 'agent' that edits /app; snapshots and restores are the real ones."""
    _usage = staticmethod(oc_usage)

    def __init__(self, tmp: Path, statuses, nudge=True, branch_timeout=600):
        self.box = tmp / "box"
        self.app = self.box / "app"
        self.logs_dir = tmp / "host"
        self.environment_logs_dir = self.box / "logs"
        self.fv_python, self.fv_snapdir = sys.executable, str(self.box / "var-sl-snap")
        self.options = SimpleNamespace(max_rounds=2, initial_timeout_sec=60, branch_timeout_sec=branch_timeout,
                                       rung_timeout_sec=60, nudge=nudge, tasks_dir=None)
        self.statuses, self.seen = list(statuses), []

    def _fv_fixed_roots(self):
        return [str(self.app), str(self.environment_logs_dir / "xdg")]

    def _fv_main_log(self):
        return f"{self.environment_logs_dir}/opencode.txt"

    def _fv_sol(self, lang):
        return str(self.app / "Solution.lean")

    async def _fv_run_harness(self, instruction, environment, context):
        self.app.mkdir(parents=True)
        (self.app / "Solution.lean").write_text("S0\n")
        (self.environment_logs_dir / "xdg").mkdir(parents=True)
        (self.environment_logs_dir / "xdg" / "session.json").write_text("s0")
        Path(self._fv_main_log()).write_text(_ev() + "\n")

    async def _rung(self, environment, rnd):
        return {"status": self.statuses.pop(0), "feedback": f"fix {rnd}"}

    def _round_cmd(self, message, out, budget):
        label = Path(out).stem
        self.seen.append((label, message, (self.app / "Solution.lean").read_text(),
                          sorted(p.name for p in self.app.iterdir()), self._fv_session))
        sol = self.app / "Solution.lean"
        return (f"echo {label} >> {sol}; touch {self.app}/scratch_{label}; echo s >> "
                f"{self.environment_logs_dir}/xdg/session.json; echo {shlex.quote(_ev())} | tee -a "
                f"{self._fv_main_log()} {out}", {})


def _run(fake):
    asyncio.run(fake._fv_flow("do it", FakeEnv(), None))
    return [json.loads(line) for line in (fake.logs_dir / "fv.jsonl").read_text().splitlines()]


def test_flow_b_fires_twice_then_c_restores_s0(tmp_path):
    fake = FakeFV(tmp_path, ["UNPROVED", "CHEAT_REJECTED", "PROVED"])
    recs = _run(fake)
    s = recs[-1]
    assert s["ev"] == "summary" and s["A"]["stop"] == "end_turn" and s["A"]["usage"]["session"] == "ses_A"
    assert [x[0] for x in fake.seen] == ["B1", "B2", "C1", "C2"]
    assert [x[1] for x in fake.seen] == ["fix 1", "fix 2", NUDGE[0], NUDGE[1]]
    assert all(x[4] == "ses_A" for x in fake.seen)  # every round resumes the A session
    assert fake.seen[1][2] == "S0\nB1\n"  # B continues its own trajectory
    assert fake.seen[2][2] == "S0\n" and fake.seen[2][3] == ["Solution.lean"]  # C starts from exactly S0
    assert fake.seen[3][2] == "S0\nC1\n"
    assert s["B"]["fired"] and s["B"]["rounds"] == 2 and s["B"]["checks"] == ["UNPROVED", "CHEAT_REJECTED", "PROVED"]
    assert s["B"]["final"] == "PROVED" and s["B"]["usage"]["in"] == 20 and s["B"]["stops"] == ["end_turn"] * 2
    assert s["C"]["restored"] and s["C"]["restore_ok"] and s["C"]["rounds"] == 2 and s["C"]["usage"]["in"] == 20
    assert list(s["snaps"]) == ["S0", "B1", "B2", "C1", "C2"] and all(v["download_ok"] for v in s["snaps"].values())
    assert len({v["sol_sha"] for v in s["snaps"].values()}) == 5
    restore = next(r for r in recs if r["ev"] == "restore")
    assert restore["ok"] and restore["digest"] == restore["expect"] == s["snaps"]["S0"]["digest"]
    checks = [r for r in recs if r["ev"] == "check"]
    assert [c["fired"] for c in checks] == [True, True, False] and checks[-1]["final"]
    host = fake.logs_dir / "fv" / "snap"
    assert sorted(p.name for p in host.glob("*.tar.gz")) == sorted(f"{x}.tar.gz" for x in s["snaps"])
    assert not Path(fake.fv_snapdir).exists()  # the in-container snapshot dir is removed at the end
    assert len(Path(fake._fv_main_log()).read_text().splitlines()) == 5  # A + 4 rounds: Harbor totals see all


def test_flow_gate_silent_c_continues_from_s0_without_restore(tmp_path):
    fake = FakeFV(tmp_path, ["PROVED"])
    recs = _run(fake)
    s = recs[-1]
    assert [x[0] for x in fake.seen] == ["C1", "C2"] and fake.seen[0][2] == "S0\n"
    assert not s["B"]["fired"] and s["B"]["rounds"] == 0 and s["B"]["checks"] == ["PROVED"]
    assert not s["C"]["restored"] and s["C"]["restore_ok"]
    assert not any(r["ev"] == "restore" for r in recs) and any(r["ev"] == "verify" and r["ok"] for r in recs)


def test_flow_check_error_does_not_resume_and_no_c_for_ab_only(tmp_path):
    fake = FakeFV(tmp_path, ["CHECK_ERROR"], nudge=False)
    s = _run(fake)[-1]
    assert fake.seen == [] and s["B"]["checks"] == ["CHECK_ERROR"] and s["C"] == {} and list(s["snaps"]) == ["S0"]


def test_flow_one_repair_round_then_proved(tmp_path):
    fake = FakeFV(tmp_path, ["SPEC_TAMPERED", "PROVED"])
    s = _run(fake)[-1]
    assert [x[0] for x in fake.seen] == ["B1", "C1", "C2"] and s["B"]["final"] == "PROVED"
    assert fake.seen[1][2] == "S0\n" and s["C"]["restored"]


def test_flow_branch_time_cap_skips_rounds(tmp_path):
    fake = FakeFV(tmp_path, ["UNPROVED"], branch_timeout=10)
    recs = _run(fake)
    s = recs[-1]
    assert fake.seen == [] and s["B"]["fired"] and s["B"]["rounds"] == 0 and s["C"]["rounds"] == 0
    assert [r["branch"] for r in recs if r["ev"] == "skip"] == ["B", "C"]
