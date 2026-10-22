"""Same-trajectory FV agents for Millennium Harness FV v1.0 (docs/research/win-plan-synthesis.md §3).

  -a bench.agents_fv:FVBranch     --ak tasks_dir=<built tasks> --ak max_rounds=2   (OpenCode: A, B and C in one trial)
  -a bench.agents_fv:ClaudeCodeFV --ak tasks_dir=<built tasks> --ak max_rounds=2   (Claude Code: A and B)

A  The harness runs until its first stop, S0 (capped at initial_timeout_sec). S0 is tarred into root-only
   /var/sl-snap with a sha256 manifest and downloaded to agent/fv/snap/. It holds /app, the agent's home and /tmp
   entries, and the harness session dirs.
B  The Millennium Harness rung runs in the sandbox: `python3 -I -m millennium_harness.<lang> check`, as the agent
   user, on a root-owned copy of the file. On any status except PROVED or CHECK_ERROR, the SAME session resumes with
   the rung's feedback: at most max_rounds rounds and branch_timeout_sec of agent time, with a snapshot and download
   after every round.
C  OpenCode only. If B resumed: kill the agent's processes, restore S0 and check that its manifest matches. Otherwise
   continue from S0 after the same manifest check. Then run two unconditional frozen NUDGE rounds, with a snapshot
   and download after each.
Every event (tokens, cost, timings, stop reasons, snapshot hashes) is appended to agent/fv.jsonl. A/B/C outcomes
come from the fresh-container regrade of the snapshots, not from this agent.
The pure helpers and FVMixin need only the stdlib (unit tests run without Harbor). Run harbor with PYTHONPATH=<repo>.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import pwd
import secrets
import shlex
import shutil
import signal
import stat
import tarfile
import time
from pathlib import Path

SNAPDIR = "/var/sl-snap"
PY = "/usr/local/bin/python3"
SOLUTION = {"lean": "/app/Solution.lean", "dafny": "/app/solution.dfy"}
FIRE = {"UNPROVED", "CHEAT_REJECTED", "SPEC_TAMPERED"}  # resume; PROVED / CHECK_ERROR / GATE_ERROR do not
STOPS = {"stop": "end_turn", "end_turn": "end_turn", "length": "length", "max_tokens": "length"}
MIN_ROUND_SECS = 30
NUDGE = [  # frozen C texts (plan §3)
    "Before you finish: run the verify command yourself. Make sure it reports no errors and no `declaration uses "
    "'sorry'` warning, that you used no escape hatch (sorry, admit, axiom, native_decide, assume, {:axiom}, "
    "{:verify false}, external_body…), and that every frozen part is unchanged. Fix anything wrong; otherwise reply "
    "DONE.",
    "Check once more exactly as above and fix anything that remains; otherwise reply DONE.",
]


# ---------------------------------------------------------------- pure helpers (host side)

def rung_cmd(lang: str, solution: str, frozen: str, work: str, rnd: int | None = None,
             max_rounds: int | None = None) -> str:
    """The one rung invocation, shared by the in-sandbox gate and tests/test.sh. It runs isolated Python in a clean
    environment against the root-owned package and toolchain binaries, and `work` is the only directory it can
    write. The Lean CLI takes --round/--max-rounds; the Dafny CLI takes them positionally."""
    tail = "" if rnd is None else f" --round {rnd} --max-rounds {max_rounds}" if lang == "lean" else f" {rnd} {max_rounds}"
    return (f"cd {work} && env -i PATH=/usr/local/bin:/usr/bin:/bin HOME={work} TMPDIR={work} LANG=C.UTF-8 "
            f"MILLENNIUM_LEAN=/opt/lean/bin/lean MILLENNIUM_DAFNY=/opt/dafny/dafny DOTNET_CLI_TELEMETRY_OPTOUT=1 "
            f"{PY} -I -m millennium_harness.{lang} check {solution} {frozen}{tail}")


def parse_rung(stdout: str | None, copy: str, solution: str) -> dict:
    """The rung prints its JSON verdict on the last line. Paths of the private copy are mapped back to /app."""
    lines = (stdout or "").strip().splitlines()
    try:
        r = json.loads(lines[-1])
    except (IndexError, ValueError):
        r = None
    if not isinstance(r, dict) or "status" not in r:
        return {"status": "GATE_ERROR", "tail": (stdout or "")[-500:]}
    for k in ("feedback", "verify"):
        if isinstance(r.get(k), str):
            r[k] = r[k].replace(copy, solution)
    return r


def fallback_feedback(r: dict, rnd: int, max_rounds: int) -> str:
    v = "".join(f"- {x}\n" for x in r.get("violations") or [])
    return (f"⟦Millennium Harness⟧ ✗ {r.get('status')} · round {rnd}/{max_rounds}\n"
            + (f"Rule violations:\n{v}" if v else "")
            + "The file does not verify honestly yet. Fix it and run the verify command again.")


def _events(text: str | None):
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e, dict):
                yield e


def _usage0() -> dict:
    return {"in": 0, "out": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0, "cost": 0.0, "steps": 0,
            "errors": 0, "reason": None, "session": None}


def oc_usage(text: str | None) -> dict:
    """Token and cost totals, the last finish reason and the session id from `opencode run --format=json` output."""
    u = _usage0()
    for e in _events(text):
        u["session"] = u["session"] or e.get("sessionID")
        if e.get("type") == "error":
            u["errors"] += 1
        elif e.get("type") == "step_finish":
            p = e.get("part") or {}
            t = p.get("tokens") or {}
            c = t.get("cache") or {}
            u["in"] += t.get("input") or 0
            u["out"] += t.get("output") or 0
            u["reasoning"] += t.get("reasoning") or 0
            u["cache_read"] += c.get("read") or 0
            u["cache_write"] += c.get("write") or 0
            u["cost"] += p.get("cost") or 0
            u["steps"] += 1
            u["reason"] = p.get("reason")
    return u


def cc_usage(text: str | None) -> dict:
    """The same fields from `claude --output-format=stream-json` output. Each call's `result` event carries its usage."""
    u = _usage0()
    for e in _events(text):
        u["session"] = u["session"] or e.get("session_id")
        if e.get("type") == "assistant":
            u["reason"] = (e.get("message") or {}).get("stop_reason") or u["reason"]
        elif e.get("type") == "result":
            us = e.get("usage") or {}
            u["in"] += us.get("input_tokens") or 0
            u["out"] += us.get("output_tokens") or 0
            u["cache_read"] += us.get("cache_read_input_tokens") or 0
            u["cache_write"] += us.get("cache_creation_input_tokens") or 0
            u["cost"] += e.get("total_cost_usd") or 0
            u["steps"] += e.get("num_turns") or 0
            bad = e.get("is_error") or str(e.get("subtype", "")).startswith("error")
            u["errors"] += bool(bad)
            u["reason"] = e.get("stop_reason") or ("max_tokens" if u["reason"] == "max_tokens" else
                                                   "end_turn" if e.get("subtype") == "success" else e.get("subtype"))
        elif e.get("type") == "error":
            u["errors"] += 1
    return u


def stop_reason(u: dict) -> str:
    """end_turn | length | error | <raw reason> | none."""
    if u.get("errors"):
        return "error"
    r = u.get("reason")
    return STOPS.get(r, r or "none")


def add_usage(acc: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            acc[k] = acc.get(k, 0) + v
    return acc


def oc_round_cmd(model: str, session: str | None, flags: str, message: str, outs: list[str], budget: int) -> str:
    """`opencode run` resumes one session: by id if known, else the latest. Capped at `budget` s; every file in outs
    gets the output appended."""
    resume = f"--session {shlex.quote(session)}" if session else "--continue"
    flags = f"{flags} " if flags else ""
    return (f"timeout -k 15 {budget} opencode --model={model} run --format=json {resume} {flags}--thinking "
            f"--dangerously-skip-permissions -- {shlex.quote(message)} 2>&1 </dev/null "
            f"| stdbuf -oL tee -a {' '.join(map(shlex.quote, outs))}")


def cc_round_cmd(var: str, flags: str, outs: list[str], budget: int) -> str:
    """`claude --continue`. The message comes from env var `var`, never from the command line."""
    flags = f"{flags} " if flags else ""
    return (f'export PATH="$HOME/.local/bin:$PATH"; printf "%s" "${var}" | timeout -k 15 {budget} claude --verbose '
            f"--output-format=stream-json {flags}--continue --print 2>&1 | tee -a {' '.join(map(shlex.quote, outs))}")


# ---------------------------------------------------------------- snapshot script (runs as root in the sandbox)
# These functions are shipped verbatim as SNAP_SCRIPT (`python3 -I -c SNAP_SCRIPT op label snapdir user roots...`).
# They use only the stdlib, and the tests also call them directly.

def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _digest(man):
    return hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()


def fs_manifest(roots):
    """{path: kind:mode:uid:gid[:sha256]} for everything under `roots`, without following symlinks (stored as
    l:uid:gid:target). Sockets are skipped, as tar skips them."""
    out = {}

    def add(p):
        st = os.lstat(p)
        own, mode = f"{st.st_uid}:{st.st_gid}", f"{stat.S_IMODE(st.st_mode):o}"
        if stat.S_ISLNK(st.st_mode):
            out[p] = f"l:{own}:{os.readlink(p)}"
        elif stat.S_ISDIR(st.st_mode):
            out[p] = f"d:{mode}:{own}"
        elif stat.S_ISREG(st.st_mode):
            out[p] = f"f:{mode}:{own}:{_sha_file(p)}"
        elif not stat.S_ISSOCK(st.st_mode):
            out[p] = f"o:{mode}:{own}"

    for r in roots:
        if os.path.lexists(r):
            add(r)
            if os.path.isdir(r) and not os.path.islink(r):
                for d, dirs, files in os.walk(r):
                    for n in dirs + files:
                        add(os.path.join(d, n))
    return out


def snap_roots(fixed, user):
    """The fixed roots, plus the user's home and the user's top-level /tmp entries (unless user is '-' or unknown)."""
    roots = list(fixed)
    try:
        pw = pwd.getpwnam(user)
    except KeyError:
        return roots
    roots.append(pw.pw_dir)
    roots += sorted(e.path for e in os.scandir("/tmp") if e.stat(follow_symlinks=False).st_uid == pw.pw_uid)
    return roots


def fv_snap(label, snapdir, roots):
    os.makedirs(snapdir, exist_ok=True)
    os.chmod(snapdir, 0o700)
    man = fs_manifest(roots)
    tar = os.path.join(snapdir, label + ".tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        for r in roots:
            if os.path.lexists(r):
                t.add(r, arcname=r.lstrip("/"))
    with open(os.path.join(snapdir, label + ".json"), "w") as f:
        json.dump({"label": label, "roots": roots, "manifest": man}, f, sort_keys=True)
    return {"label": label, "tar_sha256": _sha_file(tar), "bytes": os.path.getsize(tar), "n": len(man),
            "digest": _digest(man)}


def fv_verify(label, snapdir, roots):
    with open(os.path.join(snapdir, label + ".json")) as f:
        want = json.load(f)["manifest"]
    now = fs_manifest(roots)
    diff = sorted(k for k in set(now) | set(want) if now.get(k) != want.get(k))
    return {"label": label, "ok": not diff, "n_diff": len(diff), "diff": diff[:20], "digest": _digest(now),
            "expect": _digest(want)}


def fv_restore(label, snapdir, roots):
    """Wipe the current and the snapshotted roots, then extract the snapshot. Owners, modes and links are kept."""
    with open(os.path.join(snapdir, label + ".json")) as f:
        saved = json.load(f)["roots"]
    for r in sorted(set(roots) | set(saved)):
        if os.path.isdir(r) and not os.path.islink(r):
            shutil.rmtree(r)
        elif os.path.lexists(r):
            os.unlink(r)
    kw = {"filter": "fully_trusted"} if hasattr(tarfile, "fully_trusted_filter") else {}
    with tarfile.open(os.path.join(snapdir, label + ".tar.gz")) as t:
        t.extractall("/", numeric_owner=True, **kw)


def fv_kill(user):
    """SIGKILL every live process of `user`, repeating until none are left."""
    try:
        uid = str(pwd.getpwnam(user).pw_uid)
    except KeyError:
        return {"killed": 0, "left": 0}
    killed, pids = 0, []
    for _ in range(50):
        pids = []
        for p in os.listdir("/proc"):
            if not p.isdigit() or int(p) == os.getpid():
                continue
            try:
                with open(f"/proc/{p}/status") as f:
                    st = dict(line.split(":", 1) for line in f if ":" in line)
            except OSError:
                continue
            if uid in st.get("Uid", "").split() and not st.get("State", "").strip().startswith("Z"):
                pids.append(int(p))
        if not pids:
            break
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
                killed += 1
            except OSError:
                pass
        time.sleep(0.1)
    return {"killed": killed, "left": len(pids)}


def fv_main(argv):
    op, label, snapdir, user, *fixed = argv
    if op == "kill":
        return fv_kill(user)
    if op == "digest":
        man = fs_manifest(fixed)
        return {"digest": _digest(man), "n": len(man)}
    if op == "snap":
        return fv_snap(label, snapdir, snap_roots(fixed, user))
    if op == "restore":
        fv_restore(label, snapdir, snap_roots(fixed, user))
    return fv_verify(label, snapdir, snap_roots(fixed, user))  # verify (also after restore)


SNAP_SCRIPT = ("import hashlib, json, os, pwd, shutil, signal, stat, sys, tarfile, time\n\n"
               + "\n\n".join(inspect.getsource(f) for f in (_sha_file, _digest, fs_manifest, snap_roots, fv_snap,
                                                             fv_verify, fv_restore, fv_kill, fv_main))
               + "\nprint(json.dumps(fv_main(sys.argv[1:])))\n")


# ---------------------------------------------------------------- branching (harness-agnostic)

class FVMixin:
    """A -> S0 -> B (gate) [-> restore S0 -> C (nudge)]. Each harness class supplies _fv_run_harness,
    _fv_main_log, _fv_fixed_roots, _round_cmd and _usage."""

    fv_python = PY
    fv_snapdir = SNAPDIR

    # -- small utilities
    def _fv_log(self, rec: dict) -> None:
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.logs_dir / "fv.jsonl", "a") as f:
                f.write(json.dumps({"t": round(time.time() - self._fv_t0, 1), **rec}, default=str) + "\n")
        except OSError:
            pass

    def _fv_user(self, environment) -> str:
        return str(environment.default_user or "agent")

    def _fv_task(self, environment):
        """Return (lang, host path of tests/frozen.json) for this trial's task, or (None, None)."""
        if not self.options.tasks_dir:
            return None, None
        d = Path(self.options.tasks_dir) / environment.environment_name
        for lang, sol in SOLUTION.items():
            if (d / "environment" / sol.rsplit("/", 1)[1]).exists():
                return lang, d / "tests" / "frozen.json"
        return None, None

    def _fv_sol(self, lang):
        return SOLUTION.get(lang)

    async def _root(self, environment, cmd: str):
        try:
            return await environment.exec(command=cmd, user="root")
        except Exception:  # noqa: BLE001
            return None

    async def _sh(self, environment, op: str, label: str = "-", roots: list[str] | None = None) -> dict:
        """Run SNAP_SCRIPT as root and return its JSON result, or {"error": ...}."""
        args = [op, label, self.fv_snapdir, self._fv_user(environment), *(roots or self._fv_fixed_roots())]
        cmd = f"{self.fv_python} -I -c {shlex.quote(SNAP_SCRIPT)} {' '.join(map(shlex.quote, args))}"
        try:
            res = await environment.exec(command=cmd, user="root", timeout_sec=600)
            out = (res.stdout or "").strip().splitlines()
            if res.return_code == 0 and out:
                return json.loads(out[-1])
            return {"error": ((res.stderr or "") + (res.stdout or ""))[-500:]}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:300]}

    async def _fetch(self, environment, remote: str, name: str) -> str:
        """Download a container file to agent/fv/<name> and return its text, or '' if it is missing."""
        local = self.logs_dir / "fv" / name
        try:
            await environment.download_file(remote, local)
            return local.read_text(errors="replace")
        except Exception:  # noqa: BLE001
            return ""

    # -- primitives
    async def _snap(self, environment, label: str, keep: bool = False) -> dict:
        """Kill the agent's processes, tar and manifest the state, download both, check the sha, then delete them
        from the container unless `keep` is set."""
        t = time.time()
        killed = await self._sh(environment, "kill")
        r = await self._sh(environment, "snap", label)
        rec = {"ev": "snap", "label": label, "killed": killed.get("killed"), **r}
        local = self.logs_dir / "fv" / "snap"
        try:
            # the full snapshot stays in the sandbox (for the C restore); only /app (the solution) comes down
            lite = f"{self.fv_snapdir}/{label}.app.tar.gz"
            app = self._fv_fixed_roots()[0]
            res = await self._root(environment, f"{self.fv_python} -I -c \"import tarfile,hashlib;t=tarfile.open('{lite}','w:gz');"
                                                f"t.add('{app}',arcname='app');t.close();"
                                                f"print(hashlib.sha256(open('{lite}','rb').read()).hexdigest())\"")
            lite_sha = ((getattr(res, "stdout", "") or "").split() or [None])[0]
            await environment.download_file(lite, local / f"{label}.tar.gz")
            await environment.download_file(f"{self.fv_snapdir}/{label}.json", local / f"{label}.json")
            await self._root(environment, f"rm -f {lite}")
            rec["download_ok"] = _sha_file(local / f"{label}.tar.gz") == lite_sha
            man = json.loads((local / f"{label}.json").read_text())["manifest"]
            rec["sol_sha"] = (man.get(self._fv_solution or "") or "").rsplit(":", 1)[-1] or None
        except Exception as e:  # noqa: BLE001
            rec.update(download_ok=False, download_error=str(e)[:300])
        if not keep:
            await self._root(environment, f"rm -f {self.fv_snapdir}/{label}.tar.gz {self.fv_snapdir}/{label}.json")
        rec["secs"] = round(time.time() - t, 1)
        self._fv_log(rec)
        self._fv["snaps"][label] = {k: rec.get(k) for k in ("tar_sha256", "digest", "sol_sha", "download_ok")}
        return rec

    async def _restore(self, environment, reason: str) -> dict:
        t = time.time()
        killed = await self._sh(environment, "kill")
        r = await self._sh(environment, "restore", "S0")
        rec = {"ev": "restore", "label": "S0", "reason": reason, "killed": killed.get("killed"), **r,
               "secs": round(time.time() - t, 1)}
        self._fv_log(rec)
        return rec

    async def _rung(self, environment, rnd: int) -> dict:
        """The gate check runs as the agent user on a root-owned copy of the solution. frozen.json is uploaded for
        the duration of the check only."""
        lang, frozen = self._fv_task(environment)
        if not lang or not frozen.exists():
            return {"status": "NO_TASK"}
        sol = SOLUTION[lang]
        name = sol.rsplit("/", 1)[1]
        tmp, user = f"/tmp/.sl-{secrets.token_hex(6)}", self._fv_user(environment)
        try:
            await self._root(environment, (
                f"mkdir -p {tmp}/w && chmod 755 {tmp} && chown {user} {tmp}/w && chmod 700 {tmp}/w && "
                f"/usr/sbin/runuser -u {user} -- cat {sol} > {tmp}/{name}; chmod 644 {tmp}/{name}"))
            await environment.upload_file(frozen, f"{tmp}/frozen.json")
            await self._root(environment, f"chmod 644 {tmp}/frozen.json")
            res = await environment.exec(
                command=rung_cmd(lang, f"{tmp}/{name}", f"{tmp}/frozen.json", f"{tmp}/w", rnd, self.options.max_rounds),
                timeout_sec=int(self.options.rung_timeout_sec))
            return parse_rung(res.stdout, f"{tmp}/{name}", sol)
        except Exception as e:  # noqa: BLE001
            return {"status": "GATE_ERROR", "error": str(e)[:300]}
        finally:
            await self._root(environment, f"rm -rf {tmp}")

    async def _check(self, environment, rnd: int, final: bool = False) -> dict:
        t = time.time()
        r = await self._rung(environment, rnd)
        st = r.get("status", "GATE_ERROR")
        self._fv_log({"ev": "check", "branch": "B", "round": rnd, "final": final, "status": st,
                      "fired": st in FIRE and not final, "violations": r.get("violations"), "error": r.get("error"),
                      "feedback": (r.get("feedback") or "")[:4000], "secs": round(time.time() - t, 1)})
        return r

    async def _round(self, environment, message: str, label: str, budget: float) -> dict:
        """Resume the same session once. The output also goes to the harness's main log, so Harbor's totals include
        every branch."""
        out = f"{self.environment_logs_dir}/fv/{label}.txt"
        cmd, env = self._round_cmd(message, out, int(budget))
        t, rc, err = time.time(), None, None
        try:
            res = await environment.exec(command=f"set -o pipefail; mkdir -p {self.environment_logs_dir}/fv; {cmd}",
                                         env=env, timeout_sec=int(budget) + 120)
            rc = res.return_code
        except Exception as e:  # noqa: BLE001
            err = str(e)[:300]
        u = self._usage(await self._fetch(environment, out, f"{label}.txt"))
        stop = "timeout" if rc in (124, 137) else "error" if err else stop_reason(u)
        rec = {"ev": "round", "label": label, "rc": rc, "error": err, "secs": round(time.time() - t, 1),
               "stop": stop, "usage": u}
        self._fv_log(rec)
        return rec

    async def _fv_initial(self, instruction, environment, context) -> dict:
        """Branch A: the unmodified harness run up to its first stop. Errors and timeouts are stops too."""
        t, stop, err = time.time(), None, None
        try:
            await asyncio.wait_for(self._fv_run_harness(instruction, environment, context),
                                   self.options.initial_timeout_sec)
        except asyncio.TimeoutError:
            stop = "timeout"
        except Exception as e:  # noqa: BLE001
            stop, err = "error", f"{type(e).__name__}: {str(e)[:300]}"
        u = self._usage(await self._fetch(environment, self._fv_main_log(), "A.txt"))
        self._fv_session = u.get("session")
        return {"stop": stop or stop_reason(u), "reason_raw": u.get("reason"), "secs": round(time.time() - t, 1),
                "usage": u, "error": err}

    # -- branches
    async def _branch_b(self, environment) -> bool:
        """Gate rounds from S0. Returns True if the session was resumed, i.e. the state left S0."""
        R, t0, spent = self.options.max_rounds, time.time(), 0.0
        b = self._fv["B"] = {"fired": False, "rounds": 0, "checks": [], "stops": [], "usage": {}}
        for rnd in range(1, R + 2):
            final = rnd > R  # the check after the last round is logged only
            r = await self._check(environment, rnd, final)
            b["checks"].append(r.get("status"))
            if final or r.get("status") not in FIRE:
                break
            b["fired"] = True
            left = self.options.branch_timeout_sec - spent
            if left < MIN_ROUND_SECS:
                self._fv_log({"ev": "skip", "branch": "B", "round": rnd, "reason": "branch time cap"})
                break
            x = await self._round(environment, r.get("feedback") or fallback_feedback(r, rnd, R), f"B{rnd}", left)
            spent += x["secs"]
            b["rounds"] += 1
            b["stops"].append(x["stop"])
            add_usage(b["usage"], x["usage"])
            await self._snap(environment, f"B{rnd}")
        b.update(final=b["checks"][-1], agent_secs=round(spent, 1), secs=round(time.time() - t0, 1))
        return b["rounds"] > 0

    async def _branch_c(self, environment, b_resumed: bool) -> None:
        """Compute-matched control: 2 unconditional nudge rounds from S0. It always runs after B."""
        t0, spent = time.time(), 0.0
        c = self._fv["C"] = {"rounds": 0, "stops": [], "usage": {}}
        if b_resumed:
            r = await self._restore(environment, "B resumed")
        else:
            r = await self._sh(environment, "verify", "S0")
            self._fv_log({"ev": "verify", **r})
            if not r.get("ok"):
                r = await self._restore(environment, "drift since S0")
        c.update(restored=r.get("ev") == "restore", restore_ok=bool(r.get("ok")), restore_n_diff=r.get("n_diff"))
        for rnd, text in enumerate(NUDGE, 1):
            left = self.options.branch_timeout_sec - spent
            if left < MIN_ROUND_SECS:
                self._fv_log({"ev": "skip", "branch": "C", "round": rnd, "reason": "branch time cap"})
                break
            x = await self._round(environment, text, f"C{rnd}", left)
            spent += x["secs"]
            c["rounds"] += 1
            c["stops"].append(x["stop"])
            add_usage(c["usage"], x["usage"])
            await self._snap(environment, f"C{rnd}")
        c.update(agent_secs=round(spent, 1), secs=round(time.time() - t0, 1))

    async def _fv_flow(self, instruction, environment, context) -> None:
        self._fv_t0 = time.time()
        self._fv = {"A": {}, "B": {}, "C": {}, "snaps": {}}
        self._fv_session = None
        lang, _ = self._fv_task(environment)
        self._fv_solution = self._fv_sol(lang)
        try:
            self._fv_log({"ev": "start", "agent": type(self).__name__, "lang": lang,
                          "task": environment.environment_name, "user": self._fv_user(environment),
                          "sl": await self._sh(environment, "digest", roots=["/opt/sl"])})
            # egress probe (plan K2): as the agent user, non-model hosts must be unreachable
            user = self._fv_user(environment) or "agent"
            probe = " ; ".join(f"/usr/sbin/runuser -u {user} -- curl -s -m 8 -o /dev/null -w '{h} %{{http_code}}\\n' https://{h}"
                               f" || echo '{h} blocked'" for h in ("github.com", "huggingface.co", "atcoder.jp"))
            r = await self._root(environment, probe)
            self._fv_log({"ev": "egress", "out": (getattr(r, "stdout", "") or "").strip()})
            self._fv["A"] = await self._fv_initial(instruction, environment, context)
            self._fv_log({"ev": "A", **self._fv["A"]})
            await self._snap(environment, "S0", keep=True)
            resumed = await self._branch_b(environment)
            if getattr(self.options, "nudge", False):
                await self._branch_c(environment, resumed)
            await self._root(environment, f"rm -rf {self.fv_snapdir}")
        finally:
            self._fv_log({"ev": "summary", **self._fv})


# ---------------------------------------------------------------- Harbor agents

try:
    from pydantic import Field

    from bench.agents import ClaudeCodeGated, GatedClaudeOptions, GatedOptions, OpenCodeGated
    from harbor.agents.installed.claude_code import ClaudeCode
    from harbor.agents.installed.opencode import OpenCode
except ImportError:  # unit tests run without Harbor
    OpenCodeGated = None


async def _check_baked(environment, cmd: str, version: str | None) -> None:
    """FV images bake every harness and the network is API-only, so check the version instead of installing."""
    res = await environment.exec(command=cmd)
    if res.return_code != 0 or (version and version not in (res.stdout or "")):
        raise RuntimeError(f"`{cmd}` gave {(res.stdout or res.stderr or '').strip()[:200]!r}; expected {version} "
                           "baked into the image (FV images install nothing at runtime)")


if OpenCodeGated is not None:

    class FVOptions(GatedOptions):
        tasks_dir: str | None = Field(default=None)
        initial_timeout_sec: int = Field(default=1200)
        branch_timeout_sec: int = Field(default=600)
        rung_timeout_sec: int = Field(default=900)
        nudge: bool = Field(default=True)

    class FVClaudeOptions(GatedClaudeOptions):
        tasks_dir: str | None = Field(default=None)
        initial_timeout_sec: int = Field(default=1200)
        branch_timeout_sec: int = Field(default=600)
        rung_timeout_sec: int = Field(default=900)
        nudge: bool = Field(default=False)  # C needs an exact restore of the session; Claude Code runs A/B only

    class FVBranch(FVMixin, OpenCodeGated):
        """OpenCode 1.18.35: A, B and C from one trajectory."""
        options_model = FVOptions
        options: FVOptions
        _usage = staticmethod(oc_usage)

        async def install(self, environment) -> None:
            await _check_baked(environment, "opencode --version", self._version)

        def _fv_fixed_roots(self) -> list[str]:
            d = self.environment_logs_dir / "opencode"
            return ["/app", (d / "xdg-data").as_posix(), (d / "xdg-state").as_posix()]

        def _fv_main_log(self) -> str:
            return f"{self.environment_logs_dir}/{self._OUTPUT_FILENAME}"

        async def _fv_run_harness(self, instruction, environment, context) -> None:
            await OpenCode.run(self, instruction, environment, context)

        def _round_cmd(self, message, out, budget):
            return (oc_round_cmd(self.model_name, self._fv_session, self.build_cli_flags(), message,
                                 [self._fv_main_log(), out], budget), self._env())

        async def run(self, instruction, environment, context) -> None:
            await self._fv_flow(instruction, environment, context)

    class ClaudeCodeFV(FVMixin, ClaudeCodeGated):
        """Claude Code 2.1.296: same-trajectory A and B only, resumed with `claude --continue`."""
        options_model = FVClaudeOptions
        options: FVClaudeOptions
        _usage = staticmethod(cc_usage)

        async def install(self, environment) -> None:
            await _check_baked(environment, "claude --version", self._version)

        def _fv_fixed_roots(self) -> list[str]:
            return ["/app", f"{self.environment_logs_dir}/sessions"]

        def _fv_main_log(self) -> str:
            return f"{self.environment_logs_dir}/claude-code.txt"

        async def _fv_run_harness(self, instruction, environment, context) -> None:
            await ClaudeCode.run(self, instruction, environment, context)

        def _round_cmd(self, message, out, budget):
            var = f"SL_MSG_{secrets.token_hex(4).upper()}"
            return (cc_round_cmd(var, self.build_cli_flags(), [self._fv_main_log(), out], budget),
                    {**self._cc_env(), var: message})

        async def run(self, instruction, environment, context) -> None:
            await self._fv_flow(instruction, environment, context)
