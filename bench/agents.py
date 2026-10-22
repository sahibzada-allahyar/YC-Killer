"""Harbor agents for Millennium Harness experiments. Every arm of a harness shares the same driver, install and
command line.

  -a bench.agents:OpenCodeGated   --ak gate=none|millennium|selfstress --ak max_rounds=2 --ak suite_dir=<dir>
  -a bench.agents:ClaudeCodeGated --ak gate=...                      (driver gate via `claude --continue`)
  -a bench.agents:HermesCapped                                       (compaction capped at 85K tokens)

gate=none        vanilla (Millennium Harness installed but never invoked — install parity)
gate=millennium  after the agent finishes, the sealed suite (<suite_dir>/<task>.json, built offline from fresh-context
                 specs) is uploaded, the solution checked, the suite deleted; on ✗ the frozen feedback block resumes the
                 SAME session. <= max_rounds. (gate=sorryless, the pre-rename value, is accepted too.)
gate=selfstress  compute-matched control: max_rounds UNCONDITIONAL "write a brute force + stress test" resumes.
Run harbor with PYTHONPATH=<repo root>.
"""
from __future__ import annotations

import json
import secrets
import shlex
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from harbor.agents.installed.claude_code import ClaudeCode, ClaudeCodeOptions
from harbor.agents.installed.hermes import Hermes
from harbor.agents.installed.opencode import OpenCode, OpenCodeOptions

REPO = Path(__file__).resolve().parent.parent
SL_SRC = "/opt/sl/src"

STRESS = [
    "Before you finish, stress-test your solution: in a separate file, write an obviously-correct brute-force "
    "reference implementation and a generator of small random valid inputs, compare your solution.py against the "
    "brute force on at least 200 random inputs, and fix any discrepancy in solution.py. Keep solution.py as the "
    "final answer.",
    "Repeat the stress test with a new random seed and different input sizes (include edge cases such as minimum "
    "sizes and equal elements); fix any discrepancy in solution.py.",
    "Run one more stress test round focusing on boundary values; fix any discrepancy in solution.py.",
]


class GatedOptions(OpenCodeOptions):
    gate: str = Field(default="none")
    max_rounds: int = Field(default=2)
    suite_dir: str | None = Field(default=None)
    solution_path: str = Field(default="/app/solution.py")


class GatedClaudeOptions(ClaudeCodeOptions):
    gate: str = Field(default="none")
    max_rounds: int = Field(default=2)
    suite_dir: str | None = Field(default=None)
    solution_path: str = Field(default="/app/solution.py")


class GateMixin:
    """Shared gate loop. Subclasses implement `_resume_with(environment, message)`."""

    async def _install_millennium_harness(self, environment) -> None:
        await self.exec_as_root(environment, f"mkdir -p {SL_SRC} && chmod -R a+rx /opt/sl")
        await environment.upload_dir(REPO / "millennium_harness", f"{SL_SRC}/millennium_harness")
        await self.exec_as_root(environment, f"chmod -R a+rX {SL_SRC} && rm -rf {SL_SRC}/millennium_harness/__pycache__")

    def _log(self, rec: dict[str, Any]) -> None:
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.logs_dir / "gate.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    async def _gate_check(self, environment, rnd: int) -> dict:
        task = environment.environment_name
        suite = Path(self.options.suite_dir or "") / f"{task}.json"
        if not self.options.suite_dir or not suite.exists():
            return {"status": "NO_SUITE"}
        tmp = f"/tmp/.sl-{secrets.token_hex(6)}"
        await self.exec_as_root(environment, f"mkdir -p {tmp} && chmod 777 {tmp}")
        await environment.upload_file(suite, f"{tmp}/suite.json")
        try:
            res = await self.exec_as_agent(
                environment,
                command=(f"cd /app && PYTHONPATH={SL_SRC} timeout 120 python3 -m millennium_harness.gate "
                         f"{tmp}/suite.json {shlex.quote(self.options.solution_path)} --round {rnd} "
                         f"--max-rounds {self.options.max_rounds} || true"))
            raw = (res.stdout or "").strip().splitlines()
            return json.loads(raw[-1]) if raw else {"status": "GATE_ERROR"}
        except Exception as e:  # noqa: BLE001
            return {"status": "GATE_ERROR", "error": str(e)[:300]}
        finally:
            await self.exec_as_root(environment, f"rm -rf {tmp}")

    async def _gate_loop(self, environment) -> None:
        gate, R = self.options.gate, self.options.max_rounds
        if gate == "none":
            return
        for rnd in range(1, R + 1):
            if gate == "selfstress":
                self._log({"round": rnd, "gate": gate, "action": "resume"})
                await self._resume_with(environment, STRESS[min(rnd - 1, len(STRESS) - 1)])
                continue
            r = await self._gate_check(environment, rnd)
            self._log({"round": rnd, "gate": gate, **{k: v for k, v in r.items() if k != "feedback"},
                       "fired": r.get("status") == "REFUTED"})
            if r.get("status") != "REFUTED":
                return
            await self._resume_with(environment, r["feedback"])
        if gate in ("millennium", "sorryless"):  # final state after the last round (logged only)
            r = await self._gate_check(environment, R + 1)
            self._log({"round": R + 1, "gate": gate, "final": True,
                       **{k: v for k, v in r.items() if k != "feedback"}})


class OpenCodeGated(GateMixin, OpenCode):
    options_model = GatedOptions
    options: GatedOptions

    async def install(self, environment) -> None:
        await super().install(environment)
        await self._install_millennium_harness(environment)

    def _env(self) -> dict[str, str]:
        env = dict(self.model_connection.env)
        env["OPENCODE_FAKE_VCS"] = "git"
        env["XDG_DATA_HOME"] = (self.environment_logs_dir / "opencode" / "xdg-data").as_posix()
        env["XDG_STATE_HOME"] = (self.environment_logs_dir / "opencode" / "xdg-state").as_posix()
        return env

    async def _resume_with(self, environment, message: str) -> None:
        flags = self.build_cli_flags()
        flags = (flags + " ") if flags else ""
        out = shlex.quote((self.environment_logs_dir / self._OUTPUT_FILENAME).as_posix())
        await self.exec_as_agent(
            environment,
            command=("[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "
                     f"opencode --model={self.model_name} run --format=json --continue {flags}--thinking "
                     f"--dangerously-skip-permissions -- {shlex.quote(message)} "
                     f"2>&1 </dev/null | stdbuf -oL tee -a {out}"),
            env=self._env())

    async def run(self, instruction: str, environment, context) -> None:
        await super().run(instruction, environment, context)
        await self._gate_loop(environment)


class ClaudeCodeGated(GateMixin, ClaudeCode):
    options_model = GatedClaudeOptions
    options: GatedClaudeOptions

    async def install(self, environment) -> None:
        await super().install(environment)
        await self._install_millennium_harness(environment)

    def _cc_env(self) -> dict[str, str]:
        env = self._resolve_auth_env()
        m = self._resolved_model_name()
        if m:
            env["ANTHROPIC_MODEL"] = m
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["IS_SANDBOX"] = "1"
        env.update(self._resolved_env_vars)
        env["CLAUDE_CONFIG_DIR"] = (self.environment_logs_dir / "sessions").as_posix()
        return env

    async def _resume_with(self, environment, message: str) -> None:
        flags = self.build_cli_flags()
        flags = (flags + " ") if flags else ""
        var = f"sl_msg_{secrets.token_hex(4)}"
        out = (self.environment_logs_dir / "claude-code.txt").as_posix()
        await self.exec_as_agent(
            environment,
            command=('export PATH="$HOME/.local/bin:$PATH"; '
                     f'printf "%s" "${var.upper()}" | claude --verbose --output-format=stream-json {flags}'
                     f"--continue --print 2>&1 | tee -a {out}"),
            env={**self._cc_env(), var.upper(): message})

    async def run(self, instruction: str, environment, context) -> None:
        await super().run(instruction, environment, context)
        await self._gate_loop(environment)


class HermesCapped(Hermes):
    """Hermes with compaction capped below Haiku 5.5's 100K price tier (fairness: every harness is capped).
    Also: current Hermes dropped the `version` subcommand that Harbor 0.24.0 calls."""

    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.local/bin:$PATH"; hermes --version 2>/dev/null || echo hermes-unknown'

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        command = command.replace('&& hermes version', '&& (hermes --version || hermes --help | head -1 || true)')
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)

    @staticmethod
    def _build_config_yaml(model: str, max_turns: int | None = None) -> str:
        cfg = yaml.safe_load(Hermes._build_config_yaml(model, max_turns))
        cfg["compression"] = {"enabled": True, "threshold_tokens": 85000}
        return yaml.dump(cfg, default_flow_style=False)


class CodexBridged(GateMixin, __import__("harbor.agents.installed.codex", fromlist=["Codex"]).Codex):
    """Codex CLI on Claude via OpenRouter's Responses API (Codex only speaks Responses). Harbor keeps only the last
    '/' segment of the model name; this restores the full OpenRouter slug in the `codex exec` command."""

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        if "codex exec" in command and self.model_name and "/" in self.model_name:
            short = self.model_name.split("/")[-1]
            full = self.model_name.split("/", 1)[1] if self.model_name.startswith("openai/") else self.model_name
            command = command.replace(f"--model {short} ", f"--model {full} ")
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)


class OpenClaw24(__import__("harbor.agents.installed.openclaw", fromlist=["OpenClaw"]).OpenClaw):
    """OpenClaw >=2026.9 requires Node >=24.16; Harbor 0.24.0's adapter pins Node 22. Rewrite the nvm version only."""

    @staticmethod
    def _node24(cmd: str) -> str:
        return cmd.replace("nvm install 22", "nvm install 24").replace("nvm use 22", "nvm use 24")

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        if command.startswith("mkdir -p ~/.openclaw && cp ") and self._UPLOAD_CONFIG_FILENAME in command:
            # Harbor assumes /logs/agent is bind-mounted (true for local docker, not Modal): upload explicitly.
            await environment.upload_file(self.logs_dir / self._UPLOAD_CONFIG_FILENAME,
                                          f"{self._CONTAINER_LOGS_AGENT}/{self._UPLOAD_CONFIG_FILENAME}")
            await super().exec_as_root(environment, f"chmod a+r {self._CONTAINER_LOGS_AGENT}/{self._UPLOAD_CONFIG_FILENAME}")
        return await super().exec_as_agent(environment, self._node24(command), env=env, cwd=cwd, timeout_sec=timeout_sec)

    async def exec_as_root(self, environment, command, env=None, cwd=None, timeout_sec=None):
        return await super().exec_as_root(environment, self._node24(command), env=env, cwd=cwd, timeout_sec=timeout_sec)


class MathGated(OpenCodeGated):
    """E4a: re-derivation gate for final-answer math/physics tasks. Compares /app/answer.txt with the majority of 3
    fresh-context derivations (precomputed from the problem text only; gold answers are never used)."""

    async def _gate_check(self, environment, rnd: int) -> dict:
        import asyncio
        import sys as _sys
        _sys.path.insert(0, str(REPO))
        from bench.math_gate import equivalent
        task = environment.environment_name
        red = json.loads((REPO / "runs" / "matharena" / "rederive.json").read_text()).get(task) or {}
        maj = red.get("majority")
        if not maj:
            return {"status": "NO_SUITE"}
        g = json.loads((REPO / "runs" / "matharena" / "gold.json").read_text())[task]
        res = await self.exec_as_agent(environment, "cat /app/answer.txt 2>/dev/null || true")
        ans = (res.stdout or "").strip()
        same = bool(ans) and await asyncio.to_thread(equivalent, g["cat"], g["problem"], ans, maj)
        if same:
            return {"status": "BOUNDED_OK", "n_refs": 3}
        fb = (f"⟦Millennium Harness⟧ ✗ DISAGREES · re-derivation · independent derivations written without seeing "
              f"your work agree on a different final answer · round {rnd}/{self.options.max_rounds}\n"
              f"Their answer:\n```\n{maj[:1500]}\n```\nYour answer (/app/answer.txt):\n```\n{(ans or '(empty)')[:1500]}\n```\n"
              "They can be wrong. Re-derive carefully: check limiting cases, units/dimensions and a numeric spot-check "
              "(Python is available). Then write your final answer to /app/answer.txt — keep yours if you are certain, "
              "and say why.")
        return {"status": "REFUTED", "feedback": fb, "n_refs": 3}


class DafnyGated(OpenCodeGated):
    """E4b: structural Dafny rung. After the agent stops, run millennium_harness.dafny (frozen parts byte-identical, no
    escape hatches, audit clean, `dafny verify` 0 errors) in the sandbox; on failure resume with the concrete
    violations."""

    async def _gate_check(self, environment, rnd: int) -> dict:
        task = environment.environment_name
        frozen = REPO / "runs" / "tasks" / "vericoding" / task / "tests" / "frozen.json"
        if not frozen.exists():
            return {"status": "NO_SUITE"}
        tmp = f"/tmp/.sl-{secrets.token_hex(6)}"
        await self.exec_as_root(environment, f"mkdir -p {tmp} && chmod 777 {tmp}")
        await environment.upload_file(frozen, f"{tmp}/frozen.json")
        try:
            res = await self.exec_as_agent(environment, command=(
                f"cd /app && PYTHONPATH={SL_SRC} timeout 600 python3 -m millennium_harness.dafny check "
                f"/app/solution.dfy {tmp}/frozen.json {rnd} {self.options.max_rounds} || true"))
            raw = (res.stdout or "").strip().splitlines()
            r = json.loads(raw[-1]) if raw else {"status": "GATE_ERROR"}
        except Exception as e:  # noqa: BLE001
            r = {"status": "GATE_ERROR", "error": str(e)[:300]}
        finally:
            await self.exec_as_root(environment, f"rm -rf {tmp}")
        if r.get("status") in ("PROVED", "GATE_ERROR", "NO_SUITE"):
            return {k: v for k, v in r.items() if k != "verify"}
        return {"status": "REFUTED", "dafny_status": r.get("status"), "violations": r.get("violations"),
                "feedback": r.get("feedback", "")}
