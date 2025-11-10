from __future__ import annotations
import shutil, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

@dataclass
class LeanEnvConfig:
    work_dir: Path
    max_steps: int = 6

class LeanProverEnv:
    """
    Minimal RL environment for Lean proofs.
    Task: prove nA + nB = nB + nA for random Nats; actions are tactic lines appended.
    Reward 1.0 if Lean compiles and closes the goal; else 0.0 at horizon.
    """
    def __init__(self, cfg: LeanEnvConfig):
        self.cfg = cfg
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self.actions = [
            "simp", "rfl", "simp [Nat.add_comm]", "have h := Nat.add_comm nA nB", "simpa using Nat.add_comm nA nB",
            "apply Nat.add_comm"
        ]
        self.step_count = 0
        self.lean_available = shutil.which("lean") is not None
        self._reset_problem()

    def _reset_problem(self):
        import numpy as np
        rng = np.random.default_rng(0)
        self.nA = int(rng.integers(1, 10))
        self.nB = int(rng.integers(1, 10))
        self.script: List[str] = []

    def reset(self):
        self.step_count = 0
        self._reset_problem()
        return self._state()

    def _state(self) -> Dict[str, Any]:
        return {"nA": self.nA, "nB": self.nB, "script": list(self.script), "step": self.step_count}

    def step(self, action_idx: int):
        self.step_count += 1
        if action_idx < 0 or action_idx >= len(self.actions):
            self.script.append("-- invalid action")
        else:
            self.script.append(self.actions[action_idx])

        done = False
        reward = 0.0
        success = False
        if self.lean_available:
            lean_file = self._emit_lean()
            proc = subprocess.run(["lean", "--make", str(lean_file)], cwd=str(self.cfg.work_dir),
                                  capture_output=True, text=True)
            success = (proc.returncode == 0)
            if success:
                reward = 1.0
                done = True
        # horizon
        if self.step_count >= self.cfg.max_steps:
            done = True
        return self._state(), reward, done, {"success": success}

    def _emit_lean(self) -> Path:
        code = f"""
def nA : Nat := {self.nA}
def nB : Nat := {self.nB}

theorem goal : nA + nB = nB + nA := by
  {'
  '.join(self.script)}
"""
        p = self.cfg.work_dir / "rl_goal.lean"
        p.write_text(code, encoding="utf-8")
        return p
