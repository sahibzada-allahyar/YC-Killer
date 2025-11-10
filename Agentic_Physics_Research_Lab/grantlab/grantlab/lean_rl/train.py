from __future__ import annotations
from pathlib import Path
from .env_lean import LeanProverEnv, LeanEnvConfig
from .agent_random import run_random_episode

def train_tactic_agent(episodes: int = 20, max_steps: int = 6):
    env = LeanProverEnv(LeanEnvConfig(work_dir=Path(".lean_rl"), max_steps=max_steps))
    wins = 0
    for ep in range(episodes):
        r, info = run_random_episode(env, max_steps=max_steps)
        wins += int(r > 0.0)
    return {"episodes": episodes, "wins": wins, "win_rate": wins / max(episodes,1)}
