from __future__ import annotations
import random
from .env_lean import LeanProverEnv

def run_random_episode(env: LeanProverEnv, max_steps: int = 6):
    obs = env.reset()
    for t in range(max_steps):
        a = random.randrange(len(env.actions))
        obs, reward, done, info = env.step(a)
        if done:
            return reward, info
    return 0.0, {"success": False}
