"""Validate the preregistered estimator on simulated paired data (planted effect), before the freeze.

Simulates N tasks × k seeds for arms A/B with task-level difficulty heterogeneity and a planted +delta effect,
then checks bias of the seed-mean paired difference and the 95% task-bootstrap CI coverage.
  python bench/simulate_analysis.py
"""
import math
import random

from analyze import boot_ci


def sim(n_tasks=80, seeds=3, delta=0.06, reps=300, seed=0):
    rng = random.Random(seed)
    cover = 0
    ests = []
    for _ in range(reps):
        d = []
        for _t in range(n_tasks):
            base = rng.betavariate(0.6, 0.4)  # many easy tasks, a tail of hard ones (LCB/FreshCP-like)
            pa = min(1.0, base)
            pb = min(1.0, base + delta * (1 - base) / max(1e-9, 1 - 0.6))  # gain concentrated on unsolved mass
            a = sum(rng.random() < pa for _ in range(seeds)) / seeds
            b = sum(rng.random() < pb for _ in range(seeds)) / seeds
            d.append((b - a, pb - pa))
        obs = [x for x, _ in d]
        true = sum(t for _, t in d) / n_tasks
        lo, hi = boot_ci(obs, n=1000, seed=rng.randrange(10**6))
        cover += lo <= true <= hi
        ests.append(sum(obs) / n_tasks - true)
    bias = sum(ests) / reps
    sd = math.sqrt(sum((e - bias) ** 2 for e in ests) / reps)
    return cover / reps, bias, sd


if __name__ == "__main__":
    for n in (60, 90, 130):
        cov, bias, sd = sim(n_tasks=n)
        print(f"N={n:4d} seeds=3  CI coverage={cov:.3f} (target 0.95)  bias={bias:+.4f}  sd={sd:.4f}  ≈MDE80%={2.8*sd*100:.1f}pp")
