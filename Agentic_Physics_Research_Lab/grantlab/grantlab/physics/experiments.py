from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from .models import HarmonicOscillatorParams, simulate_sho, model_linear, model_power, model_exponential

@dataclass
class PhysicsPlan:
    family: str
    # for time-series experiments
    t0: float
    t1: float
    n_times: int
    seed: int

def design_physics_experiment(hypothesis: Dict[str, Any], df: pd.DataFrame | None = None) -> Dict[str, Any]:
    fam = hypothesis["family"]
    if fam == "sho":
        return PhysicsPlan(family=fam, t0=0.0, t1=10.0, n_times=400, seed=123).__dict__
    else:
        return PhysicsPlan(family=fam, t0=0.0, t1=1.0, n_times=100, seed=123).__dict__

def run_physics_experiment(hypothesis: Dict[str, Any], plan: Dict[str, Any], df: pd.DataFrame | None = None) -> Dict[str, Any]:
    rng = np.random.default_rng(plan["seed"])
    family = hypothesis["family"]
    out = {}
    if family == "sho":
        p = hypothesis["params_init"]
        hp = HarmonicOscillatorParams(omega=p["omega"], zeta=p["zeta"], amp=p["amp"], v0=0.0, sigma=0.02)
        t = np.linspace(plan["t0"], plan["t1"], plan["n_times"])
        y = simulate_sho(hp, (plan["t0"], plan["t1"]), t)
        out = {"t": t.tolist(), "y": y.tolist(), "meta":{"family": "sho", "params_init": p}}
    else:
        # simulate synthetic x->y using initial params
        x = np.linspace(plan["t0"], plan["t1"], plan["n_times"])
        p = hypothesis["params_init"]
        if family == "linear":
            y = model_linear(x, p["a"], p["b"]) + rng.normal(0, 0.02, size=len(x))
        elif family == "power":
            y = model_power(x, p["a"], p["b"]) + rng.normal(0, 0.02, size=len(x))
        elif family == "exponential":
            y = model_exponential(x, p["a"], p["b"]) + rng.normal(0, 0.02, size=len(x))
        else:
            raise ValueError(f"unknown family {family}")
        out = {"x": x.tolist(), "y": y.tolist(), "meta":{"family": family, "params_init": p}}
    return out
