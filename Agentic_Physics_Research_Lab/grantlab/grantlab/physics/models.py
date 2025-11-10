from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Any

@dataclass
class HarmonicOscillatorParams:
    omega: float      # natural frequency
    zeta: float       # damping ratio
    amp: float        # initial amplitude
    v0: float = 0.0   # initial velocity
    sigma: float = 0.02  # obs noise std

def _sho_ode(t, y, omega, zeta):
    # y = [x, v]
    x, v = y
    dxdt = v
    dvdt = -2.0*zeta*omega*v - (omega**2)*x
    return [dxdt, dvdt]

def simulate_sho(p: HarmonicOscillatorParams, t_span: Tuple[float,float], t_eval: np.ndarray) -> np.ndarray:
    y0 = [p.amp, p.v0]
    sol = solve_ivp(lambda t, y: _sho_ode(t, y, p.omega, p.zeta),
                    t_span=t_span, y0=y0, t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9)
    x = sol.y[0]
    if p.sigma and p.sigma > 0:
        x = x + np.random.normal(0.0, p.sigma, size=len(x))
    return x

# simple regressors (power law / exponential / linear) used in hypothesis scan
def model_linear(x, a, b):
    return a*x + b

def model_power(x, a, b):
    # y = a * x^b for x>0
    return a * np.power(np.clip(x, 1e-12, None), b)

def model_exponential(x, a, b):
    # y = a * exp(b x)
    return a * np.exp(b*x)
