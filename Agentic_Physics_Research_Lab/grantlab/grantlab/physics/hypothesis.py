from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Callable, List
from .models import model_linear, model_power, model_exponential, HarmonicOscillatorParams

@dataclass
class PhysicsHypothesis:
    family: str          # "linear"|"power"|"exponential"|"sho"
    params_init: Dict[str, float]
    metric: str          # which column to fit/predict
    alpha: float         # credible/conf threshold

def _fit_simple(x, y, f: Callable, guess: List[float]):
    # tiny LSQ (2-param) by linearization where possible; else grid
    # For robustness, we will do small random restarts
    best = None
    for _ in range(8):
        g = np.array(guess) * (1.0 + 0.2*np.random.randn(len(guess)))
        # simple local search
        for _j in range(200):
            # finite-diff gradient
            eps = 1e-6
            yhat = f(x, *g)
            r = y - yhat
            loss = np.mean(r*r)
            if best is None or loss < best[0]:
                best = (loss, g.copy())
            # numeric grad (coordinate)
            for k in range(len(g)):
                g2 = g.copy()
                g2[k] += eps
                r2 = y - f(x, *g2)
                grad = (np.mean(r2*r2) - loss) / eps
                g[k] -= 1e-2*grad
    return best[1], best[0]

def _aic(n, rss, k):
    return n*np.log(rss/n + 1e-15) + 2*k

def propose_physics_hypothesis(df: pd.DataFrame, metric: str, alpha: float=0.05) -> Dict[str, Any]:
    """
    Scan simple families and pick by AIC. If df has columns t,y with oscillations -> propose SHO.
    """
    cols = df.columns
    # default: try linear/power/exp on (x,y) if present
    fams = []
    if set(["x", "y"]).issubset(cols):
        x = df["x"].values
        y = df["y"].values
        # linear
        p_lin, rss_lin = _fit_simple(x, y, model_linear, [1.0, 0.0])
        aic_lin = _aic(len(x), rss_lin, 2)
        fams.append(("linear", {"a": float(p_lin[0]), "b": float(p_lin[1])}, aic_lin))
        # power
        p_pow, rss_pow = _fit_simple(x, y, model_power, [1.0, 1.0])
        aic_pow = _aic(len(x), rss_pow, 2)
        fams.append(("power", {"a": float(p_pow[0]), "b": float(p_pow[1])}, aic_pow))
        # exponential
        p_exp, rss_exp = _fit_simple(x, y, model_exponential, [1.0, 0.0])
        aic_exp = _aic(len(x), rss_exp, 2)
        fams.append(("exponential", {"a": float(p_exp[0]), "b": float(p_exp[1])}, aic_exp))
    # SHO detection heuristic: if 't' present and metric present, inspect spectral peak
    if set(["t", metric]).issubset(cols):
        t = df["t"].values
        y = df[metric].values
        y0 = y - y.mean()
        # crude spectral estimate
        freqs = np.fft.rfftfreq(len(t), d=(t[1]-t[0] if len(t)>1 else 1.0))
        spec = np.abs(np.fft.rfft(y0))
        kmax = np.argmax(spec[1:]) + 1 if len(spec)>1 else 1
        omega = 2*np.pi*freqs[kmax] if kmax < len(freqs) else 1.0
        zeta = 0.05
        fams.append(("sho", {"omega": float(omega), "zeta": float(zeta), "amp": float(np.max(np.abs(y0)))}, np.inf))  # score later

    # select best by AIC (except sho: prefer if strong spectral peak)
    choose = min([f for f in fams if f[0]!="sho"], key=lambda z: z[2]) if any(f[0]!="sho" for f in fams) else None
    if choose is not None:
        family, params, _ = choose
        return PhysicsHypothesis(family=family, params_init=params, metric=metric, alpha=alpha).__dict__
    else:
        # fallback to SHO if proposed
        for f in fams:
            if f[0]=="sho":
                return PhysicsHypothesis(family="sho", params_init=f[1], metric=metric, alpha=alpha).__dict__
    # last resort
    return PhysicsHypothesis(family="linear", params_init={"a":1.0,"b":0.0}, metric=metric, alpha=alpha).__dict__
