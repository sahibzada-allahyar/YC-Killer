from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class MHConfig:
    step_std: float = 0.03
    iters: int = 2000
    burn: int = 500

def mh_sampler(logpdf: Callable[[np.ndarray], float],
               x0: np.ndarray,
               cfg: MHConfig = MHConfig()) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = x0.copy()
    logp = logpdf(x)
    chain = []
    for i in range(cfg.iters):
        prop = x + rng.normal(0.0, cfg.step_std, size=x.shape)
        lp2 = logpdf(prop)
        if np.log(rng.random()) < (lp2 - logp):
            x, logp = prop, lp2
        chain.append(x.copy())
    return np.array(chain[cfg.burn:])

def bayes_fit_gaussian(y: np.ndarray, yhat_fn: Callable[[np.ndarray], np.ndarray],
                       theta0: np.ndarray, sigma0: float = 0.05) -> Dict[str, Any]:
    """
    Simple Gaussian likelihood on residuals; flat priors on theta. sigma fixed.
    """
    def logpdf(theta):
        yhat = yhat_fn(theta)
        res = y - yhat
        return -0.5*np.sum((res/sigma0)**2)
    chain = mh_sampler(logpdf, theta0)
    post_mean = chain.mean(axis=0)
    hdi_low = np.percentile(chain, 2.5, axis=0)
    hdi_hi  = np.percentile(chain, 97.5, axis=0)
    return {"chain": chain, "post_mean": post_mean, "hdi_low": hdi_low, "hdi_hi": hdi_hi}
