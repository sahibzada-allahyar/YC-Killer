from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Callable

def candidate_basis():
    # small library: 1, x, x^2, sin(kx), cos(kx) with k in {1,2,3}
    def f_const(x): return np.ones_like(x)
    def f_x(x): return x
    def f_x2(x): return x*x
    def f_s1(x): return np.sin(1.0*x)
    def f_s2(x): return np.sin(2.0*x)
    def f_s3(x): return np.sin(3.0*x)
    def f_c1(x): return np.cos(1.0*x)
    def f_c2(x): return np.cos(2.0*x)
    def f_c3(x): return np.cos(3.0*x)
    return [f_const, f_x, f_x2, f_s1, f_s2, f_s3, f_c1, f_c2, f_c3]

def evolve_symbolic_fit(x: np.ndarray, y: np.ndarray, max_terms: int = 4, seed: int = 7) -> Dict[str, Any]:
    """
    Greedy forward selection over basis with LS coeffs + small random restarts.
    Persist evaluator settings (seed) to artifacts to mirror AlphaEvolve ethos of reproducibility.
    """
    rng = np.random.default_rng(seed)
    B = candidate_basis()
    chosen: List[int] = []
    best_rss = np.inf
    best = None
    for r in range(1, max_terms+1):
        candidates = [i for i in range(len(B)) if i not in chosen]
        for i in rng.choice(candidates, size=len(candidates), replace=False):
            idxs = chosen + [i]
            Phi = np.column_stack([B[j](x) for j in idxs])
            # LS fit
            coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
            rss = np.mean((y - Phi @ coef)**2)
            if rss < best_rss:
                best_rss = rss
                best = (idxs, coef)
        chosen = best[0]
    idxs, coef = best
    return {
        "basis_idxs": idxs,
        "coef": coef.tolist(),
        "rss": float(best_rss),
        "seed": seed
    }

def make_predictor(x: np.ndarray, model: Dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    B = candidate_basis()
    idxs = model["basis_idxs"]
    coef = np.array(model["coef"])
    def yhat(theta_unused):
        # theta unused; model fixed after evolution; included to reuse bayesian API
        Phi = np.column_stack([B[j](x) for j in idxs])
        return Phi @ coef
    return yhat
