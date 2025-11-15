from __future__ import annotations
from typing import Dict, List

def _bool(x) -> int:
    return 1 if x else 0

def compute_rewards(ctx: Dict, errors: List[str]) -> float:
    r = 0.0
    # Idea & keywords
    r += 0.5 * _bool(ctx.get("idea"))
    r += 0.25 * _bool(ctx.get("keywords"))

    # Methods: non-empty & ~500 words
    m = ctx.get("methods") or ""
    if m:
        words = len(m.split())
        r += 0.5
        if 350 <= words <= 800:
            r += 0.25

    # Analysis: results, plots, and printed stats
    if ctx.get("results"):
        r += 1.0
    n_plots = len(ctx.get("plots", []))
    if n_plots >= 3:
        r += 0.5
    # Printed quantitative info (engineer prints)
    stdout = ctx.get("stdout","")
    if stdout and any(k in stdout.lower() for k in ["mean","std","auc","acc","mape","rmse","sigma","log","count"]):
        r += 0.5

    # Guardrails
    stderr = ctx.get("stderr","")
    if "dummy data" in (ctx.get("results") or "") or "np.random" in (ctx.get("results") or ""):
        r -= 2.0  # harsh penalty
    if errors:
        r -= min(1.0, 0.2*len(errors))

    # Paper & Review
    if ctx.get("paper_tex"):
        r += 0.75
    if ctx.get("referee_md"):
        # rough heuristic: fewer 'flaw'/'fail' words → higher reward
        rep = ctx["referee_md"].lower()
        penalties = rep.count("flaw") + rep.count("fail") + rep.count("insufficient")
        r += max(0.0, 0.75 - 0.2*penalties)

    return r
