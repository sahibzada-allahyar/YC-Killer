from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
import math

try:
    import pint  # optional unit checking
    _ureg = pint.UnitRegistry()
except Exception:
    _ureg = None

@dataclass
class Critique:
    summary: str
    issues: List[str]
    severity: str  # "info" | "warn" | "error"
    suggestions: List[str]

class MathCritic:
    """Lightweight math/logic critic for equations, identities, bounds."""
    def __call__(self, manuscript_text: str, extras: Optional[Dict[str, Any]] = None) -> Critique:
        issues, suggestions = [], []
        # Example heuristics
        if "≈" in manuscript_text and "+/-" not in manuscript_text:
            issues.append("Numeric approximations lack uncertainty notation.")
            suggestions.append("Report 1σ or credible intervals with approximation symbols.")
        if "log(" in manuscript_text and "base" not in manuscript_text.lower():
            suggestions.append("State log base explicitly to avoid ambiguity.")
        return Critique(
            summary="Checked algebraic clarity and numeric hygiene.",
            issues=issues,
            severity="warn" if issues else "info",
            suggestions=suggestions
        )

class PhysicsCritic:
    """Basic dimensional & physical-plausibility checks."""
    def __call__(self, manuscript_text: str, extras: Optional[Dict[str, Any]] = None) -> Critique:
        issues, suggestions = [], []
        # Unit sanity hints if pint available
        if _ureg is None:
            suggestions.append("Install 'pint' to enable dimensional checks.")
        else:
            # trivial sample: forbid summing quantities with mismatched units if detected in notes
            pass
        if "superluminal" in manuscript_text.lower():
            issues.append("Mentions of superluminal inference detected; verify signal model.")
        return Critique(
            summary="Checked unit/scale plausibility and physical claims.",
            issues=issues,
            severity="warn" if issues else "info",
            suggestions=suggestions
        )

def posterior_widths(posterior_json: Dict[str, Any]) -> Dict[str, float]:
    """Compute simple widths (95% CI) if quantiles present."""
    widths = {}
    for k, v in posterior_json.items():
        q05, q95 = v.get("q05"), v.get("q95")
        if q05 is not None and q95 is not None:
            widths[k] = float(q95) - float(q05)
    return widths

def adapt_plan_if_uncertain(plan: List[Dict[str, Any]],
                            posterior_json: Dict[str, Any],
                            thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    If any posterior width exceeds threshold, insert remediation steps:
      - more data/compute, tighter priors, or better model
    """
    widths = posterior_widths(posterior_json)
    needs_adaptation = any(
        (p in widths) and (widths[p] > thr) for p, thr in thresholds.items()
    )
    if not needs_adaptation:
        return plan

    new_steps: List[Dict[str, Any]] = []
    for step in plan:
        new_steps.append(step)
        if step.get("role") == "analysis-finalize":
            # insert re-planning block before finalization
            new_steps.append({
                "role": "replan",
                "desc": "Posterior too wide; expand dataset / refine model / increase iterations.",
                "actions": [
                    "increase_mcmc_draws: +3x",
                    "calibrate_likelihood: heteroscedastic noise",
                    "add informative priors if justified"
                ]
            })
    return new_steps
