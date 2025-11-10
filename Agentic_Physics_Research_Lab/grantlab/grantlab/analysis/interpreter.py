from __future__ import annotations
from typing import Dict, Any

def interpret_results(cfg, hypothesis: Dict[str, Any], plan: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    alpha = hypothesis["alpha"]
    p = result["p_value"]
    decision = "reject_null" if p < alpha else "fail_to_reject_null"
    direction = hypothesis.get("direction", "treatment > control")
    sign = "positive" if result["mean_b"] > result["mean_a"] else "negative or zero"
    interpretation = {
        "decision": decision,
        "alpha": alpha,
        "p_value": p,
        "effect_size_d": result.get("effect_size_d"),
        "difference_mean": result["mean_b"] - result["mean_a"],
        "direction_observed": sign,
        "notes": f"Planned test={result['test']}, observed means: {result['group_b']}={result['mean_b']:.3f}, {result['group_a']}={result['mean_a']:.3f}.",
        "supports_direction": (direction.strip() == "treatment > control" and result["mean_b"] > result["mean_a"]),
    }
    return interpretation
