from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from typing import Dict, Any
from ..config import Config

@dataclass
class Hypothesis:
    type: str
    metric: str
    group_a: str
    group_b: str
    direction: str
    alpha: float

    def __repr__(self):
        return (f"Hypothesis(type={self.type}, metric={self.metric}, "
                f"{self.group_b} {self.direction.split('>')[-1] if '>' in self.direction else self.direction} {self.group_a}, alpha={self.alpha})")

def generate_hypothesis(cfg: Config, df: pd.DataFrame) -> Dict[str, Any]:
    # If groups present, infer control/treatment order by group_names
    group_names = cfg.group_names
    if len(group_names) != 2:
        raise ValueError("Exactly two groups required for mean_difference hypothesis.")
    group_a, group_b = group_names[0], group_names[1]
    hyp = Hypothesis(
        type=cfg.hypothesis.type,
        metric=cfg.hypothesis.metric,
        group_a=group_a,
        group_b=group_b,
        direction=cfg.hypothesis.direction,
        alpha=cfg.hypothesis.alpha,
    )
    return hyp.__dict__
