from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Any
from ..config import Config

@dataclass
class ExperimentPlan:
    assignment: str
    n_per_group: int
    seed: int
    test: str
    group_a: str
    group_b: str
    blocking: bool = False
    blocks: list[str] | None = None

    def __repr__(self):
        return (f"ExperimentPlan(assign={self.assignment}, n={self.n_per_group} per group, "
                f"test={self.test}, groups=({self.group_a},{self.group_b}), blocking={self.blocking})")

def design_experiment(cfg: Config, df: pd.DataFrame, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal "power" / sample size example: respect n_per_group from config.
    plan = ExperimentPlan(
        assignment=cfg.design.assignment,
        n_per_group=cfg.design.n_per_group,
        seed=cfg.design.seed,
        test=cfg.design.test,
        group_a=hypothesis["group_a"],
        group_b=hypothesis["group_b"],
        blocking=False,
        blocks=None,
    )
    return plan.__dict__
