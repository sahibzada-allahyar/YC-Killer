from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from typing import List, Optional, Dict, Any

@dataclass
class HypothesisCfg:
    type: str = "mean_difference"
    direction: str = "treatment > control"
    metric: str = "conversion"
    alpha: float = 0.05

@dataclass
class DesignCfg:
    assignment: str = "random"
    n_per_group: int = 20
    seed: int = 42
    test: str = "welch_t"  # "t","welch_t","mannwhitney"

@dataclass
class RunCfg:
    bootstrap_iters: int = 1000

@dataclass
class ReportCfg:
    out_dir: str = "outputs"
    format: str = "md"
    include_plots: bool = False

@dataclass
class Config:
    project: str
    dataset: str
    id_column: str = "id"
    outcome_column: str = "conversion"
    group_column: Optional[str] = "group"
    group_names: List[str] = field(default_factory=lambda: ["control", "treatment"])
    hypothesis: HypothesisCfg = field(default_factory=HypothesisCfg)
    design: DesignCfg = field(default_factory=DesignCfg)
    run: RunCfg = field(default_factory=RunCfg)
    report: ReportCfg = field(default_factory=ReportCfg)
    extra: Dict[str, Any] = field(default_factory=dict)

def load_config(path: Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    hyp = HypothesisCfg(**raw.get("hypothesis", {}))
    des = DesignCfg(**raw.get("design", {}))
    run = RunCfg(**raw.get("run", {}))
    rep = ReportCfg(**raw.get("report", {}))
    cfg = Config(
        project=raw["project"],
        dataset=raw["dataset"],
        id_column=raw.get("id_column", "id"),
        outcome_column=raw.get("outcome_column", "conversion"),
        group_column=raw.get("group_column", "group"),
        group_names=raw.get("group_names", ["control", "treatment"]),
        hypothesis=hyp,
        design=des,
        run=run,
        report=rep,
        extra=raw.get("extra", {}),
    )
    return cfg

def default_config_yaml() -> str:
    return """\
project: "New Experiment"
dataset: "data/dataset.csv"
id_column: "id"
outcome_column: "conversion"
group_column: ""          # leave empty to randomize
group_names: ["control","treatment"]
hypothesis:
  type: "mean_difference"
  direction: "treatment > control"
  metric: "conversion"
  alpha: 0.05
design:
  assignment: "random"
  n_per_group: 20
  seed: 42
  test: "welch_t"
run:
  bootstrap_iters: 1000
report:
  out_dir: "outputs"
  format: "md"
  include_plots: false
"""
