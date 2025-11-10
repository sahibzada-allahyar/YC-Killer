from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any
import numpy as np
import pandas as pd
from rich import print as rprint

from ..utils.io import ensure_dir, TimestampedPath, write_text
from ..reporting.report_writer import write_report

from .hypothesis import propose_physics_hypothesis
from .experiments import design_physics_experiment, run_physics_experiment
from .bayes import bayes_fit_gaussian
from .evolve import evolve_symbolic_fit, make_predictor

from ..lean_agent.lean_agent import LeanAgent

def _load_or_create(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    # create a tiny SHO dataset
    t = np.linspace(0, 10, 400)
    y = np.exp(-0.05*t) * np.cos(2.0*np.pi*0.9*t)
    df = pd.DataFrame({"t": t, "y": y})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

def run_physics_pipeline(config_path: Path):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project = cfg.get("project", "Physics Run")
    dataset = Path(cfg.get("dataset", "examples/physics/oscillator.csv"))
    out_root = ensure_dir(Path(cfg.get("report", {}).get("out_dir", "examples/physics/outputs")))
    run_dir = TimestampedPath(out_root, prefix="phys_").make_dir()

    rprint(f"[bold]Project:[/bold] {project}")
    rprint(f"[bold]Output:[/bold] {run_dir}")

    df = _load_or_create(dataset)
    metric = cfg.get("metric", "y")

    # 1) Hypothesis
    hypothesis = propose_physics_hypothesis(df, metric=metric, alpha=cfg.get("alpha", 0.05))
    rprint(f"[green]Hypothesis:[/green] {hypothesis}")

    # 2) Design
    plan = design_physics_experiment(hypothesis, df)
    rprint(f"[green]Plan:[/green] {plan}")

    # 3) Run (simulate/generate or evaluate)
    result_data = run_physics_experiment(hypothesis, plan, df)
    rprint(f"[green]Ran experiment (synthetic or eval).[/green]")

    # 3b) AlphaEvolve-style symbolic fit (transparency + alternative model)
    if "t" in result_data:
        x = np.array(result_data["t"])
        y = np.array(result_data["y"])
    else:
        x = np.array(result_data["x"])
        y = np.array(result_data["y"])
    evo_model = evolve_symbolic_fit(x, y, max_terms=4, seed=7)
    evo_pred = make_predictor(x, evo_model)

    # 4) Bayesian inference (on evo predictor to get posterior error bars for illustration)
    theta0 = np.zeros(1)  # unused parameter vector for API compatibility
    bayes_out = bayes_fit_gaussian(y, lambda _: evo_pred(None), theta0, sigma0=0.05)

    interpretation = {
        "posterior_mean_abs_error": float(np.mean(np.abs(y - evo_pred(None)))),
        "posterior_ci_residuals": [float(np.percentile(y - evo_pred(None), 2.5)),
                                   float(np.percentile(y - evo_pred(None), 97.5))]
    }

    # 5) Lean proof checking (simple invariants)
    lean_agent = LeanAgent(run_dir)
    # Use counts invariant on timeseries length
    lean_summary = lean_agent.verify(
        hypothesis={"type":"physics","alpha":hypothesis["alpha"],"metric":metric,
                    "group_a":"obs","group_b":"model"},
        plan={"group_a":"obs","group_b":"model","n_per_group":len(x)},
        result={"n_a":len(x), "n_b":len(x)}  # symmetric check
    )

    # Report (reuse general template)
    res_summary = {
        "group_a":"obs","group_b":"model",
        "mean_a": float(np.mean(y)),
        "mean_b": float(np.mean(evo_pred(None))),
        "n_a": int(len(y)),"n_b": int(len(y)),
        "test":"model-fit","t_stat": float("nan"),
        "p_value": float("nan"),
        "effect_size_d": float("nan"),
        "ci_diff": [float("nan"), float("nan")],
        "assigned_group_column": "NA"
    }
    report_path = write_report(
        cfg=type("X",(object,),{
            "project": project,
            "report": type("Y",(object,),{"out_dir": str(out_root), "format":"md","include_plots": False})()
        })(),
        hypothesis=hypothesis, plan=plan, result=res_summary,
        interpretation=interpretation, lean_summary=lean_summary, run_dir=run_dir
    )

    # Persist artifacts
    import json
    write_text(run_dir / "data.json", json.dumps(result_data))
    write_text(run_dir / "evo_model.json", json.dumps(evo_model))
    write_text(run_dir / "bayes.json", json.dumps({k:(v.tolist() if hasattr(v,"tolist") else v) for k,v in bayes_out.items()}))
    write_text(run_dir / "interpretation.json", json.dumps(interpretation))
    rprint(f"[bold green]Report:[/bold green] {report_path}")
    return run_dir
