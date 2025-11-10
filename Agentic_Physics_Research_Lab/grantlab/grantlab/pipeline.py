from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from rich import print as rprint
from .config import Config, load_config
from .utils.io import ensure_dir, write_text, TimestampedPath
from .utils.data import load_dataset
from .hypothesis.generator import generate_hypothesis
from .design.experiment_designer import design_experiment
from .runner.executor import run_experiment
from .analysis.interpreter import interpret_results
from .reporting.report_writer import write_report
from .lean_agent.lean_agent import LeanAgent

def run_pipeline(config_path: Path):
    cfg: Config = load_config(config_path)
    out_dir = ensure_dir(Path(cfg.report.out_dir))
    run_dir = TimestampedPath(out_dir, prefix="run_").make_dir()

    rprint(f"[bold]Project:[/bold] {cfg.project}")
    rprint(f"[bold]Output:[/bold] {run_dir}")

    df = load_dataset(Path(cfg.dataset))

    # 1) Hypothesis
    hypothesis = generate_hypothesis(cfg, df)
    rprint(f"[green]Hypothesis generated:[/green] {hypothesis}")

    # 2) Design
    plan = design_experiment(cfg, df, hypothesis)
    rprint(f"[green]Design:[/green] {plan}")

    # 3) Run
    result = run_experiment(cfg, df, hypothesis, plan)
    rprint(f"[green]Run complete. p={result.get('p_value'):.4f}[/green]")

    # 4) Interpret
    interpretation = interpret_results(cfg, hypothesis, plan, result)
    rprint(f"[green]Interpretation:[/green] {interpretation['decision']}")

    # 5) Lean proof checking
    lean_agent = LeanAgent(run_dir)
    lean_summary = lean_agent.verify(hypothesis=hypothesis, plan=plan, result=result)

    # 6) Report
    report_path = write_report(cfg, hypothesis, plan, result, interpretation, lean_summary, run_dir)
    rprint(f"[bold green]Report written:[/bold green] {report_path}")

    # Save artifacts
    write_text(run_dir / "hypothesis.json", str(hypothesis))
    write_text(run_dir / "plan.json", str(plan))
    write_text(run_dir / "result.json", str(result))
    write_text(run_dir / "interpretation.json", str(interpretation))
    write_text(run_dir / "lean_summary.json", str(lean_summary))

def run_lean_only(config_path: Path):
    cfg: Config = load_config(config_path)
    df = load_dataset(Path(cfg.dataset))
    hypothesis = generate_hypothesis(cfg, df)
    plan = design_experiment(cfg, df, hypothesis)
    result = run_experiment(cfg, df, hypothesis, plan)

    out_dir = ensure_dir(Path(cfg.report.out_dir))
    run_dir = TimestampedPath(out_dir, prefix="lean_").make_dir()

    lean_agent = LeanAgent(run_dir)
    summary = lean_agent.verify(hypothesis=hypothesis, plan=plan, result=result)
    from .utils.io import write_text
    write_text(run_dir / "lean_summary.json", str(summary))
    from rich import print as rprint
    rprint(summary)
