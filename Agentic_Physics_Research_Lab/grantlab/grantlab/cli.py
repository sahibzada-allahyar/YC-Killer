from __future__ import annotations
import typer
from rich import print as rprint
from pathlib import Path

from .pipeline import run_pipeline, run_lean_only
from .utils.io import write_text
from .config import load_config, default_config_yaml

# new imports
from .physics.runner_physics import run_physics_pipeline
from .paper_agent.graph_writer import write_paper_from_run
from .lean_rl.train import train_tactic_agent

app = typer.Typer(add_completion=False, help="GrantLab: end-to-end experiments + Lean proof checking + Physics copilot")

@app.command()
def init(target_dir: str = typer.Argument(..., help="Directory to create with a starter config")):
    p = Path(target_dir)
    p.mkdir(parents=True, exist_ok=True)
    cfg = p / "config.yml"
    if cfg.exists():
        rprint(f"[yellow]Config already exists at {cfg}[/yellow]")
    else:
        write_text(cfg, default_config_yaml())
        rprint(f"[green]Wrote starter config to {cfg}[/green]")

@app.command()
def run(config: str = typer.Argument(..., help="Path to config.yml")):
    run_pipeline(Path(config))

@app.command("lean-check")
def lean_check(config: str = typer.Argument(..., help="Path to config.yml")):
    run_lean_only(Path(config))

# ---------- NEW COMMANDS ----------

@app.command("physics-run")
def physics_run(config: str = typer.Argument(..., help="Physics config.yml (see examples/physics/oscillator.yml)")):
    run_physics_pipeline(Path(config))

@app.command("write-paper")
def write_paper(run_dir: str = typer.Argument(..., help="Run directory with artifacts (report, plan, results)")):
    p = Path(run_dir)
    out = write_paper_from_run(p)
    rprint(f"[green]Wrote paper to {out}[/green]")

@app.command("rl-lean-train")
def rl_lean_train(
    episodes: int = typer.Option(20, help="Number of training episodes"),
    max_steps: int = typer.Option(6, help="Max tactic steps per episode")
):
    out = train_tactic_agent(episodes=episodes, max_steps=max_steps)
    rprint(f"[green]RL session summary: {out}[/green]")

def main():
    app()

if __name__ == "__main__":
    main()
