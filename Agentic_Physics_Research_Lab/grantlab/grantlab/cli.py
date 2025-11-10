from __future__ import annotations
import typer
from rich import print as rprint
from pathlib import Path
from .pipeline import run_pipeline, run_lean_only
from .utils.io import write_text
from .config import load_config, default_config_yaml

app = typer.Typer(add_completion=False, help="GrantLab: end-to-end experiments + Lean proof checking")

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

def main():
    app()

if __name__ == "__main__":
    main()
