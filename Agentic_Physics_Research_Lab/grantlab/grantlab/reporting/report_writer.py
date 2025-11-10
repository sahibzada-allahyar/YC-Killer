from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from ..config import Config

def _env(template_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )

def write_report(cfg: Config,
                 hypothesis: Dict[str, Any],
                 plan: Dict[str, Any],
                 result: Dict[str, Any],
                 interpretation: Dict[str, Any],
                 lean_summary: Dict[str, Any],
                 run_dir: Path) -> Path:
    tdir = Path(__file__).parent / "templates"
    env = _env(tdir)
    template = env.get_template("report.md.j2")
    md = template.render(
        project=cfg.project,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        cfg=cfg,
        hypothesis=hypothesis,
        plan=plan,
        result=result,
        interpretation=interpretation,
        lean=lean_summary,
    )
    out = run_dir / "report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    return out
