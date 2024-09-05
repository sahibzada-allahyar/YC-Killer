from __future__ import annotations

import json
from pathlib import Path
from typing import List

from jinja2 import Template

from ..core.datatypes import DT
from ..core.registry import list_transforms
from .llm_client import chat
from .prompts import ALPHA_PROMPT

_OUT = Path("artifacts/alphas")
_OUT.mkdir(parents=True, exist_ok=True)


def generate(idea_file: Path, n: int = 5) -> Path:
    idea = json.loads(idea_file.read_text())["idea"]
    tmpl = Template(ALPHA_PROMPT)
    user = tmpl.render(
        idea=idea, n=n, cols=DT.list(), transforms=list_transforms()
    )
    resp = chat(system="You are an alpha feature engineer.", user=user)
    alphas: List[dict] = json.loads(resp)
    path = _OUT / f"alpha_{idea_file.stem}_{len(alphas)}.json"
    path.write_text(json.dumps(alphas, indent=2))
    return path
