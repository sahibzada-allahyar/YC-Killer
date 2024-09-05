"""
Basic strategy idea generator.
"""
from __future__ import annotations

import json
from pathlib import Path

from .llm_client import chat
from .prompts import IDEA_PROMPT

_ARTIFACTS = Path("artifacts/ideas")
_ARTIFACTS.mkdir(parents=True, exist_ok=True)


def generate() -> Path:
    """Generate a new trading strategy idea."""
    response = chat(system="You are a quantitative strategist.", user=IDEA_PROMPT)
    data = json.loads(response)
    idea_text = data["idea"]
    fname = _ARTIFACTS / f"idea_{hash(idea_text) & 0xFFFFFFFF:x}.json"
    fname.write_text(json.dumps(data, indent=2))
    return fname
