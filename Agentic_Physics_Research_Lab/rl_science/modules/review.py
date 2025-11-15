from __future__ import annotations
from typing import Dict, Any
import re

def referee(ctx: Dict[str,Any]) -> str:
    """
    Lightweight referee: flags missing plots, short methods, missing stats in results, etc.
    Mirrors the Reviewer prompt's spirit (Figure 8), without multimodal OCR.
    """
    issues = []
    methods = ctx.get("methods","")
    results = ctx.get("results","")
    plots = ctx.get("plots", [])

    if len(methods.split()) < 250:
        issues.append("Methods too short for reproducibility.")
    if "