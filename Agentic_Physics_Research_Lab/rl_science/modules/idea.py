from __future__ import annotations
from typing import List

def generate_ideas(data_description: str) -> List[str]:
    return [
        "Quantify structure-to-outcome maps with uncertainty-aware pipelines.",
        "Compare graph vs. tabular encodings for downstream predictions.",
        "Automate ablations to isolate causal features in results.",
        "Design adaptive plans that re-route after failed code execution.",
        "Integrate reviewer feedback into active improvement loops."
    ]

def critique_ideas(ideas: List[str], data_description: str) -> List[str]:
    return [f"Critique: '{s}' — needs measurable success criteria and data paths." for s in ideas]

def select_and_improve(ideas: List[str], critiques: List[str]) -> str:
    best = ideas[0]
    return best + " Success = pass unit tests, >=3 plots, and reviewer score ≥5."
