from __future__ import annotations

TEMPLATE = """\
We proceed in four stages: (i) Data ingestion with explicit absolute paths and schema checks.
(ii) Analysis code by an Engineer that prints all quantitative stats to stdout (no file I/O reliance).
(iii) Plot generation with log-scale heuristics for high dynamic ranges. (iv) A Researcher writes
a ~500-word results narrative referencing plots and key metrics. We forbid dummy data generation
and abort if inputs are missing. We cap plan complexity to ≤ 6 steps and allow adaptive re-planning
on failures.
"""

def design(data_description: str, idea: str) -> str:
    return TEMPLATE + f"\nData paths: {data_description[:200]}"
