from __future__ import annotations
from typing import List

UNESCO = ["Computer science", "Artificial intelligence", "Scientific methods", "Data analysis"]
AAAI = ["Planning", "Multiagent systems", "Scientific discovery", "Tool use"]
AAS  = ["Methods: data analysis"]  # keep minimal; extend per domain

def select_keywords(idea: str, data_desc: str, k: int = 6) -> List[str]:
    pool = UNESCO + AAAI + AAS
    # naive rank by presence
    ranked = sorted(pool, key=lambda w: (w.lower() in (idea+data_desc).lower()), reverse=True)
    return ranked[:k]
