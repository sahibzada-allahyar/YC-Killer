from __future__ import annotations
import json
import os
from typing import Dict, Any, List
import re

def has_min_plots(plot_dir: str, n: int) -> bool:
    if not os.path.isdir(plot_dir):
        return False
    cnt = sum(1 for f in os.listdir(plot_dir) if f.lower().endswith((".png",".pdf",".jpg",".svg")))
    return cnt >= n

def posterior_widths_ok(posteriors_json_path: str, targets: Dict[str, Dict[str, float]]) -> bool:
    if not os.path.isfile(posteriors_json_path):
        return False
    P = json.load(open(posteriors_json_path))
    for k, spec in targets.items():
        if k not in P: return False
        q05, q95 = P[k].get("q05"), P[k].get("q95")
        if q05 is None or q95 is None: return False
        if (q95 - q05) > spec["width_95_max"]: return False
    return True

def code_executed_ok(log_text: str) -> bool:
    return "EXECUTION_OK" in log_text and "TRACEBACK" not in log_text.upper()

def citations_present(md_text: str) -> bool:
    # crude: require at least one [cite]
    return bool(re.search(r"\\cite{|\\citep{|\\citet{", md_text))
