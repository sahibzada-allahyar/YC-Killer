from __future__ import annotations
from typing import Dict, Any
from ..utils.exec_sandbox import run_python_code
from ..utils.plotting import ensure_matplotlib_agg

ENGINEER_PREAMBLE = """
import os,sys,math,statistics
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Engineer instructions from the paper:
# - Adjust axes/binning and choose log scales when ranges span orders of magnitude
# - Print ALL quantitative info needed for interpretation (researcher has NO file I/O)
# - Avoid creating dummy/synthetic data. Fail if data missing.
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 200)
"""

def run(data_description: str, idea: str, methods: str) -> Dict[str, Any]:
    ensure_matplotlib_agg()
    if not data_description:
        raise ValueError("No data description provided.")
    code = f"""{ENGINEER_PREAMBLE}
print("## ENGINEER: starting analysis")
print("## IDEA:", {repr(idea[:200])} if {bool(idea)} else "N/A")
print("## METHODS_LEN:", {len(methods.split())} if {bool(methods)} else 0)

# === USER-SPECIFIC LOADING HOOK ===
# Expect paths in the data_description; do not synthesize data.
desc = {repr(data_description)}
if "path" not in desc.lower() and "/" not in desc:
    raise RuntimeError("Data path not found in description; aborting as per 'no dummy data' rule.")

# Example: try to find CSVs in provided path keywords (you can extend this per project)
import re
m = re.search(r"(/[^\\s]+\\.csv)", desc)
if not m:
    print("WARN: no explicit CSV found in description; engineer prints only a stub.")
    print("STATS: none")
else:
    csv_path = m.group(1)
    df = pd.read_csv(csv_path)
    print("DATA_SHAPE:", df.shape)
    # Basic numeric summary
    numeric = df.select_dtypes(include=[float,int])
    if not numeric.empty:
        stats = numeric.describe().T
        print("NUMERIC_DESCRIBE:\\n", stats)
        # Quick heuristics for log scale
        # (if dynamic range > 1e3, suggest log)
        dyn = (numeric.max() - numeric.min()).replace(0,1)
        suggest_log = (dyn > 1e3).sum()
        print("SUGGEST_LOG_COLUMNS:", int(suggest_log))

        # Plot up to 3 histograms
        import matplotlib.pyplot as plt
        cols = list(numeric.columns)[:3]
        for c in cols:
            s = numeric[c].dropna()
            if s.empty: 
                continue
            plt.figure()
            try:
                s.plot(kind="hist", bins=50)
            except Exception:
                plt.hist(s.values, bins=50)
            plt.title(f"Hist {c}")
            plt.tight_layout()
            plt.savefig(f"plots/hist_{c}.png")

    print("## ENGINEER: done")
"""
    stdout, stderr, artifacts = run_python_code(code, workdir=".")
    results_md = "\n".join([
        "### Analysis Log",
        "