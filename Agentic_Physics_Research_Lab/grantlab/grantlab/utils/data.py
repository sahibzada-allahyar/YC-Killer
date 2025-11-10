from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic normalization: strip col names
    df.columns = [c.strip() for c in df.columns]
    return df
