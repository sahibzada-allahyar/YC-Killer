"""
Basic data loader for OHLC data.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List

from ..core.datatypes import DT


def load_ohlc(symbol: str, freq: str, root: str | Path) -> Dict[str, np.ndarray]:
    """Load OHLC data for a symbol and frequency."""
    files: List[Path] = sorted(Path(root, symbol, "ohlc", freq).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data for {symbol} {freq} under {root}")
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("ts")
    d = {c: df[c].to_numpy(float) for c in df.columns if c in DT.list()}
    d["timestamp"] = df["ts"].to_numpy(dtype="datetime64[us]")
    return d
