"""
Feature compilation & persistence.

We hash the (frequency, expression) pair so re‑runs do not duplicate work.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl

from ..core.parser import evaluate
from ..core.datatypes import DT


def _hash(freq: str, expr: str) -> str:
    payload = json.dumps({"f": freq, "e": expr}).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


@dataclass(slots=True, frozen=True)
class FeatureSet:
    freq: str
    expr: str
    hash: str

    @classmethod
    def from_expr(cls, freq: str, expr: str):
        return cls(freq=freq, expr=expr, hash=_hash(freq, expr))

    # ------------------------------------------------------------------ #
    def build(
        self, symbol: str, data_root: Path, feature_root: Path
    ) -> Path:  # returns file written
        from .loader import load_ohlc

        arrays: Dict[str, np.ndarray] = load_ohlc(symbol, self.freq, data_root)
        y = evaluate(self.expr, arrays)
        out_df = pl.DataFrame(
            {
                "timestamp": arrays["timestamp"],
                "ticker": symbol,
                "alpha": y,
            }
        ).drop_nulls("alpha")
        dest_folder = feature_root / self.hash
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest = dest_folder / f"{symbol}_{self.freq}.parquet"
        out_df.write_parquet(dest)
        return dest
