"""
Basic long/short backtesting engine.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def _neutralise(alpha: np.ndarray) -> np.ndarray:
    """Neutralize alpha to be market neutral."""
    z = alpha - np.nanmean(alpha)
    denom = np.nansum(np.abs(z))
    denom = denom if denom != 0 else 1.0
    return 10_000 * z / denom


def backtest(df: pl.DataFrame, lookahead: int = 1):
    """Basic backtesting function."""
    df = df.sort(["timestamp", "ticker"])
    df = df.with_columns(
        (pl.col("close").pct_change(periods=lookahead).shift(-lookahead)).alias("ret"),
    ).drop_nulls("ret")

    # Simple PnL calculation
    def _pnl(grp: pl.DataFrame):
        pos = _neutralise(grp["alpha"].to_numpy())
        pnl = pos * grp["ret"].to_numpy()
        return pl.Series("pnl", pnl)

    out = df.group_by("timestamp", maintain_order=True).map_groups(_pnl)
    curve = out["pnl"].cumsum()
    out = out.insert_column(0, "cum_pnl", curve)
    return out
