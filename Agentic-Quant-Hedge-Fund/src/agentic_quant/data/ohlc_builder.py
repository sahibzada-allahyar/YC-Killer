"""
Build mid‑price OHLC(+V) parquet partitions from raw Tardis quote+trade csv.gz.

Expected directory structure:

raw/
 └── binance/
     └── futures/
         └── quotes/
             binance-futures_quotes_20250101_BTCUSDT.csv.gz
         └── trades/
             binance-futures_trades_20250101_BTCUSDT.csv.gz

Output:

processed/{symbol}/ohlc/{freq}/YYYY/MM/*.parquet
"""
from __future__ import annotations

import gzip
import pathlib
from typing import Iterable

import polars as pl
from tqdm import tqdm

from ..core.datatypes import DT

_FREQS = ["1min", "5min", "15min", "1h", "4h", "12h", "1d"]


def _read_quotes(path: pathlib.Path) -> pl.DataFrame:
    return (
        pl.read_csv(path, compression="gzip")
        .with_columns(pl.col("timestamp").cast(pl.Int64).alias("ts"))
        .with_columns(
            (pl.col("bid_price") + pl.col("ask_price")) / 2.0
        ).rename({"literal": DT.OPEN_BID.value})  # placeholder, we just keep both
    )


def _resample_mid(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    return (
        df.sort(by="ts")
        .with_columns(pl.col("ts").cast(pl.Datetime("us")))
        .groupby_dynamic(index_column="ts", every=freq, closed="right", label="right")
        .agg(
            [
                pl.col("mid_price").first().alias(DT.OPEN.value),
                pl.col("mid_price").max().alias(DT.HIGH.value),
                pl.col("mid_price").min().alias(DT.LOW.value),
                pl.col("mid_price").last().alias(DT.CLOSE.value),
                pl.col("ask_price").first().alias(DT.OPEN_ASK.value),
                pl.col("ask_price").last().alias(DT.CLOSE_ASK.value),
                pl.col("bid_price").first().alias(DT.OPEN_BID.value),
                pl.col("bid_price").last().alias(DT.CLOSE_BID.value),
            ]
        )
        .drop_nulls(DT.CLOSE.value)
    )


def build_for_symbol(symbol: str, quotes_dir: pathlib.Path, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(quotes_dir.glob(f"*_{symbol}.csv.gz"))
    if not files:
        return

    for freq in _FREQS:
        frames: list[pl.DataFrame] = []
        for f in tqdm(files, desc=f"{symbol} {freq}", leave=False):
            dfq = _read_quotes(f)
            frames.append(_resample_mid(dfq, freq))
        if not frames:
            continue

        df = pl.concat(frames)
        partitioned = (
            df.with_columns(
                pl.col("ts").dt.year().alias("year"),
                pl.col("ts").dt.month().alias("month"),
            )
            .partition_by(["year", "month"], as_dict=True)
        )
        for (y, m), part in partitioned.items():
            folder = out_dir / freq / f"{y:04d}/{m:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            part.write_parquet(folder / f"{symbol}_{freq}_{y}{m:02d}.parquet")
