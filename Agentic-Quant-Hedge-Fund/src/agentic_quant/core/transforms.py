"""
Vectorised implementations for every operator allowed in prompts.
NumPy + Numba are used; callers pass **polars.Series** or **numpy.ndarray**.
All functions **must** have pure inputs, deterministic outputs, and side‑effect‑free.
"""
from __future__ import annotations

import numpy as np
import numba as nb
from functools import wraps
from typing import Callable, Dict, Tuple

__all__ = ["registry", "arity"]

# --------------------------------------------------------------------- #
# Helper decorators                                                     #
# --------------------------------------------------------------------- #
registry: Dict[str, Callable] = {}
arity: Dict[str, Tuple[int, int]] = {}   # name -> (min_args, max_args)


def _register(name: str, n_args: Tuple[int, int]):
    def deco(fn: Callable):
        registry[name] = fn
        arity[name] = n_args
        return wraps(fn)(fn)

    return deco


# --------------------------------------------------------------------- #
# Element‑wise ops (numba‑jit where it pays)                            #
# --------------------------------------------------------------------- #
@_register("add", (2, 2))
@nb.njit(cache=True, fastmath=True)
def add(x1, x2):
    return x1 + x2


@_register("sub", (2, 2))
@nb.njit(cache=True, fastmath=True)
def sub(x1, x2):
    return x1 - x2


@_register("mul", (2, 2))
@nb.njit(cache=True, fastmath=True)
def mul(x1, x2):
    return x1 * x2


@_register("div", (2, 2))
@nb.njit(cache=True, fastmath=True)
def div(x1, x2):
    out = x1 / x2
    out[~np.isfinite(out)] = np.nan
    return out


@_register("sqrt", (1, 1))
@nb.njit(cache=True, fastmath=True)
def sqrt(x):
    y = np.empty_like(x)
    for i in range(x.size):
        xi = x[i]
        if xi >= 0:
            y[i] = np.sqrt(xi)
        else:
            y[i] = np.sign(xi) * np.sqrt(np.abs(xi))
    return y


@_register("log", (1, 1))
@nb.njit(cache=True, fastmath=True)
def log(x):
    y = np.empty_like(x)
    for i in range(x.size):
        xi = x[i]
        if xi > 0:
            y[i] = np.log(xi)
        else:
            y[i] = np.sign(xi) * np.log(np.abs(xi))
    return y


@_register("abs", (1, 1))
@nb.njit(cache=True, fastmath=True)
def abs_(x):
    return np.abs(x)


@_register("neg", (1, 1))
@nb.njit(cache=True, fastmath=True)
def neg(x):
    return -x


@_register("inv", (1, 1))
@nb.njit(cache=True, fastmath=True)
def inv(x):
    y = 1.0 / x
    y[~np.isfinite(y)] = np.nan
    return y


@_register("max", (2, 2))
@nb.njit(cache=True, fastmath=True)
def max_(x1, x2):
    return np.maximum(x1, x2)


@_register("min", (2, 2))
@nb.njit(cache=True, fastmath=True)
def min_(x1, x2):
    return np.minimum(x1, x2)


@_register("sign", (1, 1))
@nb.njit(cache=True, fastmath=True)
def sign(x):
    return np.sign(x)


@_register("power", (2, 2))
@nb.njit(cache=True, fastmath=True)
def power(x1, x2):
    y = np.power(x1, x2)
    y[~np.isfinite(y)] = np.nan
    return y


# --------------------------------------------------------------------- #
# Rolling helpers – implemented via numpy stride tricks for speed       #
# --------------------------------------------------------------------- #
def _roll_view(a: np.ndarray, window: int):
    """Return 2‑d strided view with trailing 'window' observations."""
    shape = (a.shape[0] - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _rolling_apply(a: np.ndarray, w: int, func) -> np.ndarray:
    if w < 2:
        raise ValueError("window must be >=2")
    out = np.full_like(a, np.nan, dtype=np.float64)
    if a.size < w:
        return out
    view = _roll_view(a, w)
    res = func(view)
    out[w - 1 :] = res
    return out


@_register("delay", (2, 2))
def delay(x, d):
    d = int(d)
    if d < 1:
        raise ValueError("delay must be >=1")
    return np.concatenate((np.full(d, np.nan), x[:-d]))


@_register("delta", (2, 2))
def delta(x, d):
    return x - delay(x, d)


# Rolling sum/mean/min/max/stddev etc. implemented with numpy for clarity
@_register("ts_sum", (2, 2))
def ts_sum(x, d):
    return _rolling_apply(x, int(d), lambda v: np.nansum(v, axis=1))


@_register("ts_mean", (2, 2))
def ts_mean(x, d):
    return _rolling_apply(x, int(d), lambda v: np.nanmean(v, axis=1))


@_register("ts_min", (2, 2))
def ts_min(x, d):
    return _rolling_apply(x, int(d), lambda v: np.nanmin(v, axis=1))


@_register("ts_max", (2, 2))
def ts_max(x, d):
    return _rolling_apply(x, int(d), lambda v: np.nanmax(v, axis=1))


@_register("ts_stddev", (2, 2))
def ts_stddev(x, d):
    return _rolling_apply(x, int(d), lambda v: np.nanstd(v, axis=1, ddof=1))


@_register("ts_zscore", (2, 2))
def ts_zscore(x, d):
    m = ts_mean(x, d)
    s = ts_stddev(x, d)
    return (x - m) / s


@_register("decay_linear", (2, 2))
def decay_linear(x, d):
    d = int(d)
    weights = 2 * np.arange(1, d + 1) / (d * (d + 1))
    return _rolling_apply(x, d, lambda v: np.dot(v, weights))


# --- ranking ----------------------------------------------------------------#
# rank, ts_rank, ts_argmin, ts_argmax, ts_product etc. left as exercise
# They follow same pattern: build view + numpy ops.

# -----------------------------------------------------------------------------#
# Conditional operators – implemented with numpy where
# -----------------------------------------------------------------------------#
@_register("ifcondition_g", (4, 4))
def if_g(c1, c2, x1, x2):
    return np.where(c1 > c2, x1, x2)


@_register("ifcondition_ge", (4, 4))
def if_ge(c1, c2, x1, x2):
    return np.where(c1 >= c2, x1, x2)


@_register("ifcondition_e", (4, 4))
def if_e(c1, c2, x1, x2):
    return np.where(c1 == c2, x1, x2)
