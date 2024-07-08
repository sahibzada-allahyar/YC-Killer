"""
Thin re‑export layer so downstream code can access transforms and arities
without importing numba jit implementations directly.
"""
from typing import Callable, Dict, Tuple

from .transforms import registry as _impls, arity as _arity

__all__ = ["get", "arity", "list_transforms"]

_Registry = Dict[str, Callable]


def get(name: str) -> Callable:
    try:
        return _impls[name]
    except KeyError as e:
        raise ValueError(f"Unknown transform '{name}'") from e


def arity(name: str) -> Tuple[int, int]:
    return _arity[name]


def list_transforms() -> list[str]:
    return sorted(_impls.keys())
