"""
Prefix‑notation expression → compiled callable.

* Grammar via Lark
* Validation against transform registry and datatypes
* Compilation to a fast NumPy function that expects a dict(col_name -> np.ndarray)
"""
from __future__ import annotations

import textwrap
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from lark import Lark, Transformer, v_args

from .datatypes import DT
from .registry import arity, get as transform_get

_GRAMMAR = textwrap.dedent(
    r"""
    ?start: expr
    ?expr: SYMBOL                        -> column
         | NUMBER                        -> number
         | "(" SYMBOL expr+ ")"          -> call
         | SYMBOL expr+                  -> call   // prefix but without ()
    SYMBOL: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /-?\d+(\.\d+)?/
    %ignore /\s+/
"""
)

_parser = Lark(_GRAMMAR, parser="lalr", maybe_placeholders=False)


@v_args(inline=True)
class _BuildAST(Transformer):
    def column(self, tok):
        return ("col", tok.value)

    def number(self, tok):
        return ("num", float(tok.value))

    def call(self, name_tok, *args):
        return ("call", name_tok.value, list(args))


def _validate(node) -> None:
    typ = node[0]
    if typ == "call":
        _, name, args = node
        if name not in transform_get.__globals__["registry"]:
            raise ValueError(f"Unknown transform '{name}'")
        min_n, max_n = arity(name)
        if not (min_n <= len(args) <= max_n):
            raise ValueError(f"{name} expects {min_n}–{max_n} args, got {len(args)}")
        for a in args:
            _validate(a)
    elif typ == "col":
        _, col = node
        if col not in DT.list():
            raise ValueError(f"Unknown column '{col}'")
    # nums always fine


def _compile(node) -> str:
    typ = node[0]
    if typ == "num":
        return repr(node[1])
    if typ == "col":
        return f'd["{node[1]}"]'
    # call
    _, name, args = node
    py_args = ", ".join(_compile(a) for a in args)
    return f"_t('{name}')({py_args})"


@lru_cache(maxsize=512)
def compile_expr(expr: str):
    """Return a python function f(dict[str, ndarray]) -> ndarray."""
    tree = _parser.parse(expr)
    ast = _BuildAST().transform(tree)
    _validate(ast)
    body = _compile(ast)
    src = (
        "def _f(d, _t):\n"
        "    import numpy as np  # local alias\n"
        f"    return {body}\n"
    )
    local: Dict[str, object] = {}
    exec(src, {}, local)
    return local["_f"]


def evaluate(expr: str, data: Dict[str, np.ndarray]) -> np.ndarray:
    return compile_expr(expr)(data, transform_get)
