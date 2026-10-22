"""Run a candidate solution on inputs, the same way LiveCodeBench's grader does (same import prelude/arg parsing)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

PRELUDE = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import List, Dict, Set, Tuple, Optional, Union, Any
import string, re, datetime, collections, heapq, bisect, copy, math, random, statistics, itertools, functools, operator, io, sys, json
sys.setrecursionlimit(50000)
"""

_FUNC_HARNESS = r'''
import json, signal, sys, io, importlib.util
_real = sys.stdout; sys.stdout = io.StringIO()
class _T(Exception): pass
def _a(s, f): raise _T()
def _parse(inp):
    out = []
    for part in inp.split("\n"):
        if part.strip():
            try: out.append(json.loads(part))
            except json.JSONDecodeError: out.append(part)
    return out
spec = importlib.util.spec_from_file_location("solution", sys.argv[1])
mod = importlib.util.module_from_spec(spec); sys.modules["solution"] = mod
req = json.loads(sys.stdin.read())
res = []
try:
    spec.loader.exec_module(mod)
    inst = mod.Solution()
    fn = getattr(inst, req["func"], None) or getattr(inst, [m for m in dir(inst) if not m.startswith("_")][0])
except BaseException as e:
    _real.write(json.dumps({"fatal": f"{type(e).__name__}: {str(e)[:300]}"})); sys.exit(0)
signal.signal(signal.SIGALRM, _a)
for inp in req["inputs"]:
    args = _parse(inp)
    signal.setitimer(signal.ITIMER_REAL, req["timeout"])
    try:
        try: r = fn(*args)
        except TypeError as e:
            if "takes" in str(e) and "arguments" in str(e): r = fn(args)
            else: raise
        if isinstance(r, tuple): r = list(r)
        try: s = json.dumps(r)
        except TypeError: s = json.dumps(repr(r))
        res.append({"out": s, "err": None})
    except _T:
        res.append({"out": None, "err": "timeout"})
    except BaseException as e:
        res.append({"out": None, "err": f"{type(e).__name__}: {str(e)[:200]}"})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
_real.write(json.dumps({"res": res}))
'''


def prepare(code: str, workdir: str) -> str:
    """Write candidate code with the LCB import prelude; return its path."""
    os.makedirs(workdir, exist_ok=True)
    path = os.path.abspath(os.path.join(workdir, "solution.py"))
    with open(path, "w") as f:
        f.write(PRELUDE + "\n" + code)
    return path


def run_stdin(path: str, inputs: list[str], timeout: float = 2.0, workers: int = 4) -> list[tuple[str | None, str | None]]:
    def one(inp):
        try:
            p = subprocess.run([sys.executable, path], input=inp, capture_output=True, text=True,
                               timeout=timeout, cwd=os.path.dirname(path))
            if p.returncode != 0:
                return None, f"exit {p.returncode}: {p.stderr.strip()[-300:]}"
            return p.stdout, None
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, inputs))


def run_functional(path: str, func: str, inputs: list[str], timeout: float = 2.0) -> list[tuple[str | None, str | None]]:
    if not inputs:
        return []
    with tempfile.NamedTemporaryFile("w", suffix="_harness.py", delete=False) as h:
        h.write(_FUNC_HARNESS)
        hpath = h.name
    try:
        p = subprocess.run([sys.executable, hpath, path], input=json.dumps(
            {"func": func, "inputs": inputs, "timeout": timeout}), capture_output=True, text=True,
            timeout=timeout * len(inputs) + 20, cwd=os.path.dirname(path))
        d = json.loads(p.stdout or "{}")
    except subprocess.TimeoutExpired:
        return [(None, "timeout")] * len(inputs)
    except json.JSONDecodeError:
        return [(None, f"harness crashed: {p.stderr.strip()[-300:]}")] * len(inputs)
    finally:
        os.unlink(hpath)
    if "fatal" in d:
        return [(None, d["fatal"])] * len(inputs)
    res = d.get("res", [])
    res += [{"out": None, "err": "harness died"}] * (len(inputs) - len(res))
    return [(r["out"], r["err"]) for r in res]


def run(kind: str, path: str, inputs: list[str], func: str | None = None, timeout: float = 2.0, workers: int = 4):
    if kind == "functional":
        return run_functional(path, func, inputs, timeout)
    return run_stdin(path, inputs, timeout, workers)
