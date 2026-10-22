"""Isolated subprocess that loads ONE spec module and serves a single JSON command on stdin.

Usage: python -m millennium_harness._specworker SPEC.py   (stdin: {"cmd": ..., ...}; stdout: JSON)
Spec code is LLM-written, so it always runs in its own process with per-call alarms.
"""
from __future__ import annotations

import importlib.util
import json
import signal
import sys
import time


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def _call(fn, *args, timeout=2.0):
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn(*args), None
    except _Timeout:
        return None, "timeout"
    except BaseException as e:  # noqa: BLE001 — spec code can raise anything, incl. SystemExit
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _load(path):
    spec = importlib.util.spec_from_file_location("sl_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample(mod, n, seed, timeout):
    from hypothesis import HealthCheck, Phase, given, settings
    from hypothesis import seed as hseed

    out, deadline = [], time.time() + timeout

    @hseed(seed)
    @settings(max_examples=n, database=None, deadline=None, phases=[Phase.generate],
              suppress_health_check=list(HealthCheck))
    @given(mod.gen())
    def t(x):
        if time.time() > deadline:
            return
        if isinstance(x, str):
            out.append(x)

    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout + 2)
        t()
    except _Timeout:
        pass
    except BaseException as e:  # noqa: BLE001
        if not out:
            return {"inputs": [], "error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return {"inputs": uniq}


def main():
    sys.setrecursionlimit(20000)
    req = json.loads(sys.stdin.read())
    try:
        mod = _load(sys.argv[1])
    except BaseException as e:  # noqa: BLE001
        print(json.dumps({"load_error": f"{type(e).__name__}: {str(e)[:300]}"}))
        return
    cmd = req["cmd"]
    if cmd == "info":
        res = {"applicable": bool(getattr(mod, "APPLICABLE", False)),
               "has_gen": hasattr(mod, "gen"), "has_ref": hasattr(mod, "ref"),
               "has_validate": hasattr(mod, "validate"), "has_same": hasattr(mod, "same")}
    elif cmd == "sample":
        res = _sample(mod, req.get("n", 120), req.get("seed", 0), req.get("timeout", 8.0))
    elif cmd == "edge":
        f = getattr(mod, "edge_cases", None)
        xs, err = _call(f, timeout=5.0) if f else ([], "missing")
        res = {"inputs": [x for x in (xs or []) if isinstance(x, str)][:60], "error": err}
    elif cmd == "validate":
        res = {"ok": [bool(_call(mod.validate, x, timeout=1.0)[0]) for x in req["inputs"]]}
    elif cmd == "ref":
        outs, errs = [], []
        for x in req["inputs"]:
            o, e = _call(mod.ref, x, timeout=req.get("timeout", 2.0))
            outs.append(o if isinstance(o, str) else (None if o is None else str(o)))
            errs.append(e)
        res = {"out": outs, "err": errs}
    elif cmd == "same":
        if not hasattr(mod, "same"):
            res = {"ok": None}
        else:
            res = {"ok": [bool(_call(mod.same, i, e, g, timeout=2.0)[0]) for i, e, g in req["triples"]]}
    else:
        res = {"error": f"unknown cmd {cmd}"}
    sys.stdout.write(json.dumps(res))


if __name__ == "__main__":
    main()
