"""FV study pool: model-free cell filters over vericoding, cross-language components, contamination marks.

  .venv/bin/python -m bench.fv.pool        # writes runs/fv/pool.json

Components are a union-find over every record (any qa-issue, any source) linked by (source, source-id) across
languages, by every qa-near-duplicate-group, and by numpy function name (`Np<X>-spec` <-> `<module>_<x>`). A component
is SOLVED if any member has a published file under vericoding's `vericoded/` (runs/fv/vericoded_paths.txt). Cells keep
qa-issue==0 records of non-SOLVED components only. A component's key is its smallest member id.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "runs" / "dafny" / "vericoding.jsonl"
VERICODED = ROOT / "runs" / "fv" / "vericoded_paths.txt"
E4B_LOG = ROOT / "runs" / "dafny" / "valid_tasks.json"
OUT = ROOT / "runs" / "fv" / "pool.json"

CELLS = {  # cell -> (language, sources)
    "L-fv": ("lean", ("fvapps",)),
    "L-apps": ("lean", ("apps",)),
    "L-np": ("lean", ("numpy_triple", "numpy_simple")),
    "D-np": ("dafny", ("numpy_triple", "numpy_simple")),
    "D-apps": ("dafny", ("apps",)),
}
EXCLUDED_SOURCES = ("dafnybench", "humaneval", "verina", "verified_cogen", "clever", "bignum")
E4B_SEED, E4B_N = 20261011, 60  # bench/adapters/vericoding.py build() defaults
# FVAPPS train/0023 -> fvapps_000023, assuming fvapps_N is FVAPPS row N (not verified offline against FVAPPS metadata).
HUMAN_BASELINE = "fvapps_000023"
GUARDED = "Assurance level: guarded_and_plausible"

NUMPY_MODULES = sorted((
    "array_creation", "array_manipulation", "bitwise_operations", "constants", "data_type_routines", "datetime_support",
    "fft", "indexing_slicing", "io_operations", "linalg", "logic_functions", "mathematical_functions", "ndarray",
    "polynomial_chebyshev", "polynomial_hermite", "polynomial_hermite_e", "polynomial_laguerre", "polynomial_legendre",
    "polynomial_polybase", "polynomial_polynomial", "polynomial_polyutils", "random", "set_operations",
    "sorting_searching", "statistics", "strings", "ufunc", "ufuncs"), key=len, reverse=True)
NUMPY_NAMESPACES = {"strings": "strings.", "ufunc": "ufunc.", "ufuncs": "ufunc."}  # np.strings.add is not np.add
NUMPY_ALIASES = {"amax": "max", "amin": "min", "around": "round", "truedivide": "divide", "rowstack": "vstack",
                 "msort": "sort", "in1d": "isin"}
# numpy_simple names whose numpy_triple twin has another name (over-merging is the safe side)
SIMPLE_ALIASES = {"abs": "absolute", "intersect": "intersect1d", "poly": "polyfromroots", "uniqueall": "unique",
                  "center": "strings.center", "isalpha": "strings.isalpha"}

LEAN_CODE = ("vc-preamble", "vc-helpers", "vc-definitions", "vc-theorems", "vc-postamble")
OUTSIDE_HOLES = ("vc-preamble", "vc-helpers", "vc-postamble")  # agent sorry holes live in definitions/theorems
FLOAT = re.compile(r"\bFloat")
SORRY = re.compile(r"\bsorry\b")
AXIOM = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable)\s+)*axiom\b", re.M)
MATHLIB = re.compile(r"^\s*import\s+(?:Mathlib|Batteries)\b", re.M)


def load(path=DATA):
    return [json.loads(line) for line in open(path)]


def numpy_name(r) -> str:
    """Normalized numpy function name: 'NpCumSum-spec', 'mathematical_functions_cumsum' -> 'cumsum'; 'strings_add' ->
    'strings.add'."""
    sid = r["source-id"]
    if r["source"] == "numpy_simple":
        name = sid.removeprefix("Np").removesuffix("-spec").lower()
        return SIMPLE_ALIASES.get(name, NUMPY_ALIASES.get(name, name))
    mod = next(m for m in NUMPY_MODULES if sid.startswith(m + "_"))
    name = sid[len(mod) + 1:].replace("_", "").lower()
    return NUMPY_NAMESPACES.get(mod, "") + NUMPY_ALIASES.get(name, name)


def components(rows) -> dict[str, str]:
    """id -> component key (smallest member id) via union-find over source-id, near-dup group and numpy name."""
    parent = {r["id"]: r["id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = defaultdict(list)
    for r in rows:
        links["sid", r["source"], r["source-id"]].append(r["id"])
        if r.get("qa-near-duplicate-group"):
            links["dup", r["qa-near-duplicate-group"]].append(r["id"])
        if r["source"] in ("numpy_triple", "numpy_simple"):
            links["np", numpy_name(r)].append(r["id"])
    for ids in links.values():
        root = find(ids[0])
        for i in ids[1:]:
            parent[find(i)] = root
    members = defaultdict(list)
    for i in parent:
        members[find(i)].append(i)
    return {i: min(m) for m in members.values() for i in m}


def vericoded_ids(path=VERICODED) -> set[str]:
    """Task ids with a published verified solution (DA0000_vericoded.dfy, lean/apps/<model>/LA0003.lean, ...)."""
    ids = set()
    for line in path.read_text().split():
        m = re.match(r"[A-Z]{2}\d{4}", line.rsplit("/", 1)[-1])
        if m:
            ids.add(m.group(0))
    return ids


def e4b_ids(rows) -> list[str]:
    """The 60 E4b tasks: bench/adapters/vericoding.py's seeded pick, checked against the logged valid/invalid ids."""
    pool = [r for r in rows if r["language"] == "dafny" and str(r.get("qa-issue")) == "0" and r["source"] == "apps"]
    pick = {r["id"] for r in random.Random(E4B_SEED).sample(pool, E4B_N)}
    log = json.loads(E4B_LOG.read_text())
    assert pick == {i.upper() for i in log["valid"] + log["invalid"]}, "E4b pick does not reproduce"
    return sorted(pick)


def reasons(cell, r, solved, e4b) -> list[str]:
    """Every exclusion reason that applies to record r as a candidate for cell (empty list = keep)."""
    if str(r.get("qa-issue")) != "0":
        return ["qa_issue"]  # counted alone, so the other counts are over qa-issue==0 records
    out = []
    if solved:
        out.append("solved_component")
    if r["language"] == "lean":
        code = [r.get(f) or "" for f in LEAN_CODE]
        if any(MATHLIB.search(c) for c in code):
            out.append("mathlib_import")
        if cell == "L-fv":
            if GUARDED not in (r.get("vc-postamble") or ""):
                out.append("not_guarded_and_plausible")
            if r["source-id"] == HUMAN_BASELINE:
                out.append("human_baseline")
        else:
            if AXIOM.search(r.get("vc-preamble") or ""):
                out.append("preamble_axiom")
            if any(SORRY.search(r.get(f) or "") for f in OUTSIDE_HOLES):
                out.append("extra_sorry")
            if any(FLOAT.search(c) for c in code + [r.get("vc-description") or ""]):
                out.append("float")
    if cell == "D-apps" and r["id"] in e4b:
        out.append("e4b_task")
    return out


def build(rows, solved_ids) -> dict:
    comp = components(rows)
    solved = {comp[i] for i in solved_ids if i in comp}
    e4b = e4b_ids(rows)
    cells, counts, excluded = {}, {}, {}
    for cell, (lang, sources) in CELLS.items():
        cand = [r for r in rows if r["language"] == lang and r["source"] in sources]
        why = {r["id"]: reasons(cell, r, comp[r["id"]] in solved, e4b) for r in cand}
        cells[cell] = sorted(i for i, w in why.items() if not w)
        counts[cell] = {"candidates": len(cand), "tasks": len(cells[cell]),
                        "components": len({comp[i] for i in cells[cell]})}
        excluded[cell] = dict(sorted(Counter(x for w in why.values() for x in w).items()))
    keep = {i for ids in cells.values() for i in ids} | set(e4b)
    return {
        "inputs": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (DATA, VERICODED, E4B_LOG)},
        "cells": cells,
        "counts": counts,
        "excluded": excluded,
        "excluded_sources": dict(sorted(Counter(r["source"] for r in rows if r["source"] in EXCLUDED_SOURCES).items())),
        "components_total": len(set(comp.values())),
        "components_solved": len(solved),
        "component": {i: comp[i] for i in sorted(keep)},
        "e4b": {"ids": e4b, "components": sorted({comp[i] for i in e4b})},
        "notes": [
            "Components span all 12,504 records (any qa-issue); cells keep qa-issue==0 only.",
            "Verus not attempted (scoped out 2026-10-11); excluded sources are never cell candidates.",
            f"FVAPPS human baseline train/0023 mapped to {HUMAN_BASELINE} assuming fvapps_N is FVAPPS row N (not "
            "verified against FVAPPS metadata offline); that record is 'unguarded', so L-fv excludes it either way.",
            "excluded counts are per reason over qa-issue==0 candidates; one task can have several reasons.",
        ],
    }


def main():
    pool = build(load(), vericoded_ids())
    OUT.write_text(json.dumps(pool, indent=1) + "\n")
    for cell, c in pool["counts"].items():
        print(f"{cell:7s} {c['tasks']:5d} tasks {c['components']:5d} components  of {c['candidates']:5d}  "
              f"excluded {pool['excluded'][cell]}")
    print(f"components {pool['components_total']} (SOLVED {pool['components_solved']}) -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
