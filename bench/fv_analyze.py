"""Frozen confirmatory analysis for Millennium Harness FV v1.0 (docs/research/win-plan-synthesis.md §6). Offline,
model-free.

  .venv/bin/python bench/fv_analyze.py --oc <job> [--cc <job>] --family H1,H2[,H3] [--json out.json]
                                       [--curated runs/fv/curated.json | none] [--dev]

<job> is runs/fv/grades/<job>.jsonl (or a path to one), written by bench/fv/regrade.py. flatten() maps its nested rows
onto this flat schema (flat rows are read as they are):
  id, cell, part ('test' | 'dev'), stop_reason (end_turn | length | timeout | error | ...), b_fired, infra_fail,
  {A,B,C}_strict, {A,B,C}_lenient  fresh-container status of the solution at S0 / after B / after C,
  sha_{A,B,C}                      sha256 of the solution file in that snapshot,
  cost_{A,B,C}, secs_{A,B,C}      A = spend and time up to S0; B and C = the EXTRA spend and time of that branch.
Claude Code jobs have no C keys. Read when present: infra and warnings (lists; warnings never exclude), exception
(Harbor), egress (the K2 probe output),
external_fetch (step-19 transcript URL scan), c_rounds, c_restore_ok, gate (in-sandbox statuses; gate_B = the last,
S8), started_at, snap_strict (strict status of every snapshot, S5), claims_done (S4), memo (S6), tok_B, tok_C (S1).

Usable trials. Every exclusion applies to all branches of the trial together. A trial is unusable if
  - regrade flagged it (infra non-empty), or Harbor recorded an agent-phase exception (agent timeout, cancellation,
    ...; verifier-phase exceptions come after the snapshots and do not count, as in regrade);
  - a branch status is missing or outside PROVED / UNPROVED / CHEAT_REJECTED / SPEC_TAMPERED (CHECK_ERROR = a grader
    crash or a task whose frozen parts do not resolve; branches: A, B, and C in an FVBranch job);
  - an in-sandbox gate status is outside the rung's five (GATE_ERROR, NO_TASK: B never had a working gate);
  - FVBranch: the restore of S0 before C failed (c_restore_ok false), or C ran no round (c_rounds 0);
  - K2: the egress probe reached a non-model host (any HTTP code but 000) or external_fetch is set.
These mirror bench/fv/regrade.py's `infra` rules (VALID, GATE_OK, EGRESS and VERIFIER_EXC are checked against its
source by the tests), so a current grades file is excluded exactly by `infra`; the checks here also cover older or
partial files. Only c_rounds 0 and external_fetch go beyond regrade (neither occurs in its current flow).
An id with no usable trial is excluded; otherwise its earliest usable trial by started_at is used (file order only
when a start time is missing; reported). Every trial must be part 'test' (--dev: part 'dev', a labelled dry run).
With a curation file (runs/fv/curated.json by default), ids whose status is not OK (TRIVIAL_THEOREMS, VACUOUS,
BROKEN, ...) are dropped and counted; in a TEST analysis, so are ids the file does not cover.

Per task, under grader g in {strict, lenient}:
  d_nat = +1 if A fails, B is PROVED, the S0 stop was end_turn and B's file sha differs from S0's;
          -1 if A is PROVED and B is not; otherwise 0.
  e     = [B PROVED] - [C PROVED], all stops.
Family (Holm, one-sided alpha 0.025; its size is fixed at the tag via --family):
  H1 E[d_nat] > 0.03 (OpenCode)    H2 E[e] > -0.05 (OpenCode)    H3 E[d_nat] > 0.03 (Claude Code)
p = the share of 10k task-bootstrap means at or below the null (tasks sorted by id, so file order does not matter).
For H1 and H3, when no d = -1 occurs, the exact binomial P(Bin(N, 0.03) >= #wins) is also computed and the larger p
is used. H1 and H3 must hold under both graders: Holm sees max(p_strict, p_lenient) for them (an intersection-union
test) and p_strict for H2, and a hypothesis is claimed exactly when Holm rejects it. Secondaries S1-S8 are estimates
with 95% task-bootstrap CIs only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRADES = ROOT / "runs" / "fv" / "grades"
CURATED = ROOT / "runs" / "fv" / "curated.json"
ALPHA = 0.025
NULL = {"H1": 0.03, "H2": -0.05, "H3": 0.03}
N_BOOT = 10_000
SEED = 20261012  # the split.json seed; never derived from a commit hash
GRADERS = ("strict", "lenient")
LANG = {"L": "lean", "D": "dafny", "V": "verus"}
NAN = float("nan")
VALID = ("PROVED", "UNPROVED", "CHEAT_REJECTED", "SPEC_TAMPERED")  # grader statuses a usable trial may have
GATE_OK = (*VALID, "CHECK_ERROR")  # the rung's statuses; the in-sandbox gate also logs GATE_ERROR and NO_TASK
EGRESS = re.compile(r"^\s*(\S+)\s+(\d{3})\s*$")  # a probe line "<host> <http code>"; 000 = no connection
VERIFIER_EXC = {"VerifierTimeoutError", "RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError",
                "DownloadVerifierDirError", "AddTestsDirError"}  # Harbor's verifier phase: after the agent finished
K2_MAX, K11_MAX = 0.02, 0.10


# ---------------------------------------------------------------- per-task outcomes

def proved(r: dict, arm: str, g: str = "strict") -> int:
    return int(r.get(f"{arm}_{g}") == "PROVED")


def d_nat(r: dict, g: str = "strict") -> int:
    a, b = proved(r, "A", g), proved(r, "B", g)
    if not a and b and r.get("stop_reason") == "end_turn" and r.get("sha_B") and r["sha_B"] != r.get("sha_A"):
        return 1
    return -1 if a and not b else 0


def e_bc(r: dict, g: str = "strict") -> int:
    return proved(r, "B", g) - proved(r, "C", g)


def lang(r: dict) -> str:
    return LANG.get(str(r.get("cell") or "?")[:1], "?")


def num(r: dict, k: str) -> float:
    return float(r.get(k) or 0.0)


# ---------------------------------------------------------------- loading

KEEP = ("cell", "part", "agent", "trial", "infra", "warnings", "exception", "egress", "external_fetch", "c_rounds",
        "c_restore_ok", "gate", "started_at", "claims_done", "memo")


def flatten(t: dict) -> dict:
    """The flat schema above, from either a flat row or bench/fv/regrade.py's nested row (task, s0_stop, A/B/C =
    {sha, strict, lenient} or null, usage/secs per branch, snaps = {label: {strict, ...}}, infra = list of problems,
    gate = in-sandbox statuses, plus the fields in KEEP, which are carried through unchanged)."""
    if "id" in t:
        return t
    r = {"id": t.get("task"), "stop_reason": t.get("s0_stop"), "b_fired": bool(t.get("b_fired")),
         "infra_fail": bool(t.get("infra")), **{k: t[k] for k in KEEP if k in t}}
    if t.get("gate"):
        r["gate_B"] = t["gate"][-1]  # the check of B's final file
    if isinstance(t.get("snaps"), dict):
        r["snap_strict"] = {k: (v or {}).get("strict") for k, v in t["snaps"].items()}
    usage, secs = t.get("usage") or {}, t.get("secs") or {}
    for x in "ABC":
        b = t.get(x)
        if not isinstance(b, dict):
            continue
        u = usage.get(x) or {}
        r.update({f"{x}_{g}": b.get(g) for g in GRADERS})
        r.update({f"sha_{x}": b.get("sha"), f"cost_{x}": u.get("cost"), f"secs_{x}": secs.get(x),
                  f"tok_{x}": sum(u.get(k) or 0 for k in ("in", "out", "reasoning", "cache_read", "cache_write"))})
    return r


def egress_hits(t: dict) -> list[str]:
    """K2: probe lines showing a non-model host answered (any HTTP code but 000), and the transcript-scan flag."""
    e = t.get("egress") or ""
    lines = ([f"{k} {v}" for k, v in e.items()] if isinstance(e, dict) else [str(x) for x in e]
             if isinstance(e, (list, tuple)) else str(e).splitlines())
    hits = [f"{m.group(1)} {m.group(2)}" for line in lines if (m := EGRESS.match(line)) and m.group(2) != "000"]
    if t.get("external_fetch"):
        hits.append(f"external fetch: {str(t['external_fetch'])[:200]}")
    return hits


def problems(t: dict, has_c: bool) -> list[str]:
    """Why a flattened trial is unusable; [] if it is usable (the rules in the module docstring)."""
    p = []
    if t.get("infra_fail") or t.get("infra"):
        p.append("infra")
    if t.get("exception") and t["exception"] not in VERIFIER_EXC:
        p.append("Harbor exception")
    st = [t.get(f"{x}_{g}") for x in ("ABC" if has_c else "AB") for g in GRADERS]
    if None in st:
        p.append("missing status")
    p += sorted({f"status {s}" for s in st if s is not None and s not in VALID})
    gates = t.get("gate") or ([t["gate_B"]] if t.get("gate_B") else [])
    if any(s not in GATE_OK for s in gates):
        p.append("gate error")
    if has_c and t.get("c_restore_ok") is False:
        p.append("C restore failed")
    if has_c and t.get("c_rounds") == 0:
        p.append("C ran no round")
    if egress_hits(t):
        p.append("egress")
    return p


def _start(t: dict) -> float | None:
    try:
        return datetime.fromisoformat(str(t["started_at"]).replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, ValueError):
        return None


def curation(path) -> dict | None:
    """{task id: curation status} from bench/fv/curate.py's output, or None without a file."""
    if path is None or not Path(path).is_file():
        return None
    return {i: (v or {}).get("status") for i, v in (json.loads(Path(path).read_text()).get("tasks") or {}).items()}


def dedupe(trials: list[dict], curated: dict | None = None, need_curated: bool = False) -> tuple[list[dict], dict]:
    """One row per id, sorted by id: its earliest usable trial. `curated` ({id: status}) drops ids whose status is not
    OK, and with need_curated also ids it does not list; these are not counted as exclusions."""
    trials = [flatten(t) for t in trials]
    has_c = any("C_strict" in t or t.get("agent") == "FVBranch" for t in trials)
    by = defaultdict(list)
    for t in trials:
        by[t["id"]].append(t)
    rows, excluded, dup, unordered, egress, dropped, why = [], [], [], [], [], {}, Counter()
    check_errors = Counter(f"{x}_{g}" for t in trials for x in "ABC" for g in GRADERS
                           if t.get(f"{x}_{g}") == "CHECK_ERROR")
    def tally(k):
        return Counter(re.sub(r"\d+", "#", str(m))[:100] for t in trials if isinstance(t.get(k), (list, tuple))
                       for m in t[k])
    infra_msgs, warn_msgs = tally("infra"), tally("warnings")
    for i in sorted(by, key=str):
        if curated is not None:
            cs = curated.get(i, "not curated" if need_curated else "OK")
            if cs != "OK":
                dropped[i] = cs
                continue
        ps = [(t, problems(t, has_c)) for t in by[i]]
        for _, p in ps:
            why.update(p)
        if any("egress" in p for _, p in ps):
            egress.append(i)
        ok = [t for t, p in ps if not p]
        if not ok:
            excluded.append(i)
            continue
        if len(ok) > 1:
            dup.append(i)
            starts = [_start(t) for t in ok]
            if None in starts:
                unordered.append(i)
            else:
                ok = [t for _, _, t in sorted(zip(starts, range(len(ok)), ok))]
        rows.append(ok[0])
    n = max(1, len(by) - len(dropped))
    return rows, {"trials": len(trials), "ids": len(by) - len(dropped), "excluded": excluded,
                  "reasons": dict(why), "duplicates": dup, "unordered": unordered, "has_c": has_c,
                  "infra_rate": len(excluded) / n, "check_errors": dict(check_errors),
                  "infra_msgs": dict(infra_msgs.most_common(8)), "warnings": dict(warn_msgs.most_common(8)),
                  "egress": egress,
                  "egress_rate": len(egress) / n, "curation": dict(Counter(dropped.values())),
                  "curation_ids": sorted(dropped, key=str)}


def load(job: str, part: str = "test", curated=CURATED) -> tuple[list[dict], dict]:
    """A grades file; every trial must be part `part` (TEST is never analysed by accident as DEV, nor DEV as TEST)."""
    p = Path(job) if job.endswith(".jsonl") else GRADES / f"{job}.jsonl"
    with open(p) as f:
        trials = [flatten(json.loads(line)) for line in f if line.strip()]
    parts = Counter(str(t.get("part")) for t in trials)
    if set(parts) != {part}:
        raise SystemExit(f"{p}: every trial must be part {part!r}; found {dict(parts)}")
    cur = curation(curated)
    if cur is None and part == "test":
        raise SystemExit(f"a TEST analysis needs the curation file (not found: {curated})")
    rows, info = dedupe(trials, cur, need_curated=part == "test")
    return rows, {"path": str(p), "part": part, "curated": str(curated) if cur is not None else None, **info}


# ---------------------------------------------------------------- statistics

def boot_weights(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(n_boot, n) resample counts: row b says how often each task is drawn in task-bootstrap resample b (n tasks
    drawn with replacement)."""
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n)) + n * np.arange(n_boot)[:, None]
    return np.bincount(idx.ravel(), minlength=n_boot * n).reshape(n_boot, n).astype(float)


def ci(x, W: np.ndarray, den=None) -> dict:
    """sum(x) / sum(den) (den = 1s gives the mean) with a 95% percentile CI over the resamples in W."""
    x = np.asarray(x, float)
    den = np.ones_like(x) if den is None else np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = x.sum() / den.sum() if den.sum() else NAN
        bs = (W @ x) / (W @ den)
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs.size else (NAN, NAN)
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "n": int(den.sum())}


def binom_sf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) >= k), exact."""
    if k <= 0:
        return 1.0
    lp, lq, c = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    return min(1.0, sum(math.exp(c - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq)
                        for j in range(k, n + 1)))


def clopper_pearson(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    a = (1 - conf) / 2

    def solve(k, target):  # p with P(Bin(n, p) >= k) = target; increasing in p
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if binom_sf(k, n, mid) < target else (lo, mid)
        return (lo + hi) / 2
    if n == 0:
        return 0.0, 1.0
    return (0.0 if x == 0 else solve(x, a)), (1.0 if x >= n else solve(x + 1, 1 - a))


def one_sided(x, null: float, W: np.ndarray, exact: bool = False) -> dict:
    """Test E[x] > null. p_boot = share of bootstrap means <= null. With exact=True and no x = -1 (x in {0, 1}), also
    the exact binomial p vs Bin(N, null); the larger p is used."""
    x = np.asarray(x, float)
    n = len(x)
    m = W @ x / n
    wins, losses = int((x > 0).sum()), int((x < 0).sum())
    pb = float(np.mean(m <= null + 1e-12))
    pbin = binom_sf(wins, n, null) if exact and losses == 0 else None
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"n": n, "est": float(x.mean()), "lo": float(lo), "hi": float(hi), "wins": wins, "losses": losses,
            "null": null, "p_boot": pb, "p_binom": pbin, "p": max(pb, pbin or 0.0)}


def holm(ps: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step-down: the i-th smallest p (0-based) is compared with alpha / (m - i) until the first failure."""
    m, out, going, adj = len(ps), {}, True, 0.0
    for i, k in enumerate(sorted(ps, key=lambda k: (ps[k], k))):
        crit = alpha / (m - i)
        adj = max(adj, min(1.0, (m - i) * ps[k]))
        going = going and ps[k] <= crit
        out[k] = {"p": ps[k], "crit": crit, "p_holm": adj, "reject": going}
    return out


# ---------------------------------------------------------------- per-job analysis

def core(rows: list[dict], W: np.ndarray, g: str, has_c: bool) -> dict:
    A = [proved(r, "A", g) for r in rows]
    B = [proved(r, "B", g) for r in rows]
    fired = [bool(r.get("b_fired")) for r in rows]
    d = [d_nat(r, g) for r in rows]
    out = {"n": len(rows), "A": ci(A, W), "B": ci(B, W), "d_nat": ci(d, W),
           "dB_all": ci(np.subtract(B, A), W), "harm_B": ci([a and not b for a, b in zip(A, B)], W),
           "fired": sum(fired), "W_nat": d.count(1), "L": d.count(-1),
           "W_all": sum(not a and b for a, b in zip(A, B)),
           "W_silent": sum(not a and b and not f for a, b, f in zip(A, B, fired))}  # must be 0: B = S0 when silent
    if has_c:
        C = [proved(r, "C", g) for r in rows]
        e = np.subtract(B, C)
        out.update(C=ci(C, W), e=ci(e, W), dC=ci(np.subtract(C, A), W),
                   harm_C=ci([a and not c for a, c in zip(A, C)], W), e_fired=ci(e * fired, W, fired))
    return out


def cost(rows: list[dict], W: np.ndarray, has_c: bool) -> dict:
    cA, cB = [num(r, "cost_A") for r in rows], [num(r, "cost_B") for r in rows]
    tA, tB = [num(r, "secs_A") for r in rows], [num(r, "secs_B") for r in rows]
    A, B = [proved(r, "A") for r in rows], [proved(r, "B") for r in rows]
    fired = [bool(r.get("b_fired")) for r in rows]

    def pct(t):
        return {"p50": float(np.percentile(t, 50)), "p90": float(np.percentile(t, 90))}
    out = {"extra_B": ci(cB, W), "extra_B_fired": ci(np.multiply(cB, fired), W, fired),
           "usd_per_verified_A": ci(cA, W, A), "usd_per_verified_B": ci(np.add(cA, cB), W, B),
           "secs_A": pct(tA), "secs_B": pct(np.add(tA, tB))}
    if has_c:
        cC, tC, C = [num(r, "cost_C") for r in rows], [num(r, "secs_C") for r in rows], [proved(r, "C") for r in rows]
        out.update(extra_C=ci(cC, W), ratio_C_over_B=ci(cC, W, cB), usd_per_verified_C=ci(np.add(cA, cC), W, C),
                   secs_C=pct(np.add(tA, tC)))
        if any("tok_B" in r for r in rows):
            out["tok_ratio_C_over_B"] = ci([num(r, "tok_C") for r in rows], W, [num(r, "tok_B") for r in rows])
    return out


def strata(rows: list[dict], key, has_c: bool, n_boot: int, seed: int) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return {k: core(v, boot_weights(len(v), n_boot, seed), "strict", has_c) for k, v in sorted(groups.items())}


def snap_statuses(r: dict, x: str) -> set:
    """Strict statuses of every snapshot branch x produced (A: S0; B: B1, B2; C: C1, C2), from regrade's snaps; the
    final status for a flat row without them."""
    sn = r.get("snap_strict")
    if not isinstance(sn, dict):
        return {r.get(f"{x}_strict")} - {None}
    return {s for k, s in sn.items() if s and str(k)[:1] == "SBC"["ABC".index(x)]}


def analyze_job(rows: list[dict], has_c: bool, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    if not rows:
        raise SystemExit("no usable trials")
    rows = sorted(rows, key=lambda r: str(r["id"]))  # bootstrap columns by id, never by file order
    W = boot_weights(len(rows), n_boot, seed)
    out = {g: core(rows, W, g, has_c) for g in GRADERS}  # S1 and S3 (lenient)
    out["gain"] = {g: one_sided([d_nat(r, g) for r in rows], NULL["H1"], W, exact=True) for g in GRADERS}
    if has_c:
        out["noninf"] = {g: one_sided([e_bc(r, g) for r in rows], NULL["H2"], W) for g in GRADERS}
    out["cost"] = cost(rows, W, has_c)
    out["S2"] = {"cell": strata(rows, lambda r: r.get("cell") or "?", has_c, n_boot, seed),
                 "verifier": strata(rows, lang, has_c, n_boot, seed)}
    if any("claims_done" in r for r in rows):  # S4 false-done census
        fd = [not proved(r, "A") and bool(r.get("claims_done")) for r in rows]
        out["S4"] = {"all": ci(fd, W), "of_A_fail": ci(fd, W, [not proved(r, "A") for r in rows])}
    br = "ABC" if has_c else "AB"
    snap = {x: [snap_statuses(r, x) for r in rows] for x in br}  # S5: attempts in any snapshot, not only the last
    out["S5"] = {x: {"final": dict(Counter(r[f"{x}_strict"] for r in rows)),
                     "any_snapshot": dict(Counter(s for ss in snap[x] for s in ss))} for x in br}
    out["S5_examples"] = {x: {s: [r["id"] for r, ss in zip(rows, snap[x]) if s in ss][:5]
                              for s in ("CHEAT_REJECTED", "SPEC_TAMPERED")} for x in br}
    if any("memo" in r for r in rows):
        out["S6"] = strata(rows, lambda r: str(r.get("memo")), has_c, n_boot, seed)
    out["S7"] = strata(rows, lambda r: r.get("stop_reason") or "none", has_c, n_boot, seed)
    if any("gate_B" in r for r in rows):  # S8 receipt soundness: gate PROVED but the fresh strict grader disagrees
        gp = [r for r in rows if r.get("gate_B") == "PROVED"]
        bad = [r["id"] for r in gp if not proved(r, "B")]
        out["S8"] = {"gate_proved": len(gp), "incidents": len(bad), "ids": bad,
                     "cp95": clopper_pearson(len(bad), len(gp))}
    return out


def family_p(h: str, t: dict) -> float:
    """The p Holm sees. H1 and H3 must hold under both graders: max(p_strict, p_lenient), an intersection-union
    test, so Holm never steps down past an H1/H3 that is not claimed. H2: p_strict."""
    return max(t["strict"]["p"], t["lenient"]["p"]) if h in ("H1", "H3") else t["strict"]["p"]


def verdict(tests: dict, family: list[str]) -> dict:
    """Holm over the fixed family on family_p; a hypothesis is claimed exactly when Holm rejects it."""
    v = holm({h: family_p(h, tests[h]) for h in family})
    for h in family:
        v[h].update(p_strict=tests[h]["strict"]["p"], claim=v[h]["reject"])
        if h in ("H1", "H3"):
            v[h]["p_lenient"] = tests[h]["lenient"]["p"]
    return v


def analyze(oc: list[dict], cc: list[dict] | None = None, family=("H1", "H2"), n_boot: int = N_BOOT,
            seed: int = SEED) -> dict:
    family = list(family)
    if not {"H1", "H2"} <= set(family) <= {"H1", "H2", "H3"}:
        raise SystemExit(f"family must be H1,H2 or H1,H2,H3, not {family}")
    if "H3" in family and not cc:
        raise SystemExit("H3 is in the family but there is no Claude Code job")
    res = {"family": family, "alpha": ALPHA, "n_boot": n_boot, "seed": seed, "oc": analyze_job(oc, True, n_boot, seed)}
    tests = {"H1": res["oc"]["gain"], "H2": res["oc"]["noninf"]}
    if cc:
        res["cc"] = analyze_job(cc, False, n_boot, seed)
        tests["H3"] = res["cc"]["gain"]  # descriptive only when H3 is not in the family
    res["tests"] = tests
    res["verdict"] = verdict(tests, family)
    return res


# ---------------------------------------------------------------- report

def pp(c: dict, sign: bool = True) -> str:
    f = "{:+.1f}" if sign else "{:.1f}"
    return f"{f.format(100 * c['est'])} [{f.format(100 * c['lo'])}, {f.format(100 * c['hi'])}]"


def usd(c: dict) -> str:
    return f"{c['est']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]"


def core_line(c: dict) -> str:
    s = (f"n {c['n']:4d}  A {pp(c['A'], False)}  B {pp(c['B'], False)}  d_nat {pp(c['d_nat'])}pp "
         f"(W_nat {c['W_nat']}, L {c['L']})")
    return s + (f"  C {pp(c['C'], False)}  e {pp(c['e'])}pp" if "C" in c else "")


def info_lines(job: str, n: int, i: dict) -> list[str]:
    flag = f"  ** >{100 * K11_MAX:.0f}% exclusions (K11): report and flag **" if i["infra_rate"] > K11_MAX else ""
    out = [f"[{job.upper()}] {i['path']} (part {i.get('part', '?')}): {n} tasks from {i['trials']} trials of "
           f"{i['ids']} ids; {len(i['excluded'])} ids excluded ({100 * i['infra_rate']:.1f}%){flag}",
           f"  unusable trials by reason {i['reasons'] or '{}'}; CHECK_ERROR statuses by branch "
           f"{i['check_errors'] or '{}'}"]
    for k, name in (("infra_msgs", "infra entries"), ("warnings", "warnings (do not exclude)")):
        if i.get(k):
            out.append(f"  regrade {name}, most common: {i[k]}")
    if i["egress"]:
        k2 = "  ** K2: above 2%, halt and fix **" if i["egress_rate"] > K2_MAX else ""
        out.append(f"  K2 egress: {len(i['egress'])} ids ({100 * i['egress_rate']:.1f}%) {i['egress'][:10]}{k2}")
    if i.get("curated") is None:
        out.append("  no curation file: TRIVIAL_THEOREMS / VACUOUS / BROKEN tasks are NOT dropped")
    elif i["curation"]:
        out.append(f"  dropped by curation (not in the sample) {i['curation']}: {i['curation_ids'][:10]}")
    if i["duplicates"]:
        out.append(f"  {len(i['duplicates'])} ids with >1 usable trial (earliest kept)"
                   + (f"; no start time for {i['unordered']}, file order used" if i["unordered"] else ""))
    return out


def report(res: dict, info: dict) -> None:
    dev = any(i.get("part") == "dev" for i in info.values())
    print(("** DEV DRY RUN: not confirmatory **\n" if dev else "")
          + f"== Millennium Harness FV v1.0 frozen analysis: family {','.join(res['family'])}, "
          f"Holm one-sided alpha {ALPHA}, {res['n_boot']} bootstrap resamples (seed {res['seed']}) ==")
    for job in ("oc", "cc"):
        if job not in res:
            continue
        j = res[job]
        print("\n" + "\n".join(info_lines(job, j["strict"]["n"], info[job])))
        for g in GRADERS:
            print(f"  {g:8s} {core_line(j[g])}")
    print("\nConfirmatory (H1/H3: Holm on max(p strict, p lenient); H2: p strict)")
    for h, t in res["tests"].items():
        s, le = t["strict"], t["lenient"]
        pbin = f"{s['p_binom']:.4f}" if s["p_binom"] is not None else f"n/a ({s['losses']} losses)"
        line = (f"  {h} E[{'e' if h == 'H2' else 'd_nat'}] > {s['null']:+.2f}: {pp(s)}pp  p_boot {s['p_boot']:.4f}  "
                + (f"p_binom {pbin}  " if h != "H2" else "") + f"p strict {s['p']:.4f}"
                + (f"  p lenient {le['p']:.4f} ({pp(le)}pp)" if h != "H2" else ""))
        v = res["verdict"].get(h)
        if v is None:
            print(line + "  (descriptive: not in the family)")
            continue
        print(line + f"  p {v['p']:.4f}  crit {v['crit']:.4f}  -> {'CLAIM' if v['claim'] else 'not significant'}")
    oc, cs = res["oc"], res["oc"]["cost"]
    st = oc["strict"]
    print("\nSecondary (estimates and 95% CIs only)")
    print(f"  S1 dB(all stops) {pp(st['dB_all'])}pp  dC {pp(st['dC'])}pp  C harm {pp(st['harm_C'], False)}%  "
          f"B harm {pp(st['harm_B'], False)}%  B-C on fired ({st['fired']}) {pp(st['e_fired'])}pp  "
          f"W_all {st['W_all']} (gate-silent {st['W_silent']})")
    print(f"     extra $/task B {usd(cs['extra_B'])}  C {usd(cs['extra_C'])}  C/B {cs['ratio_C_over_B']['est']:.2f}x "
          f"[{cs['ratio_C_over_B']['lo']:.2f}, {cs['ratio_C_over_B']['hi']:.2f}]"
          + (f"  tokens C/B {cs['tok_ratio_C_over_B']['est']:.2f}x" if "tok_ratio_C_over_B" in cs else ""))
    print("     $/honest-verified " + "  ".join(f"{x} {usd(cs['usd_per_verified_' + x])}" for x in "ABC")
          + "  time p50/p90 " + "  ".join(f"{x} {cs['secs_' + x]['p50']:.0f}/{cs['secs_' + x]['p90']:.0f}s"
                                          for x in "ABC"))
    for job in ("oc", "cc"):
        if job not in res:
            continue
        j = res[job]
        for name, s in (("S2 cell", j["S2"]["cell"]), ("S2 verifier", j["S2"]["verifier"]),
                        ("S6 memo", j.get("S6", {})), ("S7 stop", j["S7"])):
            for k, c in s.items():
                print(f"  {job.upper()} {name:11s} {k:9s} {core_line(c)}")
        print(f"  {job.upper()} S3 lenient   {core_line(j['lenient'])}")
        if "S4" in j:
            print(f"  {job.upper()} S4 false-done {pp(j['S4']['all'], False)}% of all, "
                  f"{pp(j['S4']['of_A_fail'], False)}% of failing S0")
        print(f"  {job.upper()} S5 strict statuses {j['S5']}  cheat/tamper ids (any snapshot) {j['S5_examples']}")
        if "S8" in j:
            s8 = j["S8"]
            print(f"  {job.upper()} S8 receipt soundness {s8['incidents']}/{s8['gate_proved']} "
                  f"(95% CP upper {100 * s8['cp95'][1]:.2f}%) {s8['ids']}")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oc", required=True, help="OpenCode FVBranch job (A, B, C)")
    ap.add_argument("--cc", default=None, help="Claude Code ClaudeCodeFV job (A, B)")
    ap.add_argument("--family", required=True, help="H1,H2 or H1,H2,H3, as fixed at the tag")
    ap.add_argument("--json", default=None, help="also write the full results here")
    ap.add_argument("--curated", default=str(CURATED), help="curation statuses (bench/fv/curate.py), or 'none'")
    ap.add_argument("--dev", action="store_true", help="dry run on DEV grades (part 'dev'); never confirmatory")
    a = ap.parse_args(argv)
    part, cur = ("dev" if a.dev else "test"), (None if a.curated == "none" else a.curated)
    oc, info = load(a.oc, part, cur)
    infos = {"oc": info}
    if not info["has_c"]:
        raise SystemExit(f"{a.oc} has no C branch; --oc must be an FVBranch job")
    cc = None
    if a.cc:
        cc, infos["cc"] = load(a.cc, part, cur)
    res = analyze(oc, cc, a.family.split(","))
    res["info"] = infos
    report(res, infos)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
