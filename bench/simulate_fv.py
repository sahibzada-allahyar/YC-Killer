"""Power, size and CI coverage of the frozen FV tests (bench/fv_analyze.py) on simulated trials with planted effects,
from measured dev parameters (plan §4 power check, §6 MDE table). No model calls, no Modal.

  .venv/bin/python bench/simulate_fv.py --from runs/fv/grades/<dev job>.jsonl [--g <g_plan>] [--m 3] [--n 150,188,240]
  .venv/bin/python bench/simulate_fv.py --p0 0.5 --f 0.45 --g 0.25 --nat 0.9
H3 is H1 on the Claude Code job: rerun with --from <Claude Code dev job> --n 50,100 and read the H1 rows.

Model per task: A ~ Bern(p0). A failing S0 fires the gate with prob phi = min(1, f / (1 - p0)) (f = share of trials
where the gate fired on a failing S0). The S0 stop of a fired trial is end_turn with prob nat; B repairs it (B PROVED)
with prob g after an end_turn stop and with prob g_other after any other stop (default g). A PROVED S0 is broken by B
with prob h and by C with prob hc. C repairs a failing S0 with marginal prob gc (default: as often as B, i.e. B = C)
and with prob `both` when B repaired it too. A share `ban` of failing S0 files fail only the strict grader (ban-only:
every branch is lenient-PROVED there).
  E[d_nat] = (1 - p0) phi nat g - p0 h          E[e] = P(B PROVED) - P(C PROVED)
H1/H3 are tested as fv_analyze claims them: max(p strict, p lenient) <= level. Prints the planted effects; formula
and simulated power at the Holm worst step (alpha / m) for each N; the N needed for 80% power; each test's size at its
null boundary and Holm's family-wise error there; and 95% CI coverage. --from measures every parameter from a DEV
grades file (TEST files are refused before and after the tag); flags override (e.g. --g g_plan, plan §4).
Measured sizes at the null boundary (defaults, h = hc in {0, 0.03}, 6,000 reps, MC SE 0.002): the H1/H3 test is
conservative (about 0.005: the exact-binomial guard), while the H2 bootstrap test is NOT: 0.025-0.028 at N 100 and
240, 0.029-0.031 at N 150 (a percentile bootstrap of a mean is slightly liberal at these N).
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import fv_analyze as FA  # noqa: E402

DEFAULTS = {"p0": 0.5, "f": 0.45, "g": 0.25, "g_other": None, "nat": 0.9, "h": 0.0, "gc": None, "both": 0.6,
            "hc": 0.0, "ban": 0.0}
POWER = 0.80
Z = NormalDist()
S0 = math.sqrt(FA.NULL["H1"] * (1 - FA.NULL["H1"]))


def probs(P: dict) -> dict:
    """Derived probabilities, the planted E[d_nat] and E[e], and the per-task variances. C's marginal repair rate
    pic is gc clipped to [pib * both, pib * both + 1 - pib]."""
    p0, both = P["p0"], P["both"]
    phi = min(1.0, P["f"] / (1 - p0)) if p0 < 1 else 0.0
    pib = phi * (P["nat"] * P["g"] + (1 - P["nat"]) * g_other(P))  # P(B repairs | A fails)
    pic = pib if P["gc"] is None else P["gc"]  # marginal P(C repairs | A fails)
    c0 = min(1.0, max(0.0, (pic - pib * both) / (1 - pib))) if pib < 1 else 0.0  # P(C repairs | A fails, B did not)
    win, loss = (1 - p0) * phi * P["nat"] * P["g"], p0 * P["h"]
    up = (1 - p0) * pib * (1 - both) + p0 * P["hc"] * (1 - P["h"])  # e = +1
    dn = (1 - p0) * (1 - pib) * c0 + p0 * P["h"] * (1 - P["hc"])  # e = -1
    Ed, Ee = win - loss, up - dn
    return {"phi": phi, "pib": pib, "pic": pib * both + (1 - pib) * c0, "c0": c0, "Ed": Ed,
            "Ed_lenient": win * (1 - P["ban"]) - loss, "var_d": win + loss - Ed ** 2, "Ee": Ee,
            "var_e": up + dn - Ee ** 2}


def g_other(P: dict) -> float:
    return P["g"] if P.get("g_other") is None else P["g_other"]


def draw(n: int, P: dict, rng) -> dict:
    """n simulated trials as boolean columns: A, B, C (strict PROVED), fired, natural (end_turn), ban."""
    q, u = probs(P), rng.random((8, n))
    A = u[0] < P["p0"]
    fired, natural = ~A & (u[1] < q["phi"]), u[6] < P["nat"]
    rep_b, harm_b = fired & (u[2] < np.where(natural, P["g"], g_other(P))), A & (u[3] < P["h"])
    C = (A & ~(u[4] < P["hc"])) | (~A & (u[5] < np.where(rep_b, P["both"], q["c0"])))
    return {"A": A, "B": (A & ~harm_b) | rep_b, "C": C, "fired": fired | harm_b, "natural": natural,
            "ban": ~A & (u[7] < P["ban"])}


def outcomes(x: dict, lenient: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """d_nat and e per trial (a repair always changes the file, so the sha condition holds)."""
    A, B, C = (x[k] | x["ban"] if lenient else x[k] for k in "ABC")
    return (~A & B & x["natural"]).astype(int) - (A & ~B).astype(int), B.astype(int) - C.astype(int)


def to_rows(x: dict, cells=("L-fv", "L-apps", "D-np"), part: str = "dev") -> list[dict]:
    """The simulated trials in the runs/fv/grades schema, with fixed synthetic costs and times."""
    rows = []
    for i in range(len(x["A"])):
        f, ban = bool(x["fired"][i]), bool(x["ban"][i])
        r = {"id": f"SIM{i:05d}", "cell": cells[i % len(cells)], "part": part, "b_fired": f, "infra_fail": False,
             "stop_reason": "end_turn" if x["natural"][i] else "length",
             "sha_A": f"s{i}", "sha_B": f"s{i}b" if f else f"s{i}", "sha_C": f"s{i}c",
             "cost_A": 0.02, "cost_B": 0.012 if f else 0.0, "cost_C": 0.009,
             "secs_A": 300.0, "secs_B": 120.0 if f else 0.0, "secs_C": 100.0}
        for k in "ABC":
            r[f"{k}_strict"] = "PROVED" if x[k][i] else "CHEAT_REJECTED" if ban else "UNPROVED"
            r[f"{k}_lenient"] = "PROVED" if x[k][i] or ban else "UNPROVED"
        rows.append(r)
    return rows


def tests(x: dict, W: np.ndarray) -> dict:
    """The frozen tests on one simulated job, shaped like fv_analyze's res["tests"] entries (H2: strict only)."""
    (d, e), (dl, _) = outcomes(x), outcomes(x, lenient=True)
    return {"H1": {g: FA.one_sided(v, FA.NULL["H1"], W, exact=True) for g, v in (("strict", d), ("lenient", dl))},
            "H2": {"strict": FA.one_sided(e, FA.NULL["H2"], W)}}


def simulate(n: int, P: dict, alpha: float, reps: int = 300, n_boot: int = 2000, seed: int = 0) -> dict:
    """Rejection rates of the H1 test (d_nat, exact guard, both graders) and the H2 test (e) at level alpha, and how
    often the 95% strict-grader bootstrap CI covers the planted mean."""
    q, rng, hit = probs(P), np.random.default_rng(seed), Counter()
    for _ in range(reps):
        x = draw(n, P, rng)
        W = FA.boot_weights(n, n_boot, int(rng.integers(2 ** 31)))
        t = tests(x, W)
        t1, t2 = t["H1"]["strict"], t["H2"]["strict"]
        hit["H1"] += FA.family_p("H1", t["H1"]) <= alpha
        hit["H2"] += FA.family_p("H2", t["H2"]) <= alpha
        hit["cover_d"] += t1["lo"] <= q["Ed"] <= t1["hi"]
        hit["cover_e"] += t2["lo"] <= q["Ee"] <= t2["hi"]
    return {k: hit[k] / reps for k in ("H1", "H2", "cover_d", "cover_e")}


def at_null(P: dict) -> tuple[dict, dict]:
    """P moved to the H1 boundary (g with E[d_nat] = 0.03) and to the H2 boundary (gc with E[e] = -0.05)."""
    q = probs(P)
    p1 = {**P, "g": (FA.NULL["H1"] + P["p0"] * P["h"]) / ((1 - P["p0"]) * q["phi"] * P["nat"])}
    p2 = {**P, "gc": q["pib"] + (P["p0"] * (P["hc"] - P["h"]) - FA.NULL["H2"]) / (1 - P["p0"])}
    return p1, p2


def size(n: int, P: dict, m: int, reps: int = 300, n_boot: int = 2000, seed: int = 1) -> dict:
    """Every hypothesis at its null boundary, on independent draws: each test's rejection rate at alpha (the most
    lenient Holm step), and the rate at which Holm rejects anything (family-wise error)."""
    p1, p2 = at_null(P)
    rng, rej, fwer = np.random.default_rng(seed), Counter(), 0
    for _ in range(reps):
        ps = {}
        for h, Q in zip(("H1", "H2", "H3")[:m], (p1, p2, p1)):
            x = draw(n, Q, rng)
            W = FA.boot_weights(n, n_boot, int(rng.integers(2 ** 31)))
            ps[h] = FA.family_p(h, tests(x, W)["H2" if h == "H2" else "H1"])
            rej[h] += ps[h] <= FA.ALPHA
        fwer += any(v["reject"] for v in FA.holm(ps).values())
    return {**{h: rej[h] / reps for h in ps}, "fwer": fwer / reps, "Ed": probs(p1)["Ed"], "Ee": probs(p2)["Ee"]}


def formula_power(n: int, q: dict, h: str, alpha: float) -> float:
    z = Z.inv_cdf(1 - alpha)
    if h == "H2":
        return Z.cdf(math.sqrt(n) * (q["Ee"] - FA.NULL["H2"]) / math.sqrt(q["var_e"]) - z) if q["var_e"] > 0 else 0.0
    if q["var_d"] <= 0:
        return 0.0
    return Z.cdf((math.sqrt(n) * (q["Ed"] - FA.NULL["H1"]) - z * S0) / math.sqrt(q["var_d"]))


def formula_n(q: dict, h: str, alpha: float, power: float = POWER) -> float:
    """Plan §6: N = ((z_a sd_null + z_b sd) / (effect - null))^2 for H1/H3 and ((z_a + z_b) sd_e / (E[e] + 0.05))^2."""
    z, zb = Z.inv_cdf(1 - alpha), Z.inv_cdf(power)
    eff = q["Ee"] - FA.NULL["H2"] if h == "H2" else q["Ed"] - FA.NULL["H1"]
    top = (z + zb) * math.sqrt(q["var_e"]) if h == "H2" else z * S0 + zb * math.sqrt(q["var_d"])
    return math.ceil((top / eff) ** 2) if eff > 0 else math.inf


def n_needed(P: dict, h: str, alpha: float, reps: int, n_boot: int, seed: int) -> int | None:
    """Smallest N (to about 2%) whose simulated power reaches 80%, by bisection from the formula N."""
    start = formula_n(probs(P), h, alpha)
    if start == math.inf:
        return None
    lo, hi = max(10, start // 3), max(20, 2 * start)
    while simulate(hi, P, alpha, reps, n_boot, seed)[h] < POWER:
        lo, hi = hi, hi * 2
        if hi > 20000:
            return None
    while hi - lo > max(2, lo // 50):
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if simulate(mid, P, alpha, reps, n_boot, seed)[h] >= POWER else (mid, hi)
    return hi


def measure(rows: list[dict]) -> dict:
    """Model parameters from a DEV grades file (strict grader). nat, g and g_other are all measured on the fired
    failing-S0 trials, the population the model applies them to. TEST rows are refused."""
    test = sum(r.get("part") == "test" for r in rows)
    if test:
        raise SystemExit(f"refusing to measure from {test} TEST rows: parameters come from DEV only (plan §5)")

    def rate(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0
    ok = [r for r in rows if FA.proved(r, "A")]
    bad = [r for r in rows if not FA.proved(r, "A")]
    fired = [r for r in bad if r.get("b_fired")]
    nat = [r for r in fired if r.get("stop_reason") == "end_turn"]
    other = [r for r in fired if r.get("stop_reason") != "end_turn"]
    P = {"p0": len(ok) / len(rows), "f": len(fired) / len(rows), "g": rate(FA.proved(r, "B") for r in nat or fired),
         "nat": rate(r.get("stop_reason") == "end_turn" for r in fired), "h": rate(not FA.proved(r, "B") for r in ok),
         "ban": rate(FA.proved(r, "A", "lenient") for r in bad)}
    if other:
        P["g_other"] = rate(FA.proved(r, "B") for r in other)
    if any(r.get("C_strict") is not None for r in rows):
        rep = [r for r in bad if FA.proved(r, "B")]
        P.update(gc=rate(FA.proved(r, "C") for r in bad), hc=rate(not FA.proved(r, "C") for r in ok))
        if rep:
            P["both"] = rate(FA.proved(r, "C") for r in rep)
    return P


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default=None, help="grades file (or job name) to measure parameters from")
    for k in DEFAULTS:
        ap.add_argument(f"--{k}", type=float, default=None)
    ap.add_argument("--m", type=int, default=3, choices=(2, 3), help="family size fixed at the tag")
    ap.add_argument("--n", default="150,188,240", help="N values for the power table")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples per simulated analysis")
    ap.add_argument("--seed", type=int, default=FA.SEED)
    a = ap.parse_args(argv)
    P = dict(DEFAULTS)
    if a.src:
        rows, info = FA.load(a.src, part="dev")
        P.update(measure(rows))
        d = [FA.d_nat(r) for r in rows]
        print(f"measured from {info['path']} ({len(rows)} tasks): empirical mean d_nat {100 * sum(d) / len(d):+.1f}pp "
              f"(W_nat {d.count(1)}, L {d.count(-1)})")
    P.update({k: getattr(a, k) for k in DEFAULTS if getattr(a, k) is not None})
    q, aw, ns = probs(P), FA.ALPHA / a.m, [int(x) for x in a.n.split(",")]
    print("params " + "  ".join(f"{k}={({'gc': 'B=C', 'g_other': '=g'}.get(k) if v is None else f'{v:.3f}')}"
                                for k, v in P.items())
          + f"  (C repairs a failing S0 w.p. {q['pic']:.3f}; Holm worst step alpha/{a.m} = {aw:.4f})")
    print(f"planted E[d_nat] {100 * q['Ed']:+.2f}pp (lenient {100 * q['Ed_lenient']:+.2f}pp, "
          f"SD {math.sqrt(q['var_d']):.3f})  E[e] {100 * q['Ee']:+.2f}pp (SD {math.sqrt(q['var_e']):.3f})")
    print("power at the Holm worst step: the plan §6 normal formula, and simulation of the frozen tests "
          "(authoritative; the bootstrap p uses the observed SD, so the formula is optimistic for H1/H3)")
    print(f"\n{'N':>5s}  {'H1 formula':>10s} {'H1 sim':>7s}  {'H2 formula':>10s} {'H2 sim':>7s}  "
          f"{'cover d':>7s} {'cover e':>7s}")
    for n in ns:
        s = simulate(n, P, aw, a.reps, a.n_boot, a.seed)
        print(f"{n:5d}  {formula_power(n, q, 'H1', aw):10.2f} {s['H1']:7.2f}  {formula_power(n, q, 'H2', aw):10.2f} "
              f"{s['H2']:7.2f}  {s['cover_d']:7.2f} {s['cover_e']:7.2f}")
    for h in ("H1", "H2"):
        fn, sn = formula_n(q, h, aw), n_needed(P, h, aw, a.reps, a.n_boot, a.seed)
        print(f"N for {100 * POWER:.0f}% power, {h} at alpha/{a.m}: formula {fn}, "
              f"simulated {sn or 'n/a (>20000 or no effect)'}")
    sz = size(max(ns), P, a.m, a.reps, a.n_boot, a.seed + 1)
    print(f"\nsize at the null boundary, N={max(ns)}, alpha {FA.ALPHA} (E[d_nat] {100 * sz['Ed']:+.2f}pp, "
          f"E[e] {100 * sz['Ee']:+.2f}pp): " + "  ".join(f"{h} {sz[h]:.3f}" for h in ("H1", "H2", "H3") if h in sz)
          + f"  Holm FWER {sz['fwer']:.3f}  (MC SE about {math.sqrt(FA.ALPHA * (1 - FA.ALPHA) / a.reps):.3f})")


if __name__ == "__main__":
    main(sys.argv[1:])
