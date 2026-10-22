"""E1 analysis: selectors on a fixed candidate pool + G1 go/no-go. Expected values over random tie-breaks.

  python bench/e1_analyze.py [runs/e1]
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e1")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def pool(cs):
    P = [c for c in cs if c.get("pub")]
    return P or [c for c in cs if c.get("extracted")] or cs


def acc(cs):
    return mean(c.get("hidden", False) for c in cs) if cs else 0.0


def sel_spec(cs, key):
    P = pool(cs)
    scored = [c for c in P if c.get(key)]
    if not scored:
        return acc(P), False
    m = min(c[key]["n_dis"] for c in scored)
    return acc([c for c in scored if c[key]["n_dis"] == m]), True


def sel_codet(cs):
    P = pool(cs)
    scored = [c for c in P if c.get("codet_pass") is not None]
    if not scored:
        return acc(P)
    groups = defaultdict(list)
    for c in scored:
        groups[tuple(c["codet_pass"])].append(c)
    score = {k: len(v) * sum(k) for k, v in groups.items()}
    best = max(score.values())
    return acc([c for k, v in groups.items() if score[k] == best for c in v])


def sel_mbr(cs):
    P = pool(cs)
    scored = [c for c in P if c.get("mbr_sig")]
    if not scored:
        return acc(P)
    cnt = Counter(c["mbr_sig"] for c in scored)
    best = max(cnt.values())
    return acc([c for c in scored if cnt[c["mbr_sig"]] == best])


def sel_judge(cs, pick):
    P = pool(cs)
    ids = {c["i"]: c for c in P}
    if pick is None or pick not in ids:
        return acc(P)
    return float(ids[pick].get("hidden", False))


def sel_spec_judge(cs, pick, key="s4"):
    """Exploratory: judge's pick if it survives the spec (min disagreement set), else the spec's set."""
    P = pool(cs)
    scored = [c for c in P if c.get(key)]
    if not scored:
        return sel_judge(cs, pick)
    m = min(c[key]["n_dis"] for c in scored)
    best = [c for c in scored if c[key]["n_dis"] == m]
    ids = {c["i"]: c for c in best}
    if pick in ids:
        return float(ids[pick].get("hidden", False))
    return acc(best)


def boot(diffs, n=10000, seed=0):
    rng = random.Random(seed)
    k = len(diffs)
    ms = sorted(mean(diffs[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def main():
    ev = [json.loads(l) for l in open(D / "eval.jsonl")]
    judge = json.loads((D / "judge.json").read_text()) if (D / "judge.json").exists() else {}
    rows = []
    for e in ev:
        cs = e["cands"]
        r = {"pid": e["pid"], "kind": e["kind"], "diff": e["difficulty"]}
        r["S0_random"] = acc(cs)
        r["S1_public"] = acc(pool(cs))
        r["S2_codet"] = sel_codet(cs)
        r["S3_mbr"] = sel_mbr(cs)
        r["S4_millennium"], r["spec_ok"] = sel_spec(cs, "s4")
        r["S4_k1"], _ = sel_spec(cs, "s4_k1")
        r["S4_nofilter"], _ = sel_spec(cs, "s4_nofilter")
        r["S4_7cands"], _ = sel_spec(cs[:7], "s4")
        if judge:
            r["S5_judge"] = sel_judge(cs, judge.get(e["pid"]))
            r["S6_spec+judge"] = sel_spec_judge(cs, judge.get(e["pid"]))
        r["Sinf_oracle"] = float(any(c.get("hidden") for c in cs))
        rows.append(r)

    keys = [k for k in rows[0] if k.startswith("S")]
    print(f"\nE1 selector study — {len(rows)} problems × {len(ev[0]['cands'])} candidates (Haiku 5.5, effort medium)\n")
    print(f"{'selector':16s} {'pass@1':>7s}   Δ vs S1 [95% CI]")
    for k in keys:
        d = [r[k] - r["S1_public"] for r in rows]
        lo, hi = boot(d)
        print(f"{k:16s} {100*mean(r[k] for r in rows):6.1f}%   {100*mean(d):+5.1f}pp [{100*lo:+.1f}, {100*hi:+.1f}]")

    # G1 diagnostics (candidate-level, problems with a valid k=3 suite)
    ok_pids = {e["pid"] for e in ev if e["suites"]["s4"]["status"] == "OK"}
    fa_n = fa_d = rc_n = rc_d = 0
    for e in ev:
        if e["pid"] not in ok_pids:
            continue
        for c in e["cands"]:
            if not c.get("s4"):
                continue
            if c.get("hidden"):
                fa_d += 1; fa_n += c["s4"]["n_dis"] > 0
            elif c.get("pub"):
                rc_d += 1; rc_n += c["s4"]["n_dis"] > 0
    headroom = mean(r["Sinf_oracle"] - r["S1_public"] for r in rows)
    spec_rate = len(ok_pids) / len(ev)
    fa = fa_n / fa_d if fa_d else float("nan")
    rc = rc_n / rc_d if rc_d else float("nan")
    d4 = [r["S4_millennium"] - r["S1_public"] for r in rows]
    lo, hi = boot(d4)
    print(f"\nheadroom (S∞−S1)         {100*headroom:5.1f}pp   (GO needs ≥10)")
    print(f"valid k=3 spec rate       {100*spec_rate:5.1f}%    (GO needs ≥60)")
    print(f"false-alarm rate          {100*fa:5.1f}%    ({fa_n}/{fa_d} hidden-correct flagged; GO needs <10, KILL >20)")
    print(f"recall on pub✓/hidden✗    {100*rc:5.1f}%    ({rc_n}/{rc_d})")
    print(f"S4−S1                     {100*mean(d4):+5.1f}pp [{100*lo:+.1f}, {100*hi:+.1f}]  (GO: ≥+3 & CI>0; KILL: <+1)")
    go = headroom >= .10 and spec_rate >= .60 and fa < .10 and mean(d4) >= .03 and lo > 0
    kill = mean(d4) < .01 or fa > .20
    print("\nG1:", "GO" if go else ("KILL" if kill else "AMBIGUOUS"))
    by = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["diff"])].append(r["S4_millennium"] - r["S1_public"])
    print("strata Δ(S4−S1):", {f"{k[0]}/{k[1]}": f"{100*mean(v):+.1f}pp n={len(v)}" for k, v in sorted(by.items())})
    (D / "selectors.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
