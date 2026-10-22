"""Frozen FV analysis (bench/fv_analyze.py) and its simulator (bench/simulate_fv.py) on synthetic data only:
outcome rules, usable-trial rules, Holm with the intersection-union H1/H3 p, the exact-binomial guard, the DEV/TEST
guards, planted effects recovered, and each test's size at its null boundary."""
import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from bench import fv_analyze as FA
from bench import simulate_fv as SIM

P = dict(SIM.DEFAULTS)


def trial(i="T1", a="UNPROVED", b="PROVED", c="UNPROVED", stop="end_turn", sha_b="s1", **kw):
    r = {"id": i, "cell": "L-fv", "stop_reason": stop, "b_fired": b != a, "infra_fail": False,
         "sha_A": "s0", "sha_B": sha_b, "sha_C": "s2", "cost_A": 0.02, "cost_B": 0.01, "cost_C": 0.01,
         "secs_A": 100, "secs_B": 50, "secs_C": 50}
    for g in FA.GRADERS:
        r.update({f"A_{g}": a, f"B_{g}": b, f"C_{g}": c})
    return {**r, **kw}


def test_d_nat_and_e_rules():
    assert FA.d_nat(trial()) == 1
    assert FA.d_nat(trial(stop="length")) == 0  # repairs after a non-natural stop are not counted
    assert FA.d_nat(trial(sha_b="s0")) == 0  # B's file must differ from S0's
    assert FA.d_nat(trial(a="PROVED", b="UNPROVED")) == -1
    assert FA.d_nat(trial(a="PROVED", b="PROVED")) == 0
    assert FA.d_nat(trial(a="CHEAT_REJECTED", b="SPEC_TAMPERED")) == 0
    assert FA.e_bc(trial()) == 1 and FA.e_bc(trial(c="PROVED")) == 0 and FA.e_bc(trial(b="UNPROVED", c="PROVED")) == -1
    r = trial(A_lenient="PROVED")  # ban-only S0: strict flip, no lenient flip
    assert FA.d_nat(r, "strict") == 1 and FA.d_nat(r, "lenient") == 0


def test_dedupe_excludes_infra_and_uses_retry():
    rows, info = FA.dedupe([trial("T1", infra_fail=True), trial("T2"), trial("T1"), trial("T3", infra_fail=True),
                            trial("T4", C_strict=None)])
    assert [r["id"] for r in rows] == ["T1", "T2"]
    assert info["excluded"] == ["T3", "T4"] and info["has_c"] and info["infra_rate"] == 0.5
    cc = [{k: v for k, v in trial(i).items() if not k.startswith("C_")} for i in ("X", "Y")]
    rows, info = FA.dedupe(cc)
    assert len(rows) == 2 and not info["has_c"]


def nested(task, a="UNPROVED", b="PROVED", c="UNPROVED", infra=(), with_c=True, **kw):
    """A row as bench/fv/regrade.py writes it (an FVBranch trial; with_c=False: Claude Code)."""
    def br(label, sha, s):
        return {"label": label, "sha": sha, "strict": s, "lenient": s}
    snaps = {"S0": {"sha": "s0", "strict": a, "lenient": a}, "B1": {"sha": "s1", "strict": b, "lenient": b}}
    if with_c:
        snaps.update(C1={"sha": "s2", "strict": c, "lenient": c}, C2={"sha": "s2", "strict": c, "lenient": c})
    return {"job": "j", "trial": f"{task.lower()}__abc", "task": task, "cell": "L-apps", "part": "test", "lang": "lean",
            "agent": "FVBranch" if with_c else "ClaudeCodeFV", "s0_stop": "end_turn", "b_fired": True,
            "b_rounds": 1, "c_rounds": 2 if with_c else None, "c_restore_ok": True if with_c else None,
            "gate": ["UNPROVED", b], "egress": "github.com 000\ngithub.com blocked\nhuggingface.co 000\natcoder.jp 000",
            "A": br("S0", "s0", a), "B": br("B1", "s1", b), "C": br("C2", "s2", c) if with_c else None,
            "snaps": snaps, "usage": {"A": {"cost": 0.02, "in": 9}, "B": {"cost": 0.01, "in": 3, "out": 1},
                                      "C": {"cost": 0.02, "in": 5} if with_c else None},
            "secs": {"A": 90.0, "B": 40.0, "C": 30.0 if with_c else None}, "infra": list(infra), "warnings": [],
            "exception": None,
            "started_at": "2026-10-20T10:00:00.000000", **kw}


def test_regrade_rows_are_flattened():
    rows, info = FA.dedupe([nested("LA1"), nested("LA2", infra=["no B1 snapshot"])])
    assert [r["id"] for r in rows] == ["LA1"] and info["excluded"] == ["LA2"] and info["has_c"]
    r = rows[0]
    assert FA.d_nat(r) == 1 and FA.e_bc(r) == 1 and r["gate_B"] == "PROVED" and r["stop_reason"] == "end_turn"
    assert (r["cost_B"], r["tok_B"], r["secs_C"], r["sha_A"], r["cell"]) == (0.01, 4, 30.0, "s0", "L-apps")
    rows, info = FA.dedupe([nested("LA3", with_c=False)])
    assert len(rows) == 1 and not info["has_c"] and "C_strict" not in rows[0]


def test_binomial_and_clopper_pearson():
    assert FA.binom_sf(0, 50, 0.03) == 1.0
    brute = sum(math.comb(10, j) * 0.3 ** j * 0.7 ** (10 - j) for j in range(4, 11))
    assert FA.binom_sf(4, 10, 0.3) == pytest.approx(brute, rel=1e-9)
    lo, hi = FA.clopper_pearson(0, 100)
    assert lo == 0 and hi == pytest.approx(1 - 0.025 ** (1 / 100), abs=1e-6)
    lo, hi = FA.clopper_pearson(5, 20)
    assert FA.binom_sf(5, 20, lo) == pytest.approx(0.025, abs=1e-6)
    assert 1 - FA.binom_sf(6, 20, hi) == pytest.approx(0.025, abs=1e-6)


def test_holm():
    v = FA.holm({"H1": 0.001, "H2": 0.02, "H3": 0.03})
    assert v["H1"]["reject"] and not v["H2"]["reject"] and not v["H3"]["reject"]
    assert v["H1"]["crit"] == pytest.approx(0.025 / 3) and v["H2"]["crit"] == pytest.approx(0.0125)
    v = FA.holm({"H1": 0.01, "H2": 0.005})
    assert v["H1"]["reject"] and v["H2"]["reject"] and v["H1"]["p_holm"] == pytest.approx(0.01)
    v = FA.holm({"H1": 0.009, "H2": 0.0001, "H3": 0.5})  # step-down: 0.0001 <= alpha/3, 0.009 <= alpha/2, stop
    assert [v[h]["reject"] for h in ("H1", "H2", "H3")] == [True, True, False]


def test_bootstrap_and_exact_guard():
    W = FA.boot_weights(50, 2000, 7)
    assert (W.sum(1) == 50).all() and (W == FA.boot_weights(50, 2000, 7)).all()
    d = [1] * 4 + [0] * 46
    t = FA.one_sided(d, 0.03, W, exact=True)
    assert t["p_binom"] == pytest.approx(FA.binom_sf(4, 50, 0.03)) and t["p"] == max(t["p_boot"], t["p_binom"])
    t = FA.one_sided([-1] + d[1:], 0.03, W, exact=True)  # a loss disables the exact test
    assert t["p_binom"] is None and t["p"] == t["p_boot"] and t["losses"] == 1
    assert FA.one_sided([0] * 50, 0.03, W, exact=True)["p"] == 1.0


def test_vectorised_outcomes_match_row_rules():
    Q = {**P, "h": 0.05, "hc": 0.05, "ban": 0.3, "gc": 0.1}
    x = SIM.draw(3000, Q, np.random.default_rng(3))
    rows = SIM.to_rows(x)
    for lenient, g in ((False, "strict"), (True, "lenient")):
        d, e = SIM.outcomes(x, lenient)
        assert list(d) == [FA.d_nat(r, g) for r in rows] and list(e) == [FA.e_bc(r, g) for r in rows]


def test_measure_recovers_planted_parameters():
    Q = {**P, "h": 0.05, "hc": 0.04, "ban": 0.3, "gc": 0.2, "both": 0.7, "g_other": 0.1}
    m = SIM.measure(SIM.to_rows(SIM.draw(40000, Q, np.random.default_rng(4))))
    assert set(m) == set(Q)
    for k, v in m.items():
        assert v == pytest.approx(Q[k], abs=0.02), k


def test_planted_effect_recovered_and_claimed():
    Q = {**P, "gc": 0.05, "both": 0.2}  # B repairs 22.5% of failing S0s, C 5%: H1 and H2 both true
    q = SIM.probs(Q)
    oc = SIM.to_rows(SIM.draw(400, Q, np.random.default_rng(5)))
    res = FA.analyze(oc, family=("H1", "H2"), n_boot=4000)
    h1, h2 = res["tests"]["H1"]["strict"], res["tests"]["H2"]["strict"]
    assert h1["lo"] <= q["Ed"] <= h1["hi"] and abs(h1["est"] - q["Ed"]) < 0.04
    assert h2["lo"] <= q["Ee"] <= h2["hi"]
    assert res["verdict"]["H1"]["claim"] and res["verdict"]["H2"]["claim"]
    st = res["oc"]["strict"]
    assert st["W_silent"] == 0 and st["W_nat"] == h1["wins"] and st["L"] == 0
    c = res["oc"]["cost"]
    n_fired = sum(r["b_fired"] for r in oc)
    assert c["ratio_C_over_B"]["est"] == pytest.approx(0.009 * 400 / (0.012 * n_fired))
    assert c["extra_B_fired"]["est"] == pytest.approx(0.012)


def test_ban_only_flips_cannot_carry_the_claim():
    Q = {**P, "ban": 1.0}  # every failing S0 fails only the strict grader
    res = FA.analyze(SIM.to_rows(SIM.draw(400, Q, np.random.default_rng(6))), n_boot=4000)
    v = res["verdict"]["H1"]
    assert v["p_strict"] <= FA.ALPHA / 2 and v["p_lenient"] == 1.0 and v["p"] == 1.0
    assert not v["reject"] and not v["claim"]
    assert res["tests"]["H1"]["lenient"]["est"] == 0


def test_size_at_null_boundary_and_power():
    # Review bug 6: 2,000 reps (MC SE 0.0035) and a bound of alpha + 3 SE, which a size of 2 alpha cannot pass. H1/H3
    # are conservative (exact-binomial guard); H2's bootstrap test runs at about its nominal size.
    reps = 2000
    sz = SIM.size(200, P, m=3, reps=reps, n_boot=500, seed=11)
    assert sz["Ed"] == pytest.approx(0.03) and sz["Ee"] == pytest.approx(-0.05)
    bound = FA.ALPHA + 3 * math.sqrt(FA.ALPHA * (1 - FA.ALPHA) / reps)
    assert max(sz["H1"], sz["H2"], sz["H3"], sz["fwer"]) <= bound
    assert max(sz["H1"], sz["H3"]) <= FA.ALPHA / 2
    assert SIM.simulate(200, {**P, "g": 0.0}, FA.ALPHA, reps=100, n_boot=1000)["H1"] == 0  # no effect: no rejection
    s = SIM.simulate(240, P, FA.ALPHA / 3, reps=100, n_boot=1000, seed=12)
    assert s["H1"] >= 0.85 and s["cover_d"] >= 0.85 and s["cover_e"] >= 0.85


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_main_end_to_end(tmp_path, capsys):
    oc = SIM.to_rows(SIM.draw(150, P, np.random.default_rng(8)), part="test")
    oc = [{**oc[0], "infra_fail": True}] + oc + [{**oc[1], "id": "SIMX", "infra_fail": True}]
    oc[1]["gate_B"] = oc[1]["B_strict"]
    cc = [{k: v for k, v in r.items() if not k.endswith("_C") and not k.startswith("C_")}
          for r in SIM.to_rows(SIM.draw(80, P, np.random.default_rng(9)), part="test")]
    cur = tmp_path / "curated.json"
    cur.write_text(json.dumps({"tasks": {r["id"]: {"status": "OK"} for r in oc + cc}}))
    out = tmp_path / "res.json"
    fam = ["--family", "H1,H2,H3", "--curated", str(cur)]
    FA.main(["--oc", write(tmp_path / "oc.jsonl", oc), "--cc", write(tmp_path / "cc.jsonl", cc), *fam,
             "--json", str(out)])
    res = json.loads(out.read_text())
    assert set(res["verdict"]) == {"H1", "H2", "H3"} and res["oc"]["strict"]["n"] == 150
    assert res["info"]["oc"]["excluded"] == ["SIMX"] and "C" not in res["cc"]["strict"]
    assert res["oc"]["S8"]["gate_proved"] in (0, 1) and res["oc"]["S8"]["incidents"] == 0
    assert "Confirmatory" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        FA.main(["--oc", str(tmp_path / "oc.jsonl"), *fam])
    with pytest.raises(SystemExit):
        FA.main(["--oc", str(tmp_path / "cc.jsonl"), "--family", "H1,H2", "--curated", str(cur)])
    with pytest.raises(SystemExit, match="curation file"):  # TEST needs the curation file
        FA.main(["--oc", str(tmp_path / "oc.jsonl"), "--family", "H1,H2", "--curated", "none"])


# ---------------------------------------------------------------- regressions for the adversarial review

def test_bug1_c_not_run_restore_failed_exception_egress_excluded_and_retry_used():
    bad = {"C ran no round": nested("LA1", c_rounds=0),
           "C restore failed": nested("LA2", c_restore_ok=False),
           "Harbor exception": nested("LA3", exception="AgentTimeoutError"),
           "egress": nested("LA4", egress="github.com 200\nhuggingface.co 000\natcoder.jp 000"),
           "gate error": nested("LA5", gate=["GATE_ERROR"])}
    fetch = nested("LA6", external_fetch=["curl https://github.com/x"])
    retry = nested("LA1", started_at="2026-10-20T11:00:00", trial="la1__retry")
    rows, info = FA.dedupe([*bad.values(), fetch, retry, nested("LA7")])
    assert [r["id"] for r in rows] == ["LA1", "LA7"] and rows[0]["trial"] == "la1__retry"
    assert info["excluded"] == ["LA2", "LA3", "LA4", "LA5", "LA6"]
    for why, t in bad.items():
        assert why in FA.problems(FA.flatten(t), True), why
    assert info["reasons"]["egress"] == 2 and info["egress"] == ["LA4", "LA6"]
    assert info["egress_rate"] == pytest.approx(2 / 7) and info["infra_rate"] == pytest.approx(5 / 7)
    r = rows[1]  # the fields regrade writes are carried through flatten
    assert (r["c_rounds"], r["c_restore_ok"], r["part"], r["exception"], r["agent"]) == (2, True, "test", None,
                                                                                          "FVBranch")
    assert not FA.egress_hits(r)  # 000 and "blocked" lines are the probe failing, as it must
    assert FA.egress_hits({"egress": ["github.com 403"]}) == ["github.com 403"]  # any answer is egress
    assert FA.egress_hits({"egress": {"atcoder.jp": "000", "github.com": "200"}}) == ["github.com 200"]
    assert FA.problems(FA.flatten(nested("CC1", with_c=False)), False) == []  # Claude Code: no C rules
    # an exception in Harbor's verifier phase (after the snapshots) does not exclude, as in regrade
    assert FA.problems(FA.flatten(nested("LA8", exception="VerifierTimeoutError")), True) == []


def test_usable_rules_match_regrade_source():
    """The status, gate, egress and exception rules are the ones bench/fv/regrade.py writes into `infra`."""
    src = Path(FA.ROOT, "bench", "fv", "regrade.py").read_text()
    ns = {"re": re}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(x, "id", None) in ("VERDICTS", "GATE_OK", "EGRESS",
                                                                           "VERIFIER_EXC") for x in node.targets):
            exec(compile(ast.Module([node], []), "regrade.py", "exec"), ns)
    assert tuple(ns["VERDICTS"]) == FA.VALID and tuple(ns["GATE_OK"]) == FA.GATE_OK
    assert ns["EGRESS"].pattern == FA.EGRESS.pattern and set(ns["VERIFIER_EXC"]) == FA.VERIFIER_EXC


def test_bug2_holm_never_steps_down_past_an_unclaimed_h1():
    def t(ps, pl=None):
        return {"strict": {"p": ps}, "lenient": {"p": ps if pl is None else pl}}
    # strict H1 rejects, lenient does not: H1 is not claimed, so H2 must still face alpha/2, not alpha
    v = FA.verdict({"H1": t(0.001, 0.5), "H2": t(0.02)}, ["H1", "H2"])
    assert not v["H1"]["claim"] and not v["H2"]["claim"] and v["H2"]["crit"] == pytest.approx(FA.ALPHA / 2)
    assert v["H1"]["p"] == 0.5 and v["H1"]["p_strict"] == 0.001 and v["H1"]["p_lenient"] == 0.5
    v = FA.verdict({"H1": t(0.001, 0.004), "H2": t(0.02)}, ["H1", "H2"])  # both graders reject: step down to alpha
    assert v["H1"]["claim"] and v["H2"]["claim"] and v["H1"]["p"] == 0.004
    v = FA.verdict({"H1": t(0.001, 0.004), "H2": t(0.001), "H3": t(0.001, 0.2)}, ["H1", "H2", "H3"])
    assert [v[h]["claim"] for h in ("H1", "H2", "H3")] == [True, True, False]
    assert all(v[h]["claim"] == v[h]["reject"] for h in v)


def test_bug3_check_error_is_unusable_and_counted():
    rows, info = FA.dedupe([nested("LA1", a="CHECK_ERROR"), nested("LA2", c="CHECK_ERROR"), nested("LA3"),
                            nested("LA4", b="TRIVIAL_THEOREMS")])
    assert [r["id"] for r in rows] == ["LA3"] and info["excluded"] == ["LA1", "LA2", "LA4"]
    assert info["check_errors"] == {"A_strict": 1, "A_lenient": 1, "C_strict": 1, "C_lenient": 1}
    assert info["reasons"]["status CHECK_ERROR"] == 2 and info["reasons"]["status TRIVIAL_THEOREMS"] == 1
    # a CHECK_ERROR S0 with a later clean retry: the retry is used
    rows, _ = FA.dedupe([nested("LA1", a="CHECK_ERROR"), nested("LA1", trial="la1__2")])
    assert rows[0]["trial"] == "la1__2" and FA.d_nat(rows[0]) == 1


def test_bug4_dev_test_split_guards(tmp_path):
    cur = tmp_path / "curated.json"
    cur.write_text(json.dumps({"tasks": {"LA1": {"status": "OK"}, "LA2": {"status": "OK"}}}))
    dev = write(tmp_path / "dev.jsonl", [nested("LA1", part="dev"), nested("LA2", part="dev")])
    mixed = write(tmp_path / "mixed.jsonl", [nested("LA1", part="dev"), nested("LA2")])
    test = write(tmp_path / "test.jsonl", [nested("LA1"), nested("LA2")])
    for path in (dev, mixed):
        with pytest.raises(SystemExit, match="part 'test'"):
            FA.load(path, curated=cur)
    with pytest.raises(SystemExit, match="part 'dev'"):
        FA.load(test, part="dev", curated=cur)
    rows, info = FA.load(dev, part="dev", curated=None)  # a DEV dry run may go without the curation file
    assert len(rows) == 2 and info["part"] == "dev"
    with pytest.raises(SystemExit, match="TEST rows"):
        SIM.measure([FA.flatten(nested("LA1"))])
    with pytest.raises(SystemExit, match="part 'dev'"):
        SIM.main(["--from", test, "--n", "50", "--reps", "2"])


def test_curation_drops_trivial_and_uncurated_tasks():
    cur = {"LA1": "OK", "LA2": "TRIVIAL_THEOREMS", "LA3": "VACUOUS"}
    trials = [nested(i) for i in ("LA1", "LA2", "LA3", "LA4")]
    rows, info = FA.dedupe(trials, cur, need_curated=True)
    assert [r["id"] for r in rows] == ["LA1"] and info["excluded"] == [] and info["ids"] == 1
    assert info["curation"] == {"TRIVIAL_THEOREMS": 1, "VACUOUS": 1, "not curated": 1}
    rows, info = FA.dedupe(trials, cur)  # DEV: tasks curation has not reached yet stay
    assert [r["id"] for r in rows] == ["LA1", "LA4"] and info["infra_rate"] == 0


def test_bug5_measure_nat_and_g_on_the_fired_population():
    # stop reasons differ between fired and unfired failing S0s: nat must come from the fired ones
    rows = ([{"id": f"F{i}", "A_strict": "UNPROVED", "A_lenient": "UNPROVED", "b_fired": True,
              "stop_reason": "end_turn", "B_strict": "PROVED" if i < 4 else "UNPROVED"} for i in range(10)]
            + [{"id": f"O{i}", "A_strict": "UNPROVED", "A_lenient": "UNPROVED", "b_fired": True,
                "stop_reason": "length", "B_strict": "PROVED" if i < 1 else "UNPROVED"} for i in range(10)]
            + [{"id": f"U{i}", "A_strict": "UNPROVED", "A_lenient": "UNPROVED", "b_fired": False,
                "stop_reason": "length", "B_strict": "UNPROVED"} for i in range(20)])
    m = SIM.measure(rows)
    assert (m["nat"], m["g"], m["g_other"], m["f"]) == (0.5, 0.4, 0.1, 0.5)
    # planted E[d_nat] and P(B repairs) match what a simulated job shows, also with phi < 1 and g_other != g
    Q = {**P, "p0": 0.3, "f": 0.35, "g": 0.4, "g_other": 0.05, "nat": 0.6}
    q, x = SIM.probs(Q), SIM.draw(200000, Q, np.random.default_rng(13))
    assert q["phi"] < 1
    d, _ = SIM.outcomes(x)
    assert d.mean() == pytest.approx(q["Ed"], abs=0.003)
    assert (x["B"] & ~x["A"]).sum() / (~x["A"]).sum() == pytest.approx(q["pib"], abs=0.004)
    m = SIM.measure(SIM.to_rows(SIM.draw(40000, Q, np.random.default_rng(14))))
    assert SIM.probs({**P, **m})["Ed"] == pytest.approx(q["Ed"], abs=0.006)


def test_bug7_results_do_not_depend_on_file_order_and_retries_go_by_start_time():
    rows = SIM.to_rows(SIM.draw(120, P, np.random.default_rng(15)))
    a = FA.analyze(rows, n_boot=2000)
    b = FA.analyze(rows[::-1], n_boot=2000)
    assert a["tests"]["H1"]["strict"] == b["tests"]["H1"]["strict"]
    assert a["tests"]["H2"]["strict"] == b["tests"]["H2"]["strict"]
    late = nested("LA1", b="UNPROVED", trial="la1__aaa", started_at="2026-10-20T12:00:00")
    early = nested("LA1", trial="la1__zzz", started_at="2026-10-20T09:00:00")
    for order in ([late, early], [early, late]):
        rows, info = FA.dedupe(order)
        assert rows[0]["trial"] == "la1__zzz" and info["duplicates"] == ["LA1"] and info["unordered"] == []
    rows, info = FA.dedupe([late, {**early, "started_at": None}])  # no start time: file order, and it is reported
    assert rows[0]["trial"] == "la1__aaa" and info["unordered"] == ["LA1"]


def test_bug8_s5_counts_attempts_in_every_snapshot():
    t = nested("LA1")
    t["snaps"] = {"S0": {"strict": "UNPROVED"}, "B1": {"strict": "CHEAT_REJECTED"}, "B2": {"strict": "PROVED"},
                  "C1": {"strict": "SPEC_TAMPERED"}, "C2": {"strict": None}}  # C2: a snapshot that failed integrity
    t["B"]["label"] = "B2"
    rows, _ = FA.dedupe([t, nested("LA2")])
    s5 = FA.analyze_job(rows, True, n_boot=200)
    assert s5["S5"]["B"]["final"] == {"PROVED": 2} and s5["S5"]["B"]["any_snapshot"]["CHEAT_REJECTED"] == 1
    assert s5["S5_examples"]["B"]["CHEAT_REJECTED"] == ["LA1"] and s5["S5_examples"]["A"]["CHEAT_REJECTED"] == []
    assert s5["S5_examples"]["C"]["SPEC_TAMPERED"] == ["LA1"]
