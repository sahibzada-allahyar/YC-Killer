"""FV pool/split invariants. Hermetic union-find tests always run; the rest check runs/fv/{pool,split}.json and skip
when those gitignored files (or vericoding.jsonl) are absent."""
import hashlib
import json

import pytest

from bench.fv import pool as P
from bench.fv import split as S

built = pytest.mark.skipif(not (S.POOL.exists() and S.OUT.exists()), reason="run bench.fv.pool and bench.fv.split")
has_data = pytest.mark.skipif(not (P.DATA.exists() and P.VERICODED.exists() and P.E4B_LOG.exists()),
                              reason="vericoding inputs absent")


def row(i, lang, source, sid, dup=""):
    return {"id": i, "language": lang, "source": source, "source-id": sid, "qa-near-duplicate-group": dup}


def test_numpy_names():
    def name(src, sid):
        return P.numpy_name(row("X", "lean", src, sid))
    assert name("numpy_simple", "NpCumSum-spec") == name("numpy_triple", "mathematical_functions_cumsum") == "cumsum"
    assert name("numpy_simple", "NpAbs-spec") == name("numpy_triple", "mathematical_functions_absolute")
    assert name("numpy_simple", "NpCenter-spec") == name("numpy_triple", "strings_center") == "strings.center"
    assert name("numpy_triple", "polynomial_hermite_e_hermeadd") == "hermeadd"
    assert name("numpy_triple", "statistics_amax") == name("numpy_triple", "statistics_max")
    assert name("numpy_triple", "strings_add") != name("numpy_triple", "mathematical_functions_add")


def test_components_union_find():
    comp = P.components([
        row("DT0001", "dafny", "numpy_triple", "mathematical_functions_cumsum"),
        row("LS0005", "lean", "numpy_simple", "NpCumSum-spec", "DupLS00"),
        row("LS0009", "lean", "numpy_simple", "NpSort-spec", "DupLS00"),
        row("VT0003", "verus", "numpy_triple", "sorting_searching_sort"),
        row("DT0100", "dafny", "numpy_triple", "strings_add"),
        row("DT0101", "dafny", "numpy_triple", "mathematical_functions_add"),
        row("LA0001", "lean", "apps", "apps_test_7"),
        row("DA0001", "dafny", "apps", "apps_test_7"),
        row("VA0002", "verus", "apps", "apps_test_8"),
    ])
    assert comp["DT0001"] == comp["LS0005"] == comp["LS0009"] == comp["VT0003"] == "DT0001"
    assert comp["LA0001"] == comp["DA0001"] == "DA0001"
    assert comp["VA0002"] == "VA0002" and comp["DT0100"] != comp["DT0101"]


def test_is_dev_rate_and_forcing():
    keys = [f"K{i:05d}" for i in range(20000)]
    rate = sum(map(S.is_dev, keys)) / len(keys)
    assert 0.28 < rate < 0.32
    test_key = next(k for k in keys if not S.is_dev(k))
    assert S.is_dev(test_key, frozenset({test_key}))
    assert [S.is_dev(k) for k in keys[:50]] == [S.is_dev(k) for k in keys[:50]]


@pytest.fixture(scope="module")
def files():
    raw = S.POOL.read_bytes()
    return json.loads(raw), json.loads(S.OUT.read_text()), hashlib.sha256(raw).hexdigest()


@built
def test_split_is_deterministic_and_current(files):
    pool, split, sha = files
    assert split["pool_sha256"] == sha and split["seed"] == 20261012 and split["salt"] == "fv-split-v2|"
    assert S.build(pool, sha) == S.build(pool, sha) == split


@built
def test_no_component_straddles_dev_and_test(files):
    pool, split, _ = files
    forced = frozenset(pool["e4b"]["components"])
    part = {}
    for name in ("dev", "test"):
        for cell, comps in split[name].items():
            for c in comps:
                assert part.setdefault(c["comp"], name) == name, c["comp"]
                assert S.is_dev(c["comp"], forced) == (name == "dev")
                assert all(pool["component"][i] == c["comp"] for i in c["ids"])
    for cell, ids in pool["cells"].items():
        listed = [i for name in ("dev", "test") for c in split[name][cell] for i in c["ids"]]
        assert sorted(listed) == ids  # every pool task is in exactly one part
        assert split["dev_ids"][cell] == [i for c in split["dev"][cell] for i in c["ids"]]
    assert all(part.get(c, "dev") == "dev" for c in forced)


@built
def test_e4b_tasks_out_of_cells(files):
    pool, _, _ = files
    assert len(pool["e4b"]["ids"]) == 60
    cell_ids = {i for ids in pool["cells"].values() for i in ids}
    assert not cell_ids & set(pool["e4b"]["ids"])


@built
@has_data
def test_no_solved_component_in_any_cell(files):
    pool, _, _ = files
    rows = P.load()
    assert P.build(rows, P.vericoded_ids()) == pool  # pool.json is current and deterministic
    comp, solved_ids = P.components(rows), P.vericoded_ids()
    solved = {comp[i] for i in solved_ids}
    for cell, ids in pool["cells"].items():
        assert not set(ids) & solved_ids, cell
        assert not {comp[i] for i in ids} & solved, cell
    by_id = {r["id"]: r for r in rows}
    for cell, (lang, sources) in P.CELLS.items():
        for i in pool["cells"][cell]:
            r = by_id[i]
            assert r["language"] == lang and r["source"] in sources and r["qa-issue"] == 0
            assert r["source"] not in P.EXCLUDED_SOURCES


@built
def test_counts_sane(files):
    pool, split, _ = files
    for cell, ids in pool["cells"].items():
        n, c = pool["counts"][cell], split["counts"][cell]
        assert ids and n["tasks"] == len(ids) == len(set(ids)) >= n["components"]
        assert c["dev_components"] + c["test_components"] == n["components"]
        assert c["dev_tasks"] + c["test_tasks"] == n["tasks"]
        if n["components"] >= 20:
            assert 0.15 <= c["dev_components"] / n["components"] <= 0.45, cell
    assert pool["counts"]["L-fv"]["tasks"] == 978  # guarded_and_plausible, qa-issue==0 (FINDINGS 2026-10-11)
    assert "LF0023" not in pool["cells"]["L-fv"]  # FVAPPS human baseline


@built
def test_pick_test_one_task_per_component(files):
    _, split, _ = files
    cells = list(split["test"])
    for subset in (cells, ["L-apps", "D-apps"], ["L-np", "D-np"], cells[:1]):
        picked = S.pick_test(split, subset)
        assert picked == S.pick_test(split, subset)
        comp_of = {i: (cell, c["comp"]) for cell in subset for c in split["test"][cell] for i in c["ids"]}
        seen = [comp_of[i][1] for cell, ids in picked.items() for i in ids]
        assert len(seen) == len(set(seen))  # at most 1 task per TEST component
        assert all(comp_of[i][0] == cell for cell, ids in picked.items() for i in ids)
        assert set(seen) == {c["comp"] for cell in subset for c in split["test"][cell]}
    dev = {i for ids in split["dev_ids"].values() for i in ids}
    assert not dev & {i for ids in S.pick_test(split, cells).values() for i in ids}
