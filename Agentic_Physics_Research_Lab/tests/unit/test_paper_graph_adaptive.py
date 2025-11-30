import importlib
import pytest

def _load_symbols():
    try:
        mod = importlib.import_module("grantlab.papers.graph")
    except Exception:
        pytest.skip("grantlab.papers.graph not importable")
    build_graph = getattr(mod, "build_graph", None)
    adaptive_update = getattr(mod, "adaptive_plan_update", None)
    if build_graph is None or adaptive_update is None:
        pytest.skip("build_graph/adaptive_plan_update not exposed in grantlab.papers.graph")
    return build_graph, adaptive_update

@pytest.mark.optional
def test_adaptive_plan_update_on_wide_posteriors():
    build_graph, adaptive_update = _load_symbols()
    G = build_graph()
    updated, new_nodes = adaptive_update(G, posterior_width=0.42)
    assert isinstance(updated, bool)
    if updated:
        assert (isinstance(new_nodes, list) or new_nodes is None)
