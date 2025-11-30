import importlib
import pytest

@pytest.mark.optional
def test_mcp_costing_policy_smoke():
    try:
        mod = importlib.import_module("grantlab.mcp.policy")
    except Exception:
        pytest.skip("grantlab.mcp.policy not importable")
    build = getattr(mod, "build_mcp_delegation_policy", None)
    if build is None:
        pytest.skip("build_mcp_delegation_policy not found")
    p = build(cost_budget=0.5, latency_budget_ms=1000)
    assert p is not None
