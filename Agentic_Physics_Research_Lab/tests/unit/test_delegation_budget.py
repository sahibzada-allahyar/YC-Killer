import importlib
import pytest
from dataclasses import dataclass

Policy = None

def setup_module():
    global Policy
    try:
        mod = importlib.import_module("grantlab.policy.delegation")
        Policy = getattr(mod, "DelegationPolicy", None)
    except Exception:
        Policy = None

@pytest.mark.optional
def test_budget_enforcement_basic():
    if Policy is None:
        pytest.skip("grantlab.policy.delegation.DelegationPolicy not found")

    policy = Policy(cost_budget=1.0, latency_budget_ms=200, trace=True)

    @dataclass
    class Task:
        name: str
        cost: float
        latency_ms: int

    assert policy.allow(Task("small", cost=0.2, latency_ms=50))
    assert not policy.allow(Task("big_cost", cost=0.9, latency_ms=50))

    policy = Policy(cost_budget=10.0, latency_budget_ms=100, trace=True)
    assert policy.allow(Task("fast", cost=0.1, latency_ms=20))
    assert not policy.allow(Task("slow", cost=0.1, latency_ms=200))
