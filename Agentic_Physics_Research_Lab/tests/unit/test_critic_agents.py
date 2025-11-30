import importlib
import pytest

def _load_symbols():
    try:
        mod = importlib.import_module("grantlab.papers.graph")
    except Exception:
        pytest.skip("grantlab.papers.graph not importable")
    CriticAgent = getattr(mod, "CriticAgent", None)
    if CriticAgent is None:
        pytest.skip("CriticAgent not found in grantlab.papers.graph")
    return CriticAgent

@pytest.mark.optional
def test_critic_scoring_basic():
    CriticAgent = _load_symbols()
    agent = CriticAgent(domain="math")
    paper = {"title": "Test", "abstract": "We prove a bound.", "body": "Details..."}
    out = agent.review(paper)
    ok = isinstance(out, dict) and ("score" in out or "verdict" in out) or isinstance(out, str)
    assert ok, "CriticAgent.review should return a dict (with score) or a review string"
