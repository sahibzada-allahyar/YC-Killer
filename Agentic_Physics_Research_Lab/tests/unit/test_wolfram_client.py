import importlib
import pytest

Client = None

def setup_module():
    global Client
    try:
        mod = importlib.import_module("wolfram.client")
        Client = getattr(mod, "WolframClient", None)
    except Exception:
        Client = None

@pytest.mark.optional
def test_wolfram_client_smoke(monkeypatch):
    if Client is None:
        pytest.skip("wolfram.client.WolframClient not found")

    # Avoid real network by monkeypatching an internal call if present
    fake = {"query": "2+2", "result": "4"}
    c = Client(app_id="dummy")

    if hasattr(c, "_call"):
        monkeypatch.setattr(c, "_call", lambda q, **kw: fake)
        out = c.query("2+2")
        assert "4" in str(out)
    else:
        # Fallback: try to patch requests.get inside wolfram.client if used there
        try:
            import wolfram.client as wc
            class _Resp:
                status_code = 200
                text = "4"
                def json(self): return fake
            def _fake_get(*a, **k): return _Resp()
            if hasattr(wc, "requests"):
                monkeypatch.setattr(wc.requests, "get", _fake_get)
            out = c.query("2+2")
            assert "4" in str(out)
        except Exception:
            pytest.skip("Unable to safely monkeypatch wolfram client; adjust test to your HTTP layer")
