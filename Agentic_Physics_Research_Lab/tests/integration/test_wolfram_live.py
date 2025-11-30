import os
import importlib
import pytest

@pytest.mark.network
def test_wolfram_live_addition():
    try:
        mod = importlib.import_module("wolfram.client")
    except Exception:
        pytest.skip("wolfram.client not importable")

    Client = getattr(mod, "WolframClient", None)
    if Client is None:
        pytest.skip("WolframClient class missing")

    app_id = os.environ.get("WOLFRAM_APP_ID")
    if not app_id:
        pytest.skip("WOLFRAM_APP_ID not set")
    c = Client(app_id=app_id)
    out = c.query("2+2")
    assert "4" in str(out)
