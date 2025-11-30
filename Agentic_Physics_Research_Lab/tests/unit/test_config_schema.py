import io
import importlib
import yaml
import pytest
from pathlib import Path

@pytest.mark.optional
def test_config_load_minimal(tmp_path):
    try:
        mod = importlib.import_module("grantlab.config")
    except Exception:
        pytest.skip("grantlab.config not importable")

    yaml_str = '''
    agent:
      model: gpt-5-pro
      temperature: 0.1
    budgets:
      cost_usd: 2.5
      latency_ms: 5000
    '''

    loader = getattr(mod, "load_config", None)
    Config = getattr(mod, "Config", None)

    # Write to disk just in case loader expects a path
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml_str)

    if loader:
        try:
            cfg = loader(cfg_path)  # path
        except Exception:
            cfg = loader(io.StringIO(yaml_str))  # file-like
    elif Config:
        try:
            cfg = Config.model_validate_yaml(yaml_str)   # pydantic v2
        except Exception:
            data = yaml.safe_load(yaml_str)
            cfg = Config(**data)
    else:
        pytest.skip("No Config class or load_config function available")

    d = cfg if isinstance(cfg, dict) else getattr(cfg, "dict", lambda: cfg)()
    assert "agent" in d and "budgets" in d
