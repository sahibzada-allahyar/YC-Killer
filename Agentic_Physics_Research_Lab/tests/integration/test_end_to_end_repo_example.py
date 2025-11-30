import os
from pathlib import Path
import importlib
import pytest

@pytest.mark.optional
def test_repo_example_end_to_end():
    cfg = Path("examples/simple_experiment/config.yml")
    if not cfg.exists():
        pytest.skip("examples/simple_experiment/config.yml not found in repo")

    try:
        mod = importlib.import_module("grantlab.pipeline")
    except Exception:
        pytest.skip("grantlab.pipeline not importable")

    run_pipeline = getattr(mod, "run_pipeline", None)
    if run_pipeline is None:
        pytest.skip("grantlab.pipeline.run_pipeline not available")

    # Run the pipeline; it writes outputs to examples/simple_experiment/outputs
    run_pipeline(cfg)

    out_dir = Path("examples/simple_experiment/outputs")
    reports = list(out_dir.glob("run_*/report.md"))
    assert reports, "No report generated under examples/simple_experiment/outputs/run_*/report.md"
