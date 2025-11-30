import os
import random
import pathlib
import numpy as np
import pytest

SEED = int(os.environ.get("GRANTLAB_TEST_SEED", "1337"))

@pytest.fixture(autouse=True, scope="session")
def _seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass
    yield

@pytest.fixture
def tmp_project_dir(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    return d

def import_or_skip(modname, reason=None):
    try:
        module = __import__(modname, fromlist=["*"])
        return module
    except Exception as e:
        pytest.skip(reason or f"Optional dependency '{modname}' not importable: {e}")
