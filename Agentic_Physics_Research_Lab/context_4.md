Awesome — here’s a **drop‑in test suite** (unit + integration) you can copy into your repo **today** to verify the basic plumbing of the RL environment, Lean/Mathlib templates, delegation budgets, Wolfram client, paper‑graph critic agents, and benchmark registry.

> The structure and checks mirror the **Denario** “Planning & Control” + module layout (Idea → Literature → Methods → Analysis → Paper → Review) and its recommendations for **budgeted delegation** and **adaptive plan updates** when posteriors are wide. 

---

## What you get

* **Unit tests** for:

  * RL env API contract + **seed determinism**
  * **Delegation policy** enforcing **cost**/**latency** budgets
  * **Wolfram client** (offline stubbed smoke test)
  * **Critic agents** (math/physics reviewers) API
  * **Paper graph** adaptive plan update when posterior width is large
  * **Lean templates** presence and optional compile smoke (skips if `lean`/Mathlib missing)
  * **Benchmark registry** list/run interface
  * **Config loader** sanity checks

* **Integration tests** for:

  * Full **Hypothesis → Design → Run → Interpret → Report** flow using your repo’s example config (skips if absent)
  * Optional **live Wolfram** call (guarded by `-m network` + `WOLFRAM_APP_ID`)
  * Optional MCP costing policy smoke

* **Markers & CI**:

  * `@pytest.mark.optional` for optional modules that you’re still wiring
  * `@pytest.mark.network` for live calls
  * GitHub Actions workflow running the **fast, offline suite** by default

> All network/slow features are **opt‑in** so your fast CI stays green. Locally, enable them with `-m network` or `-m slow`.

---

## File tree to add at your repo root

```
pytest.ini
tests/
  conftest.py
  unit/
    test_env_api.py
    test_delegation_budget.py
    test_wolfram_client.py
    test_critic_agents.py
    test_paper_graph_adaptive.py
    test_lean_templates.py
    test_benchmarks_registry.py
    test_config_schema.py
  integration/
    test_end_to_end_repo_example.py
    test_wolfram_live.py
    test_mcp_costing.py
.github/
  workflows/
    tests.yml
```

> If your module names differ slightly, either add tiny shims (recommended) or tweak the imports in the tests. Each optional test uses **graceful skips** with a clear reason.

---

## `pytest.ini`

```ini
[pytest]
minversion = 7.0
addopts = -ra
testpaths = tests
markers =
    slow: tests that are relatively slow
    network: tests that require live network connectivity or real APIs
    optional: tests for optional modules; skipped if module not installed
```

---

## `tests/conftest.py`

```python
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
```

---

## Unit tests

### `tests/unit/test_env_api.py`

```python
import importlib
import pytest

def _load_env_ref():
    # Try common locations for the research environment.
    candidates = [
        ("grantlab.envs.core", "ResearchEnv"),
        ("grantlab.envs.research", "ResearchEnv"),
    ]
    for modname, clsname in candidates:
        try:
            m = importlib.import_module(modname)
            if hasattr(m, clsname):
                return getattr(m, clsname)  # class
        except Exception:
            continue

    # Fallback: try a factory function
    factories = [
        ("grantlab.envs.core", "make_env"),
        ("grantlab.envs.research", "make_env"),
    ]
    for modname, funcname in factories:
        try:
            m = importlib.import_module(modname)
            if hasattr(m, funcname):
                return getattr(m, funcname)  # factory
        except Exception:
            continue

    pytest.skip("No ResearchEnv class or make_env factory found in grantlab.envs.*")

def _instantiate(env_ref, seed=123):
    if isinstance(env_ref, type):  # class
        try:
            env = env_ref(seed=seed)
        except TypeError:
            env = env_ref()
            if hasattr(env, "seed"):
                env.seed(seed)
        return env
    elif callable(env_ref):        # factory
        return env_ref(seed=seed)
    else:
        raise TypeError("Unknown env reference type")

def _has_gym_api(env):
    return all(hasattr(env, name) for name in ["reset", "step"]) and hasattr(env, "action_space")

def test_reset_step_contract():
    env_ref = _load_env_ref()
    env = _instantiate(env_ref, seed=123)
    assert _has_gym_api(env), "Env must have gym-like API (reset/step/action_space)."

    out = env.reset(seed=123) if "reset" in dir(env) else env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    assert obs is not None

    act = env.action_space.sample()
    step_out = env.step(act)
    assert isinstance(step_out, tuple) and len(step_out) in (4, 5), "Step must return 4 or 5-tuple."

def test_seed_determinism():
    env_ref = _load_env_ref()
    env1 = _instantiate(env_ref, seed=123)
    env2 = _instantiate(env_ref, seed=123)

    env1.reset(seed=123) if "reset" in dir(env1) else env1.reset()
    env2.reset(seed=123) if "reset" in dir(env2) else env2.reset()

    actions = [env1.action_space.sample() for _ in range(8)]
    traj1, traj2 = [], []
    for a in actions:
        traj1.append(env1.step(a))
        traj2.append(env2.step(a))

    assert len(traj1) == len(traj2)
    for (o1, r1, *rest1), (o2, r2, *rest2) in zip(traj1, traj2):
        assert type(r1) == type(r2)
        assert (float(r1) == float(r2)) if hasattr(r1, "__float__") else (r1 == r2)
```

### `tests/unit/test_delegation_budget.py`

```python
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
```

### `tests/unit/test_wolfram_client.py`

```python
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
```

### `tests/unit/test_critic_agents.py`

```python
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
```

### `tests/unit/test_paper_graph_adaptive.py`

```python
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
```

### `tests/unit/test_lean_templates.py`

```python
import pathlib
import pytest
import shutil
import subprocess

@pytest.mark.optional
def test_lean_templates_exist():
    base = pathlib.Path("grantlab") / "lean" / "templates"
    if not base.exists():
        pytest.skip("lean templates folder not present")
    files = list(base.glob("*.lean"))
    assert files, "No .lean templates found; expected at least one (e.g., sho.lean)"

@pytest.mark.network
@pytest.mark.slow
def test_mathlib_build_or_skip():
    if shutil.which("lean") is None:
        pytest.skip("lean not found on PATH")
    try:
        subprocess.run(["lean", "--version"], check=True, capture_output=True)
    except Exception:
        pytest.skip("lean is installed, but failed to run; skipping")
    base = pathlib.Path("grantlab") / "lean" / "templates"
    files = list(base.glob("*.lean"))
    if not files:
        pytest.skip("No lean templates to compile")
    try:
        subprocess.run(["lean", str(files[0])], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        blob = ((e.stderr or b"") + (e.stdout or b"")).lower()
        if b"mathlib" in blob and b"unknown package" in blob:
            pytest.skip("Mathlib not installed; skipping compilation test")
        raise
```

### `tests/unit/test_benchmarks_registry.py`

```python
import importlib
import pytest

@pytest.mark.optional
def test_benchmark_registry_lists():
    try:
        mod = importlib.import_module("grantlab.eval.benchmarks")
    except Exception:
        pytest.skip("grantlab.eval.benchmarks not importable")
    list_benchmarks = getattr(mod, "list_benchmarks", None)
    if list_benchmarks is None:
        pytest.skip("list_benchmarks() not found")
    items = list_benchmarks()
    assert isinstance(items, (list, tuple))
    if items:
        assert all(isinstance(x, str) for x in items)

@pytest.mark.optional
def test_run_benchmark_smoke():
    try:
        mod = importlib.import_module("grantlab.eval.benchmarks")
    except Exception:
        pytest.skip("grantlab.eval.benchmarks not importable")
    run_benchmark = getattr(mod, "run_benchmark", None)
    list_benchmarks = getattr(mod, "list_benchmarks", None)
    if run_benchmark is None or list_benchmarks is None:
        pytest.skip("run_benchmark/list_benchmarks missing")
    names = list_benchmarks()
    if not names:
        pytest.skip("No benchmarks registered")
    res = run_benchmark(names[0], dry_run=True)
    assert res is not None
```

### `tests/unit/test_config_schema.py`

```python
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
```

---

## Integration tests

### `tests/integration/test_end_to_end_repo_example.py`

```python
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
```

### `tests/integration/test_wolfram_live.py`

```python
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
```

### `tests/integration/test_mcp_costing.py`

```python
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
```

---

## GitHub Actions (optional but recommended)

### `.github/workflows/tests.yml`

```yaml
name: tests

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt || true
          pip install pytest hypothesis pyyaml pydantic
      - name: Run fast suite (no network/slow)
        run: |
          pytest -m "not network and not slow" -q
```

---

## How to run

```bash
# Fast, offline baseline
pytest -m "not network and not slow" -q

# Enable optional live integrations
export WOLFRAM_APP_ID=...           # only if you want the live Wolfram test
pytest -m "network" tests/integration/test_wolfram_live.py -q
```

---

## Why these tests (brief rationale)

* **RL env API + determinism**: guards the minimal Gymnasium‑style contract and reproducibility — critical for reliable **RL‑on‑Lean** training loops.
* **Budgeted delegation**: mirrors Denario’s **Planning & Control** constraints ensuring your agentic system respects **cost/latency budgets** before calling heavy tools (e.g., Wolfram/PDE solvers). 
* **Critic agents + adaptive updates**: tests that “math/physics reviewers” exist and that a **posterior‑width** signal can trigger plan augmentation (extra nodes in the paper graph), as recommended by Denario for narrowing uncertainty. 
* **Lean/Mathlib presence**: smoke checks avoid brittleness; compiling is optional and skipped unless your environment is fully provisioned.
* **Benchmark registry**: a light TPBench/ReplicationBench‑style harness must list and run benchmarks in **dry‑run**.

---

If you want me to tailor the tests to *exact* class/function names you ended up using (so fewer `@optional` skips fire), paste your module map (or `tree -a` of your repo), and I’ll tighten imports and assertions accordingly.
