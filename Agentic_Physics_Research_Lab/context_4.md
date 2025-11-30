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
- [x] Copied

---

## `tests/conftest.py`
- [x] Copied

---

## Unit tests

### `tests/unit/test_env_api.py`
- [x] Copied



### `tests/unit/test_delegation_budget.py`
- [x] Copied



### `tests/unit/test_wolfram_client.py`
- [x] Copied



### `tests/unit/test_critic_agents.py`
- [x] Copied



### `tests/unit/test_paper_graph_adaptive.py`
- [x] Copied



### `tests/unit/test_lean_templates.py`
- [x] Copied



### `tests/unit/test_benchmarks_registry.py`
- [x] Copied



### `tests/unit/test_config_schema.py`
- [x] Copied



---

## Integration tests

### `tests/integration/test_end_to_end_repo_example.py`
- [x] Copied



### `tests/integration/test_wolfram_live.py`
- [x] Copied



### `tests/integration/test_mcp_costing.py`
- [x] Copied



---

## GitHub Actions (optional but recommended)

### `.github/workflows/tests.yml`
- [x] Copied



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
