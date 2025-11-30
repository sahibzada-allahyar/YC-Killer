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
