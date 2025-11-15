from rl_science.utils.exec_sandbox import run_python_code

def test_run_python_code():
    code = "print('hello'); import matplotlib.pyplot as plt; plt.figure(); plt.plot([1,2],[3,4]); plt.savefig('plots/a.png')"
    out, err, art = run_python_code(code)
    assert "hello" in out
    assert len(art.get("plots", [])) >= 1
