import shutil
from pathlib import Path
from grantlab.lean_agent.lean_agent import LeanAgent

def test_lean_agent(tmp_path: Path):
    agent = LeanAgent(tmp_path)
    hyp = {"type":"mean_difference", "metric":"conversion", "group_a":"control", "group_b":"treatment", "direction":"treatment > control", "alpha":0.05}
    plan = {"group_a":"control","group_b":"treatment","n_per_group":20,"assignment":"random","seed":0,"test":"welch_t"}
    res = {"n_a":20,"n_b":20}
    summary = agent.verify(hyp, plan, res)
    assert "file" in summary
    if shutil.which("lean"):
        assert summary["available"] is True
    else:
        assert summary["available"] is False
