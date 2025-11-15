from rl_science.envs.scientific_env import ScientificEnv, Action

def test_env_happy_path(tmp_path, monkeypatch):
    env = ScientificEnv(max_steps=10)
    obs = env.reset(data_description=str(tmp_path / "data.csv"))
    # Without real data, we still exercise pipeline
    obs, r, done, info = env.step(Action.CALL_IDEA_MAKER)
    obs, r, done, info = env.step(Action.RUN_METHODS)
    obs, r, done, info = env.step(Action.RUN_ANALYSIS)  # may warn if CSV not found
    obs, r, done, info = env.step(Action.WRITE_PAPER)
    obs, r, done, info = env.step(Action.RUN_REVIEW)
    assert obs["paper_ready"] is True or obs["review_ready"] is True
