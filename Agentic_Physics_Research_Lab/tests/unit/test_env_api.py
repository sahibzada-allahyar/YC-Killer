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
