from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats
from ..config import Config

def _randomize_groups(df: pd.DataFrame, plan: Dict[str, Any], id_col: str, group_col: str) -> pd.DataFrame:
    rng = np.random.default_rng(plan["seed"])
    shuffled = df.sample(frac=1.0, random_state=plan["seed"]).reset_index(drop=True)
    n = plan["n_per_group"]
    gA, gB = plan["group_a"], plan["group_b"]
    labels = [gA]*n + [gB]*n
    if len(labels) > len(shuffled):
        raise ValueError("Not enough rows to assign n_per_group per group.")
    shuffled.loc[:len(labels)-1, group_col] = labels
    return shuffled

def _extract_groups(df: pd.DataFrame, outcome: str, group_col: str, gA: str, gB: str):
    a = df[df[group_col] == gA][outcome].dropna().values
    b = df[df[group_col] == gB][outcome].dropna().values
    return a, b

def _effect_size_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    # Pooled SD per Cohen's d (unequal n)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
    return (np.mean(b) - np.mean(a)) / sp if sp and not np.isclose(sp, 0.0) else np.nan

def _ci_diff_means_welch(a: np.ndarray, b: np.ndarray, alpha: float):
    # Welch-Satterthwaite CI
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va/na + vb/nb)
    # df
    df = (va/na + vb/nb)**2 / ((va**2)/((na**2)*(na-1)) + (vb**2)/((nb**2)*(nb-1)))
    tcrit = stats.t.ppf(1 - alpha/2, df)
    diff = mb - ma
    return (diff - tcrit*se, diff + tcrit*se)

def run_experiment(cfg: Config, df: pd.DataFrame, hypothesis: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    group_col = cfg.group_column or "_assigned_group"
    if (cfg.group_column is None) or (cfg.group_column == "") or (group_col not in df.columns):
        df = df.copy()
        df[group_col] = None
        df = _randomize_groups(df, plan, cfg.id_column, group_col)

    a, b = _extract_groups(df, cfg.outcome_column, group_col, plan["group_a"], plan["group_b"])
    test = plan["test"]
    if test in ("t", "welch_t"):
        equal_var = (test == "t")
        tstat, p = stats.ttest_ind(b, a, equal_var=equal_var, nan_policy="omit")
    elif test == "mannwhitney":
        u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
        tstat = np.nan
    else:
        raise ValueError(f"Unknown test: {test}")

    es = _effect_size_cohens_d(a, b)
    ci_low, ci_high = _ci_diff_means_welch(a, b, hypothesis["alpha"])
    result = {
        "group_a": plan["group_a"],
        "group_b": plan["group_b"],
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "test": test,
        "t_stat": float(tstat),
        "p_value": float(p),
        "effect_size_d": float(es) if es is not None else None,
        "ci_diff": [float(ci_low), float(ci_high)],
        "assigned_group_column": group_col,
    }
    return result
