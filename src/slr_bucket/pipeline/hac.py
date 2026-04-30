"""
hac.py — HAC regression guard for the SLR event study.

Provides pre-flight design-matrix checks and a safe OLS+HAC estimator.
Guards against rank deficiency, near-singular matrices, and low
observation-to-parameter ratios that would invalidate Newey-West inference.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


MIN_OBS_PER_PARAM  = 10   # HAC standard: at least 10 obs per parameter
MAX_CONDITION_NUMBER = 1e10  # above this, design matrix is near-singular


class RankDeficiencyError(ValueError):
    pass


def check_design_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    min_obs_per_param: int = MIN_OBS_PER_PARAM,
    max_condition: float   = MAX_CONDITION_NUMBER,
    raise_on_fail: bool    = False,
) -> dict:
    """
    Check design matrix for rank deficiency, obs/param ratio, and collinearity.
    Returns a diagnostics dict.
    """
    n, k   = X.shape
    issues = []

    obs_param_ratio = n / k if k > 0 else 0
    if obs_param_ratio < min_obs_per_param:
        issues.append(
            f"Low obs/param ratio: {n} obs / {k} params = {obs_param_ratio:.1f} "
            f"(minimum recommended: {min_obs_per_param})"
        )

    try:
        rank = np.linalg.matrix_rank(X.values.astype(float))
        if rank < k:
            issues.append(f"Rank deficiency: rank={rank} < k={k} (dropped {k-rank} columns)")
    except Exception:
        rank = np.nan

    try:
        cond = float(np.linalg.cond(X.values.astype(float)))
        if cond > max_condition:
            issues.append(f"Near-singular: condition number = {cond:.2e}")
    except Exception:
        cond = np.nan

    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if const_cols:
        issues.append(f"Zero-variance columns: {const_cols}")

    # Pooled regression check: post:treasury_based must vary
    if "treasury_based" in X.columns or any("treasury_based" in c for c in X.columns):
        tb_cols = [c for c in X.columns if "treasury_based" in c and "post" in c]
        for col in tb_cols:
            if X[col].nunique(dropna=True) <= 1:
                issues.append(
                    f"post:treasury_based is constant — pooled panel must include "
                    f"both treasury_based=1 and treasury_based=0 series."
                )

    result = {
        "n": n, "k": k, "rank": rank,
        "obs_param_ratio":  obs_param_ratio,
        "condition_number": cond,
        "const_cols":       const_cols,
        "issues":           issues,
        "ok":               len(issues) == 0,
    }

    if issues and raise_on_fail:
        raise RankDeficiencyError("\n".join(issues))
    elif issues:
        for msg in issues:
            warnings.warn(f"[hac] {msg}")

    return result


def safe_hac_regression(
    y: pd.Series,
    X: pd.DataFrame,
    hac_lags: int = 5,
    min_obs_per_param: int = MIN_OBS_PER_PARAM,
) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """
    Fit OLS with Newey-West HAC standard errors.
    Returns None if the design is problematic; otherwise returns robust results.
    """
    X = X.dropna(axis=1, how="all")
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=const_cols, errors="ignore")

    common = y.index.intersection(X.index)
    y = y.loc[common]
    X = X.loc[common]

    mask = y.notna() & X.notna().all(axis=1)
    y = y[mask]
    X = X[mask]

    if len(y) < 8:
        warnings.warn(f"Too few observations: {len(y)}")
        return None

    n, k = len(y), X.shape[1]
    if n < min_obs_per_param * k:
        warnings.warn(f"Low obs/param ratio: {n}/{k} = {n/k:.1f}")

    try:
        X_const = sm.add_constant(X.astype(float), has_constant="add")
        res     = sm.OLS(y.astype(float), X_const).fit()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "covariance of constraints")
            robust = res.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
        return robust
    except Exception as exc:
        warnings.warn(f"Regression failed: {exc}")
        return None
