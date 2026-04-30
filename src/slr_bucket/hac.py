"""
Fix 6: Guard against HAC covariance rank deficiency.

Root causes:
1. Small N (equity N=14-16) with many regressors -> model underidentified
2. treasury_based constant within single-strategy run -> collinear interaction
3. Empty bin dummies for some windows/tenors -> zero-variance columns

After Fix 3 (proper pooling), N will be large enough. But we add explicit
guards so future regressions fail informatively rather than silently.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).parent.parent.parent

MIN_OBS_PER_PARAM = 10   # HAC standard: at least 10 obs per parameter
MAX_CONDITION_NUMBER = 1e10  # above this, design matrix is near-singular


class RankDeficiencyError(ValueError):
    pass


def check_design_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    min_obs_per_param: int = MIN_OBS_PER_PARAM,
    max_condition: float = MAX_CONDITION_NUMBER,
    raise_on_fail: bool = False,
) -> dict:
    """
    Check the design matrix for rank deficiency, obs/param ratio, and collinearity.
    Returns dict with diagnostics.
    """
    n, k = X.shape
    issues = []

    # 1. Obs/param ratio
    obs_param_ratio = n / k if k > 0 else 0
    if obs_param_ratio < min_obs_per_param:
        issues.append(
            f"Low obs/param ratio: {n} obs / {k} params = {obs_param_ratio:.1f} "
            f"(minimum recommended: {min_obs_per_param})"
        )

    # 2. Rank check
    try:
        rank = np.linalg.matrix_rank(X.values.astype(float))
        if rank < k:
            issues.append(f"Rank deficiency: rank={rank} < k={k} (dropped {k-rank} columns)")
    except Exception:
        rank = np.nan

    # 3. Condition number
    try:
        cond = float(np.linalg.cond(X.values.astype(float)))
        if cond > max_condition:
            issues.append(f"Near-singular: condition number = {cond:.2e}")
    except Exception:
        cond = np.nan

    # 4. Zero-variance columns
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if const_cols:
        issues.append(f"Zero-variance columns: {const_cols}")

    # 5. treasury_based check (for pooled regressions)
    if "treasury_based" in X.columns or any("treasury_based" in c for c in X.columns):
        if hasattr(y, "name") and "treasury_based" in str(y.name):
            pass
        tb_cols = [c for c in X.columns if "treasury_based" in c and "post" in c]
        if tb_cols:
            for col in tb_cols:
                if X[col].nunique(dropna=True) <= 1:
                    issues.append(
                        f"post:treasury_based interaction is constant (={X[col].iloc[0]:.0f}) "
                        f"-- all observations have the same treasury_based value. "
                        f"The pooled panel must include BOTH treasury_based=1 and 0 series."
                    )

    result = {
        "n": n,
        "k": k,
        "rank": rank,
        "obs_param_ratio": obs_param_ratio,
        "condition_number": cond,
        "const_cols": const_cols,
        "issues": issues,
        "ok": len(issues) == 0,
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
    Fit OLS with HAC standard errors, with pre-flight checks.
    Returns None if the design is too problematic; otherwise returns robust results.
    """
    # Drop all-NaN or zero-variance columns
    X = X.dropna(axis=1, how="all")
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=const_cols, errors="ignore")

    # Align y and X
    common = y.index.intersection(X.index)
    y = y.loc[common]
    X = X.loc[common]

    # Drop rows with any NaN
    mask = y.notna() & X.notna().all(axis=1)
    y = y[mask]
    X = X[mask]

    if len(y) < 8:
        warnings.warn(f"[hac] Too few observations: {len(y)}")
        return None

    # Check obs/param ratio
    n, k = len(y), X.shape[1]
    if n < min_obs_per_param * k:
        warnings.warn(
            f"[hac] Low obs/param ratio: {n}/{k} = {n/k:.1f} "
            f"(min recommended: {min_obs_per_param})"
        )

    try:
        X_const = sm.add_constant(X.astype(float), has_constant="add")
        res = sm.OLS(y.astype(float), X_const).fit()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "covariance of constraints")
            robust = res.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
        return robust
    except Exception as exc:
        warnings.warn(f"[hac] Regression failed: {exc}")
        return None
