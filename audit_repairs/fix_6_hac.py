"""
Fix 6: Guard against HAC covariance rank deficiency.

Root causes:
1. Small N (equity N=14-16) with many regressors → model underidentified
2. treasury_based constant within single-strategy run → collinear interaction
3. Empty bin dummies for some windows/tenors → zero-variance columns

After Fix 3 (proper pooling), N will be large enough. But we add explicit
guards so future regressions fail informatively rather than silently.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


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
                        f"— all observations have the same treasury_based value. "
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
            warnings.warn(f"[fix_6_hac] {msg}")

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
        warnings.warn(f"[fix_6_hac] Too few observations: {len(y)}")
        return None

    # Check obs/param ratio
    n, k = len(y), X.shape[1]
    if n < min_obs_per_param * k:
        warnings.warn(
            f"[fix_6_hac] Low obs/param ratio: {n}/{k} = {n/k:.1f} "
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
        warnings.warn(f"[fix_6_hac] Regression failed: {exc}")
        return None


def demonstrate_rank_deficiency_bug() -> None:
    """
    Reproduce the collinearity bug: single-strategy panel with treasury_based=1 for all.
    Shows that post:treasury_based ≡ post in this case.
    """
    np.random.seed(42)
    n = 50
    post = (np.arange(n) >= 25).astype(float)
    # All treasury_based=1 (single-strategy run)
    treas = np.ones(n)
    y = 20 + (-5) * post + np.random.normal(0, 2, n)

    X_buggy = pd.DataFrame({
        "post": post,
        "treasury_based": treas,
        "post_x_treas": post * treas,  # ≡ post when treas=1
    })

    print("=== Buggy design (single-strategy, all treasury_based=1) ===")
    diag = check_design_matrix(sm.add_constant(X_buggy.astype(float)), pd.Series(y))
    print(f"  Issues: {diag['issues']}")
    print(f"  Rank: {diag['rank']} vs k: {diag['k']}")
    print(f"  Correlation(post, post_x_treas): {np.corrcoef(post, post * treas)[0,1]:.4f}")

    # Fixed: pooled panel mixes treasury_based=0 and 1
    treas_mixed = np.tile([1, 0], n // 2)[:n]
    X_fixed = pd.DataFrame({
        "post": post,
        "treasury_based": treas_mixed,
        "post_x_treas": post * treas_mixed,
    })

    print("\n=== Fixed design (pooled, mixed treasury_based) ===")
    diag2 = check_design_matrix(sm.add_constant(X_fixed.astype(float)), pd.Series(y))
    print(f"  Issues: {diag2['issues']}")
    print(f"  Rank: {diag2['rank']} vs k: {diag2['k']}")
    print(f"  Correlation(post, post_x_treas): {np.corrcoef(post, post * treas_mixed)[0,1]:.4f}")
    print("  post:treasury_based now separately identified.")


if __name__ == "__main__":
    print("=" * 70)
    print("FIX 6: HAC rank deficiency guard")
    print("=" * 70)

    demonstrate_rank_deficiency_bug()

    print("\n=== Minimum obs/param requirements ===")
    print(f"  MIN_OBS_PER_PARAM = {MIN_OBS_PER_PARAM}")
    print()
    specs = [
        ("Pooled W=60, 17 series FE + 2 coefs + 3 controls", 17 + 2 + 3),
        ("Pooled W=20, 17 series FE + 2 coefs + 3 controls", 17 + 2 + 3),
        ("Per-series W=60, 1 coef + 3 controls", 4),
        ("Per-series W=20, 1 coef + 3 controls", 4),
        ("Buggy equity W=60 with 12 params", 12),
    ]
    for label, k in specs:
        min_n = MIN_OBS_PER_PARAM * k
        print(f"  {label}: need N >= {min_n}")

    print()
    print("After Fix 3 (proper pooling):")
    print("  W=60 pooled: N ≈ 17 × 121 = 2057 → obs/param = 2057/22 ≈ 93 ✓")
    print("  W=20 pooled: N ≈ 17 × 41  =  697 → obs/param =  697/22 ≈ 32 ✓")
