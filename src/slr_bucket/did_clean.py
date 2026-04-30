"""
Fix 9: Panel DiD with pre-COVID 2019 baseline (clean entry estimate).

Problem: The event-window jump for entry (2020-04-01) uses a +-60-day
pre-period spanning Jan 23 - Mar 31, 2020 -- the peak of the COVID crash.
UST SF was dislocated by 100-300 bps during this period, so the pre-period
"baseline" is artificially elevated and the post-period comparison is ambiguous.

Solution: A full-panel DiD using 2019 as the clean pre-COVID reference.

Sample:
  pre_period:  2019-01-01 to 2020-01-31  (N~252 dates, pre-COVID, clean)
  relief:      2020-04-01 to 2021-03-31  (N~252 dates, SLR exclusion active)
  post_relief: 2021-04-01 to 2021-12-31  (N~185 dates, post-expiry)
  EXCLUDED:    2020-02-01 to 2020-03-31  (COVID crash; dropped from sample)

Model:
  y_abs_bps ~ relief + post_relief
            + relief:treasury_based + post_relief:treasury_based
            + C(series_id) + controls
  HAC SE, Newey-West 5 lags
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).parent.parent.parent

HAC_LAGS = 5


# -- Helpers -------------------------------------------------------------------

def nw_fit(
    y: pd.Series,
    X: pd.DataFrame,
    lags: int = HAC_LAGS,
) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """OLS + Newey-West HAC. Returns None on failure."""
    try:
        X_c = sm.add_constant(X.astype(float), has_constant="add")
        res = sm.OLS(y.astype(float), X_c).fit()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "covariance of constraints")
            robust = res.get_robustcov_results(cov_type="HAC", maxlags=lags)
        return robust
    except Exception as exc:
        warnings.warn(f"[did_clean] nw_fit failed: {exc}")
        return None


def extract_coef(
    res: sm.regression.linear_model.RegressionResultsWrapper,
    name: str,
) -> tuple[float, float, float]:
    """Extract (coef, SE, t-stat) for a named regressor. Returns (NaN, NaN, NaN) if absent."""
    names = list(res.model.exog_names)
    if name not in names:
        return np.nan, np.nan, np.nan
    idx = names.index(name)
    coef = float(res.params[idx])
    se = float(res.bse[idx])
    t = coef / se if se > 0 else np.nan
    return coef, se, t


# -- Main estimation -----------------------------------------------------------

def build_did_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the DiD sample with period indicators.

    Drops Feb-Mar 2020 (COVID crash).
    Adds: relief (0/1), post_relief (0/1), period label.
    """
    df = panel.copy()
    df = df[df["date"].between("2019-01-01", "2021-12-31")].copy()

    # Exclude COVID crash window
    crash_mask = df["date"].between("2020-02-01", "2020-03-31")
    df = df[~crash_mask].copy()

    # Period indicators
    pre_mask = df["date"].between("2019-01-01", "2020-01-31")
    relief_mask = df["date"].between("2020-04-01", "2021-03-31")
    post_mask = df["date"].between("2021-04-01", "2021-12-31")

    df["relief"] = relief_mask.astype(int)
    df["post_relief"] = post_mask.astype(int)
    df["period"] = "excluded"
    df.loc[pre_mask, "period"] = "pre"
    df.loc[relief_mask, "period"] = "relief"
    df.loc[post_mask, "period"] = "post_relief"

    # Keep only the three defined periods
    df = df[df["period"] != "excluded"].copy()

    return df


def run_did_clean_baseline(
    panel: pd.DataFrame,
    controls: list[str],
    spec_name: str = "TOTAL",
) -> dict:
    """
    Full-panel DiD using 2019 as the clean pre-COVID reference.

    Parameters
    ----------
    panel : full loaded panel (all series, 2019-2021)
    controls : list of control variable column names
    spec_name : label for this specification (e.g., "TOTAL", "DIRECT")

    Returns
    -------
    dict with coefficient estimates, SEs, t-stats, N, R2
    """
    df = build_did_sample(panel)

    # Verify both treasury_based values present
    tb_vals = df["treasury_based"].dropna().unique()
    if len(tb_vals) < 2:
        return {"spec": spec_name, "error": f"treasury_based constant={tb_vals}"}

    # Build interactions
    df["relief_x_treas"] = df["relief"] * df["treasury_based"]
    df["post_relief_x_treas"] = df["post_relief"] * df["treasury_based"]

    # Available controls (silently drop missing)
    avail_ctrl = [c for c in controls if c in df.columns]

    keep = ["y_abs_bps", "relief", "post_relief", "relief_x_treas",
            "post_relief_x_treas", "treasury_based", "series_id"] + avail_ctrl
    reg = df[keep].copy()

    for c in ["y_abs_bps", "relief", "post_relief", "relief_x_treas",
              "post_relief_x_treas", "treasury_based"] + avail_ctrl:
        reg[c] = pd.to_numeric(reg[c], errors="coerce")
    reg = reg.dropna()

    # Series FE dummies
    fe = pd.get_dummies(reg["series_id"].astype(str), prefix="fe", drop_first=True)

    X_cols = ["relief", "post_relief", "relief_x_treas", "post_relief_x_treas"] + avail_ctrl
    X = pd.concat(
        [reg[X_cols].reset_index(drop=True), fe.reset_index(drop=True)],
        axis=1,
    )
    y = reg["y_abs_bps"].reset_index(drop=True)

    # Obs/param check
    n, k = len(y), X.shape[1] + 1  # +1 for intercept
    if n < 10 * k:
        warnings.warn(f"[did_clean] Low obs/param: {n}/{k} = {n/k:.1f}")

    robust = nw_fit(y, X, lags=HAC_LAGS)
    if robust is None:
        return {"spec": spec_name, "error": "fit_failed", "n": len(reg)}

    coef_relief, se_relief, t_relief = extract_coef(robust, "relief")
    coef_post, se_post, t_post = extract_coef(robust, "post_relief")
    coef_rx, se_rx, t_rx = extract_coef(robust, "relief_x_treas")
    coef_px, se_px, t_px = extract_coef(robust, "post_relief_x_treas")

    # Period N counts
    n_pre = int((df["period"] == "pre").sum())
    n_relief = int((df["period"] == "relief").sum())
    n_post = int((df["period"] == "post_relief").sum())

    return {
        "spec": spec_name,
        "sample": "2019 pre-COVID baseline (excl Feb-Mar 2020)",
        "n": len(reg),
        "n_pre_period": n_pre,
        "n_relief_period": n_relief,
        "n_post_period": n_post,
        "r2": float(robust.rsquared),
        # Relief period (non-Treasury)
        "coef_relief": round(coef_relief, 4),
        "se_relief": round(se_relief, 4),
        "t_relief": round(t_relief, 3),
        # Relief period (Treasury differential = DiD entry estimate)
        "coef_relief_x_treas": round(coef_rx, 4),
        "se_relief_x_treas": round(se_rx, 4),
        "t_relief_x_treas": round(t_rx, 3),
        # Post-relief period (non-Treasury)
        "coef_post_relief": round(coef_post, 4),
        "se_post_relief": round(se_post, 4),
        "t_post_relief": round(t_post, 3),
        # Post-relief period (Treasury differential = DiD exit estimate)
        "coef_post_relief_x_treas": round(coef_px, 4),
        "se_post_relief_x_treas": round(se_px, 4),
        "t_post_relief_x_treas": round(t_px, 3),
        "controls_used": avail_ctrl,
    }


def run_did_both_specs(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Run the clean DiD for both TOTAL and DIRECT control specifications.
    Returns a two-row DataFrame.
    """
    TOTAL_CONTROLS = ["VIX", "HY_OAS", "BAA10Y"]
    DIRECT_CONTROLS = ["VIX", "HY_OAS", "BAA10Y", "SOFR_rate", "spr_tgcr",
                       "issu_7_bil", "issu_14_bil", "issu_30_bil"]

    rows = []
    for spec, ctrl in [("TOTAL", TOTAL_CONTROLS), ("DIRECT", DIRECT_CONTROLS)]:
        print(f"  Running DiD clean baseline ({spec})...")
        result = run_did_clean_baseline(panel, ctrl, spec_name=spec)
        rows.append(result)
        if "error" not in result:
            print(f"    relief_x_treas:     {result['coef_relief_x_treas']:+.3f} "
                  f"(SE={result['se_relief_x_treas']:.3f}, t={result['t_relief_x_treas']:.2f})")
            print(f"    post_relief_x_treas:{result['coef_post_relief_x_treas']:+.3f} "
                  f"(SE={result['se_post_relief_x_treas']:.3f}, t={result['t_post_relief_x_treas']:.2f})")
            print(f"    N={result['n']:,}, R2={result['r2']:.3f}")
        else:
            print(f"    ERROR: {result['error']}")

    return pd.DataFrame(rows)
