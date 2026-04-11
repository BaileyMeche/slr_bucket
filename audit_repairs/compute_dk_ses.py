"""
Driscoll-Kraay Standard Errors — Robustness Check
===================================================
Computes DK SEs for the pooled jump regressions (entry/exit, W=60, TOTAL/DIRECT specs).
DK SEs allow for cross-sectional dependence in addition to Newey-West HAC.

Reference: Driscoll & Kraay (1998), Rev. Econ. Stat.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Make repo root and audit_repairs importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "audit_repairs"))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from repaired_pipeline import (
    load_full_panel,
    add_event_time_all_series,
    TOTAL_CONTROLS,
    DIRECT_CONTROLS,
    HAC_LAGS,
)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DK_LAGS = 5  # matches HAC_LAGS used in main regressions


# ── Driscoll-Kraay SE implementation ──────────────────────────────────────────

def compute_dk_ses(
    y: np.ndarray,
    X: np.ndarray,
    dates: np.ndarray,
    lags: int = DK_LAGS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Driscoll-Kraay (1998) standard errors for panel OLS.

    Parameters
    ----------
    y : (N,) outcome
    X : (N, k) design matrix (including constant if desired)
    dates : (N,) array of date values (same index as y, X)
    lags : Newey-West bandwidth for the cross-sectionally summed scores

    Returns
    -------
    beta : (k,) OLS coefficients
    se_dk : (k,) DK standard errors
    """
    N = len(y)
    k = X.shape[1]

    # OLS
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta  # (N,) residuals

    # Unique dates and score matrix S: shape (T, k)
    unique_dates = np.sort(np.unique(dates))
    T = len(unique_dates)
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    t_idx = np.array([date_to_idx[d] for d in dates])

    # S[t] = sum_{i: date_i=t} x_it * e_it
    S = np.zeros((T, k))
    for i in range(N):
        S[t_idx[i]] += X[i] * e[i]

    # Newey-West on S: Omega = (1/T) sum_t s_t s_t' + HAC terms
    Omega = S.T @ S / T
    for h in range(1, lags + 1):
        w = 1.0 - h / (lags + 1)
        cross = S[h:].T @ S[:T - h] / T
        Omega += w * (cross + cross.T)

    # Sandwich: V_DK = (X'X/T)^{-1} * Omega * (X'X/T)^{-1} / T
    XtX = X.T @ X / T
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    V_DK = XtX_inv @ Omega @ XtX_inv / T
    se_dk = np.sqrt(np.abs(np.diag(V_DK)))

    return beta, se_dk


# ── Build regression design for one spec ──────────────────────────────────────

def build_design(
    panel: pd.DataFrame,
    event_date: str,
    window: int,
    controls: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Returns y, X (with constant), dates array, and column names.
    Replicates the design from repaired_pipeline.run_pooled_jump.
    """
    work = add_event_time_all_series(panel, event_date)
    work = work[work["event_time"].between(-window, window)].copy()
    work["post"] = (work["event_time"] >= 0).astype(int)

    avail_ctrl = [c for c in controls if c in work.columns]
    keep_cols = ["y_abs_bps", "post", "treasury_based", "series_id", "date"] + avail_ctrl
    reg = work[keep_cols].dropna().copy()

    for c in ["y_abs_bps", "post", "treasury_based"] + avail_ctrl:
        reg[c] = pd.to_numeric(reg[c], errors="coerce")
    reg = reg.dropna().reset_index(drop=True)

    reg["post_x_treas"] = reg["post"] * reg["treasury_based"]

    # Series FE dummies (drop first to avoid perfect collinearity)
    fe = pd.get_dummies(reg["series_id"].astype(str), prefix="fe", drop_first=True)
    fe = fe.astype(int)

    # Build X without constant first (nw_fit adds constant via sm.add_constant)
    # We'll add constant here for DK manual computation
    core_cols = ["post", "post_x_treas"] + avail_ctrl
    X_df = pd.concat(
        [reg[core_cols].reset_index(drop=True), fe.reset_index(drop=True)],
        axis=1,
    )

    # Drop zero-variance columns
    const_mask = X_df.nunique() <= 1
    X_df = X_df.loc[:, ~const_mask]

    # Add constant
    X_df.insert(0, "const", 1.0)

    col_names = list(X_df.columns)
    y = reg["y_abs_bps"].values.astype(float)
    X = X_df.values.astype(float)
    dates = reg["date"].values

    return y, X, dates, col_names, reg


def nw_fit_for_dk(y_arr, X_arr, col_names, lags=HAC_LAGS):
    """Run statsmodels HAC (Newey-West) on pre-built arrays to get NW SEs."""
    y_s = pd.Series(y_arr)
    X_df = pd.DataFrame(X_arr, columns=col_names)
    # sm.add_constant would add another const; we already have one
    mask = y_s.notna() & X_df.notna().all(axis=1)
    y2 = y_s[mask].astype(float)
    X2 = X_df[mask].astype(float)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = sm.OLS(y2, X2).fit()
            return res.get_robustcov_results(cov_type="HAC", maxlags=lags)
    except Exception as exc:
        warnings.warn(f"NW fit failed: {exc}")
        return None


# ── Main computation ───────────────────────────────────────────────────────────

def run_dk_robustness(panel: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("TOTAL", TOTAL_CONTROLS),
        ("DIRECT", DIRECT_CONTROLS),
    ]
    events = [
        ("entry", "2020-04-01"),
        ("exit", "2021-03-31"),
    ]
    window = 60

    rows = []

    for event_label, event_date in events:
        for spec_name, controls in specs:
            print(f"\n{'='*60}")
            print(f"  {event_label.upper()} W={window}  spec={spec_name}")
            print(f"{'='*60}")

            try:
                y, X, dates, col_names, reg = build_design(
                    panel, event_date, window, controls
                )
            except Exception as exc:
                print(f"  ERROR building design: {exc}")
                continue

            print(f"  N obs={len(y)}, N series={reg['series_id'].nunique()}, "
                  f"N dates={len(np.unique(dates))}, k={X.shape[1]}")

            # ── Newey-West (from statsmodels, replicating main pipeline) ──
            nw_res = nw_fit_for_dk(y, X, col_names, lags=HAC_LAGS)
            if nw_res is None:
                print("  NW fit failed.")
                continue

            nw_names = list(nw_res.model.exog_names)

            def _get_nw(frag):
                if frag in nw_names:
                    i = nw_names.index(frag)
                elif any(frag in n for n in nw_names):
                    i = nw_names.index([n for n in nw_names if frag in n][0])
                else:
                    return np.nan, np.nan
                return float(nw_res.params[i]), float(nw_res.bse[i])

            coef_post_nw, se_post_nw = _get_nw("post")
            # post_x_treas — exact name
            if "post_x_treas" in nw_names:
                i = nw_names.index("post_x_treas")
                coef_inter_nw, se_inter_nw = float(nw_res.params[i]), float(nw_res.bse[i])
            else:
                coef_inter_nw, se_inter_nw = np.nan, np.nan

            # ── Driscoll-Kraay ──
            beta_dk, se_dk_arr = compute_dk_ses(y, X, dates, lags=DK_LAGS)

            def _get_dk(frag):
                if frag in col_names:
                    i = col_names.index(frag)
                elif any(frag in n for n in col_names):
                    i = col_names.index([n for n in col_names if frag in n][0])
                else:
                    return np.nan, np.nan
                return float(beta_dk[i]), float(se_dk_arr[i])

            coef_post_dk, se_post_dk = _get_dk("post")
            coef_inter_dk, se_inter_dk = _get_dk("post_x_treas")

            # t-stats
            def _t(c, s):
                return c / s if (np.isfinite(s) and s > 0) else np.nan

            t_post_nw = _t(coef_post_nw, se_post_nw)
            t_inter_nw = _t(coef_inter_nw, se_inter_nw)
            t_post_dk = _t(coef_post_dk, se_post_dk)
            t_inter_dk = _t(coef_inter_dk, se_inter_dk)

            print(f"\n  Post coefficient:")
            print(f"    coef = {coef_post_nw:+.3f} bps")
            print(f"    NW:  SE={se_post_nw:.3f}  t={t_post_nw:.2f}")
            print(f"    DK:  SE={se_post_dk:.3f}  t={t_post_dk:.2f}")

            print(f"\n  Post x Treasury coefficient:")
            print(f"    coef = {coef_inter_nw:+.3f} bps")
            print(f"    NW:  SE={se_inter_nw:.3f}  t={t_inter_nw:.2f}")
            print(f"    DK:  SE={se_inter_dk:.3f}  t={t_inter_dk:.2f}")

            rows.append({
                "event": event_label,
                "window": window,
                "spec": spec_name,
                "n_obs": int(len(y)),
                "n_series": int(reg["series_id"].nunique()),
                # Post
                "coef_post": round(coef_post_nw, 4),
                "se_nw_post": round(se_post_nw, 4),
                "t_nw_post": round(t_post_nw, 3) if np.isfinite(t_post_nw) else np.nan,
                "se_dk_post": round(se_post_dk, 4),
                "t_dk_post": round(t_post_dk, 3) if np.isfinite(t_post_dk) else np.nan,
                # Post x Treasury
                "coef_post_x_treas": round(coef_inter_nw, 4),
                "se_nw_post_x_treas": round(se_inter_nw, 4),
                "t_nw_post_x_treas": round(t_inter_nw, 3) if np.isfinite(t_inter_nw) else np.nan,
                "se_dk_post_x_treas": round(se_inter_dk, 4),
                "t_dk_post_x_treas": round(t_inter_dk, 3) if np.isfinite(t_inter_dk) else np.nan,
            })

    return pd.DataFrame(rows)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Loading full panel...")
    panel = load_full_panel()

    print("\nRunning Driscoll-Kraay robustness checks...")
    results = run_dk_robustness(panel)

    out_path = OUT_DIR / "dk_robustness.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Summary table ──
    print("\n" + "=" * 80)
    print("SUMMARY: NW vs DK t-statistics")
    print("=" * 80)
    print(f"\n{'Event':<8} {'W':<5} {'Spec':<8}  "
          f"{'coef_post':>10}  {'t_NW_post':>10}  {'t_DK_post':>10}  "
          f"{'coef_inter':>11}  {'t_NW_inter':>11}  {'t_DK_inter':>11}")
    print("-" * 95)
    for _, row in results.iterrows():
        print(f"{row['event']:<8} {row['window']:<5} {row['spec']:<8}  "
              f"{row['coef_post']:>+10.3f}  {row['t_nw_post']:>10.2f}  {row['t_dk_post']:>10.2f}  "
              f"{row['coef_post_x_treas']:>+11.3f}  {row['t_nw_post_x_treas']:>11.2f}  {row['t_dk_post_x_treas']:>11.2f}")
    print("=" * 80)

    print("\nKey finding (exit W=60):")
    exit_rows = results[results["event"] == "exit"]
    for _, row in exit_rows.iterrows():
        sig_dk = "|t|>2 (significant)" if abs(row["t_dk_post_x_treas"]) > 2 else "NOT significant"
        print(f"  spec={row['spec']}: Post*Treasury t_NW={row['t_nw_post_x_treas']:.2f}, "
              f"t_DK={row['t_dk_post_x_treas']:.2f} -> {sig_dk}")

    return results


if __name__ == "__main__":
    main()
