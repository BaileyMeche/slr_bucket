"""
regression.py: Shared regression helpers and pipeline constants.

Extracted from repaired_pipeline.py. Provides:
  - Constants: EVENTS, WINDOWS, HAC_LAGS, TENOR_SUBSET, EQUITY_INDICES,
               SAMPLE_START, SAMPLE_END, TOTAL_CONTROLS, DIRECT_CONTROLS, MAIN_EVENTS
  - Functions: load_controls(), load_full_panel(), add_event_time_all_series(),
               nw_fit(), extract_coef(), run_pooled_jump(), run_series_level_jumps(),
               run_binned_event_study(), compute_summary_stats()
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from slr_bucket.load import stack_all_outcomes
from slr_bucket.winsorize import winsorize_panel
from slr_bucket.issuance import build_issuance_controls
from slr_bucket.hac import MIN_OBS_PER_PARAM

REPO_ROOT = Path(__file__).parent.parent.parent

# -- Configuration -------------------------------------------------------------
EVENTS = ["2020-04-01", "2021-03-19", "2021-03-31"]
WINDOWS = [20, 60]
HAC_LAGS = 5
TENOR_SUBSET = [2, 5, 10]
EQUITY_INDICES = ["SPX", "NDX", "INDU"]
EVENT_BINS = [(-60, -41), (-40, -21), (-20, -1), (0, 0), (1, 20), (21, 40), (41, 60)]
SAMPLE_START = "2019-01-01"
SAMPLE_END = "2021-12-31"

TOTAL_CONTROLS = ["VIX", "HY_OAS", "BAA10Y"]
DIRECT_CONTROLS = ["VIX", "HY_OAS", "BAA10Y", "SOFR_rate", "spr_tgcr",
                   "issu_7_bil", "issu_14_bil", "issu_30_bil"]

# Main table events (exclude 2021-03-19 due to overlap with 2021-03-31)
MAIN_EVENTS = ["2020-04-01", "2021-03-31"]

SERIES_DIR = REPO_ROOT / "data" / "series"


# -- Data loading --------------------------------------------------------------

def load_controls() -> pd.DataFrame:
    """Load and merge all control variables."""
    # VIX, HY spreads
    ctrl = pd.read_csv(REPO_ROOT / "data/raw/event_inputs/controls_vix_creditspreads_fred.csv")
    ctrl["date"] = pd.to_datetime(ctrl["date"])

    # Repo rates
    repo = pd.read_csv(REPO_ROOT / "data/raw/event_inputs/repo_rates_combined.csv")
    repo["date"] = pd.to_datetime(repo["date"])

    # Compute funding spreads
    if "spr_tgcr" not in repo.columns and {"SOFR", "TGCR"}.issubset(repo.columns):
        repo["spr_tgcr"] = pd.to_numeric(repo["TGCR"], errors="coerce") - \
                            pd.to_numeric(repo["SOFR"], errors="coerce")
    if "spr_effr" not in repo.columns and {"SOFR", "BGCR"}.issubset(repo.columns):
        repo["spr_effr"] = pd.to_numeric(repo["BGCR"], errors="coerce") - \
                            pd.to_numeric(repo["SOFR"], errors="coerce")

    repo = repo.rename(columns={"SOFR": "SOFR_rate"})
    keep_repo = [c for c in ["date", "SOFR_rate", "spr_tgcr", "spr_effr"] if c in repo.columns]

    controls = ctrl.merge(repo[keep_repo], on="date", how="outer").sort_values("date")

    # Issuance controls: rolling 7/14/30-day total Treasury issuance
    issuance_path = REPO_ROOT / "data" / "raw" / "event_inputs" / "treasury_issuance_by_tenor_fiscaldata.csv"
    if issuance_path.exists():
        issu = build_issuance_controls(issuance_path, date_range=(SAMPLE_START, SAMPLE_END))
        controls = controls.merge(issu, on="date", how="left")
        print("  Issuance controls merged: issu_7_bil, issu_14_bil, issu_30_bil")
    else:
        print(f"  WARNING: Issuance file not found at {issuance_path}")

    return controls


def load_full_panel() -> pd.DataFrame:
    """Load all outcomes, merge controls, filter to sample window."""
    OUT_DIR = REPO_ROOT / "_outputs"
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading outcomes (corrected unit loader)...")
    panel = stack_all_outcomes(SERIES_DIR, tenor_subset=TENOR_SUBSET, equity_indices=EQUITY_INDICES)

    # Winsorize COVID-spike outliers: p1/p99 by series before merging controls
    print("Winsorizing outcome variables (p1/p99 by series)...")
    panel = winsorize_panel(
        panel,
        pct_lo=1,
        pct_hi=99,
        by_series=True,
        log_path=OUT_DIR / "fix7_winsorize_log.csv",
    )

    print("Loading controls (including issuance controls)...")
    controls = load_controls()

    # Merge
    panel = panel.merge(controls, on="date", how="left")

    # Filter to sample
    panel = panel[panel["date"].between(SAMPLE_START, SAMPLE_END)].copy()
    print(f"Full panel (post-filter): {len(panel):,} rows, "
          f"{panel['series_id'].nunique()} series, "
          f"{panel['date'].nunique()} dates")

    return panel


# -- Event-time assignment -----------------------------------------------------

def add_event_time_all_series(panel: pd.DataFrame, event_date: str) -> pd.DataFrame:
    """Assign trading-day event_time within each series."""
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    t0 = pd.Timestamp(event_date)

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        dates = g["date"].values
        idx0 = int(np.searchsorted(dates, np.datetime64(t0), side="left"))
        if idx0 >= len(dates):
            idx0 = len(dates) - 1
        g["event_t0_used"] = pd.Timestamp(dates[idx0])
        g["event_time"] = np.arange(len(g), dtype=int) - idx0
        return g

    return out.groupby("series_id", group_keys=False).apply(_apply)


# -- HAC regression helpers ----------------------------------------------------

def nw_fit(y: pd.Series, X: pd.DataFrame, lags: int = HAC_LAGS):
    """Fit OLS with Newey-West HAC covariance. Returns robust result or None."""
    # Drop zero-variance columns
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=const_cols, errors="ignore")
    if X.empty:
        return None

    X_c = sm.add_constant(X.astype(float), has_constant="add")
    common = y.index.intersection(X_c.index)
    y2, X2 = y.loc[common].astype(float), X_c.loc[common]
    mask = y2.notna() & X2.notna().all(axis=1)
    y2, X2 = y2[mask], X2[mask]

    if len(y2) < 8 or y2.nunique() < 2:
        return None

    # obs/param guard
    n_params = X2.shape[1]
    if len(y2) < MIN_OBS_PER_PARAM * n_params:
        warnings.warn(f"Low obs/param ratio: {len(y2)}/{n_params} = {len(y2)/n_params:.1f}")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "covariance of constraints")
            res = sm.OLS(y2, X2).fit()
            return res.get_robustcov_results(cov_type="HAC", maxlags=lags)
    except Exception as exc:
        warnings.warn(f"Regression failed: {exc}")
        return None


def extract_coef(robust, name_fragment: str) -> tuple[float, float]:
    """Extract coefficient and SE by partial name match."""
    names = list(robust.model.exog_names)
    # exact match first
    if name_fragment in names:
        idx = names.index(name_fragment)
        return float(robust.params[idx]), float(robust.bse[idx])
    # partial match
    matches = [n for n in names if name_fragment in n]
    if matches:
        idx = names.index(matches[0])
        return float(robust.params[idx]), float(robust.bse[idx])
    return np.nan, np.nan


# -- Table 1: Summary statistics -----------------------------------------------

def compute_summary_stats(panel: pd.DataFrame) -> pd.DataFrame:
    regimes = {
        "pre":    ("2019-01-01", "2020-03-31"),
        "relief": ("2020-04-01", "2021-03-31"),
        "post":   ("2021-04-01", "2021-12-31"),
    }

    rows = []
    for series_id, g in panel.groupby("series_id"):
        strategy = g["strategy"].iloc[0]
        tenor = g["tenor"].iloc[0]
        treas = int(g["treasury_based"].iloc[0])

        for regime, (start, end) in regimes.items():
            sub = g[g["date"].between(start, end)]
            y = sub["y_bps"].dropna()
            ya = sub["y_abs_bps"].dropna()
            if len(ya) == 0:
                continue
            rows.append({
                "strategy": strategy,
                "series_id": series_id,
                "tenor": tenor,
                "treasury_based": treas,
                "regime": regime,
                "N_days": len(ya),
                "mean_W": float(y.mean()),
                "median_W": float(y.median()),
                "mean_absW": float(ya.mean()),
                "median_absW": float(ya.median()),
                "p5_W": float(y.quantile(0.05)),
                "p95_W": float(y.quantile(0.95)),
            })

    return pd.DataFrame(rows).sort_values(["strategy", "series_id", "regime"])


# -- Pooled jump regression ----------------------------------------------------

def run_pooled_jump(
    panel: pd.DataFrame,
    event_date: str,
    window: int,
    controls: list[str],
    spec: str,
    event_window_data: pd.DataFrame | None = None,
) -> dict:
    """
    Pooled jump regression with Post x TreasuryBased interaction.
    Requires both treasury_based=1 and treasury_based=0 series in the panel.
    """
    if event_window_data is None:
        work = add_event_time_all_series(panel, event_date)
        work = work[work["event_time"].between(-window, window)].copy()
    else:
        work = event_window_data.copy()

    work["post"] = (work["event_time"] >= 0).astype(int)

    # Verify pooled design: treasury_based must vary
    tb_vals = work["treasury_based"].dropna().unique()
    if len(tb_vals) < 2:
        return {
            "event": event_date, "window": window, "spec": spec,
            "error": f"treasury_based is constant={tb_vals[0]} -- not a pooled regression",
        }

    # Build design matrix
    avail_ctrl = [c for c in controls if c in work.columns]
    keep_cols = ["y_abs_bps", "post", "treasury_based", "series_id"] + avail_ctrl
    reg = work[keep_cols].dropna().copy()

    for c in ["y_abs_bps", "post", "treasury_based"] + avail_ctrl:
        reg[c] = pd.to_numeric(reg[c], errors="coerce")
    reg = reg.dropna()

    # Interaction
    reg["post_x_treas"] = reg["post"] * reg["treasury_based"]

    # Series FE dummies (drop first)
    fe = pd.get_dummies(reg["series_id"].astype(str), prefix="fe", drop_first=True)

    X = pd.concat(
        [reg[["post", "post_x_treas"] + avail_ctrl].reset_index(drop=True),
         fe.reset_index(drop=True)],
        axis=1,
    )
    y = reg["y_abs_bps"].reset_index(drop=True)

    robust = nw_fit(y, X, lags=HAC_LAGS)
    if robust is None:
        return {"event": event_date, "window": window, "spec": spec, "error": "fit_failed", "n": len(reg)}

    coef_post, se_post = extract_coef(robust, "post")
    # Specifically get post_x_treas (not post_x)
    names = list(robust.model.exog_names)
    inter_matches = [n for n in names if n == "post_x_treas"]
    if inter_matches:
        idx = names.index(inter_matches[0])
        coef_inter, se_inter = float(robust.params[idx]), float(robust.bse[idx])
    else:
        coef_inter, se_inter = np.nan, np.nan

    # Ctrl coefs
    ctrl_coefs = {}
    for c in avail_ctrl:
        v, se = extract_coef(robust, c)
        ctrl_coefs[f"coef_{c}"] = v
        ctrl_coefs[f"se_{c}"] = se

    result = {
        "event": event_date,
        "window": window,
        "spec": spec,
        "n": int(robust.nobs),
        "n_treasury_series": int((reg.groupby("series_id")["treasury_based"].first() == 1).sum()),
        "n_nontreasury_series": int((reg.groupby("series_id")["treasury_based"].first() == 0).sum()),
        "r2": float(robust.rsquared),
        "coef_post": coef_post,
        "se_post": se_post,
        "t_post": coef_post / se_post if (np.isfinite(se_post) and se_post > 0) else np.nan,
        "coef_post_x_treas": coef_inter,
        "se_post_x_treas": se_inter,
        "t_post_x_treas": coef_inter / se_inter if (np.isfinite(se_inter) and se_inter > 0) else np.nan,
        **ctrl_coefs,
    }
    return result


# -- Series-level jump regressions ---------------------------------------------

def run_series_level_jumps(
    panel: pd.DataFrame,
    event_date: str,
    window: int,
    controls: list[str],
    spec: str,
) -> list[dict]:
    """Per-series jump regressions."""
    work = add_event_time_all_series(panel, event_date)
    work = work[work["event_time"].between(-window, window)].copy()
    work["post"] = (work["event_time"] >= 0).astype(int)

    rows = []
    for sid, g in work.groupby("series_id"):
        avail = [c for c in controls if c in g.columns]
        keep = ["y_abs_bps", "post"] + avail
        reg = g[keep].dropna().copy()
        for c in keep:
            reg[c] = pd.to_numeric(reg[c], errors="coerce")
        reg = reg.dropna()

        if len(reg) < 8 or reg["post"].nunique() < 2:
            continue

        X = reg[["post"] + avail]
        robust = nw_fit(reg["y_abs_bps"], X)
        if robust is None:
            continue

        coef, se = extract_coef(robust, "post")
        strategy = g["strategy"].iloc[0]
        treas = int(g["treasury_based"].iloc[0])

        rows.append({
            "series_id": sid,
            "strategy": strategy,
            "treasury_based": treas,
            "event": event_date,
            "window": window,
            "spec": spec,
            "n": int(robust.nobs),
            "coef_post": coef,
            "se_post": se,
            "t_post": coef / se if (np.isfinite(se) and se > 0) else np.nan,
            "ci_low": coef - 1.96 * se if np.isfinite(se) else np.nan,
            "ci_high": coef + 1.96 * se if np.isfinite(se) else np.nan,
        })

    return rows


# -- Binned event study --------------------------------------------------------

def run_binned_event_study(
    panel: pd.DataFrame,
    event_date: str,
    bins: list[tuple[int, int]],
    controls: list[str],
    series_id: str | None = None,
) -> pd.DataFrame:
    """Binned event-study path for a single series or pooled."""
    import re

    work = add_event_time_all_series(panel, event_date)

    # Assign bins
    et = work["event_time"]
    work["bin"] = pd.NA
    for lo, hi in bins:
        m = (et >= lo) & (et <= hi)
        work.loc[m, "bin"] = f"bin_{lo}_{hi}"

    if series_id is not None:
        work = work[work["series_id"] == series_id].copy()

    work = work.dropna(subset=["bin", "y_abs_bps"]).copy()
    if len(work) == 0:
        return pd.DataFrame()

    dummies = pd.get_dummies(work["bin"], prefix="")
    dummies.columns = [c.strip("_") for c in dummies.columns]

    # Reference bin: pre-event [-20,-1]
    ref_label = "bin_-20_-1"
    if ref_label in dummies.columns:
        dummies = dummies.drop(columns=[ref_label])
    elif len(dummies.columns) > 0:
        dummies = dummies.drop(columns=[dummies.columns[0]])

    avail = [c for c in controls if c in work.columns]
    X = pd.concat(
        [dummies.reset_index(drop=True),
         work[avail].reset_index(drop=True)],
        axis=1,
    )
    y = work["y_abs_bps"].reset_index(drop=True)

    robust = nw_fit(y, X)
    if robust is None:
        return pd.DataFrame()

    names = list(robust.model.exog_names)
    out = []
    for col in dummies.columns:
        if col not in names:
            continue
        coef, se = extract_coef(robust, col)
        match = re.search(r"(-?\d+)_(-?\d+)", col)
        bin_mid = 0.5 * (int(match.group(1)) + int(match.group(2))) if match else np.nan
        out.append({"bin_label": col, "bin_mid": bin_mid, "estimate": coef, "se": se,
                    "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se,
                    "n": int(robust.nobs)})

    return pd.DataFrame(out)
