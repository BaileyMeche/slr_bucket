"""
Fix 3: Correct pooled panel regression across ALL strategies simultaneously.

Root cause: The notebook uses ACTIVE = "ust_spot_fut" (or a single strategy),
so panel_long contains only one strategy. The "pooled" regression with
post:treasury_based is run on a single-strategy panel where treasury_based
is constant (all 0 or all 1), making the interaction perfectly collinear
with post.

Fix: Stack ALL strategies into a single panel and run:
    y_abs_bps ~ post + post:treasury_based + C(series_id) + controls

The interaction post:treasury_based is identified by comparing:
  - treasury_based=1: TIPS-Treasury (2y,5y,10y), UST SF (2y,5y,10y) — 6 series
  - treasury_based=0: CIP (AUD,CAD,CHF,EUR,GBP,JPY,NZD,SEK), Equity (SPX,NDX,INDU) — 11 series

Per the hypothesis:
  - Post coefficient: effect for non-Treasury baseline (CIP + equity)
  - Post:treasury_based coefficient: DIFFERENTIAL effect for Treasury series
  - At entry (2020-04-01): both should be negative (compression), Treasury more negative
  - At exit  (2021-03-31): both may be positive (widening), Treasury more positive
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# ── Event-time utilities ────────────────────────────────────────────────────────

def add_event_time_trading(
    df: pd.DataFrame,
    event_date: str,
    group_col: str = "series_id",
) -> pd.DataFrame:
    """
    Assign trading-day event_time within each series.
    event_time=0 is the first trading date >= event_date in each series' own sample.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
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

    return out.groupby(group_col, group_keys=False).apply(_apply)


def compute_window_overlap(
    event_dates: list[str],
    window: int,
) -> list[dict]:
    """
    For each consecutive pair of events, compute trading-day overlap of
    the ±window windows.
    """
    rows = []
    bday = pd.tseries.offsets.BusinessDay
    for i in range(len(event_dates) - 1):
        e1 = pd.Timestamp(event_dates[i])
        e2 = pd.Timestamp(event_dates[i + 1])
        gap_cal = (e2 - e1).days
        gap_td = len(pd.bdate_range(e1, e2)) - 1  # trading days between

        # post-event boundary of e1: e1 + window t.d. (approx via BusinessDay)
        post_end_e1 = e1 + bday(window)
        # pre-event boundary of e2: e2 - window t.d.
        pre_start_e2 = e2 - bday(window)

        overlap = max(0, (post_end_e1 - pre_start_e2).days)
        overlap_td = max(0, len(pd.bdate_range(pre_start_e2, post_end_e1)) - 1)

        rows.append({
            "event_1": str(e1.date()),
            "event_2": str(e2.date()),
            "gap_cal_days": gap_cal,
            "gap_trading_days": gap_td,
            "window": window,
            "overlap_cal_days": overlap,
            "overlap_trading_days": overlap_td,
            "overlapping": overlap > 0,
        })
    return rows


# ── Regression engine ───────────────────────────────────────────────────────────

def _nw_cov(model, lags: int = 5):
    return model.get_robustcov_results(cov_type="HAC", maxlags=lags)


MIN_OBS_PER_PARAM = 10  # at least 10 obs per parameter


def run_pooled_jump_corrected(
    panel: pd.DataFrame,
    y_col: str,
    event_date: str,
    window: int,
    controls: list[str] | None = None,
    hac_lags: int = 5,
    series_fe_col: str = "series_id",
    interact_treasury: bool = True,
    truncate_window_at_half_gap: bool = True,
    next_event_date: str | None = None,
) -> dict:
    """
    Pooled event-window jump regression across all strategies.

    Model (interact_treasury=True):
        y_abs_bps ~ post + post:treasury_based + C(series_id) + controls

    Returns dict with estimates, SEs, N, and metadata.
    """
    # Apply event time per series
    work = add_event_time_trading(panel, event_date, group_col=series_fe_col)

    # Handle window truncation for overlapping events
    effective_window = window
    if truncate_window_at_half_gap and next_event_date is not None:
        e1 = pd.Timestamp(event_date)
        e2 = pd.Timestamp(next_event_date)
        gap_td = len(pd.bdate_range(e1, e2, freq="B")) - 1
        if gap_td < 2 * window:
            effective_window = gap_td // 2
            if effective_window < 3:
                warnings.warn(
                    f"Events {event_date} and {next_event_date} are only {gap_td} t.d. apart; "
                    f"effective window truncated to {effective_window} t.d. "
                    f"Consider combining into a single exit event."
                )

    # Filter to window
    sub = work[work["event_time"].between(-effective_window, effective_window)].copy()
    sub["post"] = (sub["event_time"] >= 0).astype(int)

    # Need variation in pre and post
    if sub["post"].nunique() < 2:
        return {"event": event_date, "window": effective_window, "error": "no_variation_pre_post", "n": len(sub)}

    # Resolve available controls
    avail_controls = [c for c in (controls or []) if c in sub.columns]

    # Build design matrix components
    if interact_treasury:
        base_terms = ["post", "np.float64(treasury_based)", "post:np.float64(treasury_based)"]
    else:
        base_terms = ["post"]

    ctrl_terms = avail_controls
    fe_term = f"C({series_fe_col})"
    all_terms = base_terms + ctrl_terms + [fe_term]
    formula = f"{y_col} ~ {' + '.join(all_terms)}"

    # Prepare regression data
    keep_cols = [y_col, "post", series_fe_col] + avail_controls
    if interact_treasury:
        keep_cols += ["treasury_based"]
    reg = sub[keep_cols].dropna().copy()

    # Coerce numerics
    for c in [y_col, "post"] + avail_controls:
        reg[c] = pd.to_numeric(reg[c], errors="coerce")
    if interact_treasury:
        reg["treasury_based"] = pd.to_numeric(reg["treasury_based"], errors="coerce").fillna(0)

    reg = reg.dropna()

    if len(reg) < 8:
        return {"event": event_date, "window": effective_window, "error": "too_few_obs", "n": len(reg)}

    # Check obs-to-param ratio (Bug 6 guard)
    n_fe = reg[series_fe_col].nunique()
    n_params_approx = len(base_terms) + len(ctrl_terms) + n_fe
    if len(reg) < MIN_OBS_PER_PARAM * n_params_approx:
        warnings.warn(
            f"Low obs/param ratio: {len(reg)} obs, ~{n_params_approx} params "
            f"(event={event_date}, window={effective_window})"
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "covariance of constraints")
            res = ols(formula, data=reg).fit()
            robust = _nw_cov(res, lags=hac_lags)
    except Exception as exc:
        return {"event": event_date, "window": effective_window, "error": str(exc), "n": len(reg)}

    names = list(robust.model.exog_names)
    out = {
        "event": event_date,
        "window": effective_window,
        "n": int(robust.nobs),
        "n_series": int(reg[series_fe_col].nunique()),
        "n_treasury": int(reg[reg["treasury_based"] == 1][series_fe_col].nunique()) if interact_treasury else np.nan,
        "n_nontreasury": int(reg[reg["treasury_based"] == 0][series_fe_col].nunique()) if interact_treasury else np.nan,
        "r2": float(res.rsquared),
    }

    # Extract coefficients
    def _get(name_fragment: str) -> tuple[float, float]:
        """Find coefficient matching name_fragment in exog_names."""
        matches = [n for n in names if name_fragment in n]
        if not matches:
            return np.nan, np.nan
        n = matches[0]
        idx = names.index(n)
        return float(robust.params[idx]), float(robust.bse[idx])

    coef_post, se_post = _get("post")
    out["coef_post"] = coef_post
    out["se_post"] = se_post
    out["t_post"] = coef_post / se_post if se_post > 0 else np.nan

    if interact_treasury:
        coef_inter, se_inter = _get("treasury_based")
        # the interaction coefficient: post:treasury_based
        inter_matches = [n for n in names if "post" in n and "treasury_based" in n]
        if inter_matches:
            idx = names.index(inter_matches[0])
            coef_inter = float(robust.params[idx])
            se_inter = float(robust.bse[idx])
        else:
            coef_inter, se_inter = np.nan, np.nan
        out["coef_post_x_treas"] = coef_inter
        out["se_post_x_treas"] = se_inter
        out["t_post_x_treas"] = coef_inter / se_inter if se_inter > 0 else np.nan

    return out


def run_series_level_jump(
    panel: pd.DataFrame,
    y_col: str,
    event_date: str,
    window: int,
    controls: list[str] | None = None,
    hac_lags: int = 5,
    series_fe_col: str = "series_id",
    effective_window: int | None = None,
) -> pd.DataFrame:
    """
    Run per-series jump regressions (no TreasuryBased interaction needed — constant within series).
    Returns one row per series_id.
    """
    from src.slr_bucket.econometrics.event_study import add_event_time

    ew = effective_window if effective_window is not None else window
    rows = []

    for sid, g in panel.groupby(series_fe_col):
        work = add_event_time(g, event_date)
        sub = work[work["event_time"].between(-ew, ew)].copy()
        if len(sub) < 8:
            continue
        sub["post"] = (sub["event_time"] >= 0).astype(int)
        if sub["post"].nunique() < 2:
            continue

        avail = [c for c in (controls or []) if c in sub.columns]
        keep = [y_col, "post"] + avail
        reg = sub[keep].dropna().copy()
        for c in [y_col, "post"] + avail:
            reg[c] = pd.to_numeric(reg[c], errors="coerce")
        reg = reg.dropna()

        if len(reg) < 8:
            continue

        X = sm.add_constant(reg[["post"] + avail].astype(float), has_constant="add")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "covariance of constraints")
                res = sm.OLS(reg[y_col].astype(float), X).fit()
                robust = _nw_cov(res, lags=hac_lags)
        except Exception:
            continue

        names = list(robust.model.exog_names)
        if "post" not in names:
            continue
        idx = names.index("post")
        coef = float(robust.params[idx])
        se = float(robust.bse[idx])

        rows.append({
            "series_id": sid,
            "event": event_date,
            "window": ew,
            "n": int(robust.nobs),
            "coef_post": coef,
            "se_post": se,
            "t_post": coef / se if se > 0 else np.nan,
            "ci_low": coef - 1.96 * se,
            "ci_high": coef + 1.96 * se,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from audit_repairs.fix_1_units import stack_all_outcomes

    repo_root = Path(__file__).parent.parent
    series_dir = repo_root / "data" / "series"
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("FIX 3: Pooled regression diagnostic")
    print("=" * 70)

    # Check window overlaps
    events = ["2020-04-01", "2021-03-19", "2021-03-31"]
    for W in [20, 60]:
        print(f"\nWindow overlap check (W={W}):")
        overlaps = compute_window_overlap(events, W)
        for o in overlaps:
            status = "OVERLAP" if o["overlapping"] else "OK"
            print(f"  {status}: {o['event_1']} vs {o['event_2']}: "
                  f"gap={o['gap_trading_days']} t.d., overlap={o['overlap_trading_days']} t.d.")

    print("\nLoading corrected panel...")
    panel = stack_all_outcomes(series_dir)
    print(f"Panel: {len(panel):,} rows, strategies: {sorted(panel['strategy'].unique())}")

    # Load controls
    ctrl_fred = pd.read_csv(repo_root / "data/raw/event_inputs/controls_vix_creditspreads_fred.csv")
    ctrl_fred["date"] = pd.to_datetime(ctrl_fred["date"])
    ctrl_repo = pd.read_csv(repo_root / "data/raw/event_inputs/repo_rates_combined.csv")
    ctrl_repo["date"] = pd.to_datetime(ctrl_repo["date"])
    # Compute spreads
    if "spr_tgcr" not in ctrl_repo.columns and {"SOFR", "TGCR"}.issubset(ctrl_repo.columns):
        ctrl_repo["spr_tgcr"] = pd.to_numeric(ctrl_repo["TGCR"], errors="coerce") - \
                                  pd.to_numeric(ctrl_repo["SOFR"], errors="coerce")
    if "spr_effr" not in ctrl_repo.columns and {"SOFR", "BGCR"}.issubset(ctrl_repo.columns):
        ctrl_repo["spr_effr"] = pd.to_numeric(ctrl_repo["BGCR"], errors="coerce") - \
                                  pd.to_numeric(ctrl_repo["SOFR"], errors="coerce")
    ctrl_repo = ctrl_repo.rename(columns={"SOFR": "SOFR_rate"})

    controls_df = ctrl_fred.merge(ctrl_repo[["date", "SOFR_rate", "spr_tgcr", "spr_effr"]],
                                   on="date", how="outer")
    panel = panel.merge(controls_df, on="date", how="left")

    # Run pooled regressions
    total_controls = ["VIX", "HY_OAS", "BAA10Y"]
    direct_controls = ["VIX", "HY_OAS", "BAA10Y", "SOFR_rate", "spr_tgcr"]
    event_specs = [
        ("2020-04-01", None),
        ("2021-03-19", "2021-03-31"),  # gap=9 t.d.; pass next event for truncation
        ("2021-03-31", None),
    ]

    pooled_rows = []
    for event, next_ev in event_specs:
        for W in [20, 60]:
            for spec, ctrl_set in [("TOTAL", total_controls), ("DIRECT", direct_controls)]:
                result = run_pooled_jump_corrected(
                    panel, y_col="y_abs_bps", event_date=event, window=W,
                    controls=ctrl_set, interact_treasury=True,
                    next_event_date=next_ev,
                )
                result["spec"] = spec
                pooled_rows.append(result)
                if "error" not in result:
                    print(f"\n[POOLED] event={event} W={W} {spec}: "
                          f"N={result['n']}, "
                          f"post={result.get('coef_post',np.nan):.2f}({result.get('se_post',np.nan):.2f}), "
                          f"post×treas={result.get('coef_post_x_treas',np.nan):.2f}({result.get('se_post_x_treas',np.nan):.2f})")

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out_dir / "fix3_pooled_jump.csv", index=False)
    print(f"\nSaved: {out_dir}/fix3_pooled_jump.csv")

    # Verify: treasury_based is NOT constant in the pooled panel
    print("\nVerification — treasury_based distribution in pooled panel:")
    print(panel.groupby("strategy")["treasury_based"].first().reset_index().to_string(index=False))
    treas_share = panel["treasury_based"].mean()
    assert 0 < treas_share < 1, "treasury_based must have both 0 and 1 values in the pooled panel"
    print(f"treasury_based share: {treas_share:.2%} (should be between 0 and 1)")
    print("PASS: post:treasury_based interaction is properly identified in the pooled panel.")
