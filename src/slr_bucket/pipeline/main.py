"""
main.py — SLR Event Study Pipeline

Runs the full analysis from raw data to publication outputs:
  1. Load and stack all arbitrage spread series (basis points, unit-corrected)
  2. Winsorize COVID-spike outliers (p1/p99 by series)
  3. Merge control variables (VIX, HY spreads, SOFR, rolling Treasury issuance)
  4. Pooled jump regressions: Post x TreasuryBased interaction, HAC SEs
  5. Series-level jump regressions
  6. LaTeX regression table
  7. Diagnostic time-series and event-path plots
  8. Panel DiD with 2019 pre-COVID baseline (clean entry estimate)
  9. Formatted result summary with significance stars

Outputs (in _output/pipeline/):
  jump_estimates_pooled.csv
  jump_estimates_by_series.csv
  did_clean_baseline.csv
  key_results_formatted.txt
  regression_table_v2.tex
  robustness_table.csv
  summary_stats_repaired.csv
  winsorize_log.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Resolve repo root (src/slr_bucket/pipeline/main.py → root)
REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings(
    "ignore",
    message=".*DataFrameGroupBy.apply operated on the grouping columns.*",
    category=FutureWarning,
)

from slr_bucket.pipeline.loader    import stack_all_outcomes
from slr_bucket.pipeline.windows   import check_all_overlaps, get_clean_events_and_windows, trading_day_gap
from slr_bucket.pipeline.hac       import check_design_matrix, MIN_OBS_PER_PARAM
from slr_bucket.pipeline.winsorize import winsorize_panel
from slr_bucket.pipeline.issuance  import build_issuance_controls
from slr_bucket.pipeline.did       import run_did_both_specs
from slr_bucket.pipeline.results   import format_results

OUT_DIR = REPO_ROOT / "_output" / "pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "diagnostic_plots").mkdir(exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
EVENTS        = ["2020-04-01", "2021-03-19", "2021-03-31"]
WINDOWS       = [20, 60]
HAC_LAGS      = 5
TENOR_SUBSET  = [2, 5, 10]
EQUITY_INDICES = ["SPX", "NDX", "INDU"]
EVENT_BINS    = [(-60, -41), (-40, -21), (-20, -1), (0, 0), (1, 20), (21, 40), (41, 60)]
SAMPLE_START  = "2019-01-01"
SAMPLE_END    = "2021-12-31"

TOTAL_CONTROLS  = ["VIX", "HY_OAS", "BAA10Y"]
DIRECT_CONTROLS = ["VIX", "HY_OAS", "BAA10Y", "SOFR_rate", "spr_tgcr",
                   "issu_7_bil", "issu_14_bil", "issu_30_bil"]

MAIN_EVENTS = ["2020-04-01", "2021-03-31"]   # excludes 2021-03-19 (overlap)

SERIES_DIR = REPO_ROOT / "data" / "series"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_controls() -> pd.DataFrame:
    """Load and merge all control variables."""
    ctrl = pd.read_csv(REPO_ROOT / "data/raw/event_inputs/controls_vix_creditspreads_fred.csv")
    ctrl["date"] = pd.to_datetime(ctrl["date"])

    repo = pd.read_csv(REPO_ROOT / "data/raw/event_inputs/repo_rates_combined.csv")
    repo["date"] = pd.to_datetime(repo["date"])

    if "spr_tgcr" not in repo.columns and {"SOFR", "TGCR"}.issubset(repo.columns):
        repo["spr_tgcr"] = (
            pd.to_numeric(repo["TGCR"], errors="coerce") -
            pd.to_numeric(repo["SOFR"], errors="coerce")
        )
    if "spr_effr" not in repo.columns and {"SOFR", "BGCR"}.issubset(repo.columns):
        repo["spr_effr"] = (
            pd.to_numeric(repo["BGCR"], errors="coerce") -
            pd.to_numeric(repo["SOFR"], errors="coerce")
        )

    repo = repo.rename(columns={"SOFR": "SOFR_rate"})
    keep_repo = [c for c in ["date", "SOFR_rate", "spr_tgcr", "spr_effr"] if c in repo.columns]
    controls  = ctrl.merge(repo[keep_repo], on="date", how="outer").sort_values("date")

    issuance_path = REPO_ROOT / "data/raw/event_inputs/treasury_issuance_by_tenor_fiscaldata.csv"
    if issuance_path.exists():
        issu = build_issuance_controls(issuance_path, date_range=(SAMPLE_START, SAMPLE_END))
        controls = controls.merge(issu, on="date", how="left")
        print("  Issuance controls merged: issu_7_bil, issu_14_bil, issu_30_bil")
    else:
        print(f"  WARNING: Issuance file not found at {issuance_path}")

    return controls


def load_full_panel() -> pd.DataFrame:
    """Load all outcomes, merge controls, filter to sample window."""
    print("Loading outcomes (unit-aware loader)...")
    panel = stack_all_outcomes(SERIES_DIR, tenor_subset=TENOR_SUBSET, equity_indices=EQUITY_INDICES)

    print("Winsorizing outcome variables (p1/p99 by series)...")
    panel = winsorize_panel(
        panel,
        pct_lo=1,
        pct_hi=99,
        by_series=True,
        log_path=OUT_DIR / "winsorize_log.csv",
    )

    print("Loading controls...")
    controls = load_controls()

    panel = panel.merge(controls, on="date", how="left")
    panel = panel[panel["date"].between(SAMPLE_START, SAMPLE_END)].copy()
    print(
        f"Full panel: {len(panel):,} rows, "
        f"{panel['series_id'].nunique()} series, "
        f"{panel['date'].nunique()} dates"
    )
    return panel


# ── Event-time assignment ──────────────────────────────────────────────────────

def add_event_time_all_series(panel: pd.DataFrame, event_date: str) -> pd.DataFrame:
    """Assign trading-day event_time within each series."""
    out  = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    t0   = pd.Timestamp(event_date)

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        g     = g.sort_values("date").copy()
        dates = g["date"].values
        idx0  = int(np.searchsorted(dates, np.datetime64(t0), side="left"))
        if idx0 >= len(dates):
            idx0 = len(dates) - 1
        g["event_t0_used"] = pd.Timestamp(dates[idx0])
        g["event_time"]    = np.arange(len(g), dtype=int) - idx0
        return g

    return out.groupby("series_id", group_keys=False).apply(_apply)


# ── HAC regression helpers ─────────────────────────────────────────────────────

def nw_fit(y: pd.Series, X: pd.DataFrame, lags: int = HAC_LAGS):
    """Fit OLS with Newey-West HAC covariance. Returns robust result or None."""
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=const_cols, errors="ignore")
    if X.empty:
        return None

    X_c    = sm.add_constant(X.astype(float), has_constant="add")
    common = y.index.intersection(X_c.index)
    y2, X2 = y.loc[common].astype(float), X_c.loc[common]
    mask   = y2.notna() & X2.notna().all(axis=1)
    y2, X2 = y2[mask], X2[mask]

    if len(y2) < 8 or y2.nunique() < 2:
        return None

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
    names   = list(robust.model.exog_names)
    if name_fragment in names:
        idx = names.index(name_fragment)
        return float(robust.params[idx]), float(robust.bse[idx])
    matches = [n for n in names if name_fragment in n]
    if matches:
        idx = names.index(matches[0])
        return float(robust.params[idx]), float(robust.bse[idx])
    return np.nan, np.nan


# ── Summary statistics ─────────────────────────────────────────────────────────

def compute_summary_stats(panel: pd.DataFrame) -> pd.DataFrame:
    regimes = {
        "pre":    ("2019-01-01", "2020-03-31"),
        "relief": ("2020-04-01", "2021-03-31"),
        "post":   ("2021-04-01", "2021-12-31"),
    }
    rows = []
    for series_id, g in panel.groupby("series_id"):
        strategy = g["strategy"].iloc[0]
        tenor    = g["tenor"].iloc[0]
        treas    = int(g["treasury_based"].iloc[0])
        for regime, (start, end) in regimes.items():
            sub = g[g["date"].between(start, end)]
            y   = sub["y_bps"].dropna()
            ya  = sub["y_abs_bps"].dropna()
            if len(ya) == 0:
                continue
            rows.append({
                "strategy": strategy, "series_id": series_id, "tenor": tenor,
                "treasury_based": treas, "regime": regime,
                "N_days": len(ya),
                "mean_W": float(y.mean()), "median_W": float(y.median()),
                "mean_absW": float(ya.mean()), "median_absW": float(ya.median()),
                "p5_W": float(y.quantile(0.05)), "p95_W": float(y.quantile(0.95)),
            })
    return pd.DataFrame(rows).sort_values(["strategy", "series_id", "regime"])


# ── Pooled jump regression ─────────────────────────────────────────────────────

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

    tb_vals = work["treasury_based"].dropna().unique()
    if len(tb_vals) < 2:
        return {
            "event": event_date, "window": window, "spec": spec,
            "error": f"treasury_based is constant={tb_vals[0]}",
        }

    avail_ctrl = [c for c in controls if c in work.columns]
    keep_cols  = ["y_abs_bps", "post", "treasury_based", "series_id"] + avail_ctrl
    reg        = work[keep_cols].dropna().copy()
    for c in ["y_abs_bps", "post", "treasury_based"] + avail_ctrl:
        reg[c] = pd.to_numeric(reg[c], errors="coerce")
    reg = reg.dropna()

    reg["post_x_treas"] = reg["post"] * reg["treasury_based"]
    fe = pd.get_dummies(reg["series_id"].astype(str), prefix="fe", drop_first=True)
    X  = pd.concat(
        [reg[["post", "post_x_treas"] + avail_ctrl].reset_index(drop=True),
         fe.reset_index(drop=True)],
        axis=1,
    )
    y = reg["y_abs_bps"].reset_index(drop=True)

    robust = nw_fit(y, X, lags=HAC_LAGS)
    if robust is None:
        return {"event": event_date, "window": window, "spec": spec, "error": "fit_failed", "n": len(reg)}

    coef_post, se_post = extract_coef(robust, "post")
    names = list(robust.model.exog_names)
    inter_matches = [n for n in names if n == "post_x_treas"]
    if inter_matches:
        idx = names.index(inter_matches[0])
        coef_inter, se_inter = float(robust.params[idx]), float(robust.bse[idx])
    else:
        coef_inter, se_inter = np.nan, np.nan

    ctrl_coefs = {}
    for c in avail_ctrl:
        v, se = extract_coef(robust, c)
        ctrl_coefs[f"coef_{c}"] = v
        ctrl_coefs[f"se_{c}"]   = se

    return {
        "event":   event_date, "window": window, "spec": spec,
        "n":       int(robust.nobs),
        "n_treasury_series":    int((reg.groupby("series_id")["treasury_based"].first() == 1).sum()),
        "n_nontreasury_series": int((reg.groupby("series_id")["treasury_based"].first() == 0).sum()),
        "r2":      float(robust.rsquared),
        "coef_post":        coef_post,
        "se_post":          se_post,
        "t_post":           coef_post / se_post if (np.isfinite(se_post) and se_post > 0) else np.nan,
        "coef_post_x_treas": coef_inter,
        "se_post_x_treas":   se_inter,
        "t_post_x_treas":    coef_inter / se_inter if (np.isfinite(se_inter) and se_inter > 0) else np.nan,
        **ctrl_coefs,
    }


# ── Series-level jump regressions ──────────────────────────────────────────────

def run_series_level_jumps(
    panel: pd.DataFrame,
    event_date: str,
    window: int,
    controls: list[str],
    spec: str,
) -> list[dict]:
    """Per-series jump regressions (no interaction — treasury_based constant per series)."""
    work = add_event_time_all_series(panel, event_date)
    work = work[work["event_time"].between(-window, window)].copy()
    work["post"] = (work["event_time"] >= 0).astype(int)

    rows = []
    for sid, g in work.groupby("series_id"):
        avail = [c for c in controls if c in g.columns]
        keep  = ["y_abs_bps", "post"] + avail
        reg   = g[keep].dropna().copy()
        for c in keep:
            reg[c] = pd.to_numeric(reg[c], errors="coerce")
        reg = reg.dropna()
        if len(reg) < 8 or reg["post"].nunique() < 2:
            continue
        robust = nw_fit(reg["y_abs_bps"], reg[["post"] + avail])
        if robust is None:
            continue
        coef, se = extract_coef(robust, "post")
        rows.append({
            "series_id":      sid,
            "strategy":       g["strategy"].iloc[0],
            "treasury_based": int(g["treasury_based"].iloc[0]),
            "event": event_date, "window": window, "spec": spec,
            "n":   int(robust.nobs),
            "coef_post": coef, "se_post": se,
            "t_post":    coef / se if (np.isfinite(se) and se > 0) else np.nan,
            "ci_low":  coef - 1.96 * se if np.isfinite(se) else np.nan,
            "ci_high": coef + 1.96 * se if np.isfinite(se) else np.nan,
        })
    return rows


# ── Binned event study ─────────────────────────────────────────────────────────

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
    et   = work["event_time"]
    work["bin"] = pd.NA
    for lo, hi in bins:
        work.loc[(et >= lo) & (et <= hi), "bin"] = f"bin_{lo}_{hi}"

    if series_id is not None:
        work = work[work["series_id"] == series_id].copy()
    work = work.dropna(subset=["bin", "y_abs_bps"]).copy()
    if len(work) == 0:
        return pd.DataFrame()

    dummies = pd.get_dummies(work["bin"], prefix="")
    dummies.columns = [c.strip("_") for c in dummies.columns]
    ref_label = "bin_-20_-1"
    if ref_label in dummies.columns:
        dummies = dummies.drop(columns=[ref_label])
    elif len(dummies.columns) > 0:
        dummies = dummies.drop(columns=[dummies.columns[0]])

    avail = [c for c in controls if c in work.columns]
    X     = pd.concat(
        [dummies.reset_index(drop=True), work[avail].reset_index(drop=True)],
        axis=1,
    )
    y = work["y_abs_bps"].reset_index(drop=True)

    robust = nw_fit(y, X)
    if robust is None:
        return pd.DataFrame()

    names = list(robust.model.exog_names)
    out   = []
    for col in dummies.columns:
        if col not in names:
            continue
        coef, se = extract_coef(robust, col)
        m = re.search(r"(-?\d+)_(-?\d+)", col)
        bin_mid = 0.5 * (int(m.group(1)) + int(m.group(2))) if m else np.nan
        out.append({
            "bin_label": col, "bin_mid": bin_mid,
            "estimate": coef, "se": se,
            "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se,
            "n": int(robust.nobs),
        })
    return pd.DataFrame(out)


# ── LaTeX table ────────────────────────────────────────────────────────────────

def make_latex_table(pooled: pd.DataFrame, by_series: pd.DataFrame) -> str:
    """Generate LaTeX regression table."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Event-Window Jump Estimates: SLR Relief Entry and Exit}",
        r"\label{tab:jump_estimates}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\hline\hline",
        r" & \multicolumn{3}{c}{Entry: 2020-04-01} & \multicolumn{3}{c}{Exit: 2021-03-31} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & W=20 TOTAL & W=20 DIRECT & W=60 TOTAL & W=20 TOTAL & W=20 DIRECT & W=60 TOTAL \\",
        r"\hline",
        r"\multicolumn{7}{l}{\textit{Panel A: Pooled across all strategies}} \\",
    ]

    def _fmt(coef, se, t_stat=None):
        if not np.isfinite(coef):
            return "---"
        s = ""
        if t_stat is not None and np.isfinite(t_stat):
            a = abs(t_stat)
            s = "***" if a > 2.576 else "**" if a > 1.960 else "*" if a > 1.645 else ""
        return rf"{coef:.2f}{s} & ({se:.2f})"

    for coef_col, se_col, t_col, label in [
        ("coef_post",         "se_post",         "t_post",
         r"$\hat\beta_{\text{Post}}$"),
        ("coef_post_x_treas", "se_post_x_treas", "t_post_x_treas",
         r"$\hat\beta_{\text{Post}\times\text{Treasury}}$"),
    ]:
        coef_row = [label]
        se_row   = [""]
        for ev in ["2020-04-01", "2021-03-31"]:
            for W in [20, 20, 60]:
                mask = (pooled["event"] == ev) & (pooled["window"] == W)
                sub  = pooled[mask]
                if sub.empty:
                    coef_row.append("---"); se_row.append("")
                else:
                    row = sub.iloc[0]
                    c   = row.get(coef_col, np.nan)
                    s   = row.get(se_col,   np.nan)
                    t   = row.get(t_col,    np.nan)
                    coef_row.append(
                        (f"{c:.2f}" + ("***" if abs(t) > 2.576 else "**" if abs(t) > 1.96
                                       else "*" if abs(t) > 1.645 else ""))
                        if np.isfinite(c) else "---"
                    )
                    se_row.append(f"({s:.2f})" if np.isfinite(s) else "")
        lines.append(" & ".join(coef_row) + r" \\")
        lines.append(" & ".join(se_row)   + r" \\")

    n_row = ["$N$"]
    for ev in ["2020-04-01", "2021-03-31"]:
        for W in [20, 20, 60]:
            mask = (pooled["event"] == ev) & (pooled["window"] == W)
            sub  = pooled[mask]
            n_row.append(str(int(sub["n"].iloc[0])) if not sub.empty and "n" in sub.columns else "0")
    lines.append(" & ".join(n_row) + r" \\")

    lines += [
        r"\hline\hline",
        r"\multicolumn{7}{l}{\footnotesize HAC standard errors (Newey-West, 5 lags) in parentheses.} \\",
        r"\multicolumn{7}{l}{\footnotesize * p<0.10, ** p<0.05, *** p<0.01.} \\",
        r"\multicolumn{7}{l}{\footnotesize Series FE included in pooled regressions.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Diagnostic plots ───────────────────────────────────────────────────────────

def plot_time_series(panel: pd.DataFrame, out_dir: Path) -> None:
    """Plot |W| time series for each strategy with event dates marked."""
    event_colors = {"2020-04-01": "green", "2021-03-19": "orange", "2021-03-31": "red"}
    strategies   = sorted(panel["strategy"].unique())
    fig, axes    = plt.subplots(len(strategies), 1, figsize=(12, 3 * len(strategies)), sharex=True)
    if len(strategies) == 1:
        axes = [axes]
    for ax, strat in zip(axes, strategies):
        g = panel[panel["strategy"] == strat]
        for sid, sg in g.groupby("series_id"):
            sg = sg.sort_values("date")
            ax.plot(sg["date"], sg["y_abs_bps"], lw=0.8, alpha=0.6, label=sid)
        for ev, col in event_colors.items():
            ax.axvline(pd.Timestamp(ev), color=col, lw=1.5, linestyle="--")
        ax.set_ylabel("|W| (bps)")
        ax.set_title(f"Strategy: {strat}")
        ax.legend(fontsize=6, ncol=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Arbitrage Spread Magnitude |W| by Strategy", fontsize=12)
    fig.tight_layout()
    path = out_dir / "diagnostic_plots" / "time_series_all_strategies.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_event_paths(panel: pd.DataFrame, event_date: str, out_dir: Path, controls: list[str] | None = None) -> None:
    """Binned event-path plots for each strategy."""
    controls   = controls or TOTAL_CONTROLS
    strategies = sorted(panel["strategy"].unique())
    fig, axes  = plt.subplots(1, len(strategies), figsize=(4 * len(strategies), 4), sharey=False)
    if len(strategies) == 1:
        axes = [axes]
    for ax, strat in zip(axes, strategies):
        g   = panel[panel["strategy"] == strat]
        sid = sorted(g["series_id"].unique())[0]
        bin_df = run_binned_event_study(panel, event_date, EVENT_BINS, controls, series_id=sid)
        if bin_df.empty:
            ax.set_title(f"{strat}\n(no data)")
            continue
        bin_df = bin_df.sort_values("bin_mid")
        ax.errorbar(bin_df["bin_mid"], bin_df["estimate"],
                    yerr=1.96 * bin_df["se"], fmt="o-", capsize=3, lw=1.5)
        ax.axvline(0, color="red", lw=1, linestyle="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Event time (trading days)")
        ax.set_ylabel("Delta|W| vs pre-period (bps)")
        ax.set_title(f"{strat}\n({sid})")
    fig.suptitle(f"Event paths -- {event_date}", fontsize=11)
    fig.tight_layout()
    fname = f"event_path_{event_date.replace('-', '')}.png"
    fig.savefig(out_dir / "diagnostic_plots" / fname, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved: {out_dir}/diagnostic_plots/{fname}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("SLR Event Study Pipeline")
    print("=" * 70)

    # Step 1: Overlap check
    print("\n[Step 1] Event window overlap check")
    overlap_df = check_all_overlaps(EVENTS, WINDOWS)
    for _, row in overlap_df.iterrows():
        if row["is_overlap"]:
            print(
                f"  OVERLAP: {row['event_1']} vs {row['event_2']}, "
                f"W={row['window']}, overlap={row['overlap_trading_days']} t.d."
            )
    print(f"  Using MAIN_EVENTS = {MAIN_EVENTS} for regression tables")

    # Step 2: Load data
    print("\n[Step 2] Loading panel")
    panel = load_full_panel()

    # Step 3: Summary statistics
    print("\n[Step 3] Computing summary statistics")
    summary = compute_summary_stats(panel)
    summary.to_csv(OUT_DIR / "summary_stats_repaired.csv", index=False)
    print(f"  Saved: {OUT_DIR}/summary_stats_repaired.csv")
    print("\n  Mean |W| by strategy/regime:")
    for (strat, regime), g in summary.groupby(["strategy", "regime"]):
        print(f"    {strat:20s} {regime:8s}: {g['mean_absW'].mean():.2f} bps")

    # Step 4: Pooled jump regressions
    print("\n[Step 4] Pooled jump regressions")
    pooled_rows = []
    for event in MAIN_EVENTS:
        for W in WINDOWS:
            for spec, ctrl_set in [("TOTAL", TOTAL_CONTROLS), ("DIRECT", DIRECT_CONTROLS)]:
                result = run_pooled_jump(panel, event, W, ctrl_set, spec)
                pooled_rows.append(result)
                if "error" not in result:
                    print(
                        f"  event={event} W={W:2d} {spec:6s}: "
                        f"N={result['n']:5d}, "
                        f"post={result.get('coef_post', np.nan):+7.2f} "
                        f"({result.get('se_post', np.nan):.2f}), "
                        f"post*T={result.get('coef_post_x_treas', np.nan):+7.2f} "
                        f"({result.get('se_post_x_treas', np.nan):.2f})"
                    )
                else:
                    print(f"  event={event} W={W:2d} {spec:6s}: ERROR={result['error']}")

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(OUT_DIR / "jump_estimates_pooled.csv", index=False)
    print(f"\n  Saved: {OUT_DIR}/jump_estimates_pooled.csv")

    # Step 5: Series-level jump regressions
    print("\n[Step 5] Series-level jump regressions")
    series_rows = []
    for event in MAIN_EVENTS:
        for W in WINDOWS:
            for spec, ctrl_set in [("TOTAL", TOTAL_CONTROLS), ("DIRECT", DIRECT_CONTROLS)]:
                series_rows.extend(run_series_level_jumps(panel, event, W, ctrl_set, spec))
    series_df = pd.DataFrame(series_rows)
    series_df.to_csv(OUT_DIR / "jump_estimates_by_series.csv", index=False)
    print(f"  Saved: {OUT_DIR}/jump_estimates_by_series.csv ({len(series_df)} rows)")

    if not series_df.empty:
        print("\n  Key series estimates (W=60, DIRECT):")
        sub = series_df[(series_df["window"] == 60) & (series_df["spec"] == "DIRECT")]
        for _, row in sub.sort_values(["treasury_based", "series_id"], ascending=[False, True]).iterrows():
            print(
                f"    {row['series_id']:25s} {row['event']} "
                f"post={row['coef_post']:+7.2f} ({row['se_post']:.2f}) N={row['n']:4d}"
            )

    # Step 6: LaTeX table
    print("\n[Step 6] Generating LaTeX table")
    if not pooled_df.empty and not series_df.empty:
        tex = make_latex_table(pooled_df, series_df)
        with open(OUT_DIR / "regression_table_repaired.tex", "w") as f:
            f.write(tex)
        print(f"  Saved: {OUT_DIR}/regression_table_repaired.tex")

    # Step 7: Diagnostic plots
    print("\n[Step 7] Generating diagnostic plots")
    plot_time_series(panel, OUT_DIR)
    for event in MAIN_EVENTS:
        plot_event_paths(panel, event, OUT_DIR, controls=TOTAL_CONTROLS)

    # Step 8: Clean DiD
    print("\n[Step 8] Panel DiD (2019 pre-COVID baseline)")
    did_df = run_did_both_specs(panel)
    did_df.to_csv(OUT_DIR / "did_clean_baseline.csv", index=False)
    print(f"  Saved: {OUT_DIR}/did_clean_baseline.csv")

    # Step 9: Format results
    print("\n[Step 9] Formatting results")
    format_results(pooled_df, did_df, out_dir=OUT_DIR)

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print(f"All outputs saved to: {OUT_DIR}")
    print("=" * 70)

    # Hypothesis check
    print("\n[Check] Theoretical prediction verification")
    entry_pooled = pooled_df[
        (pooled_df["event"] == "2020-04-01") & (pooled_df["window"] == 60)
    ]
    for _, row in entry_pooled.iterrows():
        post  = row.get("coef_post",         np.nan)
        inter = row.get("coef_post_x_treas", np.nan)
        print(f"\n  Entry W=60 {row.get('spec', '?')}:")
        print(f"    Post (non-Treasury baseline): {post:+.2f} bps")
        print(f"    Post x Treasury (differential): {inter:+.2f} bps")
        if np.isfinite(post) and np.isfinite(inter):
            print(f"    Total Treasury effect: {post + inter:+.2f} bps")
            print(f"    Post sign:  {'PASS' if post < 0 else 'FAIL'}: compression")
            print(f"    Post*T sign: {'PASS' if inter < 0 else 'WARN'}: more compression")


if __name__ == "__main__":
    main()
