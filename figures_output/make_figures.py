"""
make_figures.py — Publication-quality figures for SLR exclusion paper.
All 6 figures saved to figures_output/ as PDF + PNG.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path

# ── Global style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "ust":    "#1a3a6b",
    "tips":   "#2e86ab",
    "cip":    "#b5451b",
    "equity": "#5c4a72",
    "entry":  "#c0392b",
    "exit":   "#27ae60",
    "shade":  "#f0f0f0",
    "ci":     "#aaaaaa",
}

SLR_ENTRY = pd.Timestamp("2020-04-01")
SLR_EXIT  = pd.Timestamp("2021-03-31")
REPO_ROOT = Path(__file__).parent.parent
OUT_DIR   = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_panel():
    """Build master long-format panel: date | series_id | strategy | tenor |
    y_bps | y_abs_bps | treasury_based | regime"""
    records = []

    # 1. CIP
    cip = pd.read_csv(REPO_ROOT / "data/series/cip_spreads_3m_bps.csv",
                      parse_dates=["Date"])
    cip_cols = {
        "CIP_AUD_ln": "AUD", "CIP_CAD_ln": "CAD", "CIP_CHF_ln": "CHF",
        "CIP_EUR_ln": "EUR", "CIP_GBP_ln": "GBP", "CIP_JPY_ln": "JPY",
        "CIP_NZD_ln": "NZD", "CIP_SEK_ln": "SEK",
    }
    for col, tenor in cip_cols.items():
        tmp = cip[["Date", col]].dropna().copy()
        tmp.columns = ["date", "y_bps"]
        tmp["series_id"] = f"cip_{tenor.lower()}"
        tmp["strategy"] = "cip"
        tmp["tenor"] = tenor
        tmp["treasury_based"] = 0
        records.append(tmp)

    # 2. UST SF (use 2Y, 5Y, 10Y only)
    ust = pd.read_csv(REPO_ROOT / "data/series/treasury_sf_output.csv",
                      parse_dates=["Date"])
    ust_cols = {
        "Treasury_SF_2Y": "2Y",
        "Treasury_SF_5Y": "5Y",
        "Treasury_SF_10Y": "10Y",
    }
    for col, tenor in ust_cols.items():
        tmp = ust[["Date", col]].dropna().copy()
        tmp.columns = ["date", "y_bps"]
        tmp["series_id"] = f"ust_sf_{tenor.lower()}"
        tmp["strategy"] = "ust_spot_fut"
        tmp["tenor"] = tenor
        tmp["treasury_based"] = 1
        records.append(tmp)

    # 3. TIPS (arb_2, arb_5, arb_10)
    tips = pd.read_parquet(REPO_ROOT / "data/series/tips_treasury_implied_rf_2010.parquet")
    tips = tips.reset_index() if tips.index.name == "date" else tips
    # Ensure date column name
    date_col = "date" if "date" in tips.columns else tips.columns[0]
    tips[date_col] = pd.to_datetime(tips[date_col])
    tips_cols = {"arb_2": "2Y", "arb_5": "5Y", "arb_10": "10Y"}
    for col, tenor in tips_cols.items():
        tmp = tips[[date_col, col]].dropna().copy()
        tmp.columns = ["date", "y_bps"]
        tmp["series_id"] = f"tips_treas_{tenor.lower()}"
        tmp["strategy"] = "tips"
        tmp["tenor"] = tenor
        tmp["treasury_based"] = 1
        records.append(tmp)

    # 4. Equity
    eq_files = {
        "equity_spot_spread_SPX.csv": ("eq_spx",  "SPX",  "spread_SPX"),
        "equity_spot_spread_NDX.csv": ("eq_ndx",  "NDX",  "spread_NDX"),
        "equity_spot_spread_INDU.csv": ("eq_indu", "INDU", "spread_INDU"),
    }
    for fname, (sid, tenor, col) in eq_files.items():
        df = pd.read_csv(REPO_ROOT / "data/series" / fname, parse_dates=["Date"])
        tmp = df[["Date", col]].dropna().copy()
        tmp.columns = ["date", "y_bps"]
        tmp["series_id"] = sid
        tmp["strategy"] = "eq_spot_fut"
        tmp["tenor"] = tenor
        tmp["treasury_based"] = 0
        records.append(tmp)

    panel = pd.concat(records, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])

    # ── CRITICAL unit check ──────────────────────────────────────────────────
    ust_pre = panel[
        (panel["strategy"] == "ust_spot_fut") &
        (panel["date"] >= "2019-01-01") &
        (panel["date"] <= "2020-03-31")
    ]["y_bps"].abs().mean()
    print(f"[UNIT CHECK] UST SF pre-period mean |W|: {ust_pre:.2f} bps")
    if ust_pre > 200:
        raise ValueError(
            f"UST SF pre-period mean |W| = {ust_pre:.2f} bps — series appears "
            "NOT in bps. Halt. Check treasury_sf_output.csv scale."
        )
    if ust_pre < 5:
        raise ValueError(
            f"UST SF pre-period mean |W| = {ust_pre:.2f} bps — suspiciously small. "
            "Check data."
        )
    print("[UNIT CHECK] PASSED.")

    # ── Restrict to analysis window ─────────────────────────────────────────
    panel = panel[
        (panel["date"] >= "2019-01-01") & (panel["date"] <= "2021-12-31")
    ].copy()

    # ── Winsorise within-series at p1/p99 ──────────────────────────────────
    def winsorise_series(grp):
        lo = grp["y_bps"].quantile(0.01)
        hi = grp["y_bps"].quantile(0.99)
        grp = grp.copy()
        grp["y_bps"] = grp["y_bps"].clip(lo, hi)
        return grp

    panel = panel.groupby("series_id", group_keys=False).apply(winsorise_series)
    panel["y_abs_bps"] = panel["y_bps"].abs()

    # ── Regime labels ───────────────────────────────────────────────────────
    def assign_regime(d):
        if d < SLR_ENTRY:
            return "pre"
        elif d <= SLR_EXIT:
            return "relief"
        else:
            return "post"

    panel["regime"] = panel["date"].map(assign_regime)

    return panel.reset_index(drop=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_slr_shading(ax):
    ax.axvspan(SLR_ENTRY, SLR_EXIT, alpha=0.12, color="#cccccc", zorder=0)
    ax.axvline(SLR_ENTRY, color=COLORS["entry"], ls='--', lw=0.9, zorder=2)
    ax.axvline(SLR_EXIT,  color=COLORS["exit"],  ls='--', lw=0.9, zorder=2)


def format_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)


def sig_stars(t):
    at = abs(t)
    if at > 2.576:
        return "***"
    elif at > 1.960:
        return "**"
    elif at > 1.645:
        return "*"
    return ""


def save_fig(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}")
    print(f"  Saved {stem}.pdf / .png")


# ── Figure 1 — Time Series Overview (2×2) ─────────────────────────────────────

def fig1_series_overview(panel):
    print("Generating Figure 1...")
    strategies = [
        ("ust_spot_fut", "UST Spot-Futures",  COLORS["ust"]),
        ("tips",         "TIPS-Treasury",      COLORS["tips"]),
        ("cip",          "CIP (3m FX)",         COLORS["cip"]),
        ("eq_spot_fut",  "Equity Spot-Futures", COLORS["equity"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True)
    axes_flat = axes.flatten()

    for idx, (strat, title, color) in enumerate(strategies):
        ax = axes_flat[idx]
        sub = panel[panel["strategy"] == strat].copy()
        grp = (sub.groupby("date")["y_abs_bps"]
                  .agg(["mean", "std"])
                  .reset_index()
                  .sort_values("date"))

        grp["mean_sm"] = grp["mean"].rolling(5, min_periods=1).mean()
        grp["std_sm"]  = grp["std"].rolling(5,  min_periods=1).mean()

        add_slr_shading(ax)
        ax.fill_between(grp["date"],
                        grp["mean_sm"] - grp["std_sm"],
                        grp["mean_sm"] + grp["std_sm"],
                        alpha=0.2, color=color, lw=0)
        ax.plot(grp["date"], grp["mean_sm"], color=color, lw=1.2, label=title)

        if idx == 0:
            ymax = ax.get_ylim()[1]
            ax.text(SLR_ENTRY + pd.Timedelta(days=5), 0.93,
                    "Entry", transform=ax.get_xaxis_transform(),
                    color=COLORS["entry"], fontsize=7, va='top')
            ax.text(SLR_EXIT + pd.Timedelta(days=5), 0.93,
                    "Exit", transform=ax.get_xaxis_transform(),
                    color=COLORS["exit"], fontsize=7, va='top')

        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Spread |W| (bps)", fontsize=8)
        format_xaxis(ax)

    fig.tight_layout(pad=1.2)
    save_fig(fig, "fig1_series_overview")
    plt.close(fig)

    # Caption file
    caption = (
        "Figure 1. Time-series evolution of arbitrage spreads, 2019–2021. "
        "Each panel shows the cross-series mean (solid line) and ±1 standard deviation band (shaded) "
        "of absolute spread |W| in basis points, smoothed with a 5-day rolling average. "
        "The gray shaded region denotes the SLR exclusion relief period (April 1, 2020 – March 31, 2021). "
        "Dashed vertical lines mark the entry (red) and exit (green) dates."
    )
    (OUT_DIR / "fig1_series_overview_caption.txt").write_text(caption)
    print("  Caption saved.")


# ── Figure 2 — UST SF Detail by Tenor ─────────────────────────────────────────

def fig2_ust_sf_detail(panel):
    print("Generating Figure 2...")

    tenors = [
        ("2Y", "#1a3a6b", "solid",  "2-year"),
        ("5Y", "#2c5282", "dashed", "5-year"),
        ("10Y","#4a7ba7", "dotted", "10-year"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.5))
    add_slr_shading(ax)

    pre_mask = (panel["date"] >= "2019-01-01") & (panel["date"] <= "2020-01-31")

    for tenor, color, ls, label in tenors:
        sid = f"ust_sf_{tenor.lower()}"
        sub = panel[panel["series_id"] == sid].copy().sort_values("date")
        sub["y_sm"] = sub["y_abs_bps"].rolling(5, min_periods=1).mean()

        ax.plot(sub["date"], sub["y_sm"], color=color, ls=ls, lw=1.2, label=label)

        # Pre-period mean reference line
        pre_mean = sub.loc[pre_mask[sub.index], "y_abs_bps"].mean()
        if np.isfinite(pre_mean):
            ax.axhline(pre_mean, color=color, ls=':', lw=0.8, alpha=0.6)
            ax.text(pd.Timestamp("2021-11-01"), pre_mean,
                    f"{tenor} pre: {pre_mean:.1f}",
                    color=color, fontsize=6.5, va='center')

    ax.set_ylabel("Spread |W| (bps)", fontsize=8)
    ax.set_title("UST Spot-Futures Spread by Tenor", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    format_xaxis(ax)
    fig.tight_layout(pad=0.8)
    save_fig(fig, "fig2_ust_sf_detail")
    plt.close(fig)

    caption = (
        "Figure 2. UST spot-futures arbitrage spread by tenor, 2019–2021. "
        "Lines show the absolute spread |W| for 2-year (solid), 5-year (dashed), "
        "and 10-year (dotted) maturities, smoothed with a 5-day rolling average. "
        "Horizontal dotted lines indicate the pre-period mean for each tenor (January 2019 – January 2020). "
        "The gray shaded region and dashed vertical lines are as in Figure 1."
    )
    (OUT_DIR / "fig2_ust_sf_detail_caption.txt").write_text(caption)
    print("  Caption saved.")


# ── Figure 3 — Exit Event Window Paths ────────────────────────────────────────

def fig3_exit_event_paths(panel):
    print("Generating Figure 3...")

    EXIT_DATE = pd.Timestamp("2021-03-31")
    W = 60

    # Compute event-time for each series
    groups = {
        "UST SF":  ("ust_spot_fut", COLORS["ust"],    "solid"),
        "TIPS":    ("tips",         COLORS["tips"],   "dashed"),
        "CIP":     ("cip",          COLORS["cip"],    "solid"),
        "Equity":  ("eq_spot_fut",  COLORS["equity"], "dashed"),
    }

    def build_event_paths(strategy_filter, exit_date, w):
        sub = panel[panel["strategy"] == strategy_filter].copy()
        series_list = sub["series_id"].unique()

        demeaned_list = []
        for sid in series_list:
            s = sub[sub["series_id"] == sid].sort_values("date").reset_index(drop=True)
            dates = s["date"].tolist()
            if exit_date not in dates:
                # Find nearest
                idx = np.searchsorted(dates, exit_date)
                if idx >= len(dates):
                    continue
                exit_idx = idx
            else:
                exit_idx = dates.index(exit_date)

            s["et"] = np.arange(len(s)) - exit_idx
            s_win = s[(s["et"] >= -w) & (s["et"] <= w)].copy()

            # Demean by pre-window mean
            pre_mean = s_win.loc[s_win["et"] < 0, "y_abs_bps"].mean()
            if np.isfinite(pre_mean) and pre_mean != 0:
                s_win = s_win.copy()
                s_win["y_dm"] = s_win["y_abs_bps"] - pre_mean
                demeaned_list.append(s_win[["et", "y_dm"]])

        if not demeaned_list:
            return pd.DataFrame(columns=["et", "mean", "se"])

        all_dm = pd.concat(demeaned_list, ignore_index=True)
        grp = all_dm.groupby("et")["y_dm"].agg(
            mean="mean",
            se=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        ).reset_index()
        return grp

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=False)

    # Left: Treasury-based
    for label, (strat, color, ls) in [
        ("UST SF", ("ust_spot_fut", COLORS["ust"],  "solid")),
        ("TIPS",   ("tips",         COLORS["tips"], "dashed")),
    ]:
        grp = build_event_paths(strat, EXIT_DATE, W)
        if grp.empty:
            continue
        ax_l.plot(grp["et"], grp["mean"], color=color, ls=ls, lw=1.2, label=label)
        ax_l.fill_between(grp["et"],
                          grp["mean"] - grp["se"],
                          grp["mean"] + grp["se"],
                          alpha=0.20, color=color, lw=0)

    ax_l.axvline(0, color='black', ls='--', lw=0.8)
    ax_l.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
    ax_l.set_xlabel("Trading Days Relative to SLR Exit (March 31, 2021)", fontsize=8)
    ax_l.set_ylabel("Demeaned |W| (bps)", fontsize=8)
    ax_l.set_title("Treasury-Based Strategies", fontsize=9)
    ax_l.legend(fontsize=7)

    # Right: Non-Treasury
    for label, (strat, color, ls) in [
        ("CIP",    ("cip",         COLORS["cip"],    "solid")),
        ("Equity", ("eq_spot_fut", COLORS["equity"], "dashed")),
    ]:
        grp = build_event_paths(strat, EXIT_DATE, W)
        if grp.empty:
            continue
        ax_r.plot(grp["et"], grp["mean"], color=color, ls=ls, lw=1.2, label=label)
        ax_r.fill_between(grp["et"],
                          grp["mean"] - grp["se"],
                          grp["mean"] + grp["se"],
                          alpha=0.20, color=color, lw=0)

    ax_r.axvline(0, color='black', ls='--', lw=0.8)
    ax_r.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
    ax_r.set_xlabel("Trading Days Relative to SLR Exit (March 31, 2021)", fontsize=8)
    ax_r.set_ylabel("Demeaned |W| (bps)", fontsize=8)
    ax_r.set_title("Non-Treasury Strategies (Control)", fontsize=9)
    ax_r.legend(fontsize=7)

    fig.tight_layout(pad=1.0)
    save_fig(fig, "fig3_exit_event_paths")
    plt.close(fig)


# ── Figure 4 — Coefficient Plot (Forest Plot) ─────────────────────────────────

def fig4_coef_plot():
    print("Generating Figure 4...")

    pooled = pd.read_csv(REPO_ROOT / "audit_repairs/outputs/jump_estimates_pooled.csv")
    pooled["event"] = pd.to_datetime(pooled["event"])

    did_path = REPO_ROOT / "audit_repairs/outputs/did_clean_baseline.csv"
    did_exists = did_path.exists()
    if did_exists:
        did = pd.read_csv(did_path)
        print("  Using did_clean_baseline.csv for DiD rows.")
    else:
        print("  did_clean_baseline.csv not found — using HARDCODED_FALLBACK values.")

    # Build rows list: (label, coef, se, color_group)
    rows = []

    EXIT_DATE = pd.Timestamp("2021-03-31")
    ENTRY_DATE = pd.Timestamp("2020-04-01")

    for event_date, group_label, color_key in [
        (EXIT_DATE,  "Exit Event",  COLORS["ust"]),
        (ENTRY_DATE, "Entry Event", COLORS["cip"]),
    ]:
        for w in [60, 20]:
            for spec in ["TOTAL", "DIRECT"]:
                sub = pooled[
                    (pooled["event"] == event_date) &
                    (pooled["window"] == w) &
                    (pooled["spec"] == spec)
                ]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                rows.append({
                    "label": f"W={w} {spec}",
                    "coef":  r["coef_post_x_treas"],
                    "se":    r["se_post_x_treas"],
                    "group": group_label,
                    "color": color_key,
                })

    # DiD rows
    did_color = "#27ae60"
    if did_exists:
        for _, row in did.iterrows():
            spec = row["spec"]
            rows.append({
                "label": f"Relief×Treas {spec}",
                "coef":  row["coef_relief_x_treas"],
                "se":    row["se_relief_x_treas"],
                "group": "Full-sample DiD",
                "color": did_color,
            })
            rows.append({
                "label": f"PostRelief×Treas {spec}",
                "coef":  row["coef_post_relief_x_treas"],
                "se":    row["se_post_relief_x_treas"],
                "group": "Full-sample DiD",
                "color": did_color,
            })
    else:
        # HARDCODED_FALLBACK
        fallback = [
            ("Relief×Treas TOTAL",       -10.76, 1.15),
            ("Relief×Treas DIRECT",      -10.77, 1.14),
            ("PostRelief×Treas TOTAL",    -9.03, 1.16),
            ("PostRelief×Treas DIRECT",   -9.04, 1.16),
        ]
        for label, coef, se in fallback:
            rows.append({"label": label, "coef": coef, "se": se,
                         "group": "Full-sample DiD", "color": did_color})

    # Assign y positions with gaps between groups
    groups_order = ["Exit Event", "Entry Event", "Full-sample DiD"]
    y_pos = []
    y = 0
    prev_group = None
    for r in rows:
        if prev_group is not None and r["group"] != prev_group:
            y += 0.5  # gap between groups
        y_pos.append(y)
        y += 1
        prev_group = r["group"]

    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    group_label_positions = {}
    for i, r in enumerate(rows):
        grp = r["group"]
        if grp not in group_label_positions:
            group_label_positions[grp] = []
        group_label_positions[grp].append(y_pos[i])

    # Draw points
    for i, (r, yp) in enumerate(zip(rows, y_pos)):
        coef = r["coef"]
        se   = r["se"]
        t    = coef / se if se > 0 else 0
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        stars = sig_stars(t)

        hollow = abs(t) <= 1.645
        fmt = 'o'
        fs  = 'none' if hollow else 'full'

        ax.errorbar(coef, yp,
                    xerr=[[coef - ci_lo], [ci_hi - coef]],
                    fmt=fmt, color=r["color"], fillstyle=fs,
                    ms=5, capsize=2, elinewidth=0.8, lw=0.8)

        if stars:
            ax.text(ci_hi + 0.4, yp, stars, va='center', fontsize=7,
                    color=r["color"])

    # Group labels in left margin
    for grp, positions in group_label_positions.items():
        mid = np.mean(positions)
        ax.text(-0.01, mid, grp,
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=7.5,
                fontstyle='italic', rotation=90)

    # y-axis tick labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel("Estimated Post × Treasury (basis points)", fontsize=8)
    ax.set_title("Pooled DiD Estimates: Post × Treasury Coefficient", fontsize=9)

    fig.tight_layout(pad=1.0)
    save_fig(fig, "fig4_coef_plot")
    plt.close(fig)


# ── Figure 5 — Series-Level Coefficients ──────────────────────────────────────

def fig5_series_level_coefs():
    print("Generating Figure 5...")

    by_series = pd.read_csv(REPO_ROOT / "audit_repairs/outputs/jump_estimates_by_series.csv")

    # Filter to exit W=60 DIRECT
    sub = by_series[
        (by_series["window"] == 60) &
        (by_series["spec"] == "DIRECT") &
        (by_series["event"] == "2021-03-31")
    ].copy()

    # Add tenor for sorting UST/TIPS
    def get_tenor_num(sid):
        for t in ["2y", "5y", "10y", "20y"]:
            if t in sid:
                return int(t.replace("y", ""))
        return 99

    sub["tenor_num"] = sub["series_id"].apply(get_tenor_num)

    # Build ordered list: UST SF → TIPS → CIP → Equity
    ust_rows  = sub[sub["strategy"] == "ust_spot_fut"].sort_values("tenor_num")
    tips_rows = sub[sub["strategy"] == "tips_treas"].sort_values("tenor_num")
    cip_rows  = sub[sub["strategy"] == "cip"].sort_values("coef_post", ascending=False)
    eq_rows   = sub[sub["strategy"] == "eq_spot_fut"].sort_values("series_id")

    ordered = pd.concat([ust_rows, tips_rows, cip_rows, eq_rows], ignore_index=True)

    # Nice labels
    label_map = {
        "ust_sf_2y": "UST SF 2Y", "ust_sf_5y": "UST SF 5Y", "ust_sf_10y": "UST SF 10Y",
        "tips_treas_2y": "TIPS 2Y", "tips_treas_5y": "TIPS 5Y", "tips_treas_10y": "TIPS 10Y",
        "cip_aud": "CIP AUD", "cip_cad": "CIP CAD", "cip_chf": "CIP CHF",
        "cip_eur": "CIP EUR", "cip_gbp": "CIP GBP", "cip_jpy": "CIP JPY",
        "cip_nzd": "CIP NZD", "cip_sek": "CIP SEK",
        "eq_indu": "Equity INDU", "eq_ndx": "Equity NDX", "eq_spx": "Equity SPX",
    }

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    group_colors = {
        "ust_spot_fut": COLORS["ust"],
        "tips_treas":   COLORS["tips"],
        "cip":          COLORS["cip"],
        "eq_spot_fut":  COLORS["equity"],
    }

    y_positions = list(range(len(ordered)))
    group_positions = {}

    for i, (_, row) in enumerate(ordered.iterrows()):
        coef = row["coef_post"]
        se   = row["se_post"]
        t    = row["t_post"]
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        stars = sig_stars(t)

        strat = row["strategy"]
        color = group_colors.get(strat, "gray")

        # Marker style
        if strat == "ust_spot_fut":
            fs = 'full'
        elif strat == "tips_treas":
            fs = 'none'
        elif strat == "cip":
            fs = 'full' if abs(t) > 1.645 else 'none'
        else:
            fs = 'none'

        ax.errorbar(coef, i,
                    xerr=[[coef - ci_lo], [ci_hi - coef]],
                    fmt='o', color=color, fillstyle=fs,
                    ms=5, capsize=2, elinewidth=0.8, lw=0.8)

        if stars:
            ax.text(ci_hi + 0.5, i, stars, va='center', fontsize=7, color=color)

        grp_name = {"ust_spot_fut": "Treasury: UST SF",
                    "tips_treas":   "Treasury: TIPS",
                    "cip":          "Non-Treasury: CIP",
                    "eq_spot_fut":  "Non-Treasury: Equity"}[strat]
        if grp_name not in group_positions:
            group_positions[grp_name] = []
        group_positions[grp_name].append(i)

    # Group labels
    for grp_name, positions in group_positions.items():
        mid = np.mean(positions)
        ax.text(-0.01, mid, grp_name,
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=7,
                fontstyle='italic', rotation=90)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [label_map.get(row["series_id"], row["series_id"])
         for _, row in ordered.iterrows()],
        fontsize=7.5
    )
    ax.invert_yaxis()
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel("Post-Expiry Coefficient (bps, exit event W=60)", fontsize=8)
    ax.set_title("Series-Level Post Coefficients: Exit W=60 DIRECT", fontsize=9)

    fig.tight_layout(pad=1.0)
    save_fig(fig, "fig5_series_level_coefs")
    plt.close(fig)


# ── Figure 6 — Regime Box Plots ───────────────────────────────────────────────

def fig6_regime_boxplots(panel):
    print("Generating Figure 6...")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))
    axes_flat = axes.flatten()

    REGIME_COLORS = {
        "pre":    "#d0d0d0",
        "relief": "#d4e6f1",
        "post":   "#d5f5e3",
    }
    REGIME_DARK = {
        "pre":    "#808080",
        "relief": "#2980b9",
        "post":   "#27ae60",
    }
    regimes = ["pre", "relief", "post"]
    regime_labels = {"pre": "Pre", "relief": "Relief", "post": "Post"}

    # Define sub-series for each strategy
    strategy_specs = [
        {
            "strategy": "ust_spot_fut",
            "title": "UST Spot-Futures",
            "subseries": [
                ("ust_sf_2y",  "2Y"),
                ("ust_sf_5y",  "5Y"),
                ("ust_sf_10y", "10Y"),
            ],
            "color_key": "ust",
        },
        {
            "strategy": "tips",
            "title": "TIPS-Treasury",
            "subseries": [
                ("tips_treas_2y",  "TIPS 2Y"),
                ("tips_treas_5y",  "TIPS 5Y"),
                ("tips_treas_10y", "TIPS 10Y"),
            ],
            "color_key": "tips",
        },
        {
            "strategy": "cip",
            "title": "CIP (3m FX)",
            "subseries": [
                ("cip_eur", "EUR"),
                ("cip_jpy", "JPY"),
                ("cip_gbp", "GBP"),
                # 'Other' = mean of AUD, CAD, NZD, SEK
                (["cip_aud", "cip_cad", "cip_nzd", "cip_sek"], "Other"),
            ],
            "color_key": "cip",
        },
        {
            "strategy": "eq_spot_fut",
            "title": "Equity Spot-Futures",
            "subseries": [
                ("eq_spx",  "SPX"),
                ("eq_ndx",  "NDX"),
                ("eq_indu", "INDU"),
            ],
            "color_key": "equity",
        },
    ]

    for ax_idx, spec in enumerate(strategy_specs):
        ax = axes_flat[ax_idx]
        subseries = spec["subseries"]
        n_sub = len(subseries)
        n_reg = len(regimes)

        group_width = n_reg + 0.6
        positions = []
        tick_positions = []
        tick_labels_sub = []
        data_list = []

        box_colors = []
        dark_colors = []

        for sub_i, (sid, slabel) in enumerate(subseries):
            group_start = sub_i * group_width
            group_center = group_start + (n_reg - 1) / 2.0

            tick_positions.append(group_center)
            tick_labels_sub.append(slabel)

            for reg_i, reg in enumerate(regimes):
                pos = group_start + reg_i
                positions.append(pos)

                if isinstance(sid, list):
                    # 'Other' group: average across series
                    vals = panel[
                        (panel["series_id"].isin(sid)) &
                        (panel["regime"] == reg)
                    ].groupby("date")["y_abs_bps"].mean()
                else:
                    vals = panel[
                        (panel["series_id"] == sid) &
                        (panel["regime"] == reg)
                    ]["y_abs_bps"]

                data_list.append(vals.dropna().values)
                box_colors.append(REGIME_COLORS[reg])
                dark_colors.append(REGIME_DARK[reg])

        # Draw boxplots
        bp = ax.boxplot(data_list, positions=positions, widths=0.65,
                        patch_artist=True, whis=[5, 95],
                        medianprops=dict(color="black", lw=0.8),
                        whiskerprops=dict(lw=0.6),
                        capprops=dict(lw=0.6),
                        flierprops=dict(marker='.', ms=2, alpha=0.3, lw=0))

        for patch, fc in zip(bp['boxes'], box_colors):
            patch.set_facecolor(fc)
            patch.set_linewidth(0.6)

        # Overlay mean diamonds
        for di, (pos, dc, vals) in enumerate(zip(positions, dark_colors, data_list)):
            if len(vals) > 0:
                m = np.mean(vals)
                ax.plot(pos, m, 'd', color=dc, ms=4, zorder=5, lw=0)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_sub, fontsize=7.5)
        ax.set_ylabel("Spread |W| (bps)", fontsize=8)
        ax.set_title(spec["title"], fontsize=9)

        # Add regime legend to first panel only
        if ax_idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=REGIME_COLORS["pre"],    edgecolor='gray', lw=0.5, label="Pre"),
                Patch(facecolor=REGIME_COLORS["relief"], edgecolor='gray', lw=0.5, label="Relief"),
                Patch(facecolor=REGIME_COLORS["post"],   edgecolor='gray', lw=0.5, label="Post"),
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    fig.tight_layout(pad=1.2)
    save_fig(fig, "fig6_regime_boxplots")
    plt.close(fig)


# ── FIGURE MANIFEST and LaTeX ──────────────────────────────────────────────────

def write_manifest(did_fallback_used):
    manifest_lines = ["# Figure Manifest\n"]

    figures = [
        {
            "stem": "fig1_series_overview",
            "title": "Figure 1 — Time Series Overview",
            "label": "fig:series_overview",
            "data": ["data/series/cip_spreads_3m_bps.csv",
                     "data/series/treasury_sf_output.csv",
                     "data/series/tips_treasury_implied_rf_2010.parquet",
                     "data/series/equity_spot_spread_SPX.csv",
                     "data/series/equity_spot_spread_NDX.csv",
                     "data/series/equity_spot_spread_INDU.csv"],
            "fallback": None,
        },
        {
            "stem": "fig2_ust_sf_detail",
            "title": "Figure 2 — UST SF Detail by Tenor",
            "label": "fig:ust_sf_detail",
            "data": ["data/series/treasury_sf_output.csv"],
            "fallback": None,
        },
        {
            "stem": "fig3_exit_event_paths",
            "title": "Figure 3 — Exit Event Window Paths",
            "label": "fig:exit_event_paths",
            "data": ["Panel data from all series"],
            "fallback": None,
        },
        {
            "stem": "fig4_coef_plot",
            "title": "Figure 4 — Coefficient Plot (Forest Plot)",
            "label": "fig:coef_plot",
            "data": ["audit_repairs/outputs/jump_estimates_pooled.csv",
                     "audit_repairs/outputs/did_clean_baseline.csv"],
            "fallback": (
                "HARDCODED_FALLBACK used for DiD rows: "
                "Relief×Treas TOTAL coef=-10.76 se=1.15; "
                "Relief×Treas DIRECT coef=-10.77 se=1.14; "
                "PostRelief×Treas TOTAL coef=-9.03 se=1.16; "
                "PostRelief×Treas DIRECT coef=-9.04 se=1.16"
            ) if did_fallback_used else None,
        },
        {
            "stem": "fig5_series_level_coefs",
            "title": "Figure 5 — Series-Level Coefficients (Exit W=60 DIRECT)",
            "label": "fig:series_level_coefs",
            "data": ["audit_repairs/outputs/jump_estimates_by_series.csv"],
            "fallback": None,
        },
        {
            "stem": "fig6_regime_boxplots",
            "title": "Figure 6 — Regime Box Plots",
            "label": "fig:regime_boxplots",
            "data": ["Panel data from all series"],
            "fallback": None,
        },
    ]

    for f in figures:
        manifest_lines.append(f"## {f['title']}\n")
        manifest_lines.append(f"**Files:** `{f['stem']}.pdf`, `{f['stem']}.png`\n\n")
        manifest_lines.append(f"**Data sources:**\n")
        for d in f["data"]:
            manifest_lines.append(f"- {d}\n")
        manifest_lines.append("\n")
        if f["fallback"]:
            manifest_lines.append(f"**FLAG — HARDCODED_FALLBACK:** {f['fallback']}\n\n")
        manifest_lines.append(
            "**LaTeX:**\n"
            "```latex\n"
            "\\begin{figure}[htbp]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=0.95\\textwidth]{{figures_output/{f['stem']}.pdf}}\n"
            "  \\caption{...}\n"
            f"  \\label{{{f['label']}}}\n"
            "\\end{figure}\n"
            "```\n\n"
        )

    (OUT_DIR / "FIGURE_MANIFEST.md").write_text("".join(manifest_lines))
    print("  FIGURE_MANIFEST.md saved.")

    # figures.tex
    tex_lines = [
        "% figures.tex — auto-generated by make_figures.py\n",
        "% \\input{figures_output/figures.tex} in your main .tex file\n\n",
    ]
    caption_texts = {
        "fig1_series_overview": (
            "Time-series evolution of arbitrage spreads, 2019--2021. "
            "Each panel shows the cross-series mean (solid line) and $\\pm$1 standard deviation band (shaded) "
            "of absolute spread $|W|$ in basis points, smoothed with a 5-day rolling average. "
            "The gray shaded region denotes the SLR exclusion relief period (April 1, 2020 -- March 31, 2021). "
            "Dashed vertical lines mark the entry (red) and exit (green) dates."
        ),
        "fig2_ust_sf_detail": (
            "UST spot-futures arbitrage spread by tenor, 2019--2021. "
            "Lines show the absolute spread $|W|$ for 2-year (solid), 5-year (dashed), "
            "and 10-year (dotted) maturities, smoothed with a 5-day rolling average. "
            "Horizontal dotted lines indicate the pre-period mean (January 2019 -- January 2020). "
        ),
        "fig3_exit_event_paths": (
            "Event-time paths around the SLR exclusion exit (March 31, 2021). "
            "Left panel: Treasury-based strategies (UST spot-futures and TIPS-Treasury). "
            "Right panel: non-Treasury control strategies (CIP and Equity spot-futures). "
            "Series are demeaned by their pre-event mean (trading days $-60$ to $-1$) before averaging. "
            "Shaded bands denote $\\pm$1 standard error. Dashed vertical line at day 0 marks the exit date."
        ),
        "fig4_coef_plot": (
            "Pooled difference-in-differences estimates of the Post$\\times$Treasury coefficient. "
            "Each row reports a point estimate with 95\\% confidence interval. "
            "Exit and Entry event estimates use windows $W\\in\\{20,60\\}$ trading days; "
            "Full-sample DiD estimates span 2019--2021. "
            "Filled markers indicate $|t|>1.645$; stars denote $|t|>1.96$ (**) and $|t|>2.576$ (***)."
        ),
        "fig5_series_level_coefs": (
            "Series-level post-expiry coefficients for the exit event (March 31, 2021), "
            "window $W=60$, DIRECT specification. "
            "Each row is an individual series; error bars show 95\\% confidence intervals. "
            "Filled markers indicate $|t|>1.645$."
        ),
        "fig6_regime_boxplots": (
            "Distribution of absolute arbitrage spreads $|W|$ by regime and strategy. "
            "Box spans the interquartile range; whiskers extend to the 5th and 95th percentiles; "
            "diamonds mark the within-group mean. "
            "Pre = January 2019 -- March 31, 2020; Relief = April 1, 2020 -- March 31, 2021; "
            "Post = April 1 -- December 31, 2021."
        ),
    }

    for f in figures:
        stem  = f["stem"]
        label = f["label"]
        cap   = caption_texts.get(stem, "...")
        tex_lines.append(
            f"\\begin{{figure}}[htbp]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width=0.95\\textwidth]{{figures_output/{stem}.pdf}}\n"
            f"  \\caption{{{cap}}}\n"
            f"  \\label{{{label}}}\n"
            f"\\end{{figure}}\n\n"
        )

    (OUT_DIR / "figures.tex").write_text("".join(tex_lines))
    print("  figures.tex saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== SLR Exclusion Paper — Figure Generation ===\n")

    print("Loading panel data...")
    panel = load_panel()
    print(f"  Panel shape: {panel.shape}")
    print(f"  Strategies: {panel['strategy'].unique().tolist()}")
    print(f"  Date range: {panel['date'].min()} to {panel['date'].max()}")
    print()

    fig1_series_overview(panel)
    fig2_ust_sf_detail(panel)
    fig3_exit_event_paths(panel)
    fig4_coef_plot()
    fig5_series_level_coefs()
    fig6_regime_boxplots(panel)

    # Check if DiD fallback was used
    did_path = REPO_ROOT / "audit_repairs/outputs/did_clean_baseline.csv"
    did_fallback = not did_path.exists()
    write_manifest(did_fallback)

    print("\n=== All figures complete. ===")
    print("Files saved to:", OUT_DIR)

    # Verify file sizes
    print("\nFile size check:")
    for stem in ["fig1_series_overview", "fig2_ust_sf_detail", "fig3_exit_event_paths",
                 "fig4_coef_plot", "fig5_series_level_coefs", "fig6_regime_boxplots"]:
        for ext in ("pdf", "png"):
            p = OUT_DIR / f"{stem}.{ext}"
            if p.exists():
                size_kb = p.stat().st_size / 1024
                ok = "OK" if size_kb > 50 else "WARNING: small file"
                print(f"  {stem}.{ext}: {size_kb:.0f} KB  [{ok}]")
            else:
                print(f"  {stem}.{ext}: MISSING")


if __name__ == "__main__":
    main()
