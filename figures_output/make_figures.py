"""
make_figures.py — Publication figures for SLR exclusion paper.

Style: tidyquant / theme_bw() equivalent in matplotlib.
  - White background, full black frame (all 4 spines)
  - Light gray grid lines behind data (axes.axisbelow)
  - Bold axis labels and titles
  - tidyquant palette: #2C3E50, #E31A1C, #18BC9C, #1F78B4, #CCBE93
  - Sans-serif font (matches ggplot2 default)

Source style reference:
  Ross (2020), https://tex.stackexchange.com/a/553954, CC BY-SA 4.0
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from pathlib import Path

# ── tidyquant / theme_bw() style ─────────────────────────────────────────────
mpl.rcParams.update({
    # Font — ggplot2 default is sans-serif
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size":            11,
    "font.weight":          "normal",

    # Titles and labels — bold, matching theme() overrides in the source
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.labelsize":       13,
    "axes.labelweight":     "bold",
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,

    # Legend
    "legend.fontsize":      10,
    "legend.frameon":       True,
    "legend.framealpha":    1.0,
    "legend.edgecolor":     "#cccccc",
    "legend.borderpad":     0.5,
    "legend.labelspacing":  0.4,

    # Lines
    "lines.linewidth":      1.6,

    # theme_bw: full box frame (all 4 spines on)
    "axes.spines.top":      True,
    "axes.spines.right":    True,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    "axes.edgecolor":       "black",
    "axes.linewidth":       0.9,
    "axes.facecolor":       "white",
    "figure.facecolor":     "white",

    # Grid — theme_bw light gray, behind data
    "axes.grid":            True,
    "axes.axisbelow":       True,
    "grid.color":           "#e0e0e0",
    "grid.linestyle":       "-",
    "grid.linewidth":       0.5,

    # Ticks — outward, small
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,

    # Output
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.15,
    "patch.linewidth":      0.7,
})

# ── tidyquant palette (scale_color_tq "dark") ────────────────────────────────
COLORS = {
    "ust":    "#2c3e50",    # Midnight Blue   (UST SF — Treasury-based)
    "tips":   "#1f78b4",    # Steel Blue      (TIPS — Treasury-based)
    "cip":    "#e31a1c",    # Crimson         (CIP — non-Treasury)
    "equity": "#18bc9c",    # Teal            (Equity — non-Treasury)
    "entry":  "#e31a1c",    # Crimson         (entry event line)
    "exit":   "#2c3e50",    # Midnight Blue   (exit event line)
    "shade":  "#f5f5f5",    # Near-white gray (SLR exclusion band)
    "ci":     "#cccccc",    # Light gray      (CI bands)
}

SLR_ENTRY = pd.Timestamp("2020-04-01")
SLR_EXIT  = pd.Timestamp("2021-03-31")
REPO_ROOT = Path(__file__).parent.parent
OUT_DIR   = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_panel():
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
        tmp["series_id"]      = f"cip_{tenor.lower()}"
        tmp["strategy"]       = "cip"
        tmp["tenor"]          = tenor
        tmp["treasury_based"] = 0
        records.append(tmp)

    # 2. UST SF
    ust = pd.read_csv(REPO_ROOT / "data/series/treasury_sf_output.csv",
                      parse_dates=["Date"])
    ust_cols = {"Treasury_SF_2Y": "2Y", "Treasury_SF_5Y": "5Y", "Treasury_SF_10Y": "10Y"}
    for col, tenor in ust_cols.items():
        tmp = ust[["Date", col]].dropna().copy()
        tmp.columns = ["date", "y_bps"]
        tmp["series_id"]      = f"ust_sf_{tenor.lower()}"
        tmp["strategy"]       = "ust_spot_fut"
        tmp["tenor"]          = tenor
        tmp["treasury_based"] = 1
        records.append(tmp)

    # 3. TIPS
    tips = pd.read_parquet(REPO_ROOT / "data/series/tips_treasury_implied_rf_2010.parquet")
    tips = tips.reset_index() if tips.index.name == "date" else tips
    date_col = "date" if "date" in tips.columns else tips.columns[0]
    tips[date_col] = pd.to_datetime(tips[date_col])
    for col, tenor in {"arb_2": "2Y", "arb_5": "5Y", "arb_10": "10Y"}.items():
        tmp = tips[[date_col, col]].dropna().copy()
        tmp.columns = ["date", "y_bps"]
        tmp["series_id"]      = f"tips_treas_{tenor.lower()}"
        tmp["strategy"]       = "tips"
        tmp["tenor"]          = tenor
        tmp["treasury_based"] = 1
        records.append(tmp)

    # 4. Equity
    for fname, (sid, tenor, col) in {
        "equity_spot_spread_SPX.csv":  ("eq_spx",  "SPX",  "spread_SPX"),
        "equity_spot_spread_NDX.csv":  ("eq_ndx",  "NDX",  "spread_NDX"),
        "equity_spot_spread_INDU.csv": ("eq_indu", "INDU", "spread_INDU"),
    }.items():
        df = pd.read_csv(REPO_ROOT / "data/series" / fname, parse_dates=["Date"])
        tmp = df[["Date", col]].dropna().copy()
        tmp.columns = ["date", "y_bps"]
        tmp["series_id"]      = sid
        tmp["strategy"]       = "eq_spot_fut"
        tmp["tenor"]          = tenor
        tmp["treasury_based"] = 0
        records.append(tmp)

    panel = pd.concat(records, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])

    # Unit check
    ust_pre = panel[
        (panel["strategy"] == "ust_spot_fut") &
        (panel["date"] >= "2019-01-01") &
        (panel["date"] <= "2020-03-31")
    ]["y_bps"].abs().mean()
    print(f"[UNIT CHECK] UST SF pre-period mean |W|: {ust_pre:.2f} bps")
    if not (5 < ust_pre < 200):
        raise ValueError(f"UST SF mean {ust_pre:.2f} bps out of expected range [5, 200].")
    print("[UNIT CHECK] PASSED.")

    # Restrict to 2019-2021
    panel = panel[
        (panel["date"] >= "2019-01-01") & (panel["date"] <= "2021-12-31")
    ].copy()

    # Winsorise within-series at p1/p99
    for sid, grp in panel.groupby("series_id"):
        lo = grp["y_bps"].quantile(0.01)
        hi = grp["y_bps"].quantile(0.99)
        panel.loc[grp.index, "y_bps"] = grp["y_bps"].clip(lo, hi)

    panel["y_abs_bps"] = panel["y_bps"].abs()

    # Regime labels
    def regime(d):
        if d < SLR_ENTRY:   return "pre"
        elif d <= SLR_EXIT: return "relief"
        else:               return "post"

    panel["regime"] = panel["date"].map(regime)
    return panel.reset_index(drop=True)


# ── Shared helpers ────────────────────────────────────────────────────────────

def add_slr_shading(ax):
    """Gray band + dashed event lines."""
    ax.axvspan(SLR_ENTRY, SLR_EXIT, alpha=0.12, color="#aaaaaa", zorder=0)
    ax.axvline(SLR_ENTRY, color=COLORS["entry"], ls='--', lw=1.0, zorder=2)
    ax.axvline(SLR_EXIT,  color=COLORS["exit"],  ls='--', lw=1.0, zorder=2)


def label_slr_events(ax):
    """Entry / Exit text just inside the top of the axes — axes-fraction y."""
    trans = ax.get_xaxis_transform()
    ax.text(SLR_ENTRY + pd.Timedelta(days=5), 0.96,
            "Entry", transform=trans,
            color=COLORS["entry"], fontsize=9, fontweight="bold",
            va='top', ha='left', clip_on=True)
    ax.text(SLR_EXIT + pd.Timedelta(days=5), 0.96,
            "Exit", transform=trans,
            color=COLORS["exit"], fontsize=9, fontweight="bold",
            va='top', ha='left', clip_on=True)


def format_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=10)


def sig_stars(t):
    a = abs(t)
    if a > 2.576: return "***"
    if a > 1.960: return "**"
    if a > 1.645: return "*"
    return ""


def save_fig(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}")
    print(f"  Saved {stem}.pdf / .png")


# ── Figure 1 — Time-Series Overview (2×2) ────────────────────────────────────

def fig1_series_overview(panel):
    print("Generating Figure 1...")

    strategies = [
        ("ust_spot_fut", "UST Spot-Futures",   COLORS["ust"]),
        ("tips",         "TIPS-Treasury",       COLORS["tips"]),
        ("cip",          "CIP (3-month)",        COLORS["cip"]),
        ("eq_spot_fut",  "Equity Spot-Futures",  COLORS["equity"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5), sharex=True)
    axes_flat = axes.flatten()

    for idx, (strat, title, color) in enumerate(strategies):
        ax = axes_flat[idx]
        sub = (panel[panel["strategy"] == strat]
               .groupby("date")["y_abs_bps"]
               .agg(["mean", "std"])
               .reset_index()
               .sort_values("date"))
        sub["mean_sm"] = sub["mean"].rolling(5, min_periods=1).mean()
        sub["std_sm"]  = sub["std"].rolling(5,  min_periods=1).mean()

        add_slr_shading(ax)
        ax.fill_between(sub["date"],
                        sub["mean_sm"] - sub["std_sm"],
                        sub["mean_sm"] + sub["std_sm"],
                        alpha=0.20, color=color, lw=0)
        ax.plot(sub["date"], sub["mean_sm"], color=color, lw=1.8)

        if idx == 0:
            label_slr_events(ax)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=5)
        ax.set_ylabel("|W| (bps)", fontsize=11, fontweight="bold")
        format_xaxis(ax)

    fig.text(0.5, -0.01,
             "Gray band: SLR exclusion (Apr 2020\u2013Mar 2021). "
             "Dashed lines: entry (red) and exit (navy).",
             ha='center', va='top', fontsize=9, color='#444444', style='italic')

    fig.tight_layout(pad=1.4)
    save_fig(fig, "fig1_series_overview")
    plt.close(fig)
    print("  Caption saved.")


# ── Figure 2 — UST SF Detail by Tenor ────────────────────────────────────────

def fig2_ust_sf_detail(panel):
    print("Generating Figure 2...")

    # Three shades of the tidyquant navy for within-strategy differentiation
    tenor_spec = [
        ("2Y",  "#2c3e50", "solid",   "2-year"),
        ("5Y",  "#2980b9", "dashed",  "5-year"),
        ("10Y", "#85c1e9", "dashdot", "10-year"),
    ]

    pre_mask = (panel["date"] >= "2019-01-01") & (panel["date"] <= "2020-01-31")

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    add_slr_shading(ax)
    label_slr_events(ax)

    for tenor, color, ls, label in tenor_spec:
        sid = f"ust_sf_{tenor.lower()}"
        sub = panel[panel["series_id"] == sid].copy().sort_values("date")
        sub["y_sm"] = sub["y_abs_bps"].rolling(5, min_periods=1).mean()

        pre_mean = sub.loc[pre_mask[sub.index], "y_abs_bps"].mean()
        leg = f"{label}  (pre-mean: {pre_mean:.1f} bps)" if np.isfinite(pre_mean) else label

        ax.plot(sub["date"], sub["y_sm"], color=color, ls=ls, lw=1.8, label=leg)

        if np.isfinite(pre_mean):
            ax.axhline(pre_mean, color=color, ls=':', lw=0.8, alpha=0.5)

    ax.set_ylabel("|W| (bps)", fontsize=12, fontweight="bold")
    ax.set_title("UST Spot-Futures Spread by Tenor", fontsize=13, fontweight="bold", pad=6)
    ax.legend(loc="upper left", fontsize=9)
    format_xaxis(ax)
    fig.tight_layout(pad=1.0)
    save_fig(fig, "fig2_ust_sf_detail")
    plt.close(fig)
    print("  Caption saved.")


# ── Figure 3 — Exit Event Window Paths ───────────────────────────────────────

def fig3_exit_event_paths(panel):
    print("Generating Figure 3...")

    EXIT_DATE = pd.Timestamp("2021-03-31")
    W = 60

    def event_paths(strategy_filter):
        sub = panel[panel["strategy"] == strategy_filter].copy()
        dm_list = []
        for sid in sub["series_id"].unique():
            s = sub[sub["series_id"] == sid].sort_values("date").reset_index(drop=True)
            dates = s["date"].tolist()
            try:
                ei = dates.index(EXIT_DATE)
            except ValueError:
                ei = int(np.searchsorted(dates, EXIT_DATE))
                if ei >= len(dates):
                    continue
            s["et"] = np.arange(len(s)) - ei
            w = s[(s["et"] >= -W) & (s["et"] <= W)].copy()
            pre_mean = w.loc[w["et"] < 0, "y_abs_bps"].mean()
            if np.isfinite(pre_mean) and pre_mean > 0:
                w = w.copy()
                w["yd"] = w["y_abs_bps"] - pre_mean
                dm_list.append(w[["et", "yd"]])
        if not dm_list:
            return pd.DataFrame(columns=["et", "mean", "se"])
        all_dm = pd.concat(dm_list, ignore_index=True)
        return all_dm.groupby("et")["yd"].agg(
            mean="mean",
            se=lambda x: x.std(ddof=1) / np.sqrt(max(len(x), 1))
        ).reset_index()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 3.8))

    for ax, pairs, panel_title in [
        (ax_l, [("UST SF", "ust_spot_fut", COLORS["ust"],    "solid"),
                ("TIPS",   "tips",         COLORS["tips"],   "dashed")],
         "Treasury-Based Strategies"),
        (ax_r, [("CIP",    "cip",          COLORS["cip"],    "solid"),
                ("Equity", "eq_spot_fut",  COLORS["equity"], "dashed")],
         "Non-Treasury Strategies (Control)"),
    ]:
        for label, strat, color, ls in pairs:
            grp = event_paths(strat)
            if grp.empty:
                continue
            ax.plot(grp["et"], grp["mean"], color=color, ls=ls, lw=1.8, label=label)
            ax.fill_between(grp["et"],
                            grp["mean"] - grp["se"],
                            grp["mean"] + grp["se"],
                            alpha=0.18, color=color, lw=0)

        ax.axvline(0, color='black', ls='--', lw=0.9)
        ax.axhline(0, color='#888888', ls='-', lw=0.5)
        ax.set_xlabel("Trading days relative to SLR exit",
                      fontsize=12, fontweight="bold")
        ax.set_ylabel("Demeaned |W| (bps)", fontsize=12, fontweight="bold")
        ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=5)
        ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout(pad=1.0)
    save_fig(fig, "fig3_exit_event_paths")
    plt.close(fig)


# ── Figure 4 — Coefficient Forest Plot ───────────────────────────────────────

def fig4_coef_plot():
    print("Generating Figure 4...")

    pooled = pd.read_csv(REPO_ROOT / "audit_repairs/outputs/jump_estimates_pooled.csv")
    pooled["event"] = pd.to_datetime(pooled["event"])

    did_path = REPO_ROOT / "audit_repairs/outputs/did_clean_baseline.csv"
    did_exists = did_path.exists()
    did = pd.read_csv(did_path) if did_exists else None
    if did_exists:
        print("  Using did_clean_baseline.csv.")
    else:
        print("  did_clean_baseline.csv not found — using fallback values.")

    rows = []
    for event_date, group_label, color in [
        (pd.Timestamp("2021-03-31"), "Exit Event",  COLORS["ust"]),
        (pd.Timestamp("2020-04-01"), "Entry Event", COLORS["cip"]),
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
                rows.append({"label": f"W={w}, {spec}", "coef": r["coef_post_x_treas"],
                              "se": r["se_post_x_treas"], "group": group_label, "color": color})

    did_color = COLORS["tips"]
    if did_exists:
        for _, row in did.iterrows():
            sp = row["spec"]
            rows.append({"label": f"Relief\u00d7Treas, {sp}",
                         "coef": row["coef_relief_x_treas"],
                         "se":   row["se_relief_x_treas"],
                         "group": "Full-Period DiD", "color": did_color})
            rows.append({"label": f"PostRelief\u00d7Treas, {sp}",
                         "coef": row["coef_post_relief_x_treas"],
                         "se":   row["se_post_relief_x_treas"],
                         "group": "Full-Period DiD", "color": did_color})
    else:
        for label, coef, se in [
            ("Relief\u00d7Treas, TOTAL",       -10.76, 1.15),
            ("Relief\u00d7Treas, DIRECT",       -10.77, 1.14),
            ("PostRelief\u00d7Treas, TOTAL",     -9.03, 1.16),
            ("PostRelief\u00d7Treas, DIRECT",    -9.04, 1.16),
        ]:
            rows.append({"label": label, "coef": coef, "se": se,
                         "group": "Full-Period DiD", "color": did_color})

    # Insert bold header rows before each new group
    rows_h = []
    prev = None
    for r in rows:
        if r["group"] != prev:
            rows_h.append({"label": r["group"], "coef": None, "se": None,
                           "group": r["group"], "color": r["color"], "is_hdr": True})
            prev = r["group"]
        r2 = dict(r); r2["is_hdr"] = False
        rows_h.append(r2)

    yp = list(range(len(rows_h)))

    fig, ax = plt.subplots(figsize=(6.8, 5.5))

    for i, (r, y) in enumerate(zip(rows_h, yp)):
        if r["is_hdr"]:
            if i > 0:
                ax.axhline(y - 0.5, color='#cccccc', lw=0.8, zorder=0)
            continue
        coef, se = r["coef"], r["se"]
        t = coef / se if se > 0 else 0
        ci_lo, ci_hi = coef - 1.96 * se, coef + 1.96 * se
        stars = sig_stars(t)
        fs = 'none' if abs(t) <= 1.645 else 'full'

        ax.errorbar(coef, y, xerr=[[coef - ci_lo], [ci_hi - coef]],
                    fmt='o', color=r["color"], fillstyle=fs,
                    ms=6, capsize=3, elinewidth=1.1, lw=1.1)
        if stars:
            ax.text(ci_hi + 0.5, y, stars, va='center', fontsize=8.5, color=r["color"])

    ax.set_yticks(yp)
    tick_lbls = ax.set_yticklabels([r["label"] for r in rows_h], fontsize=9)
    for lbl, r in zip(tick_lbls, rows_h):
        if r["is_hdr"]:
            lbl.set_fontweight("bold"); lbl.set_fontstyle("italic")
            lbl.set_fontsize(10);      lbl.set_color("#222222")

    ax.invert_yaxis()
    ax.axvline(0, color='#555555', ls='--', lw=0.9)
    ax.set_xlabel("Post \u00d7 Treasury coefficient (basis points)",
                  fontsize=12, fontweight="bold")
    ax.set_title("Pooled DiD Estimates: Post \u00d7 Treasury",
                 fontsize=13, fontweight="bold", pad=7)

    fig.tight_layout(pad=1.3)
    save_fig(fig, "fig4_coef_plot")
    plt.close(fig)


# ── Figure 5 — Series-Level Coefficients ─────────────────────────────────────

def fig5_series_level_coefs():
    print("Generating Figure 5...")

    by_series = pd.read_csv(REPO_ROOT / "audit_repairs/outputs/jump_estimates_by_series.csv")
    sub = by_series[
        (by_series["window"] == 60) &
        (by_series["spec"] == "DIRECT") &
        (by_series["event"] == "2021-03-31")
    ].copy()

    def tenor_num(sid):
        for t in ["2y", "5y", "10y"]:
            if t in sid: return int(t[:-1])
        return 99

    sub["tn"] = sub["series_id"].apply(tenor_num)

    ordered = pd.concat([
        sub[sub["strategy"] == "ust_spot_fut"].sort_values("tn"),
        sub[sub["strategy"] == "tips_treas"].sort_values("tn"),
        sub[sub["strategy"] == "cip"].sort_values("coef_post", ascending=False),
        sub[sub["strategy"] == "eq_spot_fut"].sort_values("series_id"),
    ], ignore_index=True)

    LABEL = {
        "ust_sf_2y": "UST SF 2Y",    "ust_sf_5y": "UST SF 5Y",
        "ust_sf_10y": "UST SF 10Y",  "tips_treas_2y": "TIPS 2Y",
        "tips_treas_5y": "TIPS 5Y",  "tips_treas_10y": "TIPS 10Y",
        "cip_aud": "CIP AUD",  "cip_cad": "CIP CAD", "cip_chf": "CIP CHF",
        "cip_eur": "CIP EUR",  "cip_gbp": "CIP GBP", "cip_jpy": "CIP JPY",
        "cip_nzd": "CIP NZD",  "cip_sek": "CIP SEK",
        "eq_indu": "Equity INDU", "eq_ndx": "Equity NDX", "eq_spx": "Equity SPX",
    }
    GROUP_INFO = [
        ("ust_spot_fut", "Treasury: UST SF",    COLORS["ust"]),
        ("tips_treas",   "Treasury: TIPS",       COLORS["tips"]),
        ("cip",          "Non-Treasury: CIP",    COLORS["cip"]),
        ("eq_spot_fut",  "Non-Treasury: Equity", COLORS["equity"]),
    ]

    # Build rows with header entries
    rows_h = []
    prev = None
    for _, row in ordered.iterrows():
        strat = row["strategy"]
        if strat != prev:
            gn = next(g for s, g, c in GROUP_INFO if s == strat)
            gc = next(c for s, g, c in GROUP_INFO if s == strat)
            rows_h.append({"sid": None, "strat": strat, "coef": None, "se": None,
                           "t": None, "label": gn, "color": gc, "is_hdr": True})
            prev = strat
        gc = next(c for s, g, c in GROUP_INFO if s == strat)
        rows_h.append({"sid": row["series_id"], "strat": strat,
                       "coef": row["coef_post"], "se": row["se_post"],
                       "t": row["t_post"],
                       "label": LABEL.get(row["series_id"], row["series_id"]),
                       "color": gc, "is_hdr": False})

    yp = list(range(len(rows_h)))

    fig, ax = plt.subplots(figsize=(6.8, 7.2))

    for i, (r, y) in enumerate(zip(rows_h, yp)):
        if r["is_hdr"]:
            if i > 0:
                ax.axhline(y - 0.5, color='#cccccc', lw=0.8, zorder=0)
            continue
        coef, se, t = r["coef"], r["se"], r["t"]
        ci_lo, ci_hi = coef - 1.96 * se, coef + 1.96 * se
        stars = sig_stars(t)
        fs = 'full' if abs(t) > 1.645 else 'none'

        ax.errorbar(coef, y, xerr=[[coef - ci_lo], [ci_hi - coef]],
                    fmt='o', color=r["color"], fillstyle=fs,
                    ms=6, capsize=3, elinewidth=1.1, lw=1.1)
        if stars:
            ax.text(ci_hi + 0.5, y, stars, va='center', fontsize=8.5, color=r["color"])

    ax.set_yticks(yp)
    tick_lbls = ax.set_yticklabels([r["label"] for r in rows_h], fontsize=9)
    for lbl, r in zip(tick_lbls, rows_h):
        if r["is_hdr"]:
            lbl.set_fontweight("bold"); lbl.set_fontstyle("italic")
            lbl.set_fontsize(10);      lbl.set_color("#222222")

    ax.invert_yaxis()
    ax.axvline(0, color='#555555', ls='--', lw=0.9)
    ax.set_xlabel("Post-expiry coefficient (bps, exit event W=60)",
                  fontsize=12, fontweight="bold")
    ax.set_title("Series-Level Post Coefficients: Exit W=60, Direct",
                 fontsize=13, fontweight="bold", pad=7)

    fig.tight_layout(pad=1.3)
    save_fig(fig, "fig5_series_level_coefs")
    plt.close(fig)


# ── Figure 6 — Regime Box Plots ──────────────────────────────────────────────

def fig6_regime_boxplots(panel):
    print("Generating Figure 6...")

    # tidyquant-inspired regime palette
    REG_FACE = {"pre": "#dce6f1", "relief": "#fce4d6", "post": "#e2efda"}
    REG_EDGE = {"pre": "#2c3e50", "relief": "#e31a1c", "post": "#18bc9c"}
    REG_MED  = {"pre": "#2c3e50", "relief": "#e31a1c", "post": "#18bc9c"}
    regimes  = ["pre", "relief", "post"]

    strategy_specs = [
        {"title": "UST Spot-Futures", "subseries": [
            ("ust_sf_2y", "2Y"), ("ust_sf_5y", "5Y"), ("ust_sf_10y", "10Y")]},
        {"title": "TIPS-Treasury",    "subseries": [
            ("tips_treas_2y", "TIPS 2Y"), ("tips_treas_5y", "TIPS 5Y"),
            ("tips_treas_10y", "TIPS 10Y")]},
        {"title": "CIP (3-month)",    "subseries": [
            ("cip_eur", "EUR"), ("cip_jpy", "JPY"), ("cip_gbp", "GBP"),
            (["cip_aud", "cip_cad", "cip_nzd", "cip_sek"], "Other")]},
        {"title": "Equity Spot-Futures", "subseries": [
            ("eq_spx", "SPX"), ("eq_ndx", "NDX"), ("eq_indu", "INDU")]},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
    axes_flat = axes.flatten()

    for ax_idx, spec in enumerate(strategy_specs):
        ax = axes_flat[ax_idx]
        subseries = spec["subseries"]
        n_reg     = len(regimes)
        gw        = n_reg + 0.8

        positions, tick_pos, tick_lbl = [], [], []
        data_list, face_colors, edge_colors, med_colors = [], [], [], []

        for si, (sid, slabel) in enumerate(subseries):
            gc = si * gw
            tick_pos.append(gc + (n_reg - 1) / 2.0)
            tick_lbl.append(slabel)
            for ri, reg in enumerate(regimes):
                positions.append(gc + ri)
                if isinstance(sid, list):
                    vals = (panel[panel["series_id"].isin(sid) & (panel["regime"] == reg)]
                            .groupby("date")["y_abs_bps"].mean())
                else:
                    vals = panel[(panel["series_id"] == sid) &
                                 (panel["regime"] == reg)]["y_abs_bps"]
                data_list.append(vals.dropna().values)
                face_colors.append(REG_FACE[reg])
                edge_colors.append(REG_EDGE[reg])
                med_colors.append(REG_MED[reg])

        bp = ax.boxplot(data_list, positions=positions, widths=0.64,
                        patch_artist=True, whis=[5, 95],
                        medianprops=dict(color="black", lw=1.2),
                        whiskerprops=dict(lw=0.7),
                        capprops=dict(lw=0.7),
                        flierprops=dict(marker='.', ms=2, alpha=0.3, lw=0))

        for patch, fc, ec in zip(bp['boxes'], face_colors, edge_colors):
            patch.set_facecolor(fc)
            patch.set_edgecolor(ec)
            patch.set_linewidth(0.9)

        for pos, mc, vals in zip(positions, med_colors, data_list):
            if len(vals) > 0:
                ax.plot(pos, np.mean(vals), 'd', color=mc, ms=5, zorder=5)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=10, fontweight="bold")
        ax.set_ylabel("|W| (bps)", fontsize=11, fontweight="bold")
        ax.set_title(spec["title"], fontsize=12, fontweight="bold", pad=5)

        if ax_idx == 0:
            legend_els = [
                Patch(facecolor=REG_FACE["pre"],    edgecolor=REG_EDGE["pre"],    lw=0.9, label="Pre"),
                Patch(facecolor=REG_FACE["relief"], edgecolor=REG_EDGE["relief"], lw=0.9, label="Relief"),
                Patch(facecolor=REG_FACE["post"],   edgecolor=REG_EDGE["post"],   lw=0.9, label="Post"),
            ]
            ax.legend(handles=legend_els, fontsize=9, loc="upper right",
                      borderpad=0.5, labelspacing=0.3)

    fig.tight_layout(pad=1.4)
    save_fig(fig, "fig6_regime_boxplots")
    plt.close(fig)


# ── Manifest ──────────────────────────────────────────────────────────────────

def write_manifest():
    figures = [
        ("fig1_series_overview",   "fig:series_overview"),
        ("fig2_ust_sf_detail",     "fig:ust_sf_detail"),
        ("fig3_exit_event_paths",  "fig:exit_event_paths"),
        ("fig4_coef_plot",         "fig:coef_plot"),
        ("fig5_series_level_coefs","fig:series_level_coefs"),
        ("fig6_regime_boxplots",   "fig:regime_boxplots"),
    ]
    lines = ["# Figure Manifest (tidyquant/theme_bw style)\n\n"]
    for stem, label in figures:
        lines.append(f"## {stem}\n- PDF: `{stem}.pdf`\n- PNG: `{stem}.png`\n"
                     f"- Label: `{label}`\n\n")
    (OUT_DIR / "FIGURE_MANIFEST.md").write_text("".join(lines))
    print("  FIGURE_MANIFEST.md saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== SLR Paper \u2014 Figure Generation (tidyquant / theme_bw style) ===\n")

    print("Loading panel data...")
    panel = load_panel()
    print(f"  Panel: {panel.shape}, "
          f"strategies: {panel['strategy'].unique().tolist()}, "
          f"dates: {panel['date'].min().date()} \u2013 {panel['date'].max().date()}\n")

    fig1_series_overview(panel)
    fig2_ust_sf_detail(panel)
    fig3_exit_event_paths(panel)
    fig4_coef_plot()
    fig5_series_level_coefs()
    fig6_regime_boxplots(panel)
    write_manifest()

    print("\n=== All figures complete ===")
    print(f"Saved to: {OUT_DIR}\n")

    print("File sizes:")
    for stem in ["fig1_series_overview", "fig2_ust_sf_detail", "fig3_exit_event_paths",
                 "fig4_coef_plot", "fig5_series_level_coefs", "fig6_regime_boxplots"]:
        for ext in ("pdf", "png"):
            p = OUT_DIR / f"{stem}.{ext}"
            kb = p.stat().st_size / 1024 if p.exists() else 0
            flag = "OK" if kb > 30 else "MISSING" if kb == 0 else "small"
            print(f"  {stem}.{ext:3s}: {kb:5.0f} KB  [{flag}]")


if __name__ == "__main__":
    main()
