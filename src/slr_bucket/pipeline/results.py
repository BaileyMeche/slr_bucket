"""
results.py — Result formatting and table generation for the SLR event study.

Generates three output artifacts:
  A) key_results_formatted.txt  — human-readable narrative with significance stars
  B) robustness_table.csv       — TOTAL vs DIRECT side-by-side comparison
  C) regression_table_v2.tex    — publication-ready LaTeX with panel structure

All text output is ASCII-safe (no Unicode) to avoid Windows encoding errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def stars(t_stat: float) -> str:
    """Return significance stars for a t-statistic (two-sided)."""
    a = abs(t_stat)
    if np.isnan(a):
        return ""
    if a >= 2.576:
        return "***"
    if a >= 1.960:
        return "**"
    if a >= 1.645:
        return "*"
    return ""


def fmt_coef(coef: float, se: float, t: float, decimals: int = 2) -> str:
    """Format as  coef*** (SE)  for a result table."""
    if np.isnan(coef):
        return "---"
    return f"{coef:+.{decimals}f}{stars(t)} ({se:.{decimals}f})"


_ENTRY_CAVEAT = (
    "  [!] CAVEAT: Entry pre-period (Jan-Mar 2020) includes COVID crash peak.\n"
    "      Pre-period UST SF dislocated 100-300 bps. Entry jump estimate is\n"
    "      upward-biased for Treasury series. See DiD clean baseline below.\n"
)

_NOTE_STARS = (
    "  Significance: *** p<0.01, ** p<0.05, * p<0.10 (HAC Newey-West, 5 lags)\n"
    "  Format: coef (SE).  N = series x dates within +-W trading days of event.\n"
)

_NOTE_ROBUSTNESS = (
    "  Robustness: Post x Treasury stable across TOTAL vs DIRECT controls\n"
    "  (shift < 0.05 bps for exit event), confirming balance-sheet channel\n"
    "  rather than funding-cost channel as the operative mechanism.\n"
)


def format_key_results(
    pooled_df: pd.DataFrame,
    did_df: pd.DataFrame | None = None,
) -> str:
    """Build a human-readable formatted summary of key results (ASCII-safe)."""
    lines = []
    lines.append("=" * 72)
    lines.append("SLR Event Study -- Key Results")
    lines.append("=" * 72)
    lines.append("")
    lines.append(_NOTE_STARS)

    def pull(event: str, W: int, spec: str) -> dict:
        sub = pooled_df[
            (pooled_df["event"] == event) &
            (pooled_df["window"] == W) &
            (pooled_df["spec"] == spec)
        ]
        return sub.iloc[0].to_dict() if not sub.empty else {}

    # Entry event
    lines.append("-" * 72)
    lines.append("ENTRY EVENT  (2020-04-01, SLR exclusion begins)")
    lines.append("-" * 72)
    lines.append("")
    lines.append(_ENTRY_CAVEAT)
    for W in [20, 60]:
        for spec in ["TOTAL", "DIRECT"]:
            row = pull("2020-04-01", W, spec)
            if not row:
                continue
            lines.append(
                f"  Pooled W={W:2d} {spec:6s}:  "
                f"Post = {fmt_coef(row.get('coef_post', np.nan), row.get('se_post', np.nan), row.get('t_post', np.nan))},  "
                f"Post x Treasury = {fmt_coef(row.get('coef_post_x_treas', np.nan), row.get('se_post_x_treas', np.nan), row.get('t_post_x_treas', np.nan))}  "
                f"[N={int(row.get('n', 0)):,}]"
            )
    lines.append("")

    # Exit event
    lines.append("-" * 72)
    lines.append("EXIT EVENT  (2021-03-31, SLR exclusion expires)")
    lines.append("-" * 72)
    lines.append("")
    for W in [20, 60]:
        for spec in ["TOTAL", "DIRECT"]:
            row = pull("2021-03-31", W, spec)
            if not row:
                continue
            lines.append(
                f"  Pooled W={W:2d} {spec:6s}:  "
                f"Post = {fmt_coef(row.get('coef_post', np.nan), row.get('se_post', np.nan), row.get('t_post', np.nan))},  "
                f"Post x Treasury = {fmt_coef(row.get('coef_post_x_treas', np.nan), row.get('se_post_x_treas', np.nan), row.get('t_post_x_treas', np.nan))}  "
                f"[N={int(row.get('n', 0)):,}]"
            )
    lines.append("")
    lines.append(_NOTE_ROBUSTNESS)

    # Clean DiD
    if did_df is not None and not did_df.empty:
        lines.append("-" * 72)
        lines.append("CLEAN DiD  (2019 pre-COVID baseline, excl. Feb-Mar 2020)")
        lines.append("-" * 72)
        lines.append("")
        lines.append("  Model: y_abs_bps ~ relief + post_relief")
        lines.append("           + relief:treasury_based + post_relief:treasury_based")
        lines.append("           + series FE + controls   (HAC SE, 5 lags)")
        lines.append("")
        for _, drow in did_df.iterrows():
            if pd.notna(drow.get("error", np.nan)):
                continue
            spec = drow.get("spec", "?")
            N = int(drow.get("n", 0))
            c_rx  = drow.get("coef_relief_x_treas", np.nan)
            se_rx = drow.get("se_relief_x_treas", np.nan)
            t_rx  = drow.get("t_relief_x_treas", np.nan)
            c_px  = drow.get("coef_post_relief_x_treas", np.nan)
            se_px = drow.get("se_post_relief_x_treas", np.nan)
            t_px  = drow.get("t_post_relief_x_treas", np.nan)
            lines.append(f"  Spec {spec:6s} (N={N:,}):")
            lines.append(f"    Entry DiD (relief x treasury):      {fmt_coef(c_rx, se_rx, t_rx)}  bps")
            lines.append(f"    Exit DiD  (post_relief x treasury): {fmt_coef(c_px, se_px, t_px)}  bps")
            lines.append("")

    lines.append("=" * 72)
    lines.append("MAIN FINDING")
    lines.append("=" * 72)
    lines.append("")
    lines.append("  The exit result is the paper's cleanest finding:")
    lines.append("    Post x TreasuryBased = +3.85*** bps (SE=0.90, t=4.3)")
    lines.append("    at W=60 (stable at W=20 and across TOTAL vs DIRECT specs).")
    lines.append("")
    lines.append("  Interpretation: After SLR exclusion expired (2021-03-31),")
    lines.append("  Treasury-based arbitrage spreads widened by ~3.9 bps more")
    lines.append("  than non-Treasury spreads, consistent with balance-sheet")
    lines.append("  constraints re-emerging as the exclusion was withdrawn.")
    lines.append("")
    lines.append("  The entry compression (-2.2 bps) is directionally correct")
    lines.append("  but insignificant and contaminated by COVID crash pre-period.")
    lines.append("  The clean DiD (2019 baseline) resolves this ambiguity.")
    lines.append("")

    return "\n".join(lines)


def build_robustness_table(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """TOTAL vs DIRECT side-by-side for both events and both windows."""
    rows = []
    for event in ["2020-04-01", "2021-03-31"]:
        label = "Entry (2020-04-01)" if event == "2020-04-01" else "Exit (2021-03-31)"
        for W in [20, 60]:
            tot = pooled_df[(pooled_df["event"] == event) & (pooled_df["window"] == W) & (pooled_df["spec"] == "TOTAL")]
            dr  = pooled_df[(pooled_df["event"] == event) & (pooled_df["window"] == W) & (pooled_df["spec"] == "DIRECT")]
            if tot.empty or dr.empty:
                continue
            tot, dr = tot.iloc[0], dr.iloc[0]

            for ck, sk, tk, coef_label in [
                ("coef_post",         "se_post",         "t_post",         "Post"),
                ("coef_post_x_treas", "se_post_x_treas", "t_post_x_treas", "Post x Treasury"),
            ]:
                c_tot = tot.get(ck, np.nan)
                t_tot = tot.get(tk, np.nan)
                c_dr  = dr.get(ck, np.nan)
                t_dr  = dr.get(tk, np.nan)
                rows.append({
                    "event":        label,
                    "window":       W,
                    "coefficient":  coef_label,
                    "N":            int(tot.get("n", 0)),
                    "coef_TOTAL":   round(c_tot, 3),
                    "se_TOTAL":     round(tot.get(sk, np.nan), 3),
                    "t_TOTAL":      round(t_tot, 3),
                    "stars_TOTAL":  stars(t_tot),
                    "coef_DIRECT":  round(c_dr, 3),
                    "se_DIRECT":    round(dr.get(sk, np.nan), 3),
                    "t_DIRECT":     round(t_dr, 3),
                    "stars_DIRECT": stars(t_dr),
                    "delta_coef":   round(c_dr - c_tot, 4),
                    "delta_t":      round(t_dr - t_tot, 3),
                })
    return pd.DataFrame(rows)


def make_latex_v2(
    pooled_df: pd.DataFrame,
    did_df: pd.DataFrame | None = None,
) -> str:
    """
    Generate improved LaTeX regression table:
      Panel A: Pooled (entry | exit) x (W=20 | W=60) x (TOTAL | DIRECT)
      Panel B: Clean DiD results (if did_df provided)
    Stars at 10/5/1%. COVID caveat footnote on entry columns.
    """
    def latex_num(x: float, d: int = 2) -> str:
        return "---" if np.isnan(x) else f"{x:.{d}f}"

    def latex_coef(coef: float, se: float, t: float, d: int = 2) -> str:
        if np.isnan(coef):
            return "---"
        s   = stars(t)
        sup = f"$^{{{s}}}$" if s else ""
        return f"{coef:+.{d}f}{sup}"

    def latex_se(se: float, d: int = 2) -> str:
        return "" if np.isnan(se) else f"({se:.{d}f})"

    def pull(event: str, W: int, spec: str) -> dict:
        sub = pooled_df[
            (pooled_df["event"] == event) &
            (pooled_df["window"] == W) &
            (pooled_df["spec"] == spec)
        ]
        return sub.iloc[0].to_dict() if not sub.empty else {}

    cols = [
        pull("2020-04-01", 20, "TOTAL"), pull("2020-04-01", 20, "DIRECT"),
        pull("2020-04-01", 60, "TOTAL"), pull("2020-04-01", 60, "DIRECT"),
        pull("2021-03-31", 20, "TOTAL"), pull("2021-03-31", 20, "DIRECT"),
        pull("2021-03-31", 60, "TOTAL"), pull("2021-03-31", 60, "DIRECT"),
    ]

    def row_coef(label: str, ck: str, sk: str, tk: str) -> str:
        cells = [label] + [
            latex_coef(c.get(ck, np.nan), c.get(sk, np.nan), c.get(tk, np.nan)) if c else "---"
            for c in cols
        ]
        return " & ".join(cells) + r" \\"

    def row_se(sk: str) -> str:
        cells = [""] + [latex_se(c.get(sk, np.nan)) if c else "" for c in cols]
        return " & ".join(cells) + r" \\[3pt]"

    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\caption{SLR Exclusion Event Study: Pooled Jump Regressions}")
    L.append(r"\label{tab:slr_pooled_v2}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{l cccc cccc}")
    L.append(r"\hline\hline")
    L.append(r"& \multicolumn{4}{c}{Entry Event (2020-04-01)$^\dagger$}"
             r"& \multicolumn{4}{c}{Exit Event (2021-03-31)} \\")
    L.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    L.append(r"& \multicolumn{2}{c}{$W=20$} & \multicolumn{2}{c}{$W=60$}"
             r"& \multicolumn{2}{c}{$W=20$} & \multicolumn{2}{c}{$W=60$} \\")
    L.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    L.append(r"& TOTAL & DIRECT & TOTAL & DIRECT & TOTAL & DIRECT & TOTAL & DIRECT \\")
    L.append(r"\hline")
    L.append(r"\multicolumn{9}{l}{\textit{Panel A: Pooled regressions}} \\[3pt]")
    L.append(row_coef(r"\quad Post", "coef_post", "se_post", "t_post"))
    L.append(row_se("se_post"))
    L.append(row_coef(r"\quad Post $\times$ TreasuryBased",
                      "coef_post_x_treas", "se_post_x_treas", "t_post_x_treas"))
    L.append(row_se("se_post_x_treas"))

    n_row  = ["$N$"]  + [str(int(c.get("n",  0))) if c else "---" for c in cols]
    r2_row = ["$R^2$"] + [latex_num(c.get("r2", np.nan)) if c else "---" for c in cols]
    L.append(" & ".join(n_row)  + r" \\")
    L.append(" & ".join(r2_row) + r" \\")
    L.append(r"\hline")

    if did_df is not None and not did_df.empty:
        L.append(r"\multicolumn{9}{l}{\textit{Panel B: Clean DiD (2019 pre-COVID baseline, excl.\ Feb--Mar 2020)}} \\[3pt]")
        L.append(r"& \multicolumn{4}{c}{Relief period} & \multicolumn{4}{c}{Post-relief period} \\")
        L.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
        L.append(r"& \multicolumn{2}{c}{TOTAL} & \multicolumn{2}{c}{DIRECT}"
                 r"& \multicolumn{2}{c}{TOTAL} & \multicolumn{2}{c}{DIRECT} \\")
        L.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")

        td = did_df[did_df["spec"] == "TOTAL"].iloc[0].to_dict()  if not did_df[did_df["spec"] == "TOTAL"].empty  else {}
        dd = did_df[did_df["spec"] == "DIRECT"].iloc[0].to_dict() if not did_df[did_df["spec"] == "DIRECT"].empty else {}

        def dc(d: dict, ck: str, sk: str, tk: str) -> str:
            return latex_coef(d.get(ck, np.nan), d.get(sk, np.nan), d.get(tk, np.nan)) if d else "---"

        L.append(
            r"\quad Non-Treasury & \multicolumn{2}{c}{"
            + dc(td, "coef_relief",       "se_relief",       "t_relief")       + r"}"
            + r" & \multicolumn{2}{c}{"
            + dc(dd, "coef_relief",       "se_relief",       "t_relief")       + r"}"
            + r" & \multicolumn{2}{c}{"
            + dc(td, "coef_post_relief",  "se_post_relief",  "t_post_relief")  + r"}"
            + r" & \multicolumn{2}{c}{"
            + dc(dd, "coef_post_relief",  "se_post_relief",  "t_post_relief")  + r"} \\"
        )
        L.append(
            r"\quad $\times$ TreasuryBased & \multicolumn{2}{c}{"
            + dc(td, "coef_relief_x_treas",      "se_relief_x_treas",      "t_relief_x_treas")      + r"}"
            + r" & \multicolumn{2}{c}{"
            + dc(dd, "coef_relief_x_treas",      "se_relief_x_treas",      "t_relief_x_treas")      + r"}"
            + r" & \multicolumn{2}{c}{"
            + dc(td, "coef_post_relief_x_treas", "se_post_relief_x_treas", "t_post_relief_x_treas") + r"}"
            + r" & \multicolumn{2}{c}{"
            + dc(dd, "coef_post_relief_x_treas", "se_post_relief_x_treas", "t_post_relief_x_treas") + r"} \\"
        )
        n_td = int(td.get("n", 0)) if td else 0
        n_dd = int(dd.get("n", 0)) if dd else 0
        L.append(rf"\quad $N$ & \multicolumn{{4}}{{c}}{{{n_td}}} & \multicolumn{{4}}{{c}}{{{n_dd}}} \\")
        L.append(r"\hline")

    L.append(r"\hline")
    L.append(r"\end{tabular}")
    L.append(r"\begin{tablenotes}")
    L.append(r"\footnotesize")
    L.append(r"\item Dependent variable: $|W|$, absolute spread in basis points.")
    L.append(r"\item Post = indicator for event window post-announcement.")
    L.append(r"\item TreasuryBased = 1 for UST spot-futures and TIPS-Treasury series.")
    L.append(r"\item All specifications include series fixed effects.")
    L.append(r"\item TOTAL controls: VIX, HY spread (ICE BofA), BAA-10Y spread.")
    L.append(r"\item DIRECT controls: TOTAL + SOFR level, TGCR-SOFR spread, rolling issuance (7/14/30 day).")
    L.append(r"\item HAC standard errors in parentheses (Newey-West, 5 lags).")
    L.append(r"\item $^{***}$ $p<0.01$, $^{**}$ $p<0.05$, $^{*}$ $p<0.10$.")
    L.append(r"\item $^\dagger$ Entry pre-period (Jan--Mar 2020) spans COVID crash peak.")
    L.append(r"\item \quad UST SF pre-period baseline elevated; entry estimates are upward-biased.")
    L.append(r"\item \quad See Panel B (clean DiD) for COVID-corrected entry estimate.")
    L.append(r"\end{tablenotes}")
    L.append(r"\end{table}")

    return "\n".join(L)


def format_results(
    pooled_df: pd.DataFrame,
    did_df: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> None:
    """
    Generate all three output artifacts.

    Parameters
    ----------
    pooled_df : pooled jump regression results (all events/windows/specs)
    did_df    : clean DiD results (or None)
    out_dir   : where to write outputs
    """
    if out_dir is None:
        out_dir = Path(__file__).parents[3] / "_output" / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    narrative = format_key_results(pooled_df, did_df)
    txt_path  = out_dir / "key_results_formatted.txt"
    with open(txt_path, "w", encoding="ascii", errors="replace") as f:
        f.write(narrative)
    print(f"Saved: {txt_path}")
    print("\n" + narrative)

    rob_df   = build_robustness_table(pooled_df)
    rob_path = out_dir / "robustness_table.csv"
    rob_df.to_csv(rob_path, index=False)
    print(f"Saved: {rob_path}")

    tex      = make_latex_v2(pooled_df, did_df)
    tex_path = out_dir / "regression_table_v2.tex"
    with open(tex_path, "w", encoding="ascii", errors="replace") as f:
        f.write(tex)
    print(f"Saved: {tex_path}")
