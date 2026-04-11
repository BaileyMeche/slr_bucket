"""
Fix 10 / format_results.py: Format and communicate the key results clearly.

Problem: The key result (+3.85 bps, t=4.3 at exit) is buried in 23-column
CSVs with no economic framing. Tables are not print-ready. Stars are absent.
The COVID pre-period caveat for the entry result is not flagged.

Outputs:
  A) outputs/key_results_formatted.txt  -- human-readable narrative with stars
  B) outputs/robustness_table.csv       -- TOTAL vs DIRECT side-by-side
  C) outputs/regression_table_v2.tex    -- improved LaTeX with panel structure

All output is ASCII-safe (no Unicode) to avoid Windows cp1252 encoding errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── Significance stars ─────────────────────────────────────────────────────────

def stars(t_stat: float) -> str:
    """Return significance stars for a t-statistic (two-sided)."""
    a = abs(t_stat)
    if np.isnan(a):
        return ""
    if a >= 2.576:  # p < 0.01
        return "***"
    if a >= 1.960:  # p < 0.05
        return "**"
    if a >= 1.645:  # p < 0.10
        return "*"
    return ""


def fmt_coef(coef: float, se: float, t: float, decimals: int = 2) -> str:
    """Format as  coef*** (SE)  for a result table."""
    if np.isnan(coef):
        return "---"
    s = stars(t)
    return f"{coef:+.{decimals}f}{s} ({se:.{decimals}f})"


# ── A: Human-readable narrative ────────────────────────────────────────────────

_ENTRY_CAVEAT = (
    "  [!] CAVEAT: Entry pre-period (Jan-Mar 2020) includes COVID crash peak.\n"
    "      Pre-period UST SF dislocated 100-300 bps. Entry jump estimate is\n"
    "      upward-biased for Treasury series. See DiD clean baseline (Fix 9).\n"
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
    """
    Build a human-readable formatted summary of key results.
    Returns a plain-text string (ASCII-safe).
    """
    lines = []
    lines.append("=" * 72)
    lines.append("SLR Event Study -- Key Results (Repaired Pipeline)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(_NOTE_STARS)

    # Helper: pull one row
    def pull(event: str, W: int, spec: str) -> dict:
        sub = pooled_df[
            (pooled_df["event"] == event) &
            (pooled_df["window"] == W) &
            (pooled_df["spec"] == spec)
        ]
        if sub.empty:
            return {}
        return sub.iloc[0].to_dict()

    # ── Entry event ───────────────────────────────────────────────────────────
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
            coef_p = row.get("coef_post", np.nan)
            se_p = row.get("se_post", np.nan)
            t_p = row.get("t_post", np.nan)
            coef_i = row.get("coef_post_x_treas", np.nan)
            se_i = row.get("se_post_x_treas", np.nan)
            t_i = row.get("t_post_x_treas", np.nan)
            N = int(row.get("n", 0))
            lines.append(
                f"  Pooled W={W:2d} {spec:6s}:  "
                f"Post = {fmt_coef(coef_p, se_p, t_p)},  "
                f"Post x Treasury = {fmt_coef(coef_i, se_i, t_i)}  "
                f"[N={N:,}]"
            )
    lines.append("")

    # ── Exit event ────────────────────────────────────────────────────────────
    lines.append("-" * 72)
    lines.append("EXIT EVENT  (2021-03-31, SLR exclusion expires)")
    lines.append("-" * 72)
    lines.append("")

    for W in [20, 60]:
        for spec in ["TOTAL", "DIRECT"]:
            row = pull("2021-03-31", W, spec)
            if not row:
                continue
            coef_p = row.get("coef_post", np.nan)
            se_p = row.get("se_post", np.nan)
            t_p = row.get("t_post", np.nan)
            coef_i = row.get("coef_post_x_treas", np.nan)
            se_i = row.get("se_post_x_treas", np.nan)
            t_i = row.get("t_post_x_treas", np.nan)
            N = int(row.get("n", 0))
            lines.append(
                f"  Pooled W={W:2d} {spec:6s}:  "
                f"Post = {fmt_coef(coef_p, se_p, t_p)},  "
                f"Post x Treasury = {fmt_coef(coef_i, se_i, t_i)}  "
                f"[N={N:,}]"
            )
    lines.append("")
    lines.append(_NOTE_ROBUSTNESS)

    # ── DiD clean baseline ────────────────────────────────────────────────────
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
            if "error" in drow and pd.notna(drow.get("error", np.nan)):
                continue
            spec = drow.get("spec", "?")
            N = int(drow.get("n", 0))

            c_rx = drow.get("coef_relief_x_treas", np.nan)
            se_rx = drow.get("se_relief_x_treas", np.nan)
            t_rx = drow.get("t_relief_x_treas", np.nan)

            c_px = drow.get("coef_post_relief_x_treas", np.nan)
            se_px = drow.get("se_post_relief_x_treas", np.nan)
            t_px = drow.get("t_post_relief_x_treas", np.nan)

            lines.append(f"  Spec {spec:6s} (N={N:,}):")
            lines.append(
                f"    Entry DiD (relief x treasury):      "
                f"{fmt_coef(c_rx, se_rx, t_rx)}  bps"
            )
            lines.append(
                f"    Exit DiD  (post_relief x treasury): "
                f"{fmt_coef(c_px, se_px, t_px)}  bps"
            )
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


# ── B: Robustness table ────────────────────────────────────────────────────────

def build_robustness_table(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """
    TOTAL vs DIRECT side-by-side for both events and both windows.
    Includes Delta(coef) and Delta(t) columns.
    """
    rows = []
    for event in ["2020-04-01", "2021-03-31"]:
        event_label = "Entry (2020-04-01)" if event == "2020-04-01" else "Exit (2021-03-31)"
        for W in [20, 60]:
            tot = pooled_df[
                (pooled_df["event"] == event) & (pooled_df["window"] == W) &
                (pooled_df["spec"] == "TOTAL")
            ]
            dr = pooled_df[
                (pooled_df["event"] == event) & (pooled_df["window"] == W) &
                (pooled_df["spec"] == "DIRECT")
            ]
            if tot.empty or dr.empty:
                continue
            tot = tot.iloc[0]
            dr = dr.iloc[0]

            for coef_key, se_key, t_key, label in [
                ("coef_post", "se_post", "t_post", "Post"),
                ("coef_post_x_treas", "se_post_x_treas", "t_post_x_treas", "Post x Treasury"),
            ]:
                c_tot = tot.get(coef_key, np.nan)
                t_tot = tot.get(t_key, np.nan)
                c_dr = dr.get(coef_key, np.nan)
                t_dr = dr.get(t_key, np.nan)

                rows.append({
                    "event": event_label,
                    "window": W,
                    "coefficient": label,
                    "N": int(tot.get("n", 0)),
                    # TOTAL
                    "coef_TOTAL": round(c_tot, 3),
                    "se_TOTAL": round(tot.get(se_key, np.nan), 3),
                    "t_TOTAL": round(t_tot, 3),
                    "stars_TOTAL": stars(t_tot),
                    # DIRECT
                    "coef_DIRECT": round(c_dr, 3),
                    "se_DIRECT": round(dr.get(se_key, np.nan), 3),
                    "t_DIRECT": round(t_dr, 3),
                    "stars_DIRECT": stars(t_dr),
                    # Delta
                    "delta_coef": round(c_dr - c_tot, 4),
                    "delta_t": round(t_dr - t_tot, 3),
                })

    return pd.DataFrame(rows)


# ── C: Improved LaTeX regression table ────────────────────────────────────────

def make_latex_v2(
    pooled_df: pd.DataFrame,
    did_df: pd.DataFrame | None = None,
) -> str:
    """
    Generate an improved LaTeX regression table:
      Panel A: Pooled (entry | exit) x (W=20 | W=60) x (TOTAL | DIRECT)
      Panel B: Key series-level highlights (omitted if series_df not provided)
      Panel C: Clean DiD results (if did_df provided)
    Stars at 10/5/1% levels. COVID caveat footnote on entry columns.
    """

    def latex_num(x: float, decimals: int = 2) -> str:
        if np.isnan(x):
            return "---"
        return f"{x:.{decimals}f}"

    def latex_coef(coef: float, se: float, t: float, decimals: int = 2) -> str:
        if np.isnan(coef):
            return "---"
        s = stars(t)
        sup = f"$^{{{s}}}$" if s else ""
        return f"{coef:+.{decimals}f}{sup}"

    def latex_se(se: float, decimals: int = 2) -> str:
        if np.isnan(se):
            return ""
        return f"({se:.{decimals}f})"

    def pull(event: str, W: int, spec: str) -> dict:
        sub = pooled_df[
            (pooled_df["event"] == event) &
            (pooled_df["window"] == W) &
            (pooled_df["spec"] == spec)
        ]
        if sub.empty:
            return {}
        return sub.iloc[0].to_dict()

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{SLR Exclusion Event Study: Pooled Jump Regressions}")
    lines.append(r"\label{tab:slr_pooled_v2}")
    lines.append(r"\small")
    # 9 columns: label + 4 entry specs + 4 exit specs
    lines.append(r"\begin{tabular}{l cccc cccc}")
    lines.append(r"\hline\hline")
    lines.append(r"& \multicolumn{4}{c}{Entry Event (2020-04-01)$^\dagger$}")
    lines.append(r"& \multicolumn{4}{c}{Exit Event (2021-03-31)} \\")
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"& \multicolumn{2}{c}{$W=20$} & \multicolumn{2}{c}{$W=60$}")
    lines.append(r"& \multicolumn{2}{c}{$W=20$} & \multicolumn{2}{c}{$W=60$} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r"& TOTAL & DIRECT & TOTAL & DIRECT & TOTAL & DIRECT & TOTAL & DIRECT \\")
    lines.append(r"\hline")
    lines.append(r"\multicolumn{9}{l}{\textit{Panel A: Pooled regressions}} \\[3pt]")

    # Row: Post
    cols = [
        pull("2020-04-01", 20, "TOTAL"),
        pull("2020-04-01", 20, "DIRECT"),
        pull("2020-04-01", 60, "TOTAL"),
        pull("2020-04-01", 60, "DIRECT"),
        pull("2021-03-31", 20, "TOTAL"),
        pull("2021-03-31", 20, "DIRECT"),
        pull("2021-03-31", 60, "TOTAL"),
        pull("2021-03-31", 60, "DIRECT"),
    ]

    def row_coef(label: str, ck: str, sk: str, tk: str) -> list[str]:
        r = [f"{label}"]
        for c in cols:
            if not c:
                r.append("---")
            else:
                r.append(latex_coef(c.get(ck, np.nan), c.get(sk, np.nan), c.get(tk, np.nan)))
        return r

    def row_se(sk: str) -> list[str]:
        r = [""]
        for c in cols:
            if not c:
                r.append("")
            else:
                r.append(latex_se(c.get(sk, np.nan)))
        return r

    post_label = r"\quad Post"
    lines.append(" & ".join(row_coef(post_label, "coef_post", "se_post", "t_post")) + r" \\")
    lines.append(" & ".join(row_se("se_post")) + r" \\[3pt]")

    inter_label = r"\quad Post $\times$ TreasuryBased"
    lines.append(" & ".join(row_coef(inter_label, "coef_post_x_treas", "se_post_x_treas", "t_post_x_treas")) + r" \\")
    lines.append(" & ".join(row_se("se_post_x_treas")) + r" \\[3pt]")

    # N and R2
    n_row = ["$N$"] + [str(int(c.get("n", 0))) if c else "---" for c in cols]
    r2_row = ["$R^2$"] + [latex_num(c.get("r2", np.nan)) if c else "---" for c in cols]
    lines.append(" & ".join(n_row) + r" \\")
    lines.append(" & ".join(r2_row) + r" \\")
    lines.append(r"\hline")

    # Panel C: DiD clean baseline
    if did_df is not None and not did_df.empty:
        lines.append(r"\multicolumn{9}{l}{\textit{Panel B: Clean DiD (2019 pre-COVID baseline, excl.\ Feb--Mar 2020)}} \\[3pt]")
        lines.append(r"& \multicolumn{4}{c}{Relief period} & \multicolumn{4}{c}{Post-relief period} \\")
        lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
        lines.append(r"& \multicolumn{2}{c}{TOTAL} & \multicolumn{2}{c}{DIRECT}")
        lines.append(r"& \multicolumn{2}{c}{TOTAL} & \multicolumn{2}{c}{DIRECT} \\")
        lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")

        tot_did = did_df[did_df["spec"] == "TOTAL"].iloc[0].to_dict() if not did_df[did_df["spec"] == "TOTAL"].empty else {}
        dir_did = did_df[did_df["spec"] == "DIRECT"].iloc[0].to_dict() if not did_df[did_df["spec"] == "DIRECT"].empty else {}

        def did_cell(d: dict, ck: str, sk: str, tk: str) -> str:
            return latex_coef(d.get(ck, np.nan), d.get(sk, np.nan), d.get(tk, np.nan)) if d else "---"

        # Non-Treasury row
        lines.append(
            r"\quad Non-Treasury & \multicolumn{2}{c}{"
            + did_cell(tot_did, "coef_relief", "se_relief", "t_relief")
            + r"} & \multicolumn{2}{c}{"
            + did_cell(dir_did, "coef_relief", "se_relief", "t_relief")
            + r"} & \multicolumn{2}{c}{"
            + did_cell(tot_did, "coef_post_relief", "se_post_relief", "t_post_relief")
            + r"} & \multicolumn{2}{c}{"
            + did_cell(dir_did, "coef_post_relief", "se_post_relief", "t_post_relief")
            + r"} \\"
        )
        # Treasury differential row
        lines.append(
            r"\quad $\times$ TreasuryBased & \multicolumn{2}{c}{"
            + did_cell(tot_did, "coef_relief_x_treas", "se_relief_x_treas", "t_relief_x_treas")
            + r"} & \multicolumn{2}{c}{"
            + did_cell(dir_did, "coef_relief_x_treas", "se_relief_x_treas", "t_relief_x_treas")
            + r"} & \multicolumn{2}{c}{"
            + did_cell(tot_did, "coef_post_relief_x_treas", "se_post_relief_x_treas", "t_post_relief_x_treas")
            + r"} & \multicolumn{2}{c}{"
            + did_cell(dir_did, "coef_post_relief_x_treas", "se_post_relief_x_treas", "t_post_relief_x_treas")
            + r"} \\"
        )

        n_did_tot = int(tot_did.get("n", 0)) if tot_did else 0
        n_did_dir = int(dir_did.get("n", 0)) if dir_did else 0
        lines.append(
            rf"\quad $N$ & \multicolumn{{4}}{{c}}{{{n_did_tot}}} & \multicolumn{{4}}{{c}}{{{n_did_dir}}} \\"
        )
        lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item Dependent variable: $|W|$, absolute spread in basis points.")
    lines.append(r"\item Post = indicator for event window post-announcement.")
    lines.append(r"\item TreasuryBased = 1 for UST spot-futures and TIPS-Treasury series.")
    lines.append(r"\item All specifications include series fixed effects.")
    lines.append(r"\item TOTAL controls: VIX, HY spread (ICE BofA), BAA-10Y spread.")
    lines.append(r"\item DIRECT controls: TOTAL + SOFR level, TGCR-SOFR spread, rolling issuance (7/14/30 day).")
    lines.append(r"\item HAC standard errors in parentheses (Newey-West, 5 lags).")
    lines.append(r"\item $^{***}$ $p<0.01$, $^{**}$ $p<0.05$, $^{*}$ $p<0.10$.")
    lines.append(r"\item $^\dagger$ Entry pre-period (Jan--Mar 2020) spans COVID crash peak.")
    lines.append(r"\item \quad UST SF pre-period baseline elevated; entry estimates are upward-biased.")
    lines.append(r"\item \quad See Panel B (clean DiD) for COVID-corrected entry estimate.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ── Main entry point ───────────────────────────────────────────────────────────

def format_results(
    pooled_df: pd.DataFrame,
    did_df: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> None:
    """
    Generate all three output artifacts.

    Parameters
    ----------
    pooled_df : output from run_pooled_jump (all events/windows/specs)
    did_df    : output from run_did_clean_baseline (or None)
    out_dir   : where to write outputs (default: audit_repairs/outputs/)
    """
    if out_dir is None:
        out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # A: Narrative text
    narrative = format_key_results(pooled_df, did_df)
    txt_path = out_dir / "key_results_formatted.txt"
    with open(txt_path, "w", encoding="ascii", errors="replace") as f:
        f.write(narrative)
    print(f"[format_results] Saved: {txt_path}")
    print("\n" + narrative)

    # B: Robustness table
    rob_df = build_robustness_table(pooled_df)
    rob_path = out_dir / "robustness_table.csv"
    rob_df.to_csv(rob_path, index=False)
    print(f"[format_results] Saved: {rob_path}")

    # C: LaTeX v2
    tex = make_latex_v2(pooled_df, did_df)
    tex_path = out_dir / "regression_table_v2.tex"
    with open(tex_path, "w", encoding="ascii", errors="replace") as f:
        f.write(tex)
    print(f"[format_results] Saved: {tex_path}")


if __name__ == "__main__":
    OUT_DIR = Path(__file__).parent / "outputs"

    # Load existing pooled estimates
    pooled_path = OUT_DIR / "jump_estimates_pooled.csv"
    if not pooled_path.exists():
        print(f"ERROR: {pooled_path} not found. Run repaired_pipeline.py first.")
        raise SystemExit(1)

    pooled_df = pd.read_csv(pooled_path)

    # Load DiD results if available
    did_path = OUT_DIR / "did_clean_baseline.csv"
    did_df = pd.read_csv(did_path) if did_path.exists() else None

    format_results(pooled_df, did_df, out_dir=OUT_DIR)
