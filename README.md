# Balance-Sheet Capacity and Cross-Strategy Treasury Arbitrage: SLR Exclusion Event Study

# Reproduction Guide — SLR Exclusion Event Study

This document describes every file needed to reproduce the results in
`presentations/draft.tex` and the exact sequence of commands to run them.

---

## Repository layout

```
slr_bucket/
├── data/
│   ├── raw/                          # Original source data (read-only)
│   │   ├── cip_bloomberg.xlsx        # CIP spreads, Bloomberg
│   │   ├── OIS.xlsx                  # Overnight index swap rates
│   │   ├── equity_spot_bloomberg.parquet
│   │   ├── treasury_spot_futures.xlsx
│   │   └── event_inputs/
│   │       ├── controls_vix_creditspreads_fred.csv   # VIX, HY_OAS, BAA10Y
│   │       ├── repo_rates_combined.csv               # SOFR, TGCR, BGCR
│   │       └── treasury_issuance_by_tenor_fiscaldata.csv
│   └── series/                       # Pre-processed series, all in bps
│       ├── cip_spreads_3m_bps.csv         # CIP_AUD_ln … CIP_JPY_ln
│       ├── treasury_sf_output.csv          # Treasury_SF_2Y … Treasury_SF_30Y
│       ├── tips_treasury_implied_rf_2010.parquet   # arb_2, arb_5, arb_10
│       ├── equity_spot_spread_SPX.csv
│       ├── equity_spot_spread_NDX.csv
│       └── equity_spot_spread_INDU.csv
│
├── outputs/
│   ├── investigation_outputs/            # Supplementary iagnostics
│   │   ├── task1_cip_aud.py              # CIP-AUD anomaly / Japanese FY-end
│   │   ├── task2_pretrend.py             # Dynamic DiD, 12-bin pre-trend F-test
│   │   ├── task3_reversion.py            # Wald test H0: mu2 = mu4
│   │   ├── task4_ar1.py                  # AR(1) coefficients for all 17 series
│   │   ├── task5_direct_baseline.py      # SOFR effect on non-Treasury baseline
│   │   ├── pretrend_regression_results.csv
│   │   └── reversion_test_results.txt
│   │
│   ├── figures_output/                   # Figure generation code and outputs
│      ├── make_figures.py               # Produces all 6 publication figures
│      ├── figures.tex                   # \input{}-ready figure │environments
│      ├── FIGURE_MANIFEST.md            # Per-figure data sources │and captions
│      ├── fig1_series_overview.pdf/.png
│      ├── fig2_ust_sf_detail.pdf/.png
│      ├── fig3_exit_event_paths.pdf/.png
│      ├── fig4_coef_plot.pdf/.png
│      ├── fig5_series_level_coefs.pdf/.png
│      └── fig6_regime_boxplots.pdf/.png││
├── presentations/
|    ├── draft.tex                     # Full thesis draft
|    └── references.bib                # BibTeX bibliography (10 entries)
└── audit_repairs/                    # All pipeline code
   ├── fix_1_units.py                # stack_all_outcomes(); unit-safe loader
   ├── fix_2_ust_sf.py               # UST SF sign correction
   ├── fix_3_equity.py               # Equity spread_SPX selection
   ├── fix_4_pooled.py               # Full 17-series panel assembly
   ├── fix_5_overlap.py              # Event-window overlap check
   ├── fix_6_controls.py             # HAC SE and controls wiring
   ├── fix_7_winsorize.py            # p1/p99 within-series winsorization
   ├── fix_8_issuance.py             # Rolling issuance controls
   ├── fix_9_did_clean.py            # Clean DiD with 2019 pre-COVID baseline
   ├── format_results.py             # Formatted text/CSV/LaTeX output
   ├── repaired_pipeline.py          # MASTER pipeline — run this
   ├── compute_dk_ses.py             # Driscoll-Kraay SE computation
   └── outputs/                      # All pipeline outputs (generated)
       ├── jump_estimates_pooled.csv
       ├── jump_estimates_by_series.csv
       ├── summary_stats_repaired.csv
       ├── did_clean_baseline.csv
       ├── dk_robustness.csv
       ├── robustness_table.csv
       ├── key_results_formatted.txt
       ├── regression_table_v2.tex
       ├── fix7_winsorize_log.csv
       └── references.bib

```

## Software requirements

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| Python | 3.10 | All pipeline scripts |
| pandas | 1.5 | Data manipulation |
| numpy | 1.23 | Numerical operations |
| scipy | 1.9 | OLS, F-tests |
| statsmodels | 0.13 | HAC (Newey-West) SE |
| matplotlib | 3.6 | All figures |
| pyarrow | 10.0 | Reading .parquet files |
| pdflatex | any recent | Compiling draft.tex |

Install Python dependencies:
```bash
pip install pandas numpy scipy statsmodels matplotlib pyarrow openpyxl
```

---

## Reproduction workflow

Run all commands from the **worktree root** (`slr_bucket/` or the worktree copy).
The `-W ignore` flag suppresses pandas GroupBy deprecation warnings.

---

### Step 1 — Run the master pipeline

This single command executes all 8 pipeline steps in order:

```bash
python -W ignore audit_repairs/repaired_pipeline.py
```

**What it does (in order):**

| Step | Action | Key output |
|------|--------|------------|
| 1 | Event-window overlap check | Prints overlap warning; drops 2021-03-19 from main tables |
| 2 | Load full panel | 17 series × 2019-2021 daily, winsorized at p1/p99 per series |
| 3 | Summary statistics | `outputs/summary_stats_repaired.csv` |
| 4 | Pooled DiD regressions | `outputs/jump_estimates_pooled.csv` (8 rows: 2 events × 2 windows × 2 specs) |
| 4.5 | Clean DiD (2019 baseline) | `outputs/did_clean_baseline.csv` |
| 5 | Series-level regressions | `outputs/jump_estimates_by_series.csv` |
| 6 | LaTeX regression table | `outputs/regression_table_v2.tex` |
| 7 | Format results | `outputs/key_results_formatted.txt`, `outputs/robustness_table.csv` |
| 8 | Hypothesis check | Prints pass/fail for all 3 pre-registered predictions |

---

### Step 2 — Compute Driscoll-Kraay standard errors

```bash
python -W ignore audit_repairs/compute_dk_ses.py
```

**Output:** `audit_repairs/outputs/dk_robustness.csv`

---

### Step 3 — Run supplementary diagnostics

These are independent and can be run in any order:

```bash
# CIP-AUD anomaly and Japanese fiscal year-end analysis
python -W ignore investigation_outputs/task1_cip_aud.py

# Dynamic DiD: 12-bin pre/post event paths + F-test of pre-trends
python -W ignore investigation_outputs/task2_pretrend.py

# Wald test for partial reversion H0: mu_relief = mu_post_relief
python -W ignore investigation_outputs/task3_reversion.py

# AR(1) persistence for all 17 series
python -W ignore investigation_outputs/task4_ar1.py

# SOFR mechanical relationship with non-Treasury baseline
python -W ignore investigation_outputs/task5_direct_baseline.py
```

---

### Step 4 — Generate publication figures

```bash
python -W ignore figures_output/make_figures.py
```

**Outputs:** `figures_output/fig1_*.pdf` through `fig6_*.pdf` (and matching `.png`)

This script loads the panel data using the same `stack_all_outcomes()` + winsorize
path as the master pipeline, ensuring figures and regression tables use identical data.

**Figures produced:**

| File | Content |
|------|---------|
| `fig1_series_overview.pdf` | Time-series mean ± 1 SD of |W| across all 17 series, 2019-2021 |
| `fig2_ust_sf_detail.pdf` | UST spot-futures by tenor (2Y, 5Y, 10Y) |
| `fig3_exit_event_paths.pdf` | Event-time paths ±60 days around 2021-03-31 |
| `fig4_coef_plot.pdf` | Forest plot: all 8 Post x Treasury point estimates |
| `fig5_series_level_coefs.pdf` | Per-series exit coefficients (W=60, DIRECT) |
| `fig6_regime_boxplots.pdf` | IQR boxplots by regime (Pre / Relief / Post) |


---

## Table-to-file map

Every numbered table in `presentations/draft.tex`, in document order, with the
exact repository file that contains its underlying data and the script that
generates that file.

| # | LaTeX label | Caption (abbreviated) | Source CSV | Generated by | draft.tex line |
|---|-------------|----------------------|------------|--------------|---------------|
| 1 | `tab:data_sources` | Data Sources and Coverage | *(manually written — no CSV)* | n/a | 116 |
| 2 | `tab:pretrend` | Event-Study Dynamics Around SLR Exit, W=60 | `investigation_outputs/pretrend_regression_results.csv` | `investigation_outputs/task2_pretrend.py` | 298 |
| 3 | `tab:series_level` | Series-Level Jump Estimates: Exit, W=60, DIRECT | `audit_repairs/outputs/jump_estimates_by_series.csv` | `audit_repairs/repaired_pipeline.py` (Step 5) | 708 |
| 4 | `tab:dk_robustness` | Driscoll-Kraay Robustness: W=60 Pooled | `audit_repairs/outputs/dk_robustness.csv` | `audit_repairs/compute_dk_ses.py` | 864 |
| 5 | `tab:summary_stats` | Summary Statistics for Arbitrage Spreads | `audit_repairs/outputs/summary_stats_repaired.csv` | `audit_repairs/repaired_pipeline.py` (Step 3) | 990 |
| 6 | `tab:pooled_jump` | SLR Exclusion Event Study: Pooled Jump Regressions | `audit_repairs/outputs/jump_estimates_pooled.csv` | `audit_repairs/repaired_pipeline.py` (Step 4) | 1045 |
| 7 | `tab:robustness` | Robustness: Total vs. Direct Specification | `audit_repairs/outputs/robustness_table.csv` | `audit_repairs/repaired_pipeline.py` + `format_results.py` | 1128 |
| 8 | `tab:winsorize` | Winsorization Log: Per-Series Clipping Counts | `audit_repairs/outputs/fix7_winsorize_log.csv` | `audit_repairs/fix_7_winsorize.py` (called by pipeline) | 1171 |

---

## Data provenance

| Series class | Raw source | Pre-processing | Processed file |
|---|---|---|---|
| CIP | `data/raw/cip_bloomberg.xlsx` | Log-price difference, annualize | `data/series/cip_spreads_3m_bps.csv` |
| UST spot-futures | `data/raw/treasury_spot_futures.xlsx` | Basis = spot - futures (carry-adjusted) | `data/series/treasury_sf_output.csv` |
| TIPS-Treasury | `data/raw/OIS.xlsx` + TIPS data | Real-nominal no-arbitrage condition | `data/series/tips_treasury_implied_rf_2010.parquet` |
| Equity SF | `data/raw/equity_spot_bloomberg.parquet` | Dividend-adjusted fair-value basis | `data/series/equity_spot_spread_*.csv` |
| Controls | FRED (VIX, HY_OAS, BAA10Y) | None | `data/raw/event_inputs/controls_vix_creditspreads_fred.csv` |
| Repo controls | NY Fed (SOFR, TGCR, BGCR) | spr_tgcr = TGCR - SOFR | `data/raw/event_inputs/repo_rates_combined.csv` |
| Issuance | TreasuryDirect | 7/14/30-day rolling sum, lag 1 day, /1e9 | built in `fix_8_issuance.py` |

All `data/series/` files are already in basis points. The loader (`fix_1_units.py`)
applies a pass-through — no scaling is done at load time.

---

## Key design choices that affect reproducibility

1. **Winsorization** — Applied at p1/p99 within each series using only 2019-2021
   in-sample data. Thresholds are logged to `outputs/fix7_winsorize_log.csv`.
   Changing `PCT_HI` in `fix_7_winsorize.py` is the main robustness lever.

2. **HAC lags** — Set to 5 (`HAC_LAGS = 5` in `repaired_pipeline.py`).
   AR(1) > 0.93 for 15/17 series justifies at least 5 lags; results are robust to
   lags 3–10 (see `outputs/robustness_table.csv`).

3. **Event-window overlap** — 2021-03-19 dropped from main tables because its
   W=20 window overlaps with the 2021-03-31 exit by 33 trading days.

4. **Clean DiD exclusion window** — 2020-02-01 through 2020-03-31 excluded as
   the COVID crash period. Boundary dates can be adjusted in `fix_9_did_clean.py`.

5. **Series classification** — `treasury_based = 1` for UST spot-futures and
   TIPS-Treasury (6 series); `treasury_based = 0` for CIP and equity SF (11 series).
   This is hard-coded in `fix_1_units.py`.

---

## Verifying the main result

After running Steps 1-4, run from the worktree root (all three checks should pass):

```python
import pandas as pd

# ── Check 1: Pooled jump — exit event, W=60, TOTAL ────────────────────────
# Column name is "event" (not "event_date"); values are date strings e.g. "2021-03-31"
df = pd.read_csv("audit_repairs/outputs/jump_estimates_pooled.csv")
row = df[(df["event"] == "2021-03-31") & (df["window"] == 60) & (df["spec"] == "TOTAL")]
print(row[["coef_post_x_treas", "se_post_x_treas", "t_post_x_treas", "n"]].to_string())
# Expected: coef=3.836, se=0.889, t=4.318, n=2057
```

```python
# ── Check 2: Driscoll-Kraay robustness — exit event, W=60, TOTAL ──────────
# Column name is "event" (not "event_date"); values are "entry"/"exit" strings
# DK column names: "se_dk_post_x_treas", "t_dk_post_x_treas"
df2 = pd.read_csv("audit_repairs/outputs/dk_robustness.csv")
row2 = df2[(df2["event"] == "exit") & (df2["window"] == 60) & (df2["spec"] == "TOTAL")]
print(row2[["t_dk_post_x_treas", "se_dk_post_x_treas"]].to_string())
# Expected: t_dk_post_x_treas=5.618, se_dk_post_x_treas=0.683
```

```python
# ── Check 3: Clean DiD — TOTAL spec ───────────────────────────────────────
# This CSV uses "spec" directly; no event/window filter needed
df3 = pd.read_csv("audit_repairs/outputs/did_clean_baseline.csv")
row3 = df3[df3["spec"] == "TOTAL"]
print(row3[["coef_relief_x_treas", "t_relief_x_treas",
            "coef_post_relief_x_treas", "t_post_relief_x_treas"]].to_string())
# Expected: coef_relief_x_treas=-10.763, t_relief_x_treas=-9.366
#           coef_post_relief_x_treas=-9.026, t_post_relief_x_treas=-7.789
```
