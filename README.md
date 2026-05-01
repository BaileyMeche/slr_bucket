# Balance-Sheet Capacity and Cross-Strategy Treasury Arbitrage: SLR Exclusion Event Study

This repository implements the full empirical pipeline for the SLR exclusion
event study. The analysis tests whether Treasury-based arbitrage spreads
compressed more than non-Treasury spreads during the Fed's 2020–2021
Supplementary Leverage Ratio (SLR) exclusion window.

## Abstract
Regulatory leverage constraints are a leading candidate explanation forlimited intermediation in U.S. Treasury markets. In this paper, I exploit the Federal Reserve's temporary exclusion of U.S. Treasury securities and Federal Reserve deposits from the Supplementary Leverage Ratio (SLR) denominator (April 1, 2020 to March 31, 2021) to test whether SLR relief differentially compressed Treasury-based arbitrage spreads relative to non-Treasury controls. Using a pooled difference-in-differences design across 17 daily arbitrage spread series in four strategy classes (Treasury spot-futures, TIPS-Treasury, covered interest parity, and equity spot-futures), I find that during the relief window Treasury spreads were compressed by 10.76 basis points more than non-Treasury spreads relative to the 2019 pre-COVID baseline, and  re-widened  at expiry by 3.84 basis points in the subsequent 60 trading days. Non-Treasury series show no comparable break at the SLR event dates. This behavior is consistent with balance-sheet segmentation rather than a general intermediary funding shock and yields an arbitrage-spread-based estimate of the leverage ratio friction in Treasury markets.

---

## Quick start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd slr_bucket

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate slr

# 3. Run the full pipeline (analysis + figures + tests)
doit                   # Windows cmd / PowerShell
./doit                 # Git Bash
python -m doit         # any shell with env activated
```

The first run takes ~5–10 minutes (136 HAC regressions, 6 publication figures).
Subsequent runs are instant unless source files changed (`doit` tracks fingerprints).

---

## Repository layout

```
slr_bucket/
├── src/
│   └── slr_bucket/
│       ├── config.py                    # PipelineConfig dataclass
│       ├── io.py                        # Data catalog, path resolution
│       ├── load.py                      # Raw data loaders
│       ├── outcomes.py                  # Outcome variable construction
│       ├── validation.py                # Schema validation helpers
│       ├── runner.py                    # Run-directory setup utilities
│       ├── econometrics/
│       │   └── event_study.py           # Core event-study estimators
│       ├── pipeline/                    # Main analysis pipeline
│       │   ├── main.py                  # Entry point — run this
│       │   ├── loader.py                # Unit-aware series loader
│       │   ├── windows.py               # Event window overlap detection
│       │   ├── hac.py                   # HAC regression guard
│       │   ├── winsorize.py             # Percentile winsorization
│       │   ├── issuance.py              # Rolling Treasury issuance controls
│       │   ├── did.py                   # Panel DiD (2019 baseline)
│       │   └── results.py               # Result formatting and LaTeX tables
│       └── figures/
│           └── make_figures.py          # All 6 publication figures
│
├── data/
│   ├── raw/                             # Original source data (read-only)
│   │   ├── cip_bloomberg.xlsx
│   │   ├── OIS.xlsx
│   │   ├── equity_spot_bloomberg.parquet
│   │   ├── treasury_spot_futures.xlsx
│   │   └── event_inputs/
│   │       ├── controls_vix_creditspreads_fred.csv   # VIX, HY_OAS, BAA10Y
│   │       ├── repo_rates_combined.csv               # SOFR, TGCR, BGCR
│   │       └── treasury_issuance_by_tenor_fiscaldata.csv
│   └── series/                          # Pre-processed series, all in bps
│       ├── cip_spreads_3m_bps.csv
│       ├── treasury_sf_output.csv
│       ├── tips_treasury_implied_rf_2010.parquet
│       ├── equity_spot_spread_SPX.csv
│       ├── equity_spot_spread_NDX.csv
│       └── equity_spot_spread_INDU.csv
│
├── docs/
│   ├── AUDIT_LOG.md                     # Chronological bug discovery record
│   ├── FINAL_REPORT.md                  # Final audit summary and results
│   └── figures.tex                      # \input{}-ready figure environments
│
├── notebooks/
│   └── metric_analysis.ipynb            # Interactive exploration notebook
│
├── presentations/
│   ├── draft.tex                        # Full thesis draft
│   └── references.bib
│
├── tests/
│   ├── test_event_study.py
│   └── test_io.py
│
├── _output/                             # GENERATED — created by doit, deleted by doit clean
│   ├── pipeline/                        # All regression outputs (CSV, TEX, TXT)
│   └── figures/                         # All publication figures (PDF, PNG)
│
├── environment.yml                      # Conda environment spec (create with: conda env create -f environment.yml)
├── dodo.py                              # doit task automation
├── doit                                 # Git Bash wrapper (./doit)
└── doit.bat                             # Windows cmd/PowerShell wrapper (doit)
```

The `_output/` directory is not tracked in version control — it is created on
the first run and deleted by `doit clean`. All source code lives in `src/`.

---

## Environment setup

### Step 1 — Install conda

If you do not have Anaconda or Miniconda installed, download from:
- Miniconda (minimal): <https://docs.conda.io/en/latest/miniconda.html>
- Anaconda (full): <https://www.anaconda.com/download>

### Step 2 — Create the project environment

```bash
conda env create -f environment.yml
```

This creates a conda environment named **`slr_bucket`** with all required
packages. Run this once from the project root directory.

### Step 3 — Activate the environment

```bash
conda activate slr_bucket
```

You must activate the environment in every new shell session before running
`doit` or any pipeline script.

### Updating the environment

If `environment.yml` changes (e.g., new packages added):

```bash
conda env update -f environment.yml --prune
```

### Removing the environment

```bash
conda env remove -n slr_bucket
```

---

## Running the pipeline

### With doit (recommended)

`doit` tracks file fingerprints and skips tasks whose inputs have not changed.

**Windows cmd / PowerShell** (run from the project root):

```
doit              # pipeline + figures + tests
doit pipeline     # analysis pipeline only
doit figures      # publication figures only
doit tests        # pytest suite only
doit clean        # delete all generated outputs (_output/)
doit list         # show all available tasks
```

**Git Bash (MINGW64)**:

```bash
./doit
./doit pipeline
./doit clean
```

**Any shell with the conda env activated**:

```bash
python -m doit [task]
```

After the first full run, doit re-executes a task only if an input changed:

| Task | Reruns when… |
|------|--------------|
| `pipeline` | Any file in `src/slr_bucket/pipeline/` or raw data changes |
| `figures` | `make_figures.py` or `jump_estimates_*.csv` changes |
| `tests` | Any file in `tests/` or `src/` changes |

#### Conda environment override

`doit` and the wrappers auto-detect the conda environment by searching for
`slr_bucket` (then `risk` for backward compatibility) in common install paths.
To override explicitly:

```
set DOIT_CONDA_ENV=my_env    # Windows cmd
$env:DOIT_CONDA_ENV="my_env" # PowerShell
export DOIT_CONDA_ENV=my_env # bash/zsh
doit
```

### Manual invocation

```bash
conda activate slr_bucket
cd slr_bucket

# Run the analysis pipeline
python src/slr_bucket/pipeline/main.py

# Generate publication figures (requires pipeline outputs)
python src/slr_bucket/figures/make_figures.py
```

---

## Software requirements

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| Python | 3.12 | All pipeline scripts |
| pandas | 2.2+ | Data manipulation |
| numpy | 2.0+ | Numerical operations |
| statsmodels | 0.14+ | HAC (Newey-West) SE |
| matplotlib | 3.9+ | All figures |
| pyarrow | 16.0+ | Reading .parquet files |
| openpyxl | 3.1+ | Reading .xlsx raw data |
| scipy | 1.13+ | Statistical utilities |
| doit | 0.36+ | Task automation |
| pytest | 9.0+ | Test suite |

All packages are specified in `environment.yml` and installed automatically by
`conda env create`.

---

## Pipeline steps

`src/slr_bucket/pipeline/main.py` runs the full analysis in sequence:

| Step | Action | Output |
|------|--------|--------|
| 1 | Event-window overlap check | Drops 2021-03-19 from main tables |
| 2 | Load panel (17 series, 2019–2021) | Winsorized at p1/p99 per series |
| 3 | Summary statistics | `_output/pipeline/summary_stats_repaired.csv` |
| 4 | Pooled DiD regressions | `_output/pipeline/jump_estimates_pooled.csv` |
| 5 | Series-level regressions | `_output/pipeline/jump_estimates_by_series.csv` |
| 6 | LaTeX regression table | `_output/pipeline/regression_table_repaired.tex` |
| 7 | Diagnostic plots | `_output/pipeline/diagnostic_plots/` |
| 8 | Panel DiD (2019 baseline) | `_output/pipeline/did_clean_baseline.csv` |
| 9 | Formatted result summary | `_output/pipeline/key_results_formatted.txt` |

The pipeline also writes `regression_table_v2.tex` and `robustness_table.csv`
in step 9 via `results.py`.

---

## Module descriptions

### `src/slr_bucket/pipeline/`

| Module | Description |
|--------|-------------|
| `main.py` | Pipeline entry point. Orchestrates all steps. |
| `loader.py` | Loads all four strategy types with explicit unit handling. All `data/series/` files are already in basis points — no heuristic scaling is applied. |
| `windows.py` | Event window overlap detection. The two 2021 exit events (Mar 19 and Mar 31) overlap at W=20 and W=60; the announcement date is excluded from main tables. |
| `hac.py` | Pre-flight design-matrix checks and safe OLS+HAC estimator. Guards against rank deficiency and low obs/param ratios. |
| `winsorize.py` | Clips `y_bps` and `y_abs_bps` at p1/p99 within each series before merging controls. COVID-spike outliers (EUR CIP: 150.9 bps, JPY CIP: 245 bps) are the primary targets. |
| `issuance.py` | Builds rolling 7/14/30-day Treasury issuance totals from bi-weekly auction data. Lagged 1 day for point-in-time discipline. |
| `did.py` | Full-panel DiD using 2019 as the clean pre-COVID reference. Excludes Feb–Mar 2020 (COVID crash). Key estimates: `relief_x_treas` (entry DiD) and `post_relief_x_treas` (exit DiD). |
| `results.py` | Generates `key_results_formatted.txt`, `robustness_table.csv`, and `regression_table_v2.tex`. |

### `src/slr_bucket/figures/`

| Module | Description |
|--------|-------------|
| `make_figures.py` | Generates all 6 publication figures in PDF + PNG. Reads panel data from `data/series/` and regression estimates from `_output/pipeline/`. |

---

## Key results

**Exit event (2021-03-31, W=60, TOTAL controls):**

```
Post x TreasuryBased = +3.85*** bps  (SE=0.89, t=4.3, N=2,057)
```

Interpretation: After the SLR exclusion expired, Treasury-based arbitrage
spreads widened by ~3.9 bps more than non-Treasury spreads — consistent with
balance-sheet constraints re-emerging as the exclusion was withdrawn.

**Robustness:** Stable across W=20 and W=60, and across TOTAL vs. DIRECT
control specifications (shift < 0.05 bps), confirming the balance-sheet
channel rather than the funding-cost channel.

**Entry event caveat:** The W=60 pre-period for entry (2020-04-01) spans
Jan–Mar 2020 — the COVID crash peak. UST SF was dislocated 100–300 bps.
The clean DiD using 2019 as the pre-period resolves this:
`relief_x_treas` = negative (compression during exclusion, significant).

---

## Verification

After running the pipeline, verify the main result:

```python
import pandas as pd

df = pd.read_csv("_output/pipeline/jump_estimates_pooled.csv")
row = df[(df["event"] == "2021-03-31") & (df["window"] == 60) & (df["spec"] == "TOTAL")]
print(row[["coef_post_x_treas", "se_post_x_treas", "t_post_x_treas", "n"]].to_string())
# Expected: coef=3.836, se=0.889, t=4.318, n=2057
```

```python
df3 = pd.read_csv("_output/pipeline/did_clean_baseline.csv")
row3 = df3[df3["spec"] == "TOTAL"]
print(row3[["coef_relief_x_treas", "t_relief_x_treas",
            "coef_post_relief_x_treas", "t_post_relief_x_treas"]].to_string())
# Expected: coef_relief_x_treas=-10.763, t_relief_x_treas=-9.366
#           coef_post_relief_x_treas=-9.026, t_post_relief_x_treas=-7.789
```

---

## Table-to-file map

| Table | LaTeX label | Source file |
|-------|-------------|-------------|
| Pooled jump regressions | `tab:pooled_jump` | `_output/pipeline/jump_estimates_pooled.csv` |
| Series-level estimates | `tab:series_level` | `_output/pipeline/jump_estimates_by_series.csv` |
| Summary statistics | `tab:summary_stats` | `_output/pipeline/summary_stats_repaired.csv` |
| Robustness (TOTAL vs DIRECT) | `tab:robustness` | `_output/pipeline/robustness_table.csv` |
| Winsorization log | `tab:winsorize` | `_output/pipeline/winsorize_log.csv` |
| Clean DiD | `tab:did_clean` | `_output/pipeline/did_clean_baseline.csv` |

---

## Data provenance

| Series | Raw source | Processed file |
|--------|------------|---------------|
| CIP | `data/raw/cip_bloomberg.xlsx` | `data/series/cip_spreads_3m_bps.csv` |
| UST spot-futures | `data/raw/treasury_spot_futures.xlsx` | `data/series/treasury_sf_output.csv` |
| TIPS-Treasury | `data/raw/OIS.xlsx` + TIPS data | `data/series/tips_treasury_implied_rf_2010.parquet` |
| Equity SF | `data/raw/equity_spot_bloomberg.parquet` | `data/series/equity_spot_spread_*.csv` |
| Controls | FRED (VIX, HY_OAS, BAA10Y) | `data/raw/event_inputs/controls_vix_creditspreads_fred.csv` |
| Repo controls | NY Fed (SOFR, TGCR, BGCR) | `data/raw/event_inputs/repo_rates_combined.csv` |
| Issuance | TreasuryDirect | Built by `src/slr_bucket/pipeline/issuance.py` |

All `data/series/` files are in basis points. The loader applies no scaling.

---

## Key design choices

1. **Winsorization** — p1/p99 within each series, 2019–2021 in-sample only.
   Log saved to `_output/pipeline/winsorize_log.csv`.

2. **HAC lags** — 5 (`HAC_LAGS = 5` in `main.py`). AR(1) > 0.93 for 15/17
   series. Results robust to lags 3–10 (`robustness_table.csv`).

3. **Event window** — 2021-03-19 dropped from main tables (W=20 window
   overlaps with 2021-03-31 exit by 33 trading days).

4. **Clean DiD exclusion** — 2020-02-01 through 2020-03-31 excluded as the
   COVID crash period. Adjustable in `src/slr_bucket/pipeline/did.py`.

5. **Series classification** — `treasury_based = 1` for UST spot-futures and
   TIPS-Treasury (6 series); `treasury_based = 0` for CIP and equity SF (11 series).
   Hard-coded in `src/slr_bucket/pipeline/loader.py`.

---

## Troubleshooting

### "slr_bucket / risk env not found" warning at doit startup

`dodo.py` searches for the conda env in common Anaconda/Miniconda install
directories. If your conda is installed elsewhere, either:

```bash
# Option A: activate the env first, then run python -m doit
conda activate slr_bucket
python -m doit

# Option B: set the env name override
export DOIT_CONDA_ENV=slr_bucket   # bash
doit
```

### doit re-runs the pipeline on every invocation

This happens when a listed target file is missing (e.g., `_output/` was
partially cleaned). Run `doit clean` to wipe all outputs, then `doit` to
rebuild from scratch.

### Stale `.doit.db` after copying the project

If you copied the project from another location, the `.doit.db` fingerprint
database may contain stale paths. Delete it and let doit rebuild:

```bash
del .doit.db          # Windows cmd
rm .doit.db           # bash / Git Bash
```

Then run `doit` again.
