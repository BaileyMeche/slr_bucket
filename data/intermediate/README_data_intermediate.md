## README `data/intermediate/*` (pipeline-assembled panels/caches)

These are produced after the raw inputs above are merged into analysis-ready panels. These “intermediate” files are best understood as artifacts of that notebook’s assembly stage.

### 1) `spreads.csv`

**What it is**

* Outcome series for the project: daily TIPS–Treasury wedge by tenor.
* V8 defines the wedge as ((y^{TIPS}*{t,\tau} + \pi^{swap}*{t,\tau}) - y^{Nom}_{t,\tau}) in **bps** .

**How we used it**

* This is the **dependent variable** (W_{t,\tau}) feeding:

  * Layer 1 event studies and jump estimates
  * Layer 2 weekly mechanism changes

---

### 2) `sofr_repo.csv`

**What it is**

* Cleaned daily secured-funding control slice used for DIRECT:

  * includes `SOFR` and repo bases (e.g., `spr_tgcr`, `spr_effr`) as used in DIRECT .

**How we used it**

* Merged into the daily wedge panel (by `date`) to form DIRECT control set in Layer 1.

---

### 3) `analysis_panel.csv`

**What it is**

* Master merged dataset at the unit of observation used by the pipeline (likely `date × tenor`):

  * wedge outcome
  * control sets (VIX/HY_OAS/BAA10Y + issuance + repo controls)
  * event-time variables / indicators
  * mechanism proxies (bank exposure, dealer utilization) aligned to weekly when needed

**How we used it**

* This is the main “design matrix” table used to run the regressions described in your V8 design:

  * event-time regressions with controls 
  * mechanism regressions with `Relief_x_bank` and `Relief_x_util` described in the V8 outputs 

---

### 4) `crsp_treasury_agg.csv`

**What it is**

* Aggregated CRSP Treasury data (likely yields, outstanding, or identifiers) used for auxiliary construction or validation.

**How we used it**

* Not referenced in the V8 control manifest excerpt you provided (V8 uses issuance from FiscalData, not CRSP issuance) .
* Most likely an **optional/legacy** intermediate for alternate issuance measures or nominal curve diagnostics.

---

### 5) `fed_assets_proxy.csv`

**What it is**

* Daily/weekly proxy for Fed balance sheet size or holdings (e.g., SOMA / H.4.1 aggregates).

**How we used it**

* Not part of the V8 control manifest in the report excerpt , so it is **not used in the core V8 regressions** as currently summarized.
* Likely created for exploratory robustness (QE / Fed footprint controls).

---

### 6) `purchase_proxy_holdings.csv`

**What it is**

* Proxy for Fed purchase flows or holdings (SOMA holdings-based measure).

**How we used it**

* Same status as `fed_assets_proxy.csv`: not in the V8 control manifest snippet .
* Likely an **optional** mechanism/control you experimented with for policy channel separation.

--- 

# Origins
### What created `slr_episode/intermediate/`

The **V8 run notebook** is the origin of the SLR episode pipeline artifacts: `slr_main_pipeline_v8.ipynb` is explicitly identified as the run notebook in your V8 report. 
The `intermediate/` directory is the **staging layer** where that notebook writes “analysis-ready” panels built from raw sources before producing the packaged `_output/event_outputs_v8/` tables.

What I can verify from the V8 report about the raw inputs used in that staging layer:

* **Outcome construction input:** `_data/tips_treasury_implied_rf_2010.csv` 
* **Swap leg (manual):** `data_manual/treasury_inflation_swaps.csv` 
* **Controls used in V8:** `VIX`, `HY_OAS`, `BAA10Y`, issuance bins (`issu_7_bil`, `issu_14_bil`, `issu_30_bil`), and for DIRECT also `SOFR`, `spr_tgcr`, `spr_effr`. 
* **Mechanism inputs in V8:** (i) aggregate bank “exemption-eligible share” proxy and (ii) dealer utilization proxy built from NY Fed Primary Dealer series; the chosen utilization pair is `NYPD-PD_RP_TIPS_TOT-A | NYPD-PD_RRP_TIPS_TOT-A`. 

Below is the dependency trace for each `intermediate/` file, mapping **which raw files feed it**.

---

## `slr_episode/intermediate/spreads.csv`

**What it is**

* The **daily dependent variable panel**: the TIPS–Treasury arbitrage wedge by tenor.

**How it was built**

* Constructed from the “three legs” definition used in V8:
  \(W_{t,\tau} = (y^{TIPS}*{t,\tau} + \pi^{swap}*{t,\tau}) - y^{Nom}_{t,\tau}\), reported in **bps**. 
* Uses the repo’s canonical daily file and the manual swap file.

**Raw sources used**

* `_data/tips_treasury_implied_rf_2010.csv` (daily TIPS + nominal legs) 
* `data_manual/treasury_inflation_swaps.csv` (inflation swap fixed leg) 

**Created by**

* `slr_main_pipeline_v8.ipynb` (staging step before event studies) 

---

## `slr_episode/intermediate/sofr_repo.csv`

**What it is**

* A **daily funding-controls slice** used for the DIRECT specification (secured/near-secured funding and bases).

**How it was built / used**

* DIRECT controls listed in the V8 manifest include `SOFR`, `spr_tgcr`, `spr_effr`. 
* This file is the “cleaned + aligned” version merged into the daily panel for Layer 1 and weekly-averaged for Layer 2.

**Raw sources used (from your event_inputs layer)**

* `_output/event_inputs/repo_rates_combined.(csv|parquet)` **or** fallback `_output/event_inputs/repo_rates_fred.(csv|parquet)`
  (the V8 control manifest confirms these series are present and used in-window). 

**Created by**

* `slr_main_pipeline_v8.ipynb` 

---

## `slr_episode/intermediate/analysis_panel.csv`

**What it is**

* The **master merged analysis dataset** (the design matrix) used to run:

  * Layer 1 event studies (TOTAL + DIRECT),
  * and to assemble the weekly mechanism panel.

**How it was built / used**

* Merge spine is (at minimum) `date × tenor` using `spreads.csv`.
* Adds control sets exactly as recorded in V8:

  * Risk/credit controls: `VIX`, `HY_OAS`, `BAA10Y` 
  * Issuance bins: `issu_7_bil`, `issu_14_bil`, `issu_30_bil` 
  * Funding controls for DIRECT: `SOFR`, `spr_tgcr`, `spr_effr` 
* Adds mechanism-layer proxies:

  * Bank exposure proxy (aggregate exemption-eligible share; predetermined/lagged) 
  * Dealer utilization proxy from primary dealer series (TIPS repo/reverse repo pair) 

**Raw sources used**

* From `_data/` + `data_manual/` via `spreads.csv` (see above) 
* From `_output/event_inputs/`:

  * `controls_vix_creditspreads_fred.(csv|parquet)` (provides `VIX`, `HY_OAS`, `BAA10Y`) 
  * `treasury_issuance_by_tenor_fiscaldata.(csv|parquet)` (transformed into issuance-bin controls used in V8) 
  * `repo_rates_combined.(csv|parquet)` → `sofr_repo.csv` 
  * `bank_exposure_y9c_agg_daily.csv` (the “bank exemption-eligible share proxy”) 
  * `primary_dealer_stats_ofr_stfm_nypd_long.(csv|parquet)` (utilization proxy inputs) 

**Created by**

* `slr_main_pipeline_v8.ipynb` 

---

## `slr_episode/intermediate/crsp_treasury_agg.csv`

**What it is**

* An **aggregated CRSP Treasury** auxiliary table (likely a supply/outstanding or mapping object).

**What I can and cannot verify**

* It is **not referenced in the V8 controls manifest** (V8 controls are the ones listed above). 
* Therefore, in the **core V8 specification** it is either:

  1. unused (legacy artifact), or
  2. used in a side branch not reflected in `controls_manifest_v8.csv`.

**Likely raw sources (cannot verify from uploaded files here)**

* WRDS CRSP Treasuries tables (e.g., `crsp_a_treasuries.*`) or a locally cached CRSP export.

**Created by**

* Almost certainly `slr_main_pipeline_v8.ipynb`, but I cannot confirm the exact upstream raw filename without the notebook/script source.

---

## `slr_episode/intermediate/fed_assets_proxy.csv`, & `slr_episode/intermediate/purchase_proxy_holdings.csv`

**What they are**

* Fed balance-sheet / purchase footprint proxies (used as optional macro/liquidity controls in many versions of the design).

**What I can and cannot verify**

* They are **not included** in the V8 control manifest table (so they are not part of the “official” V8 TOTAL/DIRECT control sets). 
* That strongly suggests they are:

  * optional robustness controls, or
  * constructed for exploration and left in `intermediate/`.

**Likely raw sources (cannot verify from uploaded files here)**

* FRED series (e.g., Fed total assets / Treasury holdings) **or** H.4.1/SOMA holdings downloads.

**Created by**

* Almost certainly `slr_main_pipeline_v8.ipynb` (or a helper script it calls), but exact provenance requires the code.
