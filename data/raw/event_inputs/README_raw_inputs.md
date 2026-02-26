## README: `data/raw/event_inputs/*` (raw pulls / inputs)

### 1) `bank_exposure_proxy_call_wrds.csv` (+ `.parquet`)

**What it is**

* **Bank-level (panel)** Call Report / Y-9C-style proxy pulled from WRDS “banksamp/bank_all/bank schedule” tables (WRDS Bank product).
* Frequency: **quarterly** (report dates).
* Core fields (as used in your outputs/logs):

  * `bank_id` (RSSD ID)
  * `report_date`
  * `assets` (total assets proxy)
  * `reserves` (reserve balances proxy)
  * `ust` (Treasury holdings proxy where available)
  * derived shares: `reserves_share = reserves/assets`, `ust_share = ust/assets` (where non-null), and “exempt-eligible” proxies.

**How we used it**

* This is a **fallback / auxiliary** balance-sheet tightness measure when you don’t have (or don’t want to rely on) the FR Y-9C parse.
* Conceptually: approximates the **share of assets that would become SLR-exempt under the relief (Treasuries + reserves)**, but WRDS schedule mapping is incomplete (your logs show very low UST coverage).


---

### 2) `bank_exposure_agg_call_wrds.csv` (+ `.parquet`)

**What it is**

* **Aggregate (system-wide)** version of the WRDS Call proxy above.
* Frequency: **quarterly**.
* Typical fields:

  * `report_date`
  * `total_assets`, `total_reserves`, `total_ust`
  * `agg_reserves_share`, `agg_ust_share`
  * `agg_exempt_share` (or proxy): ((\text{total reserves} + \text{total UST}) / \text{total assets})

**How we used it**

* Provides the **time-series** \(z(\text{bank exposure}_t)\) used in the mechanism interaction \(Relief_t \times z(\text{bank exposure}_t)\) when you are operating with WRDS call-report proxies rather than Y-9C.

---

### 3) `bank_exposure_y9c_bank_quarterly.csv`

**What it is**

* **Bank-level** FR Y-9C extracted from your manually downloaded BHCF files (2020–2021 quarters).
* Frequency: **quarterly** (as-reported).

**How we used it**

* Source-of-truth to construct “exempt-eligible exposure” for each bank:

  * Treasuries and reserve balances (and any other excluded items you configured).
* Serves two purposes:

  1. bank-level inspection / validation (who carries the exemption-eligible balance sheet?)
  2. aggregation to system-wide series used in mechanism.

---

### 4) `bank_exposure_y9c_agg_quarterly.csv`

**What it is**

* **Aggregate** quarterly system-wide Y-9C balance-sheet series.
* Frequency: **quarterly** (report dates).
* Fields: aggregate totals and shares (assets, reserves, Treasuries, and exempt-eligible share).

**How we used it**

* Primary aggregate “bank capacity / exemption exposure” series at the quarterly frequency.

---

### 5) `bank_exposure_y9c_agg_daily.csv`

**What it is**

* **Daily** version of the Y-9C aggregate exposure, created by mapping quarterly points to daily dates (typically forward-fill).
* Frequency: **daily**.
* This is the cleanest “bank exposure proxy” you can merge into daily wedge panels.

**How we used it**

* This is the **V8 mechanism bank proxy** used in the interaction:

  * The V8 report describes the mechanism regression using “aggregate exemption-eligible share proxy” .
* In the V8 results, that proxy is treated as **predetermined/lagged** in the mechanism layer (you do this to mitigate reverse causality).

---

### 6) `controls_vix_creditspreads_fred.csv` (+ `.parquet`)

**What it is**

* Daily macro/risk controls (FRED sourced):

  * `VIX`
  * `HY_OAS`
  * `BAA10Y` (Moody’s Baa minus 10y Treasury, depending on your series choice; your file shows `BAA10Y`).

**How we used it**

* Included in **TOTAL** and **DIRECT** control sets in V8 .
* These controls enter:

  * Layer 1 event-study regressions (X_t)
  * Layer 2 weekly mechanism regressions (weekly-averaged).

---

### 7) `repo_rates_combined.csv` (+ `.parquet`)

**What it is**

* Daily secured-funding / reference-rate panel:

  * at minimum `SOFR`, plus derived bases used in DIRECT: `spr_tgcr`, `spr_effr` (as described in the V8 control manifest) .
* In your earlier debugging, TGCR/BGCR sometimes fail from FRED; combined file is your “best available” merged output.

**How we used it**

* Direct channel controls for **DIRECT** specification in Layer 1:

  * the V8 report explicitly lists `SOFR`, `spr_tgcr`, `spr_effr` in DIRECT .
* Used to diagnose whether event effects attenuate once funding conditions are controlled for.

---

### 8) `repo_rates_fred.csv` (+ `.parquet`)

**What it is**

* Earlier FRED-only repo/reference attempt (typically SOFR/EFFR/IORB, and sometimes TGCR/BGCR if available).

**How we used it**

* **Fallback** if combined/OFR-based pull fails.

---

### 9) `primary_dealer_stats_ofr_stfm_nypd_long.csv` (+ `.parquet`)

**What it is**

* Weekly Primary Dealer Statistics series via OFR STFM “nypd” dataset.
* Long format:

  * `date` (weekly)
  * `mnemonic` (e.g., `NYPD-PD_RP_TIPS_TOT-A`)
  * `series_name`
  * `value`

**How we used it**

* Mechanism-layer “dealer utilization/capacity” proxy:

  * V8 report states the dealer utilization proxy is constructed from NY Fed Primary Dealer series and shows the chosen utilization pair (`NYPD-PD_RP_TIPS_TOT-A|NYPD-PD_RRP_TIPS_TOT-A`) .
* You use this to build a **lagged utilization index** entering (Relief_t \times z(util_{t-1})).

---

### 10) `treasury_issuance_by_tenor_fiscaldata.csv` (+ `.parquet`)

**What it is**

* Treasury issuance aggregated from FiscalData.
* Likely fields:

  * `issue_date`
  * `tenor_bucket`
  * `issuance_amount`
  * `n_issues`

**How we used it**

* Builds issuance controls that appear in V8 as `issu_7_bil`, `issu_14_bil`, `issu_30_bil` (the V8 report lists issuance controls in both TOTAL and DIRECT) .

---
