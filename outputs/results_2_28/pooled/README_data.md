Below is a complete **README.md** written in Markdown that documents the mathematical construction of all pooled outputs in your pipeline.

---

# README — Construction of Summary Statistics and Regression Outputs

This document provides a **technical description** of the construction of all output tables and regression files produced by the pooled multi-strategy analysis pipeline.

This document defines all variables formally and describes the exact mathematical operations used to construct each output file.

This is a construction-level reference (no interpretation of quantitative magnitudes).

---

# 1. Core Data Objects

## 1.1 Series Indexing

Let:

* ( i ) index a **series**
  (e.g., TIPS–Treasury 2y wedge, CIP AUD 3m basis, Equity spot–futures INDU, etc.)

* ( t ) index **daily dates**

Each series carries metadata:

* `strategy` ( = g(i) \in { \text{tips_treas}, \text{ust_spot_fut}, \text{cip}, \text{eq_spot_fut} } )
* `tenor` ( = \tau(i) ) (e.g., 2, 5, 10, 3m, index)
* `treasury_based` ( = TB_i \in {0,1} )

---

## 1.2 Spread Construction

### Raw spread

$$
y^{raw}_{i,t}
$$

### Conversion to basis points

$$
y^{bps}*{i,t} = \text{to\_bps}(y^{raw}*{i,t})
$$

(The function `to_bps()` multiplies by 10,000 if units appear to be in percent/log units.)

### Strategy sign normalization

$$
W_{i,t} = s(i)\cdot y^{bps}_{i,t}, \quad s(i)\in{-1,+1}
$$

This ensures a consistent orientation across strategies.

### Absolute dislocation magnitude

$$
|W|*{i,t} = |W*{i,t}|
$$

Many regressions use \( |W|_{i,t} \) as the dependent variable.

---

## 1.3 Regime Definitions

Dates are partitioned into regimes:

* **Pre**
  $$
  t \in [2019\text{-}01\text{-}01;2020\text{-}03\text{-}31]
  $$

* **Relief**
  $$
  t \in [2020\text{-}04\text{-}01;2021\text{-}03\text{-}31]
  $$

* **Post**
  $$
  t \in [2021\text{-}04\text{-}01;2021\text{-}12\text{-}31]
  $$

Let \( \mathcal T \_r \) denote the set of dates in regime \( r \).

---

## 1.4 Event Time

For event date \( t_0 \), define trading-day event time:


$$k_{i,t}(t_0) \in \mathbb{Z}
$$

where:

* \( k = 0 \) is the event date
* \( k < 0 \) are trading days before
* \( k > 0 \) are trading days after

Define post indicator:

$$
Post_{i,t}(t_0) = \mathbf{1}{k_{i,t}(t_0) \ge 0}
$$

---

# 2. Output Files

---

# 2.1 `summary_stats.csv` and `table1_levels_nearzero.csv`

Grouped by:

$$
(strategy=g,; tenor=\tau,; regime=r)
$$

Define stacked sample:

$$
\mathcal S_{g,\tau,r}
=
{(i,t) : g(i)=g, \ \tau(i)=\tau, \ t\in\mathcal T_r, \ W_{i,t}\text{ observed}}
$$

Let:

$$
n_{g,\tau,r} = |\mathcal S_{g,\tau,r}|
$$

### Columns

* `N_days`
  $$
  n_{g,\tau,r}
  $$

* `mean_W`
  $$
  \frac{1}{n_{g,\tau,r}} \sum_{(i,t)\in\mathcal S_{g,\tau,r}} W_{i,t}
  $$

* `median_W`
  $$
  \text{median}{W_{i,t}}
  $$

* `p5_W`, `p95_W`
  $$
  Q_{0.05}(W),\quad Q_{0.95}(W)
  $$

* `mean_absW`
  $$
  \frac{1}{n_{g,\tau,r}} \sum |W_{i,t}|
  $$

* `share_absW_le_d` for ( d \in {5,10} )
  $$
  \frac{1}{n_{g,\tau,r}} \sum \mathbf{1}{|W_{i,t}|\le d}
  $$

This equals the empirical CDF evaluated at threshold ( d ).

---

# 2.2 `table4_nearzero_relief.csv`

Restricted to relief regime only.

For ( r = \text{relief} ):

$$
\mathcal S_{g,\tau,\text{relief}}
$$

Columns:

* `N_days`
  $$
  |\mathcal S_{g,\tau,\text{relief}}|
  $$

* `mean_absW`
  $$
  \frac{1}{N}\sum |W_{i,t}|
  $$

* `median_absW`
  $$
  \text{median}{|W_{i,t}|}
  $$

* `share_absW_le_d`
  $$
  \frac{1}{N}\sum \mathbf{1}{|W_{i,t}|\le d}
  $$

---

# 2.3 `jump_by_strategy.csv`

Event-window jump regressions by strategy.

For event ( t_0 ), window ( W ):

$$
\mathcal S(g,t_0,W) = 
{(i,t) : g(i)=g,; |k_{i,t}(t_0)|\le W}
$$

Regression:

$$
|W|_{i,t}
=
\alpha_i
+
\beta, Post_{i,t}(t_0)
+
\Gamma'X_t^{spec}
+
\varepsilon_{i,t}
$$

* `coef_post` = ( \widehat{\beta} )
* `se_post` = HAC SE
* `N` = number of stacked observations
* `N_dates` = number of distinct calendar dates in sample

---

# 2.4 `table2_key_coefs.csv`

Pooled event-window regression across strategies.

$$
|W|_{i,t}=
\alpha_i
+
\beta, Post_{i,t}
+
\delta, (Post_{i,t} \times TB_i)
+
\Gamma'X_t^{spec}
+
\varepsilon_{i,t}
$$

Columns:

* `coef`

  * if `term=post` → ( \widehat{\beta} )
  * if `term=post:treasury_based` → ( \widehat{\delta} )

* `se` = HAC SE

* `n_obs` = stacked observations used

---

# 2.5 `eventstudy_paths_by_strategy.csv`

Binned event study by strategy.

Define bins ( B_j = [a_j,b_j] ).

Indicator:

$$
D^{(j)}*{i,t} = \mathbf{1}{k*{i,t}(t_0)\in B_j}
$$

Regression:

$$
|W|_{i,t}=
\alpha_i
+
\sum_{j\ne ref} \beta_j D^{(j)}*{i,t}
+
\Gamma'X_t^{spec}
+
\varepsilon*{i,t}
$$

Columns:

* `estimate` = ( \widehat{\beta}_j )
* `se`
* `ci_low` = ( \widehat{\beta}_j - 1.96,se )
* `ci_high` = ( \widehat{\beta}_j + 1.96,se )
* `bin_mid` = ( \frac{a_j + b_j}{2} )

---

# 2.6 `table3_relief_regression.csv`

Relief-period panel regression.

Define group-level daily average:

$$
Y_{g,t} = \frac{1}{|\mathcal I_g(t)|} \sum_{i:TB_i=g} |W|_{i,t}
$$

Define:

$$
Relief_t = \mathbf{1}{t \in \text{relief period}}
$$

Regression:

$$
Y_{g,t}=
\alpha_g
+
\beta, Relief_t
+
\delta, (Relief_t \times g)
+
\Gamma'X_t^{spec}
+
u_{g,t}
$$

---

# 2.7 `layer2_weekly_changes_by_strategy.csv`

Weekly differenced mechanism regression.

Weekly mean:

$$
\bar{Y}_{i,w}=
\text{mean}*{t\in w} |W|*{i,t}
$$

Weekly change:

$$
\Delta \bar{Y}_{i,w}
= \bar{Y}_{i,w} - 
\bar{Y}_{i,w-1}
$$

Regression:

$$
\Delta \bar{Y}_{i,w}
=
a
+
b_1 Relief_w
+
b_2 (Relief_w z^{exempt}_w)
+
b_3 (Relief_w z^{util}*w)
+
\Theta' \Delta X_w
+
e*{i,w}
$$

---

# 2.8 `layer2_weekly_changes_pooled.csv`

Adds Treasury interaction terms:

$$
\Delta \bar{Y}_{i,w}
=
a
+
b_1 Relief_w
+
b_2 Relief_w z^{exempt}_w
+
b_3 Relief_w z^{util}_w
+
c_1 Relief_w TB_i
+
c_2 Relief_w z^{exempt}_w TB_i
+
c_3 Relief_w z^{util}*w TB_i
+
\Theta' \Delta X_w
+
e*{i,w}
$$