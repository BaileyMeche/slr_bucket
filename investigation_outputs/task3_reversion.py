import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, 'audit_repairs')
out = Path('investigation_outputs')
out.mkdir(exist_ok=True)

from repaired_pipeline import load_full_panel, TOTAL_CONTROLS, HAC_LAGS

panel = load_full_panel()

# ── Replicate clean DiD specification ──
pre = panel[panel['date'].between('2019-01-01','2020-01-31')].copy()
relief = panel[panel['date'].between('2020-04-01','2021-03-31')].copy()
post = panel[panel['date'].between('2021-04-01','2021-12-31')].copy()

pre['relief'] = 0; pre['post_relief'] = 0
relief['relief'] = 1; relief['post_relief'] = 0
post['relief'] = 0; post['post_relief'] = 1

sample = pd.concat([pre, relief, post], ignore_index=True)
sample['relief_x_treas'] = sample['relief'] * sample['treasury_based']
sample['post_relief_x_treas'] = sample['post_relief'] * sample['treasury_based']

avail_ctrl = [c for c in TOTAL_CONTROLS if c in sample.columns]
reg_cols = ['y_abs_bps','relief','post_relief','treasury_based','relief_x_treas','post_relief_x_treas','series_id'] + avail_ctrl
reg = sample[reg_cols].dropna().copy()
for c in reg_cols:
    if c != 'series_id':
        reg[c] = pd.to_numeric(reg[c], errors='coerce')
reg = reg.dropna()

fe = pd.get_dummies(reg['series_id'].astype(str), prefix='fe', drop_first=True).astype(float)
X_cols = ['relief','post_relief','treasury_based','relief_x_treas','post_relief_x_treas'] + avail_ctrl
X = pd.concat([reg[X_cols].reset_index(drop=True), fe.reset_index(drop=True)], axis=1).astype(float)
X = sm.add_constant(X)
y = reg['y_abs_bps'].reset_index(drop=True).astype(float)

print(f"Clean DiD sample: N={len(y)}, params={X.shape[1]}")

res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

mu2 = float(res.params['relief_x_treas'])
mu4 = float(res.params['post_relief_x_treas'])
se_mu2 = float(res.bse['relief_x_treas'])
se_mu4 = float(res.bse['post_relief_x_treas'])
t_mu2 = float(res.tvalues['relief_x_treas'])
t_mu4 = float(res.tvalues['post_relief_x_treas'])
cov_mu2_mu4 = float(res.cov_params().loc['relief_x_treas','post_relief_x_treas'])

print(f"\nmu2 (relief x treas):      {mu2:.4f} (SE={se_mu2:.4f}, t={t_mu2:.3f})")
print(f"mu4 (post_relief x treas): {mu4:.4f} (SE={se_mu4:.4f}, t={t_mu4:.3f})")
print(f"Cov(mu2, mu4) = {cov_mu2_mu4:.6f}")

# ── F-test H0: mu2 = mu4 ──
diff = mu2 - mu4
var_diff = se_mu2**2 + se_mu4**2 - 2*cov_mu2_mu4
se_diff = np.sqrt(max(var_diff, 1e-12))
t_diff = diff / se_diff
df_resid = len(y) - X.shape[1]
p_value = 2 * (1 - stats.t.cdf(abs(t_diff), df=df_resid))
f_stat = t_diff**2
f_pval = 1 - stats.f.cdf(f_stat, 1, df_resid)

ci_lo = diff - 1.96 * se_diff
ci_hi = diff + 1.96 * se_diff

pct_reversed = abs(diff) / abs(mu2) * 100 if mu2 != 0 else np.nan

print(f"\n=== F-test: H0: mu2 = mu4 ===")
print(f"mu2 - mu4 = {diff:.4f} bps")
print(f"SE(mu2 - mu4) = {se_diff:.4f}")
print(f"t-stat = {t_diff:.4f}")
print(f"F-stat = {f_stat:.4f} (1, {df_resid}) df")
print(f"p-value = {f_pval:.4f}")
print(f"95% CI for mu2-mu4: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"% of relief compression reversed: {pct_reversed:.1f}%")

interp = "REJECT H0 (p<0.05)" if f_pval < 0.05 else "FAIL TO REJECT H0 (p>0.05)"
print(f"Interpretation: {interp}")

results_txt = f"""=== Partial Reversion Test: H0: mu2 = mu4 ===
Clean DiD TOTAL specification, N={len(y)}

mu2 (relief x treas):      {mu2:.4f} bps  (SE={se_mu2:.4f}, t={t_mu2:.3f})
mu4 (post_relief x treas): {mu4:.4f} bps  (SE={se_mu4:.4f}, t={t_mu4:.3f})
Cov(mu2, mu4):             {cov_mu2_mu4:.6f}

Difference (mu2 - mu4):    {diff:.4f} bps
SE(difference):            {se_diff:.4f}
t-statistic:               {t_diff:.4f}
F-statistic (1,{df_resid}) df: {f_stat:.4f}
p-value:                   {f_pval:.4f}

95% CI for (mu2-mu4):      [{ci_lo:.4f}, {ci_hi:.4f}]
% of compression reversed: {pct_reversed:.1f}%
Interpretation:            {interp}
"""
with open(out / 'reversion_test_results.txt', 'w') as f:
    f.write(results_txt)
print("\nSaved: reversion_test_results.txt")
print("\n=== TASK 3 COMPLETE ===")
