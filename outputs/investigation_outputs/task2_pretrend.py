import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, 'audit_repairs')
out = Path('investigation_outputs')
out.mkdir(exist_ok=True)

from repaired_pipeline import load_full_panel, add_event_time_all_series, TOTAL_CONTROLS, HAC_LAGS

panel = load_full_panel()

# Build W=60 exit event window
work = add_event_time_all_series(panel, '2021-03-31')
work = work[work['event_time'].between(-60, 60)].copy()

print(f"Panel shape: {work.shape}, series: {work['series_id'].nunique()}, N obs: {len(work)}")

# ── Define 10-day bins ──
bins = [
    (-60, -51, 'bin_m60'),
    (-50, -41, 'bin_m50'),
    (-40, -31, 'bin_m40'),
    (-30, -21, 'bin_m30'),
    (-20, -11, 'bin_m20'),
    (-10, -1,  'bin_m10'),
    (0,   9,   'bin_p0'),
    (10, 19,   'bin_p10'),
    (20, 29,   'bin_p20'),
    (30, 39,   'bin_p30'),
    (40, 49,   'bin_p40'),
    (50, 60,   'bin_p50'),
]

for lo, hi, name in bins:
    work[name] = ((work['event_time'] >= lo) & (work['event_time'] <= hi)).astype(int)

# Interaction: bin x TreasuryBased
bin_names = [b[2] for b in bins]
inter_names = [f"inter_{b}" for b in bin_names]
for b, ib in zip(bin_names, inter_names):
    work[ib] = work[b] * work['treasury_based']

# ── Build design matrix ──
avail_ctrl = [c for c in TOTAL_CONTROLS if c in work.columns]
reg_cols = ['y_abs_bps', 'treasury_based', 'series_id'] + inter_names + avail_ctrl
reg = work[reg_cols].copy()
for c in ['y_abs_bps', 'treasury_based'] + inter_names + avail_ctrl:
    reg[c] = pd.to_numeric(reg[c], errors='coerce')
reg = reg.dropna()

# Series FE
fe = pd.get_dummies(reg['series_id'].astype(str), prefix='fe', drop_first=True).astype(float)
X_cols = inter_names + avail_ctrl
X = pd.concat([reg[X_cols].reset_index(drop=True), fe.reset_index(drop=True)], axis=1).astype(float)
y = reg['y_abs_bps'].reset_index(drop=True).astype(float)

# Add constant
X = sm.add_constant(X)

print(f"\nDesign matrix: {X.shape[1]} params, {len(y)} obs")

# Fit with HAC SE
res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

# ── Extract bin interaction coefficients ──
print("\n=== DYNAMIC DiD COEFFICIENTS ===")
print(f"{'Bin':15s} {'Coef':>8s} {'SE':>8s} {'t':>8s} {'Stars':>6s}")
print("-" * 50)

results_rows = []
pre_coefs = []
pre_ses = []
for b, ib in zip(bin_names, inter_names):
    if ib in res.params.index:
        coef = float(res.params[ib])
        se = float(res.bse[ib])
        t = float(res.tvalues[ib])
        pv = float(res.pvalues[ib])
        stars = '***' if pv < 0.01 else '**' if pv < 0.05 else '*' if pv < 0.10 else ''
        lo_val = [x[0] for x in bins if x[2] == b][0]
        hi_val = [x[1] for x in bins if x[2] == b][0]
        label = f"[{lo_val:+d},{hi_val:+d}]"
        print(f"{label:15s} {coef:+8.3f} {se:8.3f} {t:+8.3f} {stars:>6s}")
        is_pre = lo_val < 0
        results_rows.append({'bin': label, 'coef': coef, 'se': se, 't': t, 'pvalue': pv, 'stars': stars, 'is_pre': is_pre})
        if is_pre:
            pre_coefs.append(coef)
            pre_ses.append(se)
    else:
        print(f"{ib}: NOT FOUND IN RESULTS")

# ── F-test of pre-event leads ──
pre_inter = [ib for b, ib in zip(bin_names, inter_names) if b in ['bin_m60','bin_m50','bin_m40','bin_m30','bin_m20','bin_m10']]
pre_in_params = [p for p in pre_inter if p in res.params.index]
print(f"\nPre-event bins in model: {pre_in_params}")

f_stat, f_pval = np.nan, np.nan
if len(pre_in_params) >= 2:
    # Manual Wald test
    R = np.zeros((len(pre_in_params), len(res.params)))
    for i, p in enumerate(pre_in_params):
        idx = list(res.params.index).index(p)
        R[i, idx] = 1
    b_vec = res.params.values
    V = res.cov_params().values
    Rb = R @ b_vec
    RVR = R @ V @ R.T
    try:
        W_stat = float(Rb @ np.linalg.solve(RVR, Rb)) / len(pre_in_params)
        f_stat = W_stat
        f_pval = float(1 - stats.f.cdf(W_stat, len(pre_in_params), len(y) - X.shape[1]))
        print(f"\nWald F-stat: {f_stat:.4f}, p = {f_pval:.4f}")
    except Exception as e2:
        print(f"Manual Wald error: {e2}")
else:
    print("Not enough pre-event bins for F-test")

# ── Save results ──
df_res = pd.DataFrame(results_rows)
df_res.to_csv(out / 'pretrend_regression_results.csv', index=False)
print(f"\nSaved: pretrend_regression_results.csv")
print(f"F-test: F={f_stat:.4f}, p={f_pval:.4f}")

# ── Plot ──
fig, ax = plt.subplots(figsize=(9, 4))

def bin_mid(label):
    lo, hi = label.strip('[]').split(',')
    return (int(lo) + int(hi)) / 2

x_all = [bin_mid(r['bin']) for _, r in df_res.iterrows()]
coefs = df_res['coef'].values
ses = df_res['se'].values

pre_mask = df_res['is_pre'].values
colors = ['gray' if p else '#1a3a6b' for p in pre_mask]

for i, (x, c, col) in enumerate(zip(x_all, coefs, colors)):
    ax.errorbar([x], [c], yerr=[[1.96*ses[i]],[1.96*ses[i]]],
                fmt='o', color=col, ecolor=col, elinewidth=1.2, capsize=3, ms=5)

ax.axvline(0, color='black', ls='--', lw=1)
ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
ax.set_xlabel('Event Time (trading days, exit = 0)')
ax.set_ylabel('Interaction Coef (bps)')
fstat_str = f"{f_stat:.2f}" if not np.isnan(f_stat) else "N/A"
fpval_str = f"{f_pval:.3f}" if not np.isnan(f_pval) else "N/A"
ax.set_title(f'Dynamic DiD: Post x Treasury by Bin -- Exit Event W=60 TOTAL\n(F-test pre-leads: F={fstat_str}, p={fpval_str})')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(out / 'pretrend_event_study.png', dpi=150)
print("Saved: pretrend_event_study.png")
plt.close()

# ── Print LaTeX table rows ──
print("\n=== LaTeX table rows for tab:pretrend ===")
for _, row in df_res.iterrows():
    stars_str = f'^{{{row["stars"]}}}' if row['stars'] else ''
    print(f"${row['bin']}$ & ${row['coef']:+.2f}{stars_str}$ & ${row['se']:.2f}$ & ${row['t']:+.2f}$ \\\\")

print("\n=== TASK 2 COMPLETE ===")
