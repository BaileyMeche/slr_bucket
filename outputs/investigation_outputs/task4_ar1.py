import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

sys.path.insert(0, 'audit_repairs')
out = Path('investigation_outputs')
out.mkdir(exist_ok=True)

from repaired_pipeline import load_full_panel

panel = load_full_panel()

print(f"{'Series':25s} {'AR(1) coef':>12s} {'SE':>8s} {'t':>8s} {'N':>6s} {'Flag':>10s}")
print("-" * 75)

rows = []
for sid, g in panel.groupby('series_id'):
    s = g.sort_values('date')['y_abs_bps'].dropna()
    if len(s) < 10:
        continue
    y = s.values[1:]
    x = s.values[:-1]
    X = sm.add_constant(x)
    try:
        res = sm.OLS(y, X).fit()
        ar1 = float(res.params[1])
        se = float(res.bse[1])
        t = float(res.tvalues[1])
        flag = 'LOW AR1' if ar1 < 0.90 else ''
        print(f"{sid:25s} {ar1:12.4f} {se:8.4f} {t:8.2f} {len(y):6d} {flag:>10s}")
        rows.append({'series_id': sid, 'ar1_coef': ar1, 'se': se, 't': t, 'n': len(y), 'flag': flag})
    except Exception as e:
        print(f"{sid:25s} ERROR: {e}")

df = pd.DataFrame(rows)
df.to_csv(out / 'ar1_coefficients.csv', index=False)
print(f"\nSaved: ar1_coefficients.csv")
print(f"\nSummary: {(df['ar1_coef'] > 0.93).sum()} of {len(df)} series have AR(1) > 0.93")
print(f"         {(df['ar1_coef'] < 0.90).sum()} series have AR(1) < 0.90 (below claim)")
print(f"\nMin AR(1): {df['ar1_coef'].min():.4f} ({df.loc[df['ar1_coef'].idxmin(),'series_id']})")
print(f"Max AR(1): {df['ar1_coef'].max():.4f} ({df.loc[df['ar1_coef'].idxmax(),'series_id']})")
print("\n=== TASK 4 COMPLETE ===")
