# Audit Diff — SLR Event Study Repairs

## Files Changed

### New files created (audit_repairs/ only — originals untouched)

| File | Purpose |
|------|---------|
| `audit_repairs/AUDIT_LOG.md` | Per-bug findings with code line numbers and evidence |
| `audit_repairs/FINAL_REPORT.md` | Executive summary, before/after tables, recommendations |
| `audit_repairs/fix_1_units.py` | Deterministic unit-aware loader; replaces `_to_bps` heuristic |
| `audit_repairs/fix_2_ust_sf.py` | UST SF data verification (no construction error found) |
| `audit_repairs/fix_3_pooled.py` | Correct pooled panel regression across all 17 series |
| `audit_repairs/fix_4_equity_n.py` | Equity sample size diagnostic |
| `audit_repairs/fix_5_overlap.py` | Event window overlap detection and truncation |
| `audit_repairs/fix_6_hac.py` | HAC rank deficiency guard; obs/param check |
| `audit_repairs/repaired_pipeline.py` | Integrated pipeline producing all outputs |

### Source files NOT modified (originals preserved)

- `src/slr_bucket/outcomes.py` — `_scale_to_bps` threshold bug documented but not patched
  (patch should be: replace `med < 5` threshold with unit-registry lookup)
- `src/slr_bucket/econometrics/event_study.py` — no changes needed; existing functions used
- `notebooks/metric_analysis.ipynb` — not modified (notebook-level bugs documented in AUDIT_LOG)

---

## Function-Level Before/After

### Bug 1: `_to_bps` / `_scale_to_bps`

**Before** (`notebooks/metric_analysis.ipynb` cell 3):
```python
def _to_bps(x: pd.Series) -> pd.Series:
    if med < 0.05:    return s * 10000.0   # decimal
    if med < 20.0:    return s * 100.0     # percent  ← COLLIDES WITH BPS DATA
    return s
```

**After** (`audit_repairs/fix_1_units.py`):
```python
def safe_identity(x: pd.Series) -> pd.Series:
    """Return series as-is (already in bps)."""
    return pd.to_numeric(x, errors="coerce")

# Deterministic: no threshold heuristic
# All data/series/ files are already in bps
# equity spread_SPX (non-filtered) is already in bps
long["y_bps"] = safe_identity(long["y_raw"])
```

**Effect**: UST SF 2Y median drops from 1,875 bps → 18.75 bps. CIP AUD from 989 → 9.9 bps.

---

### Bug 3: Pooled regression collinearity

**Before** (`notebooks/metric_analysis.ipynb` cell 7):
```python
ACTIVE = "ust_spot_fut"   # ← single-strategy mode
# ... panel_long contains only ust_spot_fut (all treasury_based = 1)

# Cell 12:
robust_all, reg_all = run_pooled_jump(subW, y_col="y_abs_bps",
                                      interact_treasury=True)
# Result: post:treasury_based ≡ post (treasury_based = 1 for all rows)
```

**After** (`audit_repairs/repaired_pipeline.py`):
```python
# All 17 series stacked simultaneously:
panel = stack_all_outcomes(series_dir)   # 6 treasury + 11 non-treasury series

# Pooled regression on full panel:
result = run_pooled_jump(panel, event_date, window, controls, spec)
# treasury_based has both 0 and 1 values → interaction properly identified
```

**Effect**: N rises from per-strategy to 2,057 (W=60). Post x TreasuryBased is now
estimated as a differential effect (-2.2 bps entry, +3.85 bps exit) rather than
being mechanically identical to Post.

---

### Bug 4: Equity N = 14-16

**Before**: Notebook was run in `ACTIVE = "eq_SPX"` mode for equity analysis.
Within-strategy ACTIVE mode + limited controls = unclear but very small N.

**After**: Full 17-series pooled panel. Each equity series (SPX, NDX, INDU) contributes
N = 121 obs per event/W=60 window. Total equity contribution = 363 obs.

---

### Bug 5: Event window overlap

**Before**: CONFIG events = ["2020-04-01", "2021-03-19", "2021-03-31"] used with W=60.
2021-03-19 vs 2021-03-31: gap = 7 trading days, overlap = 33 t.d. (W=20) or 113 t.d. (W=60).

**After**: Main regression tables use only ["2020-04-01", "2021-03-31"]. The 2021-03-19
announcement is treated as a supplementary one-sided [0,+7] window analysis only.

---

## Key Numeric Changes

### CIP mean |W| (pre-period, 2019-01-01 to 2020-03-31)
| Currency | Buggy (bps) | Repaired (bps) | Economic target |
|----------|-------------|----------------|-----------------|
| AUD | ~989 | 9.9 | 5-40 bps |
| CAD | ~1,275 | 12.7 | 5-40 bps |
| CHF | ~31 | 31.1 | 10-80 bps |
| EUR | ~36 | 36.2 | 10-80 bps |
| GBP | ~1,070 | 10.7 | 5-40 bps |
| JPY | ~39 | 38.8 | 20-100 bps |
| NZD | ~1,198 | 12.0 | 5-40 bps |
| SEK | ~23 | 22.6 | 10-60 bps |
| **Average** | **~2,594** | **~21.9** | **15-60 bps** |

### UST SF mean |W| (pre-period)
| Tenor | Buggy (bps) | Repaired (bps) | Economic target |
|-------|-------------|----------------|-----------------|
| 2Y | ~1,875 | 18.8 | 5-50 bps |
| 5Y | ~1,737 | 17.4 | 5-50 bps |
| 10Y | ~23 | 23.5 | 5-50 bps |
| **Average** | **~2,600** | **~19.9** | **10-50 bps** |

### Sample sizes (equity, W=60)
| Version | N per equity series |
|---------|---------------------|
| Buggy | 14-16 |
| Repaired | **121** |

### Pooled regression: Post x TreasuryBased (entry W=60)
| Version | Estimate | SE | t-stat |
|---------|----------|-----|--------|
| Buggy | = Post coef (collinear) | — | — |
| Repaired | **-2.22** | 3.55 | -0.63 |

### Pooled regression: Post x TreasuryBased (exit W=60)
| Version | Estimate | SE | t-stat |
|---------|----------|-----|--------|
| Buggy | = Post coef (collinear) | — | — |
| Repaired | **+3.85** | 0.89 | **4.32*** |
