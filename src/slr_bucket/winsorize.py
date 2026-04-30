"""
Fix 7: Winsorize CIP and UST SF COVID-spike outliers.

Problem: EUR CIP hit 150.9 bps on 2020-03-16; JPY has 12 obs >150 bps;
CHF has 2 obs >150 bps. These high-leverage points distort HAC covariance
and pull entry-period means upward, making pre-period baselines appear elevated.

Solution: Winsorize y_bps and y_abs_bps at p=1 / p=99 WITHIN each series_id,
using only the 2019-2021 in-sample data. Controls are untouched.

Expected effects (pre-winsorize vs post):
  CIP_EUR  99th pct: 150.9 bps -> ~103 bps
  CIP_JPY  99th pct: ~245 bps  -> ~180 bps
  All other series: p99 well below 100 bps, few/no clips

Clipping counts are logged to _outputs/fix7_winsorize_log.csv.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent

PCT_LO = 1
PCT_HI = 99


def winsorize_series(s: pd.Series, pct_lo: float, pct_hi: float) -> tuple[pd.Series, float, float, int, int]:
    """
    Clip a series at the given percentiles.
    Returns (clipped_series, lo_bound, hi_bound, n_clipped_lo, n_clipped_hi).
    """
    lo = s.quantile(pct_lo / 100.0)
    hi = s.quantile(pct_hi / 100.0)
    n_lo = int((s < lo).sum())
    n_hi = int((s > hi).sum())
    clipped = s.clip(lower=lo, upper=hi)
    return clipped, float(lo), float(hi), n_lo, n_hi


def winsorize_panel(
    panel: pd.DataFrame,
    pct_lo: float = PCT_LO,
    pct_hi: float = PCT_HI,
    by_series: bool = True,
    log_path: Path | None = None,
) -> pd.DataFrame:
    """
    Winsorize y_bps and y_abs_bps at pct_lo / pct_hi within each series_id.

    Applied AFTER loading, BEFORE merging controls.
    Records clipping counts per series; saves log to log_path if provided.

    Parameters
    ----------
    panel : DataFrame with columns [series_id, y_bps, y_abs_bps]
    pct_lo, pct_hi : percentile bounds (default 1 / 99)
    by_series : if True, compute bounds separately per series_id
    log_path : where to write the clipping log CSV (None = no file)

    Returns
    -------
    panel with y_bps and y_abs_bps winsorized in-place (copy returned)
    """
    panel = panel.copy()
    log_rows = []

    if by_series:
        groups = [(sid, grp) for sid, grp in panel.groupby("series_id")]
    else:
        groups = [("ALL", panel)]

    for sid, grp in groups:
        idx = grp.index

        for col in ["y_bps", "y_abs_bps"]:
            if col not in panel.columns:
                continue
            s = pd.to_numeric(panel.loc[idx, col], errors="coerce").dropna()
            if s.empty:
                continue

            clipped, lo, hi, n_lo, n_hi = winsorize_series(s, pct_lo, pct_hi)
            panel.loc[s.index, col] = clipped

            log_rows.append({
                "series_id": sid,
                "col": col,
                f"p{int(pct_lo)}_bound": round(lo, 4),
                f"p{int(pct_hi)}_bound": round(hi, 4),
                "n_clipped_lo": n_lo,
                "n_clipped_hi": n_hi,
                "n_total": len(s),
                "pct_clipped": round((n_lo + n_hi) / len(s) * 100, 2) if len(s) > 0 else 0,
            })

    log_df = pd.DataFrame(log_rows)

    # Print summary for series with any clipping
    clipped_any = log_df[(log_df["n_clipped_lo"] > 0) | (log_df["n_clipped_hi"] > 0)]
    if not clipped_any.empty:
        print(f"\n[winsorize] Observations clipped at p{int(pct_lo)}/p{int(pct_hi)}:")
        for _, row in clipped_any.iterrows():
            print(f"  {row['series_id']:25s} {row['col']:12s}: "
                  f"lo_clips={row['n_clipped_lo']:3d}, hi_clips={row['n_clipped_hi']:3d}  "
                  f"bounds=[{row[f'p{int(pct_lo)}_bound']:.1f}, {row[f'p{int(pct_hi)}_bound']:.1f}] bps")
    else:
        print("[winsorize] No observations clipped.")

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(log_path, index=False)
        print(f"[winsorize] Log saved: {log_path}")

    return panel


def show_pre_winsorize_extremes(panel: pd.DataFrame) -> None:
    """Print per-series 99th percentile and max for context."""
    print("\n=== Pre-winsorize extremes (y_abs_bps) ===")
    print(f"{'series_id':25s} {'p99':>8s} {'max':>8s} {'n_gt_100':>10s} {'n_gt_150':>10s}")
    print("-" * 65)
    for sid, grp in panel.groupby("series_id"):
        s = pd.to_numeric(grp["y_abs_bps"], errors="coerce").dropna()
        if s.empty:
            continue
        p99 = s.quantile(0.99)
        smax = s.max()
        n_gt100 = int((s > 100).sum())
        n_gt150 = int((s > 150).sum())
        flag = " <-- spike" if n_gt150 > 0 else ""
        print(f"{sid:25s} {p99:8.1f} {smax:8.1f} {n_gt100:10d} {n_gt150:10d}{flag}")


if __name__ == "__main__":
    from slr_bucket.load import stack_all_outcomes

    SERIES_DIR = REPO_ROOT / "data" / "series"
    OUT_DIR = REPO_ROOT / "_outputs"
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Winsorize COVID-spike outliers")
    print("=" * 70)

    panel = stack_all_outcomes(SERIES_DIR)
    panel = panel[panel["date"].between("2019-01-01", "2021-12-31")].copy()

    print(f"\nPanel loaded: {len(panel):,} rows, {panel['series_id'].nunique()} series")
    show_pre_winsorize_extremes(panel)

    panel_w = winsorize_panel(
        panel,
        pct_lo=1,
        pct_hi=99,
        by_series=True,
        log_path=OUT_DIR / "fix7_winsorize_log.csv",
    )

    print("\n=== Post-winsorize extremes (y_abs_bps) ===")
    show_pre_winsorize_extremes(panel_w)
