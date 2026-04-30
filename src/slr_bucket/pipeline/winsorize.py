"""
winsorize.py — Percentile winsorization for the SLR event study panel.

COVID-spike outliers (EUR CIP: 150.9 bps on 2020-03-16; JPY: 12 obs >150 bps)
have outsized leverage on HAC covariance. Winsorizing at p1/p99 within each
series removes these high-leverage points without discarding the stress episode.

y_bps and y_abs_bps are clipped; controls are untouched.
Clipping counts are logged per series to a CSV audit file.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PCT_LO = 1
PCT_HI = 99


def winsorize_series(
    s: pd.Series, pct_lo: float, pct_hi: float
) -> tuple[pd.Series, float, float, int, int]:
    """
    Clip a series at the given percentiles.
    Returns (clipped, lo_bound, hi_bound, n_clipped_lo, n_clipped_hi).
    """
    lo = s.quantile(pct_lo / 100.0)
    hi = s.quantile(pct_hi / 100.0)
    n_lo = int((s < lo).sum())
    n_hi = int((s > hi).sum())
    return s.clip(lower=lo, upper=hi), float(lo), float(hi), n_lo, n_hi


def winsorize_panel(
    panel: pd.DataFrame,
    pct_lo: float = PCT_LO,
    pct_hi: float = PCT_HI,
    by_series: bool = True,
    log_path: Path | None = None,
) -> pd.DataFrame:
    """
    Winsorize y_bps and y_abs_bps at pct_lo / pct_hi.

    Applied after loading, before merging controls.
    Records clipping counts per series; saves log to log_path if provided.

    Parameters
    ----------
    panel     : DataFrame with columns [series_id, y_bps, y_abs_bps]
    pct_lo    : lower percentile bound (default 1)
    pct_hi    : upper percentile bound (default 99)
    by_series : compute bounds separately per series_id if True
    log_path  : where to write the clipping log CSV (None = no file)
    """
    panel    = panel.copy()
    log_rows = []

    groups = (
        [(sid, grp) for sid, grp in panel.groupby("series_id")]
        if by_series
        else [("ALL", panel)]
    )

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
                "series_id":             sid,
                "col":                   col,
                f"p{int(pct_lo)}_bound": round(lo, 4),
                f"p{int(pct_hi)}_bound": round(hi, 4),
                "n_clipped_lo":          n_lo,
                "n_clipped_hi":          n_hi,
                "n_total":               len(s),
                "pct_clipped":           round((n_lo + n_hi) / len(s) * 100, 2),
            })

    log_df = pd.DataFrame(log_rows)
    clipped_any = log_df[(log_df["n_clipped_lo"] > 0) | (log_df["n_clipped_hi"] > 0)]

    if not clipped_any.empty:
        print(f"Winsorization clips at p{int(pct_lo)}/p{int(pct_hi)}:")
        for _, row in clipped_any.iterrows():
            print(
                f"  {row['series_id']:25s} {row['col']:12s}: "
                f"lo={row['n_clipped_lo']:3d}, hi={row['n_clipped_hi']:3d}  "
                f"bounds=[{row[f'p{int(pct_lo)}_bound']:.1f}, "
                f"{row[f'p{int(pct_hi)}_bound']:.1f}] bps"
            )
    else:
        print("Winsorization: no observations clipped.")

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(log_path, index=False)
        print(f"Winsorization log saved: {log_path}")

    return panel
