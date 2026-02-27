from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .io import load_any_table, as_daily_date


@dataclass(frozen=True)
class OutcomeSpec:
    name: str
    strategy: str
    source: str  # dataset name (without extension) resolvable under data/series
    loader: str  # loader key


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # common date columns
    for cand in ["date", "Date", "DATE", "observation_date", "timestamp"]:
        if cand in out.columns:
            out = out.rename(columns={cand: "date"})
            break
    if "date" not in out.columns:
        raise KeyError(f"Outcome loader expected a date column. Columns={list(out.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["date"] = as_daily_date(out["date"])
    return out


def _scale_to_bps(x: pd.Series) -> pd.Series:
    """Heuristic: if typical magnitude looks like % units, convert to bps."""
    s = pd.to_numeric(x, errors="coerce")
    med = float(s.abs().median()) if s.notna().any() else np.nan
    # If median abs is < 5, treat as percent units (e.g., 0.40 == 40 bps)
    if np.isfinite(med) and med < 5:
        return s * 100.0
    return s


def load_tips_treasury_arb(path: Path, pattern: str = "arb_") -> pd.DataFrame:
    df = _ensure_date(load_any_table(path))
    cols = [c for c in df.columns if str(c).startswith(pattern)]
    if not cols:
        raise ValueError(f"No columns starting with '{pattern}' in {path.name}")
    long = df[["date", *cols]].melt(id_vars=["date"], var_name="series", value_name="y_raw")
    long["tenor"] = long["series"].str.extract(r"(\d+)").astype(float)
    long["y_bps"] = _scale_to_bps(long["y_raw"])
    long["strategy"] = "TIPS_Treasury"
    long["treasury_based"] = 1
    long["series"] = long["series"].astype(str)
    return long[["date","strategy","series","tenor","y_bps","treasury_based"]].dropna(subset=["date","y_bps"])


def load_treasury_spot_futures(path: Path) -> pd.DataFrame:
    df = _ensure_date(load_any_table(path))
    # columns like Treasury_SF_2Y, Treasury_SF_10Y...
    cols = [c for c in df.columns if str(c).lower().startswith("treasury_sf_") or str(c).lower().startswith("treasury_sf")]
    if not cols:
        # allow Treasury_SF_2Y etc but with different prefix
        cols = [c for c in df.columns if "SF" in str(c) and "Treasury" in str(c)]
    if not cols:
        raise ValueError(f"No Treasury spot-futures columns found in {path.name}")
    long = df[["date", *cols]].melt(id_vars=["date"], var_name="series", value_name="y_raw")
    long["tenor"] = long["series"].str.extract(r"(\d+)Y").astype(float)
    long["y_bps"] = _scale_to_bps(long["y_raw"])
    long["strategy"] = "Treasury_SpotFutures"
    long["treasury_based"] = 1
    long["series"] = long["series"].astype(str)
    return long[["date","strategy","series","tenor","y_bps","treasury_based"]].dropna(subset=["date","y_bps"])


def load_cip_basis(path: Path, tenor_years: float = 0.25) -> pd.DataFrame:
    df = _ensure_date(load_any_table(path))
    cols = [c for c in df.columns if str(c).startswith("CIP_")]
    if not cols:
        raise ValueError(f"No CIP_* columns found in {path.name}")
    long = df[["date", *cols]].melt(id_vars=["date"], var_name="series", value_name="y_raw")
    long["series"] = long["series"].astype(str)
    # normalize series name
    long["series"] = long["series"].str.replace("_ln", "", regex=False)
    long["tenor"] = tenor_years
    long["y_bps"] = _scale_to_bps(long["y_raw"])
    long["strategy"] = "CIP"
    long["treasury_based"] = 0
    return long[["date","strategy","series","tenor","y_bps","treasury_based"]].dropna(subset=["date","y_bps"])


def load_equity_spot_futures(path: Path, index_code: str) -> pd.DataFrame:
    df = _ensure_date(load_any_table(path))
    # prefer filtered if present
    cand1 = f"spread_{index_code}_filtered"
    cand2 = f"spread_{index_code}"
    if cand1 in df.columns:
        spread = df[cand1]
    elif cand2 in df.columns:
        spread = df[cand2]
    else:
        # fallback: find first column starting with spread_
        cols = [c for c in df.columns if str(c).lower().startswith("spread_")]
        if not cols:
            raise ValueError(f"No spread column found in {path.name}")
        spread = df[cols[0]]
        cand2 = cols[0]
    y_bps = _scale_to_bps(spread)
    out = pd.DataFrame({
        "date": df["date"],
        "strategy": "Equity_SpotFutures",
        "series": f"EQ_SF_{index_code}",
        "tenor": np.nan,
        "y_bps": y_bps,
        "treasury_based": 0,
    })
    return out.dropna(subset=["date","y_bps"])


def stack_outcomes(series_dir: Path) -> pd.DataFrame:
    """Load all outcomes needed for the multi-strategy pipeline from data/series."""
    parts: list[pd.DataFrame] = []

    # TIPS-Treasury (arb_*): parquet
    parts.append(load_tips_treasury_arb(series_dir / "tips_treasury_implied_rf_2010.parquet"))

    # Treasury spot-futures
    parts.append(load_treasury_spot_futures(series_dir / "treasury_sf_output.csv"))

    # CIP basis (3m)
    parts.append(load_cip_basis(series_dir / "cip_spreads_3m_bps.csv", tenor_years=0.25))

    # Equity spot-futures (indices)
    for idx in ["SPX", "NDX", "INDU"]:
        p = series_dir / f"equity_spot_spread_{idx}.csv"
        if p.exists():
            parts.append(load_equity_spot_futures(p, idx))

    out = pd.concat(parts, ignore_index=True)
    # add magnitude column used for baseline hypothesis (dislocation size)
    out["y_abs_bps"] = out["y_bps"].abs()
    return out
