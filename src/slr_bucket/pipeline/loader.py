"""
loader.py — Unit-aware series loader for the SLR event study panel.

All data/series/ files are pre-processed to basis points. This module loads
each strategy type with explicit unit handling, avoiding heuristic scaling.

CIP, UST SF, and TIPS series are returned as-is (already in bps).
Equity series use the non-filtered spread column (spread_SPX, etc.)
which is already expressed in basis points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ── Unit registry ──────────────────────────────────────────────────────────────
# "bps": already in basis points, return as-is
# "pct": multiply by 100 to get bps
# "decimal": multiply by 10000 to get bps
SERIES_UNITS: dict[str, str] = {
    "cip_spreads_3m_bps":            "bps",
    "tips_treasury_implied_rf_2010":  "bps",
    "treasury_sf_output":             "bps",
}

EQUITY_SPREAD_COL_PREFERENCE = {
    "SPX":  "spread_SPX",   # median ~43.6 bps
    "NDX":  "spread_NDX",   # median ~40.1 bps
    "INDU": "spread_INDU",  # median ~47.0 bps
}


def safe_identity(x: pd.Series) -> pd.Series:
    """Return series as-is (already in bps)."""
    return pd.to_numeric(x, errors="coerce")


def scale_pct_to_bps(x: pd.Series) -> pd.Series:
    """Multiply by 100: percent → bps."""
    return pd.to_numeric(x, errors="coerce") * 100.0


def scale_decimal_to_bps(x: pd.Series) -> pd.Series:
    """Multiply by 10000: decimal fraction → bps."""
    return pd.to_numeric(x, errors="coerce") * 10000.0


UNIT_CONVERTERS = {
    "bps":     safe_identity,
    "pct":     scale_pct_to_bps,
    "decimal": scale_decimal_to_bps,
}


def load_cip_bps(path: Path) -> pd.DataFrame:
    """Load CIP 3-month spreads. Values are already in basis points."""
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    cip_cols = [
        "CIP_AUD_ln", "CIP_CAD_ln", "CIP_CHF_ln", "CIP_EUR_ln",
        "CIP_GBP_ln", "CIP_JPY_ln", "CIP_NZD_ln", "CIP_SEK_ln",
    ]
    available = [c for c in cip_cols if c in df.columns]
    assert available, f"No CIP_* columns in {path.name}"

    long = df[["date", *available]].melt("date", var_name="raw_name", value_name="y_raw")
    long["currency"] = long["raw_name"].str.replace("CIP_", "").str.replace("_ln", "")
    long["series_id"]      = "cip_" + long["currency"].str.lower()
    long["strategy"]       = "cip"
    long["tenor"]          = 0.25
    long["treasury_based"] = 0
    long["y_bps"]          = safe_identity(long["y_raw"])
    long["y_abs_bps"]      = long["y_bps"].abs()

    diag = long.groupby("series_id")["y_bps"].agg(["count", lambda x: x.abs().median()])
    diag.columns = ["n", "median_abs_bps"]
    print("CIP series medians (bps):")
    print(diag.to_string())

    return long[["date", "strategy", "series_id", "tenor", "treasury_based",
                 "y_bps", "y_abs_bps"]].dropna(subset=["date", "y_bps"])


def load_ust_sf_bps(path: Path, tenors: list[int] | None = None) -> pd.DataFrame:
    """
    Load UST spot-futures basis. Values are already in basis points.

    Sign convention: raw values are negative (futures cheap to spot);
    multiplied by -1 so the spread is expressed as unsigned mispricing.
    """
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if c.lower() in ("date",))
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    sf_cols = [c for c in df.columns if c.upper().startswith("TREASURY_SF_")]
    assert sf_cols, f"No Treasury_SF_* columns in {path.name}"

    long = df[["date", *sf_cols]].melt("date", var_name="raw_name", value_name="y_raw")
    long["tenor"] = long["raw_name"].str.extract(r"(\d+)Y").astype(float)
    if tenors is not None:
        long = long[long["tenor"].isin(tenors)]

    long["series_id"]      = "ust_sf_" + long["tenor"].astype(int).astype(str) + "y"
    long["strategy"]       = "ust_spot_fut"
    long["treasury_based"] = 1
    long["y_bps"]          = safe_identity(long["y_raw"]) * (-1)
    long["y_abs_bps"]      = long["y_bps"].abs()

    diag = long.groupby("series_id")["y_bps"].agg(["count", lambda x: x.abs().median()])
    diag.columns = ["n", "median_abs_bps"]
    print("UST SF series medians (bps):")
    print(diag.to_string())

    return long[["date", "strategy", "series_id", "tenor", "treasury_based",
                 "y_bps", "y_abs_bps"]].dropna(subset=["date", "y_bps"])


def load_tips_treas_bps(path: Path, tenors: list[int] | None = None) -> pd.DataFrame:
    """Load TIPS-Treasury arbitrage spreads. Values are already in basis points."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    arb_cols = [c for c in df.columns if c.startswith("arb_")]
    assert arb_cols, f"No arb_* columns in {path.name}"

    long = df[["date", *arb_cols]].melt("date", var_name="raw_name", value_name="y_raw")
    long["tenor"] = long["raw_name"].str.extract(r"arb_(\d+)").astype(float)
    if tenors is not None:
        long = long[long["tenor"].isin(tenors)]

    long["series_id"]      = "tips_treas_" + long["tenor"].astype(int).astype(str) + "y"
    long["strategy"]       = "tips_treas"
    long["treasury_based"] = 1
    long["y_bps"]          = safe_identity(long["y_raw"])
    long["y_abs_bps"]      = long["y_bps"].abs()

    diag = long.groupby("series_id")["y_bps"].agg(["count", lambda x: x.abs().median()])
    diag.columns = ["n", "median_abs_bps"]
    print("TIPS-Treasury series medians (bps):")
    print(diag.to_string())

    return long[["date", "strategy", "series_id", "tenor", "treasury_based",
                 "y_bps", "y_abs_bps"]].dropna(subset=["date", "y_bps"])


def load_equity_sf_bps(series_dir: Path, indices: list[str] | None = None) -> pd.DataFrame:
    """
    Load equity spot-futures spreads.

    Uses the non-filtered column (spread_SPX, spread_NDX, spread_INDU),
    which is already in basis points (median ~40-47 bps).
    """
    if indices is None:
        indices = ["SPX", "NDX", "INDU"]

    parts = []
    for idx in indices:
        path = series_dir / f"equity_spot_spread_{idx}.csv"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue

        df = pd.read_csv(path)
        date_col = next(c for c in df.columns if c.lower() in ("date",))
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        col = EQUITY_SPREAD_COL_PREFERENCE[idx]
        if col not in df.columns:
            col_filt = f"spread_{idx}_filtered"
            if col_filt in df.columns:
                print(f"WARNING: using {col_filt} (x100) for {idx} — preferred column not found")
                y = pd.to_numeric(df[col_filt], errors="coerce") * 100.0
            else:
                print(f"ERROR: no spread column found for {idx}, skipping")
                continue
        else:
            y = pd.to_numeric(df[col], errors="coerce")

        med = float(y.abs().median())
        print(f"Equity {idx} ({col}): median_abs = {med:.2f} bps")

        tmp = pd.DataFrame({
            "date":           df["date"],
            "strategy":       "eq_spot_fut",
            "series_id":      f"eq_{idx.lower()}",
            "tenor":          np.nan,
            "treasury_based": 0,
            "y_bps":          y,
        })
        tmp["y_abs_bps"] = tmp["y_bps"].abs()
        parts.append(tmp[["date", "strategy", "series_id", "tenor", "treasury_based",
                           "y_bps", "y_abs_bps"]].dropna(subset=["date", "y_bps"]))

    return pd.concat(parts, ignore_index=True)


def stack_all_outcomes(
    series_dir: Path,
    tenor_subset: list[int] | None = None,
    equity_indices: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and stack all four strategy types into a single long-format panel.

    All series are returned in basis points.
    Strategies:
      - tips_treas:   TIPS-Treasury arbitrage (treasury_based=1)
      - ust_spot_fut: UST spot-futures basis (treasury_based=1)
      - cip:          Covered interest parity deviations (treasury_based=0)
      - eq_spot_fut:  Equity spot-futures basis (treasury_based=0)
    """
    if tenor_subset is None:
        tenor_subset = [2, 5, 10]
    if equity_indices is None:
        equity_indices = ["SPX", "NDX", "INDU"]

    parts: list[pd.DataFrame] = []

    tips_path = series_dir / "tips_treasury_implied_rf_2010.parquet"
    parts.append(load_tips_treas_bps(tips_path, tenors=tenor_subset))

    sf_path = series_dir / "treasury_sf_output.csv"
    parts.append(load_ust_sf_bps(sf_path, tenors=tenor_subset))

    cip_path = series_dir / "cip_spreads_3m_bps.csv"
    parts.append(load_cip_bps(cip_path))

    parts.append(load_equity_sf_bps(series_dir, indices=equity_indices))

    out = pd.concat(parts, ignore_index=True)
    print(f"\nStacked panel: {len(out):,} rows, {out['series_id'].nunique()} unique series")
    return out
