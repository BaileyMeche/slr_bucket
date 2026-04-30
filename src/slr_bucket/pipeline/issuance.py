"""
issuance.py — Rolling Treasury issuance controls.

Builds daily rolling issuance totals from bi-weekly auction data:
  issu_7_bil  = rolling 7-calendar-day total issuance (all tenors), lagged 1 day
  issu_14_bil = rolling 14-day total, lagged 1 day
  issu_30_bil = rolling 30-day total, lagged 1 day

Source: treasury_issuance_by_tenor_fiscaldata.csv (bi-weekly auction dates).

Expected values:
  Normal weeks: issu_7_bil ~ 20-100 B (Treasury auctions ~2x/week)
  March 2020 COVID relief: issu_30_bil ~ 350-450 B
  2019 pre-COVID baseline: issu_30_bil ~ 200-300 B
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_issuance_raw(path: Path) -> pd.DataFrame:
    """
    Load the bi-weekly Treasury issuance CSV.

    Expects columns: issue_date (or date), tenor_bucket, issuance_amount.
    Returns a DataFrame with standardised column names.
    """
    df = pd.read_csv(path)

    date_col = next(
        (c for c in df.columns if c.lower() in ("issue_date", "date", "auction_date")),
        None,
    )
    if date_col is None:
        raise ValueError(f"No date column found in {path.name}. Columns: {list(df.columns)}")
    df = df.rename(columns={date_col: "issue_date"})
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df = df.dropna(subset=["issue_date"])

    amt_col = next(
        (c for c in df.columns if "amount" in c.lower() or "issuance" in c.lower()),
        None,
    )
    if amt_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        amt_col = numeric_cols[0] if numeric_cols else None
    if amt_col is None:
        raise ValueError(f"No amount column found in {path.name}. Columns: {list(df.columns)}")
    df = df.rename(columns={amt_col: "issuance_amount"})
    df["issuance_amount"] = pd.to_numeric(df["issuance_amount"], errors="coerce").fillna(0)

    return df[["issue_date", "issuance_amount"]]


def build_issuance_controls(
    path: Path,
    date_range: tuple[str, str] = ("2019-01-01", "2021-12-31"),
) -> pd.DataFrame:
    """
    Build daily rolling Treasury issuance controls from the bi-weekly auction CSV.

    Parameters
    ----------
    path       : Path to treasury_issuance_by_tenor_fiscaldata.csv
    date_range : (start, end) strings for the output date index

    Returns
    -------
    DataFrame with columns [date, issu_7_bil, issu_14_bil, issu_30_bil].
    One row per calendar day in date_range.
    """
    raw   = load_issuance_raw(path)
    daily = (
        raw.groupby("issue_date")["issuance_amount"]
        .sum()
        .reset_index()
        .rename(columns={"issue_date": "date"})
    )

    start, end = date_range
    cal   = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    daily = cal.merge(daily, on="date", how="left")
    daily["issuance_amount"] = daily["issuance_amount"].fillna(0.0)
    daily = daily.sort_values("date").reset_index(drop=True)

    amt = daily["issuance_amount"]
    daily["issu_7_raw"]  = amt.rolling(window=7,  min_periods=0).sum()
    daily["issu_14_raw"] = amt.rolling(window=14, min_periods=0).sum()
    daily["issu_30_raw"] = amt.rolling(window=30, min_periods=0).sum()

    # Lag 1 calendar day (point-in-time)
    daily["issu_7_bil"]  = daily["issu_7_raw"].shift(1)  / 1e9
    daily["issu_14_bil"] = daily["issu_14_raw"].shift(1) / 1e9
    daily["issu_30_bil"] = daily["issu_30_raw"].shift(1) / 1e9

    result = daily[["date", "issu_7_bil", "issu_14_bil", "issu_30_bil"]].copy()
    result[["issu_7_bil", "issu_14_bil", "issu_30_bil"]] = (
        result[["issu_7_bil", "issu_14_bil", "issu_30_bil"]].fillna(0.0)
    )
    return result
