"""
Fix 8: Build rolling Treasury issuance controls from bi-weekly auction data.

Problem: issu_7_bil, issu_14_bil, issu_30_bil are listed in DIRECT_CONTROLS
but were silently dropped because load_controls() never processed the issuance CSV.
The bi-weekly CSV (treasury_issuance_by_tenor_fiscaldata.csv) contains individual
auction amounts; it must be aggregated to rolling daily totals.

Implementation:
  - Aggregate all tenor buckets to daily total issuance (0 on non-auction days)
  - Compute rolling 7/14/30-day sums
  - Lag by 1 calendar day (point-in-time discipline)
  - Convert to billions (divide by 1e9)
  - Return a daily DataFrame covering the requested date_range
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent


def load_issuance_raw(path: Path) -> pd.DataFrame:
    """
    Load the bi-weekly treasury issuance CSV.

    Expects columns: issue_date (or date), tenor_bucket, issuance_amount.
    Returns a DataFrame with standardised column names.
    """
    df = pd.read_csv(path)

    # Normalise date column
    date_col = next(
        (c for c in df.columns if c.lower() in ("issue_date", "date", "auction_date")),
        None,
    )
    if date_col is None:
        raise ValueError(f"No date column found in {path.name}. Columns: {list(df.columns)}")
    df = df.rename(columns={date_col: "issue_date"})
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df = df.dropna(subset=["issue_date"])

    # Normalise amount column
    amt_col = next(
        (c for c in df.columns if "amount" in c.lower() or "issuance" in c.lower()),
        None,
    )
    if amt_col is None:
        # Try any numeric column that isn't the date
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
    From the bi-weekly treasury issuance CSV, build:
      issu_7_bil  = rolling 7-calendar-day total issuance (all tenors), lagged 1 day
      issu_14_bil = rolling 14-day total, lagged 1 day
      issu_30_bil = rolling 30-day total, lagged 1 day

    Parameters
    ----------
    path : Path to treasury_issuance_by_tenor_fiscaldata.csv
    date_range : (start, end) strings for the output date index

    Returns
    -------
    DataFrame with columns [date, issu_7_bil, issu_14_bil, issu_30_bil]
    One row per calendar day in date_range.
    """
    raw = load_issuance_raw(path)

    # Aggregate all tenors: sum issuance_amount per auction date
    daily = (
        raw.groupby("issue_date")["issuance_amount"]
        .sum()
        .reset_index()
        .rename(columns={"issue_date": "date"})
    )

    # Build full daily calendar
    start, end = date_range
    cal = pd.DataFrame(
        {"date": pd.date_range(start, end, freq="D")}
    )
    daily = cal.merge(daily, on="date", how="left")
    daily["issuance_amount"] = daily["issuance_amount"].fillna(0.0)
    daily = daily.sort_values("date").reset_index(drop=True)

    # Rolling sums (calendar-day windows, min_periods=0)
    amt = daily["issuance_amount"]
    daily["issu_7_raw"] = amt.rolling(window=7, min_periods=0).sum()
    daily["issu_14_raw"] = amt.rolling(window=14, min_periods=0).sum()
    daily["issu_30_raw"] = amt.rolling(window=30, min_periods=0).sum()

    # Lag 1 calendar day (point-in-time: today's reg uses yesterday's issuance)
    daily["issu_7_bil"] = daily["issu_7_raw"].shift(1) / 1e9
    daily["issu_14_bil"] = daily["issu_14_raw"].shift(1) / 1e9
    daily["issu_30_bil"] = daily["issu_30_raw"].shift(1) / 1e9

    # Drop intermediate columns; keep only output
    result = daily[["date", "issu_7_bil", "issu_14_bil", "issu_30_bil"]].copy()

    # Forward-fill any leading NaN from the lag (first day has NaN after shift)
    result[["issu_7_bil", "issu_14_bil", "issu_30_bil"]] = (
        result[["issu_7_bil", "issu_14_bil", "issu_30_bil"]].fillna(0.0)
    )

    return result


if __name__ == "__main__":
    path = REPO_ROOT / "data" / "raw" / "event_inputs" / "treasury_issuance_by_tenor_fiscaldata.csv"
    OUT_DIR = REPO_ROOT / "_outputs"
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Build rolling issuance controls")
    print("=" * 70)

    raw = load_issuance_raw(path)
    print(f"\nRaw issuance data: {len(raw):,} rows")
    print(f"Date range: {raw['issue_date'].min().date()} to {raw['issue_date'].max().date()}")

    result = build_issuance_controls(path, date_range=("2019-01-01", "2021-12-31"))
    print(f"\nBuilt daily issuance controls: {len(result):,} rows")

    result.to_csv(OUT_DIR / "fix8_issuance_controls.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/fix8_issuance_controls.csv")
