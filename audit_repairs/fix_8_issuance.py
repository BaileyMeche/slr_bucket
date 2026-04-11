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

Expected values:
  - Normal weeks: issu_7_bil ~ 20-100 B (Tsy auctions ~2x/week)
  - March 2020 COVID relief: issu_30_bil ~ 350-450 B
  - 2019 pre-COVID baseline: issu_30_bil ~ 200-300 B
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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


def print_issuance_spot_check(df: pd.DataFrame) -> None:
    """Print spot-check values for key dates."""
    check_dates = [
        "2019-01-15",   # normal week
        "2020-03-16",   # COVID crash (week of March 9-20)
        "2020-04-01",   # SLR entry event
        "2020-04-30",   # one month post-entry
        "2021-03-31",   # SLR exit event
    ]
    print("\n=== Issuance controls spot-check ===")
    print(f"{'date':12s} {'issu_7_bil':>12s} {'issu_14_bil':>13s} {'issu_30_bil':>13s}")
    print("-" * 55)
    for d in check_dates:
        row = df[df["date"] == d]
        if row.empty:
            print(f"{d:12s} {'N/A':>12s}")
            continue
        r = row.iloc[0]
        print(f"{d:12s} {r['issu_7_bil']:12.1f} {r['issu_14_bil']:13.1f} {r['issu_30_bil']:13.1f}")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    path = repo_root / "data" / "raw" / "event_inputs" / "treasury_issuance_by_tenor_fiscaldata.csv"
    OUT_DIR = Path(__file__).parent / "outputs"
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("FIX 8: Build rolling issuance controls")
    print("=" * 70)

    raw = load_issuance_raw(path)
    print(f"\nRaw issuance data: {len(raw):,} rows")
    print(f"Date range: {raw['issue_date'].min().date()} to {raw['issue_date'].max().date()}")
    print(f"Unique auction dates: {raw['issue_date'].nunique()}")
    print(f"Median gap between auctions: "
          f"{raw['issue_date'].sort_values().diff().dt.days.median():.0f} days")
    total_B = raw["issuance_amount"].sum() / 1e9
    print(f"Total issuance in dataset: ${total_B:,.0f} B")

    result = build_issuance_controls(path, date_range=("2019-01-01", "2021-12-31"))
    print(f"\nBuilt daily issuance controls: {len(result):,} rows")

    print_issuance_spot_check(result)

    # Summary stats by year
    print("\n=== Annual summary (issu_30_bil) ===")
    result["year"] = result["date"].dt.year
    for yr, g in result.groupby("year"):
        s = g["issu_30_bil"].dropna()
        print(f"  {yr}: mean={s.mean():.1f} B, median={s.median():.1f} B, "
              f"max={s.max():.1f} B, min={s.min():.1f} B")

    result.drop(columns=["year"], errors="ignore").to_csv(
        OUT_DIR / "fix8_issuance_controls.csv", index=False
    )
    print(f"\nSaved: {OUT_DIR}/fix8_issuance_controls.csv")

    print("\n=== Conclusion ===")
    print("issu_7_bil, issu_14_bil, issu_30_bil are now available as daily controls.")
    print("They should be merged into load_controls() in repaired_pipeline.py.")
