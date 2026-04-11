"""
Fix 5: Event window overlap detection and remediation.

The two 2021 exit events are 12 calendar days (≈9 trading days) apart:
  - 2021-03-19: SLR expiry announcement
  - 2021-03-31: SLR expiry effective date

With windows of ±20 or ±60 trading days, both windows overlap severely.
For W=20: the post-window of 2021-03-19 (extends to ~2021-04-16) fully
          overlaps with the pre-window of 2021-03-31 (extends back to ~2021-03-03).

Strategy:
  1. Drop 2021-03-19 from per-event regression tables (analytically dominated)
  2. Treat 2021-03-31 as the single "exit" event
  3. Add overlap assertions to the regression engine
  4. For the announcement: use a one-sided [0, +9] window for 2021-03-19 only
     to capture the announcement-to-expiry period
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def trading_day_gap(d1: str, d2: str) -> int:
    """Number of trading days between d1 and d2 (exclusive of both endpoints)."""
    bdays = pd.bdate_range(d1, d2, freq="B")
    return max(0, len(bdays) - 2)  # exclude endpoints


def window_overlap_trading_days(
    event1: str,
    event2: str,
    window: int,
) -> int:
    """
    Compute overlap in trading days between the post-window of event1
    and the pre-window of event2.

    post-window of event1: [event1, event1 + W]
    pre-window of event2: [event2 - W, event2)
    Overlap = max(0, (event1 + W) - (event2 - W)) in trading days
    """
    e1 = pd.Timestamp(event1)
    e2 = pd.Timestamp(event2)
    bday = pd.tseries.offsets.BusinessDay

    post_end_e1 = e1 + bday(window)
    pre_start_e2 = e2 - bday(window)

    if post_end_e1 < pre_start_e2:
        return 0

    overlap_dates = pd.bdate_range(pre_start_e2, post_end_e1, freq="B")
    return len(overlap_dates)


def check_all_overlaps(
    event_dates: list[str],
    windows: list[int],
    raise_on_overlap: bool = False,
) -> pd.DataFrame:
    """
    Check all consecutive event pairs for window overlap.
    Returns a DataFrame with overlap statistics.
    """
    rows = []
    for i in range(len(event_dates) - 1):
        e1, e2 = event_dates[i], event_dates[i + 1]
        gap_td = trading_day_gap(e1, e2)

        for W in windows:
            overlap_td = window_overlap_trading_days(e1, e2, W)
            is_overlap = overlap_td > 0

            row = {
                "event_1": e1,
                "event_2": e2,
                "gap_trading_days": gap_td,
                "window": W,
                "overlap_trading_days": overlap_td,
                "is_overlap": is_overlap,
                "recommended_max_window": gap_td // 2,
            }
            rows.append(row)

            if is_overlap:
                msg = (
                    f"Window overlap: events {e1} and {e2} are {gap_td} t.d. apart "
                    f"but window=±{W} creates {overlap_td} t.d. of overlap. "
                    f"Recommended max window: ±{gap_td // 2} t.d."
                )
                if raise_on_overlap:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)

    return pd.DataFrame(rows)


def get_clean_events_and_windows(
    event_dates: list[str],
    windows: list[int],
) -> list[dict]:
    """
    Return list of (event, effective_window) pairs where windows are
    truncated to avoid overlap.

    Strategy:
    - For the pair (2021-03-19, 2021-03-31): gap=9 t.d., so max window = 4 t.d.
      Since this is too restrictive, we drop 2021-03-19 from main tables and
      add it as a special one-sided [0,+9] window entry.
    - Use 2021-03-31 as the primary exit event with full windows.
    """
    result = []

    # Primary events: entry (2020-04-01) and expiry (2021-03-31)
    primary_events = [e for e in event_dates if e not in ("2021-03-19",)]

    for event in primary_events:
        for W in windows:
            result.append({
                "event": event,
                "window": W,
                "window_type": "symmetric",
                "lo": -W,
                "hi": W,
                "include_in_main_table": True,
                "note": "",
            })

    # Announcement event: one-sided [0, +9] window
    if "2021-03-19" in event_dates:
        gap = trading_day_gap("2021-03-19", "2021-03-31")
        result.append({
            "event": "2021-03-19",
            "window": gap,
            "window_type": "one_sided_post",
            "lo": 0,
            "hi": gap,
            "include_in_main_table": False,
            "note": f"announcement window [0,+{gap}] t.d. only (avoids overlap with 2021-03-31)",
        })

    return result


def filter_event_window(
    df: pd.DataFrame,
    event_date: str,
    lo: int,
    hi: int,
    series_fe_col: str = "series_id",
) -> pd.DataFrame:
    """
    Filter panel to the event window [lo, hi] trading days.
    Uses trading-day event time assigned within each series.
    """
    from audit_repairs.fix_3_pooled import add_event_time_trading
    work = add_event_time_trading(df, event_date, group_col=series_fe_col)
    return work[work["event_time"].between(lo, hi)].copy()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    events = ["2020-04-01", "2021-03-19", "2021-03-31"]
    windows = [20, 60]

    print("=" * 70)
    print("FIX 5: Event window overlap analysis")
    print("=" * 70)

    overlap_df = check_all_overlaps(events, windows, raise_on_overlap=False)
    print("\nOverlap table:")
    print(overlap_df.to_string(index=False))

    print("\nClean event specifications:")
    clean = get_clean_events_and_windows(events, windows)
    for spec in clean:
        print(f"  event={spec['event']}, window=[{spec['lo']},{spec['hi']}] t.d., "
              f"type={spec['window_type']}, main_table={spec['include_in_main_table']}"
              + (f"  ({spec['note']})" if spec["note"] else ""))

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    overlap_df.to_csv(out_dir / "fix5_overlap_diagnostic.csv", index=False)
    pd.DataFrame(clean).to_csv(out_dir / "fix5_clean_event_specs.csv", index=False)
    print(f"\nSaved: {out_dir}/fix5_overlap_diagnostic.csv")
    print(f"Saved: {out_dir}/fix5_clean_event_specs.csv")

    print("\nConclusion:")
    print("  2021-03-19 and 2021-03-31 windows overlap for W=20 and W=60.")
    print("  Primary tables use only 2020-04-01 (entry) and 2021-03-31 (exit).")
    print("  2021-03-19 gets a one-sided [0,+9] window as supplementary analysis.")
