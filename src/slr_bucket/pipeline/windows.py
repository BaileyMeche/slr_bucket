"""
windows.py — Event window overlap detection and clean event specification.

The two 2021 exit events are 12 calendar days (~9 trading days) apart:
  - 2021-03-19: SLR expiry announcement
  - 2021-03-31: SLR expiry effective date

With windows of ±20 or ±60 trading days, both windows overlap severely.
Strategy: use 2021-03-31 as the single exit event for main tables.
The announcement date (2021-03-19) is included only as a one-sided
[0, +9] supplementary window.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def trading_day_gap(d1: str, d2: str) -> int:
    """Number of trading days between d1 and d2 (exclusive of both endpoints)."""
    bdays = pd.bdate_range(d1, d2, freq="B")
    return max(0, len(bdays) - 2)


def window_overlap_trading_days(
    event1: str,
    event2: str,
    window: int,
) -> int:
    """
    Compute overlap in trading days between the post-window of event1
    and the pre-window of event2.
    """
    e1 = pd.Timestamp(event1)
    e2 = pd.Timestamp(event2)
    bday = pd.tseries.offsets.BusinessDay

    post_end_e1   = e1 + bday(window)
    pre_start_e2  = e2 - bday(window)

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
            overlap_td  = window_overlap_trading_days(e1, e2, W)
            is_overlap  = overlap_td > 0

            row = {
                "event_1":              e1,
                "event_2":              e2,
                "gap_trading_days":     gap_td,
                "window":               W,
                "overlap_trading_days": overlap_td,
                "is_overlap":           is_overlap,
                "recommended_max_window": gap_td // 2,
            }
            rows.append(row)

            if is_overlap:
                msg = (
                    f"Window overlap: events {e1} and {e2} are {gap_td} t.d. apart "
                    f"but window=\u00b1{W} creates {overlap_td} t.d. of overlap. "
                    f"Recommended max window: \u00b1{gap_td // 2} t.d."
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
    Return (event, effective_window) pairs with overlap-free specifications.

    Primary events: 2020-04-01 (entry) and 2021-03-31 (exit).
    Announcement 2021-03-19: supplementary one-sided [0, +9] window only.
    """
    result = []

    primary_events = [e for e in event_dates if e != "2021-03-19"]
    for event in primary_events:
        for W in windows:
            result.append({
                "event":               event,
                "window":              W,
                "window_type":         "symmetric",
                "lo":                  -W,
                "hi":                  W,
                "include_in_main_table": True,
                "note":                "",
            })

    if "2021-03-19" in event_dates:
        gap = trading_day_gap("2021-03-19", "2021-03-31")
        result.append({
            "event":               "2021-03-19",
            "window":              gap,
            "window_type":         "one_sided_post",
            "lo":                  0,
            "hi":                  gap,
            "include_in_main_table": False,
            "note":                f"announcement window [0,+{gap}] t.d. — avoids overlap with 2021-03-31",
        })

    return result
