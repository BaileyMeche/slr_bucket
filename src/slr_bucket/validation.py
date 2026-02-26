from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_daily_long(df: pd.DataFrame) -> pd.DataFrame:
    req = {"date", "tenor", "series", "value"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"daily_long missing required columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["date", "tenor", "series", "value"]) 
    out = out.sort_values(["series", "tenor", "date"]).drop_duplicates(["date", "tenor", "series"], keep="last")
    return out


def report_merge_quality(base: pd.DataFrame, merged: pd.DataFrame, key: str = "date") -> dict[str, float]:
    if key not in base.columns or key not in merged.columns:
        return {"match_rate": 0.0}
    base_n = base[key].nunique()
    merged_n = merged[key].nunique()
    rate = merged_n / max(base_n, 1)
    if rate < 0.5:
        raise ValueError(
            f"Catastrophic merge: match_rate={rate:.2%}. Check date alignment and available control series in /data."
        )
    logger.info("merge match rate: %.2f%%", rate * 100)
    return {"match_rate": rate}
