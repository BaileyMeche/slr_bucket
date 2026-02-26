from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


DATE_CANDIDATES = ["date", "DATE", "observation_date", "timestamp", "Time Period"]


def _infer_frequency(date_series: pd.Series) -> str:
    ordered = date_series.dropna().sort_values().drop_duplicates()
    if len(ordered) < 3:
        return "unknown"
    deltas = ordered.diff().dropna().dt.days
    med = deltas.median()
    if med <= 1:
        return "daily"
    if med <= 7:
        return "weekly"
    if med <= 31:
        return "monthly"
    return "irregular"


def load_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported file format: {path}")


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            out = df.copy()
            out = out.rename(columns={col: "date"})
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
            return out
    return df


def build_data_catalog(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".json"}:
            continue
        try:
            df = normalize_date_column(load_any_table(path))
            key_cols = [c for c in ["date", "tenor", "series", "value"] if c in df.columns]
            freq = _infer_frequency(df["date"]) if "date" in df.columns else "unknown"
            date_min = df["date"].min() if "date" in df.columns else pd.NaT
            date_max = df["date"].max() if "date" in df.columns else pd.NaT
            rows.append(
                {
                    "path": str(path),
                    "rows": len(df),
                    "columns": ",".join(map(str, df.columns[:40])),
                    "frequency": freq,
                    "date_min": date_min,
                    "date_max": date_max,
                    "suggested_join_keys": ",".join(key_cols) if key_cols else "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Catalog failed for %s: %s", path, exc)
            rows.append({"path": str(path), "rows": None, "columns": "", "frequency": "error", "date_min": None, "date_max": None, "suggested_join_keys": ""})
    return pd.DataFrame(rows)


def find_daily_long(data_dir: Path) -> pd.DataFrame:
    candidates = list(data_dir.rglob("*daily*long*.parquet")) + list(data_dir.rglob("*daily*long*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Could not locate daily_long in ./data. Add a file named like '*daily*long*.csv|parquet' with columns date, tenor, series, value."
        )
    df = normalize_date_column(load_any_table(candidates[0]))
    expected = {"date", "tenor", "series", "value"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"daily_long missing columns {sorted(missing)} from source {candidates[0]}")
    return df[["date", "tenor", "series", "value"]].copy()


def discover_funding_series(data_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    patterns = ["sofr", "tgcr", "bgcr", "repo", "ofr"]
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        for pattern in patterns:
            if pattern in name and pattern not in mapping:
                mapping[pattern] = str(path)
    return mapping
