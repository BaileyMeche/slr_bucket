from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATE_CANDIDATES = ["date", "DATE", "observation_date", "timestamp", "Time Period"]
DEFAULT_FALLBACK_ROOTS = [
    Path("."),
    Path("~/slr_episode").expanduser(),
    Path("~/data").expanduser(),
    Path("~/data/_output").expanduser(),
    Path("~/data/_data").expanduser(),
    Path("~/data/data_manual").expanduser(),
]
JOIN_KEY_CONVENTIONS = {
    "daily": ["date"],
    "weekly": ["date"],
    "quarterly": ["report_date"],
    "issuance_raw": ["issue_date"],
}


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
    if med <= 100:
        return "quarterly"
    return "irregular"


def _dataset_layer(path: Path) -> str:
    parts = set(path.parts)
    if "raw" in parts:
        return "raw"
    if "intermediate" in parts:
        return "intermediate"
    if "series" in parts:
        return "series"
    if "outputs" in parts:
        return "output"
    return "unknown"


def _join_hints(df: pd.DataFrame, frequency: str, path: Path) -> str:
    hints: list[str] = []
    conv = JOIN_KEY_CONVENTIONS.get(frequency)
    if conv and all(c in df.columns for c in conv):
        hints.append(f"{frequency}:{'+'.join(conv)}")
    if "issue_date" in df.columns:
        hints.append("issuance_raw:issue_date")
    candidate = [c for c in ["date", "report_date", "issue_date", "tenor", "tenor_bucket"] if c in df.columns]
    if candidate:
        hints.append("keys:" + "+".join(candidate))
    hints.append(f"layer:{_dataset_layer(path)}")
    return " | ".join(dict.fromkeys(hints))


def resolve_dataset_path(
    dataset_name: str,
    expected_dir: Path | None = None,
    fallback_roots: list[Path] | None = None,
    prefer_ext: tuple[str, ...] = (".parquet", ".csv"),
) -> Path:
    roots = fallback_roots or DEFAULT_FALLBACK_ROOTS
    candidates: list[Path] = []

    if expected_dir is not None:
        for ext in prefer_ext:
            p = expected_dir / f"{dataset_name}{ext}"
            if p.exists():
                return p

    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue
        for ext in prefer_ext:
            matches = sorted(root.rglob(f"{dataset_name}{ext}"))
            candidates.extend(matches)
        if candidates:
            break

    if not candidates:
        raise FileNotFoundError(f"Could not resolve dataset '{dataset_name}' with extensions {prefer_ext}.")

    # extension priority then shortest path heuristic
    candidates = sorted(candidates, key=lambda p: (prefer_ext.index(p.suffix.lower()), len(p.parts)))
    return candidates[0]


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
            freq = _infer_frequency(df["date"]) if "date" in df.columns else ("quarterly" if "report_date" in df.columns else "unknown")
            date_min = df["date"].min() if "date" in df.columns else pd.NaT
            date_max = df["date"].max() if "date" in df.columns else pd.NaT
            rows.append(
                {
                    "path": str(path),
                    "layer": _dataset_layer(path),
                    "rows": len(df),
                    "columns": ",".join(map(str, df.columns[:80])),
                    "frequency": freq,
                    "date_min": date_min,
                    "date_max": date_max,
                    "key_columns": ",".join([c for c in ["date", "report_date", "issue_date", "tenor", "tenor_bucket", "series", "value"] if c in df.columns]),
                    "join_hints": _join_hints(df, freq, path),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Catalog failed for %s: %s", path, exc)
            rows.append(
                {
                    "path": str(path),
                    "layer": _dataset_layer(path),
                    "rows": None,
                    "columns": "",
                    "frequency": "error",
                    "date_min": None,
                    "date_max": None,
                    "key_columns": "",
                    "join_hints": "",
                }
            )
    return pd.DataFrame(rows)


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
