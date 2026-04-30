"""
Fix 1: Deterministic unit-aware loader replacing _to_bps heuristic.

Root cause: _to_bps uses a median < 20 threshold to detect "percent units" and
multiply by 100. This incorrectly scales:
  - UST SF series whose stacked median is ~18 bps (< 20 threshold) -> x100 -> ~1800 bps
  - CIP currencies with low spreads (AUD, CAD, GBP, NZD, ~10-13 bps) -> x100 -> ~1200 bps

All data/series/ files are pre-processed to bps EXCEPT equity_spot_spread_*_filtered
which is in percent units. The non-_filtered equity spread columns are already in bps.

Fix: use the non-filtered equity column (spread_SPX, spread_NDX, spread_INDU) which
are already in bps. All other data/series/ files are returned as-is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

# -- Unit registry --------------------------------------------------------------
# "bps": already in basis points, return as-is
# "pct": multiply by 100 to get bps
# "decimal": multiply by 10000 to get bps
SERIES_UNITS: dict[str, str] = {
    "cip_spreads_3m_bps":        "bps",   # CIP_*_ln columns already in bps
    "tips_treasury_implied_rf_2010": "bps",  # arb_* columns already in bps
    "treasury_sf_output":        "bps",   # Treasury_SF_*Y columns already in bps
    # equity: use the non-filtered spread which is already in bps
    # (spread_SPX_filtered is in percent; avoid using it)
}

EQUITY_SPREAD_COL_PREFERENCE = {
    "SPX": "spread_SPX",    # median ~43.6 bps -- USE THIS, not _filtered
    "NDX": "spread_NDX",    # median ~40.1 bps
    "INDU": "spread_INDU",  # median ~47.0 bps
}


def safe_identity(x: pd.Series) -> pd.Series:
    """Return series as-is (already in bps)."""
    return pd.to_numeric(x, errors="coerce")


def scale_pct_to_bps(x: pd.Series) -> pd.Series:
    """Multiply by 100: percent -> bps (e.g., 0.43% -> 43 bps)."""
    return pd.to_numeric(x, errors="coerce") * 100.0


def scale_decimal_to_bps(x: pd.Series) -> pd.Series:
    """Multiply by 10000: decimal fraction -> bps (e.g., 0.0043 -> 43 bps)."""
    return pd.to_numeric(x, errors="coerce") * 10000.0


UNIT_CONVERTERS = {
    "bps": safe_identity,
    "pct": scale_pct_to_bps,
    "decimal": scale_decimal_to_bps,
}


def load_cip_bps(path: Path) -> pd.DataFrame:
    """Load CIP spreads -- already in bps, NO conversion applied."""
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
    long["series_id"] = "cip_" + long["currency"].str.lower()
    long["strategy"] = "cip"
    long["tenor"] = 0.25           # 3-month
    long["treasury_based"] = 0
    # NO _to_bps -- already in bps
    long["y_bps"] = safe_identity(long["y_raw"])
    long["y_abs_bps"] = long["y_bps"].abs()

    diag = long.groupby("series_id")["y_bps"].agg(["count", lambda x: x.abs().median()])
    diag.columns = ["n", "median_abs_bps"]
    print("[load] CIP medians after fix (all should be 5-100 bps):")
    print(diag.to_string())

    return long[["date", "strategy", "series_id", "tenor", "treasury_based",
                 "y_bps", "y_abs_bps"]].dropna(subset=["date", "y_bps"])


def load_ust_sf_bps(path: Path, tenors: list[int] | None = None) -> pd.DataFrame:
    """Load UST spot-futures basis -- already in bps, NO _to_bps conversion."""
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

    long["series_id"] = "ust_sf_" + long["tenor"].astype(int).astype(str) + "y"
    long["strategy"] = "ust_spot_fut"
    long["treasury_based"] = 1
    # NO _to_bps -- file is already in bps (typical values -15 to -25 bps median)
    # Sign convention: file values are negative (futures cheap to spot); flip to make
    # positive so deviations are measured as unsigned mispricing.
    long["y_bps"] = safe_identity(long["y_raw"]) * (-1)
    long["y_abs_bps"] = long["y_bps"].abs()

    diag = long.groupby("series_id")["y_bps"].agg(["count", lambda x: x.abs().median()])
    diag.columns = ["n", "median_abs_bps"]
    print("[load] UST SF medians after fix (should be 10-80 bps):")
    print(diag.to_string())

    return long[["date", "strategy", "series_id", "tenor", "treasury_based",
                 "y_bps", "y_abs_bps"]].dropna(subset=["date", "y_bps"])


def load_tips_treas_bps(path: Path, tenors: list[int] | None = None) -> pd.DataFrame:
    """Load TIPS-Treasury arbitrage spreads -- already in bps."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    arb_cols = [c for c in df.columns if c.startswith("arb_")]
    assert arb_cols, f"No arb_* columns in {path.name}"

    long = df[["date", *arb_cols]].melt("date", var_name="raw_name", value_name="y_raw")
    long["tenor"] = long["raw_name"].str.extract(r"arb_(\d+)").astype(float)
    if tenors is not None:
        long = long[long["tenor"].isin(tenors)]

    long["series_id"] = "tips_treas_" + long["tenor"].astype(int).astype(str) + "y"
    long["strategy"] = "tips_treas"
    long["treasury_based"] = 1
    # Data values are in bps already (median 19-25 bps range)
    long["y_bps"] = safe_identity(long["y_raw"])
    long["y_abs_bps"] = long["y_bps"].abs()

    diag = long.groupby("series_id")["y_bps"].agg(["count", lambda x: x.abs().median()])
    diag.columns = ["n", "median_abs_bps"]
    print("[load] TIPS-Treasury medians after fix (should be 15-30 bps):")
    print(diag.to_string())

    return long[["date", "strategy", "series_id", "tenor", "treasury_based",
                 "y_bps", "y_abs_bps"]].dropna(subset=["date", "y_bps"])


def load_equity_sf_bps(series_dir: Path, indices: list[str] | None = None) -> pd.DataFrame:
    """Load equity spot-futures spreads from the non-filtered column (already in bps)."""
    if indices is None:
        indices = ["SPX", "NDX", "INDU"]

    parts = []
    for idx in indices:
        path = series_dir / f"equity_spot_spread_{idx}.csv"
        if not path.exists():
            print(f"[load] WARNING: {path} not found, skipping")
            continue

        df = pd.read_csv(path)
        date_col = next(c for c in df.columns if c.lower() in ("date",))
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Prefer the non-filtered column (already in bps, median ~40-47 bps)
        col = EQUITY_SPREAD_COL_PREFERENCE[idx]
        if col not in df.columns:
            # fallback: try _filtered with explicit x100
            col_filt = f"spread_{idx}_filtered"
            if col_filt in df.columns:
                print(f"[load] WARNING: using {col_filt} (x100) for {idx} -- non-filtered not found")
                y = pd.to_numeric(df[col_filt], errors="coerce") * 100.0
            else:
                print(f"[load] ERROR: no spread column for {idx}, skipping")
                continue
        else:
            y = pd.to_numeric(df[col], errors="coerce")

        med = float(y.abs().median())
        print(f"[load] Equity {idx} ({col}): median_abs = {med:.2f} bps (target: 20-60)")

        tmp = pd.DataFrame({
            "date": df["date"],
            "strategy": "eq_spot_fut",
            "series_id": f"eq_{idx.lower()}",
            "tenor": np.nan,
            "treasury_based": 0,
            "y_bps": y,
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
    All series are in basis points after this call.
    No _to_bps heuristic is applied.
    """
    if tenor_subset is None:
        tenor_subset = [2, 5, 10]
    if equity_indices is None:
        equity_indices = ["SPX", "NDX", "INDU"]

    parts: list[pd.DataFrame] = []

    # TIPS-Treasury
    tips_path = series_dir / "tips_treasury_implied_rf_2010.parquet"
    parts.append(load_tips_treas_bps(tips_path, tenors=tenor_subset))

    # UST Spot-Futures
    sf_path = series_dir / "treasury_sf_output.csv"
    parts.append(load_ust_sf_bps(sf_path, tenors=tenor_subset))

    # CIP 3-month
    cip_path = series_dir / "cip_spreads_3m_bps.csv"
    parts.append(load_cip_bps(cip_path))

    # Equity Spot-Futures
    parts.append(load_equity_sf_bps(series_dir, indices=equity_indices))

    out = pd.concat(parts, ignore_index=True)
    print(f"\n[load] Total stacked panel: {len(out):,} rows, "
          f"{out['series_id'].nunique()} unique series")
    return out


# -- Diagnostic: compare buggy vs fixed medians ---------------------------------
def run_unit_diagnostic(series_dir: Path) -> pd.DataFrame:
    """
    Compare _to_bps output vs corrected output for each series.
    Returns a DataFrame with before/after median_abs_bps.
    """
    rows = []

    def _to_bps_buggy(x: pd.Series) -> pd.Series:
        """Original heuristic from notebook -- kept here for before/after comparison."""
        s = pd.to_numeric(x, errors="coerce")
        med = float(np.nanmedian(np.abs(s.dropna().values))) if s.notna().any() else np.nan
        if not np.isfinite(med) or med == 0:
            return s
        if med < 0.05:
            return s * 10000.0
        if med < 20.0:
            return s * 100.0
        return s

    # CIP
    cip = pd.read_csv(series_dir / "cip_spreads_3m_bps.csv")
    for col in ["CIP_AUD_ln","CIP_CAD_ln","CIP_CHF_ln","CIP_EUR_ln",
                "CIP_GBP_ln","CIP_JPY_ln","CIP_NZD_ln","CIP_SEK_ln"]:
        if col not in cip.columns:
            continue
        s = pd.to_numeric(cip[col], errors="coerce")
        buggy = _to_bps_buggy(s)
        fixed = safe_identity(s)
        rows.append({
            "series": col, "strategy": "cip",
            "raw_median_abs": s.abs().median(),
            "buggy_median_abs": buggy.abs().median(),
            "fixed_median_abs": fixed.abs().median(),
            "scale_factor_applied": buggy.abs().median() / s.abs().median() if s.abs().median() > 0 else np.nan,
        })

    # UST SF (stacked)
    sf = pd.read_csv(series_dir / "treasury_sf_output.csv")
    sf_cols = [c for c in sf.columns if c.upper().startswith("TREASURY_SF_")]
    stacked = pd.concat([pd.to_numeric(sf[c], errors="coerce") for c in sf_cols], ignore_index=True)
    buggy_sf = _to_bps_buggy(stacked)
    fixed_sf = safe_identity(stacked)
    rows.append({
        "series": "Treasury_SF_STACKED", "strategy": "ust_spot_fut",
        "raw_median_abs": stacked.abs().median(),
        "buggy_median_abs": buggy_sf.abs().median(),
        "fixed_median_abs": fixed_sf.abs().median(),
        "scale_factor_applied": buggy_sf.abs().median() / stacked.abs().median() if stacked.abs().median() > 0 else np.nan,
    })
    for c in sf_cols:
        s = pd.to_numeric(sf[c], errors="coerce")
        buggy = _to_bps_buggy(s)
        fixed = safe_identity(s)
        rows.append({
            "series": c, "strategy": "ust_spot_fut",
            "raw_median_abs": s.abs().median(),
            "buggy_median_abs": buggy.abs().median(),
            "fixed_median_abs": fixed.abs().median(),
            "scale_factor_applied": buggy.abs().median() / s.abs().median() if s.abs().median() > 0 else np.nan,
        })

    # TIPS
    tips = pd.read_parquet(series_dir / "tips_treasury_implied_rf_2010.parquet")
    arb_cols = [c for c in tips.columns if c.startswith("arb_")]
    for c in arb_cols:
        s = pd.to_numeric(tips[c], errors="coerce")
        buggy = _to_bps_buggy(s)
        fixed = safe_identity(s)
        rows.append({
            "series": c, "strategy": "tips_treas",
            "raw_median_abs": s.abs().median(),
            "buggy_median_abs": buggy.abs().median(),
            "fixed_median_abs": fixed.abs().median(),
            "scale_factor_applied": buggy.abs().median() / s.abs().median() if s.abs().median() > 0 else np.nan,
        })

    # Equity
    for idx in ["SPX", "NDX", "INDU"]:
        p = series_dir / f"equity_spot_spread_{idx}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        col_filt = f"spread_{idx}_filtered"
        col_raw = f"spread_{idx}"
        if col_filt in df.columns:
            s_filt = pd.to_numeric(df[col_filt], errors="coerce")
            buggy = _to_bps_buggy(s_filt)
            rows.append({
                "series": col_filt, "strategy": f"eq_{idx}",
                "raw_median_abs": s_filt.abs().median(),
                "buggy_median_abs": buggy.abs().median(),
                "fixed_median_abs": buggy.abs().median(),  # _filtered x100 is actually correct
                "scale_factor_applied": buggy.abs().median() / s_filt.abs().median(),
            })
        if col_raw in df.columns:
            s_raw = pd.to_numeric(df[col_raw], errors="coerce")
            rows.append({
                "series": col_raw, "strategy": f"eq_{idx}",
                "raw_median_abs": s_raw.abs().median(),
                "buggy_median_abs": s_raw.abs().median(),  # raw is already bps, buggy passes through
                "fixed_median_abs": s_raw.abs().median(),  # same
                "scale_factor_applied": 1.0,
            })

    diag = pd.DataFrame(rows)
    return diag


if __name__ == "__main__":
    series_dir = REPO_ROOT / "data" / "series"

    print("=" * 70)
    print("Unit diagnostic (before vs after)")
    print("=" * 70)
    diag = run_unit_diagnostic(series_dir)
    print("\nBefore/After comparison:")
    print(diag[["series", "strategy", "raw_median_abs", "buggy_median_abs",
                 "fixed_median_abs", "scale_factor_applied"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("Loading corrected stacked panel...")
    print("=" * 70)
    panel = stack_all_outcomes(series_dir)
    print("\nPer-series summary statistics:")
    stats = panel.groupby(["strategy", "series_id"])["y_bps"].agg(
        N="count", mean=lambda x: x.mean(), median_abs=lambda x: x.abs().median()
    ).reset_index()
    print(stats.to_string(index=False))

    out_dir = REPO_ROOT / "_outputs"
    out_dir.mkdir(exist_ok=True)
    diag.to_csv(out_dir / "fix1_unit_diagnostic.csv", index=False)
    panel.to_parquet(out_dir / "fix1_stacked_panel.parquet", index=False)
    print(f"\nSaved: {out_dir}/fix1_unit_diagnostic.csv")
    print(f"Saved: {out_dir}/fix1_stacked_panel.parquet")
