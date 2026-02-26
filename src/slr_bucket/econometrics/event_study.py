from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class JumpResult:
    event_date: str
    tenor: str
    series: str
    window: int
    spec: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    n: int


def add_event_time(df: pd.DataFrame, event_date: str, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    event = pd.Timestamp(event_date)
    out["event_time"] = (pd.to_datetime(out[date_col]) - event).dt.days
    return out


def make_bins(event_time: pd.Series, bins: Iterable[tuple[int, int]]) -> pd.Series:
    labels = [f"[{low},{high}]" for low, high in bins]
    binned = pd.Series(pd.NA, index=event_time.index, dtype="object")
    for (low, high), label in zip(bins, labels):
        mask = (event_time >= low) & (event_time <= high)
        binned.loc[mask] = label
    return binned


def _nw_cov_params(model, lags: int):
    return model.get_robustcov_results(cov_type="HAC", maxlags=lags)


def _param_lookup(robust, name: str, exog_names: list[str]) -> tuple[float, float]:
    idx = exog_names.index(name)
    return float(robust.params[idx]), float(robust.bse[idx])


def jump_estimator(
    df: pd.DataFrame,
    y_col: str,
    event_date: str,
    window: int,
    controls: list[str] | None = None,
    hac_lags: int = 5,
) -> tuple[float, float, int]:
    work = add_event_time(df, event_date)
    work = work[work["event_time"].between(-window, window)].copy()
    work["post"] = (work["event_time"] >= 0).astype(int)
    cols = ["post"] + (controls or [])
    cols = [c for c in cols if c in work.columns]
    work = work.dropna(subset=[y_col] + cols)
    if len(work) < 8:
        return np.nan, np.nan, len(work)
    X = sm.add_constant(work[cols], has_constant="add")
    model = sm.OLS(work[y_col], X).fit()
    robust = _nw_cov_params(model, lags=hac_lags)
    est, se = _param_lookup(robust, "post", list(X.columns))
    return est, se, int(robust.nobs)


def block_bootstrap_jump(
    df: pd.DataFrame,
    y_col: str,
    event_date: str,
    window: int,
    controls: list[str] | None = None,
    reps: int = 200,
    block_size: int = 5,
    seed: int = 42,
) -> float:
    work = add_event_time(df, event_date)
    work = work[work["event_time"].between(-window, window)].copy().reset_index(drop=True)
    if len(work) < 10:
        return np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        starts = rng.integers(0, max(len(work) - block_size, 1), size=max(len(work) // block_size, 1))
        idx = []
        for s in starts:
            idx.extend(range(s, min(s + block_size, len(work))))
        sample = work.iloc[idx]
        est, _, _ = jump_estimator(sample, y_col, event_date, window, controls=controls, hac_lags=1)
        if np.isfinite(est):
            vals.append(est)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan


def event_study_regression(
    df: pd.DataFrame,
    y_col: str,
    event_date: str,
    bins: list[tuple[int, int]],
    controls: list[str] | None = None,
    hac_lags: int = 5,
) -> pd.DataFrame:
    work = add_event_time(df, event_date)
    work["bin"] = make_bins(work["event_time"], bins)
    work = work.dropna(subset=["bin", y_col]).copy()
    dummies = pd.get_dummies(work["bin"], prefix="bin")
    if dummies.empty:
        return pd.DataFrame(columns=["term", "estimate", "se", "ci_low", "ci_high", "n"])
    ref = "bin_[-20,-1]" if "bin_[-20,-1]" in dummies.columns else dummies.columns[0]
    dummies = dummies.drop(columns=[ref])
    X = dummies
    if controls:
        present = [c for c in controls if c in work.columns]
        if present:
            X = pd.concat([X, work[present]], axis=1)
    joined = pd.concat([work[[y_col]], X], axis=1).dropna()
    if joined.empty:
        return pd.DataFrame(columns=["term", "estimate", "se", "ci_low", "ci_high", "n"])
    Xf = sm.add_constant(joined.drop(columns=[y_col]), has_constant="add")
    res = sm.OLS(joined[y_col], Xf).fit()
    robust = _nw_cov_params(res, hac_lags)
    names = list(Xf.columns)
    out = []
    for col in dummies.columns:
        coef, se = _param_lookup(robust, col, names)
        out.append({"term": col, "estimate": coef, "se": se, "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se, "n": int(robust.nobs)})
    return pd.DataFrame(out)
