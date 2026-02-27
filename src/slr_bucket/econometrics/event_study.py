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
    """Trading-day event time based on available sample dates."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    event = pd.Timestamp(event_date)

    valid_dates = pd.Series(out[date_col].dropna().sort_values().drop_duplicates().tolist())
    if valid_dates.empty:
        out["event_time"] = np.nan
        return out

    # Reference index: first date on/after event, else final available date.
    ref_pos = valid_dates.searchsorted(event, side="left")
    if ref_pos >= len(valid_dates):
        ref_pos = len(valid_dates) - 1

    date_to_pos = {d: i for i, d in enumerate(valid_dates)}
    out["event_time"] = out[date_col].map(lambda d: date_to_pos.get(d, np.nan)) - ref_pos
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
    # work = add_event_time(df, event_date)
    # work = work[work["event_time"].between(-window, window)].copy()
    # work["post"] = (work["event_time"] >= 0).astype(int)
    # cols = ["post"] + (controls or [])
    # cols = [c for c in cols if c in work.columns]
    # work = work.dropna(subset=[y_col] + cols)
    # if len(work) < 8:
    #     return np.nan, np.nan, len(work)
    # X = sm.add_constant(work[cols], has_constant="add")
    # model = sm.OLS(work[y_col], X).fit()
    # robust = _nw_cov_params(model, lags=hac_lags)
    # est, se = _param_lookup(robust, "post", list(X.columns))
    # return est, se, int(robust.nobs)
    # Defensive: drop duplicate columns (keeps first occurrence)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Never allow y to be a control
    controls = [c for c in (controls or []) if c != y_col]

    work = add_event_time(df, event_date)
    work = work[work["event_time"].between(-window, window)].copy()
    work["post"] = (work["event_time"] >= 0).astype(int)

    cols = ["post"] + controls
    cols = [c for c in cols if c in work.columns]

    # Coerce to numeric defensively
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=[y_col] + cols)

    # Need both pre and post to identify "post"
    if len(work) < 8 or work["post"].nunique(dropna=True) < 2:
        return np.nan, np.nan, len(work)

    X = sm.add_constant(work[cols], has_constant="add")
    model = sm.OLS(work[y_col], X).fit()
    robust = _nw_cov_params(model, lags=hac_lags)

    # If post got dropped (collinearity), bail gracefully
    if "post" not in robust.model.exog_names:
        return np.nan, np.nan, int(robust.nobs)

    est, se = _param_lookup(robust, "post", list(robust.model.exog_names))
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
    # work = add_event_time(df, event_date)
    # work["bin"] = make_bins(work["event_time"], bins)
    # work = work.dropna(subset=["bin", y_col]).copy()
    # dummies = pd.get_dummies(work["bin"], prefix="bin")
    # if dummies.empty:
    #     return pd.DataFrame(columns=["term", "estimate", "se", "ci_low", "ci_high", "n"])
    # ref = "bin_[-20,-1]" if "bin_[-20,-1]" in dummies.columns else dummies.columns[0]
    # dummies = dummies.drop(columns=[ref])
    # X = dummies
    # if controls:
    #     present = [c for c in controls if c in work.columns]
    #     if present:
    #         X = pd.concat([X, work[present]], axis=1)
    # joined = pd.concat([work[[y_col]], X], axis=1).dropna()
    # if joined.empty:
    #     return pd.DataFrame(columns=["term", "estimate", "se", "ci_low", "ci_high", "n"])
    # Xf = sm.add_constant(joined.drop(columns=[y_col]), has_constant="add")

    # #########
    # # y
    # y = joined[y_col]

    # # X: drop date + any other non-regressors
    # X = joined.drop(columns=[y_col], errors="ignore")
    # X = X.drop(columns=["date"], errors="ignore")  # <-- critical

    # # coerce any remaining non-numeric columns (defensive)
    # for col in X.columns:
    #     if not pd.api.types.is_numeric_dtype(X[col]):
    #         X[col] = pd.to_numeric(X[col], errors="coerce")

    # # drop all-NaN / constant columns (optional but helps robustness)
    # X = X.dropna(axis=1, how="all")
    # const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    # X = X.drop(columns=const_cols, errors="ignore")

    # # align + drop missing
    # reg = pd.concat([y, X], axis=1).dropna()
    # y = reg[y_col].astype(float)
    # X = reg.drop(columns=[y_col]).astype(float)

    # X = sm.add_constant(X, has_constant="add")
    # res = sm.OLS(y, X).fit()
    # #########
    
    # # res = sm.OLS(joined[y_col], Xf).fit()
    # robust = _nw_cov_params(res, hac_lags)
    # names = list(Xf.columns)
    # out = []
    # for col in dummies.columns:
    #     coef, se = _param_lookup(robust, col, names)
    #     out.append({"term": col, "estimate": coef, "se": se, "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se, "n": int(robust.nobs)})
    # return pd.DataFrame(out)
    
    # Defensive: drop duplicate columns (keeps first occurrence)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Never allow y to be a control
    controls = [c for c in (controls or []) if c != y_col]

    work = add_event_time(df, event_date)
    work["bin"] = make_bins(work["event_time"], bins)

    # Coerce outcome numeric
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")

    work = work.dropna(subset=["bin", y_col]).copy()

    dummies = pd.get_dummies(work["bin"], prefix="bin")
    if dummies.empty:
        return pd.DataFrame(columns=["term", "estimate", "se", "ci_low", "ci_high", "n"])

    ref = "bin_[-20,-1]" if "bin_[-20,-1]" in dummies.columns else dummies.columns[0]
    dummies = dummies.drop(columns=[ref])

    X = dummies

    if controls:
        present = [c for c in controls if c in work.columns and c != y_col]
        if present:
            tmp = work[present].copy()
            for c in present:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            X = pd.concat([X, tmp], axis=1)

    joined = pd.concat([work[[y_col]], X], axis=1).dropna()
    if joined.empty:
        return pd.DataFrame(columns=["term", "estimate", "se", "ci_low", "ci_high", "n"])

    y = joined[y_col]
    # Force 1D endog (if y_col was duplicated somewhere, this prevents (n,k) endog)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    X = joined.drop(columns=[y_col], errors="ignore")

    # Drop all-NaN / constant columns (avoid singular designs)
    X = X.dropna(axis=1, how="all")
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=const_cols, errors="ignore")

    if X.empty:
        return pd.DataFrame(columns=["term", "estimate", "se", "ci_low", "ci_high", "n"])

    X = sm.add_constant(X.astype(float), has_constant="add")
    res = sm.OLS(y.astype(float), X).fit()
    robust = _nw_cov_params(res, hac_lags)

    names = list(robust.model.exog_names)  # correct names for fitted model

    out = []
    for col in dummies.columns:
        if col not in names:
            continue
        coef, se = _param_lookup(robust, col, names)
        out.append(
            {"term": col, "estimate": coef, "se": se,
             "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se,
             "n": int(robust.nobs)}
        )
    return pd.DataFrame(out)