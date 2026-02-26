from __future__ import annotations

import numpy as np
import pandas as pd

from slr_bucket.econometrics.event_study import add_event_time, block_bootstrap_jump, event_study_regression, jump_estimator, make_bins


def synthetic_df(n=180):
    date = pd.date_range("2020-01-01", periods=n, freq="D")
    event = pd.Timestamp("2020-03-15")
    event_time = (date - event).days
    post = (event_time >= 0).astype(int)
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = 0.8 * post + 0.3 * x + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"date": date, "y": y, "x": x})


def test_add_event_time():
    df = synthetic_df(10)
    out = add_event_time(df, "2020-01-05")
    assert "event_time" in out.columns
    assert out.loc[4, "event_time"] == 0


def test_make_bins():
    s = pd.Series([-10, -5, 0, 4, 12])
    out = make_bins(s, [(-10, -1), (0, 0), (1, 10)])
    assert out.iloc[0] == "[-10,-1]"
    assert out.iloc[2] == "[0,0]"
    assert pd.isna(out.iloc[4])


def test_jump_estimator_positive_shift():
    df = synthetic_df()
    est, se, n = jump_estimator(df, "y", "2020-03-15", window=20, controls=["x"], hac_lags=3)
    assert n > 20
    assert est > 0.2
    assert se > 0


def test_block_bootstrap_jump_runs():
    df = synthetic_df()
    bse = block_bootstrap_jump(df, "y", "2020-03-15", window=20, controls=["x"], reps=20, block_size=4, seed=1)
    assert bse >= 0


def test_event_study_regression_output():
    df = synthetic_df()
    out = event_study_regression(df, "y", "2020-03-15", bins=[(-20, -1), (0, 0), (1, 20)], controls=["x"], hac_lags=2)
    assert not out.empty
    assert {"term", "estimate", "se", "ci_low", "ci_high", "n"}.issubset(out.columns)
