from __future__ import annotations

from pathlib import Path

import pandas as pd

from slr_bucket.io import build_data_catalog, resolve_dataset_path


def test_resolve_dataset_path_prefers_parquet(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "sample.csv").write_text("date,x\n2020-01-01,1\n", encoding="utf-8")
    pd.DataFrame({"date": ["2020-01-01"], "x": [1]}).to_parquet(d / "sample.parquet", index=False)

    found = resolve_dataset_path("sample", expected_dir=d, fallback_roots=[d])
    assert found.suffix == ".parquet"


def test_build_data_catalog_has_join_hints(tmp_path: Path):
    d = tmp_path / "data" / "raw"
    d.mkdir(parents=True)
    (d / "x.csv").write_text("date,tenor,value\n2020-01-01,2,1.0\n2020-01-02,2,1.1\n", encoding="utf-8")

    catalog = build_data_catalog(tmp_path / "data")
    row = catalog.iloc[0]
    assert "join_hints" in catalog.columns
    assert "date" in str(row["key_columns"])
