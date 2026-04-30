from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil

import pandas as pd

from .config import PipelineConfig, as_serializable_dict


def prepare_run_dirs(repo_root: Path, config: PipelineConfig) -> dict[str, Path]:
    run_dir = config.resolve_run_dir(repo_root)
    dirs = {
        "run": run_dir,
        "figures": run_dir / "figures",
        "tables": run_dir / "tables",
        "data": run_dir / "data",
        "logs": run_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def refresh_latest(repo_root: Path, config: PipelineConfig, run_dir: Path) -> Path:
    latest = repo_root / config.output_root / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)
    return latest


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


def write_run_readme(run_dir: Path, config: PipelineConfig, notes: str) -> None:
    payload = as_serializable_dict(config)
    text = "# Summary pipeline run\n\n"
    text += "## Config\n\n```json\n" + json.dumps(payload, indent=2) + "\n```\n\n"
    text += "## Notes\n\n" + notes + "\n"
    (run_dir / "README.md").write_text(text, encoding="utf-8")


def write_catalog_outputs(catalog: pd.DataFrame, out_data_dir: Path) -> None:
    out_data_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_data_dir / "data_catalog.csv", index=False)
    catalog.to_parquet(out_data_dir / "data_catalog.parquet", index=False)
    catalog.to_markdown(out_data_dir / "data_catalog.md", index=False)
