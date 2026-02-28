from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = [
    "summary_pipeline_treasury_sf.ipynb",
    "summary_pipeline_cip_3m.ipynb",
    "summary_pipeline_equity_INDU.ipynb",
    "summary_pipeline_equity_NDX.ipynb",
    "summary_pipeline_equity_SPY.ipynb",
]

def run_one(nb_path: Path, workdir: Path) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(workdir)}})
    nbformat.write(nb, nb_path)  # in-place like nbconvert --inplace

def main() -> int:
    repo_root = Path.cwd()
    nb_dir = repo_root / "notebooks"
    if not nb_dir.exists():
        print(f"ERROR: expected notebooks/ directory at {nb_dir}")
        return 2
    for nb_name in NOTEBOOKS:
        nb_path = nb_dir / nb_name
        if not nb_path.exists():
            print(f"ERROR: missing notebook: {nb_path}")
            return 2
        print(f"Running {nb_name} ...")
        run_one(nb_path, nb_dir)
    print("Done. See outputs/summary_pipeline/ for run artifacts.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
