#!/usr/bin/env bash
set -euo pipefail

# Run from repo root (so notebooks resolve repo_root=Path.cwd().parent correctly when executed in notebooks/).
# Usage:
#   bash scripts/run_new_series_notebooks.sh
#
# Requirements:
#   - jupyter (nbconvert) installed
#   - python kernel available
#
# Outputs:
#   outputs/summary_pipeline/<timestamp>_<cfg_hash>/{tables,figures,data,logs}

NOTEBOOK_DIR="notebooks"

NOTEBOOKS=(
  "summary_pipeline_treasury_sf.ipynb"
  "summary_pipeline_cip_3m.ipynb"
  "summary_pipeline_equity_INDU.ipynb"
  "summary_pipeline_equity_NDX.ipynb"
  "summary_pipeline_equity_SPY.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
  echo "Running ${nb} ..."
  jupyter nbconvert --execute --to notebook --inplace "${NOTEBOOK_DIR}/${nb}"
done

echo "Done. See outputs/summary_pipeline/ for run artifacts."
