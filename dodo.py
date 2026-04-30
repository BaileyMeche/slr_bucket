"""
dodo.py — doit task automation for the SLR Bucket project.

Usage (run from the project root):
    doit              # pipeline + figures + tests  (default)
    doit pipeline     # run the analysis pipeline only
    doit figures      # regenerate publication figures only
    doit tests        # run the test suite only
    doit list         # show tasks with descriptions
    doit clean        # delete all generated outputs (_output/)

Wrappers in this directory:

  Windows cmd / PowerShell:
    doit              (doit.bat)
    doit clean

  Git Bash (MINGW64):
    ./doit
    ./doit clean

Or invoke Python explicitly (with the conda env activated):
    python -m doit [task]

CONDA ENVIRONMENT
  The project uses a dedicated conda environment.  By default dodo.py
  looks for an environment named  slr_bucket  (created from
  environment.yml).  If not found it also tries the legacy name  risk .
  You can override with the DOIT_CONDA_ENV environment variable:

      set DOIT_CONDA_ENV=my_env   # Windows cmd
      export DOIT_CONDA_ENV=my_env  # bash/zsh

  After activating the environment you can also just run:
      python -m doit [task]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Default tasks run by plain `doit`
DOIT_CONFIG = {"default_tasks": ["pipeline", "figures", "tests"]}

ROOT = Path(__file__).parent

# Force UTF-8 I/O so Windows cp1252 doesn't choke on significance stars (***)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# ---------------------------------------------------------------------------
# Python interpreter discovery
#
# Priority:
#   1. DOIT_CONDA_ENV env-var → explicit override
#   2. Search for  slr_bucket  env in common Anaconda/Miniconda locations
#   3. Search for legacy  risk  env (backward compatibility)
#   4. sys.executable if already inside a matching env  (activated env)
# ---------------------------------------------------------------------------

def _find_env_python(env_name: str) -> Path | None:
    """Return the python.exe path for a named conda env, or None."""
    # Common conda base directories (system-wide and per-user)
    home = Path.home()
    candidates = [
        Path(r"C:\ProgramData\anaconda3")     / "envs" / env_name / "python.exe",
        Path(r"C:\ProgramData\miniconda3")    / "envs" / env_name / "python.exe",
        home / "anaconda3"  / "envs" / env_name / "python.exe",
        home / "miniconda3" / "envs" / env_name / "python.exe",
        home / "AppData" / "Local" / "anaconda3"  / "envs" / env_name / "python.exe",
        home / "AppData" / "Local" / "miniconda3" / "envs" / env_name / "python.exe",
        home / "AppData" / "Local" / "Continuum" / "anaconda3" / "envs" / env_name / "python.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _conda_run_python(env_name: str) -> Path | None:
    """Ask conda itself where the named env lives."""
    conda_exe = shutil.which("conda")
    if not conda_exe:
        return None
    try:
        result = subprocess.run(
            [conda_exe, "run", "-n", env_name, "python", "-c",
             "import sys; print(sys.executable)"],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0:
            p = Path(result.stdout.strip())
            if p.exists():
                return p
    except Exception:
        pass
    return None


def _resolve_python() -> str:
    """Locate the project Python, with fallback chain."""
    # 1. Explicit override via environment variable
    override_env = os.environ.get("DOIT_CONDA_ENV", "").strip()
    if override_env:
        for finder in (_find_env_python, _conda_run_python):
            p = finder(override_env)
            if p:
                return str(p)
        print(f"[dodo] WARNING: DOIT_CONDA_ENV={override_env!r} not found; "
              f"falling back to sys.executable")

    # 2. slr_bucket env (canonical project name)
    for env in ("slr_bucket", "risk"):
        p = _find_env_python(env)
        if p:
            return str(p)

    # 3. conda run fallback (works even if env is not in standard location)
    for env in ("slr_bucket", "risk"):
        p = _conda_run_python(env)
        if p:
            return str(p)

    # 4. Current interpreter (works if the user ran `conda activate slr_bucket`)
    cur = Path(sys.executable)
    if cur.exists():
        print(f"[dodo] INFO: Using current Python: {cur}")
        print("[dodo] TIP: Create the project env with:  conda env create -f environment.yml")
        return str(cur)

    raise RuntimeError(
        "Could not locate project Python.\n"
        "Create the environment:  conda env create -f environment.yml\n"
        "Then activate it:        conda activate slr_bucket\n"
        "Or set:                  set DOIT_CONDA_ENV=<env_name>"
    )


PY = _resolve_python()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PIPELINE_SCRIPT = ROOT / "src" / "slr_bucket" / "pipeline" / "main.py"
FIGURES_SCRIPT  = ROOT / "src" / "slr_bucket" / "figures"  / "make_figures.py"

# All pipeline source files — changes here invalidate pipeline outputs
PIPELINE_SRCS = [
    ROOT / "src" / "slr_bucket" / "pipeline" / f
    for f in (
        "main.py",
        "loader.py",
        "windows.py",
        "hac.py",
        "winsorize.py",
        "issuance.py",
        "did.py",
        "results.py",
    )
]

# Raw data files the pipeline reads
DATA_DEPS = [
    ROOT / "data" / "series" / "cip_spreads_3m_bps.csv",
    ROOT / "data" / "series" / "treasury_sf_output.csv",
    ROOT / "data" / "series" / "tips_treasury_implied_rf_2010.parquet",
    ROOT / "data" / "raw" / "event_inputs" / "controls_vix_creditspreads_fred.csv",
    ROOT / "data" / "raw" / "event_inputs" / "repo_rates_combined.csv",
    ROOT / "data" / "raw" / "event_inputs" / "treasury_issuance_by_tenor_fiscaldata.csv",
]

PIPELINE_TARGETS = [
    ROOT / "_output" / "pipeline" / f
    for f in (
        "jump_estimates_pooled.csv",
        "jump_estimates_by_series.csv",
        "did_clean_baseline.csv",
        "key_results_formatted.txt",
        "regression_table_v2.tex",
        "robustness_table.csv",
        "summary_stats_repaired.csv",
        "winsorize_log.csv",
    )
]

FIGURE_TARGETS = [
    ROOT / "_output" / "figures" / f
    for f in (
        "fig1_series_overview.png",
        "fig1_series_overview.pdf",
        "fig2_ust_sf_detail.png",
        "fig2_ust_sf_detail.pdf",
        "fig3_exit_event_paths.png",
        "fig3_exit_event_paths.pdf",
        "fig4_coef_plot.png",
        "fig4_coef_plot.pdf",
        "fig5_series_level_coefs.png",
        "fig5_series_level_coefs.pdf",
        "fig6_regime_boxplots.png",
        "fig6_regime_boxplots.pdf",
    )
]

# The single generated output directory — deleted entirely by doit clean
BUILD_DIR = ROOT / "_output"


# ---------------------------------------------------------------------------
# Clean helpers
# ---------------------------------------------------------------------------

def _clean_all():
    """
    Remove the entire _output/ directory.

    Uses a retry loop to handle transient PermissionErrors on Windows/OneDrive
    (OneDrive may hold a brief lock on newly-written directories).
    """
    import time, stat

    if not BUILD_DIR.exists():
        print(f"  Nothing to clean ({BUILD_DIR.name}/ does not exist)")
        return

    def _handle_error(func, path, exc_info):
        """On PermissionError, try to clear read-only bit and retry."""
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass  # best-effort; leftover files are non-fatal

    # First attempt
    shutil.rmtree(BUILD_DIR, onerror=_handle_error)

    # If OneDrive still holds a lock, wait briefly and try once more
    if BUILD_DIR.exists():
        time.sleep(1)
        shutil.rmtree(BUILD_DIR, ignore_errors=True)

    if BUILD_DIR.exists():
        print(f"  WARNING: {BUILD_DIR.name}/ could not be fully removed "
              f"(files may be locked by OneDrive or another process).")
    else:
        print(f"  Removed {BUILD_DIR.name}/")


# ---------------------------------------------------------------------------
# Task: pipeline
# ---------------------------------------------------------------------------

def task_pipeline():
    """Run the SLR event study pipeline (produces all CSV/TEX estimates)."""
    return {
        "file_dep": [str(f) for f in PIPELINE_SRCS + DATA_DEPS],
        "targets":  [str(t) for t in PIPELINE_TARGETS],
        "actions":  [f'"{PY}" "{PIPELINE_SCRIPT}"'],
        "verbosity": 2,
        "clean":    [_clean_all],
    }


# ---------------------------------------------------------------------------
# Task: figures
# ---------------------------------------------------------------------------

def task_figures():
    """Generate all publication figures (fig1-fig6, PNG + PDF)."""
    figure_deps = [
        ROOT / "_output" / "pipeline" / "jump_estimates_pooled.csv",
        ROOT / "_output" / "pipeline" / "jump_estimates_by_series.csv",
        FIGURES_SCRIPT,
    ]
    return {
        "task_dep": ["pipeline"],
        "file_dep": [str(f) for f in figure_deps],
        "targets":  [str(t) for t in FIGURE_TARGETS],
        "actions":  [f'"{PY}" "{FIGURES_SCRIPT}"'],
        "verbosity": 2,
        "clean":    True,
    }


# ---------------------------------------------------------------------------
# Task: tests
# ---------------------------------------------------------------------------

def task_tests():
    """Run pytest suite (tests/)."""
    test_files = list((ROOT / "tests").glob("test_*.py"))
    src_files  = list((ROOT / "src").rglob("*.py"))
    return {
        "file_dep": [str(f) for f in test_files + src_files],
        "uptodate": [False],
        "actions":  [f'"{PY}" -m pytest "{ROOT / "tests"}" -v --tb=short'],
        "verbosity": 2,
    }
