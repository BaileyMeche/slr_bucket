from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_series_with_events(df: pd.DataFrame, y_col: str, events: list[str], title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df[y_col], lw=1.2)
    for e in events:
        ax.axvline(pd.Timestamp(e), color="red", linestyle="--", alpha=0.6)
    ax.set_title(title)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_event_paths(df: pd.DataFrame, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(df["term"], df["estimate"], yerr=1.96 * df["se"], fmt="o-")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticklabels(df["term"], rotation=45, ha="right")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
