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


def plot_binned_event_overlay(
    df: pd.DataFrame,
    title: str,
    outpath: Path,
    x_col: str = "bin_mid",
    kind_col: str = "kind",
) -> None:
    """Plot binned event-study paths.

    Expects rows for at least one of: baseline_bin, interaction_bin, group0_effect, group1_effect.
    Uses x_col as numeric x (bin midpoint).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    dfp = df.copy()
    dfp = dfp[pd.to_numeric(dfp[x_col], errors="coerce").notna()].copy()
    dfp[x_col] = pd.to_numeric(dfp[x_col], errors="coerce")
    for kind in ["group0_effect", "group1_effect", "baseline_bin", "interaction_bin"]:
        sub = dfp[dfp[kind_col] == kind].sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(sub[x_col], sub["estimate"], marker="o", label=kind)
        ax.fill_between(sub[x_col], sub["ci_low"], sub["ci_high"], alpha=0.15)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Event time (bin midpoint)")
    ax.set_ylabel("Effect (bps)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
