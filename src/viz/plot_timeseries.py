"""
Reusable time-series plotting utilities.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def plot_raw_signal(
    t: np.ndarray,
    signal: np.ndarray,
    title: str = "Raw Signal",
    save_path: str | None = None,
) -> None:
    """Plot a single raw time series.

    Parameters
    ----------
    save_path : str or None
        If *None* the figure is not saved.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, signal, lw=0.5, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    plt.close(fig)


def plot_delta_phi(
    t: np.ndarray,
    delta_phi: np.ndarray,
    regime_boundaries: dict | None = None,
    title: str = "Instability Functional ΔΦ(t)",
    save_path: str | None = None,
) -> None:
    """Plot ΔΦ(t) with optional vertical regime-boundary lines.

    Parameters
    ----------
    regime_boundaries : dict or None
        Mapping ``{label: time_seconds}`` for vertical lines, e.g.
        ``{"transition": 60, "unstable": 120}``.
    save_path : str or None
        If *None* the figure is not saved.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, delta_phi, lw=0.7, color="crimson")
    if regime_boundaries:
        colors = ["orange", "red", "purple", "green"]
        for (label, tx), color in zip(regime_boundaries.items(), colors):
            ax.axvline(tx, color=color, lw=1.5, ls="--", label=f"{label}: {tx} s")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔΦ(t)")
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    plt.close(fig)
