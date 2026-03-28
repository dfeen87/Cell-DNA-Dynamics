"""
Reusable plotting utilities for ICE (Energy, Information, Coherence) features.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository root is two levels above this file (src/viz/).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def plot_ice_features(
    t: np.ndarray,
    E: np.ndarray,
    I: np.ndarray,
    C: np.ndarray,
    title: str = "ICE Features",
    save_path: str | None = None,
) -> None:
    """Plot normalized ICE feature trajectories (E, I, C) on a single axes.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    E, I, C : np.ndarray
        Normalized energy, information, and coherence arrays.
    title : str
        Figure title.
    save_path : str or None
        Absolute path to save the figure.  If *None* the figure is not saved.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, E, lw=0.8, label="E(t) – energy")
    ax.plot(t, I, lw=0.8, label="I(t) – information")
    ax.plot(t, C, lw=0.8, label="C(t) – coherence")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised value (z-score)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    plt.close(fig)


def plot_ice_panel(
    t: np.ndarray,
    E: np.ndarray,
    I: np.ndarray,
    C: np.ndarray,
    delta_phi: np.ndarray,
    stage: int = 1,
    save_path: str | None = None,
) -> None:
    """Two-panel figure: ICE features and ΔΦ(t).

    Parameters
    ----------
    t : np.ndarray
        Time vector aligned to *E*, *I*, *C*, and *delta_phi*.
    E, I, C : np.ndarray
        Normalised ICE feature arrays.
    delta_phi : np.ndarray
        Instability functional ΔΦ(t).
    stage : int
        Stage number, used to choose a default *save_path*.
    save_path : str or None
        Override save location; defaults to ``figures/stage{stage}/ice_panel.png``.
    """
    if save_path is None:
        save_path = os.path.join(
            _REPO_ROOT, "figures", f"stage{stage}", "ice_panel.png"
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t, E, lw=0.7, label="E(t)")
    axes[0].plot(t, I, lw=0.7, label="I(t)")
    axes[0].plot(t, C, lw=0.7, label="C(t)")
    axes[0].set_ylabel("Normalised value")
    axes[0].set_title("ICE Features")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(t, delta_phi, lw=0.7, color="crimson")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("ΔΦ(t)")
    axes[1].set_title("Instability Functional ΔΦ(t)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {save_path}")
    plt.close(fig)
