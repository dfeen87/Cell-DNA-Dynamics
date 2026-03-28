"""
Reusable plotting utilities for null-test / surrogate comparisons.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def plot_null_comparison(
    t: np.ndarray,
    observed: np.ndarray,
    null_dict: dict[str, np.ndarray],
    title: str = "ΔΦ(t): Observed vs Null Controls",
    save_path: str | None = None,
) -> None:
    """Overlay observed ΔΦ(t) with one or more null-control curves.

    Parameters
    ----------
    t : np.ndarray
        Common time vector.
    observed : np.ndarray
        ΔΦ(t) from the original signal.
    null_dict : dict
        Mapping ``{label: delta_phi_null}`` for each control condition.
    save_path : str or None
        Defaults to ``figures/simulations/null_comparison.png``.
    """
    if save_path is None:
        save_path = os.path.join(_REPO_ROOT, "figures", "simulations", "null_comparison.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, observed, lw=1.0, color="crimson", label="Observed ΔΦ(t)", zorder=3)
    colors = ["steelblue", "darkorange", "green", "purple"]
    for (label, arr), color in zip(null_dict.items(), colors):
        nt = t[:len(arr)]
        ax.plot(nt, arr, lw=0.7, color=color, alpha=0.75, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔΦ(t)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {save_path}")
    plt.close(fig)


def plot_null_distributions(
    observed_means: dict[str, float],
    null_means: dict[str, list[float]],
    save_path: str | None = None,
) -> None:
    """Box/strip plot comparing observed regime means with null distributions.

    Parameters
    ----------
    observed_means : dict
        ``{regime: mean_delta_phi}`` for the observed signal.
    null_means : dict
        ``{regime: [mean_delta_phi, ...]}`` across null samples.
    save_path : str or None
        Defaults to ``figures/simulations/null_distributions.png``.
    """
    if save_path is None:
        save_path = os.path.join(_REPO_ROOT, "figures", "simulations", "null_distributions.png")

    regimes = list(null_means.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(regimes))
    ax.boxplot([null_means[r] for r in regimes], positions=list(positions), widths=0.4)
    for pos, r in enumerate(regimes):
        if r in observed_means:
            ax.scatter(pos + 1, observed_means[r], color="red", zorder=5,
                       s=60, label="Observed" if pos == 0 else "")
    ax.set_xticks(range(1, len(regimes) + 1))
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.set_ylabel("Mean ΔΦ(t)")
    ax.set_title("Null-test ΔΦ distributions vs Observed")
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {save_path}")
    plt.close(fig)
