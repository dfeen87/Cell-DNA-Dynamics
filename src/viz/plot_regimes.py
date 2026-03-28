"""
Reusable regime-shading plotting utilities.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REGIME_COLORS = {
    "stable":           "#2196F3",
    "pre-instability":  "#FF9800",
    "instability":      "#F44336",
    "recovery":         "#4CAF50",
}


def shade_regimes(ax, t: np.ndarray, regimes: np.ndarray, alpha: float = 0.15) -> None:
    """Fill contiguous regime spans on *ax*."""
    regime_list = list(regimes)
    n = len(regime_list)
    dt = (t[-1] - t[0]) / (n - 1) if n > 1 else 0
    i = 0
    while i < n:
        current = regime_list[i]
        j = i
        while j < n and regime_list[j] == current:
            j += 1
        x_end = t[j - 1] + dt / 2
        color = REGIME_COLORS.get(current, "#CCCCCC")
        ax.axvspan(t[i], x_end, alpha=alpha, color=color, lw=0)
        i = j


def plot_regime_overview(
    t: np.ndarray,
    delta_phi: np.ndarray,
    regimes: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Single-panel ΔΦ(t) plot with regime shading.

    Parameters
    ----------
    save_path : str or None
        Defaults to ``figures/stage5_regime_transitions/regime_overview.png``.
    """
    if save_path is None:
        save_path = os.path.join(
            _REPO_ROOT, "figures", "stage5_regime_transitions", "regime_overview.png"
        )

    fig, ax = plt.subplots(figsize=(12, 4))
    shade_regimes(ax, t, regimes)
    ax.plot(t, delta_phi, lw=0.8, color="crimson", label="ΔΦ(t)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔΦ(t)")
    ax.set_title("ΔΦ(t) with Regime Shading")

    patches = [Patch(facecolor=REGIME_COLORS[r], alpha=0.5, label=r.capitalize())
               for r in ["stable", "pre-instability", "instability", "recovery"]]
    ax.legend(handles=patches + [ax.lines[0]], loc="upper left", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {save_path}")
    plt.close(fig)
