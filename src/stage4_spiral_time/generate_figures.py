"""
Stage 4 Figure Generation – Spiral-Time Embedding
===================================================

Generates two figures using the Stage 4 operators:

1. figures/stage4_spiral_time/phi_chi_curves.png
   - Top subplot:    Adaptive phase φ(t) from compute_phi.py
   - Bottom subplot: Memory component χ(t) from compute_chi.py

2. figures/stage4_spiral_time/spiral_embedding.png
   - 3-D trajectory of (real_part, phi_part, chi_part) from the
     spiral-time operator D_Ψ evaluated on the ICE time series.

Reproducibility settings:
    seed = 0, DPI = 150, figure size as specified per figure below.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection

# ── Allow running from repository root or from this file's directory ──────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from src.core.models import ICEStateSeries
from src.stage4_spiral_time.compute_phi import compute_phi
from src.stage4_spiral_time.compute_chi import compute_chi
from src.stage4_spiral_time.spiral_operator import evaluate_spiral_operator

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 0
DPI = 150
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_PATH = os.path.join(_REPO_ROOT, "data", "synthetic", "eic_timeseries.csv")
_FIG_DIR = os.path.join(_REPO_ROOT, "figures", "stage4_spiral_time")


def _load_series(path: str) -> tuple[ICEStateSeries, np.ndarray]:
    """Load eic_timeseries.csv and return (ICEStateSeries, time array)."""
    df = pd.read_csv(path)
    t = df["time"].to_numpy(dtype=float)
    series = ICEStateSeries(
        delta_E=df["delta_E"].to_numpy(dtype=float),
        delta_I=df["delta_I"].to_numpy(dtype=float),
        delta_C=df["delta_C"].to_numpy(dtype=float),
    )
    return series, t


def plot_phi_chi_curves(
    t: np.ndarray,
    phi: np.ndarray,
    chi: np.ndarray,
    out_path: str,
) -> None:
    """Plot φ(t) and χ(t) on separate subplots and save."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, phi, color="#2563EB", linewidth=1.2, label=r"$\varphi(t)$")
    axes[0].set_ylabel(r"$\varphi(t)$", fontsize=12)
    axes[0].set_title("Adaptive Phase", fontsize=12)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    axes[1].plot(t, chi, color="#DC2626", linewidth=1.2, label=r"$\chi(t)$")
    axes[1].set_ylabel(r"$\chi(t)$", fontsize=12)
    axes[1].set_xlabel("Time $t$", fontsize=12)
    axes[1].set_title("Memory Component", fontsize=12)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.suptitle(r"Stage 4: $\varphi(t)$ and $\chi(t)$", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_spiral_embedding(
    result,
    out_path: str,
) -> None:
    """Plot the 3-D spiral-time operator embedding and save.

    The three axes correspond to the three algebraic components of D_Ψ f:
        x-axis – real_part  : temporal drift  d/dt f(γ(t))
        y-axis – phi_part   : phase-history   φ̇(t)·(v_φ·∇f)
        z-axis – chi_part   : memory contrib  χ̇(t)·(v_χ·∇f)

    Points are coloured by normalised time so the direction of travel
    along the spiral is immediately apparent.
    """
    t_norm = (result.t - result.t.min()) / (result.t.max() - result.t.min() + 1e-15)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        result.real_part,
        result.phi_part,
        result.chi_part,
        c=t_norm,
        cmap="viridis",
        s=6,
        alpha=0.8,
    )
    ax.plot(
        result.real_part,
        result.phi_part,
        result.chi_part,
        color="grey",
        linewidth=0.4,
        alpha=0.4,
    )

    ax.set_xlabel(r"$\partial_t f$ (real)", fontsize=10, labelpad=6)
    ax.set_ylabel(r"$\dot{\varphi}\,(v_\varphi\cdot\nabla f)$", fontsize=10, labelpad=6)
    ax.set_zlabel(r"$\dot{\chi}\,(v_\chi\cdot\nabla f)$", fontsize=10, labelpad=6)
    ax.set_title("Stage 4: Spiral-Time Operator Embedding", fontsize=12)

    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Normalised time", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    os.makedirs(_FIG_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    series, t = _load_series(_DATA_PATH)

    # ── 2. Compute φ(t) ───────────────────────────────────────────────────────
    phi = compute_phi(series)

    # ── 3. Compute χ(t) ───────────────────────────────────────────────────────
    chi = compute_chi(phi, t, kernel="exponential", lam=1.0)

    # ── 4–5. Plot and save φ(t) / χ(t) ───────────────────────────────────────
    plot_phi_chi_curves(
        t, phi, chi,
        out_path=os.path.join(_FIG_DIR, "phi_chi_curves.png"),
    )

    # ── 6. Compute spiral-time embedding ─────────────────────────────────────
    def f_observable(E: np.ndarray, I: np.ndarray, C: np.ndarray) -> np.ndarray:
        """L2-norm-squared of the ICE deviation vector."""
        return E ** 2 + I ** 2 + C ** 2

    result = evaluate_spiral_operator(series, f_observable, t, phi=phi, chi=chi)

    # ── 7–8. Plot and save embedding ─────────────────────────────────────────
    plot_spiral_embedding(
        result,
        out_path=os.path.join(_FIG_DIR, "spiral_embedding.png"),
    )


if __name__ == "__main__":
    main()
