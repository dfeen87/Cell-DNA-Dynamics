"""
Stage II Implementation – Cell-DNA Dynamics Regime Segmentation
================================================================
Sections:
  1. Data Loading
  2. Segmentation (smoothing + quantile thresholds)
  3. CSV Output
  4. Visualization (3-panel figure with regime shading)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_CSV   = os.path.join(REPO_ROOT, "data", "stage1", "stage_i_results.csv")
OUTPUT_CSV  = os.path.join(REPO_ROOT, "data", "stage2", "stage_ii_segmentation.csv")
OUTPUT_PNG  = os.path.join(REPO_ROOT, "figures", "stage2", "stage_ii_regimes.png")

# Savitzky–Golay smoothing parameters
SG_WINDOW  = 251   # must be odd and < number of data points
SG_POLYORD = 3

# Quantile thresholds
Q_LOW  = 0.25
Q_HIGH = 0.75

# Regime shading colours
REGIME_COLORS = {
    "stable":            "#2196F3",   # blue
    "pre-instability":   "#FF9800",   # orange
    "instability":       "#F44336",   # red
}
REGIME_ALPHA = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str = INPUT_CSV) -> pd.DataFrame:
    """Load stage_i_results.csv and return a DataFrame."""
    df = pd.read_csv(path)
    # Rename columns to the standard Stage-II names
    df = df.rename(columns={
        "time":      "t",
        "delta_phi": "DeltaPhi",
        "E":         "E",
        "I":         "I",
        "C":         "C",
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smooth ΔΦ(t) with a Savitzky–Golay filter, compute 0.25 / 0.75 quantile
    thresholds, and assign each time-point to one of three regimes:
      - stable          : smoothed ΔΦ < Q25
      - pre_instability : Q25 ≤ smoothed ΔΦ ≤ Q75
      - instability     : smoothed ΔΦ > Q75
    """
    raw_dp = df["DeltaPhi"].values
    n = len(raw_dp)

    # Validate Savitzky–Golay parameters before calling the filter
    win = SG_WINDOW if (SG_WINDOW % 2 == 1) else SG_WINDOW + 1   # must be odd
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2               # cap to data length
    if win < SG_POLYORD + 2:
        raise ValueError(
            f"SG_WINDOW ({win}) is too small for SG_POLYORD ({SG_POLYORD}). "
            "Increase SG_WINDOW or decrease SG_POLYORD."
        )

    # Smooth ΔΦ(t)
    smooth_dp = savgol_filter(raw_dp, window_length=win, polyorder=SG_POLYORD)

    # Quantile thresholds computed from the smoothed signal
    q_low_val  = np.quantile(smooth_dp, Q_LOW)
    q_high_val = np.quantile(smooth_dp, Q_HIGH)

    # Assign regimes
    regimes = np.where(
        smooth_dp > q_high_val, "instability",
        np.where(smooth_dp < q_low_val, "stable", "pre-instability")
    )

    result = df[["t", "DeltaPhi", "E", "I", "C"]].copy()
    result.insert(1, "regime", regimes)
    result["smooth_DeltaPhi"] = smooth_dp   # kept internally for plotting

    return result, q_low_val, q_high_val


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def save_results(result: pd.DataFrame, path: str = OUTPUT_CSV) -> None:
    """Save the segmentation table (without the internal smooth column)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out_cols = ["t", "regime", "DeltaPhi", "E", "I", "C"]
    result[out_cols].to_csv(path, index=False)
    print(f"CSV saved  → {path}  ({len(result)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT REGIMES
# ─────────────────────────────────────────────────────────────────────────────
def _shade_regimes(ax, t: np.ndarray, regimes: np.ndarray) -> None:
    """Fill horizontal bands for each contiguous regime block."""
    regime_list = list(regimes)
    n = len(regime_list)
    dt = (t[-1] - t[0]) / (n - 1) if n > 1 else 0   # time step
    i = 0
    while i < n:
        current = regime_list[i]
        j = i
        while j < n and regime_list[j] == current:
            j += 1
        # Extend end by half a time-step so the final sample is fully covered
        x_end = t[j - 1] + dt / 2 if j < n else t[j - 1] + dt / 2
        ax.axvspan(t[i], x_end, alpha=REGIME_ALPHA,
                   color=REGIME_COLORS[current], lw=0)
        i = j


def plot_regimes(result: pd.DataFrame,
                 q_low_val: float,
                 q_high_val: float,
                 path: str = OUTPUT_PNG) -> None:
    """
    3-panel figure:
      Panel 1 – raw ΔΦ(t) with regime shading
      Panel 2 – features E(t), I(t), C(t) with regime shading
      Panel 3 – smoothed ΔΦ(t) with regime shading and threshold lines
    """
    t       = result["t"].values
    regimes = result["regime"].values
    raw_dp  = result["DeltaPhi"].values
    smooth_dp = result["smooth_DeltaPhi"].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ── Panel 1: Raw ΔΦ(t) ──────────────────────────────────────────────────
    ax = axes[0]
    _shade_regimes(ax, t, regimes)
    ax.plot(t, raw_dp, lw=0.5, color="crimson", label="ΔΦ(t) raw")
    ax.set_ylabel("ΔΦ(t)")
    ax.set_title("Raw Instability Functional ΔΦ(t) with Regime Shading")
    ax.legend(loc="upper left", fontsize=8)

    # ── Panel 2: Features ───────────────────────────────────────────────────
    ax = axes[1]
    _shade_regimes(ax, t, regimes)
    ax.plot(t, result["E"].values, lw=0.5, label="E(t)")
    ax.plot(t, result["I"].values, lw=0.5, label="I(t)")
    ax.plot(t, result["C"].values, lw=0.5, label="C(t)")
    ax.set_ylabel("Feature value")
    ax.set_title("Features E(t), I(t), C(t) with Regime Shading")
    ax.legend(loc="upper left", fontsize=8)

    # ── Panel 3: Smoothed ΔΦ(t) ─────────────────────────────────────────────
    ax = axes[2]
    _shade_regimes(ax, t, regimes)
    ax.plot(t, smooth_dp, lw=0.8, color="darkviolet", label="ΔΦ(t) smoothed")
    ax.axhline(q_low_val,  color="orange", lw=1.2, ls="--",
               label=f"Q{int(Q_LOW*100)} = {q_low_val:.3f}")
    ax.axhline(q_high_val, color="red",    lw=1.2, ls="--",
               label=f"Q{int(Q_HIGH*100)} = {q_high_val:.3f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔΦ(t) smoothed")
    ax.set_title("Smoothed ΔΦ(t) with Quantile Thresholds and Regime Shading")
    ax.legend(loc="upper left", fontsize=8)

    # ── Shared legend patch for regimes ─────────────────────────────────────
    from matplotlib.patches import Patch
    regime_patches = [
        Patch(facecolor=REGIME_COLORS["stable"],
              alpha=0.4, label="Stable"),
        Patch(facecolor=REGIME_COLORS["pre_instability"],
              alpha=0.4, label="Pre-instability"),
        Patch(facecolor=REGIME_COLORS["instability"],
              alpha=0.4, label="Instability"),
    ]
    fig.legend(handles=regime_patches, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    result, q_low_val, q_high_val = compute_segmentation(df)
    save_results(result)
    plot_regimes(result, q_low_val, q_high_val)

    # Print regime distribution summary
    counts = result["regime"].value_counts()
    print("\n── Regime Distribution ──")
    for regime in ["stable", "pre-instability", "instability"]:
        n = counts.get(regime, 0)
        pct = 100 * n / len(result)
        print(f"  {regime:<20s}: {n:5d} points  ({pct:.1f} %)")


if __name__ == "__main__":
    main()
