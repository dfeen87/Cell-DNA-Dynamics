"""
dna_pipeline – Reproducible pipeline for DNA-associated dynamics.

This module implements the end-to-end analysis pipeline for DNA-associated
time series as described in Section 7 of the manuscript.  It is an *extension
layer* built on top of the core triadic stability framework and must not be
interpreted as part of the core manuscript.  No biological claims are made.

Pipeline steps
--------------
1. **Load**     – Accept raw DNA-associated channel arrays (caller-supplied).
2. **Extract**  – Compute per-feature time series from raw channels via
                  :func:`~src.extensions.stage7_dna.dna_feature_mapping.map_dna_to_ice`.
3. **Map → ICE** – Normalise features against a reference baseline and produce
                   an :class:`~src.core.models.ICEStateSeries` via
                   :func:`~src.extensions.stage7_dna.dna_feature_mapping.normalize_dna`.
4. **ΔΦ(t)**   – Compute the instability magnitude (Eq. 2 / Eq. 20) via
                  :func:`~src.stage5_regime_transitions.compute_delta_phi.compute_delta_phi`.
5. **Regimes**  – Segment ΔΦ(t) into stable / pre-instability / instability /
                  recovery labels via
                  :func:`~src.stage5_regime_transitions.detect_regimes.detect_regimes`.
6. **Outputs**  – Produce a manuscript-style 4-panel figure and a CSV summary.

The pipeline is a structural analogue of
:mod:`src.extensions.stage6_cellular.cellular_pipeline` adapted for the
DNA-associated observable types listed in Table 2 and Section 7 of the
manuscript.

Usage example
-------------
::

    import numpy as np
    from src.extensions.stage7_dna.dna_feature_mapping import DNAFeatureSpec
    from src.extensions.stage7_dna.dna_pipeline import (
        DNAPipeline,
        DNAPipelineConfig,
    )

    rng = np.random.default_rng(0)
    T = 9000
    raw_data = {
        "fret_eff":   rng.uniform(0.2, 0.8, T),
        "burst":      rng.exponential(1.0, T),
        "locus_x1":  np.cumsum(rng.normal(0, 0.05, T)),
        "locus_x2":  np.cumsum(rng.normal(0, 0.05, T)),
    }
    specs = [
        DNAFeatureSpec("fret_rms",  "E", "fret_energy_proxy",  ["fret_eff"],           window_size=400),
        DNAFeatureSpec("burst_cv",  "I", "burst_variability",  ["burst"],              window_size=400),
        DNAFeatureSpec("co_motion", "C", "locus_co_motion",    ["locus_x1","locus_x2"],window_size=400),
    ]
    cfg = DNAPipelineConfig(
        feature_specs=specs,
        baseline_window=(0, 3000),
        fs=50.0,
        compute_spiral=True,
    )
    pipeline = DNAPipeline(cfg)
    result = pipeline.run(raw_data)
    pipeline.save_figure(result, "outputs/dna_figure.png")
    pipeline.save_csv(result, "outputs/dna_summary.csv")
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the repository root is importable regardless of the working directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.models import ICEStateSeries, RegimeLabels  # noqa: E402
from src.extensions.stage7_dna.dna_feature_mapping import (  # noqa: E402
    DNAFeatureSpec,
    map_dna_to_ice,
    normalize_dna,
)
from src.stage4_spiral_time.compute_chi import compute_chi  # noqa: E402
from src.stage4_spiral_time.compute_phi import compute_phi  # noqa: E402
from src.stage5_regime_transitions.compute_delta_phi import compute_delta_phi  # noqa: E402
from src.stage5_regime_transitions.detect_regimes import detect_regimes  # noqa: E402


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class DNAPipelineConfig:
    """Configuration for :class:`DNAPipeline`.

    Parameters
    ----------
    feature_specs:
        List of :class:`~src.extensions.stage7_dna.dna_feature_mapping.DNAFeatureSpec`
        objects declaring how each raw channel maps to an ICE component.
        All three components (E, I, C) must be covered.
    baseline_window:
        ``(start, stop)`` *sample* indices (half-open) defining the reference
        baseline period.  Used for z-score normalisation.  Must satisfy
        ``0 <= start < stop <= T``.
    fs:
        Sampling frequency in Hz.  Used to build the time axis.  Defaults to
        ``1.0`` (unit-sample grid).
    compute_spiral:
        Whether to compute the optional spiral-time embedding (φ(t) and χ(t)).
        Defaults to ``False``.
    q_low:
        Lower quantile threshold for regime segmentation (passed to
        :func:`~src.stage5_regime_transitions.detect_regimes.detect_regimes`).
        Defaults to ``0.75``.
    q_high:
        Upper quantile threshold for regime segmentation.  Defaults to
        ``0.90``.
    detect_recovery:
        Whether to apply optional recovery-label post-processing in
        :func:`~src.stage5_regime_transitions.detect_regimes.detect_regimes`.
        Defaults to ``True``.
    min_std:
        Lower bound on standard deviation used in z-score normalisation.
        Defaults to ``1e-8``.
    kernel:
        Memory kernel for χ(t) computation.  Passed to
        :func:`~src.stage4_spiral_time.compute_chi.compute_chi`.
        Accepts ``"exponential"`` (default) or ``"gaussian"``.
    kernel_kwargs:
        Additional keyword arguments forwarded to the kernel function.
        E.g. ``{"lam": 0.5}`` for the exponential kernel.
    """

    feature_specs: list[DNAFeatureSpec]
    baseline_window: tuple[int, int]
    fs: float = 1.0
    compute_spiral: bool = False
    q_low: float = 0.75
    q_high: float = 0.90
    detect_recovery: bool = True
    min_std: float = 1e-8
    kernel: str = "exponential"
    kernel_kwargs: dict = field(default_factory=dict)


@dataclass
class DNAPipelineResult:
    """All outputs produced by :class:`DNAPipeline`.

    Attributes
    ----------
    t:
        Time axis in seconds, shape ``(T,)``.
    ice_series:
        Normalised ICE deviation series (ΔE, ΔI, ΔC).
    delta_phi:
        Instability magnitude ΔΦ(t), shape ``(T,)``.
    regimes:
        Per-time-point regime labels together with the derived thresholds.
    phi:
        Adaptive phase φ(t), shape ``(T,)``.  ``None`` when
        ``compute_spiral=False``.
    chi:
        Memory component χ(t), shape ``(T,)``.  ``None`` when
        ``compute_spiral=False``.
    regime_stats:
        Dict mapping regime label → mean ΔΦ over that regime's time points.
    """

    t: np.ndarray
    ice_series: ICEStateSeries
    delta_phi: np.ndarray
    regimes: RegimeLabels
    phi: np.ndarray | None
    chi: np.ndarray | None
    regime_stats: dict[str, float]


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class DNAPipeline:
    """Reproducible pipeline for DNA-associated time series.

    Ties together the ICE mapping, normalisation, instability-magnitude
    computation, regime segmentation, and optional spiral-time embedding
    defined for the DNA-associated extension (Section 7 of the manuscript).

    This class is an extension layer.  It does not introduce biological claims
    or modify the core manuscript modules.

    Parameters
    ----------
    config:
        A :class:`DNAPipelineConfig` instance specifying all run parameters.
    """

    def __init__(self, config: DNAPipelineConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(self, raw_data: dict[str, np.ndarray]) -> DNAPipelineResult:
        """Execute all pipeline steps and return a :class:`DNAPipelineResult`.

        Parameters
        ----------
        raw_data:
            Dictionary mapping channel names to 1-D NumPy arrays of equal
            length *T*.  Channel names must match those referenced in the
            :class:`~src.extensions.stage7_dna.dna_feature_mapping.DNAFeatureSpec`
            objects supplied in the configuration.

        Returns
        -------
        DNAPipelineResult
            Container with the outputs from every pipeline step.

        Raises
        ------
        ValueError
            If any pipeline step receives invalid inputs (propagated from the
            underlying modules).
        """
        cfg = self._cfg

        # ------------------------------------------------------------------ #
        # Step 1: Validate and echo input dimensions                          #
        # ------------------------------------------------------------------ #
        if not raw_data:
            raise ValueError("raw_data must not be empty.")
        lengths = {name: len(arr) for name, arr in raw_data.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                f"All channels in raw_data must have the same length; "
                f"found lengths: {lengths}."
            )
        T = next(iter(unique_lengths))

        # Time axis in seconds
        t = np.arange(T) / float(cfg.fs)

        # ------------------------------------------------------------------ #
        # Step 2: Extract features and map raw channels to ICE components     #
        # ------------------------------------------------------------------ #
        observables, mapping = map_dna_to_ice(raw_data, cfg.feature_specs)

        # ------------------------------------------------------------------ #
        # Step 3: Normalise → ICE                                             #
        # ------------------------------------------------------------------ #
        ice_series = normalize_dna(
            observables,
            mapping,
            cfg.baseline_window,
            min_std=cfg.min_std,
        )

        # NaN values arise in the rolling-window warm-up period (the first
        # window_size − 1 samples of each feature are undefined).  Replace
        # them with 0.0 so that the warm-up period is treated as coinciding
        # with the reference state x* = (0, 0, 0) and does not propagate
        # through downstream computations.
        ice_series = ICEStateSeries(
            delta_E=np.nan_to_num(ice_series.delta_E, nan=0.0),
            delta_I=np.nan_to_num(ice_series.delta_I, nan=0.0),
            delta_C=np.nan_to_num(ice_series.delta_C, nan=0.0),
        )

        # ------------------------------------------------------------------ #
        # Step 4: Compute ΔΦ(t)                                              #
        # ------------------------------------------------------------------ #
        delta_phi = compute_delta_phi(ice_series)

        # ------------------------------------------------------------------ #
        # Step 5: Segment regimes                                             #
        # ------------------------------------------------------------------ #
        regimes = detect_regimes(
            delta_phi,
            q_low=cfg.q_low,
            q_high=cfg.q_high,
            detect_recovery=cfg.detect_recovery,
        )

        # ------------------------------------------------------------------ #
        # Step 6 (optional): Spiral-time embedding                           #
        # ------------------------------------------------------------------ #
        phi_t: np.ndarray | None = None
        chi_t: np.ndarray | None = None

        if cfg.compute_spiral:
            phi_t = compute_phi(ice_series)
            chi_t = compute_chi(phi_t, t, kernel=cfg.kernel, **cfg.kernel_kwargs)

        # ------------------------------------------------------------------ #
        # Regime statistics (mean ΔΦ per regime)                             #
        # ------------------------------------------------------------------ #
        regime_stats = _compute_regime_stats(delta_phi, regimes)

        return DNAPipelineResult(
            t=t,
            ice_series=ice_series,
            delta_phi=delta_phi,
            regimes=regimes,
            phi=phi_t,
            chi=chi_t,
            regime_stats=regime_stats,
        )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def save_figure(
        self,
        result: DNAPipelineResult,
        path: str | Path,
        *,
        dpi: int = 300,
        regime_boundaries: Sequence[float] | None = None,
    ) -> None:
        """Save a manuscript-style 4-panel figure for the DNA pipeline result.

        Panel layout:

        1. Normalised ICE components ΔE(t), ΔI(t), ΔC(t).
        2. Instability magnitude ΔΦ(t) with regime-threshold markers.
        3. Per-time-point regime labels (colour-coded).
        4. *(When spiral was computed)* φ(t) and χ(t); otherwise a
           Savitzky–Golay smoothed ΔΦ(t) for visual reference.

        Parameters
        ----------
        result:
            Output from :meth:`run`.
        path:
            File path for the saved figure.  The format is inferred from the
            extension (e.g. ``.png``, ``.pdf``).
        dpi:
            Dots-per-inch for raster formats.  Defaults to 300.
        regime_boundaries:
            Optional list of time values (in seconds) at which vertical
            dashed lines are drawn on all panels to mark known regime
            transition times.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        t = result.t
        dp = result.delta_phi
        ice = result.ice_series
        reg = result.regimes

        # Panel 1 – ICE components
        axes[0].plot(t, ice.delta_E, lw=0.8, color="steelblue",  label="ΔE(t)")
        axes[0].plot(t, ice.delta_I, lw=0.8, color="darkorange", label="ΔI(t)")
        axes[0].plot(t, ice.delta_C, lw=0.8, color="seagreen",   label="ΔC(t)")
        axes[0].axhline(0, color="black", lw=0.5, ls=":")
        axes[0].set_ylabel("Normalised deviation")
        axes[0].set_title("DNA-Associated ICE Components (z-scored against baseline)")
        axes[0].legend(loc="upper left", fontsize=8, ncol=3)

        # Panel 2 – ΔΦ(t)
        axes[1].plot(t, dp, lw=0.9, color="crimson", label="ΔΦ(t)")
        axes[1].axhline(
            reg.theta_low,
            color="orange",
            lw=1.2,
            ls="--",
            label=f"θ_low = {reg.theta_low:.3f}",
        )
        axes[1].axhline(
            reg.theta_high,
            color="red",
            lw=1.2,
            ls="--",
            label=f"θ_high = {reg.theta_high:.3f}",
        )
        axes[1].set_ylabel("ΔΦ(t)")
        axes[1].set_title("DNA Instability Magnitude ΔΦ(t)")
        axes[1].legend(loc="upper left", fontsize=8)

        # Panel 3 – regime labels
        _REGIME_COLOURS = {
            "stable":           "steelblue",
            "pre-instability":  "orange",
            "instability":      "crimson",
            "recovery":         "seagreen",
        }
        label_arr = reg.labels
        for regime in ["stable", "pre-instability", "instability", "recovery"]:
            mask = label_arr == regime
            if mask.any():
                axes[2].scatter(
                    t[mask],
                    np.zeros(mask.sum()),
                    c=_REGIME_COLOURS[regime],
                    s=3,
                    label=regime,
                    marker="|",
                )
        axes[2].set_ylabel("Regime")
        axes[2].set_title("Regime Labels")
        axes[2].set_yticks([])
        axes[2].legend(loc="upper left", fontsize=8, ncol=4)

        # Panel 4 – spiral-time or smoothed ΔΦ
        if result.phi is not None and result.chi is not None:
            axes[3].plot(t, result.phi, lw=0.8, color="mediumpurple", label="φ(t)")
            axes[3].plot(t, result.chi, lw=0.8, color="teal",         label="χ(t)")
            axes[3].set_ylabel("Spiral-time")
            axes[3].set_title("Spiral-Time Embedding: φ(t) and χ(t)")
            axes[3].legend(loc="upper left", fontsize=8)
        else:
            from scipy.signal import savgol_filter  # local import to keep dep optional
            win_len = max(11, int(len(dp) * 0.01) // 2 * 2 + 1)  # odd, ≥ 11
            if win_len >= len(dp):
                win_len = len(dp) if len(dp) % 2 == 1 else len(dp) - 1
            smoothed = savgol_filter(dp, window_length=win_len, polyorder=3)
            axes[3].plot(t, dp,       lw=0.5, color="lightcoral", alpha=0.5, label="ΔΦ(t) raw")
            axes[3].plot(t, smoothed, lw=1.2, color="crimson",               label="ΔΦ(t) smoothed")
            axes[3].set_ylabel("ΔΦ(t)")
            axes[3].set_title("ΔΦ(t) – Smoothed (Savitzky–Golay, visualisation only)")
            axes[3].legend(loc="upper left", fontsize=8)

        # Vertical regime-boundary lines on every panel
        if regime_boundaries:
            for ax in axes:
                for x_b in regime_boundaries:
                    ax.axvline(x_b, color="gray", lw=1.0, ls=":", alpha=0.7)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        fig.savefig(str(path), dpi=dpi)
        plt.close(fig)

    def save_csv(
        self,
        result: DNAPipelineResult,
        path: str | Path,
    ) -> None:
        """Save a per-time-point CSV summary of the DNA pipeline result.

        Each row corresponds to one time sample and contains: time (s),
        ΔE, ΔI, ΔC, ΔΦ, regime label, and (when available) φ and χ.

        Parameters
        ----------
        result:
            Output from :meth:`run`.
        path:
            Destination file path.  Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        has_spiral = result.phi is not None

        header = ["t_s", "delta_E", "delta_I", "delta_C", "delta_phi", "regime"]
        if has_spiral:
            header += ["phi", "chi"]

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            T = len(result.t)
            for i in range(T):
                row = [
                    f"{result.t[i]:.6f}",
                    f"{result.ice_series.delta_E[i]:.6f}",
                    f"{result.ice_series.delta_I[i]:.6f}",
                    f"{result.ice_series.delta_C[i]:.6f}",
                    f"{result.delta_phi[i]:.6f}",
                    str(result.regimes.labels[i]),
                ]
                if has_spiral:
                    row += [
                        f"{result.phi[i]:.6f}",
                        f"{result.chi[i]:.6f}",
                    ]
                writer.writerow(row)

    def print_regime_summary(self, result: DNAPipelineResult) -> None:
        """Print a concise regime statistics table to stdout.

        Parameters
        ----------
        result:
            Output from :meth:`run`.
        """
        print("=" * 55)
        print("DNA Pipeline – Regime Summary")
        print("=" * 55)
        print(f"  θ_low  = {result.regimes.theta_low:.4f}")
        print(f"  θ_high = {result.regimes.theta_high:.4f}")
        print("-" * 55)
        print(f"  {'Regime':<20}  {'Mean ΔΦ':>10}  {'Count':>8}")
        print("-" * 55)
        labels = result.regimes.labels
        for regime in ["stable", "pre-instability", "instability", "recovery"]:
            mask = labels == regime
            count = int(mask.sum())
            mean_dp = result.regime_stats.get(regime, float("nan"))
            print(f"  {regime:<20}  {mean_dp:>10.4f}  {count:>8d}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run_dna_pipeline(
    raw_data: dict[str, np.ndarray],
    feature_specs: list[DNAFeatureSpec],
    baseline_window: tuple[int, int],
    *,
    fs: float = 1.0,
    compute_spiral: bool = False,
    q_low: float = 0.75,
    q_high: float = 0.90,
    detect_recovery: bool = True,
    min_std: float = 1e-8,
    kernel: str = "exponential",
    kernel_kwargs: dict | None = None,
    figure_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    regime_boundaries: Sequence[float] | None = None,
) -> DNAPipelineResult:
    """Convenience wrapper that constructs and runs :class:`DNAPipeline`.

    Parameters
    ----------
    raw_data:
        Raw channel arrays (see :meth:`DNAPipeline.run`).
    feature_specs:
        ICE mapping declarations (see :class:`DNAPipelineConfig`).
    baseline_window:
        ``(start, stop)`` sample indices for the reference baseline.
    fs:
        Sampling frequency in Hz.
    compute_spiral:
        Whether to compute φ(t) and χ(t).
    q_low, q_high:
        Quantile thresholds for regime segmentation.
    detect_recovery:
        Whether to apply recovery-label post-processing.
    min_std:
        Lower bound on standard deviation in z-score normalisation.
    kernel:
        Memory kernel for χ(t) (``"exponential"`` or ``"gaussian"``).
    kernel_kwargs:
        Extra arguments forwarded to the kernel.
    figure_path:
        If provided, :meth:`~DNAPipeline.save_figure` is called.
    csv_path:
        If provided, :meth:`~DNAPipeline.save_csv` is called.
    regime_boundaries:
        Optional vertical markers for :meth:`~DNAPipeline.save_figure`.

    Returns
    -------
    DNAPipelineResult
    """
    cfg = DNAPipelineConfig(
        feature_specs=feature_specs,
        baseline_window=baseline_window,
        fs=fs,
        compute_spiral=compute_spiral,
        q_low=q_low,
        q_high=q_high,
        detect_recovery=detect_recovery,
        min_std=min_std,
        kernel=kernel,
        kernel_kwargs=kernel_kwargs or {},
    )
    pipeline = DNAPipeline(cfg)
    result = pipeline.run(raw_data)

    if figure_path is not None:
        pipeline.save_figure(result, figure_path, regime_boundaries=regime_boundaries)
    if csv_path is not None:
        pipeline.save_csv(result, csv_path)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_regime_stats(
    delta_phi: np.ndarray,
    regimes: RegimeLabels,
) -> dict[str, float]:
    """Return the mean ΔΦ for each regime label."""
    stats: dict[str, float] = {}
    for regime in ["stable", "pre-instability", "instability", "recovery"]:
        mask = regimes.labels == regime
        if mask.any():
            stats[regime] = float(np.mean(delta_phi[mask]))
        else:
            stats[regime] = float("nan")
    return stats
