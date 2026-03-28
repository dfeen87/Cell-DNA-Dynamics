"""
Metrics for evaluating the triadic stability framework.

All functions operate on:

* ``delta_phi`` – the instability magnitude ΔΦ(t), a 1-D float array.
* ``regime_labels`` – per-sample regime strings drawn from
  ``{"stable", "pre-instability", "instability", "recovery"}``.

Available metrics
-----------------
* :func:`compute_auc_full`        – AUC for the full ΔΦ(t) discriminating
                                    instability from non-instability.
* :func:`compute_auc_components`  – per-component AUCs (AUC_E, AUC_I, AUC_C).
* :func:`count_regime_transitions` – total number of regime-boundary crossings.
* :func:`delta_phi_summary`       – mean, std, min, max of ΔΦ(t) per regime.

Usage
-----
>>> import numpy as np
>>> from src.simulation.metrics import compute_auc_full, delta_phi_summary
"""

from __future__ import annotations

import sys
import os

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binary_labels(regime_labels: np.ndarray, positive: str = "instability") -> np.ndarray:
    """Return a binary (0/1) array where *positive* regime is 1."""
    return (np.asarray(regime_labels) == positive).astype(int)


def _safe_auc(score: np.ndarray, y_true: np.ndarray) -> float:
    """Compute AUC, returning NaN when sklearn is unavailable or classes are degenerate."""
    if not _HAS_SKLEARN:
        return float("nan")
    valid = np.isfinite(score)
    if len(np.unique(y_true[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[valid], score[valid]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_auc_full(
    delta_phi: np.ndarray,
    regime_labels: np.ndarray,
) -> float:
    """AUC for the full ΔΦ(t) discriminating instability from all other regimes.

    Parameters
    ----------
    delta_phi : np.ndarray, shape (T,)
        Instability magnitude ΔΦ(t).
    regime_labels : np.ndarray, shape (T,), dtype object
        Per-sample regime strings.

    Returns
    -------
    auc : float
        Area under the ROC curve (``NaN`` if sklearn is unavailable or the
        label set is degenerate).
    """
    delta_phi = np.asarray(delta_phi, dtype=float)
    y = _binary_labels(regime_labels)
    return _safe_auc(delta_phi, y)


def compute_auc_components(
    delta_E: np.ndarray,
    delta_I: np.ndarray,
    delta_C: np.ndarray,
    regime_labels: np.ndarray,
) -> dict[str, float]:
    """Per-component AUCs using the absolute deviation of each ICE component.

    Parameters
    ----------
    delta_E, delta_I, delta_C : np.ndarray, shape (T,)
        Normalised ICE deviation components.
    regime_labels : np.ndarray, shape (T,)
        Per-sample regime strings.

    Returns
    -------
    aucs : dict
        Keys: ``"AUC_E"``, ``"AUC_I"``, ``"AUC_C"``.
    """
    y = _binary_labels(regime_labels)
    return {
        "AUC_E": _safe_auc(np.abs(np.asarray(delta_E, dtype=float)), y),
        "AUC_I": _safe_auc(np.abs(np.asarray(delta_I, dtype=float)), y),
        "AUC_C": _safe_auc(np.abs(np.asarray(delta_C, dtype=float)), y),
    }


def count_regime_transitions(regime_labels: np.ndarray) -> int:
    """Count the number of times the regime label changes between consecutive steps.

    Parameters
    ----------
    regime_labels : np.ndarray, shape (T,)
        Per-sample regime strings.

    Returns
    -------
    n_transitions : int
        Total number of label-change events in the series.
    """
    labels = np.asarray(regime_labels)
    if len(labels) < 2:
        return 0
    return int(np.sum(labels[:-1] != labels[1:]))


def delta_phi_summary(
    delta_phi: np.ndarray,
    regime_labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-regime summary statistics of ΔΦ(t).

    Parameters
    ----------
    delta_phi : np.ndarray, shape (T,)
        Instability magnitude ΔΦ(t).
    regime_labels : np.ndarray, shape (T,)
        Per-sample regime strings.

    Returns
    -------
    stats : dict
        ``{regime: {"mean": float, "std": float, "min": float, "max": float}}``.
        All four canonical regime names are always present; regimes with no
        samples have ``NaN`` values.
    """
    delta_phi = np.asarray(delta_phi, dtype=float)
    regime_labels = np.asarray(regime_labels)

    all_regimes = ["stable", "pre-instability", "instability", "recovery"]
    stats: dict[str, dict[str, float]] = {}

    for regime in all_regimes:
        mask = regime_labels == regime
        vals = delta_phi[mask]
        if len(vals) == 0:
            stats[regime] = {"mean": float("nan"), "std": float("nan"),
                             "min": float("nan"), "max": float("nan")}
        else:
            stats[regime] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    return stats
