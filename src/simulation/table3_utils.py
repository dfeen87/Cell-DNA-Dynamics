"""
Table 3 metric aggregation utilities.

:func:`build_table3_metrics` is the single entry-point:

1. Generate (or reload) the synthetic ICE series via
   :mod:`src.simulation.generate_synthetic`.
2. Generate (or reload) null-test surrogates via
   :mod:`src.simulation.null_tests`.
3. Compute all metrics via :mod:`src.simulation.metrics`.
4. Aggregate into a ``pandas.DataFrame`` and save to
   ``simulations/table3_metrics.csv``.

Usage
-----
>>> from src.simulation.table3_utils import build_table3_metrics
>>> df = build_table3_metrics()
>>> print(df)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from src.simulation.generate_synthetic import generate_synthetic_ice  # noqa: E402
from src.simulation.null_tests import run_null_tests, load_null_test   # noqa: E402
from src.simulation import metrics as _metrics                          # noqa: E402

_OUT_DIR = os.path.join(_REPO_ROOT, "simulations")
_OUT_CSV = os.path.join(_OUT_DIR, "table3_metrics.csv")

# Also write a copy alongside the existing placeholder in figures/simulations/
_FIG_OUT_CSV = os.path.join(_REPO_ROOT, "figures", "simulations", "table3_metrics.csv")


def build_table3_metrics(seed: int = 0, regenerate: bool = True) -> pd.DataFrame:
    """Build Table 3 by aggregating observed and null-test metrics.

    Parameters
    ----------
    seed : int
        Random seed passed to both :func:`generate_synthetic_ice` and
        :func:`run_null_tests`.
    regenerate : bool
        When ``True`` (default) the synthetic data and null-test files are
        always regenerated.  Set to ``False`` to load previously saved files
        without re-simulating.

    Returns
    -------
    df : pandas.DataFrame
        One row per condition (observed + three null tests), with columns:

        * ``condition``         – human-readable label
        * ``AUC_full``          – AUC for ΔΦ(t) (instability vs rest)
        * ``AUC_E``             – AUC for |ΔE(t)|
        * ``AUC_I``             – AUC for |ΔI(t)|
        * ``AUC_C``             – AUC for |ΔC(t)|
        * ``n_transitions``     – number of regime-label changes
        * ``mean_stable``       – mean ΔΦ during stable regime
        * ``std_stable``        – std ΔΦ during stable regime
        * ``mean_pre``          – mean ΔΦ during pre-instability
        * ``mean_instability``  – mean ΔΦ during instability regime
        * ``mean_recovery``     – mean ΔΦ during recovery regime

    Side-effects
    ------------
    Saves the DataFrame to:

    * ``simulations/table3_metrics.csv``
    * ``figures/simulations/table3_metrics.csv``
    """
    # ------------------------------------------------------------------
    # 1. Synthetic ICE series (observed condition)
    # ------------------------------------------------------------------
    series, t, regime_labels = generate_synthetic_ice(seed=seed)
    delta_E = series.delta_E
    delta_I = series.delta_I
    delta_C = series.delta_C
    obs_delta_phi = np.sqrt(delta_E ** 2 + delta_I ** 2 + delta_C ** 2)

    # ------------------------------------------------------------------
    # 2. Null-test surrogates
    # ------------------------------------------------------------------
    null_results = run_null_tests(seed=seed)

    # ------------------------------------------------------------------
    # 3. Compute metrics for the observed condition
    # ------------------------------------------------------------------
    rows: list[dict] = []

    def _make_row(
        condition: str,
        dp: np.ndarray,
        labels: np.ndarray,
        dE: np.ndarray | None = None,
        dI: np.ndarray | None = None,
        dC: np.ndarray | None = None,
    ) -> dict:
        auc_full = _metrics.compute_auc_full(dp, labels)
        if dE is not None and dI is not None and dC is not None:
            comp_aucs = _metrics.compute_auc_components(dE, dI, dC, labels)
        else:
            comp_aucs = {"AUC_E": float("nan"), "AUC_I": float("nan"), "AUC_C": float("nan")}
        n_trans = _metrics.count_regime_transitions(labels)
        summ = _metrics.delta_phi_summary(dp, labels)

        return {
            "condition": condition,
            "AUC_full": auc_full,
            "AUC_E": comp_aucs["AUC_E"],
            "AUC_I": comp_aucs["AUC_I"],
            "AUC_C": comp_aucs["AUC_C"],
            "n_transitions": n_trans,
            "mean_stable": summ["stable"]["mean"],
            "std_stable": summ["stable"]["std"],
            "mean_pre": summ["pre-instability"]["mean"],
            "mean_instability": summ["instability"]["mean"],
            "mean_recovery": summ["recovery"]["mean"],
        }

    # Observed
    rows.append(_make_row(
        "Observed (synthetic)",
        obs_delta_phi, regime_labels,
        delta_E, delta_I, delta_C,
    ))

    # Null tests – use the *observed* regime labels so the AUC comparison
    # is meaningful (same positive/negative class, different scores).
    null_labels_map = {
        "shuffled": "Shuffled-time null",
        "surrogate": "Phase-randomised null",
        "stable_control": "Stable control",
    }
    for key, label in null_labels_map.items():
        null_phi = null_results[key]
        rows.append(_make_row(label, null_phi, regime_labels))

    # ------------------------------------------------------------------
    # 4. Build DataFrame and save
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)

    os.makedirs(_OUT_DIR, exist_ok=True)
    df.to_csv(_OUT_CSV, index=False)
    print(f"Saved: {_OUT_CSV}")

    os.makedirs(os.path.dirname(_FIG_OUT_CSV), exist_ok=True)
    df.to_csv(_FIG_OUT_CSV, index=False)
    print(f"Saved: {_FIG_OUT_CSV}")

    return df


if __name__ == "__main__":
    df = build_table3_metrics()
    print(df.to_string(index=False))
