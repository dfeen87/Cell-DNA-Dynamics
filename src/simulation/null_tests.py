"""
Null-test surrogates for the triadic stability framework.

Two null hypotheses are implemented:

1. **Shuffled-time null** – the time ordering of the ΔΦ(t) series is
   destroyed by random permutation.  Any real temporal structure (e.g.
   monotone regime progression) should vanish.

2. **Phase-randomised null** – the Fourier phases of each ICE component
   are replaced with independent uniform random phases while preserving the
   power spectrum (amplitude spectrum).  This breaks nonlinear coupling
   between components without altering their individual autocorrelation.

Both surrogates are saved under ``data/synthetic/null_tests/`` and can be
reloaded with the helper functions :func:`load_null_test` and
:func:`compare_null_trajectories`.

Usage
-----
>>> from src.simulation.null_tests import run_null_tests
>>> run_null_tests()
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from src.simulation.generate_synthetic import generate_synthetic_ice  # noqa: E402

_NULL_DIR = os.path.join(_REPO_ROOT, "data", "synthetic", "null_tests")


# ---------------------------------------------------------------------------
# Null-test generators
# ---------------------------------------------------------------------------

def shuffled_time_null(
    delta_phi: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Return a time-shuffled surrogate of *delta_phi*.

    All values are retained but their temporal order is randomised so that
    any genuine regime-progression signal is destroyed.

    Parameters
    ----------
    delta_phi : np.ndarray, shape (T,)
        Original instability magnitude series.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    shuffled : np.ndarray, shape (T,)
        Randomly permuted copy of *delta_phi*.
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(delta_phi)


def phase_randomized_null(
    delta_E: np.ndarray,
    delta_I: np.ndarray,
    delta_C: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Return a phase-randomised surrogate ΔΦ(t).

    Each ICE component's Fourier phases are replaced with independent
    uniform-random values while keeping the amplitude spectrum intact.
    The surrogate ΔΦ(t) is then recomputed from the three surrogate
    components.

    Parameters
    ----------
    delta_E, delta_I, delta_C : np.ndarray, shape (T,)
        Normalised ICE deviation components.
    seed : int
        Random seed.

    Returns
    -------
    surrogate_delta_phi : np.ndarray, shape (T,)
        ΔΦ(t) reconstructed from the phase-randomised components.
    """
    rng = np.random.default_rng(seed)

    def _phase_randomize(x: np.ndarray) -> np.ndarray:
        n = len(x)
        spectrum = np.fft.rfft(x)
        amplitudes = np.abs(spectrum)
        random_phases = rng.uniform(0, 2 * np.pi, len(spectrum))
        # Preserve DC and Nyquist phase for strict real-valued output
        random_phases[0] = 0.0
        if n % 2 == 0:
            random_phases[-1] = 0.0
        new_spectrum = amplitudes * np.exp(1j * random_phases)
        surrogate = np.fft.irfft(new_spectrum, n=n)
        return surrogate

    sE = _phase_randomize(delta_E)
    sI = _phase_randomize(delta_I)
    sC = _phase_randomize(delta_C)

    return np.sqrt(sE ** 2 + sI ** 2 + sC ** 2)


def stable_control_null(
    t: np.ndarray,
    seed: int = 0,
    noise_scale: float = 0.05,
) -> np.ndarray:
    """Return a purely-stable (white-noise) control ΔΦ(t).

    Simulates a system that stays in the stable regime throughout,
    providing a low-ΔΦ baseline for visual comparison.

    Parameters
    ----------
    t : np.ndarray, shape (T,)
        Time vector.
    seed : int
        Random seed.
    noise_scale : float
        Noise amplitude for each ICE component.

    Returns
    -------
    control_delta_phi : np.ndarray, shape (T,)
        ΔΦ(t) for the stable control.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    sE = noise_scale * rng.standard_normal(n)
    sI = noise_scale * rng.standard_normal(n)
    sC = noise_scale * rng.standard_normal(n)
    return np.sqrt(sE ** 2 + sI ** 2 + sC ** 2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_null_tests(seed: int = 0) -> dict[str, np.ndarray]:
    """Generate and persist all three null-test surrogates.

    Returns a dictionary with keys ``"shuffled"``, ``"surrogate"``, and
    ``"stable_control"``, each mapping to its ΔΦ(t) surrogate array.

    Side-effects
    ------------
    Writes three CSV files under ``data/synthetic/null_tests/``:

    * ``shuffled.csv``       – columns: time, delta_phi
    * ``surrogate.csv``      – columns: time, delta_phi
    * ``stable_control.csv`` – columns: time, delta_phi
    """
    series, t, _ = generate_synthetic_ice(seed=seed)
    delta_E = series.delta_E
    delta_I = series.delta_I
    delta_C = series.delta_C
    delta_phi = np.sqrt(delta_E ** 2 + delta_I ** 2 + delta_C ** 2)

    shuffled = shuffled_time_null(delta_phi, seed=seed)
    surrogate = phase_randomized_null(delta_E, delta_I, delta_C, seed=seed)
    control = stable_control_null(t, seed=seed)

    os.makedirs(_NULL_DIR, exist_ok=True)

    results: dict[str, np.ndarray] = {
        "shuffled": shuffled,
        "surrogate": surrogate,
        "stable_control": control,
    }

    for name, arr in results.items():
        path = os.path.join(_NULL_DIR, f"{name}.csv")
        pd.DataFrame({"time": t, "delta_phi": arr}).to_csv(path, index=False)

    return results


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_null_test(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved null-test surrogate from disk.

    Parameters
    ----------
    name : str
        One of ``"shuffled"``, ``"surrogate"``, or ``"stable_control"``.

    Returns
    -------
    t : np.ndarray
        Time vector.
    delta_phi : np.ndarray
        Surrogate ΔΦ(t).
    """
    path = os.path.join(_NULL_DIR, f"{name}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Null-test file not found: {path}\n"
            "Run `run_null_tests()` first to generate the surrogates."
        )
    df = pd.read_csv(path)
    return df["time"].to_numpy(), df["delta_phi"].to_numpy()


def compare_null_trajectories(
    observed_delta_phi: np.ndarray,
    null_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compare the observed ΔΦ(t) to each null-test surrogate.

    For each surrogate, computes the mean and standard deviation of the
    *difference* ``observed − null`` as a simple scalar summary.

    Parameters
    ----------
    observed_delta_phi : np.ndarray, shape (T,)
        Observed ΔΦ(t) from the synthetic ICE series.
    null_names : list of str, optional
        Subset of null tests to compare.  Defaults to all three.

    Returns
    -------
    stats : dict
        ``{null_name: {"mean_diff": float, "std_diff": float}}``.
    """
    if null_names is None:
        null_names = ["shuffled", "surrogate", "stable_control"]

    stats: dict[str, dict[str, float]] = {}
    for name in null_names:
        _, null_phi = load_null_test(name)
        n = min(len(observed_delta_phi), len(null_phi))
        diff = observed_delta_phi[:n] - null_phi[:n]
        stats[name] = {
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff)),
        }
    return stats


if __name__ == "__main__":
    results = run_null_tests()
    for name, arr in results.items():
        print(f"  {name}: mean={arr.mean():.4f}, std={arr.std():.4f}")
    print("Saved: data/synthetic/null_tests/{shuffled,surrogate,stable_control}.csv")
