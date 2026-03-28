"""
Spiral-Time Operator D_Ψ – Stage 4 Spiral-Time Embedding
=========================================================

Implements the spiral-time trajectory operator defined in the manuscript
(Eq. 12):

    D_Ψ f(γ(t)) := d/dt f(γ(t)) + i φ̇(t) v_φ(t) · ∇f(γ(t))
                                  + j χ̇(t) v_χ(t) · ∇f(γ(t))      (12)

where:
- γ(t) = (ΔE(t), ΔI(t), ΔC(t)) is the trajectory on the ICE manifold
- f : M_ICE → R is a scalar observable
- ∇f(γ(t)) is the gradient of f at the current trajectory point in ICE space
- v_φ(t), v_χ(t) are the normalised trajectory tangent vectors encoding
  phase-history and memory-like directional contributions respectively
- φ̇(t) and χ̇(t) are the time derivatives of the adaptive phase and
  memory component
- i, j are the algebraic generators of the triadic spiral-time structure

The implementation:
- uses :class:`~src.core.models.ICEStateSeries` for the input trajectory
- estimates ∂f/∂E, ∂f/∂I, ∂f/∂C via central finite differences
- takes v_φ = v_χ = γ̇(t) / |γ̇(t)| (normalised trajectory velocity) as the
  natural tangent direction on the ICE manifold; both directional dot-products
  v · ∇f are therefore real-valued scalars at each time step
- computes all time derivatives with numpy central differences

Output
------
The function :func:`evaluate_spiral_operator` returns a
:class:`SpiralOperatorResult` whose three components correspond to the
real (temporal drift), i-part (phase-history deformation), and j-part
(memory-bandwidth deformation) contributions to D_Ψ f, each as a 1-D
time series of shape (T,).
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Allow running from the repository root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.models import ICEStateSeries
from src.stage4_spiral_time.compute_phi import compute_phi
from src.stage4_spiral_time.compute_chi import compute_chi


@dataclass
class SpiralOperatorResult:
    """Result of evaluating the spiral-time operator D_Ψ f over time.

    Holds the three components of the operator defined in Eq. (12) of the
    manuscript:

        D_Ψ f = real_part + i · phi_part + j · chi_part

    where i, j are the algebraic generators of the triadic spiral-time
    structure (not standard imaginary units).

    Attributes
    ----------
    real_part : np.ndarray, shape (T,)
        Ordinary temporal drift: d/dt f(γ(t)).
    phi_part : np.ndarray, shape (T,)
        Phase-history deformation: φ̇(t) · (v_φ · ∇f)(γ(t)).
    chi_part : np.ndarray, shape (T,)
        Memory-bandwidth deformation: χ̇(t) · (v_χ · ∇f)(γ(t)).
    t : np.ndarray, shape (T,)
        Time coordinates at which the operator was evaluated.
    """

    real_part: np.ndarray
    phi_part: np.ndarray
    chi_part: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        self.real_part = np.asarray(self.real_part, dtype=float)
        self.phi_part = np.asarray(self.phi_part, dtype=float)
        self.chi_part = np.asarray(self.chi_part, dtype=float)
        self.t = np.asarray(self.t, dtype=float)


def _numerical_gradient(
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    E: np.ndarray,
    I: np.ndarray,
    C: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate ∇f = (∂f/∂E, ∂f/∂I, ∂f/∂C) via central finite differences.

    Each partial derivative is approximated as:

        ∂f/∂E ≈ ( f(E+ε, I, C) − f(E−ε, I, C) ) / (2ε)

    and likewise for I and C.  The arrays E, I, C represent the current
    trajectory coordinates; the gradient is evaluated point-wise along
    the trajectory.

    Parameters
    ----------
    f : callable
        Scalar observable with signature ``f(E, I, C)`` accepting three
        1-D arrays of shape (T,) and returning an array of shape (T,).
    E, I, C : np.ndarray, shape (T,)
        ICE trajectory coordinates (ΔE, ΔI, ΔC) at which to evaluate ∇f.
    eps : float, optional
        Finite-difference step size.  Default is 1e-6.

    Returns
    -------
    grad_E, grad_I, grad_C : np.ndarray, shape (T,)
        Partial derivatives ∂f/∂E, ∂f/∂I, ∂f/∂C along the trajectory.
    """
    grad_E = (f(E + eps, I, C) - f(E - eps, I, C)) / (2.0 * eps)
    grad_I = (f(E, I + eps, C) - f(E, I - eps, C)) / (2.0 * eps)
    grad_C = (f(E, I, C + eps) - f(E, I, C - eps)) / (2.0 * eps)
    return (
        np.asarray(grad_E, dtype=float),
        np.asarray(grad_I, dtype=float),
        np.asarray(grad_C, dtype=float),
    )


def evaluate_spiral_operator(
    series: ICEStateSeries,
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    t: np.ndarray,
    phi: np.ndarray | None = None,
    chi: np.ndarray | None = None,
    grad_eps: float = 1e-6,
) -> SpiralOperatorResult:
    """Evaluate the spiral-time operator D_Ψ f over the ICE trajectory.

    Computes Eq. (12) from the manuscript at each time step:

        D_Ψ f(γ(t)) = d/dt f(γ(t))
                      + i · φ̇(t) · v_φ(t) · ∇f(γ(t))
                      + j · χ̇(t) · v_χ(t) · ∇f(γ(t))

    The direction vectors v_φ and v_χ are both taken as the normalised
    trajectory velocity γ̇(t) / |γ̇(t)|, which is the natural tangent
    vector on the ICE manifold.  At stationary points (|γ̇| = 0) the
    directional derivative defaults to zero.

    All time derivatives are computed with ``numpy.gradient`` (second-order
    central differences at interior points, first-order at boundaries).

    Parameters
    ----------
    series : ICEStateSeries
        Time-resolved ICE deviation series (ΔE, ΔI, ΔC), each of shape (T,).
    f : callable
        Scalar observable with signature ``f(E, I, C)`` accepting three
        1-D arrays of shape (T,) and returning an array of shape (T,).
        The spatial gradient ∇f is estimated numerically.
    t : np.ndarray, shape (T,)
        Monotonically increasing time coordinates matching the series.
    phi : np.ndarray, shape (T,), optional
        Pre-computed adaptive phase φ(t).  If omitted, computed from
        *series* via :func:`~src.stage4_spiral_time.compute_phi.compute_phi`.
    chi : np.ndarray, shape (T,), optional
        Pre-computed memory component χ(t).  If omitted, computed from
        *phi* and *t* via
        :func:`~src.stage4_spiral_time.compute_chi.compute_chi` with
        default settings (exponential kernel, λ = 1).
    grad_eps : float, optional
        Finite-difference step size for the numerical gradient of *f*.
        Default is 1e-6.

    Returns
    -------
    result : SpiralOperatorResult
        Operator components evaluated at each time step:

        - ``real_part`` – temporal drift d/dt f(γ(t)), shape (T,)
        - ``phi_part``  – phase-history contribution φ̇ · (v_φ · ∇f), shape (T,)
        - ``chi_part``  – memory contribution χ̇ · (v_χ · ∇f), shape (T,)
        - ``t``         – time coordinates, shape (T,)

    Raises
    ------
    ValueError
        If *t* is not 1-D, or if the ICE series arrays do not match the
        length of *t*.

    Examples
    --------
    >>> import numpy as np
    >>> from src.core.models import ICEStateSeries
    >>> from src.stage4_spiral_time.spiral_operator import evaluate_spiral_operator
    >>> t = np.linspace(0, 10, 200)
    >>> series = ICEStateSeries(
    ...     delta_E=np.sin(t),
    ...     delta_I=np.cos(t),
    ...     delta_C=0.1 * t,
    ... )
    >>> def f(E, I, C):
    ...     return E ** 2 + I ** 2 + C ** 2
    >>> result = evaluate_spiral_operator(series, f, t)
    >>> result.real_part.shape
    (200,)
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError(f"t must be a 1-D array; got shape {t.shape}.")

    T = len(t)
    if series.delta_E.shape != (T,):
        raise ValueError(
            f"ICEStateSeries components must have shape ({T},) to match t; "
            f"got {series.delta_E.shape}."
        )

    E = series.delta_E
    I = series.delta_I
    C = series.delta_C

    # ── Adaptive phase φ(t) and memory component χ(t) ────────────────────────
    if phi is None:
        phi = compute_phi(series)
    else:
        phi = np.asarray(phi, dtype=float)

    if chi is None:
        chi = compute_chi(phi, t)
    else:
        chi = np.asarray(chi, dtype=float)

    # ── Time derivatives φ̇(t) and χ̇(t) ──────────────────────────────────────
    phi_dot = np.gradient(phi, t)   # shape (T,)
    chi_dot = np.gradient(chi, t)   # shape (T,)

    # ── Observable f evaluated along the trajectory ───────────────────────────
    f_values = np.asarray(f(E, I, C), dtype=float)   # shape (T,)

    # ── Real part: d/dt f(γ(t)) ───────────────────────────────────────────────
    df_dt = np.gradient(f_values, t)   # shape (T,)

    # ── Spatial gradient ∇f(γ(t)) in the ICE manifold ────────────────────────
    grad_E, grad_I, grad_C = _numerical_gradient(f, E, I, C, eps=grad_eps)
    # Stack to shape (T, 3): columns are ∂f/∂E, ∂f/∂I, ∂f/∂C
    grad_f = np.stack([grad_E, grad_I, grad_C], axis=1)

    # ── Trajectory velocity γ̇(t) = (dΔE/dt, dΔI/dt, dΔC/dt) ─────────────────
    gamma_dot = np.stack(
        [np.gradient(E, t), np.gradient(I, t), np.gradient(C, t)],
        axis=1,
    )   # shape (T, 3)

    # ── Normalise γ̇ to obtain the unit tangent vector ─────────────────────────
    norm_gamma_dot = np.linalg.norm(gamma_dot, axis=1, keepdims=True)   # (T, 1)
    # At stationary points |γ̇| = 0 the trajectory has no preferred direction;
    # v_unit is set to the zero vector so that both directional-derivative
    # contributions (phi_part and chi_part) are exactly zero at those time
    # steps by design.  This is consistent with the limiting case described in
    # the manuscript (Eq. 14): when the trajectory is momentarily stationary
    # the operator reduces to the ordinary temporal derivative.
    safe_norm = np.where(norm_gamma_dot > 0.0, norm_gamma_dot, 1.0)
    v_unit = np.where(norm_gamma_dot > 0.0, gamma_dot / safe_norm, 0.0)

    # ── Directional derivatives v · ∇f (scalars along the trajectory) ─────────
    # v_φ = v_χ = v_unit (normalised trajectory tangent)
    v_dot_grad_f = np.sum(v_unit * grad_f, axis=1)   # shape (T,)

    # ── Assemble the operator components (Eq. 12) ─────────────────────────────
    real_part = df_dt                        # d/dt f(γ(t))
    phi_part = phi_dot * v_dot_grad_f        # φ̇(t) · (v_φ · ∇f)(γ(t))
    chi_part = chi_dot * v_dot_grad_f        # χ̇(t) · (v_χ · ∇f)(γ(t))

    return SpiralOperatorResult(
        real_part=real_part,
        phi_part=phi_part,
        chi_part=chi_part,
        t=t,
    )
