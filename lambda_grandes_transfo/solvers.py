from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


Vector = np.ndarray
ResidualFunction = Callable[[Vector], Vector]
ResidualFactory = Callable[[float], ResidualFunction]
ContinuationStep = Tuple[float, Vector]


@dataclass
class NewtonSettings:
    """Parameters for damped Newton iterations."""

    k_max: int = 40
    tol_R: float = 1e-12
    tol_dx: float = 1e-14

    # Armijo backtracking on ||R||
    armijo_c: float = 1e-4
    backtrack_rho: float = 0.5
    alpha_min: float = 1e-10

    # Central finite-difference Jacobian step
    fd_h: float = 1e-7


@dataclass
class ContinuationSettings:
    """Adaptive continuation step control for the load parameter T."""

    T_max: float
    dT0: float = 0.01
    dT_min: float = 1e-7
    dT_max: float = 0.1

    # Iteration-count based adaptation thresholds
    k_good: int = 4
    k_bad: int = 12
    gamma_up: float = 1.5
    gamma_down: float = 0.5


@dataclass
class NewtonResult:
    """Outcome of one Newton solve at fixed load."""

    converged: bool
    x: Vector
    n_iterations: int
    residual_norm: float


def jacobian_fd(residual: ResidualFunction, x: Vector, h: float) -> np.ndarray:
    """
    Central finite-difference Jacobian approximation.

    For each component x_j:
      J[:,j] ≈ (R(x + h e_j) - R(x - h e_j)) / (2 h)
    """
    x = np.asarray(x, dtype=float)
    residual_at_x = residual(x)
    output_dim = residual_at_x.size
    input_dim = x.size

    jacobian = np.zeros((output_dim, input_dim), dtype=float)
    for j in range(input_dim):
        perturbation = np.zeros(input_dim, dtype=float)
        perturbation[j] = h
        residual_plus = residual(x + perturbation)
        residual_minus = residual(x - perturbation)
        jacobian[:, j] = (residual_plus - residual_minus) / (2.0 * h)

    return jacobian


def newton_armijo(residual: ResidualFunction, x0: Vector, settings: NewtonSettings) -> Tuple[bool, Vector, int, float]:
    """
    Damped Newton method with Armijo backtracking on ||R||_2.

    Public return signature is kept for backward compatibility:
      (converged, x, n_iterations, residual_norm)
    """
    result = _newton_armijo_result(residual, x0, settings)
    return result.converged, result.x, result.n_iterations, result.residual_norm


def _newton_armijo_result(residual: ResidualFunction, x0: Vector, settings: NewtonSettings) -> NewtonResult:
    x = np.asarray(x0, dtype=float).copy()

    for k in range(settings.k_max):
        residual_value = residual(x)
        residual_norm = float(np.linalg.norm(residual_value))
        if residual_norm <= settings.tol_R:
            return NewtonResult(True, x, k, residual_norm)

        jacobian = jacobian_fd(residual, x, settings.fd_h)
        try:
            newton_step = np.linalg.solve(jacobian, -residual_value)
        except np.linalg.LinAlgError:
            return NewtonResult(False, x, k, residual_norm)

        step_norm = float(np.linalg.norm(newton_step))
        if step_norm <= settings.tol_dx:
            return NewtonResult(True, x, k, residual_norm)

        step_length = 1.0
        while step_length >= settings.alpha_min:
            trial_point = x + step_length * newton_step
            trial_residual_norm = float(np.linalg.norm(residual(trial_point)))
            sufficient_decrease = (1.0 - settings.armijo_c * step_length) * residual_norm
            if trial_residual_norm <= sufficient_decrease:
                x = trial_point
                break
            step_length *= settings.backtrack_rho

        if step_length < settings.alpha_min:
            return NewtonResult(False, x, k, residual_norm)

    final_residual_norm = float(np.linalg.norm(residual(x)))
    return NewtonResult(False, x, settings.k_max, final_residual_norm)


def _secant_predictor(
    current_T: float,
    target_T: float,
    current_x: Vector,
    previous_step: Optional[ContinuationStep],
) -> Vector:
    """
    First-order predictor for continuation.

    - First step: constant predictor x0 = current_x
    - Otherwise: secant extrapolation in T
    """
    if previous_step is None:
        return current_x.copy()

    previous_T, previous_x = previous_step
    denominator = current_T - previous_T
    if abs(denominator) < 1e-15:
        return current_x.copy()

    tangent_approx = (current_x - previous_x) / denominator
    return current_x + tangent_approx * (target_T - current_T)


def continuation_in_T(
    make_residual: ResidualFactory,
    stN: NewtonSettings,
    stC: ContinuationSettings,
    x_init: Vector,
) -> List[ContinuationStep]:
    """
    Adaptive continuation in the scalar load parameter T.

    Parameters
    ----------
    make_residual:
        Callable returning the nonlinear residual R_T(x) at fixed load T.
    stN:
        Newton parameters.
    stC:
        Continuation / adaptive step parameters.
    x_init:
        Initial state (often log-stretches [ln lambda_r, ln lambda_z]).

    Returns
    -------
    list of (T, x)
        Discrete path starting at T=0.
    """
    current_T = 0.0
    current_step_size = float(stC.dT0)
    current_x = np.asarray(x_init, dtype=float).copy()

    path: List[ContinuationStep] = [(current_T, current_x.copy())]
    previous_accepted_step: Optional[ContinuationStep] = None

    while current_T < stC.T_max - 1e-15:
        target_T = min(current_T + current_step_size, stC.T_max)
        predictor = _secant_predictor(
            current_T=current_T,
            target_T=target_T,
            current_x=current_x,
            previous_step=previous_accepted_step,
        )

        residual_at_target = make_residual(target_T)
        newton_result = _newton_armijo_result(residual_at_target, predictor, stN)

        if newton_result.converged:
            previous_accepted_step = (current_T, current_x.copy())
            current_T = target_T
            current_x = newton_result.x
            path.append((current_T, current_x.copy()))

            if newton_result.n_iterations <= stC.k_good:
                current_step_size = min(stC.gamma_up * current_step_size, stC.dT_max)
            elif newton_result.n_iterations >= stC.k_bad:
                current_step_size = max(stC.gamma_down * current_step_size, stC.dT_min)
        else:
            current_step_size = max(stC.gamma_down * current_step_size, stC.dT_min)
            if current_step_size <= stC.dT_min + 1e-18:
                raise RuntimeError(
                    "Continuation failed near T={:.6g}, ||R||={:.3e}".format(
                        target_T, newton_result.residual_norm
                    )
                )

    return path
