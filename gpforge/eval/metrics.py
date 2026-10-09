"""Predictive-quality metrics for Gaussian Process evaluation.

These are the primitives the quality gate consumes. The headline invariant we
verify in tests is *calibration*: on a well-specified GP the realized 95%
predictive-interval coverage must approach the nominal 95% as the test set
grows — otherwise the posterior variance is mis-scaled.
"""

from __future__ import annotations

import numpy as np

from gpforge.core.errors import EvalError

# z-values for common nominal levels (two-sided normal interval).
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def z_for_nominal(nominal: float) -> float:
    if nominal in _Z:
        return _Z[nominal]
    # General inverse via erf; validated for the standard cases.
    from math import sqrt

    return sqrt(2.0) * _erfinv(2.0 * nominal - 1.0)


def _erfinv(x: float) -> float:
    # Rational approximation (Abramowitz & Stegun 7.1.26) — deterministic.
    a = [0.0, 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - ((((a[5] * t + a[4]) * t + a[3]) * t + a[2]) * t + a[1]) * t * np.exp(-(x**2))
    return np.sign(x) * y if hasattr(np, "sign") else (1.0 if x >= 0 else -1.0) * y


def rmse(y_true: np.ndarray, mean: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    mean = np.asarray(mean, dtype=float).ravel()
    if y_true.shape != mean.shape or y_true.size == 0:
        raise EvalError("y_true and mean must be equal non-empty length")
    return float(np.sqrt(np.mean((y_true - mean) ** 2)))


def nlpd(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Mean negative log predictive density under a Gaussian predictive."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    mean = np.asarray(mean, dtype=float).ravel()
    var = np.asarray(std, dtype=float).ravel() ** 2
    if not (y_true.shape == mean.shape == var.shape) or y_true.size == 0:
        raise EvalError("y_true, mean, std must be equal non-empty length")
    if np.any(var <= 0):
        raise EvalError("predictive variance must be > 0 for NLPD")
    ll = -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (y_true - mean) ** 2 / var
    return float(-np.mean(ll))


def interval_coverage(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, nominal: float = 0.95) -> float:
    """Fraction of true values inside the two-sided normal predictive interval."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    mean = np.asarray(mean, dtype=float).ravel()
    std = np.asarray(std, dtype=float).ravel()
    z = z_for_nominal(nominal)
    inside = np.abs(y_true - mean) <= z * std
    return float(np.mean(inside))


def evaluate(model_mean, model_std, y_true, nominal: float = 0.95) -> dict:
    """One-shot metric bundle for a predictive distribution."""
    return {
        "rmse": rmse(y_true, model_mean),
        "nlpd": nlpd(y_true, model_mean, model_std),
        "coverage": interval_coverage(y_true, model_mean, model_std, nominal),
        "nominal": nominal,
    }
