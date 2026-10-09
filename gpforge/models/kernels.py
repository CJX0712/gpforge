"""Stationary and non-stationary covariance kernels.

Every kernel is positive semi-definite by construction (Mercer kernels), which
is the invariant that guarantees the posterior predictive variance is
non-negative. We implement:

* ``rbf``        — squared-exponential, infinitely differentiable
* ``matern52``   — Matérn nu=5/2, once differentiable
* ``matern32``   — Matérn nu=3/2
* ``periodic``   — Exp-Sine-Squared
* ``linear``     — dot-product (ARD-free)

Hyperparameters are held in log-space internally so unconstrained optimization
stays stable.
"""

from __future__ import annotations

import numpy as np

from gpforge.core.errors import ModelError

_VALID = {"rbf", "matern52", "matern32", "periodic", "linear"}


def kernel_matrix(
    Xa: np.ndarray,
    Xb: np.ndarray,
    kind: str,
    lengthscales: np.ndarray,
    signal_var: float,
    period: float = 1.0,
) -> np.ndarray:
    """Evaluate a kernel matrix ``K[Xa, Xb]``.

    Parameters
    ----------
    lengthscales:
        Per-dimension lengthscales ``(d,)`` (ARD). All entries must be > 0.
    signal_var:
        Output (signal) variance > 0.
    """
    Xa = np.atleast_2d(np.asarray(Xa, dtype=float))
    Xb = np.atleast_2d(np.asarray(Xb, dtype=float))
    if Xa.shape[1] != Xb.shape[1]:
        raise ModelError("Xa and Xb must share input dimension")
    if kind not in _VALID:
        raise ModelError(f"unknown kernel {kind!r}; choose from {sorted(_VALID)}")
    ls = np.asarray(lengthscales, dtype=float)
    if np.any(ls <= 0) or signal_var <= 0 or period <= 0:
        raise ModelError("lengthscales, signal_var and period must be > 0")

    # ARD-scaled pairwise squared distances.
    Xas = Xa / ls
    Xbs = Xb / ls
    sq = np.sum(Xas**2, 1)[:, None] + np.sum(Xbs**2, 1)[None, :] - 2.0 * (Xas @ Xbs.T)

    if kind == "rbf":
        return signal_var * np.exp(-0.5 * np.clip(sq, 0.0, None))
    if kind == "matern52":
        r = np.sqrt(np.clip(5.0 * sq, 0.0, None))
        return signal_var * (1.0 + r + r**2 / 3.0) * np.exp(-r)
    if kind == "matern32":
        r = np.sqrt(np.clip(3.0 * sq, 0.0, None))
        return signal_var * (1.0 + r) * np.exp(-r)
    if kind == "periodic":
        # distance on the circle (uses unscaled coords)
        diff = Xa[:, None, :] - Xb[None, :, :]
        s = np.sin(np.pi * np.sqrt(np.clip(np.sum(diff**2, 2), 0.0, None)) / period)
        return signal_var * np.exp(-2.0 * s**2 / max(np.mean(ls) ** 2, 1e-12))
    if kind == "linear":
        return signal_var * (Xa @ Xbs.T)
    raise ModelError("unreachable")  # pragma: no cover
