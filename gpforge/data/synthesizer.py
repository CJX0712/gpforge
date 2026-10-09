"""Synthetic data generators for Gaussian Process benchmarking.

The data layer produces *ground-truth* regression problems whose generative
process is itself a Gaussian Process with a known kernel. That lets us check the
single most important correctness invariant of any GP implementation: on a
well-specified model the predictive intervals must achieve their nominal
coverage, and the posterior must interpolate the training data in the
noise-free limit.
"""

from __future__ import annotations

import numpy as np

from gpforge.core.errors import DataError
from gpforge.core.seed import set_all
from gpforge.core.types import GPDataset

# Default 1-D kernel parameters for the latent generative GP.
_PRESETS: dict[str, dict] = {
    "rbf_1d": {
        "d": 1,
        "kernel": "rbf",
        "lengthscale": 0.4,
        "signal_var": 1.0,
        "bounds": (0.0, 1.0),
    },
    "matern52_1d": {
        "d": 1,
        "kernel": "matern52",
        "lengthscale": 0.35,
        "signal_var": 1.2,
        "bounds": (0.0, 1.0),
    },
    "periodic_1d": {
        "d": 1,
        "kernel": "periodic",
        "lengthscale": 0.2,
        "signal_var": 1.0,
        "period": 0.3,
        "bounds": (0.0, 1.0),
    },
    "rbf_2d": {
        "d": 2,
        "kernel": "rbf",
        "lengthscale": 0.5,
        "signal_var": 1.0,
        "bounds": (0.0, 1.0),
    },
}


def _kernel_matrix(
    Xa: np.ndarray,
    Xb: np.ndarray,
    kind: str,
    ell: float,
    sf2: float,
    period: float = 1.0,
) -> np.ndarray:
    Xa = np.atleast_2d(Xa)
    Xb = np.atleast_2d(Xb)
    if kind == "rbf":
        sq = np.sum(Xa**2, 1)[:, None] + np.sum(Xb**2, 1)[None, :] - 2.0 * (Xa @ Xb.T)
        return sf2 * np.exp(-0.5 * sq / max(ell**2, 1e-12))
    if kind == "matern52":
        sq = np.sum((Xa[:, None, :] - Xb[None, :, :]) ** 2, 2)
        r = np.sqrt(5.0 * sq / max(ell**2, 1e-12))
        return sf2 * (1.0 + r + r**2 / 3.0) * np.exp(-r)
    if kind == "periodic":
        diff = Xa[:, None, :] - Xb[None, :, :]
        s = np.sin(np.pi * np.sqrt(np.sum(diff**2, 2)) / max(period, 1e-12))
        return sf2 * np.exp(-2.0 * s**2 / max(ell**2, 1e-12))
    if kind == "linear":
        return sf2 * (Xa @ Xb.T)
    raise DataError(f"unknown kernel kind {kind!r}")


def make_dataset(
    n_train: int,
    n_test: int,
    preset: str = "rbf_1d",
    noise_std: float = 0.10,
    seed: int = 0,
) -> GPDataset:
    """Draw a train/test regression problem from a latent GP.

    The latent function ``f ~ GP(0, k)`` is sampled once (shared by train and
    test), then observations are ``y = f(x) + N(0, noise_std^2)``.
    """
    if preset not in _PRESETS:
        raise DataError(f"unknown preset {preset!r}; choose from {sorted(_PRESETS)}")
    p = dict(_PRESETS[preset])
    d = p["d"]
    bounds = p["bounds"]
    rng = set_all(seed)

    n_tot = n_train + n_test
    X = rng.uniform(bounds[0], bounds[1], size=(n_tot, d))

    # Sample the latent GP function on all inputs (shared realization).
    K = _kernel_matrix(
        X,
        X,
        p["kernel"],
        p["lengthscale"],
        p["signal_var"],
        period=p.get("period", 1.0),
    )
    K = K + 1e-9 * np.eye(n_tot)  # generative stability
    L = np.linalg.cholesky(K)
    f = L @ rng.standard_normal(n_tot)

    y = f + rng.normal(0.0, noise_std, size=n_tot)

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    if not (
        np.all(np.isfinite(X_train))
        and np.all(np.isfinite(y_train))
        and np.all(np.isfinite(X_test))
        and np.all(np.isfinite(y_test))
    ):
        raise DataError("non-finite values in generated dataset")
    return GPDataset(X_train, y_train, X_test, y_test, float(noise_std))


def make_benchmark_suite() -> list[str]:
    """Return the preset names used by the default benchmark."""
    return ["rbf_1d", "matern52_1d", "periodic_1d", "rbf_2d"]


def preset_kernel(preset: str) -> str:
    """Return the *well-specified* kernel for a preset (used by the pipeline so
    each benchmark dataset is modeled by the kernel that actually generated it)."""
    if preset not in _PRESETS:
        raise DataError(f"unknown preset {preset!r}")
    return _PRESETS[preset]["kernel"]
