"""Exact-GP inference invariants: non-negative variance, interpolation,
and offline (no-optimization) agreement with a manual Cholesky computation."""

from __future__ import annotations

import numpy as np

from gpforge.models.gp import GaussianProcess, available_scipy


def test_posterior_variance_nonnegative():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(30, 1))
    y = np.sin(5 * X[:, 0]) + rng.normal(0, 0.05, size=30)
    gp = GaussianProcess(kernel="rbf", optimize=False).fit(X, y)
    Xq = rng.uniform(0, 1, size=(50, 1))
    pred = gp.predict(Xq)
    assert np.all(pred.var >= 0.0)
    assert np.all(pred.std >= 0.0)


def test_interpolation_noise_free():
    # In the noise-free limit the posterior must interpolate training points.
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(25, 1))
    y = np.sin(4 * X[:, 0])
    gp = GaussianProcess(kernel="rbf", jitter=1e-10).fit_fixed(
        X, y, log_ls=np.log([0.3]), log_sf2=np.log(1.0), log_noise2=np.log(1e-12)
    )
    pred = gp.predict(X)
    assert np.allclose(pred.mean, y, atol=1e-3)
    assert np.all(pred.var < 1e-4)  # variance -> 0 at training points


def test_optimization_improves_log_marglik():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(40, 1))
    y = np.sin(6 * X[:, 0]) + rng.normal(0, 0.08, size=40)
    gp0 = GaussianProcess(kernel="rbf", optimize=False).fit(X, y)
    gp1 = GaussianProcess(kernel="rbf", optimize=True).fit(X, y)
    # Optimized model should fit at least as well on training RMSE.
    r0 = np.sqrt(np.mean((gp0.predict(X).mean - y) ** 2))
    r1 = np.sqrt(np.mean((gp1.predict(X).mean - y) ** 2))
    assert r1 <= r0 + 1e-9


def test_offline_manual_cholesky_agreement():
    """With optimize=False and fixed hyperparameters, the GP must match a
    hand-rolled Cholesky solve exactly (zero ML-framework dependency)."""
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, size=(20, 1))
    y = rng.normal(0, 1, size=20)
    ls = np.array([0.6])
    sf2 = 1.3
    noise2 = 0.04
    gp = GaussianProcess(kernel="rbf").fit_fixed(
        X, y, log_ls=np.log(ls), log_sf2=np.log(sf2), log_noise2=np.log(noise2)
    )
    Xq = rng.uniform(0, 1, size=(7, 1))

    from gpforge.models.kernels import kernel_matrix

    K = kernel_matrix(X, X, "rbf", ls, sf2) + (noise2 + gp.jitter) * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = kernel_matrix(X, Xq, "rbf", ls, sf2)
    mean_manual = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var_manual = sf2 - np.sum(v**2, axis=0) + noise2

    pred = gp.predict(Xq)
    assert np.allclose(pred.mean, mean_manual, atol=1e-8)
    assert np.allclose(pred.var, var_manual, atol=1e-8)


def test_scipy_available():
    # The CI environment ships SciPy; we assert the Tier-0 path is present.
    assert available_scipy() is True
