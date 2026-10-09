"""Kernel correctness: positive-definiteness, symmetry, value sanity."""

from __future__ import annotations

import numpy as np
import pytest

from gpforge.models.kernels import kernel_matrix


def test_rbf_psd():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(20, 2))
    K = kernel_matrix(X, X, "rbf", np.array([0.5, 0.5]), 1.0)
    eig = np.linalg.eigvalsh(K)
    assert np.all(eig > -1e-8)  # PSD invariant


def test_matern52_psd():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(15, 1))
    K = kernel_matrix(X, X, "matern52", np.array([0.4]), 1.2)
    eig = np.linalg.eigvalsh(K)
    assert np.all(eig > -1e-8)


def test_periodic_psd():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(15, 1))
    K = kernel_matrix(X, X, "periodic", np.array([0.2]), 1.0, period=0.3)
    eig = np.linalg.eigvalsh(K)
    assert np.all(eig > -1e-8)


def test_symmetry_and_diagonal():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, size=(10, 1))
    K = kernel_matrix(X, X, "rbf", np.array([0.5]), 1.0)
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.allclose(np.diag(K), 1.0, atol=1e-12)  # signal_var=1 on the diagonal


def test_kronecker_delta_at_zero_for_rbf():
    # Two identical inputs => covariance = signal_var.
    X = np.array([[0.3]])
    K = kernel_matrix(X, X, "rbf", np.array([0.5]), 2.0)
    assert K[0, 0] == pytest.approx(2.0)
