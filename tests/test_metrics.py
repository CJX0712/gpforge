"""Tests for predictive-quality metrics and their invariants."""

from __future__ import annotations

import numpy as np
import pytest

from gpforge.core.errors import EvalError
from gpforge.eval.metrics import interval_coverage, nlpd, rmse, z_for_nominal


def test_rmse_perfect_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y = np.array([0.0, 0.0, 0.0])
    m = np.array([1.0, -1.0, 0.0])
    assert rmse(y, m) == pytest.approx(np.sqrt((1 + 1 + 0) / 3))


def test_nlpd_finite_positive_var():
    y = np.array([0.0, 1.0])
    m = np.array([0.0, 1.0])
    s = np.array([0.5, 0.5])
    assert nlpd(y, m, s) > 0.0


def test_nlpd_rejects_nonpositive_var():
    with pytest.raises(EvalError):
        nlpd(np.array([0.0]), np.array([0.0]), np.array([0.0]))


def test_coverage_exact_normal():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20000)
    m = np.zeros_like(y)
    s = np.ones_like(y)
    cov = interval_coverage(y, m, s, nominal=0.95)
    assert cov == pytest.approx(0.95, abs=0.01)


def test_z_for_nominal_known():
    assert z_for_nominal(0.95) == pytest.approx(1.959963985, abs=1e-6)
    assert z_for_nominal(0.99) == pytest.approx(2.575829304, abs=1e-6)
