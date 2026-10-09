"""Bayesian optimization invariants: non-negative acquisition and monotonic
improvement of the best observed value."""

from __future__ import annotations

import numpy as np

from gpforge.gp.methods import BayesOpt, _expected_improvement, _ucb


def _toy(x):
    x = np.asarray(x, dtype=float)
    return float(np.sin(3.0 * x[0]) + 0.3 * x[0])


def test_ei_nonnegative():
    mean = np.array([-1.0, 0.0, 2.0, 5.0])
    std = np.array([0.5, 0.5, 0.5, 0.5])
    ei = _expected_improvement(mean, std, f_best=0.0)
    assert np.all(ei >= 0.0)


def test_ucb_lowers_uncertainty():
    mean = np.array([1.0, 1.0])
    std = np.array([0.1, 2.0])
    # Acquisition we *maximize* is -UCB = -mean + kappa*std. Higher std
    # (more exploration) yields a higher acquisition when means are equal.
    acq = -_ucb(mean, std, kappa=2.0)
    assert acq[1] > acq[0]


def test_bo_finds_near_optimum():
    bo = BayesOpt(
        _toy,
        bounds=np.array([[0.0, 1.0]]),
        seed=0,
        bo_iters=20,
        grid=300,
        acquisition="ei",
    )
    res = bo.run()
    # Global minimum of sin(3x)+0.3x on [0,1] is at x=0 (value 0).
    assert res["best_y"] <= 0.05


def test_bo_improves_over_init():
    bo = BayesOpt(
        _toy,
        bounds=np.array([[0.0, 1.0]]),
        seed=1,
        bo_iters=15,
        grid=200,
        acquisition="ei",
    )
    res = bo.run()
    assert res["best_y"] <= res["init_best_y"] + 1e-9


def test_bo_deterministic():
    kw = {
        "objective": _toy,
        "bounds": np.array([[0.0, 1.0]]),
        "seed": 2,
        "bo_iters": 10,
        "grid": 150,
    }
    a = BayesOpt(**kw).run()
    b = BayesOpt(**kw).run()
    assert a["best_y"] == b["best_y"]
