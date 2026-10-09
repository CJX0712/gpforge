"""Calibration: on a *well-specified* GP the realized coverage must approach
the nominal level (within a finite-N slack). This is the headline correctness
invariant of any credible GP implementation."""

from __future__ import annotations

from gpforge.core.seed import set_all
from gpforge.data.synthesizer import make_dataset
from gpforge.eval.metrics import interval_coverage
from gpforge.models.gp import GaussianProcess


def test_coverage_at_nominal_with_large_test():
    set_all(7)
    ds = make_dataset(80, 4000, preset="rbf_1d", noise_std=0.10, seed=7)
    gp = GaussianProcess(kernel="rbf", optimize=True).fit(ds.X_train, ds.y_train)
    pred = gp.predict(ds.X_test)
    cov = interval_coverage(ds.y_test, pred.mean, pred.std, nominal=0.95)
    # Realized coverage should be within 0.05 of the 0.95 nominal.
    assert cov >= 0.90


def test_coverage_worse_without_noise_in_interval():
    """If we (incorrectly) drop the observation-noise term from the predictive
    variance, coverage collapses — guarding the calibration invariant."""
    set_all(8)
    ds = make_dataset(80, 3000, preset="rbf_1d", noise_std=0.15, seed=8)
    gp = GaussianProcess(kernel="rbf", optimize=True).fit(ds.X_train, ds.y_train)
    pred = gp.predict(ds.X_test, full=False)  # latent variance only (no noise)
    cov_latent = interval_coverage(ds.y_test, pred.mean, pred.std, nominal=0.95)
    pred_full = gp.predict(ds.X_test, full=True)
    cov_full = interval_coverage(ds.y_test, pred.mean, pred_full.std, nominal=0.95)
    assert cov_full > cov_latent
