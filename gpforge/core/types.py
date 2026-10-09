"""Lightweight data containers (dataclasses) shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GPDataset:
    """A regression dataset for Gaussian Process fitting.

    Attributes
    ----------
    X_train, y_train:
        Training inputs ``(n, d)`` and targets ``(n,)``.
    X_test, y_test:
        Held-out inputs ``(m, d)`` and targets ``(m,)``.
    noise_std:
        The observation noise standard deviation used by the DGP (the
        generative truth, used only for reporting / coverage checks).
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    noise_std: float


@dataclass
class PredResult:
    """Posterior predictive summaries at query points."""

    mean: np.ndarray  # (q,) predictive mean
    std: np.ndarray  # (q,) predictive std (sqrt of variance)
    var: np.ndarray  # (q,) predictive variance (>= 0)


@dataclass
class FitResult:
    """Outcome of fitting a GP (or baseline) to a dataset."""

    name: str
    rmse: float
    nlpd: float  # negative log predictive density (lower better)
    coverage: float  # fraction of test y inside 95% predictive interval
    nominal: float = 0.95
    meta: dict = field(default_factory=dict)


@dataclass
class GateResult:
    """Per-dataset + aggregate fairness/quality gate verdict."""

    passed: bool
    per_dataset: dict = field(default_factory=dict)
    aggregate: dict = field(default_factory=dict)
