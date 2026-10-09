"""Protocols (structural typing) for swappable components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from gpforge.core.types import PredResult


@runtime_checkable
class Regressor(Protocol):
    """Anything that can fit a regression and predict with uncertainty."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> Regressor: ...

    def predict(self, Xq: np.ndarray) -> PredResult: ...
