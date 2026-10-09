"""Shared pytest fixtures for gpforge."""

from __future__ import annotations

import pytest

from gpforge.core.config import Config
from gpforge.core.seed import set_all
from gpforge.data.synthesizer import make_dataset


@pytest.fixture
def base_config():
    return Config(n_train=60, n_test=600, noise_std=0.10, optimize=True, seed=0)


@pytest.fixture
def a_dataset(base_config):
    return make_dataset(
        base_config.n_train,
        base_config.n_test,
        preset="rbf_1d",
        noise_std=base_config.noise_std,
        seed=0,
    )


@pytest.fixture(autouse=True)
def _reseed():
    """Every test starts from a clean global RNG state."""
    set_all(12345)
    yield
