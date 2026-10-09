"""Global deterministic seeding.

A single unidirectional entry point so every randomness source (numpy, the
python ``random`` module, and the ``PYTHONHASHSEED`` used by set/hash
ordering) is pinned before any computation. Determinism is a first-class
requirement of gpforge: two identical runs must produce byte-identical metrics.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_all(seed: int) -> np.random.Generator:
    """Seed every RNG source and return a numpy ``Generator`` for callers.

    Parameters
    ----------
    seed:
        Non-negative integer. Must be the *first* call in any reproducible
        pipeline (no other module may seed before this).

    Returns
    -------
    numpy.random.Generator
        A dedicated generator that callers should thread through rather than
        re-seeding, keeping the dependency graph acyclic.
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    os.environ["PYTHONHASHSEED"] = str(seed % (2**32))
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
