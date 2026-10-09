"""Configuration with environment-variable overrides and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass

from gpforge.core.errors import ConfigError


@dataclass
class Config:
    """Global configuration for gpforge.

    Every field may be overridden by an ``GPFORGE_*`` environment variable
    (uppercased, prefixed). Values are coerced to the declared type.
    """

    seed: int = 0
    n_train: int = 60
    n_test: int = 600
    noise_std: float = 0.10  # generative observation noise (truth)
    jitter: float = 1e-8  # numerical stability added to K
    nominal: float = 0.95  # predictive-interval coverage target
    coverage_floor: float = 0.90  # gate: realized coverage >= this
    rmse_factor: float = 1.10  # gate: gp.rmse <= rmse_factor * baseline.rmse
    optimize: bool = True  # hyperparameter optimization (L-BFGS via scipy)
    max_opt_iters: int = 60
    bo_iters: int = 25  # Bayesian-optimization iterations in demo
    bo_grid: int = 400  # acquisition grid resolution (deterministic)
    acquisition: str = "ei"  # "ei" | "ucb"
    ucb_kappa: float = 2.0

    _ENV_PREFIX = "GPFORGE_"

    def __post_init__(self) -> None:
        self._apply_env()
        self.validate()

    def _apply_env(self) -> None:
        for f in self.__dataclass_fields__:  # type: ignore[attr-defined]
            env = f"{self._ENV_PREFIX}{f.upper()}"
            if env in os.environ:
                raw = os.environ[env]
                cur = getattr(self, f)
                try:
                    if isinstance(cur, bool):
                        setattr(self, f, raw.lower() in ("1", "true", "yes"))
                    elif isinstance(cur, int):
                        setattr(self, f, int(raw))
                    elif isinstance(cur, float):
                        setattr(self, f, float(raw))
                    else:
                        setattr(self, f, raw)
                except ValueError as exc:  # pragma: no cover - bad env input
                    raise ConfigError(f"cannot parse {env}={raw!r}: {exc}") from exc

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError("seed must be >= 0")
        if self.n_train < 2 or self.n_test < 1:
            raise ConfigError("n_train>=2, n_test>=1 required")
        if self.noise_std < 0 or self.jitter <= 0:
            raise ConfigError("noise_std>=0, jitter>0 required")
        if not 0.0 < self.nominal < 1.0:
            raise ConfigError("nominal must be in (0,1)")
        if not 0.0 < self.coverage_floor <= self.nominal:
            raise ConfigError("0 < coverage_floor <= nominal required")
        if self.rmse_factor <= 0:
            raise ConfigError("rmse_factor must be > 0")
        if self.acquisition not in ("ei", "ucb"):
            raise ConfigError("acquisition must be 'ei' or 'ucb'")
        if self.max_opt_iters < 1 or self.bo_iters < 0 or self.bo_grid < 2:
            raise ConfigError("max_opt_iters>=1, bo_grid>=2 required")
