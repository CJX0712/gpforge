"""Method layer: the flagship Gaussian Process method and Bayesian optimization.

* :class:`GpMethod` wraps :class:`gpforge.models.gp.GaussianProcess` and turns a
  fit/predict into a :class:`~gpforge.core.types.FitResult` (RMSE / NLPD /
  coverage) — the unit the quality gate inspects.
* :class:`BayesOpt` runs sequential model-based optimization (Expected
  Improvement or Upper-Confidence-Bound) on a black-box objective using the GP
  posterior as a surrogate. The acquisition is maximized over a *deterministic*
  candidate grid, so the whole loop is reproducible.
"""

from __future__ import annotations

import numpy as np

from gpforge.core.errors import MethodError
from gpforge.core.types import FitResult
from gpforge.models.gp import GaussianProcess


class GpMethod:
    """Fit an exact GP to a dataset and summarize predictive quality on test."""

    name = "gpforge"

    def __init__(
        self,
        kernel: str = "rbf",
        optimize: bool = True,
        max_iters: int = 60,
        jitter: float = 1e-8,
    ):
        self.kernel = kernel
        self.optimize = optimize
        self.max_iters = max_iters
        self.jitter = jitter
        self._model: GaussianProcess | None = None

    def fit_score(self, ds, nominal: float = 0.95) -> FitResult:
        model = GaussianProcess(
            kernel=self.kernel,
            optimize=self.optimize,
            max_iters=self.max_iters,
            jitter=self.jitter,
        ).fit(ds.X_train, ds.y_train)
        self._model = model
        pred = model.predict(ds.X_test)
        resid = ds.y_test - pred.mean
        rmse = float(np.sqrt(np.mean(resid**2)))
        # Negative log predictive density under Gaussian predictive.
        ll = -0.5 * np.log(2.0 * np.pi * pred.var) - 0.5 * (resid**2) / pred.var
        nlpd = float(-np.mean(ll))
        z = np.abs(resid) <= 1.959963985 * pred.std  # 95% normal interval
        coverage = float(np.mean(z)) if len(z) else 0.0
        return FitResult(
            name=self.name,
            rmse=rmse,
            nlpd=nlpd,
            coverage=coverage,
            nominal=nominal,
            meta={
                "kernel": self.kernel,
                "optimized": self.optimize,
                "n_train": int(ds.X_train.shape[0]),
                "mean_pred_var": float(np.mean(pred.var)),
            },
        )


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard-normal CDF via erf (pure NumPy, no SciPy dependency)."""
    import math

    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def _expected_improvement(mean: np.ndarray, std: np.ndarray, f_best: float) -> np.ndarray:
    """EI for *minimization*: E[max(f_best - f(x), 0)] under N(mean, std^2)."""
    std = np.maximum(std, 1e-12)
    u = (f_best - mean) / std
    ei = (f_best - mean) * _norm_cdf(u) + std * _norm_pdf(u)
    return np.maximum(ei, 0.0)


def _ucb(mean: np.ndarray, std: np.ndarray, kappa: float) -> np.ndarray:
    """UCB for minimization: prefer low mean, penalize uncertainty -> μ - κσ."""
    return mean - kappa * std


class BayesOpt:
    """Sequential Bayesian optimization on a black-box objective.

    Parameters
    ----------
    objective:
        Callable ``f(x) -> float`` taking a 1-D array ``(d,)``.
    bounds:
        ``(d, 2)`` array of ``[lo, hi]`` per dimension.
    kernel, optimize, max_iters, jitter:
        Forwarded to the surrogate GP.
    acquisition:
        ``"ei"`` (Expected Improvement) or ``"ucb"`` (Upper-Confidence-Bound).
    n_init:
        Initial random (deterministic) observations before the loop starts.
    bo_iters:
        Optimization iterations.
    grid:
        Candidate grid resolution per dimension (1D/2D) or candidate count (d>2).
    seed:
        Master seed (the surrogate and candidate sampling are both derived from it).
    """

    name = "BayesOpt"

    def __init__(
        self,
        objective,
        bounds: np.ndarray,
        kernel: str = "rbf",
        optimize: bool = True,
        max_iters: int = 60,
        jitter: float = 1e-8,
        acquisition: str = "ei",
        n_init: int = 5,
        bo_iters: int = 25,
        grid: int = 400,
        ucb_kappa: float = 2.0,
        seed: int = 0,
    ) -> None:
        self.objective = objective
        self.bounds = np.atleast_2d(np.asarray(bounds, dtype=float))
        if self.bounds.shape[1] != 2:
            raise MethodError("bounds must be shape (d, 2)")
        self.d = self.bounds.shape[0]
        self.kernel = kernel
        self.optimize = optimize
        self.max_iters = max_iters
        self.jitter = jitter
        self.acquisition = acquisition
        self.n_init = int(n_init)
        self.bo_iters = int(bo_iters)
        self.grid = int(grid)
        self.ucb_kappa = ucb_kappa
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def _candidates(self) -> np.ndarray:
        if self.d <= 2:
            axes = [np.linspace(self.bounds[i, 0], self.bounds[i, 1], self.grid) for i in range(self.d)]
            mesh = np.meshgrid(*axes)
            cand = np.stack([m.ravel() for m in mesh], axis=1)
            return cand
        # Higher dimensions: deterministic uniform candidates + corners.
        n = self.grid * 5
        cand = self._rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n, self.d))
        corners = np.array(np.meshgrid(*[self.bounds[:, 0], self.bounds[:, 1]]).T.tolist())
        corners = corners.reshape(-1, self.d)
        return np.vstack([cand, corners])

    def run(self) -> dict:
        # Deterministic initial design.
        X0 = self._rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(max(self.n_init, 1), self.d))
        X_obs = list(X0)
        y_obs = [float(self.objective(x)) for x in X0]
        history = []
        for _ in range(self.bo_iters):
            Xa = np.asarray(X_obs)
            ya = np.asarray(y_obs)
            gp = GaussianProcess(
                kernel=self.kernel,
                optimize=self.optimize,
                max_iters=self.max_iters,
                jitter=self.jitter,
            ).fit(Xa, ya)
            cand = self._candidates()
            pred = gp.predict(cand, full=False)  # latent posterior (noise-free objective)
            f_best = float(np.min(ya))
            if self.acquisition == "ei":
                acq = _expected_improvement(pred.mean, pred.std, f_best)
            elif self.acquisition == "ucb":
                acq = -_ucb(pred.mean, pred.std, self.ucb_kappa)  # negate: maximize
            else:
                raise MethodError(f"unknown acquisition {self.acquisition!r}")
            best = int(np.argmax(acq))
            x_next = cand[best]
            y_next = float(self.objective(x_next))
            X_obs.append(x_next)
            y_obs.append(y_next)
            history.append((x_next.copy(), y_next, f_best))
        Xa = np.asarray(X_obs)
        ya = np.asarray(y_obs)
        return {
            "best_x": Xa[int(np.argmin(ya))],
            "best_y": float(np.min(ya)),
            "init_best_y": float(np.min(y_obs[: max(self.n_init, 1)])),
            "history": history,
            "X_obs": Xa,
            "y_obs": ya,
            "n_evals": len(y_obs),
        }
