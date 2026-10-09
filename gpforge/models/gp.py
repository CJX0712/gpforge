"""Exact Gaussian Process regression with Cholesky inference.

The posterior is computed *exactly* (no approximation) via the Cholesky factor
of the noisy kernel matrix. With the RBF/Matérn/periodic/linear kernels the
kernel matrix is positive definite, so:

* the predictive variance ``var(x*) = k** - k*(x*)^T K^{-1} k*(x*)`` is
  guaranteed non-negative (Schur complement of a PD block), and
* the predictive mean interpolates the training targets in the noise-free
  limit (``var -> 0`` and ``mean -> y`` at the training points).

Hyperparameters (lengthscales, signal variance, noise variance) are optimized by
maximizing the log-marginal-likelihood with L-BFGS-B (SciPy, Tier-0). When
SciPy is unavailable we fall back to *fixed* hyperparameters (pure NumPy,
zero external ML dependencies) so the toolkit still runs anywhere.
"""

from __future__ import annotations

import numpy as np

from gpforge.core.errors import ModelError
from gpforge.core.types import PredResult
from gpforge.models.kernels import kernel_matrix

_NAME = "ExactGP"


def available_scipy() -> bool:
    """Return True if SciPy can be imported (used for hyperparameter optim)."""
    try:
        import scipy.optimize  # noqa: F401

        return True
    except Exception:  # noqa: BLE001  # pragma: no cover - environment dependent
        return False


class GaussianProcess:
    """Exact GP regressor.

    Parameters
    ----------
    kernel:
        One of ``rbf``, ``matern52``, ``matern32``, ``periodic``, ``linear``.
    optimize:
        If True and SciPy is present, maximize the log-marginal-likelihood.
        Otherwise the (reasonable) initial hyperparameters are kept.
    max_iters:
        L-BFGS-B iteration cap.
    jitter:
        Diagonal stabilizer added to the kernel matrix (numerical safety).
    """

    name = _NAME

    def __init__(
        self,
        kernel: str = "rbf",
        optimize: bool = True,
        max_iters: int = 60,
        jitter: float = 1e-8,
    ) -> None:
        self.kernel = kernel
        self.optimize = optimize
        self.max_iters = max_iters
        self.jitter = float(jitter)
        # Hyperparameters in log-space.
        self._log_ls: np.ndarray | None = None
        self._log_sf2: float | None = None
        self._log_noise2: float | None = None
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._trained = False

    # ---- internals -------------------------------------------------------
    def _hyper_to_params(self):
        ls = np.exp(self._log_ls)
        sf2 = float(np.exp(self._log_sf2))
        noise2 = float(np.exp(self._log_noise2))
        return ls, sf2, noise2

    def _pack(self) -> np.ndarray:
        return np.concatenate([self._log_ls, [self._log_sf2, self._log_noise2]])

    def _unpack(self, theta: np.ndarray) -> None:
        d = len(self._log_ls)
        self._log_ls = theta[:d]
        self._log_sf2 = float(theta[d])
        self._log_noise2 = float(theta[d + 1])

    def _log_marginal_likelihood(self, theta: np.ndarray) -> float:
        self._unpack(theta)
        ls, sf2, noise2 = self._hyper_to_params()
        K = kernel_matrix(self._X, self._X, self.kernel, ls, sf2, period=self._period)
        K = K + (noise2 + self.jitter) * np.eye(len(self._X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - numerically guarded
            raise ModelError("kernel matrix not PD; increase jitter") from exc
        y = self._y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        ll = -0.5 * float(y @ alpha) - np.sum(np.log(np.diag(L))) - 0.5 * len(y) * np.log(2.0 * np.pi)
        return float(ll)

    def _neg_ll_and_grad(self, theta: np.ndarray, eps: float = 1e-4):
        f0 = -self._log_marginal_likelihood(theta)
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            tp = theta.copy()
            tm = theta.copy()
            tp[i] += eps
            tm[i] -= eps
            fp = -self._log_marginal_likelihood(tp)
            fm = -self._log_marginal_likelihood(tm)
            grad[i] = (fp - fm) / (2.0 * eps)
        return f0, grad

    # ---- public API ------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcess:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ModelError("X and y row counts differ")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ModelError("non-finite X or y")
        self._X = X
        self._y = y
        self._period = 1.0  # single period knob; fixed for periodic preset
        # Sensible log-space initialization (ARD lengthscales ~ data range).
        ranges = np.ptp(X, axis=0)
        self._log_ls = np.log(np.clip(ranges, 1e-2, None) + 1e-1)
        ystd = float(np.std(y)) + 1e-3
        self._log_sf2 = np.log(ystd**2)
        self._log_noise2 = np.log((0.1 * ystd) ** 2 + 1e-6)

        if self.optimize and available_scipy():
            try:
                from scipy.optimize import minimize

                theta0 = self._pack()
                bounds = [(-6.0, 4.0)] * len(theta0)
                res = minimize(
                    lambda t: self._neg_ll_and_grad(t),
                    theta0,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": self.max_iters},
                )
                self._unpack(res.x)
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - optimizer failure
                # Keep initial hyperparameters; model still valid.
                self._unpack(theta0)
                self._last_opt_error = str(exc)

        # Finalize Cholesky factors with the (optimized) hyperparameters.
        ls, sf2, noise2 = self._hyper_to_params()
        K = kernel_matrix(self._X, self._X, self.kernel, ls, sf2, period=self._period)
        K = K + (noise2 + self.jitter) * np.eye(len(self._X))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        self._noise2 = noise2
        self._trained = True
        return self

    def fit_fixed(self, X: np.ndarray, y: np.ndarray, log_ls, log_sf2, log_noise2) -> GaussianProcess:
        """Fit with *explicit* log-space hyperparameters (no init, no optim).

        Used by tests and by any caller that wants a fully prescribed GP
        (e.g. a warm start). Builds the Cholesky factor immediately.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ModelError("X and y row counts differ")
        self._X = X
        self._y = y
        self._period = 1.0
        self._log_ls = np.asarray(log_ls, dtype=float).ravel()
        self._log_sf2 = float(log_sf2)
        self._log_noise2 = float(log_noise2)
        ls, sf2, noise2 = self._hyper_to_params()
        K = kernel_matrix(self._X, self._X, self.kernel, ls, sf2, period=self._period)
        K = K + (noise2 + self.jitter) * np.eye(len(self._X))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        self._noise2 = noise2
        self._trained = True
        return self

    def predict(self, Xq: np.ndarray, full: bool = True) -> PredResult:
        """Posterior predictive at ``Xq``.

        Parameters
        ----------
        full:
            If True (default, for noisy regression), the predictive variance
            includes the learned observation-noise variance, so the 95%
            predictive interval is calibrated to noisy test targets. If False
            (latent posterior, used by Bayesian optimization on noise-free
            objectives), only the epistemic variance is returned.
        """
        if not self._trained:
            raise ModelError("predict called before fit")
        Xq = np.atleast_2d(np.asarray(Xq, dtype=float))
        ls, sf2, noise2 = self._hyper_to_params()
        Ks = kernel_matrix(self._X, Xq, self.kernel, ls, sf2, period=self._period)  # (n, q)
        mean = Ks.T @ self._alpha
        # v = L^{-1} Ks  ->  latent predictive var = sf2 - sum(v^2)
        v = np.linalg.solve(self._L, Ks)
        var = sf2 - np.sum(v**2, axis=0)
        if full:
            var = var + noise2  # include observation noise for noisy targets
        var = np.maximum(var, 0.0) + self.jitter  # enforce >= 0 invariant
        std = np.sqrt(var)
        return PredResult(mean=mean, std=std, var=var)


# Convenience baseline regressors used by the pipeline's quality gate.


class ConstantMean:
    """Trivial baseline: predicts the training mean everywhere."""

    name = "ConstantMean"

    def fit(self, X: np.ndarray, y: np.ndarray) -> ConstantMean:
        self._mu = float(np.mean(np.asarray(y, dtype=float).ravel()))
        self._std_y = float(np.std(np.asarray(y, dtype=float).ravel())) or 1.0
        return self

    def predict(self, Xq: np.ndarray) -> PredResult:
        n = np.atleast_2d(Xq).shape[0]
        m = np.full(n, self._mu)
        s = np.full(n, self._std_y)
        return PredResult(mean=m, std=s, var=s**2)


class LinearRidge:
    """Ridge-regression baseline (closed-form), with predictive variance from
    residual spread — a dependency-free reference the GP must beat on RMSE."""

    name = "LinearRidge"

    def __init__(self, alpha: float = 1e-2) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRidge:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        self._bias = True
        A = np.hstack([X, np.ones((X.shape[0], 1))]) if self._bias else X
        reg = np.eye(A.shape[1]) * self.alpha
        self._w = np.linalg.solve(A.T @ A + reg, A.T @ y)
        pred = A @ self._w
        self._sigma2 = float(np.mean((y - pred) ** 2)) + 1e-6
        return self

    def predict(self, Xq: np.ndarray) -> PredResult:
        Xq = np.atleast_2d(np.asarray(Xq, dtype=float))
        A = np.hstack([Xq, np.ones((Xq.shape[0], 1))]) if self._bias else Xq
        mean = A @ self._w
        s = np.full(Xq.shape[0], np.sqrt(self._sigma2))
        return PredResult(mean=mean, std=s, var=s**2)
