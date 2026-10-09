# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-09

### Added
- `GaussianProcess`: exact GP regression via Cholesky factorization, with
  RBF / Matérn-5-2 / Matérn-3-2 / periodic / linear kernels and ARD lengthscales.
- Hyperparameter optimization by maximizing the log-marginal-likelihood with
  L-BFGS-B (SciPy, Tier-0), with a fixed-hyperparameter NumPy fallback.
- `predict(..., full=...)`: posterior predictive with the observation-noise term
  included (for calibrated intervals on noisy targets) or excluded (latent
  posterior, as needed by noise-free Bayesian optimization).
- `BayesOpt`: sequential Bayesian optimization with Expected Improvement and
  Upper-Confidence-Bound acquisition on a deterministic candidate grid.
- Metrics: RMSE, negative log predictive density, predictive-interval coverage.
- `pipeline`: benchmark runner and pre-registered quality gate
  (coverage ≥ 0.90 at 95% nominal **and** GP RMSE ≤ 1.10 × ridge baseline).
- `cli`: `run` (benchmark + gate + JSON report) and `methods` subcommands.
- `examples/run_demo.py`: benchmark, gate, determinism check, ablation, BO demo.
- 29 pytest cases; `docs/architecture.md` and `docs/model_card.md`.
- CI: ruff lint/format, pytest, demo smoke on Python 3.10 & 3.13, secret scan.
- Docker image, Makefile, packaging metadata.

### Fixed
- Predictive variance originally omitted the observation-noise term, which made
  intervals severely over-confident (realized coverage 0.13–0.62 against a 0.95
  nominal). Added `+ σ²_noise` for noisy-target prediction; realized coverage is
  now 0.939–0.955.
- `np.math.erf` (removed in NumPy 2.x) replaced with `math.erf`.
- Benchmark presets now use the kernel that actually generated them
  (`preset_kernel`), instead of forcing RBF on periodic data.

### Authors
- 晨星 (Morning Star)
