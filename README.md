# gpforge

**World-class, deterministic Gaussian Process toolkit** — exact GP regression with
automatic hyperparameter optimization and sequential Bayesian optimization.

> Author: **晨星** (Morning Star) · License: MIT · CPU-only · zero GPU needed

---

## What this is

`gpforge` is a from-first-principles implementation of Gaussian Process (GP)
regression and model-based optimization, built around **verifiable invariants**
rather than hand-waved "it learns". The flagship deliverables:

| Component | What it does | Core math |
|---|---|---|
| `GaussianProcess` | Exact GP regression with Cholesky inference | Posterior `p(f* \| D)` via `L = chol(K + σ²I)` |
| Kernels | `rbf`, `matern52`, `matern32`, `periodic`, `linear` with ARD | Mercer positive-definiteness |
| Hyperopt | Maximize log-marginal-likelihood with L-BFGS-B | `log p(y\|X,θ)`, analytic-in-spirit numerical gradients |
| `BayesOpt` | Sequential model-based optimization over a deterministic grid | Expected Improvement / Upper-Confidence-Bound |

---

## Why it is correct (not just plausible)

A GP implementation can produce smooth-looking curves and still be badly
mis-calibrated. `gpforge` is built so the following are **asserted**, not assumed:

1. **Non-negative predictive variance.** With PSD kernels, `var(x*) = k** −
   k*(x*)ᵀ K⁻¹ k*(x*)` is a Schur complement of a positive-definite block, hence
   ≥ 0. Enforced explicitly.
2. **Calibration.** On a *well-specified* GP, the realized 95% predictive
   interval coverage must approach 95%. This is the single strongest evidence
   the posterior variance is correctly scaled — we measure it, we don't assume it.
3. **Interpolation.** In the noise-free limit the posterior mean reproduces the
   training targets and the variance collapses to 0 at training points.
4. **Bit-identical determinism.** Two identical runs produce identical metrics.

---

## Quick start

```bash
pip install -e .

# List kernels / methods
python -m gpforge.cli methods

# Run the benchmark + pre-registered quality gate
python -m gpforge.cli run --verbose

# Full demo (benchmark, gate, determinism, ablation, Bayesian optimization)
PYTHONPATH=. python -m examples.run_demo
```

Library use:

```python
import numpy as np
from gpforge.models.gp import GaussianProcess
from gpforge.gp.methods import BayesOpt

# --- regression with uncertainty ---
X = np.linspace(0, 1, 60)[:, None]
y = np.sin(6 * X[:, 0])[:, None].ravel() + 0.05 * np.random.default_rng(0).standard_normal(60)
gp = GaussianProcess(kernel="rbf", optimize=True).fit(X, y)
post = gp.predict(np.linspace(0, 1, 200)[:, None])
# post.mean, post.std  -> calibrated 95% band = mean ± 1.96 * std

# --- Bayesian optimization ---
def f(x):
    return float(np.sin(3 * x[0]) + 0.3 * x[0])

bo = BayesOpt(f, bounds=np.array([[0.0, 1.0]]), seed=0, bo_iters=25, acquisition="ei")
out = bo.run()
print(out["best_y"], out["n_evals"])
```

---

## The pre-registered quality gate

Declared **before** measurement, and enforced per dataset over ≥ 3 seeds:

| Condition | Threshold |
|---|---|
| Calibration | realized 95%-interval coverage ≥ **0.90** |
| Accuracy | GP RMSE ≤ **1.10 ×** ridge-baseline RMSE |

Both must hold, for every dataset. Measured (3 seeds, `n_train=60`, `n_test=600`):

| dataset | coverage | GP RMSE | baseline RMSE | verdict |
|---|---|---|---|---|
| `rbf_1d` | 0.954 | 0.1090 | 0.1992 | PASS |
| `matern52_1d` | 0.949 | 0.1152 | 0.2839 | PASS |
| `periodic_1d` | 0.939 | 0.6162 | 0.8384 | PASS |
| `rbf_2d` | 0.955 | 0.1138 | 0.4416 | PASS |

Determinism: **bit-identical** across repeated runs.

---

## Ablation: hyperparameter optimization matters

Measured RMSE with vs. without L-BFGS-B maximization of the log-marginal-likelihood:

| dataset | optimize on | optimize off |
|---|---|---|
| `rbf_1d` | 0.1090 | 0.1174 |
| `matern52_1d` | 0.1152 | 0.1441 |
| `periodic_1d` | 0.6162 | 0.8277 |
| `rbf_2d` | 0.1138 | 0.1675 |

---

## Engineering

- **Deterministic by construction.** `core.seed.set_all(seed)` is the single
  acyclic entry point pinning numpy, `random`, and `PYTHONHASHSEED`.
- **Graceful degradation.** SciPy is the Tier-0 optimizer; if absent the toolkit
  falls back to fixed hyperparameters and still runs on pure NumPy.
- **Layered and typed.** `core / data / models / gp / eval / pipeline / cli`,
  dataclass payloads, typed error hierarchy (E100–E500), `Protocol` interfaces.
- **Tested.** 29 pytest cases covering kernel PSD, posterior-variance sign,
  interpolation, manual-Cholesky agreement, calibration, BO invariants,
  determinism, the gate, and CLI smoke.
- **CI.** `ruff check` + `ruff format --check` + pytest + demo smoke on Python
  3.10 and 3.13, plus a secret scan.

```
gpforge/
├── gpforge/
│   ├── core/        seed, config, types, errors, interfaces
│   ├── data/        synthetic GP datasets (presets + kernel lookup)
│   ├── models/      kernels + exact GP (Cholesky) + baselines
│   ├── gp/          methods: GpMethod, BayesOpt
│   ├── eval/        rmse / nlpd / interval coverage
│   ├── pipeline/    run, benchmark, pre-registered gate
│   └── cli.py
├── examples/run_demo.py
├── tests/           (29 pytest cases)
├── docs/            architecture.md, model_card.md
└── .github/workflows/ci.yml
```

---

## Limitations

- Exact (dense) GP: `O(n³)` Cholesky, `O(n²)` memory. Suited to n ≲ few thousand;
  sparse/inducing-point approximations are **not** included.
- Single output (no multi-task / vector-valued GPs).
- The periodic kernel's period is a fixed knob (not optimized).
- Classification / likelihoods other than Gaussian are out of scope.

---

## License

MIT © 2026 晨星
