# Model Card — gpforge `GaussianProcess` (v0.1.0)

**Author:** 晨星 (Morning Star) · **License:** MIT · **Released:** 2026-10-09

---

## Model details

| Field | Value |
|---|---|
| Model | Exact (dense) Gaussian Process regression |
| Inference | Cholesky factorization, exact posterior — no approximation |
| Kernels | RBF, Matérn-5/2, Matérn-3/2, Periodic, Linear (all ARD-capable) |
| Hyperparameters | lengthscales, signal variance, noise variance |
| Learning | Maximization of log-marginal-likelihood (L-BFGS-B, numerical gradients) |
| Output | Predictive mean **and** calibrated predictive variance |
| Dependencies | NumPy (required), SciPy (optional optimizer) |
| Compute | CPU-only; `O(n³)` time, `O(n²)` memory |

---

## Intended use

**In scope**
- Small-to-moderate regression with meaningful uncertainty quantification
  (n up to a few thousand).
- Surrogate modeling for Bayesian optimization of expensive black-box functions
  (tens to hundreds of evaluations).
- Any setting where a *calibrated* predictive interval matters more than raw
  point accuracy.

**Out of scope**
- Large-n regression without sparsification.
- Multi-output / vector-valued problems.
- Non-Gaussian likelihoods (classification, counts).
- High-dimensional inputs (d ≳ 20) without careful kernel choice.

---

## Training data

The benchmark uses **synthetic** data whose generative process is itself a GP
with a known kernel (`rbf_1d`, `matern52_1d`, `periodic_1d`, `rbf_2d`), so the
ground truth is known and calibration can be *checked* rather than assumed.
No real-world or personal data is used.

---

## Quantitative results

Pre-registered gate, 3 seeds, `n_train = 60`, `n_test = 600`, noise σ = 0.10:

| dataset | coverage (target ≥ 0.90 @ 95% nominal) | GP RMSE | ridge-baseline RMSE | pass |
|---|---|---|---|---|
| `rbf_1d` | 0.954 | 0.1090 | 0.1992 | yes |
| `matern52_1d` | 0.949 | 0.1152 | 0.2839 | yes |
| `periodic_1d` | 0.939 | 0.6162 | 0.8384 | yes |
| `rbf_2d` | 0.955 | 0.1138 | 0.4416 | yes |

Other properties:
- **Determinism**: bit-identical metrics across repeated runs.
- **BO**: finds the global minimum of `sin(3x) + 0.3x` on `[0, 1]` (value 0 at
  `x = 0`) within 25 evaluations.
- **Ablation**: disabling hyperparameter optimization raises RMSE by 7–47%,
  confirming the marginal-likelihood objective earns its keep.

---

## Ethical considerations

This is a general-purpose regression tool; it inherits no task-specific bias
from this repository. Two generic cautions:

1. **Uncertainty is only as good as the kernel.** A mis-specified kernel produces
   confidently wrong predictions. Always check calibration, not just RMSE.
2. **Downstream decisions.** Predictive intervals are a statement about the model
   under its own assumptions; they are not a safety guarantee in high-stakes
   use (medical, financial, legal) without domain validation.

---

## Caveats and recommendations

- Verify coverage on held-out data before trusting an interval.
- For n > ~2000, use a sparse approximation (not provided here).
- Periodic structure requires either the periodic kernel or a correct period knob;
  RBF will silently underfit it.
