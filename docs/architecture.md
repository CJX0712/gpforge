# gpforge Architecture

## Design goals

1. **Correctness over convenience.** Every module exposes an invariant that a
   test asserts directly (PSD kernels, non-negative posterior variance,
   calibrated intervals, interpolation, manual-Cholesky agreement).
2. **Determinism as a contract.** One seeding entry point; no hidden RNG use.
3. **Layering with no cycles.** `core` is depended on by everyone and depends on
   nothing; `models` knows about kernels but not about the pipeline; the
   pipeline composes; `cli` / `examples` are the only entry points.

## Layer map

```
                 ┌───────────────────────────────┐
 entry points    │ cli.py        examples/       │
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────▼───────────────┐
 orchestration   │ pipeline/  run_one · benchmark│
                 │            · gate             │
                 └──┬──────────┬──────────┬──────┘
                    │          │          │
        ┌───────────▼──┐  ┌────▼─────┐  ┌─▼────────────┐
 domain │ gp/methods   │  │ eval/    │  │ data/        │
        │ GpMethod     │  │ metrics  │  │ synthesizer  │
        │ BayesOpt     │  └──────────┘  └──────────────┘
        └───────┬──────┘
                │
        ┌───────▼────────────────────────────────┐
 engine │ models/  kernels.py · gp.py            │
        │ (Cholesky inference, LML hyperopt)     │
        └───────┬────────────────────────────────┘
                │
        ┌───────▼────────────────────────────────┐
  base  │ core/  seed · config · types · errors   │
        │        · interfaces                     │
        └─────────────────────────────────────────┘
```

## Core mathematics

### Posterior inference

Given training data `(X, y)` with `y = f(X) + ε`, `f ~ GP(0, k)`, `ε ~ N(0, σ²)`:

```
K      = k(X, X) + (σ² + jitter) I
L      = chol(K)                            # O(n³)
alpha  = Lᵀ \ (L \ y)
μ(x*)  = k(X, x*)ᵀ alpha
v      = L \ k(X, x*)
var(x*)= k(x*, x*) − Σ vᵢ²   (+ σ²  when predicting a noisy target)
```

`var ≥ 0` is guaranteed because it is the Schur complement of the positive-
definite block `[K, k; kᵀ, k**]`.

### Hyperparameter learning

Parameters `θ = (log ℓ₁..log ℓ_d, log σ_f², log σ_n²)` are chosen by maximizing

```
log p(y | X, θ) = −½ yᵀ K⁻¹ y − Σ log diag(L) − (n/2) log 2π
```

with L-BFGS-B using central-difference gradients in log space (stable, and
deterministic). Bounds keep hyperparameters in a sane range.

### Bayesian optimization

1. Fit the GP to observed `(X, y)`.
2. Score a deterministic candidate grid with an acquisition:
   - **EI** (minimization): `(f_best − μ) Φ(u) + σ φ(u)`, `u = (f_best − μ)/σ`
   - **UCB** (minimization): maximize `−μ + κσ`
3. Evaluate the argmax, append to observations, repeat.

The latent posterior (`full=False`) is used for acquisition because the objective
is noise-free; the noisy posterior (`full=True`) is used for regression intervals.

## Determinism strategy

- `core.seed.set_all(seed)` seeds numpy, `random`, and `PYTHONHASHSEED`. It is the
  only sanctioned seeding call, and no module re-seeds after it.
- The BO candidate grid is `linspace`-based (1D/2D) or derived from a seeded
  generator — never from ambient global state.
- Acquisition `argmax` uses `np.argmax` on a deterministic ordering.

## Extension points

- **New kernel**: add a branch to `models/kernels.py::kernel_matrix`; the PSD
  invariant test applies automatically.
- **New acquisition**: add a function in `gp/methods.py` and a branch in
  `BayesOpt.run`.
- **New baseline**: subclass the `Regressor` protocol; add it to the pipeline.
- **New gate**: extend `_evaluate_gate` with more conditions (the gate is a pure
  function of the benchmark result).
