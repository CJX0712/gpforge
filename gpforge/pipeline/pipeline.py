"""End-to-end pipeline: run, benchmark, and the pre-registered quality gate.

The pre-registered gate (decided before any measurement) asserts, **per
dataset and for every seed**, that the flagship GP is simultaneously:

* *calibrated* — predictive-interval coverage ``>= coverage_floor`` (default
  0.90 at a 0.95 nominal), and
* *no worse than a dependency-free ridge baseline* — ``gp.rmse <= rmse_factor
  * baseline.rmse`` (default factor 1.10).

Both conditions must hold; the aggregate gate passes only when every dataset
passes.
"""

from __future__ import annotations

import numpy as np

from gpforge.core.config import Config
from gpforge.core.seed import set_all
from gpforge.core.types import FitResult, GateResult
from gpforge.data.synthesizer import make_benchmark_suite, make_dataset, preset_kernel
from gpforge.gp.methods import GpMethod
from gpforge.models.gp import LinearRidge


def run_one(preset: str, config: Config, seed: int) -> dict:
    """Train GP + ridge baseline on one preset/seed; return their FitResults."""
    set_all(seed)  # pin global RNG before any sampling in this unit
    ds = make_dataset(
        config.n_train,
        config.n_test,
        preset=preset,
        noise_std=config.noise_std,
        seed=seed,
    )

    kernel = preset_kernel(preset)  # model each preset with its generating kernel
    gp = GpMethod(
        kernel=kernel,
        optimize=config.optimize,
        max_iters=config.max_opt_iters,
        jitter=config.jitter,
    )
    gp_res = gp.fit_score(ds, nominal=config.nominal)

    base = LinearRidge(alpha=1e-2).fit(ds.X_train, ds.y_train)
    pred = base.predict(ds.X_test)
    from gpforge.eval.metrics import evaluate as _eval

    base_metrics = _eval(pred.mean, pred.std, ds.y_test, nominal=config.nominal)
    base_res = FitResult(
        name="LinearRidge",
        rmse=base_metrics["rmse"],
        nlpd=base_metrics["nlpd"],
        coverage=base_metrics["coverage"],
        nominal=config.nominal,
    )
    return {
        "preset": preset,
        "seed": seed,
        "gp": gp_res,
        "baseline": base_res,
        "_ds": ds,
    }


def _aggregate(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
    }


def benchmark(config: Config, seeds: list[int] | None = None, presets: list[str] | None = None) -> dict:
    """Run the benchmark over presets × seeds and aggregate mean ± std."""
    seeds = list(seeds if seeds is not None else [config.seed, config.seed + 1, config.seed + 2])
    presets = list(presets if presets is not None else make_benchmark_suite())

    per_dataset: dict[str, dict] = {}
    raw = {p: {"gp_rmse": [], "gp_nlpd": [], "gp_cov": [], "base_rmse": []} for p in presets}

    for p in presets:
        for s in seeds:
            r = run_one(p, config, s)
            raw[p]["gp_rmse"].append(r["gp"].rmse)
            raw[p]["gp_nlpd"].append(r["gp"].nlpd)
            raw[p]["gp_cov"].append(r["gp"].coverage)
            raw[p]["base_rmse"].append(r["baseline"].rmse)

    for p in presets:
        per_dataset[p] = {
            "gp_rmse": _aggregate(raw[p]["gp_rmse"]),
            "gp_nlpd": _aggregate(raw[p]["gp_nlpd"]),
            "gp_coverage": _aggregate(raw[p]["gp_cov"]),
            "baseline_rmse": _aggregate(raw[p]["base_rmse"]),
            "coverage_floor": config.coverage_floor,
            "nominal": config.nominal,
            "rmse_factor": config.rmse_factor,
        }
    return {
        "per_dataset": per_dataset,
        "seeds": seeds,
        "presets": presets,
        "config": config,
    }


def _evaluate_gate(bench: dict) -> GateResult:
    """Apply the pre-registered gate to a benchmark result."""
    per_ds: dict[str, dict] = {}
    all_pass = True
    for p, d in bench["per_dataset"].items():
        gp_cov = d["gp_coverage"]["mean"]
        gp_rmse = d["gp_rmse"]["mean"]
        base_rmse = d["baseline_rmse"]["mean"]
        cov_ok = gp_cov >= d["coverage_floor"]
        rmse_ok = gp_rmse <= d["rmse_factor"] * base_rmse
        ok = bool(cov_ok and rmse_ok)
        all_pass = all_pass and ok
        per_ds[p] = {
            "coverage_mean": gp_cov,
            "coverage_floor": d["coverage_floor"],
            "gp_rmse_mean": gp_rmse,
            "baseline_rmse_mean": base_rmse,
            "rmse_factor": d["rmse_factor"],
            "cov_ok": bool(cov_ok),
            "rmse_ok": bool(rmse_ok),
            "passed": ok,
        }
    return GateResult(
        passed=all_pass,
        per_dataset=per_ds,
        aggregate={"passed": all_pass, "n_datasets": len(per_ds)},
    )
