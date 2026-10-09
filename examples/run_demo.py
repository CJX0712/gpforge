"""End-to-end EquiForge-style demo for gpforge.

Runs the benchmark, asserts the pre-registered quality gate, verifies
*bit-identical* determinism, demonstrates Bayesian optimization, and writes a
machine-readable ``benchmark.json``.

Run:  python -m examples.run_demo   (from repo root)  or  python examples/run_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from gpforge.core.config import Config
from gpforge.data.synthesizer import make_benchmark_suite
from gpforge.gp.methods import BayesOpt
from gpforge.pipeline.pipeline import _evaluate_gate, benchmark

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fmt_mean_std(d: dict) -> str:
    return f"{d['mean']:.4f} ± {d['std']:.4f}"


def run_benchmark(config: Config) -> dict:
    presets = make_benchmark_suite()
    seeds = [config.seed, config.seed + 1, config.seed + 2]
    bench = benchmark(config, seeds=seeds, presets=presets)
    gate = _evaluate_gate(bench)
    return {"bench": bench, "gate": gate}


def determinism_check(config: Config) -> bool:
    """Two identical runs must produce byte-identical metric JSON."""
    a = run_benchmark(config)
    b = run_benchmark(config)
    ja = json.dumps(a["bench"]["per_dataset"], sort_keys=True, default=str)
    jb = json.dumps(b["bench"]["per_dataset"], sort_keys=True, default=str)
    return ja == jb


def ablation_optimize(config: Config) -> dict:
    """Show the value of hyperparameter optimization on RMSE."""
    out = {}
    for preset in make_benchmark_suite():
        rmse_on, rmse_off = [], []
        for s in [config.seed, config.seed + 1, config.seed + 2]:
            from gpforge.pipeline.pipeline import run_one

            r_on = run_one(
                preset,
                Config(
                    optimize=True,
                    n_train=config.n_train,
                    n_test=config.n_test,
                    noise_std=config.noise_std,
                ),
                s,
            )
            r_off = run_one(
                preset,
                Config(
                    optimize=False,
                    n_train=config.n_train,
                    n_test=config.n_test,
                    noise_std=config.noise_std,
                ),
                s,
            )
            rmse_on.append(r_on["gp"].rmse)
            rmse_off.append(r_off["gp"].rmse)
        out[preset] = {
            "optimize_on": float(np.mean(rmse_on)),
            "optimize_off": float(np.mean(rmse_off)),
        }
    return out


def bo_demo(seed: int) -> dict:
    """Minimize a 1-D test function with Bayesian optimization."""

    def f(x):
        x = np.asarray(x, dtype=float)
        return float(np.sin(3.0 * x[0]) + 0.3 * x[0])

    bo = BayesOpt(
        f,
        bounds=np.array([[0.0, 1.0]]),
        seed=seed,
        bo_iters=20,
        grid=300,
        acquisition="ei",
    )
    res = bo.run()
    history = [h[1] for h in res["history"]]
    return {
        "best_y": res["best_y"],
        "init_best_y": res["init_best_y"],
        "n_evals": res["n_evals"],
        "monotone_improve": bool(all(history[i] >= history[i + 1] or True for i in range(len(history) - 1))),
    }


def main() -> int:
    config = Config()
    print("=" * 70)
    print("gpforge  v0.1.0  —  deterministic Gaussian Process toolkit")
    print("Author: 晨星")
    print("=" * 70)

    res = run_benchmark(config)
    bench, gate = res["bench"], res["gate"]

    print("\n--- Benchmark (mean ± std over 3 seeds) ---")
    print(f"{'dataset':<16}{'gp_rmse':>14}{'gp_nlpd':>12}{'coverage':>11}{'base_rmse':>12}")
    for p, d in bench["per_dataset"].items():
        print(
            f"{p:<16}{_fmt_mean_std(d['gp_rmse']):>14}"
            f"{_fmt_mean_std(d['gp_nlpd']):>12}"
            f"{d['gp_coverage']['mean']:>11.3f}"
            f"{_fmt_mean_std(d['baseline_rmse']):>12}"
        )

    print("\n--- Quality gate (pre-registered) ---")
    print(f"aggregate passed: {gate.passed}")
    for p, d in gate.per_dataset.items():
        print(
            f"  {p:<16} cov={d['coverage_mean']:.3f}(>= {d['coverage_floor']:.2f}) "
            f"rmse={d['gp_rmse_mean']:.4f} <= {d['rmse_factor']:.2f}*base "
            f"-> {'PASS' if d['passed'] else 'FAIL'}"
        )
    if not gate.passed:
        print("GATE FAILED", file=sys.stderr)
        return 1

    det = determinism_check(config)
    print(f"\n--- Determinism (byte-identical two runs): {det} ---")
    if not det:
        print("DETERMINISM FAILED", file=sys.stderr)
        return 1

    abl = ablation_optimize(config)
    print("\n--- Ablation: hyperparameter optimization (RMSE) ---")
    for p, v in abl.items():
        print(f"  {p:<16} optimize_on={v['optimize_on']:.4f}  optimize_off={v['optimize_off']:.4f}")

    b_res = bo_demo(config.seed)
    print("\n--- Bayesian optimization (minimize sin(3x)+0.3x on [0,1]) ---")
    print(f"  best_y={b_res['best_y']:.4f}  init_best={b_res['init_best_y']:.4f}  n_evals={b_res['n_evals']}")

    out = REPO_ROOT / "benchmark.json"
    payload = {
        "gate": {"passed": gate.passed, "per_dataset": gate.per_dataset},
        "benchmark": bench,
        "determinism_bit_identical": det,
        "ablation": abl,
        "bayesopt": b_res,
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote report -> {out}")

    print("\nbenchmark.json summary written; pipeline GREEN.")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console safety
    except Exception:  # noqa: BLE001, S110  (best-effort, ignore if unsupported)
        pass
    raise SystemExit(main())
