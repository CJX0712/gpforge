"""Command-line entry point for gpforge.

Subcommands
-----------
run      Benchmark the flagship GP against the ridge baseline and apply the
         pre-registered quality gate. Writes a JSON report.
methods  Print the available kernels / methods.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gpforge.core.config import Config
from gpforge.data.synthesizer import make_benchmark_suite
from gpforge.pipeline.pipeline import _evaluate_gate, benchmark


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpforge",
        description="Deterministic Gaussian Process toolkit (author: 晨星).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="benchmark + quality gate")
    r.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help="presets to run (default: full suite)",
    )
    r.add_argument("--seeds", nargs="*", type=int, default=None, help="seeds (default: 0 1 2)")
    r.add_argument("--n-train", type=int, default=None)
    r.add_argument("--n-test", type=int, default=None)
    r.add_argument("--noise-std", type=float, default=None)
    r.add_argument("--no-optimize", action="store_true", help="disable hyperparameter optimization")
    r.add_argument("--output", type=str, default=None, help="write JSON report to this path")
    r.add_argument("--verbose", action="store_true")

    sub.add_parser("methods", help="list kernels / methods")
    return p


def _print_gate(gate, bench) -> None:
    print("\n=== QUALITY GATE ===")
    print(f"aggregate passed: {gate.passed}")
    hdr = f"{'dataset':<16}{'cov(mean)':>11}{'floor':>8}{'gpRMSE':>10}{'baseRMSE':>11}{'pass':>7}"
    print(hdr)
    for p, d in gate.per_dataset.items():
        print(
            f"{p:<16}{d['coverage_mean']:>11.3f}{d['coverage_floor']:>8.2f}"
            f"{d['gp_rmse_mean']:>10.4f}{d['baseline_rmse_mean']:>11.4f}"
            f"{'YES' if d['passed'] else 'NO':>7}"
        )


def _cmd_run(args) -> int:
    cfg = Config()
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.n_test is not None:
        cfg.n_test = args.n_test
    if args.noise_std is not None:
        cfg.noise_std = args.noise_std
    if args.no_optimize:
        cfg.optimize = False

    presets = args.presets if args.presets else make_benchmark_suite()
    seeds = args.seeds if args.seeds else [cfg.seed, cfg.seed + 1, cfg.seed + 2]

    bench = benchmark(cfg, seeds=seeds, presets=presets)
    gate = _evaluate_gate(bench)

    if args.verbose:
        for p, d in bench["per_dataset"].items():
            print(f"\n[{p}]")
            print(f"  gp_rmse      = {d['gp_rmse']['mean']:.4f} ± {d['gp_rmse']['std']:.4f}")
            print(f"  gp_nlpd      = {d['gp_nlpd']['mean']:.4f} ± {d['gp_nlpd']['std']:.4f}")
            print(f"  gp_coverage  = {d['gp_coverage']['mean']:.3f} (nominal {d['nominal']:.2f})")
            print(f"  base_rmse    = {d['baseline_rmse']['mean']:.4f} ± {d['baseline_rmse']['std']:.4f}")

    _print_gate(gate, bench)

    report = {
        "gate": {"passed": gate.passed, "per_dataset": gate.per_dataset},
        "benchmark": bench,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote report -> {args.output}")
    return 0 if gate.passed else 1


def _cmd_methods(_args) -> int:
    print("kernels : rbf, matern52, matern32, periodic, linear")
    print("methods : GpMethod (exact GP regression), BayesOpt (EI/UCB acquisition)")
    print("baselines: LinearRidge, ConstantMean")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "methods":
        return _cmd_methods(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
