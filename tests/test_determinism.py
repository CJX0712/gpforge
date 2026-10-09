"""Bit-identical determinism: identical inputs -> identical metrics."""

from __future__ import annotations

import json

from gpforge.core.config import Config
from gpforge.data.synthesizer import make_benchmark_suite
from gpforge.pipeline.pipeline import _evaluate_gate, benchmark


def _run():
    cfg = Config(seed=0)
    bench = benchmark(cfg, seeds=[0, 1, 2], presets=make_benchmark_suite())
    return json.dumps(bench["per_dataset"], sort_keys=True, default=str)


def test_determinism_bit_identical():
    assert _run() == _run()


def test_gate_deterministic():
    cfg = Config(seed=0)
    g1 = _evaluate_gate(benchmark(cfg, seeds=[0, 1, 2], presets=make_benchmark_suite()))
    g2 = _evaluate_gate(benchmark(cfg, seeds=[0, 1, 2], presets=make_benchmark_suite()))
    assert g1.passed == g2.passed
