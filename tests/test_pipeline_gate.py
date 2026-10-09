"""Pre-registered quality gate must pass on the default benchmark."""

from __future__ import annotations

from gpforge.core.config import Config
from gpforge.data.synthesizer import make_benchmark_suite
from gpforge.pipeline.pipeline import _evaluate_gate, benchmark


def test_gate_passes_default():
    cfg = Config()
    bench = benchmark(cfg, seeds=[0, 1, 2], presets=make_benchmark_suite())
    gate = _evaluate_gate(bench)
    assert gate.passed is True


def test_gate_per_dataset_fields():
    cfg = Config()
    bench = benchmark(cfg, seeds=[0, 1, 2], presets=make_benchmark_suite())
    gate = _evaluate_gate(bench)
    for d in gate.per_dataset.values():
        assert "coverage_mean" in d
        assert "gp_rmse_mean" in d
        assert "baseline_rmse_mean" in d
        assert d["coverage_mean"] >= d["coverage_floor"] - 1e-9
