"""CLI smoke tests (subprocess isolation)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "gpforge.cli", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_methods():
    proc = _run_cli(["methods"])
    assert proc.returncode == 0
    assert "kernels" in proc.stdout
    assert "GpMethod" in proc.stdout


def test_cli_run_writes_report(tmp_path):
    out_file = tmp_path / "bench.json"
    proc = _run_cli(
        [
            "run",
            "--presets",
            "rbf_1d",
            "rbf_2d",
            "--seeds",
            "0",
            "1",
            "2",
            "--n-train",
            "60",
            "--n-test",
            "400",
            "--output",
            str(out_file),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "gate" in data
    assert data["gate"]["passed"] is True
