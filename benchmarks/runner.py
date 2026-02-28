"""Centralized execution and environment utilities for benchmarks."""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Tuple

import torch


def ensure_repo_root_on_path() -> Path:
    """Ensure repository root is importable when running scripts directly."""
    # We assume this file is in benchmarks/runner.py
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def get_run_directory(run_name: str, repo_root: str | Path) -> Tuple[str, str]:
    """
    Generate a standard timestamped run directory.

    Returns:
        (relative_path, absolute_path)
    """
    today = datetime.datetime.now().strftime("%Y_%m_%d")
    now = datetime.datetime.now().strftime("%H%M%S")
    run_name = run_name.strip() or "benchmark"

    rel_path = os.path.join("benchmark_results", "runs", today, f"{now}_{run_name}")
    abs_path = os.path.join(repo_root, rel_path)
    return rel_path, abs_path


def run_and_capture(cmd: list[str], cwd: str | Path) -> str:
    """Run a command and capture its stdout and stderr, mirroring to the console."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(cwd),
    )
    assert result.stdout is not None
    assert result.stderr is not None

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def _drain(stream: Any, sink: list[str], writer: Any) -> None:
        for line in iter(stream.readline, ""):
            sink.append(line)
            writer.write(line)
            writer.flush()
        stream.close()

    t_out = threading.Thread(
        target=_drain,
        args=(result.stdout, stdout_parts, sys.stdout),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_drain,
        args=(result.stderr, stderr_parts, sys.stderr),
        daemon=True,
    )
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()

    returncode = result.wait()
    stdout_text = "".join(stdout_parts)
    stderr_text = "".join(stderr_parts)

    if returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {returncode}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{stdout_text}\n"
            f"STDERR:\n{stderr_text}"
        )
    return stdout_text


def setup_torch_device(device_arg: str) -> torch.device:
    """Standardize torch device selection."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is unavailable")
    return device


def get_torch_dtype(dtype_arg: str) -> torch.dtype:
    """Standardize torch dtype selection."""
    mapping = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    if dtype_arg not in mapping:
        raise ValueError(
            f"Unknown dtype: {dtype_arg}. Supported: {list(mapping.keys())}"
        )
    return mapping[dtype_arg]
