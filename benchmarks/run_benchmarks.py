#!/usr/bin/env python3
"""
benchmarks/run_benchmarks.py

Runs the maintained solver benchmark matrix and writes fresh .txt logs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "benchmark_results")
SPD_DIR = os.path.join(RESULTS_DIR, "latest_spd_solve_logs")
NONSPD_DIR = os.path.join(RESULTS_DIR, "latest_nonspd_solve_logs")


def _ensure_dirs() -> None:
    os.makedirs(SPD_DIR, exist_ok=True)
    os.makedirs(NONSPD_DIR, exist_ok=True)


def run_command(cmd: list[str], out_path: str) -> None:
    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {out_path}")
    with open(out_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=REPO_ROOT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        process.wait()
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            sys.exit(process.returncode)


def _run_spd(trials: int, dtype: str, timing_reps: int, warmup_reps: int) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Suite A: k < n, with n in {1024, 2048}, k in {1,16,64}
    for p_val in (1, 2, 4):
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "benchmarks.solve.matrix_solve",
            "--p",
            str(p_val),
            "--sizes",
            "1024,2048",
            "--k",
            "1,16,64",
            "--trials",
            str(trials),
            "--timing-reps",
            str(timing_reps),
            "--timing-warmup-reps",
            str(warmup_reps),
            "--dtype",
            dtype,
        ]
        out = os.path.join(
            SPD_DIR,
            f"spd_p{p_val}_klt_n_sizes1024_2048_k1_16_64_{ts}.txt",
        )
        run_command(cmd, out)

    # Suite B: k = n, with n in {256,512,1024,2048}
    for p_val in (1, 2, 4):
        for n_val in (256, 512, 1024, 2048):
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "benchmarks.solve.matrix_solve",
                "--p",
                str(p_val),
                "--sizes",
                str(n_val),
                "--k",
                str(n_val),
                "--trials",
                str(trials),
                "--timing-reps",
                str(timing_reps),
                "--timing-warmup-reps",
                str(warmup_reps),
                "--dtype",
                dtype,
            ]
            out = os.path.join(
                SPD_DIR,
                f"spd_p{p_val}_keq_n_n{n_val}_k{n_val}_{ts}.txt",
            )
            run_command(cmd, out)


def _run_nonspd(trials: int, dtype: str, timing_reps: int, warmup_reps: int) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Suite A: k < n, with n in {1024, 2048}, k in {1,16,64}
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "benchmarks.solve.matrix_solve_nonspd",
        "--p",
        "1",
        "--sizes",
        "1024,2048",
        "--k",
        "1,16,64",
        "--trials",
        str(trials),
        "--timing-reps",
        str(timing_reps),
        "--timing-warmup-reps",
        str(warmup_reps),
        "--dtype",
        dtype,
    ]
    out = os.path.join(
        NONSPD_DIR,
        f"nonspd_p1_klt_n_sizes1024_2048_k1_16_64_{ts}.txt",
    )
    run_command(cmd, out)

    # Suite B: k = n, with n in {256,512,1024,2048}
    for n_val in (256, 512, 1024, 2048):
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "benchmarks.solve.matrix_solve_nonspd",
            "--p",
            "1",
            "--sizes",
            str(n_val),
            "--k",
            str(n_val),
            "--trials",
            str(trials),
            "--timing-reps",
            str(timing_reps),
            "--timing-warmup-reps",
            str(warmup_reps),
            "--dtype",
            dtype,
        ]
        out = os.path.join(
            NONSPD_DIR,
            f"nonspd_p1_keq_n_n{n_val}_k{n_val}_{ts}.txt",
        )
        run_command(cmd, out)


def main() -> None:
    # Keep defaults aligned with current lightweight reproducibility settings.
    trials = 5
    dtype = "bf16"
    timing_reps = 5
    warmup_reps = 2

    _ensure_dirs()
    _run_spd(trials=trials, dtype=dtype, timing_reps=timing_reps, warmup_reps=warmup_reps)
    _run_nonspd(trials=trials, dtype=dtype, timing_reps=timing_reps, warmup_reps=warmup_reps)


if __name__ == "__main__":
    main()
