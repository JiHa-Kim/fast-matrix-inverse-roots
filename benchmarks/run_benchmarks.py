#!/usr/bin/env python3
"""
benchmarks/run_benchmarks.py

Runs the maintained solver benchmark matrix and generates Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from benchmarks.utils import (
    get_git_metadata,
    write_text_file,
    write_json_file,
    write_sha256_sidecar,
    format_timestamp,
    repo_relative,
)
from benchmarks.solver_utils import (
    ParsedRow,
    parse_rows,
    row_from_dict,
)
from benchmarks.solver_reporting import to_markdown, to_markdown_ab

# Bootstrap and Common Utils
try:
    from .runner import ensure_repo_root_on_path, run_and_capture, get_run_directory
except ImportError:
    from runner import ensure_repo_root_on_path, run_and_capture, get_run_directory

REPO_ROOT = ensure_repo_root_on_path()


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str  # "spd" | "nonspd"
    cmd: list[str]
    txt_out: str


def _build_specs(
    trials: int,
    dtype: str,
    timing_reps: int,
    warmup_reps: int,
    *,
    spd_dir: str,
    nonspd_dir: str,
    ts: str,
) -> list[RunSpec]:
    specs: list[RunSpec] = []

    # SPD, p in {1,2,4}, k<n (n={1024}, k={1,16,64})
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
            "1024",
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
        specs.append(
            RunSpec(
                name=f"SPD p={p_val} k<n",
                kind="spd",
                cmd=cmd,
                txt_out=os.path.join(spd_dir, f"{ts}_spd_p{p_val}_klt_n.txt"),
            )
        )

    # SPD, p in {1,2,4}, k=n for n in {256,512,1024}
    for p_val in (1, 2, 4):
        for n_val in (256, 512, 1024):
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
            specs.append(
                RunSpec(
                    name=f"SPD p={p_val} k=n={n_val}",
                    kind="spd",
                    cmd=cmd,
                    txt_out=os.path.join(
                        spd_dir, f"{ts}_spd_p{p_val}_keq_n_{n_val}.txt"
                    ),
                )
            )

    # Non-SPD p=1
    for n_val in (256, 512, 1024):
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
        specs.append(
            RunSpec(
                name=f"non-SPD p=1 k=n={n_val}",
                kind="nonspd",
                cmd=cmd,
                txt_out=os.path.join(nonspd_dir, f"{ts}_nonspd_p1_keq_n_{n_val}.txt"),
            )
        )

    # Gram RHS (M = G^T B)
    for p_val in (2, 4):
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "benchmarks.solve.matrix_solve_gram_rhs",
            "--p",
            str(p_val),
            "--m",
            "256",
            "--n",
            "1024",
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
        specs.append(
            RunSpec(
                name=f"GRAM RHS p={p_val} m<n",
                kind="spd",
                cmd=cmd,
                txt_out=os.path.join(spd_dir, f"{ts}_gram_rhs_p{p_val}.txt"),
            )
        )

    return specs


def _parse_csv_tokens(spec: str) -> list[str]:
    return [tok.strip() for tok in str(spec).split(",") if tok.strip()]


def _filter_specs(specs: Iterable[RunSpec], only_tokens: list[str]) -> list[RunSpec]:
    toks = [t.lower() for t in only_tokens if t]
    if not toks:
        return list(specs)
    out: list[RunSpec] = []
    for spec in specs:
        hay = f"{spec.name} {spec.txt_out}".lower()
        if any(tok in hay for tok in toks):
            out.append(spec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run maintained solver benchmark suites"
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--timing-reps", type=int, default=10)
    parser.add_argument("--timing-warmup-reps", type=int, default=2)
    parser.add_argument(
        "--only", type=str, default="", help="Filter specs by substring."
    )
    parser.add_argument("--run-name", type=str, default="solver_benchmarks")
    parser.add_argument(
        "--extra-args", type=str, default="", help="Extra args for all commands."
    )
    parser.add_argument(
        "--markdown", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--out", type=str, default="", help="Output markdown path.")
    parser.add_argument(
        "--prod", action="store_true", help="Update production documentation."
    )
    parser.add_argument("--ab-extra-args-a", type=str, default="")
    parser.add_argument("--ab-extra-args-b", type=str, default="")
    parser.add_argument("--ab-label-a", type=str, default="A")
    parser.add_argument("--ab-label-b", type=str, default="B")
    parser.add_argument(
        "--ab-match-on-method", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ab-interleave", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ab-out", type=str, default="")
    parser.add_argument("--ab-baseline-rows-in", type=str, default="")
    parser.add_argument("--baseline-rows-out", type=str, default="")
    parser.add_argument("--manifest-out", type=str, default="")
    parser.add_argument(
        "--integrity-checksums", action=argparse.BooleanOptionalAction, default=True
    )

    args = parser.parse_args()

    rel_run_dir, abs_run_dir = get_run_directory(str(args.run_name), REPO_ROOT)
    spd_dir = os.path.join(abs_run_dir, "spd_solve_logs")
    nonspd_dir = os.path.join(abs_run_dir, "nonspd_solve_logs")
    time_prefix = datetime.now().strftime("%H%M%S")

    # Set default paths
    if args.prod:
        args.out = os.path.join(
            REPO_ROOT, "docs", "benchmarks", "benchmark_results_production.md"
        )
    if not args.out:
        args.out = os.path.join(abs_run_dir, "solver_benchmarks.md")
    if not args.ab_out:
        args.ab_out = os.path.join(abs_run_dir, "solver_benchmarks_ab.md")
    if not args.manifest_out:
        args.manifest_out = os.path.join(abs_run_dir, "run_manifest.json")

    specs_all = _build_specs(
        trials=args.trials,
        dtype=args.dtype,
        timing_reps=args.timing_reps,
        warmup_reps=args.timing_warmup_reps,
        spd_dir=spd_dir,
        nonspd_dir=nonspd_dir,
        ts=time_prefix,
    )
    specs = _filter_specs(specs_all, _parse_csv_tokens(args.only))

    base_extra_args = shlex.split(args.extra_args)
    ab_mode = bool(
        args.ab_extra_args_a or args.ab_extra_args_b or args.ab_baseline_rows_in
    )

    rows_a: list[ParsedRow] = []
    rows_b: list[ParsedRow] = []
    run_records = []

    # Simplified execution logic
    def run_spec_variant(spec, extra, variant_label):
        cmd = spec.cmd + extra
        raw = run_and_capture(cmd, REPO_ROOT)
        rows = parse_rows(raw, spec.kind)
        write_text_file(spec.txt_out + (f".{variant_label}" if ab_mode else ""), raw)
        return rows, raw, cmd

    if ab_mode:
        a_extra = base_extra_args + shlex.split(args.ab_extra_args_a)
        b_extra = base_extra_args + shlex.split(args.ab_extra_args_b)

        if args.ab_baseline_rows_in:
            with open(args.ab_baseline_rows_in, "r", encoding="utf-8") as f:
                rows_a = [row_from_dict(r) for r in json.load(f)["rows"]]

        for spec in specs:
            if not args.ab_baseline_rows_in:
                r, raw, cmd = run_spec_variant(spec, a_extra, "A")
                rows_a.extend(r)
                run_records.append(
                    {"spec": spec.name, "variant": "A", "cmd": cmd, "parsed": len(r)}
                )

            r, raw, cmd = run_spec_variant(spec, b_extra, "B")
            rows_b.extend(r)
            run_records.append(
                {"spec": spec.name, "variant": "B", "cmd": cmd, "parsed": len(r)}
            )

        report = to_markdown_ab(
            rows_a,
            rows_b,
            label_a=args.ab_label_a,
            label_b=args.ab_label_b,
            match_on_method=args.ab_match_on_method,
            config=vars(args),
        )
        write_text_file(args.ab_out, report)
    else:
        all_rows = []
        for spec in specs:
            r, raw, cmd = run_spec_variant(spec, base_extra_args, "")
            all_rows.extend(r)
            run_records.append({"spec": spec.name, "cmd": cmd, "parsed": len(r)})

        report = to_markdown(all_rows, config=vars(args))
        write_text_file(args.out, report)

    # Simplified manifest and checksum logic
    safe_args = dict(vars(args))
    for k in ["out", "ab_out", "manifest_out", "ab_baseline_rows_in", "baseline_rows_out"]:
        if safe_args.get(k):
            safe_args[k] = repo_relative(safe_args[k], str(REPO_ROOT))

    manifest = {
        "generated_at": format_timestamp(),
        "args": safe_args,
        "git": get_git_metadata(str(REPO_ROOT)),
        "runs": run_records,
    }
    write_json_file(args.manifest_out, manifest)
    if args.integrity_checksums:
        for p in [args.out, args.ab_out, args.manifest_out]:
            if os.path.exists(p):
                write_sha256_sidecar(p)

    print(f"Done. Report at: {repo_relative(args.out, str(REPO_ROOT))}")


if __name__ == "__main__":
    main()
