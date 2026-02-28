# Guidelines & Contributing

This document outlines the standards for development, testing, and contribution to the `fast-matrix-inverse-roots` project.

## Project Structure & Module Organization

- **`fast_iroot/`**: Core library code (inverse $p$-th-root kernels, solve/apply paths, preconditioning, diagnostics).
- **`benchmarks/`**: Benchmark CLIs and solve suites.
- **`tests/`**: `pytest` suite for kernels, preconditioners, and coefficient tuning.
- **`benchmark_results/`**: Raw benchmark logs and summaries.
- **`docs/`**: Documentation, method notes, and benchmark decisions.

## Development Commands

- **Environment**: `uv sync` to install dependencies.
- **Linting**: `uv run python -m ruff check .`
- **Testing**:
  - `uv run python -m pytest -q`: Run all tests.
  - `uv run python -m pytest tests/test_verify_iroot.py -q`: Correctness/stability validation sweep.
- **Benchmarking**:
  - `uv run python benchmarks/run_benchmarks.py`: Main benchmark driver.

## Coding Style

- **Language**: Python 3.10+; 4-space indentation.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for dataclasses.
- **Typing**: Use explicit type hints (`Optional`, `Tuple`, etc.) for all public APIs.
- **Consistency**: Ensure new code follows existing patterns in `fast_iroot/`.

## Testing Guidelines

- **Framework**: `pytest`.
- **Requirements**:
  - Add unit tests for all new features.
  - Add regression tests for any bug fixes.
  - For performance changes, include benchmark results in `benchmark_results/` and update `docs/benchmark_decisions.md`.

## Commit & PR Standards

- **Commits**: Follow [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat(solve): ...`, `perf(coupled): ...`).
- **PRs**: Should include a clear description of the change, exact commands used for validation, and links to any new benchmark reports.
