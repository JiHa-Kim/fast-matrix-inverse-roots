# Implementation Status: Ideas vs Code

Date: 2026-02-26
Primary reference ideas: `ideas/p.md`, `ideas/p1.md`.

This page distinguishes what is already implemented from what is still missing to try.

## Implemented

- Coupled and uncoupled quadratic PE inverse-root kernels (`fast_iroot/coupled.py`, `fast_iroot/uncoupled.py`).
- Apply variants for `Z = A^{-1/p}B`, including SPD/non-SPD controls and workspace reuse (`fast_iroot/coupled.py`, `fast_iroot/apply.py`).
- SPD-specialized fast path for `p=2` in coupled updates (`inverse_sqrt_pe_quadratic`).
- Online scheduling hooks for coupled PE (greedy-newton, greedy-minimax local alpha, greedy-affine-opt) and interval-error schedule trimming (`fast_iroot/coeff_tuner.py`, `benchmarks/solve/matrix_solve.py`).
- Chebyshev direct-apply with Clenshaw recurrence (`apply_inverse_proot_chebyshev`) and minimax-auto degree selection (`fast_iroot/chebyshev.py`).
- SPD preconditioning/scaling modes: `none`, `frob`, `aol`, `jacobi`, `ruiz`, plus ridge and Gershgorin-based floor targeting (`fast_iroot/precond.py`).
- Non-SPD safety mechanisms for `p=1` solve paths (adaptive inverse-Newton fallback and optional exact solve fallback) (`fast_iroot/coupled.py`).
- `p=1` hybrid PE+NSRC solve path for small-`k/n` settings (`fast_iroot/nsrc.py`, `fast_iroot/apply.py`).

## Partially Implemented (core direction present, full idea missing)

- Minimax adaptation exists only in restricted families (local quadratic alpha / affine slope), not full degree-`d` interval minimax contraction tables indexed by condition number.
- Turbo-like normalization exists via scaling and `lambda_max` estimation, but no robust `lambda_min`-aware optimal scalar initialization from the inverse-root literature.
- Direct-apply Chebyshev exists, but is not part of the maintained default `p=2,4` benchmark method matrix.

## Not Implemented Yet (high-value candidates)

- `p=2,4` staged policy: interval-focused minimax early, residual-binomial late.
- Full runtime coefficient lookup by `(p, degree, kappa-bin)` for contraction objective `max | t*q(t)^p - 1 |`.
- `p=4` two-stage reduction via tuned `p=2` kernels.
- Precision-oriented residual correction pass for `p=2,4` apply kernels.
- Block-diagonal SPD preconditioning mode.
- Block-Lanczos/Krylov direct apply for `A^{-1/p}B` (`p=2,4`).
- Default benchmark exposure of `Torch-EVD-Solve` and `Chebyshev-Apply` for `p>1`.

## Why This Matters for Current Priorities

- `p=1` already has strong `torch.linalg.solve` baselines and dedicated fallback/hybrid handling.
- Remaining headroom is mainly in `p=2,4` speed-vs-relerr tradeoffs, especially for SPD solve/apply workloads where polynomial/direct-apply paths should improve throughput.
