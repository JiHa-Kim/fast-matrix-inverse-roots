# TODO

Date: 2026-02-26
Scope: prioritize unimplemented ideas from `ideas/p.md`, `ideas/p1.md`, and `ideas/1.md`, with emphasis on `p=2,4`.

## P0: Highest Priority (p=2,4)

- [ ] Implement full interval-minimax coefficient lookup by `(p, degree, kappa-bin)` for the contraction objective `max | t*q(t)^p - 1 |`.
  - Gap today: local minimax-alpha candidate exists, but not full degree-`d` runtime lookup from precomputed tables.
  - Exit check: online schedule uses nonzero minimax steps in real runs and improves total ms or relerr at fixed ms.

## P1: Next Wave

- [ ] Add stronger Turbo-style normalization for SPD: robust `lambda_min` estimation + scalar scaling target beyond current row-sum/Gershgorin proxies.
  - Gap today: `lambda_max` estimation exists; `lambda_min`-aware optimal scalar initialization is missing.

- [ ] Add block-diagonal SPD preconditioner mode in `precond_spd`.
  - Gap today: only none/frob/aol/jacobi/ruiz (diagonal/global scaling family) are available.

- [ ] Implement block-Lanczos direct apply for `A^{-1/p}B` (`p=2,4`) as an alternative to Chebyshev.
  - Gap today: no Lanczos/Krylov inverse-root apply kernel is present.

- [ ] Add a mixed-precision factorization + iterative-refinement path for SPD `p>1` when `k ~= n`.
  - Gap today: direct polynomial methods are available, but there is no explicit high-accuracy mixed-precision fallback path for fat-RHS regimes.
