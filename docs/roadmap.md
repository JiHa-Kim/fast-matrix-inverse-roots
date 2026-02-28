# Roadmap

This document outlines the planned features and priority tasks for the `fast-matrix-inverse-roots` project.

## Priority 1: Performance & Robustness

- [ ] **Stronger Turbo-style Normalization**: Implement robust $\lambda_{min}$ estimation and scalar scaling targets beyond current row-sum/Gershgorin proxies.
  - *Current Gap*: $\lambda_{max}$ estimation exists, but $\lambda_{min}$-aware optimal scalar initialization is missing.
- [ ] **Block-Diagonal SPD Preconditioning**: Add a block-diagonal mode to `precond_spd` for better spectral clustering.
  - *Current Gap*: Only diagonal/global scaling families (Jacobi, Ruiz, etc.) are currently available.

## Priority 2: New Methods

- [ ] **Block-Lanczos Direct Apply**: Implement block-Lanczos/Krylov kernels for $A^{-1/p} B$ as an alternative to Chebyshev.
  - *Current Gap*: No Lanczos-family inverse-root apply kernel is currently present.
- [ ] **Mixed-Precision Iterative Refinement**: Add a mixed-precision factorization + refinement path for SPD $p > 1$ when $k \approx n$.
  - *Current Gap*: Direct polynomial methods are available, but there is no high-accuracy fallback for fat-RHS regimes.

## Future Explorations

- **Staged Policy Optimization**: Combining minimax polynomials for early contraction with residual-binomial steps for late-stage refinement.
- **Runtime Coefficient Auto-Tuning**: Using a lookup table indexed by $(p, 	ext{degree}, \kappa)$ to select the optimal contraction objective at runtime.
