# PE-Quad (Quadratic Polynomial-Express)

PE-Quad is the primary iterative family used in this project for computing inverse $p$-th roots of SPD matrices. It uses quadratic polynomial updates for rapid convergence.

## Mathematical Form

At each step $t$, we build a quadratic polynomial:

$$
B_t = a_t I + b_t Y_t + c_t Y_t^2
$$

The state is then updated based on the mode:

- **Uncoupled**: Tracks only $X_t \approx A^{-1/p}$.
  - $Y_t = X_t^p A$
  - $X_{t+1} = X_t B_t$
- **Coupled**: Tracks $X_t$ and $Y_t$, maintaining the invariant $Y_t \approx X_t^p A$.
  - $X_{t+1} = X_t B_t$
  - $Y_{t+1} = B_t^p Y_t$ (updated using the commuting polynomial model)

## Implementations

- `inverse_proot_pe_quadratic_uncoupled`: Standard iterative root.
- `inverse_proot_pe_quadratic_coupled`: Stability-enhanced root tracking.
- `solve_spd`: High-level entrypoint that utilizes these kernels with automated preconditioning.

## Optimized Paths

- **Specializations**: Specialized kernels for $p=2$ (inverse sqrt) and $p=4$ minimize operation counts.
- **Terminal Step**: When only the result $Z = A^{-1/p} B$ is needed, the final $Y$ update is skipped to save one large **GEMM (General Matrix Multiply)**.
- **Symmetry**: Configurable symmetry guards (`symmetrize_Y`) maintain **SPD (Symmetric Positive Definite)** invariants in finite precision.

## Performance Profile

- **$p=1$**: Inverse-Newton updates often dominate in both speed and accuracy.
- **$p=2, 4$**: The coupled path is typically the fastest for production workloads, while the uncoupled path can provide better residual accuracy in some ill-conditioned cases.
- **$p > 4$**: Coupled methods remain the default choice for throughput.
