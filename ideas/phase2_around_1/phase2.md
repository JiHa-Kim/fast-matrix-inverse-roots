# Phase 2: Local Refinement for Fast Whitening (Minimal, Correct, Reproducible)

Phase 2 refines a preconditioner $Z$ for an SPD matrix $B$ with the application objective
$$
S := Z^T B Z \approx I,
$$
so that applying $U = GZ$ yields $U^T U = S$ close to identity.

This document separates what is rigorous in exact arithmetic from what must be calibrated empirically under bf16 GPU kernels.

## 1. Metrics (do not mix them)

We track two certificate errors:

- Spectral-radius deviation (Phase 2 design/control metric):
  $$
  \rho_2(S) := \|S-I\|_2 = \max_i |\lambda_i(S) - 1|.
  $$

- Frobenius whitening error (project metric):
  $$
  \delta_F(S) := \|S-I\|_F.
  $$

Always:
$$
\rho_2 \le \delta_F \le \sqrt{n}\,\rho_2.
$$

## 2. Rigorous contraction (exact arithmetic)

Assume $B \succ 0$, $S = Z^T B Z \succ 0$, and update
$$
Z^+ = Z\,q(S),
$$
where $q(\cdot)$ is a polynomial applied to $S$.

In exact arithmetic, $q(S)$ commutes with $S$ and
$$
S^+ = (Z^+)^T B Z^+ = q(S)^T S q(S) = q(S)\,S\,q(S).
$$
Therefore every eigenvalue transforms by the scalar map
$$
\lambda^+ = \lambda\,q(\lambda)^2.
$$

Define the scalar worst-case deviation for interval radius $\rho$:
$$
m_q(\rho) := \sup_{x \in [1-\rho,\,1+\rho]} \left|x\,q(x)^2 - 1\right|.
$$
Then the exact-arithmetic certificate guarantee is:
$$
\rho_2(S^+) \le m_q(\rho_2(S)).
$$

## 3. bf16 numerics: calibrated, not universal

bf16 GPU execution perturbs the exact map due to bf16 storage, bf16 GEMMs, evaluation order, and casting. There is no universal theorem-level constant that bounds this effect independent of backend and size.

We therefore use a calibrated envelope for the deployed code path (the implementation in `step_phase2_local`):
$$
\rho_2^{bf16}(S^+) \le m_q(\rho_2^{bf16}(S)) + \epsilon_{hw},
$$
and Phase 2 will plateau at an observed level $\rho_{plat}$.

Both $\epsilon_{hw}$ and $\rho_{plat}$ are measured by an end-to-end verifier on the target GPU(s), sizes, and spectra stress tests.

### Symmetry for measurement
Finite precision can introduce small skew components. For measuring eigenvalues, always symmetrize:
$$
S_{sym} := \tfrac12(S + S^T),
$$
and compute $\rho_2$ from $\lambda_i(S_{sym})$.

## 4. Two-step Phase 2 protocol (guarded)

We use two fixed low-degree polynomials:

- Transition polynomial $q_T$, valid for $\rho_2 \le \rho_{in}$.
- Terminal polynomial $q_\star$, intended for $\rho_2 \le \rho_\star$.

Nominal calibrated radii (current setup):
$$
\rho_{in} = 0.7653,
\qquad
\rho_\star = 0.0816.
$$

### Policy (minimal guards)
1) Measure $\hat\rho_2$ from $S_{sym}$ (or use a conservative surrogate; note $\rho_2 \le \delta_F$).
2) If $\hat\rho_2 \le \rho_\star$, apply terminal: $Z \leftarrow Z\,q_\star(S)$.
3) Else apply transition: $Z \leftarrow Z\,q_T(S)$, re-measure $\hat\rho_2$.
4) If $\hat\rho_2 \le \rho_\star$, apply terminal.
5) Else trigger a guard action (retry transition once, or fall back to Phase 1).

## 5. Reproducibility

- Polynomial design and scalar bf16-rounding-model checks are produced by `design_local_poly.py`.
- End-to-end bf16 kernel behavior (including calibration of $\epsilon_{hw}$ and $\rho_{plat}$) is produced by `verify_phase2_policy.py`, which runs multiple trials and spectra stress tests.