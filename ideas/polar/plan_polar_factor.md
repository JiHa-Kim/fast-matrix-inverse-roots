# Fastest path to ML-useful polar factor:
# direct odd update vs preconditioner-first vs hybrid

## 0. Core question and success criteria

Given $G \in \mathbb{R}^{m \times n}$, compute
$$
Q = \operatorname{polar}(G) = G(G^T G)^{-1/2}
$$
as fast as possible on GPU in low precision (bf16/FP16 compute), subject to runtime-checkable correctness.

We optimize *applied* orthogonality, not SVD-accurate factors.

Let $r=\min(m,n)$. Use the small-side certificate:
- if $m \ge n$: $S := Q^T Q \in \mathbb{R}^{n\times n}$
- if $m < n$:  $S := Q Q^T \in \mathbb{R}^{m\times m}$

Metrics (do not mix):
$$
\rho_2 := \|S-I\|_2,\qquad \delta_F := \|S-I\|_F,
\qquad \rho_2 \le \delta_F \le \sqrt{r}\,\rho_2.
$$

Target tiers:
- light: $\delta_F \le 0.35$
- medium: $\delta_F \le 0.20$
- strong: $\delta_F \le 0.10$

Primary benchmark: wall time to hit tier (median + p95), plus failure/guard rates.

## 1. The meta iteration and the key equivalence

### 1.1 Direct odd polynomial on $G$ (current meta)
Baseline family uses odd matrix polynomials (GEMM-friendly):
$$
X_{k+1} = a_k X_k + b_k X_k(X_k^T X_k) + c_k X_k(X_k^T X_k)^2.
$$
If $X_k = U\Sigma V^T$, then $X_{k+1} = U\big(\Sigma\,p_k(\Sigma^2)\big)V^T$ where
$$
p_k(t) = a_k + b_k t + c_k t^2,
\qquad \sigma \mapsto \sigma\,p_k(\sigma^2).
$$
This is exactly the singular-value map you are optimizing (make $\sigma \approx 1$ quickly).

Polar Express is a strong instance of this approach: GEMM-only odd updates, Frobenius-based normalization and bf16 stabilizations, and adaptive coefficient choice via minimax design. 

### 1.2 Preconditioner-first on the Gram (inverse modulus view)
Let the small-side Gram be
- if $m\ge n$: $B := G^T G \in \mathbb{R}^{n\times n}$
- else:        $B := G G^T \in \mathbb{R}^{m\times m}$

Compute/apply $Z \approx B^{-1/2}$ and return $Q = GZ$ (or $Q=ZG$ depending on orientation). Then
$$
S = Q^T Q = Z^T B Z.
$$

### 1.3 The crucial point (do not miss this)
The direct and preconditioner views do **not** enable fundamentally different singular-value maps.

Any direct odd update can be written as
$$
Q_{k+1} = Q_k\,q_k(S_k),\qquad S_k = Q_k^T Q_k,
$$
and the same $q_k$ can be applied by maintaining $Z_k$ with
$$
Z_{k+1} = Z_k\,q_k(S_k),\qquad S_k = Z_k^T B Z_k.
$$

So the preconditioner advantage is not a larger polynomial class. It is:
- iterating on the small side ($r\times r$) instead of repeatedly multiplying tall matrices,
- cleaner certificates and switch logic,
- cheaper local refinement when $m\gg n$ (or $n\gg m$),
- more freedom to use aggressive local steps because they are cheap on $r\times r$.

## 2. The three policy families to compare

### Family D (Direct-only): odd updates on $G$
Run direct odd steps until tier or cap.
This is the baseline to beat (Polar Express schedule is the must-match strong baseline). 

### Family P (Preconditioner-first): Gram-side refinement then apply once
1) Form $B$ once.
2) Iterate on small-side state/certificate to get $Z$.
3) Apply once: $Q = GZ$.
This should win when aspect ratio is large and/or you need >1-2 refinement steps.

AOL-style preprocessing can shift the frontier enough to remove an iteration in low-step regimes; include it as a preprocessing variant. 

### Family H (Hybrid): a few direct steps then small-side finish
Do 1-2 direct steps (fast global compression), then switch to Gram-side local finish and apply once.
This is the high-priority "likely winner" in the middle regime.

## 3. The reverse-engineered 2-phase structure (shared across families)

### Phase 2 (local finish, certified and aggressive)
Once $S$ is near identity, use 1-2 low-degree polynomials designed on $[1-\rho,1+\rho]$:
- exact arithmetic: eigenvalues map via $x \mapsto x q(x)^2$
- contraction number:
  $$
  m_q(\rho) := \sup_{x\in[1-\rho,1+\rho]} |x q(x)^2 - 1|
  $$
- bf16 deployment: treat deviation as calibrated envelope + guards (not a universal constant).

(Reuse the Phase 2 local designer + verifier you already built.)

### Phase 1 (global compression, bf16-safe)
Goal: enter the Phase 2 band with minimal time and low risk.
Candidates:
- Polar Express adaptive minimax schedule (direct) 
- fixed odd degree-3/5 schedules (direct)
- preprocessing that tightens the starting spectrum (Frobenius scaling, AOL-like) 

Reverse-engineering workflow:
1) Choose Phase 2 terminal band $\rho_\star$ and a 1-step local polynomial.
2) Design transition step(s) that land inside $\rho_\star$ with calibrated margin.
3) Only then pick Phase 1 policy to hit the transition radius fastest.

## 4. The most important missing operational piece: break-even experiment

Before doing giant sweeps, fit an empirical time model on your actual GPU and kernel path:

Let:
- $T_{\text{tall}}$: time for one tall-side pass involving an $(m\times n)\cdot(n\times n)$ GEMM (or symmetric equivalent)
- $T_{\text{small}}$: time for one $n\times n$ refinement step (your exact Phase 2 apply path on small side)
- $T_{\text{apply}}$: time for final apply $GZ$ (one tall GEMM)
- $T_{\text{gram}}$: time to form $B=G^T G$ once

Then compare:
- one extra direct step (typically ~2 tall passes + small work)
vs
- one extra small-side refinement step (+ maybe the eventual final apply if not yet counted)

Use this to predict an aspect-ratio frontier where preconditioner-first/hybrid overtakes direct-only.

Deliverable:
- `report_polar_break_even.md` + JSON with measured times and the inferred crossover.

## 5. Cost model (for intuition, then validate empirically)

Assume $m\ge n$ (swap similarly otherwise).

### Direct odd step
Approx FLOP scales:
- form $A=X^T X$: $m n^2$
- form $A^2$: $n^3$
- apply $X \leftarrow X p(A)$: $m n^2$
So one step ~ $2 m n^2 + n^3$ (kernel constants dominate in practice).

### Preconditioner-first
- one-time: form $B=G^T G$: $m n^2$
- per step: only $n^3$-scale work
- final: apply $Q=GZ$: $m n^2$
So big cost is 2 tall GEMMs total, independent of iteration count.

### Hybrid
Pay 1-2 direct steps (some tall GEMMs), then small-side finishing, then one final apply.

## 6. Scaling / preprocessing sweep (do not hardcode one choice)

Minimum required baseline:
- Frobenius-based normalization with a safety factor (as used in Polar Express code) 

Include as candidate options:
- AOL-like preprocessing (can change low-step frontier) 
- Gram scaling variants (e.g., based on Frobenius inner products; useful when working directly with $B$) 

Treat scaling as part of the policy, not a fixed assumption.

## 7. Benchmark suite (what to sweep)

### Synthetic sweeps
- aspect ratios: $m/n \in \{1,2,4,8,16,32\}$
- sizes: several $n$ values (cover typical ML regimes)
- spectral shapes (via controlled singular values):
  - near-flat
  - moderately decaying
  - highly ill-conditioned
  - clustered endpoints / two-mass mixtures (stress minimax)

### Real ML snapshots
Use saved training snapshots of $G$ (and/or gradients/moment estimates):
- log shape, norms, estimated spread
- run all policy families to tier or cap

## 8. What to log (non-negotiable)

Per run:
- wall time (median/p95 across repeats)
- final $\delta_F$, $\rho_2$
- number of tall passes over $G$
- number of $n\times n$ GEMMs
- guard triggers (NaN/Inf, spike, overshoot)
- which fallback used

## 9. Implementation contract: certification + guards

At each step:
1) form certificate $S$ on the small side
2) symmetrize $S_{sym} = 0.5(S+S^T)$
3) measure $\rho_2$ and $\delta_F$ (eigs on $r\times r$ are feasible up to a few thousand; otherwise use power iteration for $\rho_2$)
4) accept if within policy envelope; else guard:
   - rescale and retry once, or
   - fall back to Phase 1 safe compression

This is how you get practical "always safe" behavior without pretending there is a universal bf16 theorem constant.

## 10. Deliverables (minimal, actionable)

### Code
- `bench_polar_policies.py` (unified harness)
- `policy_direct_odd.py` (match strongest direct baseline; include Polar Express schedule) 
- `policy_precond_inv_modulus.py` (Gram-side 2-phase; small-side iterate then apply once)
- `policy_hybrid_switch.py` (1-2 direct steps then small-side finish)
- `analyze_break_even.py` (fit time model and frontier)
- `calibrate_hw_envelope.py` (estimate bf16 envelope and plateau for Phase 2)

### Reports
- `report_polar_break_even.md`
- `report_polar_aspect_ratio_sweep.md`
- `report_polar_real_snapshots.md`

## 11. Decision rule: what we ship

For each tier and shape regime:
1) choose lowest median wall time
2) require acceptable p95 and failure rate
3) publish a simple decision heuristic (aspect-ratio-aware):
   - direct-only for near-square and light targets if it wins
   - preconditioner-first for very rectangular
   - hybrid as default in the middle regime if it dominates