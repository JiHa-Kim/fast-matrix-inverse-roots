# Fast finite-precision polar: direct, Gram-side, and hybrid policies

## 0. Summary

Goal: compute an ML-useful approximation to
$$
Q = \operatorname{polar}(G) = G(G^T G)^{-1/2}
$$
as fast as possible in low precision, typically bf16 compute with higher-precision small-side certification.

We compare **policies**, not isolated polynomials:

1. **Direct**: update $G$ by odd matrix polynomials.
2. **Gram-side**: form $B = G^T G$, refine $Z \approx B^{-1/2}$ on the small side, then return $\widehat Q = GZ$.
3. **Hybrid**: do a few direct global-compression steps, switch to small-side refinement, then apply once.

The objective is **applied quality after use**, not raw approximation error of $Z$.

For $m \ge n$, the key certificate is
$$
S := \widehat Q^T \widehat Q = Z^T B Z, \qquad B = G^T G.
$$
We therefore optimize how fast we can make
$$
S \approx I,
$$
since this is what makes $\widehat Q$ approximately orthogonal.

The core exact fact is that both direct odd updates and Gram-side refinement induce the same certificate eigenvalue map
$$
x \mapsto x\,q(x)^2.
$$
So the real comparison is wall time, robustness, and when moving refinement to the small side is worth it.

---

## 1. Objective and metrics

Given $G \in \mathbb{R}^{m \times n}$, compute $\widehat Q \approx \operatorname{polar}(G)$ as fast as possible on real hardware.

For $m \ge n$,
$$
Q = G(G^T G)^{-1/2}.
$$
For $m < n$, swap sides analogously and certify on the smaller side.

Define the certificate
$$
S :=
\begin{cases}
\widehat Q^T \widehat Q, & m \ge n, \\
\widehat Q \widehat Q^T, & m < n,
\end{cases}
\qquad
E := S - I.
$$

Main metrics:
$$
\rho_2 := \|E\|_2,
\qquad
\delta_F := \|E\|_F.
$$

Always,
$$
\rho_2 \le \delta_F \le \sqrt{r}\,\rho_2,
\qquad
r := \min(m,n).
$$

If $\rho_2 < 1$, then
$$
\kappa(S) \le \frac{1+\rho_2}{1-\rho_2},
$$
so $\rho_2$ directly controls post-apply conditioning.

Target tiers:
$$
\text{light: } \delta_F \le 0.35 \quad (\text{or } 0.5 \text{ if very cheap}),
$$
$$
\text{medium: } \delta_F \le 0.20,
\qquad
\text{strong: } \delta_F \le 0.10.
$$

Primary objective:
- minimize wall time to hit a target tier,
- report median and $p95$ time,
- track failures, guard triggers, and monotonicity.

---

## 2. Two exact formulations

### 2.1 Direct odd updates on $G$

A direct step has the form
$$
X_{+} = X\,q(X^T X),
$$
where $q$ is a low-degree polynomial.

For the cubic family,
$$
X_{+} = aX + bX(X^T X) + cX(X^T X)^2.
$$

If $X = U\Sigma V^T$, then
$$
X_{+} = U\bigl(\Sigma\,q(\Sigma^2)\bigr)V^T,
$$
so singular values evolve by
$$
\sigma \mapsto \sigma\,q(\sigma^2).
$$

With the certificate
$$
S := X^T X,
$$
we get
$$
S_{+} = q(S)\,S\,q(S),
$$
hence certificate eigenvalues evolve by
$$
x \mapsto x\,q(x)^2.
$$

This is the current practical baseline: GEMM-only odd updates that drive singular values toward $1$.

### 2.2 Gram-side inverse square root, then apply once

For $m \ge n$, define
$$
B := G^T G.
$$

Compute only
$$
Z \approx B^{-1/2},
$$
then apply once:
$$
\widehat Q = GZ.
$$

The relevant certificate is
$$
S = \widehat Q^T \widehat Q = Z^T B Z,
$$
not $\|Z - B^{-1/2}\|$.

A generic Gram-side step is
$$
Z_{+} = Z\,q(S),
\qquad
S = Z^T B Z.
$$

Then again
$$
S_{+} = q(S)\,S\,q(S),
$$
so the same eigenvalue map holds:
$$
x \mapsto x\,q(x)^2.
$$

Thus direct and Gram-side methods share the same exact local scalar dynamics. The difference is not a larger function class, but cost, state location, certification, and finite-precision behavior.

---

## 3. What Gram-side refinement may buy us

Even though the certificate map is the same, Gram-side refinement may still win because:

- the refinement state is entirely on the small side,
- certification is naturally small-side,
- aggressive local steps can be much cheaper when $G$ is rectangular,
- the tall apply $\widehat Q = GZ$ can be delayed until the end,
- hybrids may beat both extremes in the middle regime.

For very rectangular $G$, forming $B$ once and iterating mostly on the small side may beat repeated tall passes. This is an empirical break-even question, not an assumption.

---

## 4. Reverse-engineered local design

Local design should be posed on the certificate.

For a candidate polynomial $q$, define
$$
m_q(\rho)
:=
\sup_{x \in [1-\rho,\,1+\rho]}
\left|x\,q(x)^2 - 1\right|.
$$

Exact arithmetic gives:
if
$$
\rho_2(S) \le \rho,
$$
then after one step
$$
\rho_2(S_{+}) \le m_q(\rho).
$$

This is the correct object to optimize.

### Reverse problem

Let
$$
u := 2^{-7},
$$
the bf16 spacing near $1$.

Use a terminal target band near identity such as
$$
[1-u,\,1+u],
$$
with the understanding that deployed bf16 margins must be calibrated empirically.

For a chosen polynomial class and degree, define
$$
R(\tau)
:=
\sup\left\{
\rho \ge 0 :
\exists q \text{ with } m_q(\rho) \le \tau
\right\}.
$$

Starting from a terminal radius $\tau_0$, build a backward chain
$$
\tau_{k+1} := R(\tau_k).
$$

This answers:
- how many aggressive local steps can be stacked backward,
- which degrees are best at each stage,
- where Phase 1 should hand off to Phase 2,
- when lower degree beats higher degree because it gives a larger certified basin.

### Basis and evaluation

For local design and deployment:
- use the **Chebyshev basis** for $d \ge 3$,
- evaluate with **Clenshaw recurrence**,
- for $d=2$, benchmark against monomial/Horner and keep whichever is faster and more stable on the target hardware.

The best basis at low degree is empirical, not assumed.

---

## 5. Two-phase policy design

### Phase 1: global compression

Phase 1 should be bf16-safe, wide-basin, cheap, and GEMM-friendly. Its job is to bring the certificate into a profitable local basin quickly and with low failure risk.

Candidates include:
- direct odd degree-$3$ or degree-$5$ schedules,
- adaptive direct schedules,
- Frobenius normalization with safety-factor scaling,
- Gram-side scaling and preprocessing variants,
- AOL-style preprocessing ideas if they reduce steps in practice.

### Phase 2: local finish

Phase 2 is a small library of aggressive cheap local steps designed by the reverse problem above.

For each candidate step, record:
- certified input radius,
- output contraction profile $m_q$,
- coefficients,
- deployed cost,
- finite-precision margin.

Recommended workflow:
1. choose the terminal target,
2. design the terminal local step,
3. design the transition step into that basin,
4. only then choose the global policy that reaches the transition basin fastest.

Use a guarded handoff:
- if measured $\rho_2 \le \rho_*$, apply the terminal step,
- otherwise use a transition step and recheck,
- if still outside, retry once or fall back.

---

## 6. Finite-precision stance

Separate:

1. **Exact arithmetic theory**: certificate dynamics and interval guarantees.
2. **Deployed bf16 behavior**: evaluation error, GEMM perturbations, plateaus, and guard logic.

Do **not** claim:
- a universal bf16 floor,
- a universal GEMM perturbation constant,
- theorem-level deployed guarantees without calibration.

Instead:
- solve the exact scalar design problem,
- test scalar bf16 emulation for candidate steps,
- calibrate matrix-level margins on real kernels, sizes, and codepaths,
- use runtime guards.

### Certification

Always symmetrize on the small side:
$$
S_{\mathrm{sym}} := \tfrac12(S + S^T).
$$

Then measure, preferably in fp32 or fp64,
$$
\rho_2 = \|S_{\mathrm{sym}} - I\|_2,
\qquad
\delta_F = \|S_{\mathrm{sym}} - I\|_F.
$$

Use eigendecomposition when the small side is cheap enough; otherwise use a reliable estimator for $\rho_2$.

### Guards

Every policy should include:
- NaN/Inf detection,
- non-monotone spike detection,
- overshoot outside the intended band,
- restart or rescale once,
- fallback to a safer step.

---

## 7. Cost model and regime split

Assume $m \ge n$.

A direct odd step typically includes:
- one tall Gram-like product $X^T X$,
- one or more small-side products,
- one tall apply back to $X$.

So the cost is roughly
$$
O(mn^2 + n^3 + mn^2),
$$
with kernel overhead, memory traffic, and casting often dominating FLOP counts.

The Gram-side route costs:
- one-time formation of
  $$B = G^T G,$$
- then only small-side refinement,
- one final apply
  $$\widehat Q = GZ.$$

So the rectangular cost is concentrated in the initial Gram build and the final apply.

The key empirical break-even experiment is to measure:
- $T_{\mathrm{gram}}$: time to form $B = G^T G$,
- $T_{\mathrm{apply}}$: time for one final apply $GZ$,
- $T_{\mathrm{direct}}$: time for one more direct step,
- $T_{\mathrm{small}}$: time for one more small-side step.

Then compare:
- one more direct step,
versus
- one more small-side step plus the eventual final apply.

This determines where direct-only, Gram-only, and hybrid policies cross over.

---

## 8. Benchmark plan

### Synthetic suite

Sweep:
$$
m/n \in \{1,2,4,8,16,32\},
$$
multiple $n$, and spectral shapes such as:
- near-flat,
- moderate decay,
- severe ill-conditioning,
- clustered endpoints,
- two-mass or adversarial mixtures.

### Real ML snapshots

Use saved training matrices or Gram snapshots whenever possible.

For each snapshot:
- log shape and aspect ratio,
- log norms and basic spread indicators,
- run all policy families to target or cap,
- compare median and tail time.

### Per-run logging

Record:
- wall time,
- median and $p95$ over repeats,
- final $\rho_2$ and $\delta_F$,
- tall passes over $G$,
- small-side GEMMs,
- hybrid switch point,
- scaling or preprocessing,
- guard triggers,
- failures and fallback use,
- monotonicity of the certificate.

The unit of comparison is the full policy, not an isolated scalar polynomial.

---

## 9. Immediate tasks

### A. Exact local reverse design

For each candidate degree and class, solve
$$
\max_{\rho,q} \rho
\quad \text{s.t.} \quad
\sup_{x \in [1-\rho,\,1+\rho]}
|x\,q(x)^2 - 1|
\le \tau
$$
for a chain of target radii $\tau$ starting near
$$
\tau_0 = 2^{-7}.
$$

Output:
- maximal certified radius,
- coefficients,
- contraction profile,
- degree comparison.

### B. Scalar bf16 deployment study

Emulate scalar bf16 evaluation for the same steps and estimate how much of the exact basin survives under the chosen evaluation scheme.

### C. Matrix-level calibration

On real bf16 kernels, estimate:
- practical plateaus,
- safe deployment margins,
- whether Chebyshev/Clenshaw or monomial/Horner is better in the low-degree regime,
- in particular whether $d=2$ should use Chebyshev or monomials.

### D. Policy comparison

Benchmark direct, Gram-only, and hybrid policies to the target tiers and determine:
- winners by aspect ratio,
- winners by target strength,
- one default fast policy,
- one safer fallback.

---

## 10. Ship rule and scope

For each target tier and shape regime, ship the policy with:
1. lowest median wall time,
2. acceptable $p95$,
3. low failure and guard-trigger rate,
4. clean runtime certification.

Expected pattern:
- direct-only may remain best near square or for light targets,
- Gram-only may win for very rectangular matrices,
- hybrid may dominate in the middle regime.

But this must be decided by benchmark evidence.

Scope order:
1. polar,
   $$
   Q = G(G^T G)^{-1/2},
   $$
2. SPD inverse square root for whitening or preconditioning,
3. inverse $r$-th roots for small integer $r$,
4. applied transforms
   $$
   G P^{-s/r}
   $$
   when the same certificate logic applies.

---

## 11. Project statement

The project is to find the fastest finite-precision policy for producing an ML-useful polar factor approximation by comparing:

- direct odd updates on $G$,
- Gram-side refinement of $Z \approx (G^T G)^{-1/2}$ followed by $\widehat Q = GZ$,
- hybrids.

The central exact object is the certificate map
$$
x \mapsto x\,q(x)^2.
$$

The central engineering task is to combine:
- reverse-engineered local design,
- bf16-safe global compression,
- calibrated deployment margins,
- aspect-ratio-aware switching,
- basis-aware evaluation,
- wall-time benchmarking,

so that we can ship one default fast policy and one safer fallback for each practical target tier.