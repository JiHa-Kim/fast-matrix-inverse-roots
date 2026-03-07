# Fast finite-precision polar and inverse roots: lean execution plan (log-centered)

## 1. Goal

Compute an ML-useful approximation to the polar factor
$$
Q = \operatorname{polar}(G) = G(G^T G)^{-1/2}
$$
as fast as possible in finite precision.

Primary objective:
$$
\text{minimize wall time to target applied quality, not raw root approximation error.}
$$

The same framework should later extend to SPD inverse square roots, inverse $r$-th roots, and applied transforms such as
$$
G P^{-s/r}.
$$

---

## 2. What quality means

Assume $m \ge n$ (otherwise swap sides and certify on the smaller side). Define
$$
B := G^T G,
\qquad
\widehat Q := GZ,
\qquad
S := \widehat Q^T \widehat Q = Z^T B Z.
$$

The error certificate is
$$
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
\kappa(S) \le \frac{1+\rho_2}{1-\rho_2}.
$$

### Log-space "uniformity" coordinate (the right global geometry)

Define the log-width of the SPD certificate:
$$
\eta(S) := \frac12 \log\!\left(\frac{\lambda_{\max}(S)}{\lambda_{\min}(S)}\right)
= \frac12 \log \kappa(S).
$$

Equivalently, if $U := GZ$ has singular values $\{s_i\}$, then $\lambda_i(S)=s_i^2$ and
$$
\eta(S) = \log\!\left(\frac{s_{\max}}{s_{\min}}\right) = \log \kappa(U).
$$

Also define a log-center drift:
$$
c(S) := \frac12\big(\log \lambda_{\max}(S) + \log \lambda_{\min}(S)\big)
= \log \sqrt{\lambda_{\max}(S)\lambda_{\min}(S)}.
$$

Interpretation:
- $\eta(S)$ measures multiplicative spread / uniformity (primary global width metric).
- $c(S)$ measures multiplicative scale drift away from $1$.

If scalar rescaling of $Z$ is allowed (cheap), then rescale
$$
Z \leftarrow e^{-c(S)/2} Z
\quad\Longrightarrow\quad
S \leftarrow e^{-c(S)} S,
$$
which recenters the spectrum multiplicatively so that
$$
\lambda\big(e^{-c(S)} S\big) \subseteq [e^{-\eta(S)},\,e^{\eta(S)}].
$$

Target tiers (evaluated on $E=S-I$):
$$
\text{light: } \delta_F \le 0.35 \quad (\text{or } 0.5 \text{ if very cheap}),
$$
$$
\text{medium: } \delta_F \le 0.20,
\qquad
\text{strong: } \delta_F \le 0.10.
$$

---

## 3. Compare policies, not isolated polynomials

We compare three policy families.

### A. Direct
Update $G$ directly with odd matrix polynomials/rationals, e.g.
$$
X_+ = X q(X^T X).
$$
For a cubic family,
$$
X_+ = aX + bX(X^T X) + cX(X^T X)^2.
$$

### B. Gram-side
Form
$$
B = G^T G,
$$
refine only
$$
Z \approx B^{-1/2},
$$
then apply once:
$$
\widehat Q = GZ.
$$

### C. Hybrid
Do a small number of direct global-compression steps, then switch to small-side refinement and apply once at the end.

The winner is the policy with the best time-to-target and acceptable tail risk.

---

## 4. Core exact fact

For a local step
$$
Z_+ = Z q(S),
\qquad
S = Z^T B Z,
$$
we get
$$
S_+ = q(S) S q(S).
$$

So certificate eigenvalues evolve by the scalar map
$$
x \mapsto \phi(x) := x q(x)^2.
$$

The same map appears for direct odd updates on $G$. Therefore direct and Gram-side methods share the same exact local scalar dynamics. The difference is cost, state location, certification, and finite-precision behavior.

### Additive local contraction metric (near $1$)
For a candidate $q$ and additive band around $1$,
$$
m_q(\rho) := \sup_{x \in [1-\rho,\,1+\rho]} |x q(x)^2 - 1|.
$$
Exact arithmetic guarantee: if $\rho_2(S)\le \rho$, then
$$
\rho_2(S_+) \le m_q(\rho).
$$

### Multiplicative (log-space) global contraction metric
For log-band $x \in [e^{-\eta},e^\eta]$, define
$$
\psi(z) := \log\!\big(\phi(e^z)\big),
\qquad z=\log x,
$$
and the output log-width
$$
\eta_\phi(\eta) :=
\frac12\Big(\sup_{|z|\le \eta}\psi(z) - \inf_{|z|\le \eta}\psi(z)\Big).
$$

If scalar recentering is allowed after the step, then the main global objective of a compression step is to shrink $\eta(S)$:
$$
\eta(S_+) \ \le\ \eta_\phi(\eta(S)).
$$

For reciprocal-symmetric updates (common in inverse-root design), $\phi(1/x)=1/\phi(x)$, hence $\psi(-z)=-\psi(z)$ and the log-center drift is automatically $0$.

Example: Mobius family $q_c(x) = \dfrac{x+c}{cx+1}$ has $\phi(x) = x\left(\dfrac{x+c}{cx+1}\right)^2$ and $q_c(1/x) = \dfrac{1}{q_c(x)}$.



---

## 5. Default design: two phases

### Phase 1: global compression (log-centered)
Use a cheap, wide-basin, bf16-safe policy to reduce log-width $\eta(S)$ quickly.

Default candidates:
- direct odd degree-$3$ or degree-$5$ schedules,
- simple adaptive direct schedules,
- Frobenius-based scaling as a fallback (upper-bound control),
- Gram-side scaling / preprocessing when clearly helpful,
- one-solve rational Gram-side compression steps if Cholesky/solve is cheap enough.

Measure and compare Phase 1 steps primarily by $\eta$ shrinkage per wall time (with guards).

### Phase 2: local finish (near 1)
Use one or two aggressive local steps designed from the certificate map once the spectrum is near $1$.

Local model:
- work near $x=1$ (additive coordinate),
- represent $q$ in shifted Chebyshev form,
- evaluate with Clenshaw,
- optimize the deployed scalar step directly in bf16.

Rationale: additive coordinates are efficient and stable once the spectrum has been multiplicatively compressed so that $x\approx 1$.

---

## 6. Exact bf16 local design problem (terminal step)

For the local scalar step, fix the deployed model
$$
t = \operatorname{rn}_{bf16}(x-1),
$$
$$
q(x) = \sum_{j=0}^d c_j T_j(t),
$$
with bf16 Clenshaw evaluation, and deployed certificate map
$$
\Phi(x) := \operatorname{rn}_{bf16}\!\left(x \cdot \operatorname{rn}_{bf16}(q(x)^2)\right).
$$

Let the terminal target set be
$$
\mathcal T_\tau := \{ y \in \mathrm{BF16} : |y-1| \le \tau \}.
$$

For each degree $d$, solve the discrete predecessor problem:
find the largest contiguous bf16 input band around $1$ such that
$$
\forall x \in \mathcal X_d,
\qquad
\Phi(x) \in \mathcal T_\tau.
$$

Score band width by the log-width
$$
\eta(\mathcal X_d) := \frac12 \log\!\left(\frac{x_{\max}}{x_{\min}}\right),
$$
for $\mathcal X_d = [x_{\min},x_{\max}] \cap \mathrm{BF16}$.

Practical rule:
- optimize $\eta$ for raw basin width,
- optimize $\eta / \text{cost}(d)$ for step efficiency.

Default degree search for terminal local steps:
$$
d \in \{2,3,4\}.
$$

---

## 7. Finite-precision stance

Separate three levels.

### A. Exact arithmetic theory
Rigorous scalar map:
$$
x \mapsto x q(x)^2,
$$
with exact contraction bounds via $m_q(\rho)$ or log-width maps $\eta_\phi$.

### B. Exact scalar bf16 deployment model
Centered-at-$1$ Chebyshev + bf16 Clenshaw predecessor solve above.

### C. Real matrix-kernel deployment
GEMM accumulation, reduction order, casts, and backend details. Must be calibrated empirically.

Do not claim:
- a universal bf16 floor,
- a universal GEMM perturbation constant,
- theorem-level deployed guarantees for full matrix kernels without calibration.

---

## 8. Certification and guards

Always certify on the small side.

Symmetrize before spectral measurement:
$$
S_{\mathrm{sym}} := \tfrac12(S + S^T).
$$

Then measure (fp32/fp64 on small side when cheap enough):
$$
\rho_2 = \|S_{\mathrm{sym}} - I\|_2,
\qquad
\delta_F = \|S_{\mathrm{sym}} - I\|_F,
\qquad
\eta = \frac12 \log\!\left(\frac{\lambda_{\max}(S_{\mathrm{sym}})}{\lambda_{\min}(S_{\mathrm{sym}})}\right).
$$

Every deployed policy should include:
- NaN/Inf detection,
- non-monotone spike detection,
- overshoot detection,
- one restart or rescale path,
- one safer fallback.

---

## 9. Cost model and regime split

Assume $m \ge n$.

A direct odd step costs roughly:
- one tall Gram-like build,
- small-side products,
- one tall apply back to $G$.

A Gram-side policy costs:
- one-time formation of $B = G^T G$,
- only small-side refinement after that,
- one final apply $GZ$.

So the regime split is empirical.

Key break-even comparison:
- one more direct step,
versus
- one more small-side step plus the final apply.

Expectation:
- direct may win near square or for light targets,
- Gram-side may win for very rectangular matrices,
- hybrid may win in the middle.

But this is a benchmark question, not a theorem.

---

## 10. Minimal benchmark plan

### Synthetic coverage
Sweep:
$$
\frac{m}{n} \in \{1,2,4,8,16,32\},
$$
with several $n$ values and spectra such as:
- flat,
- moderate decay,
- severe ill-conditioning,
- clustered endpoints,
- two-mass adversarial mixtures.

### Real snapshots
Use saved training matrices or Gram snapshots whenever possible.

### Record per run
- wall time (median and $p95$),
- final $\rho_2$, $\delta_F$, and $\eta$,
- number of tall passes,
- number of small-side GEMMs,
- switch point for hybrid,
- scaling used (including recentering),
- guard triggers,
- failures and fallback use,
- monotonicity of the certificate.

---

## 11. Default execution order

### Step 1
Lock the local evaluator to centered-at-$1$ shifted Chebyshev with Clenshaw.

### Step 2
Solve the exact scalar bf16 predecessor problem for degrees $2,3,4$ and pick:
- the widest local basin by $\eta$,
- the best $\eta / \text{cost}$ tradeoff.

### Step 3
Build a minimal Phase 1 baseline using direct odd updates with simple safe scaling and log-width tracking.

### Step 4
Add a Gram-side baseline with (optional) multiplicative recentering and refinement.

### Step 5
Benchmark three policy families:
- direct,
- Gram-side,
- hybrid,
using aspect-ratio sweeps and real snapshots.

### Step 6
For each target tier, ship:
- one default fast policy,
- one safer fallback.

---

## 12. What not to optimize directly

Do not optimize
$$
\|Z - B^{-1/r}\|
$$
unless it clearly improves the applied objective.

For polar and whitening-style use, the primary object is
$$
S = Z^T B Z \approx I,
$$
or equivalently
$$
\widehat Q^T \widehat Q \approx I.
$$

So optimize applied whitening quality and time, not abstract root approximation in isolation.

---

## 13. Immediate tasks

1. Finish the exact scalar bf16 predecessor solve for degrees $2,3,4$.
2. Choose the best terminal local step by $\eta$ and by $\eta / \text{cost}$.
3. Implement a minimal direct Phase 1 baseline with certification and log-width $\eta$ tracking.
4. Add a Gram-side baseline with refinement and optional multiplicative recentering.
5. Benchmark direct vs Gram-side vs hybrid on aspect-ratio sweeps and real snapshots.
6. Ship one default fast policy and one fallback per target tier.

---

## 14. One-sentence project statement

Find the fastest finite-precision policy for producing an ML-useful polar factor approximation by combining bf16-safe global log-width compression, exact local certificate design, small-side certification, and aspect-ratio-aware switching between direct, Gram-side, and hybrid policies.