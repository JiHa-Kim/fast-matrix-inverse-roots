# Fast Matrix Preconditioning and Ridge Analysis

If the only thing you ever do with $B^{-1/2}$ is apply it (via $U = GZ$ where $Z \approx B^{-1/2}$, $B = G^T G$), then the semi-principled objective is not "approximate $B^{-1/2}$ accurately", it is:

1. **Make the preconditioned Gram close to identity**:
   $$
   S := Z^T B Z \approx I
   $$
   because this is exactly
   $$
   U^T U = Z^T (G^T G) Z = S
   $$
   which measures how much you whiten or equalize directions.

2. **Do it in minimum wall-clock time**, under bfloat16 (bf16) stability constraints.

That immediately tells you what to sweep and what to stop on.

---

## 1. What to Measure (Certificates for Preconditioning)

### Primary Metric: Whiteness / Anisotropy After Applying
Compute in bf16 on the small side:
$$
E := S - I, \quad \delta_F := \|E\|_F, \quad \delta_2 \le \delta_F
$$
Then singular values of $U$ satisfy:
$$
\sqrt{1-\delta_2} \le \sigma_i(U) \le \sqrt{1+\delta_2}
$$

For preconditioning, you typically do not need $\delta_F \ll 1$. A practical set of targets:
* **Light preconditioning**: $\delta_F \le 0.5$ (keeps $\sigma(U)$ in a moderate band)
* **Medium preconditioning**: $\delta_F \le 0.2$
* **Strong preconditioning**: $\delta_F \le 0.1$

This is a clean stop rule that matches your real objective.

### Secondary Metric: Effective Condition Number Reduction
A cheap estimate from the same certificate:
$$
\kappa(U^T U) \le \frac{1+\delta_2}{1-\delta_2} \le \frac{1+\delta_F}{1-\delta_F}
$$
So you can stop when this upper bound falls below a target like 10 or 5. That is more directly tied to "good enough preconditioning".

### Tertiary Metric: Progress Per Second (Not Per GEMM)
Define an action as "one update of $Z \leftarrow Z q(S)$" plus your fixed-cost ops (compute $S$, measure $\delta_F$, optional restart). The metric is:
$$
\frac{\log\left(\frac{1+\delta_{t}}{1-\delta_{t}}\right) - \log\left(\frac{1+\delta_{t+1}}{1-\delta_{t+1}}\right)}{\text{wallclock}(\text{action})}
$$
Or even simpler, $\Delta \log \kappa / \Delta t$.

This automatically accounts for GEMM optimization, kernel launch overhead, casts, reductions, etc.

---

## 2. What to Sweep (Small, Structured, and Informative)

### 2.1 The Policy, Not Just the Degree
You do not want to sweep a single degree in isolation; you want to sweep a small set of *policies* that map the observed $\delta_F$ to a polynomial choice. A good minimal policy family (covers most of the Pareto front) is:

**Global (Phase 1):**
* $d \in \{3, 5\}$ bf16-safe one-sided polynomials (no-overshoot) for $[\ell, 1]$ after scaling.
* Restart block length $T_{\text{block}} \in \{1, 2\}$.

**Local (Phase 2):**
* $d=5$ weighted minimax on $[-\rho, \rho]$ with online $\rho = \gamma \delta_F$.
* Optionally include $d=3$ local as a cheaper alternative if $d=5$ is overkill.

Then sweep policies like:
* **P1**: global $d=3$ until switch, then local $d=5$ once
* **P2**: global $d=3$ until switch, then local $d=5$ twice
* **P3**: global $d=5$ once, then local $d=5$ once
* **P4**: global $d=3$ twice (or until switch), then local $d=3$ or $d=5$

This is far more informative than sweeping $d \in \{3, 5, 7, 9\}$ everywhere.

### 2.2 Switch Threshold ($\rho_{\text{switch}}$)
This is the single most important hyperparameter for "fast preconditioning."
Sweep $\rho_{\text{switch}} \in \{0.7, 0.5, 0.35, 0.2\}$.

**Interpretation:** When $\delta_F \le \rho_{\text{switch}}$, stop using global polynomials and only use local ones. A phase-2 degree-5 table suggests local is very strong once you are at or below about $0.5$.

### 2.3 Interval Inflation ($\gamma$) for Online Robustness
This is the principled knob that replaces ad hoc rescales:
$$
\rho_{\text{design}} = \gamma \delta_F
$$
Sweep $\gamma \in \{1.1, 1.25, 1.5\}$. This is where bf16 stability usually lives.

### 2.4 Safety Mode for Global Step
For Phase 1 you already implemented a bf16-safe design that returns $\mu_*$ and $m = \min g$. The only sweep you need is:
* Degree $d \in \{3, 5\}$
* (Optional) coefficient bound / shape constraint if you add it.

Based on previous results at $\ell=10^{-3}$, $d=3$ gives best "progress per GEMM", and in practice it often also wins per second because the rest of the step cost is dominated by GEMMs anyway. But you should still test the policy-level wall-clock.

### 2.5 Ridge and Jacobi Eps as Stability Knobs
These should not be huge sweeps. Use 2-3 values each:
* **Ridge**: $\delta = \alpha \cdot \frac{\mathrm{tr}(B)}{n}$, where $\alpha \in \{0, 10^{-6}, 10^{-5}\}$
* **Jacobi eps**: $\epsilon = \beta \cdot \frac{\mathrm{tr}(B)}{n}$, where $\beta \in \{10^{-8}, 10^{-6}\}$

In ML practice, these are there to prevent rare catastrophes, not to optimize performance.

---

## 3. Semi-Principled Default (Starting Point)

If you want a reasonable "good enough and fast" preconditioner:
* **Stop target**: $\delta_F \le 0.2$ (medium) or $\le 0.35$ (light).
* **Global**: bf16-safe degree 3, block length 1 or 2.
* **Switch**: $\rho_{\text{switch}} = 0.5$.
* **Local**: degree 5 weighted minimax with $\gamma = 1.25$, run 1 step, then re-check $\delta_F$, run a second only if needed.

That policy tends to be robust and cheap.

---

## 4. How to Run the Sweep Efficiently

Do it on a dataset of saved snapshots of $B = G^T G$ (or $G$) collected during training without training loops:

For each snapshot:
1. Run each policy until it reaches $\delta_F \le \eta$ or hits a max-step cap.
2. Record:
   * Total time
   * Final $\delta_F$
   * Whether any guard triggered (non-monotone $\delta_F$, NaNs, overshoot)

Then pick the policy minimizing median time subject to a small tail failure rate. This is aligned with your real objective (fast approximate preconditioning) and avoids overfitting to exact-invsqrt error, which you do not care about.

---

## 5. Ridge-Centered Analysis

Below is an analysis treating ridge as a deliberate part of the preconditioner (not just a numerical band-aid) because it fundamentally changes what "good" means.

### 5.1 What Ridge Changes Mathematically

Let $B = G^T G \succeq 0$ with eigenpairs $(\lambda_i, v_i)$. If you use a ridge $\delta \ge 0$ and apply the exact damped inverse square root:
$$
Z_\delta := (B + \delta I)^{-1/2}
$$
then the matrix you actually induce on the Gram side is:
$$
S_\delta := Z_\delta^T B Z_\delta = (B+\delta I)^{-1/2} B (B+\delta I)^{-1/2}
$$
This simplifies to:
$$
S_\delta = I - \delta (B+\delta I)^{-1}
$$
So ridge does not just "help convergence". It changes the target from $I$ to a shrinked matrix $S_\delta \prec I$.

#### Eigenvalues of the Preconditioned Gram
In the eigenbasis of $B$:
$$
s_i(\delta) := \lambda_i(S_\delta) = \frac{\lambda_i}{\lambda_i + \delta} = \frac{1}{1 + \delta/\lambda_i}
$$
**Interpretation:**
* For large $\lambda_i$, $s_i(\delta) \approx 1$.
* For small $\lambda_i$, $s_i(\delta) \approx \lambda_i/\delta$ (strongly shrunk).
* Ridge is therefore a **damping / shrinkage** mechanism: it prevents aggressive amplification of small-eigenvalue directions.

### 5.2 Ridge Sets Hard Limits on Certificates

Your online certificate is based on $E := Z^T B Z - I$.
Even if you computed $Z = Z_\delta$ exactly, you would get:
$$
E_\delta = S_\delta - I = -\delta (B+\delta I)^{-1}
$$

**Spectral-Norm Floor (Unavoidable):**
$$
\|E_\delta\|_2 = \max_i \frac{\delta}{\lambda_i + \delta} = \frac{\delta}{\lambda_{\min} + \delta}
$$
If you choose ridge $\delta$, you are implicitly saying: *"I do not want (or cannot trust) whitening beyond this level."* The stopping criterion must be consistent with the ridge choice, or you will chase an impossible target.

**Frobenius-Norm Floor:**
$$
\|E_\delta\|_F^2 = \sum_i \left(\frac{\delta}{\lambda_i + \delta}\right)^2
$$
This shows why the certificate might plateau even when the iteration is perfectly stable.

### 5.3 Ridge Controls How Hard You Can "Hit" a Vector

When you apply the preconditioner to a vector $x$, each eigendirection of $B$ is scaled by:
$$
\text{gain}_i(\delta) = \frac{1}{\sqrt{\lambda_i + \delta}}
$$
So ridge imposes an absolute cap:
$$
\text{gain}_i(\delta) \le \frac{1}{\sqrt{\delta}}
$$
This is a very direct plug-and-play tuning knob for ML. If small eigenvalues are noisy (common for minibatch covariances), you often want to cap amplification to avoid exploding steps.

### 5.4 Two Competing "Goodness" Notions

**A) Whitening Quality (Make $S$ close to $I$)**
Ridge hurts this, because $S_\delta \prec I$. A useful bound sets the floor of the smallest singular value of $U$:
$$
\lambda_{\min}(S_\delta) = \frac{\lambda_{\min}}{\lambda_{\min}+\delta} \implies \sigma_{\min}(U)^2 = \lambda_{\min}(S_\delta)
$$

**B) Stability / Robustness (Finite Precision & Statistics)**
Ridge helps here in multiple ways:
* Substantially improves condition number: $\kappa(B+\delta I) = \frac{\lambda_{\max}+\delta}{\lambda_{\min}+\delta}$.
* Reduces dynamic range and polynomial overshoot due to noise or rounding.
* Caps the maximum gain at $1/\sqrt{\delta}$.

So ridge is a classic bias-variance (or accuracy-robustness) trade: less whitening, more safety.

### 5.5 Plug-and-Play Ridge Choices

**Rule 1: Target "do not amplify more than X"**
Pick a reference scale $\lambda_{\text{ref}}$ (e.g., $\lambda_{\max}$ or $\mathrm{tr}(B)/n$).
If you want $\max_i \text{gain}_i(\delta) \le \frac{c}{\sqrt{\lambda_{\text{ref}}}}$, choose:
$$
\delta \ge \frac{\lambda_{\text{ref}}}{c^2}
$$

**Rule 2: Cap the certificate floor below a tolerance**
If you want a stop test $\|E\|_2 \le \eta$, and use a trusted minimum eigenvalue $\lambda_{\text{trust}}$:
$$
\delta \le \frac{\eta}{1-\eta}\lambda_{\text{trust}}
$$
In many ML cases, $\lambda_{\min} \approx 0$ at minibatch scale, meaning you cannot target small $\eta$ if you want damping.

**Rule 3: Iteration conditioning target**
If you want $\kappa(B+\delta I) \le \kappa_{\text{alg}}$ for faster/stabler iterations:
$$
\delta \ge \frac{\lambda_{\max} - \kappa_{\text{alg}}\lambda_{\min}}{\kappa_{\text{alg}} - 1} \quad \implies \left(\text{if } \lambda_{\min} \approx 0\right) \quad \delta \gtrsim \frac{\lambda_{\max}}{\kappa_{\text{alg}} - 1}
$$

**Rule 4: Minimum retained energy**
To ensure $\lambda_{\min}(S_\delta) \ge s_{\min}$, choose:
$$
\delta \le \lambda_{\min}\left(\frac{1}{s_{\min}} - 1\right)
$$

### 5.6 Ridge Placement Matters

With Jacobi congruence scaling $D$, there are two different ridge placements:
1. **In original coordinates**: $B_\delta = B + \delta I \implies \widetilde B = D B D + \delta D^2$. This produces a ridge term that is not uniform after scaling.
2. **After scaling (uniform in scaled space)**: $\widetilde B_\delta = D B D + \delta I$.

Neither is "right" in general. This should be an explicit option in experiments.

### 5.7 Ridge Interactions with Finite Precision

* Widens the region where low-degree polynomial steps are contractive.
* Reduces bf16 quantization effects in subsequent multiplications (avoids producing extremely large values along tiny-eigen directions).
* Guarantees $Z^T B Z$ will never be perfectly close to $I$, requiring tracking of both the raw residual $\|Z^T B Z - I\|_F$ and a "damped objective" residual.

### 5.8 Suggestions for Experiments

Instead of sweeping $\delta$ blindly, sweep it through dimensionless ratios.
Pick a reference scale $\lambda_{\text{ref}}$ and define:
$$
\hat\delta = \frac{\delta}{\lambda_{\text{ref}}}
$$
Sweep $\hat\delta$ on a small log grid, like $\hat\delta \in \{0, 10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$.
For each $\hat\delta$, report:
* Time to reach a target preconditioning level.
* Observed plateau (the ridge-induced floor).

This gives you a scale-normalized "plug-and-play" knob across models.

### 5.9 Bottom Line

Ridge is not just a stabilization trick; it is a controllable bias that:
* Caps preconditioner gain: $\|Z_\delta\|_2 \le 1/\sqrt{\delta}$
* Sets an irreducible deviation from perfect whitening: $Z_\delta^T B Z_\delta - I = -\delta(B+\delta I)^{-1}$
* Improves conditioning for iterations: $\kappa(B+\delta I) = \frac{\lambda_{\max}+\delta}{\lambda_{\min}+\delta}$

If you identify a primary specification (max amplification cap, target eigenvalue band, or target time-to-reach a ridge-aware plateau), this can be turned into an automatic "ridge policy" computed from existing estimates like $\lambda_{\max}$ or $\mathrm{tr}(B)$.
