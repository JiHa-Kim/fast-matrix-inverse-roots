# Two-step damped polar (Gram-side) for ML: robust, fixed-cost, fail-closed

This document describes a production-oriented approximation to the polar factor
$$
Q=\operatorname{polar}(G)=G(G^T G)^{-1/2},
$$
designed for ML training where the top priority is **preventing catastrophic numerical failures** (NaNs, Cholesky breakdowns). The method uses **two fixed rational steps** on the small side, with **mandatory damping** and **fail-closed restarts**.

---

## 0. Summary (what you implement)

Given $G\in\mathbb{R}^{m\times n}$ with $m\ge n$:

1. Form $B=G^T G$ (fp32 accumulate), symmetrize.
2. Dampen and normalize the Gram:
   - $\mu=\frac{1}{n}\operatorname{tr}(B)$
   - $\lambda=\tau\mu$ with $\tau=2^{-8}$
   - $\tilde B=\frac{B+\lambda I}{\mu+\lambda}$ so $\frac{1}{n}\operatorname{tr}(\tilde B)=1$ and $\tilde B\succ 0$.
3. Run exactly **two** updates $Z\leftarrow Z(S+cI)(cS+I)^{-1}$ with trace-centering each step.
4. Output $\widehat Q = G Z_{\text{full}}$ where $Z_{\text{full}} = Z/\sqrt{\mu+\lambda}$.
5. If any Cholesky fails, increase $\tau$ by powers of two and restart the whole two-step.

Recommended defaults:
- $\tau=2^{-8}=0.00390625$
- For $n\approx 256$: $(c_1,c_2)=(10.5,3.2)$
- Universal (up to $n=1024$): $(c_1,c_2)=(13.0,3.26)$

---

## 1. Definitions and what we approximate

Assume $m\ge n$. Define
$inline$
$B := G^T G\in\mathbb{R}^{n\times n},\quad B\succeq 0.$
$inline$

Exact polar factor:
$$
Q = \operatorname{polar}(G)=G\,B^{-1/2}.
$$

We compute an approximate inverse square root $Z\approx (B+\lambda I)^{-1/2}$ and output
$$
\widehat Q := GZ.
$$

The Gram of the output (the "certificate matrix") is
$$
S := \widehat Q^T\widehat Q = Z^T B Z \succ 0.
$$
If $Z=B^{-1/2}$, then $S=I$ (orthonormal columns).

---

## 2. Quality metric (what $\kappa_\star$ means)

Primary spec:
$$
\kappa(S):=\frac{\lambda_{\max}(S)}{\lambda_{\min}(S)}.
$$

Log-width:
$$
\eta(S)=\tfrac12\log\kappa(S).
$$

For $\kappa_\star=1.5$:
- $\eta_\star=\tfrac12\log(1.5)\approx 0.2027326$.
- Equivalent spectral residual threshold:
  $$
  \kappa(S)\le \kappa_\star
  \iff
  \|S-I\|_2 \le \frac{\kappa_\star-1}{\kappa_\star+1}=0.2.
  $$

Geometric interpretation: $\widehat Q$ is a near-isometry:
$$
\sqrt{\lambda_{\min}(S)}\|x\|
\le \|\widehat Qx\|
\le \sqrt{\lambda_{\max}(S)}\|x\|.
$$
If $\kappa(S)\le 1.5$, then the norm distortion ratio is at most $\sqrt{1.5}\approx 1.2247$.

**Note:** in the robust fixed-cost policy below, we do not certify $\kappa(S)$ online; we just run 2 steps deterministically and use damping + restart to eliminate catastrophic failures.

---

## 3. Conditioning relationships ($G$ vs $B$)

Key identity:
$$
\kappa(B)=\kappa(G)^2.
$$

This is why the small-side Gram can be far more ill-conditioned than $G$. Damping is used to cap effective conditioning and improve PD robustness.

---

## 4. Damping and normalization (the main safety knob)

We damp the Gram:
$$
\mu := \frac{1}{n}\operatorname{tr}(B),\qquad
\lambda := \tau \mu,
$$
and normalize:
$$
\tilde B := \frac{B+\lambda I}{\mu+\lambda}
= \frac{B+\tau\mu I}{(1+\tau)\mu}.
$$

Then $\frac{1}{n}\operatorname{tr}(\tilde B)=1$ and $\tilde B$ is strongly SPD.

Conservative worst-case cap:
$$
\kappa(\tilde B)\le \frac{n+\tau}{\tau}\approx \frac{n}{\tau}.
$$

With $\tau=2^{-8}$:
$$
\kappa_{\text{init,cap}}\approx \frac{n}{\tau}=n\cdot 2^8 = 256n.
$$

Examples:
- $n=128:\ \kappa_{\text{init,cap}}\approx 2^{15}=32768$
- $n=256:\ \kappa_{\text{init,cap}}\approx 2^{16}=65536$
- $n=512:\ \kappa_{\text{init,cap}}\approx 2^{17}=131072$
- $n=1024:\ \kappa_{\text{init,cap}}\approx 2^{18}=262144$

This cap is the design "coverage" for the 2-step contraction.

---

## 5. Two-step rational update on the Gram side

Maintain $Z\in\mathbb{R}^{n\times n}$. Each step:

1. Form the current certificate
   $$
   S := Z^T \tilde B Z.
   $$
2. Apply a right update using
   $$
   q_c(x)=\frac{x+c}{cx+1},\quad c>0,
   $$
   so
   $$
   Z \leftarrow Z\,q_c(S)
   = Z\,(S+cI)\,(cS+I)^{-1}.
   $$

We run exactly two steps with $(c_1,c_2)$.

---

## 6. Centering (prevent scale drift safely)

Scaling $Z$ by a scalar does not change $\kappa(S)$ but improves stability.

### Default: trace-centering (cannot fail)

Given $S$:
$$
\alpha := \sqrt{\frac{n}{\operatorname{tr}(S)}},\qquad
Z \leftarrow \alpha Z.
$$

Trace-centering is recommended for the "never crash" path since it has no Cholesky dependency.

### Optional: logdet-centering (if desired)

If you want the theoretically elegant centering:
$$
c_{\det}(S)=\frac{1}{n}\log\det(S),\qquad
Z \leftarrow e^{-c_{\det}(S)/2}Z.
$$

To avoid "need Cholesky before scaling", pre-scale by $s=\operatorname{tr}(S)/n$:
$$
\tilde S=\frac{S}{s},\qquad
\log\det(S)=\log\det(\tilde S)+n\log s.
$$

In the robust path, trace-centering is typically enough.

---

## 7. Recommended parameters

### Damping
Default:
$inline$
$\tau=2^{-8}=0.00390625.$
$inline$

### Rational step parameters
Two choices depending on how wide an $n$ range you want:

- **Tuned for $n\approx 256$** (cap $\approx 65536$):
  $inline$
  $(c_1,c_2)=(10.5,3.2).$
  $inline$

- **Universal up to $n=1024$** (cap $\approx 262144$):
  $inline$
  $(c_1,c_2)=(13.0,3.26).$
  $inline$

Both are intentionally moderate to reduce PD failure risk in $M=I+cS$.

---

## 8. Fail-closed restart policy (eliminate catastrophes)

Never limp forward after a PD failure. Restart with more damping.

If any Cholesky fails (typically on $M=I+cS$):
- Increase $\tau \leftarrow 4\tau$ (power-of-two jumps),
- Rebuild $\tilde B$,
- Restart both steps from $Z=I$.

Suggested ladder:
$$
2^{-8}\to 2^{-6}\to 2^{-4}.
$$

If all levels fail, fail closed to a safe fallback (e.g. $Z=\mu^{-1/2}I$ and $\widehat Q=G/\sqrt{\mu}$) to guarantee "no crash".

---

## 9. Full algorithm (pseudocode)

**Input:** $G\in\mathbb{R}^{m\times n}$, $m\ge n$, $\tau=2^{-8}$, $(c_1,c_2)$.

1. Compute $B=\mathrm{sym}(G^T G)$ (bf16 inputs with fp32 accumulation).
2. For $\tau$ in $\{2^{-8},2^{-6},2^{-4}\}$:
   1. $\mu=\operatorname{tr}(B)/n$; if invalid, return fallback.
   2. $\lambda=\tau\mu$, $\tilde B=(B+\lambda I)/(\mu+\lambda)$.
   3. Set $Z=I$.
   4. For $c$ in $\{c_1,c_2\}$:
      1. $S=\mathrm{sym}(Z^T\tilde B Z)$.
      2. Trace-center: $Z\leftarrow \sqrt{n/\operatorname{tr}(S)}\,Z$.
      3. (Optional) recompute $S$.
      4. $M=\mathrm{sym}(I+cS)$.
      5. Cholesky $M=LL^T$. If fail: restart loop with larger $\tau$.
      6. Compute $X=(cS+I)^{-1}(S+cI)$ via triangular solves.
      7. Update $Z\leftarrow ZX$.
   5. $Z_{\text{full}} = Z/\sqrt{\mu+\lambda}$.
   6. Output $\widehat Q = GZ_{\text{full}}$ and stop.
3. If all $\tau$ fail: return fallback.

---

## 10. Numerics checklist

- Use fp32 accumulation for $B=G^T G$.
- Always symmetrize $B$, $S$, and $M$:
  $inline$
  $X\leftarrow \tfrac12(X+X^T).$
  $inline$
- Keep $B,S,M,Z$ in fp32 (fp64 only for debugging).
- Use trace-centering to prevent drift (no extra Cholesky).
- On any NaN/Inf or Cholesky failure: restart with larger $\tau$.
- Avoid large $c$ values; moderate $c$ improves PD robustness of $M=I+cS$.

---

## 11. Notes on $m<n$ (not covered above)

If $m<n$, swap sides and work with the smaller dimension (left Gram). The same damping + 2-step update philosophy applies; implement once and dispatch based on the smaller side.

---