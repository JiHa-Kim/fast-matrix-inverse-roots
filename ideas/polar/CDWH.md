# Cholesky Dynamically Weighted Halley Iterations for Polar Decomposition

The key idea is to optimize for the thing we actually care about after applying the transform, not for the abstract error in an inverse square root. For a polar iterate $X$, what matters is whether $X$ is close to having all singular values at $1$, which is equivalent to making

$$
S := X^T X
$$

well conditioned and close to the identity in a multiplicative sense.

That is why the natural success metric is

$$
\kappa(X) = \sqrt{\kappa(S)},
\qquad
\eta(S) = \tfrac12 \log \kappa(S),
$$

rather than an additive matrix norm alone. Additive errors can still be useful for debugging, but they are not the fairest global objective.

The second idea is to work on the small side whenever possible. For a tall matrix $X \in \mathbb{R}^{m \times n}$ with $m \gg n$, the expensive object is the tall pass, while the informative certificate lives in the small $n \times n$ matrix $S = X^T X$. This makes it natural to do the delicate numerical work on the small side.

The DWH/QDWH baseline is valuable because it gives a principled, correctness-first reference method. Its update can be written in the safer form

$$
X_{k+1}
= \frac{b}{c} X_k
+ \left(a - \frac{b}{c}\right) X_k (I + c X_k^T X_k)^{-1},
$$

which is much better behaved numerically than a more naive rearrangement. The associated small-side propagation

$$
Q_k = (aI + bS_k)(I + cS_k)^{-1}, \qquad
S_{k+1} = Q_k S_k Q_k
$$

lets us certify progress without redoing the whole argument from scratch.

The third idea is honesty about precision. The small Gram and Cholesky factorization should be done in fp64, even if the surrounding tall operations are in float32 or bf16. That keeps the certificate trustworthy. The solves should be applied in row chunks so memory stays under control. If Cholesky fails, use tiny jitter and record it; do not quietly pretend the path was stable.

The fourth idea is to avoid expensive or fragile certification machinery when a simpler conservative bound is available. For small $n$, exact $\operatorname{eigvalsh}$ is fine. For larger $n$, a trace/logdet bound is attractive because it only needs stable small-side quantities:

$$
a := \frac{\operatorname{tr}(S)}{n}, \qquad
g := \exp\!\left(\frac{1}{n}\log\det S\right), \qquad
r := \frac{a}{g}.
$$

From this one gets an upper bound on $\kappa(S)$, and therefore on $\kappa(X)$. In practice this is much more useful than old very loose certificates, especially near the target region.

So the overall picture is simple:

1. Use an honest DWH-style baseline to define what "correct and certified" means.
2. Judge methods by certified conditioning of the applied transform, not by internal approximation error.
3. Push hard work to the small side and keep that part numerically strong.
4. Compare future aggressive rational or Zolotarev-inspired schedules against this baseline on progress per second and failure behavior.

This gives a clean foundation for the eventual fast Muon path: the fast method does not need to look like DWH, but it does need to beat DWH on the metric that actually matters.
