# Applied inverse-root roadmap: from inverse modulus to general G P^{-s/r}

## 0) Goal and sequencing (do not skip this)

We want fast applied transforms of the form
$$
G P^{-s/r},
$$
where $P \succeq 0$, $r>0$, $s>0$, using GEMM-heavy polynomial policies.

Sequencing:
1) First deliverable: applied inverse square root ($r=2$, $s=1$) for whitening/preconditioning.
2) Next: small integer $r$ (especially $r=4$).
3) Only pursue general real $r$ if integer-$r$ cases show clear practical value.

## 1) Applied objective and certificates

For inverse square root (whitening), we aim for $Z \approx P^{-1/2}$ such that
$$
S := Z^T P Z \approx I,
$$
and the project metrics are $\delta_F=\|S-I\|_F$ and $\rho_2=\|S-I\|_2$.

For more general applied roots, define the certificate by the downstream use case (do not optimize $\|Z-P^{-1/r}\|$ unless it improves the applied metric).

## 2) Coupled iteration family (bridge from inverse modulus to general applied roots)

A useful template is the coupled polynomial iteration that drives an auxiliary sequence $P_t \to I$ while updating the applied transform:
$$
G_{t+1} = G_t \left(a_{t+1}I + b_{t+1}P_t + c_{t+1}P_t^2\right)^s,
$$
$$
P_{t+1} = \left(a_{t+1}I + b_{t+1}P_t + c_{t+1}P_t^2\right)^r P_t.
$$
This connects the inverse-modulus viewpoint to later $G P^{-s/r}$ deliverables. 

## 3) Two-phase reverse-engineered policy (reuse the polar blueprint)

### Phase 2 (local finish)
Once the certificate is near identity, use 1-2 locally designed steps with exact-arithmetic interval analysis and bf16 calibrated envelope, plus guards.

### Phase 1 (global compression)
Use bf16-safe global steps + scaling/preconditioning to enter Phase 2 band quickly and safely.

## 4) Scaling/preconditioning sweep

Test small-side scalings and preprocessors, including Frobenius-inner-product based options (useful to compress spectrum into a safe interval without expensive estimates). 

Include ridge option for near-PSD/noisy inputs:
1) symmetrize $P \leftarrow 0.5(P+P^T)$
2) ridge $P_\lambda = P + \lambda I$

If ridge is used, treat the objective as changed and report the induced floor in $\|S-I\|$.

## 5) Verification and benchmarking

### Exact arithmetic verification
For each polynomial/schedule:
- scalar interval contraction checks for the relevant map
- composition behavior across multiple steps

### Deployment verification
- end-to-end bf16 kernel tests (same code path as deployment)
- calibrated envelopes, plateau behavior, and guard triggers
- evaluate progress-per-second at target tiers

## 6) Deliverables

### Code
- `policy_apply_inv_sqrt.py` (primary deliverable)
- `policy_apply_inv_rth_root.py` (for r=4 next)
- `design_local_steps.py` (Phase 2 local designer)
- `bench_applied_roots.py` (unified harness)

### Reports
- `report_inv_sqrt_policies.md` (compare direct vs preconditioner-first vs hybrid)
- `report_integer_r_roots.md` (start with r=4)
- `report_scaling_and_ridge.md`

## 7) Ship rule

Ship the policy family that minimizes wall time to a downstream-useful certificate under bf16-safe guards, and only expand to more general $r$ when the integer-$r$ results demonstrate clear practical benefit.