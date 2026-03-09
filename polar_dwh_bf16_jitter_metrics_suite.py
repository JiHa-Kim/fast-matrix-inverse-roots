#!/usr/bin/env python3
# polar_dwh_bf16_predcap_suite.py
#
# BF16-friendly polar / orthogonalization preconditioner via direct x g(x^2)
# (DWH/QDWH-style), with BF16-aware coefficient selection:
#
# Instead of "cap -> shrink tau -> extra steps", we do:
#   - choose coefficients (a,b,c) that already satisfy caps (a_cap,b_cap,c_cap)
#   - by searching over ell in [max(ell_est, ell0), 1] and minimizing predicted kappa
#   - keep tau = 1/safety (nearly 1), so each step is "full strength" but bounded
#
# Core invariants:
#   - X stored in bf16, S = X^T X formed in fp32
#   - trace-centering each iter so mean eigenvalue ~ 1
#   - Cholesky only on M = I + c S
#   - "jitter" (+delta I) is applied only to M and only if Cholesky fails
#
# Certification modes:
#   - eig (exact, O(n^3)): eigvalsh on S to get lam_min/lam_max
#   - frob (O(n^2)): uses ||S-I||_F bound to get lam_min_lb, lam_max_ub (conservative)
#   - auto: eig for n <= eig_threshold, else frob
#
# Suite mode includes "Kimi K2/GLM5-ish" shapes (some may OOM on small GPUs; script skips).

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import time
from typing import List, Tuple

import numpy as np
import torch

Tensor = torch.Tensor


# ----------------------------- utilities -----------------------------------


def symmetrize(A: Tensor) -> Tensor:
    return 0.5 * (A + A.T)


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = int(round(p * (len(ys) - 1)))
    i = max(0, min(len(ys) - 1, i))
    return float(ys[i])


def cuda_time_ms(fn):
    if not torch.cuda.is_available():
        return 0.0, fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out


def chol_on_M_with_jitter(
    M: Tensor,
    jitter_rel: float,
    max_tries: int = 10,
    allow_fp64_fallback: bool = True,
    fp64_eps_rel: float = 1e-12,
) -> Tuple[Tensor, float]:
    """
    Cholesky for M = I + cS.
    - Try fp32 cholesky_ex with delta = jitter_rel * tr(M)/n, doubling delta on failure.
    - If still fails and allow_fp64_fallback: compute minimal shift in fp64 using eigvalsh:
        shift = max(0, -lam_min(M)) + eps
      then do fp64 cholesky and return fp64 factor (cast is handled by caller through cholesky_solve dtype).
    Returns (L, used_shift_or_jitter). If fp32 succeeds, returned L is fp32 and used_shift is the jitter delta.
    If fp64 fallback triggers, returned L is fp64 and used_shift is the fp64 shift.
    """
    M = symmetrize(M)
    if not torch.isfinite(M).all():
        raise RuntimeError("Non-finite entries in M before Cholesky")

    n = M.shape[0]
    I32 = torch.eye(n, device=M.device, dtype=M.dtype)

    tr = torch.trace(M).abs()
    base = float((jitter_rel * (tr / n)).item()) if jitter_rel > 0.0 else 0.0
    delta = base

    # fp32 attempts
    for _ in range(max_tries):
        Mt = M if delta == 0.0 else (M + delta * I32)
        L, info = torch.linalg.cholesky_ex(Mt)
        if int(info.item()) == 0:
            return L, float(delta)
        if jitter_rel <= 0.0:
            break
        delta = delta * 2.0 if delta > 0.0 else base

    if not allow_fp64_fallback:
        raise RuntimeError(
            "Cholesky failed on M (fp32+jitter) and fp64 fallback disabled"
        )

    # fp64 minimal shift
    Md = symmetrize(M.double())
    evals = torch.linalg.eigvalsh(Md)
    lam_min = float(evals[0].item())
    trd = float(torch.trace(Md).abs().item())
    eps = fp64_eps_rel * (trd / n if n > 0 else 1.0)
    shift = max(0.0, -lam_min + eps)
    I64 = torch.eye(n, device=M.device, dtype=torch.float64)
    Ld = torch.linalg.cholesky(Md + shift * I64)
    return Ld, float(shift)


# ----------------------------- DWH coefficients ----------------------------


def dwh_coeffs(ell: float) -> Tuple[float, float, float]:
    """
    Dynamically Weighted Halley coefficients (a,b,c) from ell in (0,1].
    """
    ell = float(ell)
    ell = min(max(ell, 1e-12), 1.0)
    ell2 = ell * ell

    d = (4.0 * (1.0 - ell2) / (ell2 * ell2)) ** (1.0 / 3.0)
    h = math.sqrt(1.0 + d) + 0.5 * math.sqrt(
        8.0 - 4.0 * d + 8.0 * (2.0 - ell2) / (ell2 * math.sqrt(1.0 + d))
    )
    a = h
    b = (a - 1.0) * (a - 1.0) / 4.0
    c = a + b - 1.0
    return a, b, c


def phi_map(a: float, b: float, c: float, x: float) -> float:
    """
    Scalar eigenvalue map for S eigenvalues under U(S) = (aI+bS)(I+cS)^(-1):
      x_+ = x * r(x)^2, r(x) = (a + b x)/(1 + c x)
    """
    r = (a + b * x) / (1.0 + c * x)
    return x * (r * r)


def pick_capped_coeffs_by_pred(
    lam_min: float,
    lam_max: float,
    ell_est: float,
    ell0: float,
    a_cap: float,
    b_cap: float,
    c_cap: float,
    grid: int = 48,
) -> Tuple[float, float, float, float]:
    """
    Choose ell in [max(ell_est,ell0), 1] to minimize predicted kappa after one step,
    subject to coefficient caps.

    Returns (a,b,c, ell_chosen).
    If nothing fits caps, returns mild Halley-like step (a,b,c)=(3,1,3).
    """
    lo = max(float(ell_est), float(ell0))
    lo = min(max(lo, 1e-12), 1.0)

    if lo >= 1.0:
        return 1.0, 0.0, 0.0, 1.0

    # Geometric interpolation from lo to 1, biased to lower end (more resolution where it matters).
    ts = np.linspace(0.0, 1.0, int(grid))
    ells = lo * (1.0 / lo) ** ts

    best = None
    best_k = None
    best_ell = None

    lam_min = max(float(lam_min), 0.0)
    lam_max = max(float(lam_max), lam_min)

    for ell in ells:
        a, b, c = dwh_coeffs(float(ell))
        if a > a_cap or b > b_cap or c > c_cap:
            continue
        y_min = phi_map(a, b, c, lam_min)
        y_max = phi_map(a, b, c, lam_max)
        if not (math.isfinite(y_min) and math.isfinite(y_max)):
            continue
        if y_min <= 0.0 or y_max <= 0.0:
            continue
        k = y_max / y_min
        if best_k is None or k < best_k:
            best_k = k
            best = (a, b, c)
            best_ell = float(ell)

    if best is None:
        # Mild, bounded, extremely stable step:
        # r(x) = (3 + x)/(1 + 3x) corresponds to (a,b,c)=(3,1,3)
        return 3.0, 1.0, 3.0, lo

    return float(best[0]), float(best[1]), float(best[2]), float(best_ell)


# ----------------------------- synthetic matrices --------------------------


def make_matrix_from_singulars(
    m: int,
    singulars: Tensor,
    seed: int,
    device: str,
    storage_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    n = int(singulars.numel())
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    U, _ = torch.linalg.qr(
        torch.randn(m, n, generator=gen, dtype=torch.float32), mode="reduced"
    )
    V, _ = torch.linalg.qr(
        torch.randn(n, n, generator=gen, dtype=torch.float32), mode="reduced"
    )
    G = (U * singulars.float()) @ V.T
    return G.to(device=device, dtype=storage_dtype)


def make_spectrum_bank(
    n: int, kappa_G: float, bank_size: int, seed: int
) -> List[Tensor]:
    sig_max = 1.0
    sig_min = 1.0 / float(kappa_G)
    out: List[Tensor] = []

    out.append(
        torch.logspace(0.0, math.log10(sig_min), n, base=10.0, dtype=torch.float32)
    )

    t = torch.linspace(0.0, 1.0, n)
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        logs = math.log(sig_max) + (math.log(sig_min) - math.log(sig_max)) * (t**p)
        out.append(torch.exp(logs))
        logs = math.log(sig_max) + (math.log(sig_min) - math.log(sig_max)) * (
            1.0 - (1.0 - t) ** p
        )
        out.append(torch.exp(logs))

    for frac in [1 / n, 2 / n, 4 / n, 8 / n, 0.1, 0.25, 0.5, 0.75, 0.9]:
        r = max(1, min(n - 1, int(round(frac * n))))
        s = torch.full((n,), sig_min, dtype=torch.float32)
        s[:r] = sig_max
        out.append(s)

    rng = random.Random(seed)
    while len(out) < bank_size:
        u = sorted([rng.random() for _ in range(n)], reverse=True)
        logs = torch.tensor([math.log(sig_min) * x for x in u], dtype=torch.float32)
        s = torch.exp(logs)
        s[0] = sig_max
        s[-1] = sig_min
        out.append(s)

    return out[:bank_size]


# ----------------------------- certification/bounds ------------------------


@dataclasses.dataclass
class CertInfo:
    kappa_cert: float
    kappa_O: float
    # endpoints for prediction / ell selection
    lam_min_for_ell: float
    lam_min_for_pred: float
    lam_max_for_pred: float
    # for logging/debug
    mode: str


def cert_and_bounds_eig(
    S: Tensor,
    psd_clip: bool,
    cert_floor_rel: float,
    psd_repair: bool,
) -> Tuple[Tensor, CertInfo]:
    """
    Use eigvalsh to get lam_min/lam_max.
    Optionally repair PSD if lam_min_raw < 0 by shifting S <- S + shift I.
    Returns (S_used, CertInfo).
    """
    S = symmetrize(S)
    n = S.shape[0]
    Sd = S.double()
    evals = torch.linalg.eigvalsh(Sd)
    lam_min_raw = float(evals[0].item())
    lam_max_raw = float(evals[-1].item())

    shift = 0.0
    if psd_repair and psd_clip and lam_min_raw < 0.0:
        eps = cert_floor_rel * max(lam_max_raw, 1.0)
        shift = float(-lam_min_raw + eps)
        Id_mat = torch.eye(n, device=S.device, dtype=S.dtype)
        S = S + shift * Id_mat
        lam_min_raw += shift
        lam_max_raw += shift

    lam_max = max(lam_max_raw, 0.0)
    lam_min_pos = max(lam_min_raw, 0.0) if psd_clip else lam_min_raw

    lam_min_safe = max(lam_min_pos, cert_floor_rel * lam_max if lam_max > 0.0 else 0.0)
    kappa_cert = 1.0 if lam_max == 0.0 else (lam_max / max(lam_min_safe, 1e-300))
    kappa_O = math.sqrt(kappa_cert)

    # For ell: use clipped-but-unfloored lower bound (do not use cert_floor_rel)
    lam_min_for_ell = max(lam_min_pos, 0.0)

    info = CertInfo(
        kappa_cert=float(kappa_cert),
        kappa_O=float(kappa_O),
        lam_min_for_ell=float(lam_min_for_ell),
        lam_min_for_pred=float(max(lam_min_pos, 0.0)),
        lam_max_for_pred=float(max(lam_max, 0.0)),
        mode="eig",
    )
    return S, info


def cert_and_bounds_frob(
    S: Tensor,
    cert_floor_rel: float,
) -> Tuple[Tensor, CertInfo]:
    """
    Use Frobenius bound:
      ||E||_2 <= ||E||_F where E = S - I.
    Then lam(S) subset of [1 - rho, 1 + rho] with rho = ||E||_2 <= ||E||_F.

    This is conservative but O(n^2). Good for large n.

    Returns (S_used, CertInfo). S is unchanged.
    """
    S = symmetrize(S)
    n = S.shape[0]
    Id_matd_mat = torch.eye(n, device=S.device, dtype=S.dtype)
    E = S - Id_matd_mat
    rhoF = float(torch.linalg.norm(E, ord="fro").item())

    lam_min_lb = max(1.0 - rhoF, 0.0)
    lam_max_ub = 1.0 + rhoF

    lam_min_safe = max(lam_min_lb, cert_floor_rel * lam_max_ub)
    kappa_cert = lam_max_ub / max(lam_min_safe, 1e-300)
    kappa_O = math.sqrt(kappa_cert)

    info = CertInfo(
        kappa_cert=float(kappa_cert),
        kappa_O=float(kappa_O),
        lam_min_for_ell=float(lam_min_lb),
        lam_min_for_pred=float(lam_min_lb),
        lam_max_for_pred=float(lam_max_ub),
        mode="frob",
    )
    return S, info


# ----------------------------- core iteration ------------------------------


@dataclasses.dataclass
class RunSummary:
    success: bool
    final_kappa_O: float
    steps: int
    ms_gram: float
    ms_solve: float
    ms_upd: float
    ms_total: float


@torch.no_grad()
def run_polar_dwh_predcap(
    G_storage: Tensor,
    target_kappa_O: float,
    max_steps: int,
    ell0: float,
    eps_scale: float,
    safety: float,
    jitter_rel: float,
    tf32: bool,
    psd_clip: bool,
    cert_floor_rel: float,
    cert_mode: str,
    eig_threshold: int,
    psd_repair: bool,
    a_cap: float,
    b_cap: float,
    c_cap: float,
    ell_grid: int,
    allow_fp64_fallback: bool,
) -> RunSummary:
    device = G_storage.device
    m, n = G_storage.shape
    Id_matd_matd_matd_mat = torch.eye(n, device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high")

    Gf = G_storage.float()

    # Initial scaling to keep sigma_max <= 1-ish.
    fro = torch.linalg.norm(Gf, ord="fro")
    denom = float(safety) * fro + float(eps_scale)
    X = (Gf / denom).to(torch.bfloat16)

    ms_gram_sum = 0.0
    ms_solve_sum = 0.0
    ms_upd_sum = 0.0

    tau = 1.0 / float(safety)

    for it in range(1, max_steps + 1):
        # Form S in fp32.
        ms_gram, S = cuda_time_ms(lambda: symmetrize(X.float().T @ X.float()))
        ms_gram_sum += ms_gram

        # Trace centering (keeps S near I).
        mu = torch.trace(S) / n
        mu_f = float(mu.item())
        if not math.isfinite(mu_f) or mu_f <= 0.0:
            return RunSummary(
                False,
                float("inf"),
                it,
                ms_gram_sum,
                ms_solve_sum,
                ms_upd_sum,
                float("inf"),
            )
        X = (X.float() / math.sqrt(mu_f)).to(torch.bfloat16)
        S = S / mu

        # Cert + bounds.
        mode = cert_mode
        if mode == "auto":
            mode = "eig" if n <= eig_threshold else "frob"

        if mode == "eig":
            S, cert = cert_and_bounds_eig(
                S,
                psd_clip=psd_clip,
                cert_floor_rel=cert_floor_rel,
                psd_repair=psd_repair,
            )
        elif mode == "frob":
            S, cert = cert_and_bounds_frob(S, cert_floor_rel=cert_floor_rel)
        else:
            raise ValueError(f"Unknown cert_mode: {cert_mode}")

        if cert.kappa_O <= target_kappa_O:
            ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum
            return RunSummary(
                True,
                float(cert.kappa_O),
                it,
                ms_gram_sum,
                ms_solve_sum,
                ms_upd_sum,
                ms_total,
            )

        # Estimate ell from lower bound on lambda_min (unfloored for ell).
        ell_est = math.sqrt(max(cert.lam_min_for_ell, 0.0))

        # Choose capped coefficients by predicted kappa contraction (no cap-driven tau shrink).
        a, b, c, ell_chosen = pick_capped_coeffs_by_pred(
            lam_min=cert.lam_min_for_pred,
            lam_max=cert.lam_max_for_pred,
            ell_est=ell_est,
            ell0=ell0,
            a_cap=a_cap,
            b_cap=b_cap,
            c_cap=c_cap,
            grid=ell_grid,
        )

        # Solve U = (aI + bS)(I + cS)^(-1) using Cholesky on M = I + cS.
        def solve():
            M = Id_matd_matd_matd_mat + float(c) * S
            RHS = float(a) * Id_matd_matd_matd_mat + float(b) * S
            L, used = chol_on_M_with_jitter(
                M,
                jitter_rel=jitter_rel,
                max_tries=10,
                allow_fp64_fallback=allow_fp64_fallback,
                fp64_eps_rel=1e-12,
            )
            # If L is fp64 (fallback), solve in fp64 then cast back.
            if L.dtype == torch.float64:
                U64 = torch.cholesky_solve(RHS.double(), L)
                return U64.float()
            return torch.cholesky_solve(RHS, L)

        ms_solve, U = cuda_time_ms(solve)
        ms_solve_sum += ms_solve

        # Fixed damping only (tau = 1/safety), not cap-driven.
        U = (1.0 - tau) * Id_matd_matd_matd_mat + tau * U

        ms_upd, X = cuda_time_ms(lambda: (X.float() @ U).to(torch.bfloat16))
        ms_upd_sum += ms_upd

    # If we used all steps, evaluate final cert one last time for reporting.
    ms_gram, S = cuda_time_ms(lambda: symmetrize(X.float().T @ X.float()))
    ms_gram_sum += ms_gram
    mu = torch.trace(S) / n
    mu_f = float(mu.item())
    if math.isfinite(mu_f) and mu_f > 0.0:
        S = S / mu
        mode = cert_mode
        if mode == "auto":
            mode = "eig" if n <= eig_threshold else "frob"
        if mode == "eig":
            _, cert = cert_and_bounds_eig(
                S,
                psd_clip=psd_clip,
                cert_floor_rel=cert_floor_rel,
                psd_repair=psd_repair,
            )
        else:
            _, cert = cert_and_bounds_frob(S, cert_floor_rel=cert_floor_rel)
        final_kO = float(cert.kappa_O)
    else:
        final_kO = float("inf")

    ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum
    return RunSummary(
        final_kO <= target_kappa_O,
        final_kO,
        max_steps,
        ms_gram_sum,
        ms_solve_sum,
        ms_upd_sum,
        ms_total,
    )


# ----------------------------- suite shapes --------------------------------


def suite_shapes_kimi_glm5() -> List[Tuple[int, int]]:
    # Some of these will OOM on small GPUs; suite catches OOM and continues.
    return [
        (2048, 256),
        (4096, 256),
        (8192, 256),
        (8192, 1024),
        (16384, 1024),
        (8192, 2048),
        (16384, 2048),
        (28672, 4096),
        (28672, 7168),
        (32768, 8192),
    ]


# ----------------------------- CLI -----------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["demo", "bank", "suite"], default="suite")

    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--kappa_G", type=float, default=1e7)
    ap.add_argument("--target_kappa_O", type=float, default=1.22474)
    ap.add_argument("--max_steps", type=int, default=5)

    # Numerical knobs
    ap.add_argument("--ell0", type=float, default=1e-3)
    ap.add_argument("--eps_scale", type=float, default=1e-2)
    ap.add_argument("--safety", type=float, default=1.01)
    ap.add_argument("--jitter_rel", type=float, default=1e-10)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--psd_clip", action="store_true", default=True)
    ap.add_argument("--no_psd_clip", dest="psd_clip", action="store_false")
    ap.add_argument("--cert_floor_rel", type=float, default=1e-12)

    # Cert mode and PSD repair
    ap.add_argument("--cert_mode", choices=["auto", "eig", "frob"], default="auto")
    ap.add_argument("--eig_threshold", type=int, default=2048)
    ap.add_argument("--psd_repair", action="store_true", default=True)
    ap.add_argument("--no_psd_repair", dest="psd_repair", action="store_false")

    # Caps and ell-search grid
    ap.add_argument("--a_cap", type=float, default=96.0)
    ap.add_argument("--b_cap", type=float, default=8192.0)
    ap.add_argument("--c_cap", type=float, default=8192.0)
    ap.add_argument("--ell_grid", type=int, default=48)

    # Fallback
    ap.add_argument(
        "--no_fp64_fallback",
        dest="allow_fp64_fallback",
        action="store_false",
        default=True,
    )

    # Test bank / suite
    ap.add_argument("--bank_size", type=int, default=12)
    ap.add_argument("--suite_cases", type=int, default=6)
    ap.add_argument("--suite_shapes", choices=["kimi_glm5"], default="kimi_glm5")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print(
        f"device={args.device}  mode={args.mode}  kappa_G<={args.kappa_G:.3g}  "
        f"target_kappa(O)<={args.target_kappa_O:.6g}"
    )
    print(
        "knobs: "
        f"ell0={args.ell0:g} eps_scale={args.eps_scale:g} safety={args.safety:g} jitter_rel={args.jitter_rel:g} "
        f"tf32={args.tf32} psd_clip={args.psd_clip} cert_floor_rel={args.cert_floor_rel:g} "
        f"cert_mode={args.cert_mode} psd_repair={args.psd_repair} "
        f"a_cap={args.a_cap:g} b_cap={args.b_cap:g} c_cap={args.c_cap:g} ell_grid={args.ell_grid} "
        f"max_steps={args.max_steps}"
    )

    def run_one(m: int, n: int, seed: int) -> RunSummary:
        spectra = make_spectrum_bank(n, args.kappa_G, bank_size=1, seed=seed + n)
        G = make_matrix_from_singulars(m, spectra[0], seed=seed, device=args.device)
        return run_polar_dwh_predcap(
            G_storage=G,
            target_kappa_O=args.target_kappa_O,
            max_steps=args.max_steps,
            ell0=args.ell0,
            eps_scale=args.eps_scale,
            safety=args.safety,
            jitter_rel=args.jitter_rel,
            tf32=args.tf32,
            psd_clip=args.psd_clip,
            cert_floor_rel=args.cert_floor_rel,
            cert_mode=args.cert_mode,
            eig_threshold=args.eig_threshold,
            psd_repair=args.psd_repair,
            a_cap=args.a_cap,
            b_cap=args.b_cap,
            c_cap=args.c_cap,
            ell_grid=args.ell_grid,
            allow_fp64_fallback=args.allow_fp64_fallback,
        )

    if args.mode == "demo":
        res = run_one(args.m, args.n, args.seed)
        print("")
        print(
            f"demo m={args.m} n={args.n}: success={res.success} final_kappa(O)={res.final_kappa_O:.6g} steps={res.steps}"
        )
        print(
            f"  ms total={res.ms_total:.3f} (gram={res.ms_gram:.3f} solve={res.ms_solve:.3f} upd={res.ms_upd:.3f})"
        )
        return

    if args.mode == "bank":
        spectra = make_spectrum_bank(
            args.n, args.kappa_G, bank_size=args.bank_size, seed=args.seed
        )
        finals: List[float] = []
        steps: List[int] = []
        for i, s in enumerate(spectra):
            G = make_matrix_from_singulars(
                args.m, s, seed=args.seed + 1000 + i, device=args.device
            )
            try:
                res = run_polar_dwh_predcap(
                    G_storage=G,
                    target_kappa_O=args.target_kappa_O,
                    max_steps=args.max_steps,
                    ell0=args.ell0,
                    eps_scale=args.eps_scale,
                    safety=args.safety,
                    jitter_rel=args.jitter_rel,
                    tf32=args.tf32,
                    psd_clip=args.psd_clip,
                    cert_floor_rel=args.cert_floor_rel,
                    cert_mode=args.cert_mode,
                    eig_threshold=args.eig_threshold,
                    psd_repair=args.psd_repair,
                    a_cap=args.a_cap,
                    b_cap=args.b_cap,
                    c_cap=args.c_cap,
                    ell_grid=args.ell_grid,
                    allow_fp64_fallback=args.allow_fp64_fallback,
                )
                finals.append(res.final_kappa_O)
                steps.append(res.steps)
            except torch.cuda.OutOfMemoryError:
                finals.append(float("inf"))
                steps.append(0)
                torch.cuda.empty_cache()
        print("")
        print(f"bank summary (N={len(finals)}):")
        print(
            f"  success <= target: {sum(1 for x in finals if x <= args.target_kappa_O)}/{len(finals)}"
        )
        print(
            f"  worst kappa(O): {max(finals):.6g}  median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        print(
            f"  steps median: {pct([float(x) for x in steps], 0.5):.6g}  p90: {pct([float(x) for x in steps], 0.9):.6g}"
        )
        return

    # suite mode
    if args.suite_shapes == "kimi_glm5":
        shapes = suite_shapes_kimi_glm5()
    else:
        shapes = [(args.m, args.n)]

    for m, n in shapes:
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            print(
                f"\nshape m={m} n={n}  (cuda mem free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB)"
            )
        else:
            print(f"\nshape m={m} n={n}")

        spectra = make_spectrum_bank(
            n, args.kappa_G, bank_size=args.suite_cases, seed=args.seed + n
        )
        finals: List[float] = []
        steps_used: List[int] = []
        ms_total: List[float] = []
        ms_gram: List[float] = []
        ms_solve: List[float] = []
        ms_upd: List[float] = []
        successes = 0

        t0 = time.time()
        for i, s in enumerate(spectra):
            try:
                G = make_matrix_from_singulars(
                    m, s, seed=args.seed + 10000 + i, device=args.device
                )
                res = run_polar_dwh_predcap(
                    G_storage=G,
                    target_kappa_O=args.target_kappa_O,
                    max_steps=args.max_steps,
                    ell0=args.ell0,
                    eps_scale=args.eps_scale,
                    safety=args.safety,
                    jitter_rel=args.jitter_rel,
                    tf32=args.tf32,
                    psd_clip=args.psd_clip,
                    cert_floor_rel=args.cert_floor_rel,
                    cert_mode=args.cert_mode,
                    eig_threshold=args.eig_threshold,
                    psd_repair=args.psd_repair,
                    a_cap=args.a_cap,
                    b_cap=args.b_cap,
                    c_cap=args.c_cap,
                    ell_grid=args.ell_grid,
                    allow_fp64_fallback=args.allow_fp64_fallback,
                )
                finals.append(res.final_kappa_O)
                steps_used.append(res.steps)
                successes += int(res.final_kappa_O <= args.target_kappa_O)
                ms_total.append(res.ms_total)
                ms_gram.append(res.ms_gram)
                ms_solve.append(res.ms_solve)
                ms_upd.append(res.ms_upd)
            except torch.cuda.OutOfMemoryError:
                print(f"  case {i:02d} OOM (skipping)")
                finals.append(float("inf"))
                steps_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as ex:
                print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")
                finals.append(float("inf"))
                steps_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))

        dt = time.time() - t0
        print(f"  ran {len(spectra)} cases in {dt:.2f}s")
        print(f"  success <= target: {successes}/{len(spectra)}")
        print(
            f"  worst kappa(O): {max(finals):.6g}  median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        print(
            f"  steps median: {pct([float(x) for x in steps_used], 0.5):.6g}  p90: {pct([float(x) for x in steps_used], 0.9):.6g}"
        )
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        print(
            f"    ms gram  median: {pct(ms_gram, 0.5):.3f}  p90: {pct(ms_gram, 0.9):.3f}"
        )
        print(
            f"    ms solve  median: {pct(ms_solve, 0.5):.3f}  p90: {pct(ms_solve, 0.9):.3f}"
        )
        print(
            f"    ms upd   median: {pct(ms_upd, 0.5):.3f}  p90: {pct(ms_upd, 0.9):.3f}"
        )


if __name__ == "__main__":
    main()
