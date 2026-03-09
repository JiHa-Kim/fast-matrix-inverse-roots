#!/usr/bin/env python3
# polar_dwh_bf16_honest_baseline.py
#
# Correctness-first DWH/QDWH-style polar baseline for tall matrices.
#
# Key choices:
#   - Uses actual DWH coefficients from Nakatsukasa-Bai-Gygi.
#   - Uses the numerically safer update:
#       X_{k+1} = (b/c) X + (a - b/c) X (I + c X^T X)^(-1)
#   - Computes the small Gram in chunked fp64.
#   - Solves the small right-side system in chunks to control memory.
#   - Uses exact eig certification for small n, and trace/logdet bound otherwise.
#
# This is intended as an honest baseline, not the final fastest Muon path.

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


# ----------------------------- utilities ------------------------------------


def symmetrize(A: Tensor) -> Tensor:
    return 0.5 * (A + A.T)


def pct(xs: List[float], p: float) -> float:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    if not ys:
        return float("nan")
    ys.sort()
    i = int(round(p * (len(ys) - 1)))
    i = max(0, min(len(ys) - 1, i))
    return float(ys[i])


def cuda_time_ms(fn):
    if not torch.cuda.is_available():
        t0 = time.time()
        out = fn()
        return 1000.0 * (time.time() - t0), out
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_exp(x: float) -> float:
    if x >= 709.0:
        return float("inf")
    return float(math.exp(x))


def acosh_exp(logu: float) -> float:
    if logu <= 0.0:
        return 0.0
    if logu < 20.0:
        u = math.exp(logu)
        return float(math.acosh(max(u, 1.0)))
    return float(logu + math.log(2.0))


# ----------------------------- fp64 small-side ops --------------------------


@torch.no_grad()
def gram_xtx_chunked_fp64(
    X: Tensor,
    chunk_rows: int,
) -> Tensor:
    """
    Accurate small Gram S = X^T X in fp64, accumulated in row chunks.
    """
    m, n = X.shape
    device = X.device
    S = torch.zeros((n, n), device=device, dtype=torch.float64)

    for i in range(0, m, chunk_rows):
        Xi = X[i : i + chunk_rows]
        Xi = Xi.float().to(torch.float64)
        S.addmm_(Xi.T, Xi)

    return symmetrize(S)


@torch.no_grad()
def chol_with_jitter_fp64(
    A: Tensor,
    jitter_rel: float,
    max_tries: int = 8,
) -> Tuple[Tensor, float]:
    A = symmetrize(A.to(torch.float64))
    if not torch.isfinite(A).all():
        raise RuntimeError("non-finite matrix before Cholesky")

    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=torch.float64)

    scale = float((torch.trace(A).abs() / max(n, 1)).item())
    base = max(float(jitter_rel) * max(scale, 1.0), 1e-30)

    delta = 0.0
    for _ in range(max_tries):
        At = A if delta == 0.0 else (A + delta * I)
        L, info = torch.linalg.cholesky_ex(At)
        if int(info.item()) == 0:
            return L, float(delta)
        delta = base if delta == 0.0 else 2.0 * delta

    raise RuntimeError("Cholesky failed even after jitter escalation")


# ----------------------------- DWH scalar theory ----------------------------


def dwh_coeffs_from_ell(ell: float) -> Tuple[float, float, float]:
    """
    DWH coefficients from Nakatsukasa-Bai-Gygi:
      a = h(ell)
      b = (a - 1)^2 / 4
      c = a + b - 1
    """
    ell = float(min(max(ell, 1e-300), 1.0))
    ell2 = ell * ell

    d = (4.0 * (1.0 - ell2) / (ell2 * ell2)) ** (1.0 / 3.0)
    a = math.sqrt(1.0 + d) + 0.5 * math.sqrt(
        8.0 - 4.0 * d + 8.0 * (2.0 - ell2) / (ell2 * math.sqrt(1.0 + d))
    )
    b = 0.25 * (a - 1.0) * (a - 1.0)
    c = a + b - 1.0
    return float(a), float(b), float(c)


def dwh_ell_next(ell: float, a: float, b: float, c: float) -> float:
    ell = float(ell)
    return float(ell * (a + b * ell * ell) / (1.0 + c * ell * ell))


# ----------------------------- DWH matrix update ----------------------------


@torch.no_grad()
def build_q_from_s(
    S: Tensor,
    a: float,
    b: float,
    c: float,
    jitter_rel: float,
) -> Tuple[Tensor, Tensor, float]:
    """
    Build:
      M = I + c S
      Q = (a I + b S) M^{-1}
    in fp64.

    Returns:
      Q, L, shift
    where L is chol(M + shift I), useful for chunked X update.
    """
    S = symmetrize(S.to(torch.float64))
    n = S.shape[0]
    I = torch.eye(n, device=S.device, dtype=torch.float64)

    M = symmetrize(I + float(c) * S)
    R = symmetrize(float(a) * I + float(b) * S)

    L, shift = chol_with_jitter_fp64(M, jitter_rel=jitter_rel)
    Q = torch.cholesky_solve(R, L)
    Q = symmetrize(Q)
    return Q, L, float(shift)


@torch.no_grad()
def apply_dwh_update_chunked(
    X: Tensor,
    L: Tensor,
    a: float,
    b: float,
    c: float,
    rhs_chunk_rows: int,
    out_dtype: torch.dtype,
) -> Tensor:
    """
    Apply:
      X_{k+1} = (b/c) X + (a - b/c) X (I + c X^T X)^(-1)

    using the Cholesky factor L of M = I + c X^T X.
    """
    alpha = float(b / c)
    beta = float(a - b / c)

    m, n = X.shape
    device = X.device
    X_next = torch.empty((m, n), device=device, dtype=out_dtype)

    for i in range(0, m, rhs_chunk_rows):
        Xi = X[i : i + rhs_chunk_rows].float().to(torch.float64)  # (b, n)
        Yi_t = torch.cholesky_solve(Xi.T.contiguous(), L)  # (n, b)
        Yi = Yi_t.T  # (b, n)
        Zi = alpha * Xi + beta * Yi
        X_next[i : i + rhs_chunk_rows] = Zi.to(out_dtype)

    return X_next


# ----------------------------- certification --------------------------------


@torch.no_grad()
def cert_from_s(
    S: Tensor,
    cert_mode: str,
    exact_threshold: int,
    chol_jitter_rel: float,
) -> Tuple[float, float, float]:
    """
    Returns:
      kappa_O_value_or_upper_bound,
      exact_value_or_nan,
      shift_used_for_cholesky

    Modes:
      - exact: always eigvalsh
      - bound: always trace/logdet upper bound
      - auto: exact if n <= exact_threshold else bound
    """
    S = symmetrize(S.to(torch.float64))
    n = S.shape[0]

    use_exact = (cert_mode == "exact") or (cert_mode == "auto" and n <= exact_threshold)

    if use_exact:
        evals = torch.linalg.eigvalsh(S)
        lam_min = max(float(evals[0].item()), 1e-300)
        lam_max = max(float(evals[-1].item()), lam_min)
        kappa_O = float(math.sqrt(lam_max / lam_min))
        return float(kappa_O), float(kappa_O), 0.0

    L, shift = chol_with_jitter_fp64(S, jitter_rel=chol_jitter_rel)
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum().item()

    a = max(float((torch.trace(S) / n).item()), 1e-300)
    g = safe_exp(logdet / n)
    r = max(a / max(g, 1e-300), 1.0)

    logu = 0.5 * n * math.log(r)
    eta_ub = acosh_exp(logu)
    kappa_O_ub = safe_exp(eta_ub)
    return float(kappa_O_ub), float("nan"), float(shift)


# ----------------------------- synthetic matrices ---------------------------


def make_matrix_from_singulars(
    m: int,
    singulars: Tensor,
    seed: int,
    device: str,
    storage_dtype: torch.dtype,
) -> Tensor:
    n = int(singulars.numel())

    seed_all(seed)
    U, _ = torch.linalg.qr(
        torch.randn(m, n, device=device, dtype=torch.float32),
        mode="reduced",
    )

    seed_all(seed + 1)
    V, _ = torch.linalg.qr(
        torch.randn(n, n, device=device, dtype=torch.float32),
        mode="reduced",
    )

    G = (U * singulars.to(device=device, dtype=torch.float32)) @ V.T
    return G.to(dtype=storage_dtype)


def make_spectrum_bank(
    n: int, kappa_G: float, bank_size: int, seed: int
) -> List[Tensor]:
    sig_max = 1.0
    sig_min = 1.0 / float(kappa_G)
    out: List[Tensor] = []

    out.append(
        torch.logspace(0.0, math.log10(sig_min), n, base=10.0, dtype=torch.float32)
    )

    t = torch.linspace(0.0, 1.0, n, dtype=torch.float32)
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        logs1 = math.log(sig_max) + (math.log(sig_min) - math.log(sig_max)) * (t**p)
        logs2 = math.log(sig_max) + (math.log(sig_min) - math.log(sig_max)) * (
            1.0 - (1.0 - t) ** p
        )
        out.append(torch.exp(logs1))
        out.append(torch.exp(logs2))

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


def suite_shapes_kimi_glm5() -> List[Tuple[int, int]]:
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


# ----------------------------- run core -------------------------------------


@dataclasses.dataclass
class RunSummary:
    success: bool
    final_kO_cert: float
    final_kO_exact: float
    final_kO_pred: float
    steps: int
    guards: int
    ms_gram: float
    ms_solve: float
    ms_upd: float
    ms_cert: float
    ms_total: float


@torch.no_grad()
def run_one_case(
    G_storage: Tensor,
    kappa_G_upper: float,
    target_kappa_O: float,
    max_steps: int,
    iter_dtype: torch.dtype,
    cert_mode: str,
    exact_threshold: int,
    gram_chunk_rows: int,
    rhs_chunk_rows: int,
    solve_jitter_rel: float,
    cert_jitter_rel: float,
    tf32: bool,
) -> RunSummary:
    device = G_storage.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high")

    # Honest synthetic assumption:
    # spectra are generated with sigma_max = 1 and sigma_min = 1 / kappa_G_upper.
    X = G_storage.to(dtype=iter_dtype)
    ell = 1.0 / float(kappa_G_upper)

    ms_gram_sum = 0.0
    ms_solve_sum = 0.0
    ms_upd_sum = 0.0
    ms_cert_sum = 0.0
    guards = 0

    # Initial small Gram in fp64.
    ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fp64(X, gram_chunk_rows))
    ms_gram_sum += ms_gram

    final_kO_cert = float("inf")
    final_kO_exact = float("nan")
    final_kO_pred = float("inf")

    for it in range(1, max_steps + 1):
        a, b, c = dwh_coeffs_from_ell(ell)

        ms_solve, (Q, L, shift) = cuda_time_ms(
            lambda: build_q_from_s(
                S=S,
                a=a,
                b=b,
                c=c,
                jitter_rel=solve_jitter_rel,
            )
        )
        ms_solve_sum += ms_solve
        guards += int(shift > 0.0)

        # Small-side propagated certificate matrix for next step / cert.
        S_next = symmetrize(Q @ S @ Q)
        S = S_next

        ms_upd, X = cuda_time_ms(
            lambda: apply_dwh_update_chunked(
                X=X,
                L=L,
                a=a,
                b=b,
                c=c,
                rhs_chunk_rows=rhs_chunk_rows,
                out_dtype=iter_dtype,
            )
        )
        ms_upd_sum += ms_upd

        ell = dwh_ell_next(ell, a, b, c)
        final_kO_pred = 1.0 / max(ell, 1e-300)

        ms_cert, (kO_cert, kO_exact, cert_shift) = cuda_time_ms(
            lambda: cert_from_s(
                S=S,
                cert_mode=cert_mode,
                exact_threshold=exact_threshold,
                chol_jitter_rel=cert_jitter_rel,
            )
        )
        ms_cert_sum += ms_cert
        guards += int(cert_shift > 0.0)

        final_kO_cert = float(kO_cert)
        final_kO_exact = float(kO_exact)

        if final_kO_cert <= target_kappa_O:
            ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum + ms_cert_sum
            return RunSummary(
                True,
                final_kO_cert,
                final_kO_exact,
                final_kO_pred,
                it,
                guards,
                ms_gram_sum,
                ms_solve_sum,
                ms_upd_sum,
                ms_cert_sum,
                ms_total,
            )

    ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum + ms_cert_sum
    return RunSummary(
        False,
        final_kO_cert,
        final_kO_exact,
        final_kO_pred,
        max_steps,
        guards,
        ms_gram_sum,
        ms_solve_sum,
        ms_upd_sum,
        ms_cert_sum,
        ms_total,
    )


# ----------------------------- CLI ------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["demo", "bank", "suite"], default="suite")

    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--kappa_G", type=float, default=1e7)
    ap.add_argument("--target_kappa_O", type=float, default=1.22474)
    ap.add_argument("--max_steps", type=int, default=6)

    ap.add_argument("--input_dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--iter_dtype", choices=["float32", "bfloat16"], default="float32")

    ap.add_argument("--cert_mode", choices=["auto", "exact", "bound"], default="auto")
    ap.add_argument("--exact_threshold", type=int, default=1024)

    ap.add_argument("--gram_chunk_rows", type=int, default=2048)
    ap.add_argument("--rhs_chunk_rows", type=int, default=2048)

    ap.add_argument("--solve_jitter_rel", type=float, default=1e-15)
    ap.add_argument("--cert_jitter_rel", type=float, default=1e-15)

    ap.add_argument("--tf32", action="store_true")

    ap.add_argument("--bank_size", type=int, default=12)
    ap.add_argument("--suite_cases", type=int, default=6)
    ap.add_argument("--suite_shapes", choices=["kimi_glm5"], default="kimi_glm5")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    input_dtype = torch.float32 if args.input_dtype == "float32" else torch.bfloat16
    iter_dtype = torch.float32 if args.iter_dtype == "float32" else torch.bfloat16

    print(
        f"device={args.device}  mode={args.mode}  kappa_G<={args.kappa_G:.3g}  target_kappa(O)<={args.target_kappa_O:.6g}"
    )
    print(
        "knobs: "
        f"max_steps={args.max_steps} input_dtype={args.input_dtype} iter_dtype={args.iter_dtype} "
        f"cert_mode={args.cert_mode} exact_threshold={args.exact_threshold} "
        f"gram_chunk_rows={args.gram_chunk_rows} rhs_chunk_rows={args.rhs_chunk_rows} "
        f"solve_jitter_rel={args.solve_jitter_rel:g} cert_jitter_rel={args.cert_jitter_rel:g} tf32={args.tf32}"
    )
    if args.input_dtype == "bfloat16":
        print(
            "WARNING: bfloat16 input storage changes the actual synthetic spectrum. Use input_dtype=float32 for honest stress tests."
        )

    def make_case(m: int, n: int, case_seed: int) -> Tensor:
        spectra = make_spectrum_bank(n, args.kappa_G, bank_size=1, seed=case_seed + n)
        return make_matrix_from_singulars(
            m=m,
            singulars=spectra[0],
            seed=case_seed,
            device=args.device,
            storage_dtype=input_dtype,
        )

    def run_case(G: Tensor) -> RunSummary:
        return run_one_case(
            G_storage=G,
            kappa_G_upper=args.kappa_G,
            target_kappa_O=args.target_kappa_O,
            max_steps=args.max_steps,
            iter_dtype=iter_dtype,
            cert_mode=args.cert_mode,
            exact_threshold=args.exact_threshold,
            gram_chunk_rows=args.gram_chunk_rows,
            rhs_chunk_rows=args.rhs_chunk_rows,
            solve_jitter_rel=args.solve_jitter_rel,
            cert_jitter_rel=args.cert_jitter_rel,
            tf32=args.tf32,
        )

    if args.mode == "demo":
        G = make_case(args.m, args.n, args.seed)
        res = run_case(G)
        print("")
        print(
            f"demo m={args.m} n={args.n}: success={res.success} "
            f"final_kappa(O)_cert={res.final_kO_cert:.6g} "
            f"exact={res.final_kO_exact:.6g} pred={res.final_kO_pred:.6g} "
            f"steps={res.steps} guards={res.guards}"
        )
        print(
            f"  ms total={res.ms_total:.3f} "
            f"(gram={res.ms_gram:.3f} solve={res.ms_solve:.3f} upd={res.ms_upd:.3f} cert={res.ms_cert:.3f})"
        )
        return

    if args.mode == "bank":
        finals = []
        finals_exact = []
        finals_pred = []
        steps = []
        guards = []
        ms_total = []

        for i in range(args.bank_size):
            try:
                G = make_case(args.m, args.n, args.seed + 1000 + i)
                res = run_case(G)
                finals.append(res.final_kO_cert)
                finals_exact.append(res.final_kO_exact)
                finals_pred.append(res.final_kO_pred)
                steps.append(res.steps)
                guards.append(res.guards)
                ms_total.append(res.ms_total)
                del G
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                finals.append(float("inf"))
                finals_exact.append(float("nan"))
                finals_pred.append(float("inf"))
                steps.append(0)
                guards.append(0)
                ms_total.append(float("inf"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        print("")
        print(f"bank summary (N={len(finals)}):")
        print(
            f"  success <= target: {sum(1 for x in finals if x <= args.target_kappa_O)}/{len(finals)}"
        )
        print(
            f"  worst kappa(O)_cert: {max(finals):.6g}  median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        if any(math.isfinite(x) for x in finals_exact):
            print(
                f"  exact kappa(O) median: {pct(finals_exact, 0.5):.6g}  p90: {pct(finals_exact, 0.9):.6g}"
            )
        print(
            f"  pred kappa(O) median: {pct(finals_pred, 0.5):.6g}  p90: {pct(finals_pred, 0.9):.6g}"
        )
        print(f"  steps median: {pct(steps, 0.5):.6g}  p90: {pct(steps, 0.9):.6g}")
        print(f"  guards median: {pct(guards, 0.5):.6g}  p90: {pct(guards, 0.9):.6g}")
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        return

    shapes = (
        suite_shapes_kimi_glm5()
        if args.suite_shapes == "kimi_glm5"
        else [(args.m, args.n)]
    )

    for m, n in shapes:
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            print(
                f"\nshape m={m} n={n}  (cuda mem free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB)"
            )
        else:
            print(f"\nshape m={m} n={n}")

        finals = []
        finals_exact = []
        finals_pred = []
        steps_used = []
        guards_used = []
        ms_total = []
        ms_gram = []
        ms_solve = []
        ms_upd = []
        ms_cert = []
        successes = 0

        t0 = time.time()
        for i in range(args.suite_cases):
            try:
                G = make_case(m, n, args.seed + 10000 + i)
                res = run_case(G)
                finals.append(res.final_kO_cert)
                finals_exact.append(res.final_kO_exact)
                finals_pred.append(res.final_kO_pred)
                steps_used.append(res.steps)
                guards_used.append(res.guards)
                successes += int(res.final_kO_cert <= args.target_kappa_O)
                ms_total.append(res.ms_total)
                ms_gram.append(res.ms_gram)
                ms_solve.append(res.ms_solve)
                ms_upd.append(res.ms_upd)
                ms_cert.append(res.ms_cert)
                del G
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  case {i:02d} OOM (skipping)")
                finals.append(float("inf"))
                finals_exact.append(float("nan"))
                finals_pred.append(float("inf"))
                steps_used.append(0)
                guards_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))
                ms_cert.append(float("inf"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as ex:
                print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")
                finals.append(float("inf"))
                finals_exact.append(float("nan"))
                finals_pred.append(float("inf"))
                steps_used.append(0)
                guards_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))
                ms_cert.append(float("inf"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        dt = time.time() - t0
        print(f"  ran {args.suite_cases} cases in {dt:.2f}s")
        print(f"  success <= target: {successes}/{args.suite_cases}")
        print(
            f"  worst kappa(O)_cert: {max(finals):.6g}  median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        if any(math.isfinite(x) for x in finals_exact):
            print(
                f"  exact kappa(O) median: {pct(finals_exact, 0.5):.6g}  p90: {pct(finals_exact, 0.9):.6g}"
            )
        print(
            f"  pred kappa(O) median: {pct(finals_pred, 0.5):.6g}  p90: {pct(finals_pred, 0.9):.6g}"
        )
        print(
            f"  steps median: {pct(steps_used, 0.5):.6g}  p90: {pct(steps_used, 0.9):.6g}"
        )
        print(
            f"  guards median: {pct(guards_used, 0.5):.6g}  p90: {pct(guards_used, 0.9):.6g}"
        )
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        print(
            f"    ms gram  median: {pct(ms_gram, 0.5):.3f}  p90: {pct(ms_gram, 0.9):.3f}"
        )
        print(
            f"    ms solve median: {pct(ms_solve, 0.5):.3f}  p90: {pct(ms_solve, 0.9):.3f}"
        )
        print(
            f"    ms upd   median: {pct(ms_upd, 0.5):.3f}  p90: {pct(ms_upd, 0.9):.3f}"
        )
        print(
            f"    ms cert  median: {pct(ms_cert, 0.5):.3f}  p90: {pct(ms_cert, 0.9):.3f}"
        )


if __name__ == "__main__":
    main()
