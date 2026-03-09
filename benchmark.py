#!/usr/bin/env python3
import argparse
import math
import random
import time
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from fast_iroot.cdwh import run_cdwh
from fast_iroot.utils import pct


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--algo", choices=["cdwh"], default="cdwh", help="Algorithm to benchmark"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["demo", "bank", "suite"], default="suite")

    # Generic
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--kappa_G", type=float, default=1e7)
    ap.add_argument("--target_kappa_O", type=float, default=1.22474)
    ap.add_argument("--max_steps", type=int, default=6)

    ap.add_argument("--input_dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--iter_dtype", choices=["float32", "bfloat16"], default="float32")

    ap.add_argument("--cert_mode", choices=["auto", "exact", "bound"], default="auto")
    ap.add_argument("--exact_threshold", type=int, default=1024)

    # Currently kept generic as we only have cdwh to benchmark but these can be factored out later
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
        f"algo={args.algo} device={args.device}  mode={args.mode}  kappa_G<={args.kappa_G:.3g}  target_kappa(O)<={args.target_kappa_O:.6g}"
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

    def run_case(G: Tensor):
        if args.algo == "cdwh":
            return run_cdwh(
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
        else:
            raise ValueError(f"Unknown algorithm: {args.algo}")

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
        print(f"  ms total={res.ms_total:.3f}")
        for k, v in res.ms_details.items():
            print(f"    ms {k}={v:.3f}")
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

        # Aggregate logic for details:
        ms_details_lists = {}

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

                for k, v in res.ms_details.items():
                    ms_details_lists.setdefault(k, []).append(v)

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
                for k in ms_details_lists:
                    ms_details_lists[k].append(float("inf"))

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
                for k in ms_details_lists:
                    ms_details_lists[k].append(float("inf"))
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

        for k, lst in ms_details_lists.items():
            print(
                f"    ms {k:<5} median: {pct(lst, 0.5):.3f}  p90: {pct(lst, 0.9):.3f}"
            )


if __name__ == "__main__":
    main()
