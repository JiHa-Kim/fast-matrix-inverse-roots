#!/usr/bin/env python3
import math
import random
from typing import List, Tuple

import torch

from .ops import seed_all, symmetrize

Tensor = torch.Tensor


def make_spd_from_eigs(
    eigs: Tensor,
    seed: int,
    device: str,
    storage_dtype: torch.dtype,
) -> Tensor:
    n = int(eigs.numel())
    seed_all(seed)
    Q, _ = torch.linalg.qr(
        torch.randn(n, n, device=device, dtype=torch.float64),
        mode="reduced",
    )
    P = (Q * eigs.to(device=device, dtype=torch.float64)) @ Q.T
    P = symmetrize(P)
    return P.to(dtype=storage_dtype)


def make_tall_random(
    m: int,
    n: int,
    seed: int,
    device: str,
    storage_dtype: torch.dtype,
) -> Tensor:
    seed_all(seed)
    G = torch.randn(m, n, device=device, dtype=torch.float32)
    return G.to(dtype=storage_dtype)


def make_eig_bank(n: int, kappa_P: float, bank_size: int, seed: int) -> List[Tensor]:
    lam_max = 1.0
    lam_min = 1.0 / float(kappa_P)
    out: List[Tensor] = []

    out.append(
        torch.logspace(0.0, math.log10(lam_min), n, base=10.0, dtype=torch.float64)
    )

    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        logs1 = math.log(lam_max) + (math.log(lam_min) - math.log(lam_max)) * (t**p)
        logs2 = math.log(lam_max) + (math.log(lam_min) - math.log(lam_max)) * (
            1.0 - (1.0 - t) ** p
        )
        out.append(torch.exp(logs1))
        out.append(torch.exp(logs2))

    for frac in [1 / n, 2 / n, 4 / n, 8 / n, 0.1, 0.25, 0.5, 0.75, 0.9]:
        r = max(1, min(n - 1, int(round(frac * n))))
        d = torch.full((n,), lam_min, dtype=torch.float64)
        d[:r] = lam_max
        out.append(d)

    rng = random.Random(seed)
    while len(out) < bank_size:
        u = sorted([rng.random() for _ in range(n)], reverse=True)
        logs = torch.tensor([math.log(lam_min) * x for x in u], dtype=torch.float64)
        d = torch.exp(logs)
        d[0] = lam_max
        d[-1] = lam_min
        out.append(d)

    return out[:bank_size]


def suite_shapes_default() -> List[Tuple[int, int]]:
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
