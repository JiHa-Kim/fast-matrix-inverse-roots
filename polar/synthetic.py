from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
import torch

Tensor = torch.Tensor


def pct(xs: List[float], p: float) -> float:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    if not ys:
        return float("nan")
    ys.sort()
    i = int(round(p * (len(ys) - 1)))
    i = max(0, min(len(ys) - 1, i))
    return float(ys[i])


def mean_finite(xs: List[float]) -> float:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    if not ys:
        return float("nan")
    return float(sum(ys) / len(ys))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype name: {name}")


def make_matrix_from_singulars(
    m: int, singulars: Tensor, seed: int, device: str, storage_dtype: torch.dtype
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


def suite_shapes_light() -> List[Tuple[int, int]]:
    return [
        (2048, 256),
        (4096, 256),
        (8192, 256),
        (8192, 1024),
        (16384, 1024),
        (8192, 2048),
        (16384, 2048),
    ]
