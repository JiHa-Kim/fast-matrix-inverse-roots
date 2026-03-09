import math
import time
from typing import List

import torch
from torch import Tensor


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
