#!/usr/bin/env python3
import torch
from .ops import symmetrize

Tensor = torch.Tensor

@torch.no_grad()
def exact_invroot_fp64(P: Tensor, p: int) -> Tensor:
    evals, U = torch.linalg.eigh(symmetrize(P.to(torch.float64)))
    evals = torch.clamp(evals, min=1e-300)
    X = (U * evals.pow(-1.0 / float(p))) @ U.T
    return symmetrize(X)


@torch.no_grad()
def exact_root_resid_fp64(X: Tensor, P: Tensor, p: int) -> float:
    # Check || P^{1/2p} X P^{1/2p} - I ||_2
    evals, U = torch.linalg.eigh(symmetrize(P.to(torch.float64)))
    evals = torch.clamp(evals, min=1e-300)
    P12p = (U * evals.pow(0.5 / float(p))) @ U.T
    S = symmetrize(P12p @ X.to(torch.float64) @ P12p)
    e = torch.linalg.eigvalsh(S)
    lam_min = float(e[0].item())
    lam_max = float(e[-1].item())
    return float(max(abs(1.0 - lam_min), abs(lam_max - 1.0)))
