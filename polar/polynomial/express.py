from __future__ import annotations

import dataclasses

import torch

from polar.ops import gram_xtx, symmetrize

Tensor = torch.Tensor


@dataclasses.dataclass(frozen=True)
class PaperPolarExpressStep:
    a: float
    b: float
    c: float


_PE5_PAPER_COEFFS: tuple[PaperPolarExpressStep, ...] = (
    PaperPolarExpressStep(8.28721201814563, -23.595886519098837, 17.300387312530933),
    PaperPolarExpressStep(4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    PaperPolarExpressStep(3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    PaperPolarExpressStep(3.3184196573706015, -2.488488024314874, 0.51004894012372),
    PaperPolarExpressStep(2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    PaperPolarExpressStep(1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    PaperPolarExpressStep(1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    PaperPolarExpressStep(1.875, -1.25, 0.375),
)


def paper_polar_express_coeff(step_idx: int) -> PaperPolarExpressStep:
    idx = min(max(int(step_idx), 0), len(_PE5_PAPER_COEFFS) - 1)
    return _PE5_PAPER_COEFFS[idx]


@torch.no_grad()
def polar_express_paper5_step_matrix_only(
    S: Tensor,
    coeffs: PaperPolarExpressStep,
    matmul_dtype: torch.dtype,
) -> tuple[Tensor, float]:
    S_work = symmetrize(S.to(dtype=matmul_dtype))
    n = S_work.shape[0]
    I = torch.eye(n, device=S_work.device, dtype=matmul_dtype)
    S2 = symmetrize(S_work @ S_work)
    Q = symmetrize(float(coeffs.a) * I + float(coeffs.b) * S_work + float(coeffs.c) * S2)
    if not torch.isfinite(Q).all():
        raise RuntimeError("non-finite paper polar express step")
    return Q, 0.0


@torch.no_grad()
def polar_express_paper_fro_scale(
    X: Tensor,
    safety: float = 1.01,
    eps: float = 1e-7,
) -> tuple[Tensor, float]:
    fro = torch.linalg.matrix_norm(X.float(), ord="fro").clamp_min(float(eps))
    scale = float(safety) * fro
    X_scaled = X / scale.to(dtype=X.dtype)
    return X_scaled, float(scale.item())


@torch.no_grad()
def polar_express_aol_scale(
    X: Tensor,
    accum_dtype: torch.dtype,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    S = gram_xtx(X, accum_dtype)
    s = torch.rsqrt(S.abs().sum(dim=-1).clamp_min(float(eps)))
    X_scaled = X * s.unsqueeze(0).to(dtype=X.dtype)
    return X_scaled, s
