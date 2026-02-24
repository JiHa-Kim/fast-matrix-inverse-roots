from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .utils import _matmul_into, _symmetrize_inplace
from .coeffs import _affine_coeffs, _quad_coeffs


@dataclass
class IsqrtWorkspaceCoupled:
    X: torch.Tensor
    Xbuf: torch.Tensor
    Y: torch.Tensor
    Ybuf: torch.Tensor
    Y2: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor
    eye_mat: torch.Tensor


def _alloc_ws_coupled(A: torch.Tensor) -> IsqrtWorkspaceCoupled:
    shape = A.shape
    n = shape[-1]
    # IMPORTANT: do NOT .contiguous() an expanded identity; that materializes a full batch of I.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    return IsqrtWorkspaceCoupled(
        X=A.new_empty(shape),
        Xbuf=A.new_empty(shape),
        Y=A.new_empty(shape),
        Ybuf=A.new_empty(shape),
        Y2=A.new_empty(shape),
        B=A.new_empty(shape),
        B2=A.new_empty(shape),
        eye_mat=eye,
    )


def _ws_ok_coupled(ws: Optional[IsqrtWorkspaceCoupled], A: torch.Tensor) -> bool:
    if ws is None:
        return False
    return (
        ws.X.device == A.device
        and ws.X.dtype == A.dtype
        and ws.X.shape == A.shape
        and ws.eye_mat.device == A.device
        and ws.eye_mat.dtype == A.dtype
        and ws.eye_mat.shape == A.shape
    )


@torch.no_grad()
def inverse_sqrt_ns(
    A_norm: torch.Tensor,
    iters: int,
    ws: Optional[IsqrtWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspaceCoupled]:
    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)

    T = int(iters)
    for t in range(T):
        ws.B.copy_(ws.Y).mul_(-0.5)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(1.5)

        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        # NOTE: if terminal_last_step is True, ws.Y is intentionally not updated on the last step.
        if terminal_last_step and (t == T - 1):
            break

        _matmul_into(ws.Y, ws.B, ws.B2)
        _matmul_into(ws.B, ws.B2, ws.Ybuf)
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


@torch.no_grad()
def inverse_sqrt_pe_affine(
    A_norm: torch.Tensor,
    ab_t: Sequence[Tuple[float, float]] | torch.Tensor,
    ws: Optional[IsqrtWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspaceCoupled]:
    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)
    coeffs = _affine_coeffs(ab_t)

    T = len(coeffs)
    for t, (a, b) in enumerate(coeffs):
        ws.B.copy_(ws.Y).mul_(b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)

        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        # NOTE: if terminal_last_step is True, ws.Y is intentionally not updated on the last step.
        if terminal_last_step and (t == T - 1):
            break

        _matmul_into(ws.Y, ws.B, ws.B2)
        _matmul_into(ws.B, ws.B2, ws.Ybuf)
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


@torch.no_grad()
def inverse_sqrt_pe_quadratic(
    A_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[IsqrtWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspaceCoupled]:
    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs(abc_t)

    T = len(coeffs)
    for t, (a, b, c) in enumerate(coeffs):
        _matmul_into(ws.Y, ws.Y, ws.Y2)
        ws.B.copy_(ws.Y2).mul_(c)
        ws.B.add_(ws.Y, alpha=b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)

        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        # NOTE: if terminal_last_step is True, ws.Y is intentionally not updated on the last step.
        if terminal_last_step and (t == T - 1):
            break

        _matmul_into(ws.Y, ws.B, ws.B2)
        _matmul_into(ws.B, ws.B2, ws.Ybuf)
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws
