from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import torch

from .coupled import (
    IrootWorkspaceCoupled,
    InverseSolveWorkspaceCoupled,
    inverse_proot_pe_quadratic_coupled,
    inverse_solve_pe_quadratic_coupled,
)


@dataclass
class InverseApplyAutoWorkspace:
    solve_ws: Optional[InverseSolveWorkspaceCoupled] = None
    root_ws: Optional[IrootWorkspaceCoupled] = None


@torch.no_grad()
def apply_inverse(
    A_norm: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """
    Apply an iterative inverse to M_norm using a coupled quadratic PE scheme.
    This effectively computes Z ≈ A_norm^{-1} M_norm by evolving an operator.

    Note: When terminal_last_step=True, ws.Y is not advanced on the final step.
    """
    return inverse_solve_pe_quadratic_coupled(
        A_norm=A_norm,
        M_norm=M_norm,
        abc_t=abc_t,
        p_val=1,
        ws=ws,
        symmetrize_Y=symmetrize_Y,
        symmetrize_every=symmetrize_every,
        terminal_last_step=terminal_last_step,
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
    )


@torch.no_grad()
def apply_inverse_root(
    A_norm: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """
    Apply an iterative inverse p-th root to M_norm using a coupled quadratic PE scheme.
    This effectively computes Z ≈ A_norm^{-1/p} M_norm.

    Note: When terminal_last_step=True, ws.Y is not advanced on the final step.
    """
    return inverse_solve_pe_quadratic_coupled(
        A_norm=A_norm,
        M_norm=M_norm,
        abc_t=abc_t,
        p_val=p_val,
        ws=ws,
        symmetrize_Y=symmetrize_Y,
        symmetrize_every=symmetrize_every,
        terminal_last_step=terminal_last_step,
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
    )


@torch.no_grad()
def apply_inverse_root_auto(
    A_norm: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[InverseApplyAutoWorkspace] = None,
    strategy: str = "auto",
    expected_reuse: int = 1,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
) -> Tuple[torch.Tensor, InverseApplyAutoWorkspace]:
    """Apply inverse p-th root with strategy selection for single-shot vs reuse.

    Strategies:
      - `auto` (default): direct solve for single-shot (`expected_reuse <= 1`),
        materialize root then multiply for repeated reuse (`expected_reuse > 1`).
      - `direct-solve`: always run coupled solve on RHS block.
      - `materialize-root`: compute root operator then multiply (`X @ M_norm`).
    """
    if strategy not in ("auto", "direct-solve", "materialize-root"):
        raise ValueError(
            "Unknown strategy: "
            f"'{strategy}'. Supported strategies are 'auto', 'direct-solve', 'materialize-root'."
        )
    reuse = int(expected_reuse)
    if reuse < 1:
        raise ValueError(f"expected_reuse must be >= 1, got {expected_reuse}")

    if ws is None:
        ws = InverseApplyAutoWorkspace()

    use_materialize = strategy == "materialize-root" or (
        strategy == "auto" and reuse > 1
    )

    if use_materialize:
        Xn, ws.root_ws = inverse_proot_pe_quadratic_coupled(
            A_norm,
            abc_t=abc_t,
            p_val=p_val,
            ws=ws.root_ws,
            symmetrize_Y=symmetrize_Y,
            symmetrize_every=symmetrize_every,
            terminal_last_step=terminal_last_step,
            online_stop_tol=online_stop_tol,
            online_min_steps=online_min_steps,
        )
        return Xn @ M_norm, ws

    Zn, ws.solve_ws = inverse_solve_pe_quadratic_coupled(
        A_norm=A_norm,
        M_norm=M_norm,
        abc_t=abc_t,
        p_val=p_val,
        ws=ws.solve_ws,
        symmetrize_Y=symmetrize_Y,
        symmetrize_every=symmetrize_every,
        terminal_last_step=terminal_last_step,
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
    )
    return Zn, ws
