from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import torch

from .coupled import (
    IrootWorkspaceCoupled,
    InverseSolveWorkspaceCoupled,
    inverse_proot_pe_quadratic_coupled,
    inverse_solve_pe_quadratic_coupled,
)
from .precond import PrecondStats, precond_gram_spd


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
    assume_spd: bool = True,
    nonspd_adaptive: bool = False,
    nonspd_adaptive_resid_tol: float = 1.0,
    nonspd_adaptive_growth_tol: float = 1.02,
    nonspd_adaptive_check_every: int = 1,
    nonspd_safe_fallback_tol: Optional[float] = None,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """
    Apply an iterative inverse to M_norm using a coupled quadratic PE scheme.
    This effectively computes Z ≈ A_norm^{-1} M_norm by evolving an operator.
    Set assume_spd=False for general (non-symmetric) matrices.
    For non-SPD p=1 solves, set nonspd_adaptive=True to enable runtime
    inverse-Newton fallback when Y residual appears unstable.
    Optionally set nonspd_safe_fallback_tol to trigger exact solve fallback
    when final ||A Z - M||/||M|| remains above tolerance.

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
        assume_spd=assume_spd,
        nonspd_adaptive=nonspd_adaptive,
        nonspd_adaptive_resid_tol=nonspd_adaptive_resid_tol,
        nonspd_adaptive_growth_tol=nonspd_adaptive_growth_tol,
        nonspd_adaptive_check_every=nonspd_adaptive_check_every,
        nonspd_safe_fallback_tol=nonspd_safe_fallback_tol,
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
    assume_spd: bool = True,
    nonspd_adaptive: bool = False,
    nonspd_adaptive_resid_tol: float = 1.0,
    nonspd_adaptive_growth_tol: float = 1.02,
    nonspd_adaptive_check_every: int = 1,
    nonspd_safe_fallback_tol: Optional[float] = None,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """
    Apply an iterative inverse p-th root to M_norm using a coupled quadratic PE scheme.
    This effectively computes Z ≈ A_norm^{-1/p} M_norm.
    Set assume_spd=False for general (non-symmetric) matrices.

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
        assume_spd=assume_spd,
        nonspd_adaptive=nonspd_adaptive,
        nonspd_adaptive_resid_tol=nonspd_adaptive_resid_tol,
        nonspd_adaptive_growth_tol=nonspd_adaptive_growth_tol,
        nonspd_adaptive_check_every=nonspd_adaptive_check_every,
        nonspd_safe_fallback_tol=nonspd_safe_fallback_tol,
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
    assume_spd: bool = True,
    nonspd_adaptive: bool = False,
    nonspd_adaptive_resid_tol: float = 1.0,
    nonspd_adaptive_growth_tol: float = 1.02,
    nonspd_adaptive_check_every: int = 1,
    nonspd_safe_fallback_tol: Optional[float] = None,
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
            assume_spd=assume_spd,
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
        assume_spd=assume_spd,
        nonspd_adaptive=nonspd_adaptive,
        nonspd_adaptive_resid_tol=nonspd_adaptive_resid_tol,
        nonspd_adaptive_growth_tol=nonspd_adaptive_growth_tol,
        nonspd_adaptive_check_every=nonspd_adaptive_check_every,
        nonspd_safe_fallback_tol=nonspd_safe_fallback_tol,
    )
    return Zn, ws


@torch.no_grad()
def apply_inverse_sqrt_spd(
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
    """Dedicated SPD p=2 apply path."""
    return apply_inverse_root(
        A_norm,
        M_norm,
        abc_t=abc_t,
        p_val=2,
        ws=ws,
        symmetrize_Y=symmetrize_Y,
        symmetrize_every=symmetrize_every,
        terminal_last_step=terminal_last_step,
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
        assume_spd=True,
    )


@torch.no_grad()
def apply_inverse_sqrt_non_spd(
    A: torch.Tensor,
    M: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
    terminal_last_step: bool = True,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """Dedicated non-SPD p=2 apply path (no symmetry assumptions)."""
    return apply_inverse_root(
        A,
        M,
        abc_t=abc_t,
        p_val=2,
        ws=ws,
        symmetrize_Y=False,
        symmetrize_every=1,
        terminal_last_step=terminal_last_step,
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
        assume_spd=False,
    )


@torch.no_grad()
def apply_inverse_sqrt_gram_spd(
    G: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
    gram_mode: str = "col-norm",
    precond_mode: str = "none",
    eps: float = 1e-12,
    precond_ruiz_iters: int = 2,
    ridge_rel: float = 0.0,
    l_target: float = 0.05,
    lambda_max_est: str = "row_sum",
    lambda_max_power_iters: int = 8,
    lambda_max_safety: float = 1.02,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled, PrecondStats]:
    """Gram-matrix SPD p=2 apply path: precondition A=G^T G, then apply inverse sqrt."""
    A_norm, stats = precond_gram_spd(
        G,
        gram_mode=gram_mode,
        mode=precond_mode,
        eps=eps,
        ruiz_iters=precond_ruiz_iters,
        ridge_rel=ridge_rel,
        l_target=l_target,
        lambda_max_est=lambda_max_est,
        lambda_max_power_iters=lambda_max_power_iters,
        lambda_max_safety=lambda_max_safety,
    )
    Z, ws = apply_inverse_sqrt_spd(
        A_norm,
        M_norm,
        abc_t=abc_t,
        ws=ws,
        symmetrize_Y=symmetrize_Y,
        symmetrize_every=symmetrize_every,
        terminal_last_step=terminal_last_step,
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
    )
    return Z, ws, stats
