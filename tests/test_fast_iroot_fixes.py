import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from fast_iroot.utils import _addmm_into
from fast_iroot.coupled import (
    inverse_solve_pe_quadratic_coupled,
    InverseSolveWorkspaceCoupled,
)
from fast_iroot import build_pe_schedules, precond_spd
from fast_iroot.metrics import isqrt_relative_error, exact_inverse_proot


def test_addmm_into_multibatch_shape():
    torch.manual_seed(42)
    b, k, n, m = 2, 3, 4, 4
    bias = torch.randn(b, k, n, m)
    mat1 = torch.randn(b, k, n, m)
    mat2 = torch.randn(b, k, n, m)
    out = torch.empty_like(bias)

    _addmm_into(bias, mat1, mat2, beta=0.5, alpha=1.2, out=out)
    expected = 0.5 * bias + 1.2 * (mat1 @ mat2)
    assert torch.allclose(out, expected, atol=1e-5)


def test_inverse_solve_dtype_mismatch_raises():
    A_norm = torch.randn(4, 4, dtype=torch.float32)
    M_norm = torch.randn(4, 4, dtype=torch.float64)
    abc_t = [(0.1, 0.2, 0.3)]
    with pytest.raises(ValueError, match="same dtype"):
        inverse_solve_pe_quadratic_coupled(A_norm, M_norm, abc_t)


def test_inverse_solve_device_mismatch_raises():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    A_norm = torch.randn(4, 4, device="cpu")
    M_norm = torch.randn(4, 4, device="cuda")
    abc_t = [(0.1, 0.2, 0.3)]
    with pytest.raises(ValueError, match="same device"):
        inverse_solve_pe_quadratic_coupled(A_norm, M_norm, abc_t)


def test_metrics_shape_mismatch_raises():
    Xhat = torch.randn(4, 4)
    A = torch.randn(5, 5)
    with pytest.raises(ValueError, match="compatible shapes"):
        isqrt_relative_error(Xhat, A)


def test_addmm_into_non_contiguous():
    torch.manual_seed(42)
    b, k, n, m = 2, 3, 4, 4
    bias = torch.randn(b, k, n, m)
    mat1 = torch.randn(b, k, m, n).transpose(-1, -2)
    mat2 = torch.randn(b, k, m, n).transpose(-1, -2)
    out = torch.empty_like(bias)

    assert not mat1.is_contiguous()
    assert not mat2.is_contiguous()

    _addmm_into(bias, mat1, mat2, beta=0.5, alpha=1.2, out=out)
    expected = 0.5 * bias + 1.2 * (mat1 @ mat2)
    assert torch.allclose(out, expected, atol=1e-5)


def test_workspace_reuse_sanity():
    A_norm = torch.randn(4, 4, dtype=torch.float32)
    A_norm = (A_norm @ A_norm.mT) + torch.eye(4) * 0.1
    M_norm = torch.randn(4, 4, dtype=torch.float32)
    abc_t = [(0.1, 0.2, 0.3)]

    Z1, ws = inverse_solve_pe_quadratic_coupled(A_norm, M_norm, abc_t, ws=None)
    Z2, ws_reused = inverse_solve_pe_quadratic_coupled(A_norm, M_norm, abc_t, ws=ws)
    assert ws is ws_reused

    # Test reallocation on different shape
    A_large = torch.randn(5, 5)
    A_large = (A_large @ A_large.mT) + torch.eye(5) * 0.1
    M_large = torch.randn(5, 5)
    Z3, ws_new = inverse_solve_pe_quadratic_coupled(A_large, M_large, abc_t, ws=ws)
    assert ws_new.tmp.shape[-1] == 5


def test_coupled_pe_vs_exact():
    n = 10
    torch.manual_seed(42)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1

    l_target = 0.1
    A_norm, _ = precond_spd(A, mode="frob", l_target=l_target)

    M = torch.eye(n)

    abc_t, _ = build_pe_schedules(
        l_target=l_target,
        device=A_norm.device,
        p_val=2,
        coeff_mode="auto",
        coeff_seed=0,
        coeff_safety=1.0,
        coeff_no_final_safety=False,
    )
    abc_coeffs = [
        (a, b, c)
        for a, b, c in zip(
            abc_t[:, 0].tolist(), abc_t[:, 1].tolist(), abc_t[:, 2].tolist()
        )
    ]

    Z, _ = inverse_solve_pe_quadratic_coupled(A_norm, M, abc_coeffs, p_val=2)
    Z_exact = exact_inverse_proot(A_norm, p_val=2)

    rel_diff = torch.linalg.matrix_norm(Z - Z_exact) / torch.linalg.matrix_norm(Z_exact)
    assert rel_diff < 0.05
