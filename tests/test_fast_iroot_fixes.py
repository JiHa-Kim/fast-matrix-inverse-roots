import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from fast_iroot.utils import _addmm_into
from fast_iroot.coupled import inverse_solve_pe_quadratic_coupled
from fast_iroot.metrics import isqrt_relative_error


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
