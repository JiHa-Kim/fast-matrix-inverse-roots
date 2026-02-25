from typing import Optional
import torch


def _check_square(A: torch.Tensor) -> None:
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")


def _validate_p_val(p_val: int) -> None:
    if not isinstance(p_val, int) or p_val <= 0:
        raise ValueError("p_val must be a positive integer")


@torch.no_grad()
def _symmetrize_inplace(M: torch.Tensor, tmp: Optional[torch.Tensor] = None) -> None:
    if tmp is None:
        M.copy_(0.5 * (M + M.mT))
        return
    tmp.copy_(M.mT)
    M.add_(tmp).mul_(0.5)


@torch.no_grad()
def _matmul_into(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    torch.matmul(A, B, out=out)
    return out


@torch.no_grad()
def _addmm_into(
    bias: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float,
    alpha: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """out = beta * bias + alpha * (mat1 @ mat2).  Fused BLAS call."""
    if mat1.dim() == 2:
        torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha, out=out)
    elif mat1.dim() == 3:
        torch.baddbmm(bias, mat1, mat2, beta=beta, alpha=alpha, out=out)
    else:
        batch_shape = mat1.shape[:-2]
        n, m = mat1.shape[-2], mat2.shape[-1]
        k = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) else 1
        outv = out.reshape(k, n, m)
        biasv = bias.reshape(k, n, m)
        m1v = mat1.reshape(k, n, mat1.shape[-1])
        m2v = mat2.reshape(k, mat2.shape[-2], m)
        torch.baddbmm(biasv, m1v, m2v, beta=beta, alpha=alpha, out=outv)
    return out


@torch.no_grad()
def _bpow_times_y(
    B: torch.Tensor,
    Y: torch.Tensor,
    p: int,
    out: torch.Tensor,
    tmp1: torch.Tensor,
    tmp2: torch.Tensor,
) -> None:
    """Compute B^p * Y into `out` using O(log p) matmuls (binary exponentiation).

    B, Y are inputs. tmp1, tmp2 are scratch buffers (same shape).
    `out`, `tmp1`, and `tmp2` must not alias `B` or `Y`.
    """
    if p <= 0:
        out.copy_(Y)
        return
    if p == 1:
        torch.matmul(B, Y, out=out)
        return
    if p == 2:
        torch.matmul(B, Y, out=tmp1)
        torch.matmul(B, tmp1, out=out)
        return
    if p == 4:
        # B^2 -> tmp1, B^2*Y -> tmp2, B^4*Y -> out  (3 matmuls)
        torch.matmul(B, B, out=tmp1)
        torch.matmul(tmp1, Y, out=tmp2)
        torch.matmul(tmp1, tmp2, out=out)
        return

    # General binary exponentiation for p >= 3
    bits = [(p >> i) & 1 for i in range(p.bit_length())]

    cur_base = B
    cur_res = Y

    for i, bit in enumerate(bits):
        if bit:
            for buf in (out, tmp1, tmp2):
                if buf is not cur_base and buf is not cur_res:
                    next_res = buf
                    break
            torch.matmul(cur_base, cur_res, out=next_res)
            cur_res = next_res

        if i < len(bits) - 1:
            for buf in (out, tmp1, tmp2):
                if buf is not cur_base and buf is not cur_res:
                    next_base = buf
                    break
            torch.matmul(cur_base, cur_base, out=next_base)
            cur_base = next_base

    if cur_res is not out:
        out.copy_(cur_res)
