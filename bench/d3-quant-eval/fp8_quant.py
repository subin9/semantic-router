"""Per-row FP8 quantization in one Triton kernel: scale = amax(|x|) / fp8 max, q = x / scale."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rows_kernel(
    x_ptr, q_ptr, s_ptr, n_cols, stride_x, stride_q, fmax, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)  # rows * stride passes 2**31 on long prompts
    x_row = x_ptr + row * stride_x
    amax = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    scale = tl.maximum(tl.max(amax, axis=0), 1e-12) / fmax
    tl.store(s_ptr + row, scale)
    inv = 1.0 / scale
    q_row = q_ptr + row * stride_q
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(x * inv, -fmax), fmax)
        tl.store(q_row + cols, q.to(q_ptr.dtype.element_ty), mask=cols < n_cols)


def quant_rows(
    x: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] (contiguous rows) -> (q [M, K] in ``dtype``, scale [M, 1] float32)."""
    m, k = x.shape
    q = torch.empty((m, k), device=x.device, dtype=dtype)
    s = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    block = min(triton.next_power_of_2(k), 4096)
    _rows_kernel[(m,)](
        x,
        q,
        s,
        k,
        x.stride(0),
        q.stride(0),
        torch.finfo(dtype).max,
        BLOCK=block,
        num_warps=8,
    )
    return q, s
