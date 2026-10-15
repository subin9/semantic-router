"""Linear layers for d3's quantized checkpoints (compressed-tensors), and the loader that installs them.

transformers loads compressed-tensors weights without the d3 runtime's assumptions holding (it reads BF16
``Linear`` weights), so ``load_backbone`` loads the checkpoint compressed and replaces every quantized layer:

- ``Int4Linear``: ``pack-quantized`` int4 weights in groups, symmetric (GPTQ) or with zero points (AWQ), kept
  packed on the GPU. Each call unpacks the weight to BF16 with one Triton kernel and multiplies in BF16, which
  computes exactly what a BF16 layer holding the dequantized weight computes. It saves memory, not time.
- ``Float8Linear``: ``float-quantized`` FP8 weights with one scale per output channel (FP8_DYNAMIC). Each call
  quantizes the activations to FP8 per row with one Triton kernel and multiplies in ``torch._scaled_mm``. It
  needs FP8 matrix units: AMD Instinct MI300 / MI350 series, or NVIDIA compute capability 8.9 or later. On MI300
  (gfx942), whose units multiply ``float8_e4m3fnuz``, the stored E4M3 bits are reinterpreted and the scales
  doubled, which keeps every value. On Ada (8.9) the per-row scaled product takes a slow kernel from 256 rows
  on, so there the product is taken unscaled in FP32 and the scales applied after.

These are the layers of the vLLM Semantic Router model runtime (``engines/native/quantized.py``).
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from torch import nn

PACK = 8
UNPACK_BLOCK = 256
ROCM_FLOAT8 = {"gfx942": torch.float8_e4m3fnuz, "gfx950": torch.float8_e4m3fn}


@triton.jit
def _unpack_int4(
    packed_ptr,
    scale_ptr,
    zero_ptr,
    out_ptr,
    n_words,
    words_per_row,
    groups_per_row,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
    ZERO: tl.constexpr,
):
    words = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = words < n_words
    word = tl.load(packed_ptr + words, mask=mask, other=0)
    row = words // words_per_row
    column = (words % words_per_row) * 8
    scale = tl.load(
        scale_ptr + row * groups_per_row + column // GROUP, mask=mask, other=0.0
    ).to(tl.float32)
    nibbles = (word[:, None] >> (tl.arange(0, 8) * 4)[None, :]) & 0xF
    values = (nibbles - 8).to(tl.float32)
    if ZERO:
        zero_word = tl.load(
            zero_ptr + (row // 8) * groups_per_row + column // GROUP, mask=mask, other=0
        )
        values -= (((zero_word >> ((row % 8) * 4)) & 0xF) - 8).to(tl.float32)[:, None]
    values = values * scale[:, None]
    out = words[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(out_ptr + out, values.to(out_ptr.dtype.element_ty), mask=mask[:, None])


@triton.jit
def _quantize_rows(
    x_ptr, q_ptr, s_ptr, n_cols, stride_x, stride_q, fmax, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)  # rows * stride passes 2**31 on long prompts
    x_row = x_ptr + row * stride_x
    amax = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        amax = tl.maximum(
            amax,
            tl.abs(tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)),
        )
    scale = tl.maximum(tl.max(amax, axis=0), 1e-12) / fmax
    tl.store(s_ptr + row, scale)
    inv = 1.0 / scale
    q_row = q_ptr + row * stride_q
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(x * inv, -fmax), fmax)
        tl.store(q_row + cols, q.to(q_ptr.dtype.element_ty), mask=cols < n_cols)


def float8_format(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if torch.version.hip is not None:
        return ROCM_FLOAT8.get(
            torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        )
    return (
        torch.float8_e4m3fn
        if torch.cuda.get_device_capability(device) >= (8, 9)
        else None
    )


class Int4Linear(nn.Module):
    def __init__(self, packed, scale, zero, in_features, out_features, bias=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.group_size = in_features // scale.shape[1]
        self.register_buffer("weight_packed", packed.to(torch.int32).contiguous())
        self.register_buffer("weight_scale", scale.contiguous())
        self.register_buffer(
            "weight_zero_point",
            None if zero is None else zero.to(torch.int32).contiguous(),
        )
        self.bias = bias

    def unpacked(self) -> torch.Tensor:
        rows, words = self.weight_packed.shape
        out = torch.empty(
            rows, words * PACK, dtype=torch.bfloat16, device=self.weight_packed.device
        )
        n = self.weight_packed.numel()
        zero = self.weight_zero_point
        _unpack_int4[(triton.cdiv(n, UNPACK_BLOCK),)](
            self.weight_packed,
            self.weight_scale,
            self.weight_packed if zero is None else zero,
            out,
            n,
            words,
            self.weight_scale.shape[1],
            GROUP=self.group_size,
            BLOCK=UNPACK_BLOCK,
            ZERO=zero is not None,
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.unpacked().to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, int4 g{self.group_size}"
        )


class Float8Linear(nn.Module):
    def __init__(self, weight, scale, in_features, out_features, bias=None):
        super().__init__()
        fmt = float8_format(weight.device)
        if fmt is None:
            raise RuntimeError(
                "FP8 weights need an AMD Instinct MI300 / MI350 or an NVIDIA GPU of compute "
                "capability 8.9 or later"
            )
        scale = scale.float().reshape(out_features, 1)
        if fmt == torch.float8_e4m3fnuz and weight.dtype == torch.float8_e4m3fn:
            bits = weight.view(torch.int8)
            weight = torch.where(bits == -128, torch.zeros_like(bits), bits).view(
                torch.float8_e4m3fnuz
            )
            scale = scale * 2  # the same bits mean half the value in e4m3fnuz
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer("weight_scale", scale.t().contiguous())
        self.unscaled = torch.version.hip is None and torch.cuda.get_device_capability(
            weight.device
        ) == (8, 9)
        self.register_buffer("one", torch.ones((), device=weight.device))
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        rows = x.reshape(-1, shape[-1]).contiguous()
        q = torch.empty(rows.shape, device=rows.device, dtype=self.weight.dtype)
        s = torch.empty(rows.shape[0], 1, device=rows.device, dtype=torch.float32)
        if rows.shape[0]:
            block = min(triton.next_power_of_2(rows.shape[1]), 4096)
            _quantize_rows[(rows.shape[0],)](
                rows,
                q,
                s,
                rows.shape[1],
                rows.stride(0),
                q.stride(0),
                torch.finfo(self.weight.dtype).max,
                BLOCK=block,
                num_warps=8,
            )
        if self.unscaled:
            out = torch._scaled_mm(
                q,
                self.weight.t(),
                scale_a=self.one,
                scale_b=self.one,
                out_dtype=torch.float32,
            )
            out = (out * s * self.weight_scale).to(x.dtype)
        else:
            out = torch._scaled_mm(
                q,
                self.weight.t(),
                scale_a=s,
                scale_b=self.weight_scale,
                out_dtype=x.dtype,
            )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, fp8"


def _tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: t
        for name, t in (
            *module.named_parameters(recurse=False),
            *module.named_buffers(recurse=False),
        )
        if t is not None
    }


def convert(model: nn.Module) -> dict[str, int]:
    """Replace every compressed module of a ``dequantize=False`` load with its kernel layer."""
    counts = {"Int4Linear": 0, "Float8Linear": 0}
    for name, module in list(model.named_modules()):
        t = _tensors(module)
        if "weight_packed" in t:
            shape = t.get("weight_shape")
            out_features, in_features = (
                (int(shape[0]), int(shape[1]))
                if shape is not None
                else (t["weight_packed"].shape[0], t["weight_packed"].shape[1] * PACK)
            )
            new = Int4Linear(
                t["weight_packed"].data,
                t["weight_scale"].data,
                t.get("weight_zero_point"),
                in_features,
                out_features,
                t.get("bias"),
            )
        elif (
            "weight" in t
            and t["weight"].dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            and "weight_scale" in t
        ):
            out_features, in_features = t["weight"].shape
            new = Float8Linear(
                t["weight"].data,
                t["weight_scale"].data,
                in_features,
                out_features,
                t.get("bias"),
            )
        else:
            continue
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, new)
        counts[type(new).__name__] += 1
    return counts


def load_backbone(path: str, device: str = "cuda:0", **kwargs: Any):
    """A quantized ``Qwen3_5Model`` with its weights kept quantized and kernel layers installed."""
    from transformers import CompressedTensorsConfig, Qwen3_5Model

    kwargs.pop("quantization_config", None)
    kwargs.setdefault("dtype", torch.bfloat16)
    kwargs.setdefault("device_map", {"": device})
    model = Qwen3_5Model.from_pretrained(
        path, quantization_config=CompressedTensorsConfig(dequantize=False), **kwargs
    )
    model.quantized_modules = convert(model)
    if not any(model.quantized_modules.values()):
        raise RuntimeError(
            "no quantized layer found: the checkpoint did not load compressed"
        )
    return model
