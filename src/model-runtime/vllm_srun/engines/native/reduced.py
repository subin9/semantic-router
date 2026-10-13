"""Reduced copies of a backbone's linear layers (the ``max_speed`` profile).

``reduced_view`` returns a view of a backbone that shares every parameter
(embeddings, norms, rotary tables) except its ``nn.Linear`` layers, which the
view holds in a reduced form; the backbone keeps its exact weights, so
``exact`` batches never see the copy. Layers the package's records keep exact
(``DtypePolicy.reduced_exact``) are shared too. Kinds
(``DtypePolicy.reduced_gpu`` / ``reduced_cpu``):

- ``bfloat16``: BF16 weights, run under BF16 autocast (norms, softmax and the
  heads stay FP32): GPUs, and CPUs with BF16 instructions;
- ``int8``: dynamic int8 (per-channel weight scales, per-batch activation
  scales; x86 CPUs);
- ``float32-packed``: FP32 weights reordered once for oneDNN (x86 CPUs,
  ``accel/onednn.py``), for models whose exact path keeps ``F.linear``;
- ``float8``: FP8 weights with one scale per output channel, multiplied with
  activations quantized to FP8 per row at each call (``accel/triton_fp8.py``)
  on the GPU's FP8 matrix units: NVIDIA GPUs of compute capability 8.9 or
  later, AMD Instinct MI300 and MI350 series GPUs; decoders only, whose
  weights are BF16 already.
"""

from __future__ import annotations

import copy
import fnmatch
import importlib.util
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import nn

from ...accel import onednn
from ...accel.onednn import PackedLinear

if TYPE_CHECKING:
    from torch.ao.quantization import QConfig

REDUCED_AUTOCAST = {"bfloat16": torch.bfloat16}
FLOAT8 = "float8"
# The kinds a decoder takes; every kind applies to encoders.
DECODER_KINDS = frozenset({FLOAT8})
# The FP8 format each ROCm architecture's matrix units multiply.
ROCM_FLOAT8 = {"gfx942": torch.float8_e4m3fnuz, "gfx950": torch.float8_e4m3fn}


class Quantizable(Protocol):
    """A float module that ``from_float`` quantizes with the ``qconfig`` set on it."""

    qconfig: QConfig


class QuantizedLinear(Protocol):
    """A dynamic int8 linear layer, whose weight and bias are methods over its packed parameters."""

    def weight(self) -> torch.Tensor: ...

    def bias(self) -> torch.Tensor | None: ...


def bf16_linear(linear: nn.Linear) -> nn.Module:
    layer = nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        layer.weight.copy_(linear.weight)
        if linear.bias is not None:
            layer.bias.copy_(linear.bias)
    return layer.eval()


def int8_linear(linear: nn.Linear) -> nn.Module:
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
    from torch.ao.quantization import per_channel_dynamic_qconfig

    source = copy.copy(linear)
    cast(Quantizable, source).qconfig = per_channel_dynamic_qconfig
    quantized: nn.Module = DynamicLinear.from_float(source)  # type: ignore[no-untyped-call]  # torch leaves from_float unannotated
    return quantized


def float8_format(device: torch.device) -> torch.dtype | None:
    """The FP8 format ``device`` multiplies in ``torch._scaled_mm``; None where it has none."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if torch.version.hip is not None:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        return ROCM_FLOAT8.get(arch)
    if torch.cuda.get_device_capability(device) >= (8, 9):
        return torch.float8_e4m3fn
    return None


class Float8Linear(nn.Module):
    """An ``nn.Linear`` as FP8 weights with one FP32 scale per output channel.

    Each call quantizes the input per row (``accel/triton_fp8.py``) and
    multiplies the two in ``torch._scaled_mm``, which applies both scales and
    returns the input's dtype; the bias, if any, is added in that dtype.
    The FP8 matrix product needs both dimensions to be multiples of 16
    (``fits``); a view shares any other layer exact.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor
    quantize_rows: Callable[
        [torch.Tensor, torch.dtype], tuple[torch.Tensor, torch.Tensor]
    ]

    def __init__(self, linear: nn.Linear):
        super().__init__()
        from ...accel.triton_fp8 import quantize_rows

        fmt = float8_format(linear.weight.device)
        if fmt is None:
            raise ValueError(f"{linear.weight.device} has no FP8 matrix product")
        weight = linear.weight.detach().float()
        largest = torch.finfo(fmt).max
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / largest
        self.register_buffer(
            "weight", (weight / scale).clamp(-largest, largest).to(fmt)
        )
        self.register_buffer("weight_scale", scale.reshape(1, -1).contiguous())
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(linear.bias.detach(), requires_grad=False)
        )
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.quantize_rows = quantize_rows

    @staticmethod
    def fits(linear: nn.Linear) -> bool:
        return linear.in_features % 16 == 0 and linear.out_features % 16 == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.in_features)
        if rows.stride(1) != 1:
            rows = rows.contiguous()
        q, scale = self.quantize_rows(rows, self.weight.dtype)
        out = torch._scaled_mm(
            q,
            self.weight.t(),
            scale_a=scale,
            scale_b=self.weight_scale,
            out_dtype=x.dtype,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], self.out_features)


COPIES: dict[str, Callable[[nn.Linear], nn.Module]] = {
    "bfloat16": bf16_linear,
    "int8": int8_linear,
    "float32-packed": PackedLinear,
    FLOAT8: Float8Linear,
}


def unavailable(
    kind: str, device: torch.device, native_bf16: bool = True
) -> str | None:
    """Why ``kind`` cannot run on ``device``; None when it can.

    ``native_bf16`` is ``DeviceInfo.bf16``: a CPU BF16 copy needs it.
    """
    if kind not in COPIES:
        return f"unknown reduced copy {kind!r} (one of {', '.join(COPIES)})"
    if kind in ("int8", "float32-packed") and device.type != "cpu":
        return f"{kind} copies run on CPUs"
    if kind == FLOAT8:
        if float8_format(device) is None:
            return (
                "float8 copies need an NVIDIA GPU of compute capability 8.9 or later "
                "or an AMD Instinct MI300 or MI350 series GPU"
            )
        if importlib.util.find_spec("triton") is None:
            return (
                "float8 copies quantize activations with Triton, which is not installed"
            )
    if kind == "bfloat16" and device.type == "cpu" and not native_bf16:
        return (
            "this CPU computes BF16 without native instructions (AVX-512 BF16 or AMX)"
        )
    if kind == "float32-packed" and not onednn.available():
        return "oneDNN's packed linear needs an x86 CPU and a PyTorch build with oneDNN"
    if kind == "int8" and "fbgemm" not in torch.backends.quantized.supported_engines:
        return "this PyTorch build has no fbgemm"
    return None


def reduced_view(
    module: nn.Module, kind: str, exact: tuple[str, ...] = ()
) -> nn.Module:
    """``module`` with every ``nn.Linear`` replaced by its ``kind`` copy, every other parameter shared.

    ``exact`` holds ``fnmatch`` patterns of linear layer names within
    ``module`` (``"layers.*.linear_attn.in_proj_qkv"``) that the view shares
    instead, as it does a layer the kind cannot hold (``Float8Linear.fits``).
    """
    reason = unavailable(kind, next(module.parameters()).device)
    if reason:
        raise ValueError(reason)
    copy_of = COPIES[kind]

    def make(name: str, linear: nn.Linear) -> nn.Module:
        if any(fnmatch.fnmatchcase(name, p) for p in exact) or (
            kind == FLOAT8 and not Float8Linear.fits(linear)
        ):
            return linear
        return copy_of(linear)

    with torch.inference_mode(False), torch.no_grad():
        return _view(module, make, "")


def _view(
    module: nn.Module, make: Callable[[str, nn.Linear], nn.Module], name: str
) -> nn.Module:
    if isinstance(module, nn.Linear):
        return make(name, module)
    view = copy.copy(module)
    view._parameters = dict(module._parameters)
    view._buffers = dict(module._buffers)
    view._modules = type(module._modules)(
        (
            child_name,
            (
                None
                if child is None
                else _view(child, make, f"{name}.{child_name}" if name else child_name)
            ),
        )
        for child_name, child in module._modules.items()
    )
    return view


def copy_counts(view: nn.Module, source: nn.Module) -> dict[str, int]:
    """How many of the view's linear layers are reduced copies and how many it shares with ``source``."""
    shared = {id(m) for m in source.modules() if isinstance(m, nn.Linear)}
    counts = {"reduced": 0, "exact": 0}
    for layer in view.modules():
        if id(layer) in shared:
            counts["exact"] += 1
        elif isinstance(layer, (nn.Linear, PackedLinear, Float8Linear)) or callable(
            getattr(layer, "weight", None)
        ):
            counts["reduced"] += 1
    return counts


def linear_bytes(module: nn.Module, source: nn.Module | None = None) -> int:
    """Bytes the view's reduced linear layers hold (layers shared with ``source`` excluded)."""
    shared = set() if source is None else {id(m) for m in source.modules()}
    total = 0
    for layer in module.modules():
        if id(layer) in shared:
            continue
        if isinstance(layer, PackedLinear):
            tensors: list[torch.Tensor | None] = [layer.packed, layer.bias]
        elif isinstance(layer, Float8Linear):
            tensors = [layer.weight, layer.weight_scale, layer.bias]
        elif isinstance(layer, nn.Linear):
            tensors = [layer.weight, layer.bias]
        elif callable(getattr(layer, "weight", None)):
            quantized = cast(QuantizedLinear, layer)
            tensors = [quantized.weight(), quantized.bias()]
        else:
            continue
        total += sum(t.numel() * t.element_size() for t in tensors if t is not None)
    return total
