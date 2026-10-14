"""An FP8 (W8A8, FP8_DYNAMIC) d3 package in the compressed-tensors format, built on the CPU.

usage: fp8_package.py MODEL_DIR OUT_DIR --runtime SRC/model-runtime [--keep-bf16 PATTERN ...]

Every targeted projection becomes an E4M3 weight with one scale per output channel (max |w| / 448,
rounded to BF16 before the weight is divided by it), and the config asks for FP8 activations per token
at run time. ``--keep-bf16`` adds compressed-tensors ``ignore`` patterns: by default the GDN input
projections (``in_proj_qkv``, ``in_proj_z``), which the runtime's FP8 records kept exact. The vision
tower, ``in_proj_a`` / ``in_proj_b`` and everything that is not a projection stay as published.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from awq import TARGETS, finish_package

FP8_MAX = 448.0
IGNORE = [r"re:.*visual\..*", r"re:.*linear_attn\.in_proj_(a|b)$"]
KEEP = [r"re:.*linear_attn\.in_proj_(qkv|z)$"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("out")
    p.add_argument("--runtime", required=True)
    p.add_argument("--keep-bf16", nargs="*", default=KEEP)
    args = p.parse_args()
    root, out = Path(args.root), Path(args.out)
    sys.path.insert(0, args.runtime)
    from vllm_srun.families.decision3 import package as pkg
    from vllm_srun.quantization import QuantizedWeights

    holds = QuantizedWeights(
        "float8", tuple(TARGETS), tuple(IGNORE + args.keep_bf16)
    ).holds
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors, quantized = {}, 0
    for file in sorted(set(index.values())):
        with safe_open(str(root / file), framework="pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                layer = name.removesuffix(".weight")
                if not (name.endswith(".weight") and tensor.ndim == 2 and holds(layer)):
                    tensors[name] = tensor
                    continue
                w = tensor.float()
                scale = (w.abs().amax(1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
                scale = scale.to(torch.bfloat16)
                tensors[name] = (
                    (w / scale.float()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                )
                tensors[f"{layer}.weight_scale"] = scale
                quantized += 1
    save_file(tensors, str(out / "model.safetensors"))
    for path in root.iterdir():
        if path.name in (
            "model.safetensors.index.json",
            pkg.MANIFEST_NAME,
            "__pycache__",
        ) or (path.is_dir() and path.name.startswith(".")):
            continue
        if not path.name.startswith("model-"):
            (shutil.copytree if path.is_dir() else shutil.copy)(path, out / path.name)
    config = json.loads((root / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": TARGETS,
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
                "output_activations": None,
            }
        },
        "ignore": IGNORE + args.keep_bf16,
    }
    finish_package(root, out, config, pkg)
    print(out, "quantized", quantized, "projections")


if __name__ == "__main__":
    main()
