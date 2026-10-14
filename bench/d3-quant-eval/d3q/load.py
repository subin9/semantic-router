"""Load a quantized d3 package (FP8 or int4) through the package's own ``d3_runtime.D3``.

The runtime's backbone loader is swapped for ``d3q.kernels.load_backbone``, so the weights stay quantized on
the GPU; prompting, the noncausal prompt encoder, the readout, images and videos are the unmodified runtime.

    import sys; sys.path.insert(0, "d3-FP8")
    from d3q.load import load_d3
    model = load_d3("d3-FP8", device="cuda:0")
    model.system_one(state=..., questions=...)

The same interface as TheoHsiung/d3-W4A16-G128's ``d3q`` (Apache-2.0), whose structure this follows.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def runtime(root: Path):
    """The package's own ``d3_runtime`` module (never a copy)."""
    root_text = str(Path(root).resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module("d3_runtime")


def is_quantized(path: Path) -> bool:
    return "quantization_config" in json.loads((Path(path) / "config.json").read_text())


def load_d3(
    path: str | Path, device: str = "cuda:0", batch_size: int = 8, **kwargs: Any
):
    # Prompts of very different lengths fragment the caching allocator; set before CUDA starts.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    path = Path(path).expanduser()
    rt = runtime(path)
    if is_quantized(path):
        from d3q.kernels import load_backbone

        class QuantizedBackbone:
            @staticmethod
            def from_pretrained(root: str, **options: Any):
                options.pop("device_map", None)
                return load_backbone(root, device=device, **options)

        rt.backbone_class = lambda kind: QuantizedBackbone
    return rt.D3.from_pretrained(
        str(path), device=device, batch_size=batch_size, **kwargs
    )
