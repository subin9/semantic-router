"""Decision Index engine for a quantized d3 package: the package's own ``d3_engine.D3Engine``, unchanged.

    PYTHONPATH=bench/d3-quant-eval python -m decision_index pipeline --engine quant_engine:QuantEngine \
        --option model=d3-FP8 --out runs/d3-FP8

The only difference from the official run (``--engine d3_engine:D3Engine``) is how the backbone loads: for a
package whose config.json has a ``quantization_config``, the runtime's backbone loader is swapped for the
package's ``d3q.kernels.load_backbone``, so the weights stay quantized on the GPU. Prompting, batching,
options, verification and the readout are the official engine's and runtime's own code. A BF16 package
loads exactly as with ``d3_engine:D3Engine``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


class QuantEngine:
    def __new__(cls, model: str | None = None, device: str = "cuda:0", **options):
        if not model:
            raise ValueError("pass --option model=<package dir>")
        root = Path(model).expanduser().resolve()
        sys.path.insert(0, str(root))
        runtime = importlib.import_module("d3_runtime")
        from d3q.load import is_quantized

        if is_quantized(root):
            from d3q.kernels import load_backbone

            class QuantizedBackbone:
                @staticmethod
                def from_pretrained(path, **kwargs):
                    kwargs.pop("device_map", None)
                    return load_backbone(path, device=device, **kwargs)

            runtime.backbone_class = lambda kind: QuantizedBackbone
        engine = importlib.import_module("d3_engine")
        return engine.D3Engine(model=str(root), device=device, **options)
