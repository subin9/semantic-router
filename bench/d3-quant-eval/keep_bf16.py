"""An AWQ package with some decoder layers restored to the BF16 checkpoint's tensors.

usage: keep_bf16.py AWQ_DIR BF16_DIR OUT_DIR LAYER[,LAYER...] --runtime SRC/model-runtime

AWQ quantizes each layer from the BF16 model's inputs and folds its scales only into that layer's own
tensors, so a layer can be put back as published without touching the others. The restored layers are
added to ``ignore`` in the quantization_config.
"""

import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from awq import write_manifest


def main():
    awq, bf16, out = (Path(a) for a in sys.argv[1:4])
    layers = sorted(int(i) for i in sys.argv[4].split(","))
    runtime = sys.argv[sys.argv.index("--runtime") + 1]
    sys.path.insert(0, runtime)
    from vllm_srun.families.decision3 import package as pkg

    shutil.copytree(
        awq, out, ignore=shutil.ignore_patterns("model.safetensors", pkg.MANIFEST_NAME)
    )
    index = json.loads((bf16 / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    restored = {
        name: file
        for name, file in index.items()
        if any(f".layers.{i}." in name and "visual" not in name for i in layers)
    }
    tensors = {}
    with safe_open(str(awq / "model.safetensors"), framework="pt") as f:
        for name in f.keys():
            if not any(
                f".layers.{i}." in name and "visual" not in name for i in layers
            ):
                tensors[name] = f.get_tensor(name)
    for name, file in restored.items():
        with safe_open(str(bf16 / file), framework="pt") as f:
            tensors[name] = f.get_tensor(name)
    save_file(tensors, str(out / "model.safetensors"))
    config = json.loads((out / "config.json").read_text())
    config["quantization_config"]["ignore"].append(
        r"re:(.*\.)?layers\.(%s)\..*" % "|".join(str(i) for i in layers)
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_manifest(bf16, out, pkg)
    print(out, "restored", len(restored), "tensors of layers", layers)


if __name__ == "__main__":
    main()
