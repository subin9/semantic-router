"""Sensitivity of d3 to FP8 or AWQ int4 by projection type, on dev rows (never the test rows).

usage: type_sens.py MODEL_DIR CALIB_JSON DEV_JSON OUT_JSON MODE [--keep-layers 1,2,...]
Configurations: every projection quantized; each type alone quantized; every type but one quantized.
``--keep-layers`` keeps those decoder layers in BF16 throughout (from sensitivity.py).
"""

import json
import sys

import torch

import quant_eval as qe
from sensitivity import answers, compare

TYPES = {
    "mlp_gate_up": ("gate_proj", "up_proj"),
    "mlp_down": ("down_proj",),
    "attn_qkv": ("q_proj", "k_proj", "v_proj"),
    "attn_out": ("o_proj",),
    "gdn_in": ("in_proj_qkv", "in_proj_z"),
    "gdn_gates": ("in_proj_a", "in_proj_b"),
    "gdn_out": ("out_proj",),
}


def main():
    root, calib_path, dev_path, out_path, mode = sys.argv[1:6]
    keep_layers = set()
    if "--keep-layers" in sys.argv:
        keep_layers = {
            int(i)
            for i in sys.argv[sys.argv.index("--keep-layers") + 1].split(",")
            if i
        }
    dev = [r for r in json.load(open(dev_path)) if qe.usable(r)]
    from transformers import AutoModel

    model = AutoModel.from_pretrained(root, trust_remote_code=True, device="cuda:0")
    lm = model.backbone.language_model
    names = [
        n for n, _ in qe.linears(lm, None) if int(n.split(".")[1]) not in keep_layers
    ]
    kind = {
        n: next(t for t, leaves in TYPES.items() if n.rsplit(".", 1)[1] in leaves)
        for n in names
    }
    quant = {}
    if mode == "fp8":
        for n in names:
            quant[n] = qe.FP8Linear(lm.get_submodule(n))
    else:
        calib = [r for r in json.load(open(calib_path)) if qe.usable(r)]
        grouping = qe.groups(lm, None)
        stats = qe.calibrate(model, lm, grouping, calib)
        for i, members in enumerate(grouping):
            s, _, _ = qe.awq_scale(members, stats.pop(i))
            for n, lin in members:
                if n in kind:
                    quant[n] = qe.Int4Linear(lin, s.to(lin.weight.device))
        del grouping, stats
    original = {n: lm.get_submodule(n) for n in names}

    def apply(selected):
        for n in names:
            qe.replace(lm, n, quant[n] if n in selected else original[n])

    apply(set())
    base = answers(model, dev)
    report = {
        "mode": mode,
        "dev_rows": len(dev),
        "keep_layers": sorted(keep_layers),
        "configs": {},
    }
    configs = {"all": set(names)}
    for t in TYPES:
        configs[f"only:{t}"] = {n for n in names if kind[n] == t}
        configs[f"all_but:{t}"] = {n for n in names if kind[n] != t}
    for label, selected in configs.items():
        if not selected:
            continue
        apply(selected)
        report["configs"][label] = compare(base, answers(model, dev))
        print(label, json.dumps(report["configs"][label]), flush=True)
    apply(set())
    json.dump(report, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
