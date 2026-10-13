"""Per-layer sensitivity of d3 to FP8 or AWQ int4, on dev rows (never the test rows), then keep-in-BF16 sweeps.

usage: sensitivity.py MODEL_DIR CALIB_JSON DEV_JSON OUT_JSON MODE
For each decoder layer alone: quantize its projections, answer the dev rows, compare with BF16
(mean KL of the answer distribution, rows whose top answer changes). Then quantize every layer
except the k most sensitive, for several k, and report the same on the dev rows.
"""

import json
import math
import re
import sys

import torch

import quant_eval as qe


def answers(model, rows):
    out = []
    with torch.no_grad():
        for row in rows:
            ans = model.system_one(
                state=row["state"], questions={"q": qe.question(row)}
            )["answers"]["q"]
            probs = (
                ans["probabilities"]
                if "probabilities" in ans
                else {"false": 1 - ans["noul"], "true": ans["noul"]}
            )
            out.append(list(probs.values()))
    return out


def compare(base, other):
    kl, flips = 0.0, 0
    for p, q in zip(base, other):
        kl += sum(
            a * math.log(max(a, 1e-12) / max(b, 1e-12)) for a, b in zip(p, q) if a > 0
        )
        flips += p.index(max(p)) != q.index(max(q))
    return {"kl": kl / len(base), "flips": flips, "agreement": 1 - flips / len(base)}


def main():
    root, calib_path, dev_path, out_path, mode = sys.argv[1:6]
    dev = [r for r in json.load(open(dev_path)) if qe.usable(r)]
    from transformers import AutoModel

    model = AutoModel.from_pretrained(root, trust_remote_code=True, device="cuda:0")
    lm = model.backbone.language_model
    n_layers = len(lm.layers)
    names = [n for n, _ in qe.linears(lm, None)]
    by_layer = {
        i: [n for n in names if n.startswith(f"layers.{i}.")] for i in range(n_layers)
    }

    # the quantized replacement of every projection, built once
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
                quant[n] = qe.Int4Linear(lin, s.to(lin.weight.device))
        del grouping, stats
    original = {n: lm.get_submodule(n) for n in names}

    def apply(selected):
        for n in names:
            qe.replace(lm, n, quant[n] if n in selected else original[n])

    apply(set())
    base = answers(model, dev)
    report = {"mode": mode, "dev_rows": len(dev), "layers": {}, "sweep": {}}
    for i in range(n_layers):
        apply(set(by_layer[i]))
        report["layers"][i] = compare(base, answers(model, dev))
        print("layer", i, json.dumps(report["layers"][i]), flush=True)
    order = sorted(range(n_layers), key=lambda i: -report["layers"][i]["kl"])
    report["order"] = order
    for k in (0, 1, 2, 4, 6, 8, 12, 16):
        keep = set(order[:k])
        apply({n for i in range(n_layers) if i not in keep for n in by_layer[i]})
        res = compare(base, answers(model, dev))
        res["keep"] = sorted(keep)
        res["keep_regex"] = (
            r"^layers\.(%s)\." % "|".join(str(i) for i in sorted(keep))
            if keep
            else None
        )
        report["sweep"][k] = res
        print("keep", k, json.dumps(res), flush=True)
    apply(set())
    json.dump(report, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
