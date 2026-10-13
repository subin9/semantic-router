"""AWQ for d3: activation-aware int4 weights (W4A16, group 128, asymmetric), written as a compressed-tensors package.

usage: awq.py MODEL_DIR CALIB_JSON OUT_DIR --runtime SRC/model-runtime [--loss-rows 64] [--clip-tokens 4096] [--rtn]

The algorithm of AWQ (Lin et al., 2024) as AutoAWQ runs it, adapted to Qwen3.5's layers:

1. The calibration rows run through d3 once; the inputs of the first decoder layer are kept.
2. Layer by layer: the layer's BF16 output on every row, and the inputs of its projections, are recorded.
3. Per group of projections that read one input, a per-input-channel scale s = x^a / w^(1-a) (x the mean
   |activation|, w the mean group-normalised |weight| per channel) is searched over a in [0, 1) to minimise
   the squared error of the layer's output with those projections quantized as W * s, inputs / s. The scale
   is folded into what produces the input, so the BF16 model is unchanged:
   - input_layernorm (Qwen3.5 norms multiply by 1 + w) -> q/k/v_proj, or in_proj_qkv/z/b/a (b and a stay BF16)
   - post_attention_layernorm -> gate_proj, up_proj
   - up_proj (rows) -> down_proj
   GDN's out_proj (after a gated norm shared across heads) and attention's o_proj (after the output gate)
   have no linear producer to fold into, so they take no scale.
4. Per projection except q/k_proj, the clipping range of each output row and group is shrunk over 20 steps to
   minimise the error of its output on sampled tokens.
5. The layer's projections are quantized (asymmetric int4, zero points), and the next layer reads the
   BF16 layer's output.

The package keeps every file of MODEL_DIR, one ``model.safetensors`` with the projections as
``weight_packed`` / ``weight_scale`` / ``weight_zero_point`` / ``weight_shape``, ``quantization_config``
in config.json and a ``MODEL_MANIFEST.json`` computed as the model runtime verifies it (``--runtime``).
"""

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn

import quant_eval as qe

GROUP = 128
GRID = 20
CLIP_STEPS = 20
CLIP_MIN = 0.5
TARGETS = [
    r"re:(.*\.)?layers\.\d+\.mlp\.(gate|up|down)_proj$",
    r"re:(.*\.)?layers\.\d+\.self_attn\.(q|k|v|o)_proj$",
    r"re:(.*\.)?layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
]
QUANTIZED = (
    "gate_proj",
    "up_proj",
    "down_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
)


def fake_quant(w, group=GROUP):
    """Asymmetric int4 round trip per group of input columns: (dequantized, q, scale, zero), q and zero in 0..15."""
    n, k = w.shape
    g = w.float().reshape(n, k // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp(min=1e-5)
    zero = (-lo / scale).round().clamp(0, 15)
    q = ((g / scale).round() + zero).clamp(0, 15)
    return ((q - zero) * scale).reshape(n, k), q, scale, zero


def layer_inputs(model, rows):
    """Per calibration row: the first decoder layer's hidden states, and every layer's other arguments
    (a GDN layer and a full-attention layer get different masks)."""
    layers = model.backbone.language_model.layers
    hidden, kwargs = [], []

    def hook(index):
        def fn(_, args, kw):
            if index == 0:
                hidden.append((args[0] if args else kw["hidden_states"]).detach())
                kwargs.append({})
            kw = {k: v for k, v in kw.items() if k != "hidden_states"}
            kw["past_key_values"] = None
            kwargs[-1][index] = kw

        return fn

    handles = [
        m.register_forward_pre_hook(hook(i), with_kwargs=True)
        for i, m in enumerate(layers)
    ]
    with torch.no_grad():
        for row in rows:
            model.system_one(state=row["state"], questions={"q": qe.question(row)})
    for h in handles:
        h.remove()
    return hidden, kwargs


def run_layer(layer, inputs):
    """``inputs``: (hidden_states, kwargs) per row."""
    outs = []
    with torch.no_grad():
        for hidden, kwargs in inputs:
            out = layer(hidden, **kwargs)
            outs.append(out[0] if isinstance(out, tuple) else out)
    return outs


def record(layer, inputs, names):
    """Inputs of the named projections over every row, flattened to [tokens, features]."""
    feats = {n: [] for n in names}
    hooks = [
        layer.get_submodule(n).register_forward_pre_hook(
            lambda _, args, n=n: feats[n].append(
                args[0].detach().reshape(-1, args[0].shape[-1])
            )
        )
        for n in names
    ]
    outs = run_layer(layer, inputs)
    for h in hooks:
        h.remove()
    return {n: torch.cat(v) for n, v in feats.items()}, outs


def loss(layer, inputs, want):
    got = run_layer(layer, inputs)
    return sum(
        (a.float() - b.float()).pow(2).sum().item() for a, b in zip(got, want)
    ) / sum(w.numel() for w in want)


def search_scale(layer, linears, x, inputs, want):
    """The scale over the projections' input channels that minimises the layer's output error."""
    weights = [layer.get_submodule(n).weight for n in linears]
    originals = [w.data.clone() for w in weights]
    w = torch.cat([o.float() for o in originals])
    g = w.reshape(w.shape[0], -1, GROUP).abs()
    w_mean = (g / g.amax(-1, keepdim=True).clamp(min=1e-8)).reshape(w.shape).mean(0)
    x_mean = x.abs().float().mean(0)
    best = (float("inf"), None, None)
    for i in range(GRID):
        a = i / GRID
        s = (x_mean.pow(a) / w_mean.pow(1 - a).clamp(min=1e-4)).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        for weight, original in zip(weights, originals):
            weight.data = (fake_quant(original.float() * s)[0] / s).to(original.dtype)
        err = loss(layer, inputs, want)
        if err < best[0]:
            best = (err, a, s)
    for weight, original in zip(weights, originals):
        weight.data = original
    return best


def fold(layer, producer, consumers, s):
    """Divide what ``producer`` emits by ``s`` and multiply the consumers' input columns by it."""
    p = layer.get_submodule(producer)
    if producer.endswith("layernorm"):  # Qwen3.5 RMSNorm: x * (1 + w)
        p.weight.data = ((1 + p.weight.float()) / s - 1).to(p.weight.dtype)
    else:  # a linear's output rows
        p.weight.data = (p.weight.float() / s[:, None]).to(p.weight.dtype)
    for name in consumers:
        c = layer.get_submodule(name)
        c.weight.data = (c.weight.float() * s).to(c.weight.dtype)


def search_clip(weight, x, tokens):
    """Per output row and group, the clipping bound (shrunk max |w|) that minimises the output error."""
    x = x[torch.randperm(x.shape[0], device=x.device)[:tokens]].float()
    n, k = weight.shape
    groups = k // GROUP
    xg = x.reshape(-1, groups, GROUP)
    best_max = torch.empty(n, groups, 1, device=weight.device)
    for start in range(0, n, 256):
        w = weight[start : start + 256].float().reshape(-1, groups, GROUP)
        ref = torch.einsum("tgk,ogk->tog", xg, w)
        org_max = w.abs().amax(-1, keepdim=True)
        best_err = torch.full(org_max.shape, float("inf"), device=w.device)
        best = org_max.clone()
        for i in range(int(CLIP_STEPS * (1 - CLIP_MIN))):
            max_val = org_max * (1 - i / CLIP_STEPS)
            clipped = w.clamp(-max_val, max_val).reshape(w.shape[0], -1)
            q = fake_quant(clipped)[0].reshape(w.shape)
            err = (torch.einsum("tgk,ogk->tog", xg, q) - ref).pow(2).mean(0)[..., None]
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best = torch.where(better, max_val, best)
        best_max[start : start + 256] = best
    return best_max


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("calib")
    p.add_argument("out")
    p.add_argument(
        "--runtime",
        required=True,
        help="src/model-runtime of the semantic-router checkout",
    )
    p.add_argument("--loss-rows", type=int, default=64)
    p.add_argument("--clip-tokens", type=int, default=4096)
    p.add_argument(
        "--rtn",
        action="store_true",
        help="round to nearest only: no scales, no clipping (baseline)",
    )
    args = p.parse_args()
    torch.manual_seed(0)
    root, out = Path(args.root), Path(args.out)
    rows = [r for r in json.load(open(args.calib)) if qe.usable(r)]
    from transformers import AutoModel

    model = AutoModel.from_pretrained(root, trust_remote_code=True, device="cuda:0")
    lm = model.backbone.language_model
    hidden, row_kwargs = layer_inputs(model, rows)
    packed = {}
    report = {"rows": len(rows), "layers": []}
    t0 = time.time()
    for index, layer in enumerate(lm.layers):
        inputs = [(h, kw[index]) for h, kw in zip(hidden, row_kwargs)]
        loss_inputs = inputs[: args.loss_rows]
        names = [
            n
            for n, m in layer.named_modules()
            if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in QUANTIZED
        ]
        attn = "self_attn" if hasattr(layer, "self_attn") else "linear_attn"
        feats, outs = record(layer, inputs, names)
        want = outs[: args.loss_rows]
        groups = (
            [
                (
                    "input_layernorm",
                    [f"{attn}.q_proj", f"{attn}.k_proj", f"{attn}.v_proj"],
                    [],
                )
            ]
            if attn == "self_attn"
            else [
                (
                    "input_layernorm",
                    [f"{attn}.in_proj_qkv", f"{attn}.in_proj_z"],
                    [f"{attn}.in_proj_b", f"{attn}.in_proj_a"],
                )
            ]
        ) + [
            ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"], []),
            ("mlp.up_proj", ["mlp.down_proj"], []),
        ]
        info = {"layer": index}
        for producer, linears, also in [] if args.rtn else groups:
            x = feats[linears[0]]
            err, a, s = search_scale(layer, linears, x, loss_inputs, want)
            fold(layer, producer, linears + also, s)
            for n in linears:
                feats[n] = feats[n] / s.to(feats[n].dtype)
            info[producer] = {"alpha": a, "loss": err}
        info["fold_error"] = loss(layer, loss_inputs, want)
        for n in names:
            m = layer.get_submodule(n)
            if not args.rtn and not n.endswith(("q_proj", "k_proj")):
                bound = search_clip(m.weight.data, feats[n], args.clip_tokens)
                w = m.weight.data.float().reshape(m.weight.shape[0], -1, GROUP)
                m.weight.data = (
                    w.clamp(-bound, bound).reshape(m.weight.shape).to(m.weight.dtype)
                )
            deq, q, scale, zero = fake_quant(m.weight.data)
            packed[f"layers.{index}.{n}"] = (
                q.reshape(m.weight.shape),
                scale.squeeze(-1),
                zero.squeeze(-1),
            )
            m.weight.data = deq.to(m.weight.dtype)
        info["output_error"] = loss(layer, loss_inputs, want)
        hidden = outs
        del feats, want, inputs, loss_inputs
        gc.collect()
        torch.cuda.empty_cache()
        report["layers"].append(info)
        print(json.dumps(info), f"{time.time() - t0:.0f}s", flush=True)
    report["seconds"] = round(time.time() - t0)
    Path(f"{out}.awq.json").write_text(json.dumps(report, indent=1))
    write_package(model, root, out, packed, Path(args.runtime))


def pack(values, dim=1):
    """0..15 values, eight per int32 word along ``dim`` (compressed-tensors pack-quantized)."""
    if dim == 0:
        rows = -(-values.shape[0] // 8) * 8
        padded = torch.full(
            (rows, values.shape[1]), 8, dtype=values.dtype, device=values.device
        )
        padded[: values.shape[0]] = values
        return pack(padded.t()).t().contiguous()
    v = values.to(torch.int64).reshape(values.shape[0], -1, 8)
    words = sum(v[..., i] << (4 * i) for i in range(8))
    return torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)


def write_package(model, root, out, packed, runtime):
    from safetensors import safe_open
    from safetensors.torch import save_file

    sys.path.insert(0, str(runtime))
    from vllm_srun.families.decision3 import package as pkg

    out.mkdir(parents=True, exist_ok=False)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    text_prefix = next(
        k for k in index if ".layers.0." in k and "visual" not in k
    ).split("layers.0.")[0]
    state = {
        f"{text_prefix}{k}": v
        for k, v in model.backbone.language_model.state_dict().items()
    }
    tensors = {}
    for name in sorted(index):
        layer = name.removesuffix(".weight")[len(text_prefix) :]
        if name.endswith(".weight") and layer in packed:
            q, scale, zero = packed[layer]
            tensors[f"{text_prefix}{layer}.weight_packed"] = pack(q).cpu()
            tensors[f"{text_prefix}{layer}.weight_scale"] = scale.to(
                torch.bfloat16
            ).cpu()
            tensors[f"{text_prefix}{layer}.weight_zero_point"] = pack(zero, dim=0).cpu()
            tensors[f"{text_prefix}{layer}.weight_shape"] = torch.tensor(list(q.shape))
        elif name in state:
            tensors[name] = state[name].detach().to(torch.bfloat16).cpu().contiguous()
        else:  # the vision tower and anything the text model does not hold, as published
            with safe_open(str(root / index[name]), framework="pt") as f:
                tensors[name] = f.get_tensor(name)
    save_file(tensors, str(out / "model.safetensors"))
    skip = {"model.safetensors.index.json", pkg.MANIFEST_NAME, "__pycache__"}
    for path in root.iterdir():
        if path.name in skip or path.name.startswith("model-"):
            continue
        (shutil.copytree if path.is_dir() else shutil.copy)(path, out / path.name)
    config = json.loads((root / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": TARGETS,
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "group",
                    "group_size": GROUP,
                    "dynamic": False,
                    "actorder": None,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
        "ignore": [r"re:.*visual\..*", r"re:.*linear_attn\.in_proj_(a|b)$"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_manifest(root, out, pkg)


def write_manifest(root, out, pkg):
    """MODEL_MANIFEST.json of ``out`` from ``root``'s, with every file hashed as the runtime verifies it."""
    manifest = json.loads((root / pkg.MANIFEST_NAME).read_text())
    decision = json.loads((out / "decision_config.json").read_text())

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()

    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    hashes = {name: sha(out / name) for name in files}
    identity = {
        n: hashes[n] for n in ("model.safetensors", *pkg.IDENTITY_FILES) if n in hashes
    }
    manifest["model_name"] = out.name
    manifest["identity"]["model_sha256"] = pkg.model_identity(identity, decision)
    manifest["identity"]["files_sha256"] = identity
    manifest["files_sha256"] = hashes
    manifest["files_bytes"] = {n: (out / n).stat().st_size for n in files}
    (out / pkg.MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
