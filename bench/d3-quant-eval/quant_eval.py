"""d3 BF16 vs quantized: same requests, compare answer probabilities, accuracy, memory and latency.

usage: quant_eval.py MODEL_DIR EVAL_JSON OUT_JSON MODE [--calib CALIB_JSON] [--n N] [--keep-bf16 REGEX]
MODE:
  bf16   the checkpoint as published; a compressed-tensors package that ships d3q/ (for example
         TheoHsiung/d3-W4A16-G128 or d3-NVFP4) loads through d3q.load.load_d3, so it runs on its
         4-bit kernels instead of being dequantized to BF16 by transformers
  fp8    W8A8: per-output-channel FP8 weights, per-token FP8 activations, torch._scaled_mm
  awq    W4A16: int4 weights, group 128, asymmetric, with an AWQ input-channel scale searched per
         group of projections that read the same input (needs --calib); matmul in BF16 after dequant
Only the language model's projections are quantized: the vision tower, the embeddings and the FP32
readout stay as published.
"""

import argparse
import gc
import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

try:
    from fp8_quant import quant_rows as triton_rows
except ImportError:  # no Triton: the PyTorch quantizer below
    triton_rows = None

HIP = torch.version.hip is not None
FP8 = torch.float8_e4m3fnuz if HIP else torch.float8_e4m3fn
FMAX = torch.finfo(FP8).max
GROUP = 128
# projections that read the same input share one AWQ scale
SIBLINGS = [
    ("q_proj", "k_proj", "v_proj"),
    ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
    ("gate_proj", "up_proj"),
    ("o_proj",),
    ("out_proj",),
    ("down_proj",),
]
SAMPLE_TOKENS = 2048


def quant_rows(x):
    scale = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / FMAX
    return (x.float() / scale).clamp(-FMAX, FMAX).to(FP8), scale


class FP8Linear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        w, s = quant_rows(lin.weight.detach())
        self.register_buffer("w", w)
        self.register_buffer("ws", s.reshape(1, -1).contiguous())
        self.bias = lin.bias
        self.out_features = lin.out_features

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        xq, xs = triton_rows(x2, FP8) if triton_rows else quant_rows(x2)
        y = torch._scaled_mm(
            xq, self.w.t(), scale_a=xs, scale_b=self.ws, out_dtype=x.dtype
        )
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)


def int4_groups(w):
    """Asymmetric int4 per group of GROUP input channels: (q uint8 [N,K], scale, zero) in float32."""
    n, k = w.shape
    g = w.float().reshape(n, k // GROUP, GROUP)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp(min=1e-8)
    zero = (-lo / scale).round().clamp(0, 15)
    q = ((g / scale).round() + zero).clamp(0, 15)
    return q, scale, zero


def int4_dequant(q, scale, zero):
    return ((q - zero) * scale).reshape(q.shape[0], -1)


class Int4Linear(nn.Module):
    """W4A16: packed int4 weights (two per byte), dequantized to BF16 per call; x is divided by the AWQ scale."""

    def __init__(self, lin, s):
        super().__init__()
        w = lin.weight.detach()
        q, scale, zero = int4_groups(w.float() * s)
        q = q.reshape(w.shape).to(torch.uint8)
        self.register_buffer("packed", (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous())
        self.register_buffer("scale", scale.squeeze(-1).to(torch.bfloat16))
        self.register_buffer("zero", zero.squeeze(-1).to(torch.uint8))
        self.register_buffer("inv_s", (1.0 / s).to(w.dtype))
        self.bias = lin.bias
        self.shape = tuple(w.shape)

    def weight(self):
        lo, hi = self.packed & 15, self.packed >> 4
        q = (
            torch.stack((lo, hi), dim=-1)
            .reshape(self.shape[0], -1, GROUP)
            .to(torch.bfloat16)
        )
        return (
            (q - self.zero.unsqueeze(-1).to(torch.bfloat16)) * self.scale.unsqueeze(-1)
        ).reshape(self.shape)

    def forward(self, x):
        return nn.functional.linear(x * self.inv_s, self.weight(), self.bias)


def ships_d3q(root):
    """A quantized package with its own d3q/ loader (its 4-bit kernels).

    transformers dequantizes compressed-tensors INT4/NVFP4 weights to BF16 when it loads
    them (NVFP4 also loses its FP4 activations), so that path would not measure it.
    """
    path = Path(root)
    config = path / "config.json"
    return (
        config.is_file()
        and "quantization_config" in json.loads(config.read_text())
        and (path / "d3q").is_dir()
    )


def load(root):
    """(model, loader): d3q for a package that ships it, transformers otherwise."""
    if ships_d3q(root):
        path = Path(root)
        sys.path.insert(0, str(path))
        from d3q.load import load_d3

        return load_d3(path, device="cuda:0", batch_size=1), "d3q"
    model = AutoModel.from_pretrained(root, trust_remote_code=True, device="cuda:0")
    return model, "transformers"


def linears(lm, keep):
    return [
        (n, m)
        for n, m in lm.named_modules()
        if isinstance(m, nn.Linear) and not (keep and re.search(keep, n))
    ]


def replace(lm, name, new):
    parent, _, leaf = name.rpartition(".")
    setattr(lm.get_submodule(parent) if parent else lm, leaf, new)


def groups(lm, keep):
    """Lists of (name, module) that read the same input: siblings of one layer's block."""
    by_parent = {}
    for name, mod in linears(lm, keep):
        parent, _, leaf = name.rpartition(".")
        by_parent.setdefault(parent, {})[leaf] = (name, mod)
    out = []
    for leaves in by_parent.values():
        used = set()
        for sib in SIBLINGS:
            members = [leaves[s] for s in sib if s in leaves]
            if members:
                out.append(members)
                used.update(s for s in sib if s in leaves)
        out.extend([v] for k, v in leaves.items() if k not in used)
    return out


def calibrate(model, lm, grouping, rows):
    """Per group: mean |x| per input channel over all calibration tokens, and a token sample."""
    stats, hooks = {}, []
    gen = torch.Generator().manual_seed(0)

    def hook(key):
        def fn(_, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            st = stats.setdefault(
                key,
                {
                    "abs": torch.zeros(
                        x.shape[1], device=x.device, dtype=torch.float64
                    ),
                    "n": 0,
                    "sample": [],
                },
            )
            st["abs"] += x.abs().sum(0, dtype=torch.float64)
            st["n"] += x.shape[0]
            take = torch.randperm(x.shape[0], generator=gen)[
                : max(1, SAMPLE_TOKENS // 64)
            ]
            st["sample"].append(x[take.to(x.device)].clone())

        return fn

    for i, members in enumerate(grouping):
        hooks.append(members[0][1].register_forward_pre_hook(hook(i)))
    with torch.no_grad():
        for row in rows:
            model.system_one(state=row["state"], questions={"q": question(row)})
    for h in hooks:
        h.remove()
    for st in stats.values():
        st["mean"] = (st["abs"] / st["n"]).float()
        st["x"] = torch.cat(st["sample"])[-SAMPLE_TOKENS:]
        del st["sample"], st["abs"]
    return stats


@torch.no_grad()
def awq_scale(members, st, grid=20):
    w = torch.cat([m.weight.detach() for _, m in members]).float()
    x = st["x"].float()
    ref = x @ w.t()
    best, best_s, best_err = None, None, float("inf")
    for i in range(grid + 1):
        a = i / grid
        s = st["mean"].clamp(min=1e-4) ** a
        s = s / (s.max() * s.min()).sqrt()
        q, scale, zero = int4_groups(w * s)
        wq = int4_dequant(q, scale, zero) / s
        err = (x @ wq.t() - ref).pow(2).mean().item()
        if err < best_err:
            best, best_s, best_err = a, s, err
    return best_s, best, best_err


def question(row):
    kind = row["kind"]
    if kind == "choice":
        return {
            "type": "choice",
            "instructions": row["question"],
            "criteria": {o: None for o in row["options"]},
        }
    if kind == "score":
        return {
            "type": "score",
            "instructions": row["question"],
            "criteria": list(row["options"]),
        }
    return {"type": "noul", "instructions": row["question"]}


def usable(row):
    return row["kind"] == "noul" or len(set(row["options"])) == len(row["options"])


def correct(row, probs):
    if row["kind"] == "noul":
        return (probs["true"] >= 0.5) == (row["target"][0] >= 0.5)
    vals = list(probs.values())
    return vals.index(max(vals)) == row["target"].index(max(row["target"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("eval")
    p.add_argument("out")
    p.add_argument("mode", choices=["bf16", "fp8", "awq"])
    p.add_argument("--calib")
    p.add_argument("--n", type=int)
    p.add_argument(
        "--keep-bf16", default=None, help="regex of projection names kept in BF16"
    )
    args = p.parse_args()
    if args.mode != "bf16" and ships_d3q(args.root):
        p.error(f"{args.root} is already quantized: run it with mode bf16")
    rows = [r for r in json.load(open(args.eval))[: args.n] if usable(r)]
    model, loader = load(args.root)
    lm = model.backbone.language_model
    info = {"loader": loader}
    t0 = time.time()
    if args.mode == "fp8":
        names = [n for n, _ in linears(lm, args.keep_bf16)]
        for name in names:
            replace(lm, name, FP8Linear(lm.get_submodule(name)))
        info["quantized"] = len(names)
    elif args.mode == "awq":
        calib = [r for r in json.load(open(args.calib)) if usable(r)]
        grouping = groups(lm, args.keep_bf16)
        stats = calibrate(model, lm, grouping, calib)
        names = [[n for n, _ in members] for members in grouping]
        del grouping
        alphas = []
        for i, members in enumerate(names):
            mods = [(n, lm.get_submodule(n)) for n in members]
            s, a, _ = awq_scale(mods, stats.pop(i))
            alphas.append(a)
            for name, lin in mods:
                replace(lm, name, Int4Linear(lin, s.to(lin.weight.device)))
            del mods, lin
        info.update(
            quantized=sum(len(g) for g in names),
            groups=len(names),
            calib_rows=len(calib),
            alpha_mean=statistics.mean(alphas),
            alpha_zero=sum(a == 0 for a in alphas),
        )
    info["quantize_s"] = round(time.time() - t0, 1)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        model.system_one(
            state="warm-up", questions={"q": {"type": "noul", "instructions": "ok?"}}
        )
    out, times = [], []
    for row in rows:
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = model.system_one(state=row["state"], questions={"q": question(row)})
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        ans = res["answers"]["q"]
        probs = (
            ans["probabilities"]
            if "probabilities" in ans
            else {"false": 1 - ans["noul"], "true": ans["noul"]}
        )
        out.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "probs": probs,
                "correct": correct(row, probs),
            }
        )
    summary = {
        "mode": args.mode,
        "keep_bf16": args.keep_bf16,
        **info,
        "rows": len(out),
        "accuracy": round(sum(r["correct"] for r in out) / len(out), 4),
        "weights_mib": round(weights / 2**20),
        "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "median_ms": round(statistics.median(times) * 1000, 1),
        # d3q returns the runtime D3 itself; transformers wraps it in .runtime
        "fast_path": getattr(getattr(model, "runtime", model), "fast_skipped", None)
        or "on",
        "device": torch.cuda.get_device_name(),
    }
    json.dump({"summary": summary, "rows": out}, open(args.out, "w"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
