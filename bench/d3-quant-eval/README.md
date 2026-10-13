# d3 quantization eval

Scripts used to compare quantized Decision 3.0 (d3) checkpoints with the published BF16 weights. They run d3 through transformers (`trust_remote_code`), so any checkpoint transformers can load can be compared the same way, before the model runtime supports its format.

## Data

`make_sets.py OUT_DIR` downloads `tasksource/tasksource-jev-typed-decisions` at a pinned revision and writes:

| File | Rows | From | Used for |
| --- | --- | --- | --- |
| `calib.json` | 128 | train shard 0 | AWQ (or GPTQ) calibration |
| `dev200.json` | 200 | train shard 0 | per-layer sensitivity |
| `dev1000.json` | 1,000 | train shard 0 | per-projection-type sensitivity |
| `eval600.json` | 600 | test | quick BF16 vs quantized runs |
| `test_requests.jsonl` | 14,031 | test | full record through the model runtime |

Rows with a state over 6,000 characters are dropped. Choice and score rows with repeated options are skipped when the sets are read. The train sets do not overlap, and every choice (which layers or projections stay in BF16) is made on dev rows only. The dataset was republished with a different mix later on 2026-10-13, so the revision is pinned to the one these sets were built from.

## Scripts

- `quant_eval.py MODEL_DIR EVAL_JSON OUT_JSON {bf16,fp8,awq} [--calib CALIB_JSON] [--keep-bf16 REGEX]` answers every row and writes per-row probabilities, accuracy, weight memory and median latency. `fp8` is W8A8 (per-output-channel FP8 weights, per-row FP8 activations, `torch._scaled_mm`). `awq` is a simplified W4A16 AWQ (group 128, asymmetric int4, one scale per group of projections that read the same input, no clip search) with the matmul in BF16 after dequantization, so its latency is not a kernel number. `bf16` loads the checkpoint as published, which is also how a pre-quantized checkpoint is run. transformers dequantizes a compressed-tensors INT4 or NVFP4 checkpoint to BF16 when it loads it (NVFP4 also loses its FP4 activations), so a package that ships its own `d3q/` loader (for example `TheoHsiung/d3-W4A16-G128` or `TheoHsiung/d3-NVFP4`) is loaded through `d3q.load.load_d3` instead and runs on its 4-bit kernels; `loader` in the summary records which path ran. `fp8` and `awq` are refused on such a package.
- `compare.py BF16_JSON OTHER_JSON...` prints agreement with BF16 (rows whose top answer is the same) and the largest probability difference per row (median, p95, max).
- `sensitivity.py MODEL_DIR CALIB_JSON DEV_JSON OUT_JSON {fp8,awq}` quantizes one decoder layer at a time, then sweeps keeping the k most sensitive layers in BF16.
- `type_sens.py MODEL_DIR CALIB_JSON DEV_JSON OUT_JSON {fp8,awq}` does the same by projection type (MLP gate/up, MLP down, attention qkv, attention out, GDN inputs, GDN gates, GDN out): each type alone, and every type but one.
- `fp8_quant.py` is the Triton per-row activation quantizer `quant_eval.py` uses when Triton is installed.

FP8 needs `float8_e4m3fnuz` on MI300X (gfx942) or `float8_e4m3fn` on CUDA sm 8.9 or newer. Run with `D3_FAST=0` on ROCm to keep d3's own fused path out of the comparison.

## Example

```bash
python make_sets.py sets
python quant_eval.py vllm-sr/d3 sets/eval600.json bf16.json bf16
python quant_eval.py vllm-sr/d3 sets/eval600.json fp8.json fp8 --keep-bf16 'linear_attn\.in_proj_(qkv|z)$'
python quant_eval.py path/to/d3-W4A16-G128 sets/eval600.json w4a16.json bf16
python quant_eval.py path/to/d3-NVFP4 sets/eval600.json nvfp4.json bf16
python compare.py bf16.json fp8.json w4a16.json nvfp4.json
```

The last two lines run pre-quantized packages (GPTQ W4A16 group 128, and NVFP4) on their own 4-bit kernels.
