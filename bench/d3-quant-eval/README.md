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

- `quant_eval.py MODEL_DIR EVAL_JSON OUT_JSON {bf16,fp8,awq} [--calib CALIB_JSON] [--keep-bf16 REGEX]` answers every row and writes per-row probabilities, accuracy, weight memory and median latency. `fp8` is W8A8 (per-output-channel FP8 weights, per-row FP8 activations, `torch._scaled_mm`). `awq` is a simplified W4A16 AWQ (group 128, asymmetric int4, one scale per group of projections that read the same input, no clip search) with the matmul in BF16 after dequantization, so its latency is not a kernel number. `bf16` loads the checkpoint as published, which is also how a GPTQ or other pre-quantized checkpoint would be run.
- `compare.py BF16_JSON OTHER_JSON...` prints agreement with BF16 (rows whose top answer is the same) and the largest probability difference per row (median, p95, max).
- `sensitivity.py MODEL_DIR CALIB_JSON DEV_JSON OUT_JSON {fp8,awq}` quantizes one decoder layer at a time, then sweeps keeping the k most sensitive layers in BF16.
- `type_sens.py MODEL_DIR CALIB_JSON DEV_JSON OUT_JSON {fp8,awq}` does the same by projection type (MLP gate/up, MLP down, attention qkv, attention out, GDN inputs, GDN gates, GDN out): each type alone, and every type but one.
- `fp8_quant.py` is the Triton per-row activation quantizer `quant_eval.py` uses when Triton is installed.
- `awq.py MODEL_DIR CALIB_JSON OUT_DIR --runtime SRC/model-runtime [--rtn]` runs AWQ layer by layer (per-channel scales folded into the norms and `up_proj`, clipping searched per row and group) and writes an asymmetric int4 (group 128) compressed-tensors package with a `MODEL_MANIFEST.json` the model runtime verifies. `--rtn` writes plain round-to-nearest weights instead, as a baseline.
- `fp8_package.py MODEL_DIR OUT_DIR --runtime SRC/model-runtime` writes an FP8 (FP8_DYNAMIC) compressed-tensors package on the CPU, keeping the GDN input projections in BF16 by default.
- `keep_bf16.py AWQ_DIR BF16_DIR OUT_DIR LAYERS --runtime SRC/model-runtime` restores some decoder layers of an AWQ package to the BF16 checkpoint's tensors.

FP8 needs `float8_e4m3fnuz` on MI300X (gfx942) or `float8_e4m3fn` on CUDA sm 8.9 or newer. Run with `D3_FAST=0` on ROCm to keep d3's own fused path out of the comparison.

## Example

```bash
python make_sets.py sets
python quant_eval.py vllm-sr/d3 sets/eval600.json bf16.json bf16
python quant_eval.py vllm-sr/d3 sets/eval600.json fp8.json fp8 --keep-bf16 'linear_attn\.in_proj_(qkv|z)$'
python quant_eval.py path/to/d3-gptq sets/eval600.json gptq.json bf16
python compare.py bf16.json fp8.json gptq.json
```

The GPTQ line is how a pre-quantized checkpoint would be compared. It has not been run with a GPTQ d3.
