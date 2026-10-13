# Decision 3.0 float8 copy

`max_speed` can keep an FP8 copy of a decoder's linear layers next to the BF16
weights and answer approximate passes from it (`engines/native/reduced.py`,
`accel/triton_fp8.py`). This record decides which d3 packages consent to it
(`reduced={"gpu": "float8"}`): only those whose answers keep their top choice
on at least 99% of the test split. `vllm-sr/d3` (27B) does, with the GDN input
projections kept exact; the five smaller packages do not.

- **Date:** 2026-10-13.
- **Device:** one AMD Instinct MI300X (gfx942), `vllm/vllm-openai-rocm` image
  (PyTorch 2.12.0, Triton 3.7.1, FLA 0.5.2), `float8_e4m3fnuz`.
- **Models:** the six built-in d3 packages at their registry revisions.
- **Code:** this branch, on top of the Decision 3.0 graphs change.
- **Tool:** `tools/reduced_agreement.py`, which loads a package once with the
  `max_speed` options and answers every request with `run` (the exact stack)
  and `run_approximate` (the copy), in alternating order.

## Data

`tasksource/tasksource-jev-typed-decisions` at revision `28c39125` (the
dataset was republished with a different mix later the same day). Rows with
a state over 6,000 characters are dropped, and choice or score rows whose
options repeat are skipped. Which layers stay exact was chosen on rows of the
first train shard (1,000 rows, never the test split). The records below are
the whole test split: 14,031 requests of one question each, from 14,129 rows
(11,117 choice, 2,069 Noul and 943 score before the filter).

## Agreement

Every answer from the copy, against the exact answer to the same request.
"Changed" counts answers whose top choice (or Noul side) differs; accuracy is
against the dataset's target.

| Package | Exact layers | Changed | Agreement | Accuracy, exact | Accuracy, copy | Largest probability change, p95 |
| --- | --- | --- | --- | --- | --- | --- |
| `d3` (27B) | GDN inputs | 134 | 99.04% | 0.8325 | 0.8322 | 0.031 |
| `d3` (27B) | none | 138 | 99.02% | 0.8325 | 0.8308 | |
| `d3-flash` | GDN inputs | 195 | 98.61% | 0.8107 | 0.8119 | 0.043 |
| `d3-mini` | GDN inputs | 205 | 98.54% | 0.7956 | 0.7971 | 0.047 |
| `d3-nano` | GDN inputs | 372 | 97.35% | 0.7543 | 0.7531 | 0.078 |
| `d3-lite` | GDN inputs | 380 | 97.29% | 0.7176 | 0.7155 | |
| `d3-edge` | GDN inputs | 449 | 96.80% | 0.6634 | 0.6642 | |

"GDN inputs" is `layers.*.linear_attn.in_proj_qkv` and
`layers.*.linear_attn.in_proj_z`. On the 1,000 train rows, keeping them exact
cut the changed answers of `d3` from 12 to 3. On the test split the two
settings are within four answers of each other, and keeping them exact holds
accuracy (0.8322 against 0.8308), so the registry keeps them exact. Keeping
whole decoder layers exact instead did not raise agreement on 200 train rows.

These answers came from the copy run eagerly. With graphs, the first 500
requests of each of `d3`, `d3-flash` and `d3-nano` gave the same largest
probability change per request as the eager copy, to the last bit.

## Latency and memory

Median of 500 test requests on `d3`, each answered by both sides:

| Measurement | Exact | Copy |
| --- | --- | --- |
| Each request's shape seen twice before (its graph replays) | 49.6 ms | 38.6 ms |
| Each request answered once (most shapes new) | 73.8 ms | 74.1 ms |

A profile of one 456-character request with graphs replaying took 50.4 ms
exact and 41.7 ms on the copy; the device time of its kernels was 46.2 ms and
37.6 ms, of which the 400 activation quantizations took 2.5 ms. Run eagerly,
the same request took 100.5 ms exact and 139.5 ms on the copy for 46.5 ms and
36.6 ms of kernels: launching a quantization and a scaled GEMM per layer costs
more host time than FP8 saves on the device. So the copy runs only where its
graphs replay, and a shape's first use, a shape over the graph limits and an
image prompt run on the exact stack. Before that rule, an eager copy answered
the 14,031 requests in a median of 86.8 ms against 68.9 ms exact.

The copy adds 20.3 GB (20,336,494,592 bytes) to the engine's 52.2 GB of
weights; placement counts it. 400 of the 496 linear layers are in FP8.
