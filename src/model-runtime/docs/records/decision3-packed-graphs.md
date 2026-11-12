# Decision 3.0: merged packed rows on branch graphs

A merged Decision 3.0 pass packs its text rows when left-padding them would add
more than 10 % to its tokens ([lanes and packing](decision3-lanes-packing.md)).
That packed forward ran eager while the padded one replays a graph, so on a GPU
it paid its host time in full (the variable-length delta rule's host-side
preparation and every launch): on d3-mini a single-question request took 45 ms
packed and 12 ms padded. Packed rows at 10 to 20 % padding took 1.3 to 2.1
times the padded wall time, though their kernels took 1 to 8 % less. Packed
text rows now run as a static branched layout (`NativeEngine._rows_static`:
each row a `prefix_bidir` state with an empty block, which attends to itself
both ways from a zero state, as `Packed` rows do) and replay its bucket's
graph; the last layer runs on the read rows only. A batch past the branch
graphs' token limit keeps the eager packed forward, which costs less than the
static layout run eager. From 2 % padding on, packed rows now take at most the
padded wall time.

- **Date:** 2026-11-12.
- **Devices:** one AMD Instinct MI300X VF (gfx942) whose GPU was idle at every
  stage of the runs, in the router's ROCm image (`vllm-sr-rocm:latest`,
  digest `sha256:5d76c418…`: PyTorch 2.12.0+git6bbd260 on HIP 7.2.53211,
  Triton 3.7.0, FLA 0.5.2, no AITER; attention: SDPA for padded rows,
  `torch.nn.attention.varlen.varlen_attn` for eager packed rows and
  `aten._flash_attention_forward` in the static layout); one NVIDIA GeForce
  RTX 4070 Ti SUPER (sm_89, driver 610.88, WSL2; PyTorch 2.14.1+cu130,
  Triton 3.8.0, FLA 0.5.2).
- **Models:** `vllm-sr/d3-mini` at `a475367d` and `vllm-sr/d3-flash` at
  `521525ef` on the MI300X, `vllm-sr/d3-edge` at `18d61df7` on the RTX card,
  released layout, default engine arguments.
- **Requests:** 290 public `tasksource/tasksource-jev-typed-decisions` test
  requests (481 Choice and Noul questions), planned by the runtime.

## One forward

Each batch of rows ran through the family's left-padded pass (`_pass`) and
packed (`_readout` with `packed`), before (the engine's `_rows_branch` off:
packed rows eager, as on main) and after this change: wall time as the median
of 5 calls after 2 warm-up calls. Batches: the 140 requests with two or more
questions, 40 random merges of 2 to 16 requests and 40 windows of 2 to 16 rows
of similar length. Packed over padded wall time, median per padding share:

| Padding share | Batches | d3-mini before | d3-mini after | d3-flash before | d3-flash after | d3-edge (RTX) before | d3-edge (RTX) after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 to 2 % | 117 | 2.490 | 1.011 | 1.538 | 1.006 | 2.075 | 1.021 |
| 2 to 5 % | 33 | 1.785 | 0.972 | 1.109 | 0.975 | 1.315 | 0.991 |
| 5 to 10 % | 21 | 1.945 | 0.956 | 1.162 | 0.954 | 1.460 | 0.963 |
| 10 to 20 % | 10 | 2.050 | 0.951 | 1.322 | 0.956 | 1.652 | 0.936 |
| 20 to 50 % | 10 | 1.176 | 0.819 | 0.876 | 0.825 | 0.933 | 0.783 |
| 50 to 100 % | 9 | 0.714 | 0.665 | 0.678 | 0.643 | 0.637 | 0.602 |
| over 100 % | 20 | 0.400 | 0.385 | 0.397 | 0.385 | 0.326 | 0.329 |

The 40 random merges took 3,328 ms packed before and 3,044 ms after on d3-mini
(about 7,170 padded), 4,939 and 4,732 ms on d3-flash (about 11,380 padded) and
2,166 and 2,055 ms on d3-edge (about 5,860 padded). A single row packed stays 5
to 6 % slower than padded, and the 10 % threshold keeps such rows padded. A
request's rows pad 2.2 % of their tokens over the 290 requests (1.1 % at the
median of those with several questions), so a request's own forward, which
stays padded for its released numerics, has little to gain from packing.

Answers: packed against padded after this change, at most 2 decisions changed
per kind of batch on the MI300X (of 331, 368 and 484) and 5 of 484 in the RTX
card's random merges (6 before); the largest probability gap was 0.055.

## Served through the scheduler

Waves of C concurrent requests through the scheduler with the default engine
arguments (2 ms window), every request planned once, two untimed passes then a
timed one; before and after alternate in one process, twice (requests per
second):

| Model and device | C | Before | After |
| --- | ---: | ---: | ---: |
| d3-mini, MI300X | 4 | 62.6, 62.7 | 77.1, 76.8 |
| d3-mini, MI300X | 16 | 28.3, 71.0 | 76.8, 89.1 |
| d3-mini, MI300X | 64 | 88.8, 91.4 | 91.7, 92.1 |
| d3-flash, MI300X | 4 | 47.4, 47.1 | 50.7, 50.6 |
| d3-flash, MI300X | 16 | 48.9, 55.9 | 53.4, 51.3 |
| d3-flash, MI300X | 64 | 57.6, 57.5 | 57.4, 57.5 |
| d3-edge, RTX 4070 Ti SUPER | 4 | 112.4, 112.2 | 121.5, 121.6 |
| d3-edge, RTX 4070 Ti SUPER | 16 | 118.2, 119.9 | 113.5, 120.2 |
| d3-edge, RTX 4070 Ti SUPER | 64 | 118.4, 118.9 | 116.5, 117.9 |

At 4 concurrent requests merged batches are small and packed, and the graphs
add 23 % (d3-mini), 7 % (d3-flash) and 8 % (d3-edge). At 16 the runs spread
too far to tell (d3-mini's first run before is an outlier). At 64 the batches
keep the device busy longer than their launches take: the MI300X runs do not
move, and d3-edge on the RTX card came out 1 to 2 % lower in each of the four
pairs measured that day.
