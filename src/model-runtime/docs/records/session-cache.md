# Session cache

`--session-cache-gib` keeps the backbone state of earlier requests' states,
and a later request reuses it (`engines/native/session.py`): a state-once
Decision 3.0 package that reads its state both ways (`prefix_bidir`) when the
same state comes back whole, one that reads it causally (`prefix_causal`,
`causal`) when a later state starts with an earlier one. This record measures
what that saves on a repeated and on a growing state, and how far the answers
move against the same requests without the cache.

- **Date:** 2026-11-11.
- **Code:** this change.
- **Devices:** one AMD Instinct MI300X VF (gfx942) in the router's ROCm image
  (`vllm-sr-rocm:latest`, digest `sha256:6efbc9fa…`: PyTorch 2.12.0+git6bbd260
  on HIP 7.2.53211, Triton 3.7.0, FLA 0.5.2, causal-conv1d 1.7.0) with this
  change's runtime on the Python path; one NVIDIA GeForce RTX 4070 Ti SUPER
  (sm_89, 16 GB, driver 610.88, WSL2; PyTorch 2.14.1+cu130, Triton 3.8.0,
  FLA 0.5.2).
- **Models:** `vllm-sr/d3-turbo` at `689b3af1`, `prefix_bidir` as released.
  `vllm-sr/d3-flash` at `521525ef` and `vllm-sr/d3-edge` at `18d61df7` served
  under `prefix_causal` (`--attention-mode`, as `state-once.md` does): those
  weights were not built for the layout, so their rows measure the engine, not
  the layout's accuracy.
- **Tool:** `tools/session_cache_bench.py --pattern repeat|grow --runs 3`:
  `Runtime.call("decisions", ...)` in process, no HTTP; the same package served
  without (`off`) and with (`on`) the cache.

## Requests

Six long states from the public `tasksource/tasksource-jev-typed-decisions`
test split (court opinions, Wikipedia claim evidence, a retrieval puzzle), each
cut to its first 24,000 characters: 5,806 state tokens at the median. Every
call asks three questions. **Repeat:** each whole state is asked four times,
each time its own question and two others from a fixed pool, as a decision's
rules and then its `systemone` algorithm ask one deployment about the same
user message. **Grow:** each state arrives in 8 steps, step k holding the
first k/8 of its characters, as an agent loop's calls would, with the same
three questions. Each run clears the cache and
sends every state's calls in order. Tables give the median over the six
states of each state's median over 3 runs, after one untimed pass over one
state.

## A repeated state: d3-turbo on the MI300X

| Call | State tokens resumed | off | on |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 711.2 ms | 745.2 ms |
| 2 | 5,806 | 710.6 ms | 122.7 ms |
| 3 | 5,806 | 708.4 ms | 121.4 ms |
| 4 | 5,806 | 708.2 ms | 121.5 ms |

The first call saves the whole state, 4.8% slower than `off`. Every later call
with other questions computes only its question blocks against the saved
state: 5.8× faster.

## A growing state: d3-flash on the MI300X

| Step | Input tokens | State tokens resumed | off | on |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 997 | 0 | 47.7 ms | 48.3 ms |
| 2 | 1,724 | 0 | 71.4 ms | 71.6 ms |
| 3 | 2,387 | 0 | 95.4 ms | 109.4 ms |
| 4 | 3,149 | 2,048 | 120.2 ms | 101.6 ms |
| 5 | 3,861 | 2,816 | 150.8 ms | 101.2 ms |
| 6 | 4,571 | 3,584 | 173.5 ms | 103.6 ms |
| 7 | 5,261 | 4,224 | 196.2 ms | 105.0 ms |
| 8 | 5,967 | 4,864 | 230.3 ms | 106.2 ms |

Input tokens count the state once and each question's block. With the cache a
call costs about what its state added since the last call, and stays near
100 ms while `off` grows with the whole state: 2.2× at the last step. States
under 2,048 tokens are not saved and run as `off` does; step 3 is the first
save.

## A growing state: d3-edge on the RTX 4070 Ti SUPER

| Step | Input tokens | State tokens resumed | off | on |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 997 | 0 | 17.6 ms | 18.1 ms |
| 2 | 1,724 | 0 | 26.6 ms | 26.6 ms |
| 3 | 2,387 | 0 | 38.3 ms | 42.6 ms |
| 4 | 3,149 | 2,048 | 50.6 ms | 35.6 ms |
| 5 | 3,861 | 2,816 | 64.2 ms | 35.9 ms |
| 6 | 4,571 | 3,584 | 79.9 ms | 42.5 ms |
| 7 | 5,261 | 4,224 | 91.4 ms | 42.5 ms |
| 8 | 5,967 | 4,864 | 105.3 ms | 44.4 ms |

The same curve on a CUDA card with the smallest size: 2.4× at the last step.

## Answers

`on` against `off`, every call of the last run (the tool keeps each call's
answers from its last run):

| Model and pattern | Questions | Decisions that differ | Largest probability gap |
| --- | ---: | ---: | ---: |
| d3-turbo, repeat | 72 | 0 | 0.0141 |
| d3-flash, grow | 144 | 0 | 0.0085 |
| d3-edge, grow (RTX 4070 Ti SUPER) | 144 | 1 | 0.0237 |

The one decision that differs is a near tie: that call's probabilities moved
by at most 0.0027. Nothing is approximated: a resumed state computes what the whole state
computes (`tests/test_session_cache.py` checks it layer by layer for every
layout, within 1e-5 in FP32 on the CPU and within 2% of the largest hidden
value in BF16 with the fused gfx942 kernels). What moves the probabilities is
the rounding of another packing, the same order as branching moves them
([state-once](state-once.md)), which is why the argument is off by default.

## The cost of a session tree

A state the cache saves or resumes runs as a tree of that state
(`EngineModel.branch` sends it to `_run_tree`), which runs eager. One resumed
d3-flash call at the last step launched 2,077 kernels for 61.0 ms of GPU time
and took 106.3 ms; the same call without the cache launched 924 kernels for
230.5 ms of GPU time and took 241.1 ms (`torch.profiler`). The GPU time is the
tokens the call still computes; the other 45 ms is host time between kernels. A first save also
splits the recurrence at its mark and copies every layer's state out. With a
1,024-token floor (an earlier run of the same benchmark), the call of 1,724
input tokens paid 26 ms on its first save and the call of 2,387 was still no
faster than `off`, so states under 2,048 tokens are not saved.

## Memory

A saved state holds the Gated DeltaNet layers' recurrent states (FP32) and
convolution inputs, and the full-attention layers' keys and values (BF16) for
every token before its point:

| Model | Fixed per state | Per token |
| --- | ---: | ---: |
| d3-turbo (48 Gated DeltaNet and 16 attention layers) | 153.9 MB | 65,536 bytes |
| d3-flash (24 and 8) | 51.5 MB | 32,768 bytes |

After their runs, d3-turbo's cache held one 5,805-token state in 534,380,544
bytes and d3-flash's six states (marks of 2,304 to 5,632 tokens) in
1,089,208,320 bytes, which these sizes reproduce exactly.
