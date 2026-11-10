# Session cache

`--session-cache-gib` keeps the backbone state of earlier requests' states,
and a request whose state starts with the same tokens resumes from it
(`engines/native/session.py`). This record measures what that saves on a
state that grows call by call, and how far the answers move against the same
requests without the cache.

- **Date:** 2026-11-11.
- **Model:** `vllm-sr/Decision-2.0-Eos-0.8B` at the built-in revision
  `3594047d`: Qwen3.5, 18 Gated DeltaNet and 6 full-attention layers, BF16 on
  the GPU.
- **Device:** one NVIDIA GeForce RTX 4070 Ti SUPER (sm_89, 16 GB), driver
  610.88, under WSL2; PyTorch 2.14.1+cu130, Triton 3.8.0, FLA 0.5.2, no
  causal-conv1d.
- **Code:** this change.
- **Tool:** `tools/session_cache_bench.py --model PACKAGE --states STATES.jsonl
  --device cuda:0 --runs 3 --steps 8 --gib 1`: `Runtime.call("decisions", ...)`
  in process, no HTTP.

## Requests

Six long states from the public `tasksource/tasksource-jev-typed-decisions`
test split (court opinions, Wikipedia claim evidence, a retrieval puzzle), each
cut to its first 24,000 characters. Each state arrives in 8 growing steps, as a
conversation's calls would: step k holds the first k/8 of its characters. Every
call asks three questions: the row's own question, a three-way domain Choice
and a Noul. The tool serves the package three times: `off` (default engine
arguments), `tree` (every request a shared-context tree, without the cache,
which is the computation the cache path does when it resumes nothing) and `on`
(`--session-cache-gib 1`). Each run clears the cache and sends every state's
steps in order; the table is the median over the 6 states of each state's
median over 3 runs, after one untimed pass over one state.

## Latency

| Step | Input tokens (3 questions) | State tokens resumed | off | tree | on |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,640 | 0 | 54.8 ms | 46.0 ms | 54.8 ms |
| 2 | 4,797 | 0 | 101.6 ms | 51.1 ms | 60.9 ms |
| 3 | 6,790 | 1,280 | 151.8 ms | 68.8 ms | 64.8 ms |
| 4 | 9,072 | 2,048 | 199.0 ms | 89.8 ms | 65.6 ms |
| 5 | 11,232 | 2,816 | 114.2 ms | 108.0 ms | 72.0 ms |
| 6 | 13,362 | 3,456 | 129.3 ms | 129.5 ms | 72.4 ms |
| 7 | 15,432 | 4,224 | 156.6 ms | 155.8 ms | 75.7 ms |
| 8 | 17,546 | 4,864 | 178.4 ms | 177.9 ms | 76.0 ms |

The state is 776 tokens at step 1 and 5,764 at step 8 (medians; the input
tokens count it once per question). With the cache a call costs about what its
state added since the last call: 76 ms at the last step against 178 ms without
it (2.3×), and the cost stays
nearly flat as the state grows. Step 1 saves nothing (its state is shorter than
1,024 tokens) and runs as `off` does. Step 2 is the first save and costs about
10 ms more than `tree`: the recurrence runs in two calls, split at the mark,
and every layer's state is copied out. From step 3 on, every call resumes.

`off` and `tree` differ at steps 2 to 4 because `off` keeps the default
break-even (`auto_shared_tokens`, measured on an MI325X): below it a request's
questions run as padded rows, which on this card take up to 2.2× as long as
the tree. The cache path does not apply the break-even to a state it saves
or resumes.

## Answers

48 requests (144 questions) per run, `on` against `off`:

| Decisions that differ | Largest probability gap |
| ---: | ---: |
| 0 | 0.0083 |

Nothing is approximated: a resumed tree computes what the whole tree computes
(`tests/test_session_cache.py` checks it per layer on random backbones, within
1e-5 in FP32 on the CPU and within 2% of the largest hidden value in BF16 with
the fused GPU kernels). What moves the probabilities is the rounding of a
different packing, the same order as shared context itself moves them
([decision batching](rocm-mi325x-batching.md)), which is why the argument is
off by default.

## Memory

A saved state of this model takes 19.5 MB (18 recurrent states of 16 heads of
128 × 128 in FP32, and the convolution inputs) plus 12,288 bytes per token
before its point (6 full-attention layers, 2 key-value heads of 256 in BF16):
the 7 entries the last state left after its run held 445,046,784 bytes, which
these sizes reproduce exactly for its marks of 1,536 to 5,632 tokens.
