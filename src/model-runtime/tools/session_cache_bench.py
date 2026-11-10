"""A growing state answered with and without the session cache, through the served path (performance records).

    python3 tools/session_cache_bench.py --model PACKAGE --states STATES.jsonl --output OUT.json
        [--device cpu|cuda:0|rocm:0] [--steps 8] [--gib 1] [--runs 3]

Starts the runtime in process with the same package three times, ``off`` (default engine arguments), ``tree``
(every request a shared-context tree, no cache: what the cache path computes without resuming) and ``on``
(``--session-cache-gib``), and times ``Runtime.call("decisions", ...)``. Each line of STATES is
``{"id", "state", "questions"}``; the state is cut into ``--steps`` growing prefixes (step k holds the first
k / steps of its characters), which arrive in order as one conversation's calls would. Every run clears the
cache and sends each state's steps in order to every model; reported per step: the p50 milliseconds of each
model over the runs, the tokens the request counted, the state tokens the cache resumed, and per state the
largest probability gap between ``on`` and ``off`` and the decisions that differ.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vllm_srun.config import ModelConfig, ServeConfig  # noqa: E402
from vllm_srun.runtime import Runtime  # noqa: E402


def probabilities(answers: dict[str, Any]) -> list[list[float]]:
    out = []
    for answer in answers.values():
        if answer["type"] == "noul":
            out.append([1 - answer["noul"], answer["noul"]])
        else:
            out.append(list(answer["probabilities"].values()))
    return out


def call(
    runtime: Runtime, model: str, state: str, questions: dict
) -> tuple[float, dict]:
    begin = time.perf_counter()
    status, body = asyncio.run(
        runtime.call(
            "decisions", {"model": model, "state": state, "questions": questions}
        )
    )
    if status != 200:
        raise SystemExit(f"{model}: {status} {body}")
    return (time.perf_counter() - begin) * 1000, body


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gib", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    states = [json.loads(line) for line in args.states.open() if line.strip()]
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(model=args.model, name="off", device=args.device),
                ModelConfig(model=args.model, name="tree", device=args.device),
                ModelConfig(
                    model=args.model,
                    name="on",
                    device=args.device,
                    session_cache_gib=args.gib,
                ),
            ),
            result_cache_entries=0,
        )
    )
    runtime.start(background=False)
    try:
        served = {s.config.name: s for s in runtime.served}
        sharing = served["tree"].scheduler.batcher.shared
        assert sharing is not None
        sharing.threshold = 0  # no break-even: a request of any length is a tree
        cache = served["on"].model.engine_model.session
        assert cache is not None
        # One untimed pass: kernels compile and tune for the shapes first.
        cache.clear()
        for entry in states[:1]:
            for step in range(args.steps):
                cut = len(entry["state"]) * (step + 1) // args.steps
                for name in ("off", "tree", "on"):
                    call(runtime, name, entry["state"][:cut], entry["questions"])
        times: dict[tuple[str, int], dict[str, list[float]]] = {}
        rows: dict[tuple[str, int], dict[str, Any]] = {}
        for _ in range(args.runs):
            for entry in states:
                cache.clear()
                for step in range(args.steps):
                    cut = len(entry["state"]) * (step + 1) // args.steps
                    state = entry["state"][:cut]
                    key = (entry["id"], step)
                    before = cache.receipt()["resumed_tokens"]
                    ms_on, on = call(runtime, "on", state, entry["questions"])
                    resumed = cache.receipt()["resumed_tokens"] - before
                    ms_off, off = call(runtime, "off", state, entry["questions"])
                    ms_tree, _ = call(runtime, "tree", state, entry["questions"])
                    times.setdefault(key, {"off": [], "tree": [], "on": []})
                    times[key]["off"].append(ms_off)
                    times[key]["tree"].append(ms_tree)
                    times[key]["on"].append(ms_on)
                    a, b = probabilities(off["answers"]), probabilities(on["answers"])
                    gap = max(abs(x - y) for p, q in zip(a, b) for x, y in zip(p, q))
                    changed = sum(
                        max(range(len(p)), key=p.__getitem__)
                        != max(range(len(q)), key=q.__getitem__)
                        for p, q in zip(a, b)
                    )
                    rows[key] = {
                        "id": entry["id"],
                        "step": step,
                        "input_tokens": off["usage"]["input_tokens"],
                        "resumed_tokens": resumed,
                        "max_gap": gap,
                        "changed": changed,
                    }
        steps = []
        for step in range(args.steps):
            keys = [key for key in rows if key[1] == step]
            steps.append(
                {
                    "step": step,
                    "input_tokens_p50": statistics.median(
                        rows[k]["input_tokens"] for k in keys
                    ),
                    "resumed_tokens_p50": statistics.median(
                        rows[k]["resumed_tokens"] for k in keys
                    ),
                    "off_ms_p50": statistics.median(
                        statistics.median(times[k]["off"]) for k in keys
                    ),
                    "tree_ms_p50": statistics.median(
                        statistics.median(times[k]["tree"]) for k in keys
                    ),
                    "on_ms_p50": statistics.median(
                        statistics.median(times[k]["on"]) for k in keys
                    ),
                    "max_gap": max(rows[k]["max_gap"] for k in keys),
                    "changed": sum(rows[k]["changed"] for k in keys),
                }
            )
        engine = served["on"].model.engine_model
        report = {
            "model": args.model,
            "device": args.device,
            "steps": args.steps,
            "runs": args.runs,
            "states": len(states),
            "questions": sum(len(entry["questions"]) for entry in states),
            "session_cache": cache.receipt(),
            "kernels": engine.receipt().get("kernels"),
            "per_step": steps,
            "per_request": list(rows.values()),
        }
        args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        for row in steps:
            print(json.dumps(row))
    finally:
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
