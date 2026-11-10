"""States answered with and without the session cache, through the served path (performance records).

    python3 tools/session_cache_bench.py --model PACKAGE --states STATES.jsonl --output OUT.json
        [--device cpu|cuda:0|rocm:0] [--pattern grow|repeat] [--steps 8] [--gib 1] [--runs 3]
        [--attention-mode prefix_bidir|prefix_causal|causal]

Starts the runtime in process with the same package as ``off`` (default engine arguments) and ``on``
(``--session-cache-gib``) and, for a model that is not state-once, ``tree`` (every request a shared-context
tree without the cache: what the cache path computes when it resumes nothing). It times
``Runtime.call("decisions", ...)``. Each line of STATES is ``{"id", "state", "questions"}``.

- ``grow``: the state arrives in ``--steps`` growing prefixes (step k holds the first k / steps of its
  characters), as an agent loop's calls would, each with the line's questions.
- ``repeat``: the whole state arrives ``--steps`` times, each time with another question set (the line's
  first question and two from ``POOL``), as a decision's rules and then its ``systemone`` algorithm ask one deployment about it.

``--attention-mode`` first writes a copy of a Decision 3.0 package (a directory) that declares that layout
(``tools/state_once_parity.convert``). Every run clears the cache and sends each state's steps in order to
every model; reported per step: the p50 milliseconds of each model over the runs, the tokens the request
counted, the state tokens the cache resumed, and the largest probability gap between ``on`` and ``off`` and
the decisions that differ.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm_srun.config import ModelConfig, ServeConfig  # noqa: E402
from vllm_srun.runtime import Runtime  # noqa: E402

POOL: list[tuple[str, dict[str, Any]]] = [
    (
        "domain",
        {
            "type": "choice",
            "instructions": "Which domain is this?",
            "criteria": {
                "law": "Legal text",
                "science": "Science or medicine",
                "other": "Anything else",
            },
        },
    ),
    (
        "reasoning",
        {"type": "noul", "instructions": "Does answering need multi-step reasoning?"},
    ),
    (
        "urgent",
        {"type": "noul", "instructions": "Does this need a reply within the hour?"},
    ),
    (
        "tone",
        {
            "type": "choice",
            "instructions": "What is the tone of the text?",
            "criteria": {
                "neutral": "Neutral",
                "positive": "Positive",
                "negative": "Negative",
            },
        },
    ),
    (
        "personal",
        {"type": "noul", "instructions": "Does the text name a private person?"},
    ),
    (
        "length",
        {
            "type": "score",
            "instructions": "How long an answer does this need?",
            "criteria": ["one line", "a paragraph", "a page"],
        },
    ),
]


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


def requests(entry: dict[str, Any], pattern: str, steps: int) -> list[tuple[str, dict]]:
    """The calls of one state, in order: (state, questions) per step."""
    if pattern == "grow":
        return [
            (
                entry["state"][: len(entry["state"]) * (step + 1) // steps],
                entry["questions"],
            )
            for step in range(steps)
        ]
    first = dict(list(entry["questions"].items())[:1])
    return [
        (
            entry["state"],
            first | dict(POOL[(2 * step + k) % len(POOL)] for k in range(2)),
        )
        for step in range(steps)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pattern", choices=("grow", "repeat"), default="grow")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gib", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--attention-mode")
    args = parser.parse_args()
    model = args.model
    if args.attention_mode:
        from state_once_parity import convert

        source = Path(model)
        target = source.parent / f"{source.name}-{args.attention_mode}"
        if not (target / "MODEL_MANIFEST.json").is_file():
            convert(source, target, args.attention_mode)
        model = str(target)
    states = [json.loads(line) for line in args.states.open() if line.strip()]
    config = Path(model) / "decision_config.json"
    layout = (
        json.loads(config.read_text()).get("attention_mode")
        if config.is_file()
        else None
    )
    # A state-once package branches every request already: no tree model to compare.
    tree = layout not in ("prefix_bidir", "prefix_causal", "causal")
    configs = [ModelConfig(model=model, name="off", device=args.device)]
    if tree:
        configs.append(ModelConfig(model=model, name="tree", device=args.device))
    configs.append(
        ModelConfig(
            model=model, name="on", device=args.device, session_cache_gib=args.gib
        )
    )
    runtime = Runtime(ServeConfig(models=tuple(configs), result_cache_entries=0))
    runtime.start(background=False)
    try:
        served = {s.config.name: s for s in runtime.served}
        engine = served["on"].model.engine_model
        names = ["off", "on"]
        if tree:
            sharing = served["tree"].scheduler.batcher.shared
            assert sharing is not None
            sharing.threshold = 0  # no break-even: a request of any length is a tree
            names.insert(1, "tree")
        cache = engine.session
        assert cache is not None
        # One untimed pass: kernels compile and tune for the shapes first.
        for state, questions in requests(states[0], args.pattern, args.steps):
            for name in names:
                call(runtime, name, state, questions)
        times: dict[tuple[str, int], dict[str, list[float]]] = {}
        rows: dict[tuple[str, int], dict[str, Any]] = {}
        for _ in range(args.runs):
            for entry in states:
                cache.clear()
                for step, (state, questions) in enumerate(
                    requests(entry, args.pattern, args.steps)
                ):
                    key = (entry["id"], step)
                    before = cache.receipt()["resumed_tokens"]
                    ms_on, on = call(runtime, "on", state, questions)
                    resumed = cache.receipt()["resumed_tokens"] - before
                    ms_off, off = call(runtime, "off", state, questions)
                    timed = times.setdefault(key, {name: [] for name in names})
                    timed["on"].append(ms_on)
                    timed["off"].append(ms_off)
                    if "tree" in names:
                        timed["tree"].append(call(runtime, "tree", state, questions)[0])
                    a, b = probabilities(off["answers"]), probabilities(on["answers"])
                    gap = max(
                        abs(x - y)
                        for p, q in zip(a, b, strict=True)
                        for x, y in zip(p, q, strict=True)
                    )
                    changed = sum(
                        max(range(len(p)), key=p.__getitem__)
                        != max(range(len(q)), key=q.__getitem__)
                        for p, q in zip(a, b, strict=True)
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
            row = {
                "step": step,
                "input_tokens_p50": statistics.median(
                    rows[k]["input_tokens"] for k in keys
                ),
                "resumed_tokens_p50": statistics.median(
                    rows[k]["resumed_tokens"] for k in keys
                ),
                "max_gap": max(rows[k]["max_gap"] for k in keys),
                "changed": sum(rows[k]["changed"] for k in keys),
            }
            for name in names:
                row[f"{name}_ms_p50"] = statistics.median(
                    statistics.median(times[k][name]) for k in keys
                )
            steps.append(row)
        report = {
            "model": model,
            "device": args.device,
            "pattern": args.pattern,
            "attention_mode": args.attention_mode,
            "steps": args.steps,
            "runs": args.runs,
            "states": len(states),
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
