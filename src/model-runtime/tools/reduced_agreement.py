"""Agreement of a Decision 3.0 model's reduced copy (``max_speed``) with its exact path, and what it costs (GPU records).

    python3 tools/reduced_agreement.py --package DIR --requests REQUESTS.jsonl --output OUT.json
        [--kind float8] [--exact PATTERN,...] [--count N] [--device rocm:0] [--base-path DIR]

``REQUESTS.jsonl`` holds ``{"id", "state", "questions", "target"?}`` lines; ``target`` maps a question ID to
the expected choice, Noul side (``true`` / ``false``) or Score level. The package is loaded once through the
decision3 family and the native engine with the max_speed profile's options and the reduced copy ``--kind``,
the linear layers matching ``--exact`` (``fnmatch`` patterns) kept exact (by default the copy and layers the package's registry entry consents to). Every
request is answered by the exact stack (``run``, as exact requests are) and by the copy (``run_approximate``),
one at a time, in alternating order. Reports the rows whose top answer changes, the largest absolute difference of any
probability or Noul value per request (median, 95th percentile, largest), accuracy against ``target`` where
given, each side's median latency, the device memory each engine holds and the reduced engine's receipt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from vllm_srun.accel.autotune import KernelChoices  # noqa: E402
from vllm_srun.accel.cuda import CUDAAccelerator  # noqa: E402
from vllm_srun.accel.rocm import ROCmAccelerator  # noqa: E402
from vllm_srun.engines.native.engine import NativeEngine  # noqa: E402
from vllm_srun.families.decision3.family import Decision3Family  # noqa: E402
from vllm_srun.plugins.base import (  # noqa: E402
    EngineOptions,
    Job,
    PackageRef,
    RegistryOptions,
)
from vllm_srun.profiles.exact import ExactProfile  # noqa: E402
from vllm_srun.profiles.max_speed import MaxSpeedProfile  # noqa: E402
from vllm_srun.registry import builtin  # noqa: E402

ACCELERATORS = {"cuda": CUDAAccelerator, "rocm": ROCmAccelerator}


def answer(
    model: Any, profile: Any, state: Any, questions: dict[str, Any], approximate: bool
) -> dict[str, Any]:
    plan = model.plan(state, questions)
    job = Job(
        items=plan.items, deadline=None, enqueued=time.monotonic(), profile=profile.name
    )
    logits: dict[int, Any] = {}
    for batch in profile.plan([job], model.forward_token_budget()):
        indices = [index for _, part in batch.parts for index in part]
        run = model.run_approximate if approximate else model.run
        values = run([plan.items[i] for i in indices])
        logits.update(zip(indices, values, strict=True))
    answers = dict(plan.errors)
    for index, item in enumerate(plan.items):
        answers[item.question_id] = model.answer(item, logits.get(index))
    return {qid: answers[qid] for qid in plan.question_ids}


def values(reply: dict[str, Any]) -> dict[str, float]:
    if "probabilities" in reply:
        return {str(k): float(v) for k, v in reply["probabilities"].items()}
    return {"true": float(reply["noul"]), "false": 1 - float(reply["noul"])}


def top(reply: dict[str, Any]) -> str:
    probs = values(reply)
    return max(probs, key=probs.__getitem__)


def label(reply: dict[str, Any]) -> str:
    if "choice" in reply:
        return str(reply["choice"])
    if "noul" in reply:
        return "true" if reply["noul"] >= 0.5 else "false"
    legend = reply.get("legend") or {}
    return str(legend.get(top(reply), top(reply)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--kind", default=None, help="reduced copy (default: the registry's)"
    )
    parser.add_argument(
        "--exact",
        default=None,
        help="comma-separated fnmatch patterns of linear layers kept exact",
    )
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--device", default="rocm:0")
    parser.add_argument("--base-path", default=None)
    parser.add_argument(
        "--warm-each",
        action="store_true",
        help="answer each request twice untimed first, so graphs replay",
    )
    args = parser.parse_args()

    family = Decision3Family(RegistryOptions(base_path=args.base_path, offline=True))
    package = family.verify(PackageRef(root=Path(args.package)))
    spec = family.describe(package)
    kind = args.kind or spec.dtype.reduced_gpu
    if kind is None:
        raise SystemExit("the package consents to no reduced copy: pass --kind")
    exact_layers = (
        tuple(p for p in args.exact.split(",") if p)
        if args.exact is not None
        else spec.dtype.reduced_exact
    )
    reduced_spec = dataclasses.replace(
        spec,
        dtype=dataclasses.replace(
            spec.dtype, reduced_gpu=kind, reduced_exact=exact_layers
        ),
    )
    name, _, index = args.device.partition(":")
    accelerator = ACCELERATORS[name]()
    device = accelerator.devices()[int(index or 0)]
    recorded = builtin.kernel_choices(
        package.model_sha256, device.accelerator, device.arch
    )
    if recorded and KernelChoices(recorded).install() is None:
        KernelChoices(recorded).pin_thread()

    engine = NativeEngine().load(
        reduced_spec,
        accelerator,
        device,
        MaxSpeedProfile().engine_options(EngineOptions()),
    )
    model = family.load(package, reduced_spec, engine)
    receipt = engine.receipt().get("reduced")
    if not receipt or receipt.get("skipped"):
        raise SystemExit(f"no reduced copy loaded: {receipt}")
    profile = ExactProfile()
    profile.bind(model)

    requests = [
        json.loads(line)
        for line in Path(args.requests).read_text(encoding="utf-8").splitlines()
        if line
    ]
    requests = requests[: args.count]
    for request in requests[:3]:  # warm-up: kernels, autotune, graphs
        for approximate in (False, True):
            answer(model, profile, request["state"], request["questions"], approximate)
    rows, times = [], {"exact": [], "reduced": []}
    for n, request in enumerate(requests):
        out = {}
        order = (
            (("exact", False), ("reduced", True))
            if n % 2 == 0
            else (("reduced", True), ("exact", False))
        )
        for side, approximate in order:
            for _ in range(
                2 if args.warm_each else 0
            ):  # a shape is captured on its second pass
                answer(
                    model, profile, request["state"], request["questions"], approximate
                )
            torch.cuda.synchronize()
            start = time.perf_counter()
            out[side] = answer(
                model, profile, request["state"], request["questions"], approximate
            )
            torch.cuda.synchronize()
            times[side].append(time.perf_counter() - start)
        for qid, reply in out["exact"].items():
            other = out["reduced"][qid]
            a, b = values(reply), values(other)
            target = (request.get("target") or {}).get(qid)
            rows.append(
                {
                    "id": request["id"],
                    "question": qid,
                    "changed": top(reply) != top(other),
                    "max_abs_diff": max(abs(a[k] - b.get(k, 0.0)) for k in a),
                    "exact_correct": None if target is None else label(reply) == target,
                    "reduced_correct": (
                        None if target is None else label(other) == target
                    ),
                }
            )
    diffs = sorted(r["max_abs_diff"] for r in rows)
    scored = [r for r in rows if r["exact_correct"] is not None]
    summary = {
        "model": package.model_name,
        "revision": package.ref.revision,
        "device": device.name,
        "kind": kind,
        "exact_layers": list(exact_layers),
        "answers": len(rows),
        "changed": sum(r["changed"] for r in rows),
        "agreement": 1 - sum(r["changed"] for r in rows) / len(rows),
        "max_abs_diff": {
            "median": diffs[len(diffs) // 2],
            "p95": diffs[int(len(diffs) * 0.95)],
            "max": diffs[-1],
        },
        "accuracy": {
            "scored": len(scored),
            "exact": (
                sum(r["exact_correct"] for r in scored) / len(scored)
                if scored
                else None
            ),
            "reduced": (
                sum(r["reduced_correct"] for r in scored) / len(scored)
                if scored
                else None
            ),
        },
        "median_ms": {
            side: round(statistics.median(t) * 1000, 1) for side, t in times.items()
        },
        "memory_bytes": {"engine": engine.memory_bytes(), "copy": receipt["bytes"]},
        "receipt": engine.receipt().get("reduced"),
        "exact_graphs": engine.receipt().get("graphs"),
    }
    Path(args.output).write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
