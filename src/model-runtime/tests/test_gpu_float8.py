"""GPU: the float8 reduced copy (FP8 weights, per-row FP8 activations) of a decoder's linear layers."""

from __future__ import annotations

import dataclasses
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from torch import nn  # noqa: E402
from vllm_srun.engines.native.reduced import (  # noqa: E402
    Float8Linear,
    copy_counts,
    float8_format,
    reduced_view,
    unavailable,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA or ROCm GPU")
    found = torch.device("cuda:0")
    reason = unavailable("float8", found)
    if reason:
        pytest.skip(reason)
    return found


def reference(x, dtype):
    largest = torch.finfo(dtype).max
    scale = x.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / largest
    return (x.float() / scale).clamp(-largest, largest).to(dtype), scale


@pytest.mark.parametrize("shape", [(1, 64), (37, 48), (5, 5120), (3, 17408), (0, 64)])
def test_the_row_kernel_quantizes_as_pytorch_does(device, shape):
    from vllm_srun.accel.triton_fp8 import quantize_rows

    dtype = float8_format(device)
    torch.manual_seed(0)
    x = torch.randn(shape, device=device, dtype=torch.bfloat16) * 7
    if shape[0]:
        x[0, 3] = 0  # zero rows and entries keep a positive scale
    q, scale = quantize_rows(x, dtype)
    want_q, want_scale = reference(x, dtype)
    assert q.dtype == dtype and q.shape == x.shape and scale.shape == (shape[0], 1)
    torch.testing.assert_close(scale, want_scale, rtol=1e-6, atol=0)
    # x * (1 / scale) and x / scale may round to neighbouring FP8 values: at most one step
    # (an eighth of the magnitude with three mantissa bits) apart, and rarely
    got, ref = q.float(), want_q.float()
    assert (got - ref).abs().le(ref.abs() / 8 + 2**-6).all()
    assert (got != ref).sum() <= got.numel() // 100


def test_an_all_zero_row_quantizes_to_zero(device):
    from vllm_srun.accel.triton_fp8 import quantize_rows

    q, scale = quantize_rows(
        torch.zeros(2, 64, device=device, dtype=torch.bfloat16), float8_format(device)
    )
    assert (
        torch.equal(q.float(), torch.zeros(2, 64, device=device)) and (scale > 0).all()
    )


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("leading", [(9,), (2, 7), (1, 1)])
def test_a_float8_linear_computes_close_to_the_bf16_linear(device, bias, leading):
    torch.manual_seed(1)
    linear = nn.Linear(256, 512, bias=bias, device=device, dtype=torch.bfloat16)
    x = torch.randn(*leading, 256, device=device, dtype=torch.bfloat16)
    want = linear(x).float()
    got = Float8Linear(linear)(x)
    assert got.dtype == torch.bfloat16 and got.shape == want.shape
    error = (got.float() - want).norm() / want.norm()
    assert error < 0.06, error


def test_a_float8_linear_reads_strided_inputs(device):
    linear = nn.Linear(128, 64, device=device, dtype=torch.bfloat16)
    wide = torch.randn(6, 256, device=device, dtype=torch.bfloat16)
    x = wide[:, ::2]  # column stride 2
    fp8 = Float8Linear(linear)
    torch.testing.assert_close(fp8(x), fp8(x.contiguous()), rtol=0, atol=0)


def test_a_float8_linear_holds_half_the_weight_bytes(device):
    linear = nn.Linear(1024, 2048, bias=False, device=device, dtype=torch.bfloat16)
    fp8 = Float8Linear(linear)
    assert fp8.weight.element_size() == 1
    assert fp8.weight.numel() == linear.weight.numel()
    assert fp8.weight_scale.shape == (1, 2048)


def test_a_float8_linear_replays_in_a_graph(device):
    linear = nn.Linear(256, 256, device=device, dtype=torch.bfloat16)
    fp8 = Float8Linear(linear)
    x = torch.randn(33, 256, device=device, dtype=torch.bfloat16)
    eager = fp8(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fp8(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fp8(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_a_float8_view_shares_layers_fp8_cannot_hold(device):
    model = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 4), nn.Linear(4, 64)).to(
        device, torch.bfloat16
    )
    view = reduced_view(model, "float8")
    assert isinstance(view[0], Float8Linear)
    assert view[1] is model[1] and view[2] is model[2]
    assert copy_counts(view, model) == {"reduced": 1, "exact": 2}
    assert isinstance(model[0], nn.Linear)


def load_decision3(package, device, max_speed, exact=(), graphs=True):
    from vllm_srun.accel.cuda import CUDAAccelerator
    from vllm_srun.accel.rocm import ROCmAccelerator
    from vllm_srun.engines.native.engine import NativeEngine
    from vllm_srun.families.decision3.family import Decision3Family
    from vllm_srun.plugins.base import EngineOptions, PackageRef, RegistryOptions
    from vllm_srun.profiles.max_speed import MaxSpeedProfile

    family = Decision3Family(RegistryOptions(offline=True))
    verified = family.verify(PackageRef(root=package))
    spec = family.describe(verified)
    spec = dataclasses.replace(
        spec,
        dtype=dataclasses.replace(
            spec.dtype, reduced_gpu="float8", reduced_exact=exact
        ),
    )
    accelerator = ROCmAccelerator() if torch.version.hip else CUDAAccelerator()
    options = (
        MaxSpeedProfile().engine_options(EngineOptions())
        if max_speed
        else EngineOptions()
    )
    options = dataclasses.replace(options, graphs=options.graphs and graphs)
    engine = NativeEngine().load(spec, accelerator, accelerator.devices()[0], options)
    return family.load(verified, spec, engine), engine


def ask(model, questions, approximate):
    from vllm_srun.plugins.base import Job
    from vllm_srun.profiles.exact import ExactProfile

    profile = ExactProfile()
    plan = model.plan(
        "The order arrived damaged; the customer wants a refund.", questions
    )
    job = Job(
        items=plan.items, deadline=None, enqueued=time.monotonic(), profile=profile.name
    )
    run = model.run_approximate if approximate else model.run
    logits = {}
    for batch in profile.plan([job], model.forward_token_budget()):
        indices = [i for _, part in batch.parts for i in part]
        logits.update(zip(indices, run([plan.items[i] for i in indices]), strict=True))
    return {
        item.question_id: model.answer(item, logits.get(i))
        for i, item in enumerate(plan.items)
    }


QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"returns": None, "billing": None, "tech": None},
    },
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["low", "mid", "high"],
    },
}


def numbers(answers):
    out = []
    for reply in answers.values():
        out.extend(
            reply["probabilities"].values()
            if "probabilities" in reply
            else [reply["noul"]]
        )
    return torch.tensor(out)


def test_only_approximate_batches_run_the_float8_copy(device, d3_text_package):
    exact_model, _ = load_decision3(d3_text_package, device, max_speed=False)
    model, engine = load_decision3(
        d3_text_package,
        device,
        max_speed=True,
        exact=("layers.0.*", "layers.*.linear_attn.in_proj_z"),
    )
    receipt = engine.receipt()["reduced"]
    assert receipt["kind"] == "float8" and receipt["bytes"] > 0
    counts = receipt["linear_layers"]
    assert counts["reduced"] > 0 and counts["exact"] > 0
    view = engine.reduced[None]
    assert view.layers[0].mlp.gate_proj is engine.backbone.layers[0].mlp.gate_proj
    assert isinstance(view.layers[1].mlp.gate_proj, Float8Linear)
    assert (
        view.layers[1].linear_attn.in_proj_z
        is engine.backbone.layers[1].linear_attn.in_proj_z
    )
    assert isinstance(engine.backbone.layers[1].mlp.gate_proj, nn.Linear)
    want = numbers(ask(exact_model, QUESTIONS, approximate=False))
    # exact batches on a max_speed engine keep the exact weights: golden readiness and exact requests
    assert torch.equal(numbers(ask(model, QUESTIONS, approximate=False)), want)
    # a shape's first pass does not replay, so it runs on the exact stack
    assert torch.equal(numbers(ask(model, QUESTIONS, approximate=True)), want)
    got = numbers(ask(model, QUESTIONS, approximate=True))  # captured on the copy
    assert torch.equal(numbers(ask(model, QUESTIONS, approximate=True)), got)
    assert engine.receipt()["reduced"]["graphs"]["replays"] > 0
    assert not torch.equal(got, want)
    assert (got - want).abs().max() < 0.05
    # the exact path is unchanged after the copy ran
    assert torch.equal(numbers(ask(model, QUESTIONS, approximate=False)), want)


def test_the_copy_runs_the_fast_path_the_exact_stack_runs(device, d3_text_package):
    _, engine = load_decision3(d3_text_package, device, max_speed=True)
    receipt = engine.receipt()
    if receipt.get("fused_layers"):
        assert receipt["reduced"]["fused_layers"] == receipt["fused_layers"]
    assert ("graphs" in receipt) == ("graphs" in receipt["reduced"])


def test_a_decoder_without_graphs_keeps_no_copy(device, d3_text_package):
    _, engine = load_decision3(d3_text_package, device, max_speed=True, graphs=False)
    assert "graphs" in engine.receipt()["reduced"]["skipped"]
    assert not engine.reduced


def test_the_copy_costs_less_than_the_exact_linear_layers(device, d3_text_package):
    _, exact = load_decision3(d3_text_package, device, max_speed=False)
    _, fp8 = load_decision3(d3_text_package, device, max_speed=True)
    copy_bytes = fp8.receipt()["reduced"]["bytes"]
    linear = sum(
        m.weight.numel() * m.weight.element_size()
        for m in exact.backbone.modules()
        if isinstance(m, nn.Linear)
    )
    assert 0 < copy_bytes < linear
    assert fp8.memory_bytes() == exact.memory_bytes() + copy_bytes
    assert fp8.parameter_count() == exact.parameter_count()
