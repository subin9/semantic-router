"""Reduced copies of an encoder's linear layers (the max_speed profile)."""

import pytest
import torch
from torch import nn
from vllm_srun.accel.onednn import PackedLinear
from vllm_srun.engines.native.reduced import (
    linear_bytes,
    reduced_view,
    unavailable,
)

from .test_modernbert import CONFIGS, batch, native

CPU = torch.device("cpu")


def runnable(kind):
    reason = unavailable(kind, CPU)
    if reason:
        pytest.skip(reason)


@pytest.mark.parametrize("kind", ["bfloat16", "int8", "float32-packed"])
def test_a_view_shares_everything_but_its_linear_layers(kind):
    runnable(kind)
    backbone, _ = native(CONFIGS["yarn"], seed=1)
    exact = {name: value.clone() for name, value in backbone.state_dict().items()}
    view = reduced_view(backbone, kind)
    assert (
        view.embeddings.tok_embeddings.weight
        is backbone.embeddings.tok_embeddings.weight
    )
    assert view.layers[1].attn_norm.weight is backbone.layers[1].attn_norm.weight
    assert view.final_norm.weight is backbone.final_norm.weight
    assert view.layers[0].attn.Wqkv is not backbone.layers[0].attn.Wqkv
    assert not any(
        isinstance(m, nn.Linear) and m.weight.dtype == torch.float32
        for m in view.modules()
    )
    assert linear_bytes(view) > 0
    for name, value in backbone.state_dict().items():
        assert torch.equal(value, exact[name]), name


@pytest.mark.parametrize(
    ("kind", "tolerance"), [("bfloat16", 0.1), ("int8", 0.1), ("float32-packed", 1e-4)]
)
def test_a_view_computes_close_to_the_exact_backbone(kind, tolerance):
    runnable(kind)
    backbone, _ = native(CONFIGS["yarn"], seed=2)
    view = reduced_view(backbone, kind)
    view.kernels = backbone.kernels
    lengths = [9, 3, 14]
    ids, mask = batch(lengths, seed=3)
    flat = ids[mask.bool()]
    layout = backbone.packed(lengths, "cpu")
    context = (
        torch.autocast("cpu", dtype=torch.bfloat16)
        if kind == "bfloat16"
        else torch.no_grad()
    )
    with torch.inference_mode():
        expected = backbone.encode(flat, layout)[backbone.num_layers]
        with context:
            actual = view.encode(flat, layout)[backbone.num_layers]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)


def test_packed_linears_give_a_row_the_same_result_in_any_batch():
    runnable("float32-packed")
    torch.manual_seed(0)
    linear = PackedLinear(nn.Linear(64, 96))
    rows = torch.randn(300, 64)
    with torch.inference_mode():
        alone = linear(rows[:3])
        for count in (4, 17, 64, 300):
            assert torch.equal(linear(rows[:count])[:3], alone)
        assert linear(rows[None, :5]).shape == (1, 5, 96)


def test_copies_that_cannot_run_on_a_device_say_why():
    assert "unknown" in unavailable("fp4", CPU)
    assert "CPUs" in unavailable("int8", torch.device("cuda"))
    assert "CPUs" in unavailable("float32-packed", torch.device("cuda"))
    assert unavailable("bfloat16", torch.device("cuda")) is None


@pytest.mark.parametrize("profile_asks", [True, False])
def test_the_engine_runs_reduced_batches_on_the_copy_only(tmp_path, profile_asks):
    runnable("int8")
    from vllm_srun.accel.cpu import CPUAccelerator
    from vllm_srun.engines.native.engine import NativeEngine
    from vllm_srun.plugins.base import (
        BackboneSpec,
        DtypePolicy,
        EncoderBatch,
        EngineOptions,
        ModelSpec,
    )
    from vllm_srun.testing.fixtures import random_backbone, save

    config = CONFIGS["yarn"]
    state = random_backbone("modernbert", config, seed=7)
    save({f"model.{n}": v for n, v in state.items()}, tmp_path / "model.safetensors")
    spec = ModelSpec(
        "tiny",
        BackboneSpec("modernbert", config, (tmp_path / "model.safetensors",), "model."),
        DtypePolicy(autocast=None, bf16_resident=False, reduced_cpu="int8"),
        max_input_tokens=4096,
        encoder=True,
    )
    accelerator = CPUAccelerator()
    model = NativeEngine().load(
        spec,
        accelerator,
        accelerator.devices()[0],
        EngineOptions(
            reduced_precision=profile_asks, exact_kernels_only=not profile_asks
        ),
    )
    lengths = [9, 3]
    ids, mask = batch(lengths, seed=4)
    flat = ids[mask.bool()]
    exact = model.encode(EncoderBatch(flat, None, lengths=lengths)).hidden[4]
    reduced = model.encode(
        EncoderBatch(flat, None, lengths=lengths, reduced=True)
    ).hidden[4]
    again = model.encode(EncoderBatch(flat, None, lengths=lengths)).hidden[4]
    assert torch.equal(again, exact)
    assert (model.receipt().get("reduced", {}).get("kind") == "int8") is profile_asks
    assert torch.equal(reduced, exact) is not profile_asks
    torch.testing.assert_close(reduced, exact, rtol=0, atol=0.1)


def test_a_float8_copy_needs_a_gpu_with_fp8_matrix_units():
    assert "GPU" in unavailable("float8", CPU)


def test_a_view_shares_the_layers_the_records_keep_exact():
    runnable("bfloat16")
    from vllm_srun.engines.native.reduced import copy_counts

    backbone, _ = native(CONFIGS["yarn"], seed=5)
    view = reduced_view(backbone, "bfloat16", exact=("layers.0.*", "layers.*.mlp.Wo"))
    kept = [
        name
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Linear)
        and (name.startswith("layers.0.") or name.endswith(".mlp.Wo"))
    ]
    total = sum(isinstance(m, nn.Linear) for m in backbone.modules())
    assert len(kept) > len(backbone.layers)
    for name in kept:
        assert view.get_submodule(name) is backbone.get_submodule(name)
    assert view.layers[1].attn.Wqkv.weight.dtype == torch.bfloat16
    assert copy_counts(view, backbone) == {
        "reduced": total - len(kept),
        "exact": len(kept),
    }
    # the shared layers count once, with the backbone
    assert linear_bytes(view, backbone) < linear_bytes(view)


def decision3_model(package, reduced_cpu, options):
    """A decision3 fixture package on the CPU, its policy consenting to ``reduced_cpu``."""
    import dataclasses

    from vllm_srun.accel.cpu import CPUAccelerator
    from vllm_srun.engines.native.engine import NativeEngine
    from vllm_srun.families.decision3.family import Decision3Family
    from vllm_srun.plugins.base import PackageRef, RegistryOptions

    family = Decision3Family(RegistryOptions(offline=True))
    verified = family.verify(PackageRef(root=package))
    spec = family.describe(verified)
    spec = dataclasses.replace(
        spec, dtype=dataclasses.replace(spec.dtype, reduced_cpu=reduced_cpu)
    )
    accelerator = CPUAccelerator()
    return NativeEngine().load(spec, accelerator, accelerator.devices()[0], options)


@pytest.mark.parametrize(
    ("kind", "receipt"),
    [
        # A decoder takes only the decoder kinds: another leaves it exact without a word.
        ("bfloat16", None),
        # A decoder kind the device cannot run is skipped, and the receipt says why.
        ("float8", "GPU"),
    ],
)
def test_a_decoder_takes_only_decoder_copies(d3_text_package, kind, receipt):
    from vllm_srun.plugins.base import EngineOptions

    engine = decision3_model(
        d3_text_package,
        kind,
        EngineOptions(reduced_precision=True, exact_kernels_only=False),
    )
    reduced = engine.receipt().get("reduced")
    if receipt is None:
        assert reduced is None
    else:
        assert reduced["kind"] == kind and receipt in reduced["skipped"]
    assert engine.reduced == {}


def test_the_exact_profile_never_loads_a_copy(d3_text_package):
    from vllm_srun.plugins.base import EngineOptions

    engine = decision3_model(d3_text_package, "float8", EngineOptions())
    assert "reduced" not in engine.receipt()
    assert engine.reduced == {}


def test_decision3_takes_its_copy_and_exact_layers_from_the_registry(
    d3_text_package, monkeypatch
):
    from vllm_srun.families.decision3.family import Decision3Family
    from vllm_srun.plugins.base import PackageRef, RegistryOptions
    from vllm_srun.registry import builtin
    from vllm_srun.registry.tables.common import BuiltinModel

    family = Decision3Family(RegistryOptions(offline=True))
    verified = family.verify(PackageRef(root=d3_text_package))
    assert family.describe(verified).dtype.reduced_gpu is None
    entry = BuiltinModel(
        repo_id="fixture/d3",
        revision="0" * 40,
        family="decision3",
        model_sha256=verified.model_sha256,
        manifest_sha256="",
        loaded_parameters=0,
        backbone="qwen3_5_text",
        min_device_memory_gib=1,
        reduced={"gpu": "float8"},
        reduced_exact=("layers.*.linear_attn.in_proj_qkv",),
    )
    monkeypatch.setattr(
        builtin,
        "by_identity",
        lambda sha: entry if sha == verified.model_sha256 else None,
    )
    dtype = family.describe(verified).dtype
    assert dtype.reduced_gpu == "float8"
    assert dtype.reduced_cpu is None
    assert dtype.reduced_exact == ("layers.*.linear_attn.in_proj_qkv",)
