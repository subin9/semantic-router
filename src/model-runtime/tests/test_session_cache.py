"""Session cache: a growing state resumed from the backbone state an earlier request saved (``session.py``)."""

from __future__ import annotations

import asyncio
import math

import pytest
import torch
from vllm_srun.config import EngineArgs, ModelConfig, ServeConfig
from vllm_srun.engines.native.models.tree import Tree, tree_attention
from vllm_srun.engines.native.session import SessionCache
from vllm_srun.runtime import Runtime
from vllm_srun.scheduler.batching import SharedContext
from vllm_srun.testing.fixtures import write_package

# A conversation that grows call by call: each call's state is the previous one with more text after it.
TURNS = [
    "User: The blender I ordered (order 48213) arrived with a cracked jug and a scratched base. ",
    "Agent: Sorry about that. Please send photos of the box and the jug and we will arrange a replacement. ",
    "User: Photos attached. The crack runs along the handle; the base has two deep scratches. ",
    "Agent: Thanks, the photos are clear. A replacement was requested and usually ships in one business day. ",
    "User: Still no tracking number, and my card shows two express shipping charges of 14.99 each. ",
    "Agent: I can see the duplicate charge and asked billing to refund it; refunds take three to five days. ",
    "User: Nothing has shipped yet and I need the blender by Saturday for a family event. ",
]
QUESTIONS = {
    "next": {
        "type": "choice",
        "instructions": "What should support do next?",
        "criteria": {
            "refund": "refund the order",
            "replace": "expedite the replacement",
            "escalate": "hand the case to a human supervisor",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Does this need a reply within the hour?",
    },
    "mood": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "angry"],
    },
}
# Small enough that the tiny fixtures' states save and resume after a turn or two.
POLICY = {"stride": 16, "margin": 4, "min_tokens": 48}


def states(rounds: int = 3) -> list[str]:
    """The conversation's state at each call, longer each time."""
    text = "".join(TURNS * rounds)
    cuts = [len(text) * k // 6 for k in range(2, 7)]
    return [text[:cut] for cut in cuts]


def serve(package, **args) -> Runtime:
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=str(package), device="cpu", **args),),
            result_cache_entries=0,
        )
    )
    runtime.start(background=False)
    return runtime


def cache_of(runtime: Runtime) -> SessionCache:
    session = runtime.served[0].model.engine_model.session
    assert session is not None
    for name, value in POLICY.items():
        setattr(session, name, value)
    return session


def decide(runtime: Runtime, state: str, questions: dict) -> dict:
    status, body = asyncio.run(
        runtime.call("decisions", {"state": state, "questions": questions})
    )
    assert status == 200, body
    return body["answers"]


def flat(answers: dict) -> list[float]:
    """Every probability of a response, in question order."""
    out: list[float] = []
    for answer in answers.values():
        if answer["type"] == "noul":
            out.append(answer["noul"])
        else:
            out.extend(answer["probabilities"].values())
    return out


# The pure cache: marks, lookups, eviction


def test_marks_fall_on_the_stride_inside_the_state_and_past_what_resumed():
    cache = SessionCache(1 << 20, stride=4, margin=2, min_tokens=8)
    assert cache.mark(20, 0, 30) == 16
    assert cache.mark(9, 0, 30) == 0  # 7 rounds to 4, below the minimum
    assert cache.mark(20, 16, 30) == 0  # the resumed position already reaches it
    assert cache.mark(40, 0, 18) == 16  # never past the shared prefix
    assert cache.mark(0, 0, 30) == 0


def test_a_lookup_resumes_the_furthest_matching_state_and_leaves_a_token_to_compute():
    cache = SessionCache(1 << 20, stride=4, margin=2, min_tokens=6)
    tokens = list(range(100, 140))
    short = [{"key": torch.zeros(1, 1, 8, 2)}]
    long = [{"key": torch.zeros(1, 1, 16, 2)}]
    cache.insert(tokens[:8], short)
    cache.insert(tokens[:16], long)
    entry, start = cache.match(tokens[:30])
    assert start == 16 and entry is not None and entry.layers is long
    entry, start = cache.match(
        tokens[:16]
    )  # the whole prefix saved: resume the shorter one
    assert start == 8 and entry is not None and entry.layers is short
    assert cache.match([0] + tokens[1:30]) == (None, 0)  # a different first token
    assert cache.match(tokens[:6]) == (None, 0)  # at most the minimum: no lookup
    receipt = cache.receipt()
    assert receipt["hits"] == 2 and receipt["misses"] == 2
    assert receipt["resumed_tokens"] == 24 and receipt["entries"] == 2
    with pytest.raises(ValueError, match="multiple of the stride"):
        cache.insert(tokens[:5], short)


def test_entries_are_evicted_least_recently_used_under_the_byte_budget():
    layer = [{"key": torch.zeros(256, dtype=torch.float32)}]  # 1 KiB
    cache = SessionCache(2048 + 512, stride=4, margin=0, min_tokens=4)
    a, b, c = ([n] * 8 for n in (1, 2, 3))
    cache.insert(a, layer)
    cache.insert(b, layer)
    assert cache.match(a + [0])[1] == 8  # a is now the most recent
    cache.insert(c, layer)
    assert cache.match(b + [0]) == (None, 0)
    assert cache.match(a + [0])[1] == 8 and cache.match(c + [0])[1] == 8
    assert cache.receipt()["evictions"] == 1 and cache.used == 2048
    cache.insert([4] * 8, [{"key": torch.zeros(1024, dtype=torch.float32)}])
    assert cache.match([4] * 9) == (None, 0)  # larger than the whole budget: not kept


# The tree forward: a resumed tree computes what the whole tree computes


def test_tree_positions_continue_from_the_resumed_tokens():
    cpu = torch.device("cpu")
    whole = Tree(5, [2, 3], 4, cpu, False)
    assert whole.positions[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 5, 6, 7]
    resumed = Tree(2, [2, 3], 4, cpu, False, start=3, resume=[{}])
    assert resumed.positions[0].tolist() == [3, 4, 5, 6, 5, 6, 7]
    with pytest.raises(ValueError, match="exactly when"):
        Tree(2, [2], 4, cpu, False, start=3)
    with pytest.raises(ValueError, match="inside the prefix"):
        Tree(2, [2], 4, cpu, False, start=3, resume=[{}], mark=6)


def test_tree_attention_over_resumed_keys_equals_the_whole_tree():
    torch.manual_seed(0)
    prefix, start, lengths = 40, 16, [5, 7]
    total = prefix + sum(lengths)
    q, k, v = (torch.randn(1, heads, total, 8) for heads in (4, 2, 2))
    cpu = torch.device("cpu")
    whole = tree_attention(
        q, k, v, Tree(prefix, lengths, 8, cpu, False), groups=2, scaling=0.3
    )
    saved = [{"key": k[:, :, :start], "value": v[:, :, :start]}]
    tree = Tree(prefix - start, lengths, 8, cpu, False, start=start, resume=saved)
    resumed = tree_attention(
        q[:, :, start:], k[:, :, start:], v[:, :, start:], tree, groups=2, scaling=0.3
    )
    assert torch.allclose(resumed[0], whole[0, start:], atol=1e-6)


@pytest.mark.parametrize("backbone", ["qwen3", "qwen3_5"])
def test_a_tree_resumed_from_its_saved_state_matches_the_whole_tree(tmp_path, backbone):
    runtime = serve(write_package(tmp_path / backbone, backbone=backbone, seed=5))
    try:
        layers = runtime.served[0].model.engine_model.backbone
        torch.manual_seed(0)
        prefix, mark, lengths = 200, 128, [20, 10]
        ids = torch.randint(5, 300, (prefix + sum(lengths),))
        cpu = torch.device("cpu")

        def run(
            start: int, resume: list | None, save: int
        ) -> tuple[torch.Tensor, Tree]:
            tree = Tree(
                prefix - start,
                lengths,
                32,
                cpu,
                False,
                start=start,
                resume=resume,
                mark=save,
            )
            with torch.inference_mode():
                return layers.forward_tree(ids[start:][None], tree)[0], tree

        whole, _ = run(0, None, 0)
        saving, tree = run(0, None, mark)
        if backbone == "qwen3":
            assert torch.equal(saving, whole)  # keeping keys changes no number
        else:
            # The recurrence restarts its chunks at the mark: rounding moves.
            assert torch.allclose(saving, whole, atol=1e-5)
        assert len(tree.saved) == len(layers.layers)
        resumed, _ = run(mark, tree.saved, 0)
        assert torch.allclose(resumed, whole[mark:], atol=1e-5)
    finally:
        runtime.stop()


# Through the runtime: the same answers with the cache, on both decoder backbones


@pytest.mark.parametrize("backbone", ["qwen3", "qwen3_5"])
def test_a_growing_state_resumes_and_answers_as_a_fresh_forward(tmp_path, backbone):
    package = write_package(tmp_path / backbone, backbone=backbone, seed=7)
    fresh = serve(package)
    cached = serve(package, session_cache_gib=0.25)
    try:
        cache = cache_of(cached)
        for state in states():
            for questions in (QUESTIONS, {"next": QUESTIONS["next"]}):
                want = flat(decide(fresh, state, questions))
                got = flat(decide(cached, state, questions))
                assert got == pytest.approx(want, abs=2e-5)
        receipt = cache.receipt()
        assert receipt["saves"] >= 4 and receipt["hits"] >= 4
        assert receipt["resumed_tokens"] > 0 and receipt["bytes"] > 0
        card = cached.served[0].card([])
        assert card["engine_args"]["session_cache_gib"] == 0.25
    finally:
        fresh.stop()
        cached.stop()


def test_a_single_question_runs_as_a_tree_only_once_its_state_is_long_enough(
    qwen35_package,
):
    runtime = serve(qwen35_package, session_cache_gib=0.25)
    try:
        cache_of(runtime)
        model = runtime.served[0].model
        sharing = runtime.served[0].scheduler.batcher.shared
        assert isinstance(sharing, SharedContext)
        short = model.plan("A short state.", {"next": QUESTIONS["next"]}).items
        long = model.plan(states()[-1], {"next": QUESTIONS["next"]}).items
        assert long[0].state_tokens >= POLICY["min_tokens"] + POLICY["margin"]
        assert sharing.share(list(short), None) == 0
        prefix = sharing.share(list(long), None)
        assert long[0].state_tokens <= prefix <= min(long[0].gather)
    finally:
        runtime.stop()


def test_the_state_length_ends_where_the_question_text_begins(qwen35_package):
    runtime = serve(qwen35_package)
    try:
        model = runtime.served[0].model
        state = states()[0]
        (item,) = model.plan(state, {"next": QUESTIONS["next"]}).items
        alone = model.tokenizer.encode("Context:\n" + state)
        assert abs(item.state_tokens - len(alone)) <= 2
        assert item.ids[: item.state_tokens - 2] == alone[: item.state_tokens - 2]
    finally:
        runtime.stop()


# Decision 3.0 state-once packages: the state's tree under the package's layout


def d3_runtime(tmp_path, layout, gib):
    from vllm_srun.testing.decision3 import write_fixture

    package = write_fixture(tmp_path / f"d3-{layout}-{gib}", layout, seed=31)
    args = {"session_cache_gib": gib} if gib else {}
    return serve(package, dtype="float32", **args)


@pytest.mark.parametrize("layout", ["prefix_causal", "prefix_bidir", "causal"])
def test_a_state_once_tree_resumed_from_its_saved_state_matches_the_whole_tree(
    tmp_path, layout
):
    runtime = d3_runtime(tmp_path, layout, 0)
    try:
        layers = runtime.served[0].model.engine_model.backbone
        assert layers.layout == layout
        torch.manual_seed(0)
        prefix, lengths = 120, [9, 14, 6]
        # A state read both ways is saved and resumed whole; a causal one inside.
        mark = prefix if layout == "prefix_bidir" else 64
        ids = torch.randint(5, 300, (prefix + sum(lengths),))
        flags = {
            "state_bidirectional": layout == "prefix_bidir",
            "block_bidirectional": layout != "causal",
        }
        cpu = torch.device("cpu")

        def run(
            start: int, resume: list | None, save: int
        ) -> tuple[torch.Tensor, Tree]:
            tree = Tree(
                prefix - start, lengths, 16, cpu, False, start=start, resume=resume, mark=save, **flags
            )  # fmt: skip
            with torch.inference_mode():
                return layers.forward_tree(ids[start:][None], tree)[0], tree

        whole, _ = run(0, None, 0)
        saving, tree = run(0, None, mark)
        assert torch.allclose(saving, whole, atol=1e-5)
        resumed, _ = run(mark, tree.saved, 0)
        assert torch.allclose(resumed, whole[mark:], atol=1e-5)
    finally:
        runtime.stop()


@pytest.mark.parametrize("layout", ["prefix_causal", "prefix_bidir"])
def test_a_state_once_session_answers_as_the_branched_forward(tmp_path, layout):
    plain = d3_runtime(tmp_path, layout, 0)
    cached = d3_runtime(tmp_path, layout, 0.25)
    try:
        cache = cache_of(cached)
        assert cache.whole == (layout == "prefix_bidir")
        asked = [
            QUESTIONS,
            {"next": QUESTIONS["next"]},
            {"urgent": QUESTIONS["urgent"]},
        ]
        for state in states():
            # Each state again with other questions: a whole state resumes on its repeat.
            for questions in asked:
                want = flat(decide(plain, state, questions))
                got = flat(decide(cached, state, questions))
                assert got == pytest.approx(want, abs=2e-5)
        receipt = cache.receipt()
        assert receipt["saves"] >= 4 and receipt["hits"] >= 8
        assert receipt["resumed_tokens"] > 0
    finally:
        plain.stop()
        cached.stop()


def test_an_agent_loop_state_resumes_as_its_tool_results_grow(tmp_path):
    plain = d3_runtime(tmp_path, "prefix_causal", 0)
    cached = d3_runtime(tmp_path, "prefix_causal", 0.25)
    try:
        cache = cache_of(cached)
        request, results = TURNS[0], TURNS[1:] * 3
        for n in range(1, len(results), 4):
            # The Router's parts arrive with sorted keys: the request, then the tool results that grow.
            state = {"request": request, "tool_results": "\n\n".join(results[:n])}
            want = flat(decide(plain, state, QUESTIONS))
            got = flat(decide(cached, state, QUESTIONS))
            assert got == pytest.approx(want, abs=2e-5)
        receipt = cache.receipt()
        # Every call after the first resumes the previous call's state.
        assert receipt["misses"] == 1 and receipt["hits"] == 4
    finally:
        plain.stop()
        cached.stop()


@pytest.mark.parametrize("layout", ["prefix_causal", "prefix_bidir"])
def test_a_merged_forward_runs_long_states_as_trees_and_short_ones_branched(
    tmp_path, layout
):
    plain = d3_runtime(tmp_path, layout, 0)
    cached = d3_runtime(tmp_path, layout, 0.25)
    try:
        cache_of(cached)
        long, short = states()[-1], "A short state about a refund."
        want, got = (
            [
                value
                for state in (long, short, long)
                for value in runtime.served[0].model.plan(state, QUESTIONS).items
            ]
            for runtime in (plain, cached)
        )
        # One run over the items of three requests, as merged batches do: one long state twice, one short.
        expected = plain.served[0].model.run(want)
        merged = cached.served[0].model.run(got)
        assert len(merged) == len(expected)
        for left, right in zip(merged, expected, strict=True):
            assert left == pytest.approx(right, abs=2e-5)
        assert cached.served[0].model.engine_model.session.receipt()["saves"] >= 1
    finally:
        plain.stop()
        cached.stop()


def test_a_released_decision_3_package_refuses_the_session_cache(tmp_path):
    from vllm_srun.testing.decision3 import write_fixture

    package = write_fixture(tmp_path / "d3", "text", seed=31)
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(model=str(package), device="cpu", session_cache_gib=0.25),
            )
        )
    )
    try:
        with pytest.raises(RuntimeError, match="whole prompt both ways"):
            runtime.start(background=False)
    finally:
        runtime.stop()


# Engine arguments


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan, True, "1"])
def test_session_cache_gib_must_be_a_finite_number_of_zero_or_more(value):
    with pytest.raises(ValueError, match="session_cache_gib"):
        EngineArgs(session_cache_gib=value)


def test_the_session_cache_is_refused_without_the_shared_context_forward(
    qwen3_package,
):
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=str(qwen3_package),
                    device="cpu",
                    enable_shared_context=False,
                    session_cache_gib=0.25,
                ),
            )
        )
    )
    try:
        with pytest.raises(RuntimeError, match="needs the shared-context forward"):
            runtime.start(background=False)
    finally:
        runtime.stop()


def test_the_session_cache_is_refused_for_a_model_with_no_tree_forward(tmp_path):
    from vllm_srun.testing.vela2 import write_encoder_package

    package = write_encoder_package(tmp_path / "vela2")
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(model=str(package), device="cpu", session_cache_gib=0.25),
            )
        )
    )
    try:
        with pytest.raises(RuntimeError, match="session cache"):
            runtime.start(background=False)
    finally:
        runtime.stop()


# GPU: the fused gated-delta tree (fast.py) resumes as the eager one does


def _fused_device():
    """A GPU whose fused kernels run (CUDA Ampere or newer, approximate there; gfx942, exact), else skip."""
    from vllm_srun.accel.cuda import CUDAAccelerator
    from vllm_srun.accel.rocm import ROCmAccelerator

    for accelerator in (ROCmAccelerator(), CUDAAccelerator()):
        if not accelerator.available():
            continue
        device = accelerator.devices()[0]
        if device.arch == "gfx942" or (
            (device.arch or "").startswith("sm_")
            and int(device.arch.removeprefix("sm_")) >= 80
        ):
            return accelerator, device
    pytest.skip("needs a GPU the fused kernels run on")


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["causal", "prefix_causal", "prefix_bidir"])
def test_the_fused_gated_delta_tree_resumes_as_the_whole_tree(monkeypatch, layout):
    from vllm_srun.engines.native import fast

    from .test_gpu_fast_path import build, qwen3_5

    accelerator, device = _fused_device()
    module = build(qwen3_5(4, 1024, 16, 8), 7, False, accelerator, device)
    module.kernels.allow_approximate = True
    if fast.fused_unavailable(module, module.kernels) is not None:
        pytest.skip(fast.fused_unavailable(module, module.kernels))
    assert fast.install_fused(module) == len(module.layers)
    taken = []
    fused = fast._tree_gated_delta
    monkeypatch.setattr(
        fast,
        "_tree_gated_delta",
        lambda m, p, normed, tree, kernels: taken.append(tree.session)
        or fused(m, p, normed, tree, kernels),
    )
    torch.manual_seed(0)
    prefix, lengths = 700, [40, 25, 60]
    # A state read both ways is saved and resumed whole; a causal one inside.
    mark = prefix if layout == "prefix_bidir" else 512
    flags = {
        "state_bidirectional": layout == "prefix_bidir",
        "block_bidirectional": layout != "causal",
    }
    ids = torch.randint(5, 1000, (prefix + sum(lengths),))
    gpu = accelerator.torch_device(device)

    def run(start: int, resume: list | None, save: int) -> tuple[torch.Tensor, Tree]:
        tree = Tree(
            prefix - start,
            lengths,
            64,
            gpu,
            False,
            start=start,
            resume=resume,
            mark=save,
            **flags,
        )
        with torch.inference_mode(), torch.autocast(gpu.type, dtype=torch.bfloat16):
            return module.forward_tree(ids[start:][None].to(gpu), tree)[0].float(), tree

    whole, _ = run(0, None, 0)
    saving, tree = run(0, None, mark)
    resumed, _ = run(mark, tree.saved, 0)
    assert len(tree.saved) == len(module.layers) and any(taken) and not all(taken)
    scale = whole.abs().max().item()
    # BF16: the resumed rows round as another packing of the same rows would.
    assert (saving - whole).abs().max().item() <= 2e-2 * scale
    assert (resumed - whole[mark:]).abs().max().item() <= 2e-2 * scale


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["prefix_causal", "prefix_bidir"])
def test_a_state_once_session_answers_as_the_branched_forward_on_the_gpu(
    tmp_path, layout
):
    from vllm_srun.testing.decision3 import write_fixture

    _, device = _fused_device()
    name = f"{'rocm' if device.arch == 'gfx942' else 'cuda'}:{device.index}"
    package = write_fixture(tmp_path / layout, layout, seed=31)
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(model=str(package), device=name, name="plain"),
                ModelConfig(
                    model=str(package),
                    device=name,
                    name="cached",
                    session_cache_gib=0.25,
                ),
            ),
            result_cache_entries=0,
        )
    )
    runtime.start(background=False)
    try:
        cached = next(s for s in runtime.served if s.config.name == "cached")
        session = cached.model.engine_model.session
        assert session is not None
        for key, value in POLICY.items():
            setattr(session, key, value)
        for state in states():
            for questions in (QUESTIONS, {"next": QUESTIONS["next"]}):
                answers = {}
                for model in ("plain", "cached"):
                    body = {"model": model, "state": state, "questions": questions}
                    status, answer = asyncio.run(runtime.call("decisions", body))
                    assert status == 200, answer
                    answers[model] = flat(answer["answers"])
                # BF16: a tree and a branched forward round apart (test_state_once.py BF16_TOLERANCE).
                assert answers["cached"] == pytest.approx(answers["plain"], abs=0.03)
        assert session.receipt()["hits"] >= 4
    finally:
        runtime.stop()
