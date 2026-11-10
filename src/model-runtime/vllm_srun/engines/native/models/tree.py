"""Shared-context tree forward: a multi-question request's shared prefix computed once.

The prefix and every question's suffix are packed into one row with the exact
path's positions. Full-attention layers: the prefix attends causally to itself;
every suffix query attends to the prefix keys (one kernel for all queries) and
to its own suffix (one causal kernel over the padded suffix rows), and the two
results are merged by their log-sum-exp. Gated-delta layers: the prefix runs
from a zero state; the suffixes run as variable-length sequences from the
prefix-end recurrent state, each convolution window starting with the prefix's
last inputs. These are the operations of the released runtime's shared-context
switch (``decision2/shared_ctx.py`` tree mode), so the answers equal it; they
can differ from the request's own batches by rounding (``--enable-shared-context``).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from ....accel.kernels import KernelSet
    from .qwen3_5 import GatedDeltaNet


class Tree:
    """A request packed into one row: the shared prefix, then every suffix in order.

    ``rows`` / ``packed`` move suffix tokens between the packed row and a padded
    [questions, width] layout. Padding sits after each suffix and repeats its last
    token: causal attention, convolution and recurrence never carry it into a real
    token, and the readout never reads it.

    Session cache (``session.py``): ``resume`` holds the backbone state after the
    first ``start`` prefix tokens, one item per layer, and the row holds only the
    prefix tokens after them (``prefix`` counts those). ``mark`` > ``start`` asks
    every layer to keep its state after the first ``mark`` prefix tokens in
    ``saved``. Every layer takes its item with ``layer()`` once per forward, in
    layer order.

    A state-once package's tree (``models/branch.py`` layouts) reads its prefix
    both ways (``state_bidirectional``, ``prefix_bidir``) and every suffix both
    ways within itself (``block_bidirectional``, both ``prefix_*`` layouts);
    recurrent layers stay causal. A tree that resumes a whole state has no
    prefix tokens left in its row (``prefix`` 0).
    """

    is_tree = True

    def __init__(
        self,
        prefix: int,
        lengths: list[int],
        width: int,
        device: torch.device,
        padded_exact: bool,
        start: int = 0,
        resume: list[dict[str, torch.Tensor]] | None = None,
        mark: int = 0,
        state_bidirectional: bool = False,
        block_bidirectional: bool = False,
    ):
        if (start > 0) != (resume is not None):
            raise ValueError(
                "a tree resumes from a saved state exactly when it starts past 0"
            )
        if mark and not start < mark <= start + prefix:
            raise ValueError("a tree saves its state inside the prefix it computes")
        if state_bidirectional and (
            (start and prefix) or (mark and mark != start + prefix)
        ):
            raise ValueError("a state read both ways resumes and saves whole")
        if prefix < 1 and not start:
            raise ValueError("a tree needs a prefix token or a resumed state")
        self.state_bidirectional = state_bidirectional
        self.block_bidirectional = block_bidirectional
        self.prefix = prefix
        self.start = start
        self.resume = resume
        self.mark = mark
        self.saved: list[dict[str, torch.Tensor]] = []
        self._layer = 0
        self.lengths = lengths
        starts = [0]
        for length in lengths[:-1]:
            starts.append(starts[-1] + length)
        index = torch.empty((len(lengths), width), dtype=torch.long)
        valid = torch.zeros((len(lengths), width), dtype=torch.bool)
        for row, (offset, length) in enumerate(zip(starts, lengths, strict=False)):
            index[row, :length] = torch.arange(offset, offset + length)
            index[row, length:] = offset + length - 1
            valid[row, :length] = True
        self.index = index.to(device)
        # Index (not mask) gathers: a boolean mask would sync the host on every use.
        self.flat = valid.reshape(-1).nonzero().squeeze(1).to(device)
        ends = [
            offset + length for offset, length in zip(starts, lengths, strict=False)
        ]
        self.cu_seqlens_cpu = torch.tensor([0, *ends], dtype=torch.long)
        self.cu_seqlens = self.cu_seqlens_cpu.to(device)
        self._ends = ends
        self._device = device
        self._block_keys: tuple[Any, ...] | None = None
        end = start + prefix
        self.positions = torch.cat(
            [torch.arange(start, end)]
            + [torch.arange(end, end + length) for length in lengths]
        )[None].to(device)
        # The exact batch of a request whose rows differ in padded length runs SDPA with an explicit mask; the
        # prefix keeps that mask regime.
        self.prefix_mask = (
            torch.ones(prefix, prefix, dtype=torch.bool, device=device).tril()[
                None, None
            ]
            if padded_exact
            else None
        )

    def block_keys(self, context: int) -> tuple[Any, ...]:
        """For varlen attention over each suffix's keys (all ``context`` prefix keys, then its own, in a key row
        of the context and the suffixes): the key index, the query and key boundaries (device int32, host) and
        the longest query and key segments. The same for every layer of a tree, so built once.
        """
        if self._block_keys is None:
            index: list[int] = []
            keys = [0]
            for start, end in zip([0, *self._ends[:-1]], self._ends, strict=True):
                index += range(context)
                index += range(context + start, context + end)
                keys.append(len(index))
            queries = [0, *self._ends]
            self._block_keys = (
                torch.tensor(index, dtype=torch.long).to(self._device),
                torch.tensor(queries, dtype=torch.int32).to(self._device),
                queries,
                max(self.lengths),
                torch.tensor(keys, dtype=torch.int32).to(self._device),
                keys,
                context + max(self.lengths),
            )
        return self._block_keys

    def layer(self) -> dict[str, torch.Tensor] | None:
        """The next layer's resumed state (None when the tree starts at 0); call once per layer, in order."""
        index = self._layer
        self._layer += 1
        return None if self.resume is None else self.resume[index]

    @property
    def session(self) -> bool:
        """Whether the tree resumes or saves a session state, so its layers take the session path."""
        return self.resume is not None or self.mark > 0

    def keep(self, state: dict[str, torch.Tensor]) -> None:
        """Save the current layer's state at the mark (dense copies, so the forward's tensors can be freed)."""
        self.saved.append(
            {
                k: v.clone(memory_format=torch.contiguous_format)
                for k, v in state.items()
            }
        )

    def rows(self, packed: torch.Tensor) -> torch.Tensor:
        """[suffix tokens, ...] -> [questions, width, ...]."""
        return packed[self.index]

    def packed(self, rows: torch.Tensor) -> torch.Tensor:
        """[questions, width, ...] -> [suffix tokens, ...]."""
        return rows.reshape(-1, *rows.shape[2:])[self.flat]


def is_tree(mask: object) -> TypeIs[Tree]:
    """Whether a layer's mask is a shared-context ``Tree``, which runs the tree forward."""
    return bool(getattr(mask, "is_tree", False))


def _attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention output and its log-sum-exp per query ([B, H, L, D], [B, H, L])."""
    if query.is_cuda:
        out = torch.ops.aten._scaled_dot_product_efficient_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            None,
            True,
            0.0,
            causal,
            scale=scale,
        )
        return out[0], out[1][..., : query.shape[2]]
    scores = (query.float() @ key.float().transpose(-1, -2)) * scale
    if causal:
        q, k = scores.shape[-2:]
        keep = torch.ones(q, k, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~keep, -float("inf"))
    lse = scores.logsumexp(-1)
    out = (scores - lse[..., None]).exp() @ value.float()
    return out.to(query.dtype), lse


def tree_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tree: Tree,
    *,
    groups: int,
    scaling: float,
) -> torch.Tensor:
    """Attention over a packed tree ([1, H, L, D] inputs); returns [1, L, H, D] like ``common.attention``.

    A resumed tree's saved keys and values come first (``c`` of them): the
    prefix queries see them all and their own keys causally, the suffix queries
    every prefix key. A state-once tree reads its prefix (``state_bidirectional``)
    or its suffixes (``block_bidirectional``) both ways."""
    saved = tree.layer()
    dtype = query.dtype
    if query.is_cuda and torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
    p = tree.prefix
    if tree.session:
        key, value = _session_keys(key.to(dtype), value.to(dtype), tree, saved)
    c = key.shape[2] - query.shape[2]
    grouped = key, value
    if groups > 1:
        b, h, n, d = key.shape
        key = key[:, :, None].expand(b, h, groups, n, d).reshape(b, h * groups, n, d)
        value = (
            value[:, :, None].expand(b, h, groups, n, d).reshape(b, h * groups, n, d)
        )
    q, k, v = query.to(dtype), key.to(dtype), value.to(dtype)
    if not p:
        head = q[:, :, :0]
    elif c:
        head = _merge(
            _attend(q[:, :, :p], k[:, :, c : c + p], v[:, :, c : c + p], True, scaling),
            _attend(q[:, :, :p], k[:, :, :c], v[:, :, :c], False, scaling),
        ).to(dtype)
    else:
        causal = tree.prefix_mask is None and not tree.state_bidirectional
        head = F.scaled_dot_product_attention(
            q[:, :, :p], k[:, :, :p], v[:, :, :p], attn_mask=tree.prefix_mask, is_causal=causal, scale=scaling,
        )  # fmt: skip
    if tree.block_bidirectional:
        tail = _block_attention(q[:, :, p:], *grouped, c + p, tree, scaling, groups)
        return torch.cat([head[0].transpose(0, 1), tail.to(dtype)], dim=0)[None]
    out_prefix, lse_prefix = _attend(
        q[:, :, p:], k[:, :, : c + p], v[:, :, : c + p], False, scaling
    )

    def rows(x: torch.Tensor) -> torch.Tensor:
        return tree.rows(x[0].transpose(0, 1)).permute(0, 2, 1, 3)

    out_own, lse_own = _attend(
        rows(q[:, :, p:]), rows(k[:, :, c + p :]), rows(v[:, :, c + p :]), True, scaling
    )
    out_own = tree.packed(out_own.transpose(1, 2))
    lse_own = tree.packed(lse_own.transpose(1, 2))
    # softmax over [prefix; own] = the two parts weighted by sigmoid of their lse gap
    weight = torch.sigmoid(lse_prefix[0].transpose(0, 1) - lse_own)[..., None]
    tail = torch.lerp(out_own.float(), out_prefix[0].transpose(0, 1).float(), weight)
    out = torch.cat([head[0].transpose(0, 1), tail.to(dtype)], dim=0)
    return out[None]


def _block_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    context: int,
    tree: Tree,
    scaling: float,
    groups: int,
) -> torch.Tensor:
    """Every suffix query ([1, H, S, D]) over all ``context`` prefix keys and its own suffix's keys both ways, as
    a state-once block reads them, in one variable-length call (``branch._attend``: no padding, keys with
    ``H / groups`` heads); the packed outputs [S, H, D]."""
    from .branch import _attend

    index, cu_q, host_q, max_q, cu_k, host_k, max_k = tree.block_keys(context)
    q = query[0].transpose(0, 1).contiguous()
    k = key[0].transpose(0, 1)[index].contiguous()
    v = value[0].transpose(0, 1)[index].contiguous()
    return _attend(
        q, k, v, (cu_q, host_q, max_q), (cu_k, host_k, max_k), causal=False, scale=scaling, groups=groups,
    )  # fmt: skip


def _merge(
    own: tuple[torch.Tensor, torch.Tensor], other: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """One softmax over two key sets from their outputs and log-sum-exps ([B, H, L, D], [B, H, L])."""
    weight = torch.sigmoid(other[1] - own[1])[..., None]
    return torch.lerp(own[0].float(), other[0].float(), weight)


def _session_keys(
    key: torch.Tensor,
    value: torch.Tensor,
    tree: Tree,
    saved: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A layer's keys and values ([1, kv, L, D]) after the resumed ones, and its state at the mark kept."""
    if saved is not None:
        key = torch.cat([saved["key"].to(key.dtype), key], dim=2)
        value = torch.cat([saved["value"].to(value.dtype), value], dim=2)
    if tree.mark:
        tree.keep({"key": key[:, :, : tree.mark], "value": value[:, :, : tree.mark]})
    return key, value


def session_rows(
    x: torch.Tensor, saved: dict[str, torch.Tensor] | None, p: int, window: int
) -> torch.Tensor:
    """A session tree's prefix rows of ``x`` ([1, L, C]) after the ``window - 1`` convolution inputs before the
    row: the saved ones, or zeros at position 0 (what the convolution pads with)."""
    if saved is None:
        lead = x.new_zeros(1, window - 1, x.shape[-1])
    else:
        lead = saved["conv"].to(x.dtype)
    return torch.cat([lead, x[:, :p]], dim=1)


def prefix_rule(
    rule: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    tree: Tree,
    inputs: tuple[torch.Tensor, ...],
    saved: dict[str, torch.Tensor] | None,
    conv_rows: torch.Tensor,
    window: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """The gated delta rule over a session tree's prefix (q, k, v, g, beta of its row tokens) from the resumed
    state, cut at the mark, whose state and convolution inputs it keeps (``conv_rows`` holds the ``window - 1``
    inputs before the row, then the row's). Returns the prefix outputs (None for a tree that resumed its whole
    prefix) and the prefix-end state.
    """
    p = tree.prefix
    own = tree.mark - tree.start if tree.mark else p
    initial = None if saved is None else saved["S"]
    if not p:
        assert initial is not None
        return None, initial

    def run(
        lo: int, hi: int, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, g, beta = (x[:, lo:hi] for x in inputs)
        out, final = rule(
            q, k, v, g=g, beta=beta, initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        assert final is not None
        return out, final

    out, state = run(0, own, initial)
    if tree.mark:
        tree.keep({"S": state, "conv": conv_rows[:, own : own + window - 1]})
    if own < p:
        rest, state = run(own, p, state)
        out = torch.cat([out, rest], dim=1)
    return out, state


def varlen(rule: Any) -> bool:
    """Whether the bound chunked gated-delta kernel takes ``cu_seqlens`` (FLA)."""
    return "cu_seqlens" in inspect.signature(rule).parameters


def suffix_rule(
    rule: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    tree: Tree,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    start: torch.Tensor,
) -> torch.Tensor:
    """The gated delta rule over every suffix from the prefix-end state ``start`` ([1, S, ...] packed inputs)."""
    if varlen(rule):
        out, _ = rule(
            q, k, v, g=g, beta=beta, initial_state=start, output_final_state=False, use_qk_l2norm_in_kernel=True,
            cu_seqlens=tree.cu_seqlens, cu_seqlens_cpu=tree.cu_seqlens_cpu,
        )  # fmt: skip
        return out
    out_rows, _ = rule(
        tree.rows(q[0]), tree.rows(k[0]), tree.rows(v[0]), g=tree.rows(g[0]), beta=tree.rows(beta[0]),
        initial_state=start, output_final_state=False, use_qk_l2norm_in_kernel=True,
    )  # fmt: skip
    return tree.packed(out_rows)[None]


def tree_gated_delta(
    m: GatedDeltaNet, hidden_states: torch.Tensor, tree: Tree, kernels: KernelSet
) -> torch.Tensor:
    """``GatedDeltaNet.forward`` over a packed tree (no padding in the row), eager ops.

    A session tree's prefix convolution starts from the saved inputs before the
    row (zeros at position 0) and its recurrence from the saved state
    (``prefix_rule``)."""
    saved = tree.layer()
    _, length, _ = hidden_states.shape
    p, n = tree.prefix, len(tree.lengths)
    mixed = m.in_proj_qkv(hidden_states)
    z = m.in_proj_z(hidden_states).reshape(1, length, -1, m.head_v_dim)
    b = m.in_proj_b(hidden_states)
    a = m.in_proj_a(hidden_states)
    conv = kernels("causal_conv1d")
    weight, bias, window = m.conv1d.weight.squeeze(1), m.conv1d.bias, m.conv_kernel_size
    prefix_rows = None
    if tree.session:
        prefix_rows = session_rows(mixed, saved, p, window)
        head = conv(prefix_rows.transpose(1, 2), weight, bias, activation=m.activation)
        head = head[:, :, window - 1 :]
        lead = prefix_rows[:, -(window - 1) :]
    else:
        head = conv(mixed[:, :p].transpose(1, 2), weight, bias, activation=m.activation)
        lead = mixed[:, p - (window - 1) : p]
    rows = torch.cat([lead.expand(n, -1, -1), tree.rows(mixed[0, p:])], dim=1)
    tail = conv(rows.transpose(1, 2), weight, bias, activation=m.activation)[
        :, :, window - 1 :
    ]
    mixed = torch.cat(
        [head[0].transpose(0, 1), tree.packed(tail.transpose(1, 2))], dim=0
    )[None]
    query, key, value = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    query = query.reshape(1, length, -1, m.head_k_dim)
    key = key.reshape(1, length, -1, m.head_k_dim)
    value = value.reshape(1, length, -1, m.head_v_dim)
    beta = b.sigmoid()
    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    if m.num_v_heads // m.num_k_heads > 1:
        query = query.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        key = key.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
    rule = kernels("chunk_gated_delta_rule")
    if prefix_rows is not None:
        out_prefix, state = prefix_rule(
            rule, tree, (query, key, value, g, beta), saved, prefix_rows, window
        )
    else:
        out_prefix, state = rule(
            query[:, :p], key[:, :p], value[:, :p], g=g[:, :p], beta=beta[:, :p], initial_state=None,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
    start = state.expand(n, *state.shape[1:]).contiguous()
    out_suffix = suffix_rule(
        rule, tree, query[:, p:], key[:, p:], value[:, p:], g[:, p:], beta[:, p:], start
    )
    core = (
        out_suffix if out_prefix is None else torch.cat([out_prefix, out_suffix], dim=1)
    )
    core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
    out: torch.Tensor = m.out_proj(core.reshape(1, length, -1))
    return out
