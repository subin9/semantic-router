"""Session cache: the backbone state of a request's prefix, saved after its forward and resumed by later requests.

An agent, or a Router rule that reads the conversation, asks about a state that grows call after call, so most of
one request's prefix is the prefix of an earlier one. After a shared-context forward the engine saves the backbone
state at a mark inside the request's state (``mark``); a later request whose prefix starts with the same tokens
resumes from the furthest saved position it matches and computes only the tokens after it (``Tree.start``).

A saved state holds, per layer, what the layer needs to continue from the mark: a full-attention layer's keys and
values of every token before it (after norm and rotary, in the compute dtype), a Gated DeltaNet layer's recurrent
state there (float32) and the last ``window - 1`` inputs of its convolution. Nothing is approximated, so a resumed
request computes the function a fresh one computes; the rounding differs as it does between any two packings.

Marks fall on multiples of ``STRIDE`` at least ``MARGIN`` tokens before the end of the state, so the next call of a
growing state (whose last tokens may tokenize differently once text follows them) and a repeat of the same state
with other questions both find it. Lookup hashes a prefix in blocks of ``STRIDE`` tokens, one pass over its tokens,
and confirms a hit token by token. Entries live on the model's device and are evicted least recently used under a
byte budget.

A state that attends to itself both ways (a ``prefix_bidir`` state-once package) changes all its tokens' states
when text is added to it, so only the whole state is reusable: a ``whole`` cache saves each state at its end and
resumes a request whose state is exactly a saved one, which then computes its question blocks only.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch

STRIDE = 256
MARGIN = 32
# A state shorter than this is not saved: below it a session tree, which runs eager and launches about twice the
# kernels of a branched or padded forward, costs more than the tokens it skips (docs/records/session-cache.md).
MIN_TOKENS = 2048


def nbytes(value: Any) -> int:
    """Bytes of the tensors in a nested dict or list."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nbytes(v) for v in value)
    return 0


def block_hashes(tokens: list[int], stride: int) -> list[int]:
    """Chained hashes of ``tokens`` in blocks of ``stride``: entry k identifies ``tokens[: k * stride]``."""
    hashes = [0]
    for start in range(0, len(tokens) - stride + 1, stride):
        hashes.append(hash((hashes[-1], tuple(tokens[start : start + stride]))))
    return hashes


@dataclass
class Entry:
    """The backbone state after ``tokens``, one item per layer in layer order."""

    tokens: list[int]
    layers: list[dict[str, torch.Tensor]]
    bytes: int = 0

    def __post_init__(self) -> None:
        self.bytes = nbytes(self.layers)


class SessionCache:
    """Saved backbone states by the tokens before them, least recently used first out, under ``budget`` bytes."""

    def __init__(
        self,
        budget: int,
        stride: int = STRIDE,
        margin: int = MARGIN,
        min_tokens: int = MIN_TOKENS,
        whole: bool = False,
    ):
        if budget <= 0:
            raise ValueError("a session cache needs a positive byte budget")
        self.whole = whole
        self.budget = budget
        self.stride = stride
        self.margin = margin
        self.min_tokens = min_tokens
        self.entries: OrderedDict[int, Entry] = OrderedDict()
        self.used = 0
        self.hits = self.misses = self.saves = self.evictions = 0
        self.resumed_tokens = 0
        self._lock = threading.Lock()

    def mark(self, state_tokens: int, start: int, prefix: int) -> int:
        """Where to save a prefix of ``prefix`` tokens whose first ``state_tokens`` are the state, computed from
        ``start``: the last multiple of ``stride`` at least ``margin`` before the state's end (a ``whole`` cache:
        the state's end), or 0 for none (a short state, or a mark the resumed position already reaches).
        """
        if self.whole:
            fits = self.min_tokens <= state_tokens <= prefix
            return state_tokens if fits and start == 0 else 0
        mark = min(state_tokens - self.margin, prefix) // self.stride * self.stride
        return mark if mark >= self.min_tokens and mark > start else 0

    def match(self, tokens: list[int]) -> tuple[Entry | None, int]:
        """The entry and position to resume ``tokens`` (a prefix) from: the furthest saved position, before its
        last token, whose tokens all match; a ``whole`` cache resumes only all of ``tokens``, an exact saved state.
        (None, 0) for none."""
        with self._lock:
            if len(tokens) < self.min_tokens + (not self.whole) or not self.entries:
                self.misses += 1
                return None, 0
            best: tuple[int, int] | None = None
            if self.whole:
                key = hash(tuple(tokens))
                entry = self.entries.get(key)
                if entry is not None and entry.tokens == tokens:
                    best = (key, len(tokens))
            hashes = [] if self.whole else block_hashes(tokens[:-1], self.stride)
            for key, entry in self.entries.items() if hashes else ():
                saved = len(entry.tokens)
                blocks = saved // self.stride
                if blocks >= len(hashes) or (best is not None and saved <= best[1]):
                    continue
                if key == hashes[blocks] and entry.tokens == tokens[:saved]:
                    best = (key, saved)
            if best is None:
                self.misses += 1
                return None, 0
            self.entries.move_to_end(best[0])
            self.hits += 1
            self.resumed_tokens += best[1]
            return self.entries[best[0]], best[1]

    def insert(self, tokens: list[int], layers: list[dict[str, torch.Tensor]]) -> None:
        """Save the state after ``tokens`` (a multiple of ``stride`` long, unless the cache is ``whole``), evicting
        the least recently used."""
        if len(tokens) % self.stride and not self.whole:
            raise ValueError("a saved state ends on a multiple of the stride")
        entry = Entry(list(tokens), layers)
        with self._lock:
            if entry.bytes > self.budget:
                return
            key = (
                hash(tuple(entry.tokens))
                if self.whole
                else block_hashes(entry.tokens, self.stride)[-1]
            )
            old = self.entries.pop(key, None)
            if old is not None:
                self.used -= old.bytes
            while self.entries and self.used + entry.bytes > self.budget:
                _, evicted = self.entries.popitem(last=False)
                self.used -= evicted.bytes
                self.evictions += 1
            self.entries[key] = entry
            self.used += entry.bytes
            self.saves += 1

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()
            self.used = 0

    def receipt(self) -> dict[str, Any]:
        with self._lock:
            return {
                "budget_bytes": self.budget,
                "bytes": self.used,
                "entries": len(self.entries),
                "hits": self.hits,
                "misses": self.misses,
                "saves": self.saves,
                "evictions": self.evictions,
                "resumed_tokens": self.resumed_tokens,
                "stride": self.stride,
                "min_tokens": self.min_tokens,
                "whole": self.whole,
            }
