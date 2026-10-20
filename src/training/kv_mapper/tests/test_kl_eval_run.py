from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np

# Torch and Transformers are optional in the contract test environment.
# ruff: noqa: PLC0415

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.tests.test_distill import (  # noqa: E402
    VOCAB,
    _identity_tensors,
    _manifest,
)


class KlEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError as exc:
            raise unittest.SkipTest("torch or transformers not installed") from exc

        torch.manual_seed(0)
        config = Qwen3Config(
            initializer_range=0.2,
            vocab_size=VOCAB,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
        )
        self.model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        rng = random.Random(0)
        self.samples = [
            (
                [rng.randrange(VOCAB) for _ in range(rng.randrange(4, 12))],
                [rng.randrange(VOCAB) for _ in range(rng.randrange(1, 6))],
            )
            for _ in range(4)
        ]

    def test_identity_maps_match_the_native_cache(self) -> None:
        from src.training.kv_mapper.distill import LinearMapper
        from src.training.kv_mapper.kl_eval_run import score

        mapper = LinearMapper(_manifest(), _identity_tensors())
        rows = score(self.model, self.model, {"id": mapper}, self.samples, 2, 16)
        for row in rows:
            self.assertLess(row["id"]["kl"], 1e-5)
            self.assertAlmostEqual(row["id"]["nll"], row["cold_nll"], places=4)

    def test_report_pairs_every_artifact_with_the_first(self) -> None:
        from src.training.kv_mapper.distill import LinearMapper
        from src.training.kv_mapper.kl_eval_run import report, score

        rng = np.random.default_rng(0)
        noisy = {
            name: value + rng.normal(0.0, 0.3, value.shape).astype(np.float32)
            for name, value in _identity_tensors().items()
        }
        mappers = {
            "exact": LinearMapper(_manifest(), _identity_tensors()),
            "noisy": LinearMapper(_manifest(), noisy),
        }
        rows = score(self.model, self.model, mappers, self.samples, 2, 16)
        out = report("chat", rows, ["exact", "noisy"])
        self.assertEqual(out["n"], len(self.samples))
        self.assertEqual(out["kl"]["reference"], "exact")
        self.assertEqual(set(out["kl"]["delta_vs_reference"]), {"noisy"})
        self.assertGreater(out["kl"]["delta_vs_reference"]["noisy"]["mean"], 0.0)
        self.assertEqual(out["kl"]["example_ids"][0], "chat:0")


class TextWindowTests(unittest.TestCase):
    def test_window_is_the_document_start(self) -> None:
        try:
            from src.training.kv_mapper.kl_eval_run import text_window
        except ImportError as exc:
            raise unittest.SkipTest("torch or transformers not installed") from exc
        self.assertEqual(
            text_window(list(range(10)), 6, 3), ([0, 1, 2, 3, 4, 5], [6, 7, 8])
        )
        self.assertIsNone(text_window(list(range(8)), 6, 3))

    def test_datasets_must_be_pinned(self) -> None:
        try:
            from src.training.kv_mapper.kl_eval_run import parse_args
        except ImportError as exc:
            raise unittest.SkipTest("torch or transformers not installed") from exc
        with self.assertRaises(SystemExit):
            parse_args(["--artifact", "a=/x", "--output", "/tmp/o.json"])
        args = parse_args(
            [
                "--artifact",
                "a=/x",
                "--output",
                "/tmp/o.json",
                "--chat-revision",
                "c",
                "--text-revision",
                "t",
            ]
        )
        self.assertEqual(args.chat_split, "test_sft")


class _CharTemplate:
    """One token per character; a turn renders as <role>content."""

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class AgentCutTests(unittest.TestCase):
    MESSAGES = (
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "ls"},
        {"role": "tool", "content": "a.py b.py"},
        {"role": "assistant", "content": "cat a.py"},
        {"role": "tool", "content": "x" * 10},
        {"role": "assistant", "content": "done"},
    )

    def setUp(self) -> None:
        try:
            from src.training.kv_mapper.kl_eval_run import agent_cut
        except ImportError as exc:
            raise unittest.SkipTest("torch or transformers not installed") from exc
        self.cut = agent_cut
        self.tok = _CharTemplate()

    def prompt_length(self, turn: int) -> int:
        text = self.tok.apply_chat_template(
            self.MESSAGES[:turn], add_generation_prompt=True
        )
        return len(text)

    def test_cuts_before_the_last_step_that_fits(self) -> None:
        budget = self.prompt_length(5) - 1
        prefix, reply = self.cut(self.tok, self.MESSAGES, None, budget, 3)
        self.assertEqual(len(prefix), self.prompt_length(3))
        self.assertEqual("".join(map(chr, reply)), "cat")

    def test_rejects_a_cut_far_below_the_budget(self) -> None:
        budget = 2 * self.prompt_length(1) + 2
        self.assertLess(self.prompt_length(3), budget)
        self.assertIsNone(self.cut(self.tok, self.MESSAGES, None, budget // 3, 8))
        self.assertIsNotNone(self.cut(self.tok, self.MESSAGES, None, budget, 8))


if __name__ == "__main__":
    unittest.main()
