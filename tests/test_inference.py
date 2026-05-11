#!/usr/bin/env python3
"""Smoke tests for Terry inference utilities."""

import os
import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.checkpoint_manager import CheckpointManager
from tools.inference import ModelInference
from tools.tokenizer import load_tokenizer


CHECKPOINT_PATH = "checkpoints/last_model.pt"


def checkpoint_is_compatible(path: str) -> bool:
    if not os.path.exists(path):
        return False

    checkpoint = torch.load(path, map_location="cpu")
    checkpoint_vocab_size = checkpoint["model"]["embed.weight"].size(0)
    tokenizer_vocab_size = len(load_tokenizer())
    return checkpoint_vocab_size == tokenizer_vocab_size


class TestTokenizer(unittest.TestCase):
    def test_round_trip(self):
        tokenizer = load_tokenizer()
        text = "hello owner"
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.bos_token_id, 1)
        self.assertEqual(tokenizer.eos_token_id, 2)


class TestInferenceDecoding(unittest.TestCase):
    def test_decode_generated_reply_strips_literal_im_end_marker(self):
        tokenizer = load_tokenizer()
        model = ModelInference.__new__(ModelInference)
        model.tokenizer = tokenizer
        model._text_eos_token_ids = tokenizer.encode(
            tokenizer.eos_token,
            add_special_tokens=False,
        )

        prompt_length = 4
        reply_ids = tokenizer.encode(
            "hello owner\n<|im_end|>\nextra text",
            add_special_tokens=False,
        )
        generated_ids = torch.tensor([11, 12, 13, 14, *reply_ids], dtype=torch.long)

        reply = model._decode_generated_reply(generated_ids, prompt_length)
        self.assertEqual(reply, "hello owner")


@unittest.skipUnless(
    checkpoint_is_compatible(CHECKPOINT_PATH),
    "compatible Terry checkpoint required for inference smoke tests",
)
class TestInference(unittest.TestCase):
    def test_model_loading(self):
        model = ModelInference(checkpoint_path=CHECKPOINT_PATH)
        self.assertFalse(model.model.training)

    def test_generation(self):
        model = ModelInference(checkpoint_path=CHECKPOINT_PATH)
        reply = model.generate("hi terry", max_length=16, do_sample=False, temperature=0.0)
        self.assertIsInstance(reply, str)

    def test_checkpoint_manager(self):
        manager = CheckpointManager()
        latest = manager.get_latest_checkpoint()
        self.assertIsNotNone(latest)
        model = manager.load_model_from_checkpoint(latest)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
