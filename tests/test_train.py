import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ModelConfig
from model.transformer import TinyLLM
from tools.tokenizer import load_tokenizer


class TestModelInternals(unittest.TestCase):
    def setUp(self):
        self.model_cfg = ModelConfig(
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=32,
            dropout_rate=0.0,
        )
        self.tokenizer = load_tokenizer()

    def test_01_vocab_resizing(self):
        """Resizing must preserve tied weights."""
        base_vocab_size = 128
        target_vocab_size = len(self.tokenizer)

        model = TinyLLM(self.model_cfg, vocab_size=base_vocab_size)
        model.resize_token_embeddings(target_vocab_size)

        self.assertEqual(model.embed.weight.size(0), target_vocab_size)
        self.assertEqual(model.head.weight.size(0), target_vocab_size)
        self.assertIs(model.head.weight, model.embed.weight)


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.model_cfg = ModelConfig(
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=32,
            dropout_rate=0.0,
        )
        self.tokenizer = load_tokenizer()
        self.vocab_size = len(self.tokenizer)

    def test_01_overfit_one_batch(self):
        """The model should overfit a single next-token batch quickly."""
        torch.manual_seed(42)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyLLM(self.model_cfg, vocab_size=self.vocab_size).to(device)
        model.train()

        tokens = torch.randint(
            low=3,
            high=self.vocab_size,
            size=(2, self.model_cfg.max_seq_len + 1),
            device=device,
        )
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        padding_mask = torch.ones_like(input_ids, dtype=torch.bool)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        with torch.no_grad():
            initial_logits = model(input_ids, padding_mask=padding_mask)
            initial_loss = torch.nn.functional.cross_entropy(
                initial_logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=self.tokenizer.pad_token_id,
            ).item()

        final_loss = initial_loss
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, padding_mask=padding_mask)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.assertLess(
            final_loss,
            initial_loss * 0.25,
            f"Expected strong overfitting, got loss {final_loss:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
