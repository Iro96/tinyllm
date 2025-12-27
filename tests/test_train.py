import os
import shutil
import unittest
import torch
from transformers import AutoTokenizer

from config import ModelConfig
from model.transformer import TinyLLM


class TestModelInternals(unittest.TestCase):
    def setUp(self):
        self.model_cfg = ModelConfig(
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=32,
            dropout_rate=0.0,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def test_01_vocab_resizing(self):
        """
        Check that the model correctly resizes token embeddings
        and preserves weight tying.
        """
        base_vocab_size = len(self.tokenizer)
        model = TinyLLM(self.model_cfg, vocab_size=base_vocab_size)

        # Add PAD token
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        new_vocab_size = len(self.tokenizer)

        self.assertNotEqual(base_vocab_size, new_vocab_size)

        model.resize_token_embeddings(new_vocab_size)

        self.assertEqual(model.embed.weight.size(0), new_vocab_size)
        self.assertEqual(model.head.weight.size(0), new_vocab_size)

        # Weight tying must remain exact (same object)
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

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.vocab_size = len(self.tokenizer)

        self.checkpoint_dir = "test_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)

    def test_01_overfit_one_batch(self):
        """
        The model must be able to strongly overfit a single batch.
        This validates:
        - forward pass
        - backward pass
        - optimizer
        - loss wiring
        """
        torch.manual_seed(42)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyLLM(self.model_cfg, vocab_size=self.vocab_size).to(device)
        model.train()

        input_ids = torch.randint(
            0,
            self.vocab_size,
            (2, self.model_cfg.max_seq_len),
            device=device,
        )

        padding_mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = input_ids.clone()

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        with torch.no_grad():
            initial_logits = model(input_ids, padding_mask=padding_mask)
            initial_loss = torch.nn.functional.cross_entropy(
                initial_logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=self.tokenizer.pad_token_id,
            ).item()

        for _ in range(100):
            optimizer.zero_grad()
            logits = model(input_ids, padding_mask=padding_mask)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Strong overfitting requirement
        self.assertLess(
            final_loss,
            initial_loss * 0.2,
            f"Expected strong overfitting, got loss {final_loss:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
