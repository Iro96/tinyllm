#!/usr/bin/env python3
"""Quick smoke test for variable-length sequence support."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ModelConfig
from data.dataset import TokenDataset
from model.transformer import TinyLLM
from tools.tokenizer import load_tokenizer


def test_variable_length_sequences():
    """Test that the dataset and model work with dynamic padding."""
    print("Testing variable-length sequence support...")

    tokenizer = load_tokenizer()

    documents = [
        [10, 11, 12, 13, 14, 15, 16, tokenizer.eos_token_id],
        [20, 21, 22, 23, tokenizer.eos_token_id],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, tokenizer.eos_token_id],
        [40, 41, 42, tokenizer.eos_token_id],
    ]

    max_seq_len = 8
    dataset = TokenDataset(documents, max_seq_len, min_seq_len=2)

    print(f"Dataset length: {len(dataset)}")

    def collate_fn(batch):
        input_seqs, target_seqs = zip(*batch)
        max_len = max(seq.size(0) for seq in input_seqs)

        batch_inputs = torch.stack(
            [
                torch.nn.functional.pad(
                    seq,
                    (0, max_len - seq.size(0)),
                    value=tokenizer.pad_token_id,
                )
                for seq in input_seqs
            ]
        )
        batch_targets = torch.stack(
            [
                torch.nn.functional.pad(
                    seq,
                    (0, max_len - seq.size(0)),
                    value=tokenizer.pad_token_id,
                )
                for seq in target_seqs
            ]
        )
        return batch_inputs, batch_targets

    batch = [dataset[i] for i in range(min(3, len(dataset)))]
    batch_inputs, batch_targets = collate_fn(batch)
    padding_mask = batch_inputs != tokenizer.pad_token_id

    model_cfg = ModelConfig(max_seq_len=max_seq_len)
    model = TinyLLM(model_cfg, vocab_size=len(tokenizer))
    model.eval()

    with torch.no_grad():
        logits = model(batch_inputs, padding_mask=padding_mask)

    print(f"Batch shape: {batch_inputs.shape}")
    print(f"Target shape: {batch_targets.shape}")
    print(f"Model output shape: {logits.shape}")
    print(f"Causal mask cached: {model.blocks[0].attn.causal_mask is not None}")
    print("Variable-length sequence test passed.")


if __name__ == "__main__":
    test_variable_length_sequences()
