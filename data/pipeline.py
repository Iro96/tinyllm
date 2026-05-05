from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from data.dataset_builder import prepare_dataset_assets
from data.stream_dataset import StreamingTokenDataset


def build_collate_fn(tokenizer):
    """Pad each batch to the longest sequence, handling loss masks."""
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        # Handle both old (x, y) and new (x, y, mask) formats
        if len(batch[0]) == 3:
            input_seqs, target_seqs, loss_masks = zip(*batch)
            has_masks = True
        else:
            input_seqs, target_seqs = zip(*batch)
            loss_masks = None
            has_masks = False
        
        max_len = max(seq.size(0) for seq in input_seqs)

        x = torch.stack(
            [
                F.pad(seq, (0, max_len - seq.size(0)), value=pad_id)
                for seq in input_seqs
            ]
        )
        y = torch.stack(
            [
                F.pad(seq, (0, max_len - seq.size(0)), value=pad_id)
                for seq in target_seqs
            ]
        )
        
        if has_masks:
            # Pad loss masks with 0 (don't train on padding)
            loss_mask = torch.stack(
                [
                    F.pad(mask.unsqueeze(0), (0, max_len - mask.size(0)), value=0).squeeze(0)
                    for mask in loss_masks
                ]
            )
            return x, y, loss_mask
        
        return x, y

    return collate_fn


def build_dataloader(tokenizer, model_cfg, train_cfg, use_cuda):
    """Build the local Terry training dataloader.

    If the dataset or tokenized assets are missing, they are generated locally.
    """
    train_tokens = Path(train_cfg.train_tokens_path)
    valid_tokens = Path(train_cfg.valid_tokens_path)
    tokenizer_dir = Path(train_cfg.tokenizer_dir)

    if not train_tokens.exists() or not valid_tokens.exists() or not tokenizer_dir.exists():
        prepare_dataset_assets(
            train_source=train_cfg.train_source_path,
            valid_source=train_cfg.valid_source_path,
            train_tokens=train_cfg.train_tokens_path,
            valid_tokens=train_cfg.valid_tokens_path,
            tokenizer_dir=train_cfg.tokenizer_dir,
            train_samples=train_cfg.train_samples,
            valid_samples=train_cfg.valid_samples,
            seed=train_cfg.seed,
            force=False,
        )

    dataset = StreamingTokenDataset(
        path=train_cfg.train_tokens_path,
        max_seq_len=model_cfg.max_seq_len,
        min_seq_len=32,
    )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=not isinstance(dataset, IterableDataset),
        drop_last=not isinstance(dataset, IterableDataset),
        pin_memory=use_cuda,
        num_workers=0,
        collate_fn=build_collate_fn(tokenizer),
    )

    return loader
