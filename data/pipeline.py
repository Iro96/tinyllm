import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from data.dataset import TokenDataset
from data.dataset_builder import stream_wikipedia


def build_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        input_seqs, target_seqs = zip(*batch)
        max_len = max(seq.size(0) for seq in input_seqs)

        x = torch.stack([
            F.pad(seq, (0, max_len - seq.size(0)), value=pad_id)
            for seq in input_seqs
        ])
        y = torch.stack([
            F.pad(seq, (0, max_len - seq.size(0)), value=pad_id)
            for seq in target_seqs
        ])
        return x, y

    return collate_fn


def build_dataloader(tokenizer, model_cfg, train_cfg, use_cuda):
    documents = stream_wikipedia(
        tokenizer=tokenizer,
        max_seq_len=model_cfg.max_seq_len,
        max_tokens=190_000_000,
        cache_dir="D:\\gh-editor\\tinyllm\\hf",
    )

    dataset = TokenDataset(
        documents=documents,
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
