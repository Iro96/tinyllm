import math
import os
from itertools import cycle
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from config import ModelConfig, TrainConfig
from model.transformer import TinyLLM
from data.dataset import TokenDataset
from utils.logger import Logger
from utils.checkpoint import save_checkpoint


def main():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Edit total_steps
    train_cfg.total_steps = 10000

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if train_cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg.device)

    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Tokenizer (PAD ≠ EOS) - Centralized
    # ------------------------------------------------------------------
    from tools.tokenizer import load_tokenizer
    tokenizer = load_tokenizer()

    # ------------------------------------------------------------------
    # Dataset loading (document-aware)
    # ------------------------------------------------------------------
    raw = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-v1",
        split="train",
        # cache_dir="D:\\gh-editor\\tinyllm\\hf",
    )

    documents = []
    for ex in raw:
        ids = tokenizer(
            ex["text"],
            add_special_tokens=False,
            truncation=False,
        ).input_ids

        max_len = model_cfg.max_seq_len - 1  # reserve space for EOS

        for i in range(0, len(ids), max_len):
            chunk = ids[i : i + max_len]
            if len(chunk) > 1:
                chunk = chunk + [tokenizer.eos_token_id]
                documents.append(chunk)

    print(f"Loaded {len(documents)} documents")

    dataset = TokenDataset(
        documents=documents,
        max_seq_len=model_cfg.max_seq_len,
        min_seq_len=32,
    )

    # ------------------------------------------------------------------
    # Collate function (dynamic padding)
    # ------------------------------------------------------------------
    def collate_fn(batch):
        input_seqs, target_seqs = zip(*batch)
        max_len = max(seq.size(0) for seq in input_seqs)

        pad_id = tokenizer.pad_token_id

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

        return x, y

    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Model / optimizer / scheduler
    # ------------------------------------------------------------------
    vocab_size = len(tokenizer)
    model = TinyLLM(model_cfg, vocab_size=vocab_size).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    def lr_lambda(step):
        if step < train_cfg.warmup_steps:
            return step / max(1, train_cfg.warmup_steps)
        progress = (step - train_cfg.warmup_steps) / max(
            1, train_cfg.total_steps - train_cfg.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Load checkpoint if available
    # ------------------------------------------------------------------
    optimizer_step = 0
    checkpoint_path = "checkpoints/last_model.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_step = checkpoint["step"]
        # we need to advance the scheduler to the loaded step
        for _ in range(optimizer_step):
            scheduler.step()
        print(f"Resuming training from step {optimizer_step}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Training from scratch.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger = Logger()
    model.train()

    global_step = optimizer_step * train_cfg.grad_accum
    accum_loss = 0.0

    train_iterator = cycle(loader)

    while optimizer_step < train_cfg.total_steps:
        x, y = next(train_iterator)

        x = x.to(device)
        y = y.to(device)

        padding_mask = x != tokenizer.pad_token_id

        logits = model(x, padding_mask=padding_mask)

        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            y[:, 1:].contiguous().view(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction="sum",
        )

        tokens = (y[:, 1:] != tokenizer.pad_token_id).sum()

        loss = loss / tokens
        loss = loss / train_cfg.grad_accum
        accum_loss += loss.item()
        loss.backward()

        if (global_step + 1) % train_cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            optimizer_step += 1

            if optimizer_step % 50 == 0:
                avg_loss = accum_loss / 50
                accum_loss = 0.0

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=optimizer_step,
                    path="checkpoints/last_model.pt",
                )
                
                # Save tokenizer with checkpoint
                tokenizer.save_pretrained("checkpoints/tokenizer")

                logger.log(
                    step=optimizer_step,
                    loss=avg_loss,
                    lr=optimizer.param_groups[0]["lr"],
                )

        global_step += 1

    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=optimizer_step,
        path=f"releases/tinyllm_wiki103_{train_cfg.total_steps}.pt",
    )
    
    # Save final tokenizer
    tokenizer.save_pretrained(f"releases/tokenizer_{train_cfg.total_steps}")
    print("Training finished.")


if __name__ == "__main__":
    main()
