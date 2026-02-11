import math
import os
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

from config import ModelConfig, TrainConfig
from model.transformer import TinyLLM
from data.dataset import TokenDataset
from utils.logger import Logger
from utils.checkpoint import save_checkpoint


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    train_cfg.total_steps = 10_000
    set_seed(train_cfg.seed)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if train_cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg.device)

    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    if use_cuda:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"

    else:
        torch.set_num_threads(1)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    from tools.tokenizer import load_tokenizer
    tokenizer = load_tokenizer()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    use_shuffle = True

    max_tokens = 300_000_000
    documents = []
    token_count = 0

    print("Streaming Wikipedia 20231101.en...")
    raw = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        cache_dir="D:\\gh-editor\\tinyllm\\hf",
    )

    max_len = model_cfg.max_seq_len - 1

    for ex in raw:
        text = ex["text"]
        if not text.strip():
            continue

        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) < 2:
            continue

        for i in range(0, len(ids), max_len):
            chunk = ids[i : i + max_len]
            if len(chunk) > 1:
                chunk.append(tokenizer.eos_token_id)
                documents.append(chunk)
                token_count += len(chunk)
            if token_count >= max_tokens:
                break
        if token_count >= max_tokens:
            break

    print(f"Collected {len(documents)} documents (~{token_count} tokens)")

    dataset = TokenDataset(
        documents=documents,
        max_seq_len=model_cfg.max_seq_len,
        min_seq_len=32,
    )

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------
    def collate_fn(batch):
        input_seqs, target_seqs = zip(*batch)
        max_len = max(seq.size(0) for seq in input_seqs)
        pad_id = tokenizer.pad_token_id

        x = torch.stack(
            [F.pad(seq, (0, max_len - seq.size(0)), value=pad_id) for seq in input_seqs]
        )
        y = torch.stack(
            [F.pad(seq, (0, max_len - seq.size(0)), value=pad_id) for seq in target_seqs]
        )
        return x, y

    is_iterable = isinstance(dataset, IterableDataset)

    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=(use_shuffle and not is_iterable),
        drop_last=not is_iterable,
        pin_memory=use_cuda,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Model / Optimizer / Scheduler
    # ------------------------------------------------------------------
    model = TinyLLM(model_cfg, vocab_size=len(tokenizer)).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        fused=use_cuda,
    )

    def lr_lambda(step: int):
        if step < train_cfg.warmup_steps:
            return step / max(1, train_cfg.warmup_steps)
        progress = (step - train_cfg.warmup_steps) / max(
            1, train_cfg.total_steps - train_cfg.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    checkpoint_path = "checkpoints/last_model.pt"
    optimizer_step = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_step = checkpoint["step"]
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Resumed from step {optimizer_step}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    logger = Logger()
    model.train()

    accum_loss = 0.0
    micro_step = 0

    while optimizer_step < train_cfg.total_steps:
        for x, y in loader:
            if optimizer_step >= train_cfg.total_steps:
                break

            x = x.to(device)
            y = y.to(device)

            padding_mask = x != tokenizer.pad_token_id
            logits = model(x, padding_mask=padding_mask)

            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                y[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="sum",
            )

            tokens = (y[:, 1:] != tokenizer.pad_token_id).sum()
            loss = loss / tokens / train_cfg.grad_accum

            loss.backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % train_cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                optimizer_step += 1

                if optimizer_step % 50 == 0:
                    avg_loss = accum_loss / 50
                    accum_loss = 0.0

                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=optimizer_step,
                        scheduler=scheduler,
                        path=checkpoint_path,
                    )
                    tokenizer.save_pretrained("checkpoints/tokenizer")

                    logger.log(
                        step=optimizer_step,
                        loss=avg_loss,
                        lr=optimizer.param_groups[0]["lr"],
                    )

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=optimizer_step,
        scheduler=scheduler,
        path=f"releases/test_tinyllm_300mtk_{train_cfg.total_steps}.pt",
    )
    tokenizer.save_pretrained(f"tokenizer/test_tokenizer_{train_cfg.total_steps}")
    print("Training finished.")


if __name__ == "__main__":
    main()
