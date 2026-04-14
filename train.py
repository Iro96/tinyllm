import math
import random
import torch
from torch.optim import AdamW

from config import ModelConfig, TrainConfig
from model.transformer import TinyLLM
from tools.tokenizer import load_tokenizer

from data.pipeline import build_dataloader
from engine.trainer import Trainer
from utils.logger import Logger
from utils.checkpoint import save_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device(device_cfg: str):
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    else:
        torch.set_num_threads(1)

    return device


def main():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    set_seed(train_cfg.seed)
    device = setup_device(train_cfg.device)
    use_cuda = device.type == "cuda"

    tokenizer = load_tokenizer()

    # Data
    loader = build_dataloader(tokenizer, model_cfg, train_cfg, use_cuda)

    # Model
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

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        train_cfg=train_cfg,
        device=device,
        logger=Logger(),
    )

    trainer.try_resume()
    trainer.train(loader)

    # Final save
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=trainer.optimizer_step,
        scheduler=scheduler,
        path=f"releases/test_tinyllm_{train_cfg.total_steps}.pt",
    )

    tokenizer.save_pretrained(
        f"tokenizer/test_tokenizer_{train_cfg.total_steps}"
    )

    print("Training finished.")


if __name__ == "__main__":
    main()
