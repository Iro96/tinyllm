import torch
import torch.nn.functional as F

from utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        tokenizer,
        train_cfg,
        device,
        logger,
        checkpoint_path="checkpoints/last_model.pt",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.train_cfg = train_cfg
        self.device = device
        self.logger = logger
        self.checkpoint_path = checkpoint_path

        self.optimizer_step = 0
        self.micro_step = 0
        self.accum_loss = 0.0

    # --------------------------------------------------------------
    # Resume
    # --------------------------------------------------------------
    def try_resume(self):
        import os

        if not os.path.exists(self.checkpoint_path):
            return

        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.optimizer_step = checkpoint["step"]

        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        print(f"Resumed from step {self.optimizer_step}")

    # --------------------------------------------------------------
    # Step
    # --------------------------------------------------------------
    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)

        padding_mask = x != self.tokenizer.pad_token_id
        logits = self.model(x, padding_mask=padding_mask)

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            y[:, 1:].reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction="sum",
        )

        tokens = (y[:, 1:] != self.tokenizer.pad_token_id).sum()
        loss = loss / tokens / self.train_cfg.grad_accum

        loss.backward()

        self.accum_loss += loss.item()
        self.micro_step += 1

    # --------------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------------
    def optimizer_step_fn(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.optimizer_step += 1

    # --------------------------------------------------------------
    # Logging + checkpoint
    # --------------------------------------------------------------
    def maybe_log_and_save(self):
        if self.optimizer_step % 50 != 0:
            return

        avg_loss = self.accum_loss / 50
        self.accum_loss = 0.0

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.optimizer_step,
            scheduler=self.scheduler,
            path=self.checkpoint_path,
        )

        self.tokenizer.save_pretrained("checkpoints/tokenizer")

        self.logger.log(
            step=self.optimizer_step,
            loss=avg_loss,
            lr=self.optimizer.param_groups[0]["lr"],
        )

    # --------------------------------------------------------------
    # Train loop
    # --------------------------------------------------------------
    def train(self, dataloader):
        self.model.train()

        while self.optimizer_step < self.train_cfg.total_steps:
            for x, y in dataloader:
                if self.optimizer_step >= self.train_cfg.total_steps:
                    break

                self.train_step(x, y)

                if self.micro_step % self.train_cfg.grad_accum == 0:
                    self.optimizer_step_fn()
                    self.maybe_log_and_save()
