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
        scaler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.train_cfg = train_cfg
        self.device = device
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.scaler = scaler

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
    def train_step(self, x, y, loss_mask=None):
        """Train step with optional loss mask for assistant-only tokens.
        
        Args:
            x: input token IDs
            y: target token IDs (shifted by 1)
            loss_mask: optional binary mask (1=train, 0=ignore) for each position in y
        """
        x, y = x.to(self.device), y.to(self.device)
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.device)

        if self.scaler is not None:
            # Mixed precision training
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                padding_mask = x != self.tokenizer.pad_token_id
                logits = self.model(x, padding_mask=padding_mask)

                # Reshape for cross entropy
                logits_flat = logits.reshape(-1, logits.size(-1))
                y_flat = y.reshape(-1)
                
                # Compute loss with padding ignored
                loss = F.cross_entropy(
                    logits_flat,
                    y_flat,
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction="none",
                )
                
                # Apply loss mask if provided (only train on assistant tokens)
                if loss_mask is not None:
                    loss_mask_flat = loss_mask.reshape(-1).float()
                    valid_mask = (y_flat != self.tokenizer.pad_token_id)
                    effective_mask = loss_mask_flat * valid_mask.float()

                    loss = loss * effective_mask
                    tokens = effective_mask.sum()
                else:
                    # Legacy: count non-padding tokens
                    tokens = (y_flat != self.tokenizer.pad_token_id).sum()
                
                # Sum loss and normalize
                if tokens > 0:
                    loss = loss.sum() / tokens / self.train_cfg.grad_accum
                    self.scaler.scale(loss).backward()
                    self.accum_loss += loss.item()
                self.micro_step += 1
        else:
            # Standard precision training
            padding_mask = x != self.tokenizer.pad_token_id
            logits = self.model(x, padding_mask=padding_mask)

            # Reshape for cross entropy
            logits_flat = logits.reshape(-1, logits.size(-1))
            y_flat = y.reshape(-1)
            
            # Compute loss with padding ignored
            loss = F.cross_entropy(
                logits_flat,
                y_flat,
                ignore_index=self.tokenizer.pad_token_id,
                reduction="none",
            )
            
            # Apply loss mask if provided (only train on assistant tokens)
            if loss_mask is not None:
                loss_mask_flat = loss_mask.reshape(-1).float()
                valid_mask = (y_flat != self.tokenizer.pad_token_id)
                effective_mask = loss_mask_flat * valid_mask.float()

                loss = loss * effective_mask
                tokens = effective_mask.sum()
            else:
                # Legacy: count non-padding tokens
                tokens = (y_flat != self.tokenizer.pad_token_id).sum()
            
            # Sum loss and normalize
            if tokens > 0:
                loss = loss.sum() / tokens / self.train_cfg.grad_accum
                loss.backward()
                self.accum_loss += loss.item()
            self.micro_step += 1

    # --------------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------------
    def optimizer_step_fn(self):
        if self.scaler is not None:
            # Mixed precision step
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision step
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
            for batch in dataloader:
                if self.optimizer_step >= self.train_cfg.total_steps:
                    break

                # Handle both old (x, y) and new (x, y, mask) formats
                if len(batch) == 3:
                    x, y, loss_mask = batch
                    self.train_step(x, y, loss_mask)
                else:
                    x, y = batch
                    self.train_step(x, y)

                if self.micro_step % self.train_cfg.grad_accum == 0:
                    self.optimizer_step_fn()
                    self.maybe_log_and_save()
