#!/usr/bin/env python3
"""Checkpoint management utilities for TinyLLM."""

from __future__ import annotations

import os
from typing import Any

import torch

from config import ModelConfig
from model.transformer import TinyLLM
from tools.tokenizer import load_tokenizer


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints", tokenizer_path: str | None = None):
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer_path = tokenizer_path
        os.makedirs(checkpoint_dir, exist_ok=True)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoint files sorted by training step."""
        checkpoints: list[dict[str, Any]] = []

        if not os.path.exists(self.checkpoint_dir):
            return checkpoints

        for filename in os.listdir(self.checkpoint_dir):
            if not filename.endswith(".pt"):
                continue

            filepath = os.path.join(self.checkpoint_dir, filename)
            info = self.get_checkpoint_info(filepath)
            if info is None:
                continue

            info["filename"] = filename
            info["path"] = filepath
            checkpoints.append(info)

        checkpoints.sort(key=lambda item: item.get("step", 0), reverse=True)
        return checkpoints

    def get_checkpoint_info(self, checkpoint_path: str) -> dict[str, Any] | None:
        """Return metadata about a checkpoint without loading it onto GPU."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            print(f"Error reading checkpoint info from {checkpoint_path}: {exc}")
            return None

        return {
            "step": checkpoint.get("step", 0),
            "model_keys": list(checkpoint.get("model", {}).keys()),
            "optimizer_keys": list(checkpoint.get("optimizer", {}).keys()),
            "file_size": os.path.getsize(checkpoint_path),
            "file_path": checkpoint_path,
        }

    def load_model_from_checkpoint(
        self,
        checkpoint_path: str,
        device: str = "auto",
    ) -> TinyLLM:
        """Load a model from a checkpoint file."""
        if device == "auto":
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            target_device = torch.device(device)

        tokenizer = load_tokenizer(self.tokenizer_path)
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        checkpoint_vocab_size = checkpoint["model"]["embed.weight"].size(0)
        tokenizer_vocab_size = len(tokenizer)
        if checkpoint_vocab_size != tokenizer_vocab_size:
            raise ValueError(
                "Checkpoint vocabulary does not match the Terry tokenizer. "
                f"checkpoint={checkpoint_vocab_size}, tokenizer={tokenizer_vocab_size}. "
                "Please retrain or load a Terry-format checkpoint."
            )

        model = TinyLLM(
            ModelConfig(),
            vocab_size=tokenizer_vocab_size,
        ).to(target_device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    def print_checkpoint_summary(self, checkpoint_path: str):
        """Print a short summary for a checkpoint file."""
        info = self.get_checkpoint_info(checkpoint_path)
        if not info:
            print(f"No checkpoint info available for {checkpoint_path}")
            return

        print("\n=== Checkpoint Summary ===")
        print(f"File: {os.path.basename(checkpoint_path)}")
        print(f"Path: {checkpoint_path}")
        print(f"Step: {info.get('step', 'N/A')}")
        print(f"File Size: {info.get('file_size', 0) / 1024 / 1024:.2f} MB")
        print(f"Model Keys: {len(info.get('model_keys', []))}")
        print(f"Optimizer Keys: {len(info.get('optimizer_keys', []))}")

    def get_latest_checkpoint(self) -> str | None:
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[0]["path"]
        return None


def main():
    manager = CheckpointManager()
    print("=== TinyLLM Checkpoint Manager ===")

    checkpoints = manager.list_checkpoints()
    if not checkpoints:
        print("No checkpoints found.")
        return

    print("\nAvailable checkpoints:")
    for index, checkpoint in enumerate(checkpoints, start=1):
        print(f"{index}. {checkpoint['filename']} (step: {checkpoint.get('step', 'N/A')})")

    latest = manager.get_latest_checkpoint()
    if latest:
        print(f"\nLatest checkpoint: {os.path.basename(latest)}")
        manager.print_checkpoint_summary(latest)


if __name__ == "__main__":
    main()
