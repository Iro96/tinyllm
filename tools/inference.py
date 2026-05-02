from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from config import ModelConfig
from data.dataset_builder import build_generation_prompt
from model.transformer import TinyLLM
from tools.tokenizer import load_tokenizer


class ModelInference:
    """Load checkpoints and chat with the Terry model."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/last_model.pt",
        device: str = "auto",
        tokenizer_path: str | None = None,
    ):
        self.model_config = ModelConfig()
        self.device = self._get_device(device)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.model = self._load_model(checkpoint_path)

        print(f"Using device: {self.device}")
        print(f"Model loaded from: {checkpoint_path}")

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self, checkpoint_path: str) -> TinyLLM:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        checkpoint_vocab_size = checkpoint["model"]["embed.weight"].size(0)
        tokenizer_vocab_size = len(self.tokenizer)
        if checkpoint_vocab_size != tokenizer_vocab_size:
            raise ValueError(
                "Checkpoint vocabulary does not match the Terry tokenizer. "
                f"checkpoint={checkpoint_vocab_size}, tokenizer={tokenizer_vocab_size}. "
                "Please retrain or point inference at a Terry-format checkpoint."
            )

        model = TinyLLM(
            self.model_config,
            vocab_size=len(self.tokenizer),
        ).to(self.device)

        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        if "step" in checkpoint:
            print(f"Checkpoint step: {checkpoint['step']}")

        return model

    def _build_chat_input(self, prompt: str) -> torch.Tensor:
        input_ids = build_generation_prompt(self.tokenizer, prompt)
        return torch.tensor([input_ids], dtype=torch.long, device=self.device)

    def _decode_generated_reply(
        self,
        generated_ids: torch.Tensor,
        prompt_length: int,
    ) -> str:
        reply_ids = generated_ids[prompt_length:].tolist()

        if self.tokenizer.eos_token_id in reply_ids:
            stop = reply_ids.index(self.tokenizer.eos_token_id)
            reply_ids = reply_ids[:stop]

        return self.tokenizer.decode(
            reply_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate Terry's next assistant reply for a plain user prompt."""
        self.model.eval()
        input_ids = self._build_chat_input(prompt)
        prompt_length = input_ids.size(1)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_length=prompt_length + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return self._decode_generated_reply(generated_ids[0], prompt_length)

    def generate_tokens(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Compatibility wrapper for older scripts."""
        return self.generate(
            prompt=prompt,
            max_length=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

    def get_next_token_probabilities(
        self,
        prompt: str,
        top_k: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        """Return the top token probabilities for the next assistant token."""
        self.model.eval()

        input_ids = self._build_chat_input(prompt)

        with torch.no_grad():
            padding_mask = input_ids != self.tokenizer.pad_token_id
            logits = self.model(input_ids, padding_mask=padding_mask)
            scale = temperature if temperature > 0 else 1.0
            next_logits = logits[:, -1, :] / scale

            if top_p is not None and top_p < 1.0:
                next_logits = self.model._top_p_filter(next_logits, top_p)

            if top_k is not None and top_k > 0:
                next_logits = self.model._top_k_filter(next_logits, top_k)

            probs = F.softmax(next_logits, dim=-1)[0]
            k = top_k if (top_k is not None and top_k > 0) else min(20, probs.size(0))
            values, indices = torch.topk(probs, k=k)

            results = []
            for token_id, probability in zip(indices.tolist(), values.tolist()):
                token_str = self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                if not token_str and token_id == self.tokenizer.eos_token_id:
                    token_str = self.tokenizer.eos_token
                results.append((token_str, float(probability)))

        return results


def load_model(
    checkpoint_path: str = "checkpoints/last_model.pt",
    device: str = "auto",
    tokenizer_path: str | None = None,
):
    return ModelInference(checkpoint_path, device, tokenizer_path)
