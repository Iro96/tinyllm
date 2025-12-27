import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ModelConfig
from model.transformer import TinyLLM


class ModelInference:
    def __init__(self, checkpoint_path="checkpoints/last_model.pt", device="auto", tokenizer_path=None):
        self.model_config = ModelConfig()
        self.device = self._get_device(device)
        
        # Use saved tokenizer if path provided, otherwise use centralized loader
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = self._load_tokenizer()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = self._load_model(checkpoint_path)

        print(f"Using device: {self.device}")
        print(f"Model loaded from: {checkpoint_path}")
        if tokenizer_path:
            print(f"Tokenizer loaded from: {tokenizer_path}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _get_device(self, device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_tokenizer(self):
        from tools.tokenizer import load_tokenizer
        return load_tokenizer()

    def _load_model(self, checkpoint_path):
        vocab_size = len(self.tokenizer)
        model = TinyLLM(self.model_config, vocab_size=vocab_size).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        if "step" in checkpoint:
            print(f"Checkpoint step: {checkpoint['step']}")

        return model

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def get_next_token_probabilities(
        self,
        prompt: str,
        top_k: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        """Return a list of (token_str, probability) for the next token.

        Args:
            prompt: Input prompt string.
            top_k: If provided (>0), return the top-k tokens by probability.
            temperature: Temperature to divide logits by before softmax.
            top_p: If <1.0, apply nucleus filtering before softmax.
        """
        self.model.eval()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            pad_id = self.tokenizer.pad_token_id
            padding_mask = None
            if pad_id is not None:
                padding_mask = input_ids != pad_id

            logits = self.model(input_ids, padding_mask=padding_mask)
            next_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            # Apply optional filtering consistent with generation
            if top_p is not None and top_p < 1.0:
                next_logits = self.model._top_p_filter(next_logits, top_p)

            if top_k is not None and top_k > 0:
                next_logits = self.model._top_k_filter(next_logits, top_k)

            probs = F.softmax(next_logits, dim=-1)[0]

            # Select top tokens to return
            k = top_k if (top_k is not None and top_k > 0) else min(20, probs.size(0))
            values, indices = torch.topk(probs, k=k)

            results = []
            for idx, val in zip(indices.tolist(), values.tolist()):
                # Decode single token to human-readable string
                token_str = self.tokenizer.decode([idx], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                results.append((token_str, float(val)))

        return results


def load_model(checkpoint_path="checkpoints/last_model.pt", device="auto", tokenizer_path=None):
    return ModelInference(checkpoint_path, device, tokenizer_path)
