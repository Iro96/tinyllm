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


def load_model(checkpoint_path="checkpoints/last_model.pt", device="auto", tokenizer_path=None):
    return ModelInference(checkpoint_path, device, tokenizer_path)
