from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import torch

from config import ModelConfig

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|im_start|>"
EOS_TOKEN = "<|im_end|>"

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

BYTE_OFFSET = 3
VOCAB_SIZE = BYTE_OFFSET + 256

DEFAULT_TOKENIZER_DIR = Path("tokenizer/terry_byte")


class ByteTokenizer:
    """Tiny byte-level tokenizer with fixed chat special tokens.

    The first three IDs are reserved and stable across runs:
    - 0: pad
    - 1: im_start
    - 2: im_end

    All UTF-8 bytes are mapped into the remaining ID range.
    """

    def __init__(self, model_max_length: int | None = None):
        self.pad_token = PAD_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN

        self.pad_token_id = PAD_TOKEN_ID
        self.bos_token_id = BOS_TOKEN_ID
        self.eos_token_id = EOS_TOKEN_ID

        self.model_max_length = model_max_length or max(
            ModelConfig().max_seq_len,
            1_000_000,
        )

    def __len__(self) -> int:
        return VOCAB_SIZE

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ):
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be str, got {type(text)!r}")

        ids = [BYTE_OFFSET + value for value in text.encode("utf-8")]

        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)

        return ids

    def decode(
        self,
        ids: Sequence[int] | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        del clean_up_tokenization_spaces

        flat_ids = self._flatten_ids(ids)
        pieces: list[str] = []
        byte_buffer = bytearray()

        for token_id in flat_ids:
            if token_id >= BYTE_OFFSET:
                byte_buffer.append(token_id - BYTE_OFFSET)
                continue

            if byte_buffer:
                pieces.append(byte_buffer.decode("utf-8", errors="ignore"))
                byte_buffer.clear()

            if skip_special_tokens:
                continue

            if token_id == self.pad_token_id:
                pieces.append(self.pad_token)
            elif token_id == self.bos_token_id:
                pieces.append(self.bos_token)
            elif token_id == self.eos_token_id:
                pieces.append(self.eos_token)

        if byte_buffer:
            pieces.append(byte_buffer.decode("utf-8", errors="ignore"))

        return "".join(pieces)

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        return_tensors: str | None = None,
    ):
        del truncation
        input_ids = self.encode(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
        return SimpleNamespace(input_ids=input_ids)

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        tokens = []
        for token_id in ids:
            if token_id == self.pad_token_id:
                tokens.append(self.pad_token)
            elif token_id == self.bos_token_id:
                tokens.append(self.bos_token)
            elif token_id == self.eos_token_id:
                tokens.append(self.eos_token)
            else:
                tokens.append(bytes([token_id - BYTE_OFFSET]).decode("utf-8", errors="ignore"))
        return tokens

    def save_pretrained(self, save_directory: str | Path):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        config = {
            "tokenizer_type": "byte",
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "byte_offset": BYTE_OFFSET,
            "vocab_size": VOCAB_SIZE,
            "model_max_length": self.model_max_length,
        }

        with (save_path / "tokenizer_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        return (str(save_path),)

    @classmethod
    def from_pretrained(cls, load_directory: str | Path) -> "ByteTokenizer":
        config_path = Path(load_directory) / "tokenizer_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Tokenizer config not found at {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        tokenizer = cls(model_max_length=config.get("model_max_length"))
        expected = {
            "pad_token_id": PAD_TOKEN_ID,
            "bos_token_id": BOS_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Unsupported tokenizer config: {key}={config.get(key)}")

        return tokenizer

    @staticmethod
    def _flatten_ids(ids: Sequence[int] | torch.Tensor) -> list[int]:
        if isinstance(ids, torch.Tensor):
            return ids.detach().cpu().view(-1).tolist()

        if ids and isinstance(ids[0], (list, tuple)):
            flat: list[int] = []
            for item in ids:
                flat.extend(item)
            return flat

        return list(ids)


def load_tokenizer(tokenizer_path: str | Path | None = None) -> ByteTokenizer:
    if tokenizer_path is not None:
        path = Path(tokenizer_path)
        if (path / "tokenizer_config.json").exists():
            return ByteTokenizer.from_pretrained(path)

    if (DEFAULT_TOKENIZER_DIR / "tokenizer_config.json").exists():
        return ByteTokenizer.from_pretrained(DEFAULT_TOKENIZER_DIR)

    return ByteTokenizer()
