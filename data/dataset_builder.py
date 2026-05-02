from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from data.generate_terry_dataset import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
    write_dataset_splits,
)
from tools.tokenizer import load_tokenizer

DEFAULT_PROCESSED_DIR = Path("src/processed")
DEFAULT_TRAIN_TOKENS = DEFAULT_PROCESSED_DIR / "terry_train_tokens.txt"
DEFAULT_VALID_TOKENS = DEFAULT_PROCESSED_DIR / "terry_valid_tokens.txt"
DEFAULT_TOKENIZER_DIR = Path("tokenizer/terry_byte")

SYSTEM_PROMPT = (
    "you are terry, a tiny human brain.\n"
    "you speak in short, lowercase sentences.\n"
    "you are friendly, curious, and a little dumb.\n"
    "you only know the user is your owner.\n"
    "you have limited experience outside the home."
)


def iter_jsonl_records(path: str | Path) -> Iterator[dict]:
    """Yield JSONL records from a dataset split."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def encode_message(tokenizer, role: str, content: str) -> list[int]:
    """Serialize a single chat message with im_start/im_end wrappers."""
    body = f"{role}\n{content.strip()}"
    return [
        tokenizer.bos_token_id,
        *tokenizer.encode(body, add_special_tokens=False),
        tokenizer.eos_token_id,
    ]


def serialize_chat_record(
    tokenizer,
    messages: Iterable[dict[str, str]],
    include_system_prompt: bool = True,
    add_generation_prompt: bool = False,
) -> list[int]:
    """Convert chat messages into one trainable token sequence."""
    tokens: list[int] = []

    if include_system_prompt:
        tokens.extend(encode_message(tokenizer, "system", SYSTEM_PROMPT))

    for msg in messages:
        tokens.extend(
            encode_message(
                tokenizer=tokenizer,
                role=msg["role"],
                content=msg["content"],
            )
        )

    if add_generation_prompt:
        tokens.append(tokenizer.bos_token_id)
        tokens.extend(tokenizer.encode("assistant\n", add_special_tokens=False))

    return tokens


def build_generation_prompt(tokenizer, user_prompt: str) -> list[int]:
    """Build the same chat prompt format used during training."""
    return serialize_chat_record(
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": user_prompt}],
        include_system_prompt=True,
        add_generation_prompt=True,
    )


def write_tokenized_split(
    source_path: str | Path,
    target_path: str | Path,
    tokenizer,
) -> dict[str, int]:
    """Encode one JSONL split into one-doc-per-line token IDs."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    documents = 0
    total_tokens = 0

    with target.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl_records(source_path):
            ids = serialize_chat_record(tokenizer, record["messages"])
            if len(ids) < 2:
                continue

            handle.write(" ".join(map(str, ids)))
            handle.write("\n")
            documents += 1
            total_tokens += len(ids)

    return {"documents": documents, "tokens": total_tokens}


def ensure_source_dataset(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    valid_path: str | Path = DEFAULT_VALID_PATH,
    train_samples: int = 60_000,
    valid_samples: int = 2_000,
    seed: int = 42,
    force: bool = False,
) -> dict[str, object]:
    """Generate Terry JSONL files when they do not exist yet."""
    train = Path(train_path)
    valid = Path(valid_path)

    if force or not train.exists() or not valid.exists():
        return write_dataset_splits(
            train_path=train,
            valid_path=valid,
            train_samples=train_samples,
            valid_samples=valid_samples,
            seed=seed,
        )

    return {
        "train_path": str(train),
        "valid_path": str(valid),
        "train_samples": None,
        "valid_samples": None,
    }


def prepare_dataset_assets(
    train_source: str | Path = DEFAULT_TRAIN_PATH,
    valid_source: str | Path = DEFAULT_VALID_PATH,
    train_tokens: str | Path = DEFAULT_TRAIN_TOKENS,
    valid_tokens: str | Path = DEFAULT_VALID_TOKENS,
    tokenizer_dir: str | Path = DEFAULT_TOKENIZER_DIR,
    train_samples: int = 60_000,
    valid_samples: int = 2_000,
    seed: int = 42,
    force: bool = False,
) -> dict[str, object]:
    """Ensure source JSONL, tokenizer files, and tokenized splits exist."""
    ensure_source_dataset(
        train_path=train_source,
        valid_path=valid_source,
        train_samples=train_samples,
        valid_samples=valid_samples,
        seed=seed,
        force=force,
    )

    tokenizer = load_tokenizer()
    tokenizer.save_pretrained(tokenizer_dir)

    train_stats = write_tokenized_split(train_source, train_tokens, tokenizer)
    valid_stats = write_tokenized_split(valid_source, valid_tokens, tokenizer)

    return {
        "train_source": str(train_source),
        "valid_source": str(valid_source),
        "train_tokens": str(train_tokens),
        "valid_tokens": str(valid_tokens),
        "tokenizer_dir": str(tokenizer_dir),
        "train_documents": train_stats["documents"],
        "valid_documents": valid_stats["documents"],
        "train_token_count": train_stats["tokens"],
        "valid_token_count": valid_stats["tokens"],
    }
