from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from config import ModelConfig
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


def encode_message(
    tokenizer,
    role: str,
    content: str,
) -> tuple[list[int], list[int]]:
    """Encode a single chat message with role tokens included.

    Role tokens are included in the sequence as boundary markers,
    but are masked out for loss. Only assistant content tokens are supervised.
    """
    role_tokens = tokenizer.encode(f"{role}\n", add_special_tokens=False)
    content_tokens = tokenizer.encode(
        f"{content.strip()}\n<|im_end|>\n",
        add_special_tokens=False,
    )

    is_assistant = role == "assistant"
    role_mask = [0] * len(role_tokens)
    content_mask = [1 if is_assistant else 0] * len(content_tokens)

    return role_tokens + content_tokens, role_mask + content_mask


def serialize_chat_record(
    tokenizer,
    messages: Iterable[dict[str, str]],
    include_system_prompt: bool = True,
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[int]]:
    """Convert chat messages into one trainable token sequence with loss mask.

    Returns:
        Tuple of (token_ids, loss_mask) where loss_mask[i]=1 for target tokens
        that should contribute to loss. Single BOS at start.
    """
    tokens: list[int] = [tokenizer.bos_token_id]
    loss_mask: list[int] = [0]  # BOS token not included in loss

    if include_system_prompt:
        sys_tokens, sys_mask = encode_message(
            tokenizer,
            role="system",
            content=SYSTEM_PROMPT,
        )
        tokens.extend(sys_tokens)
        loss_mask.extend(sys_mask)

    for msg in messages:
        assert msg["role"] in {"system", "user", "assistant"}
        msg_tokens, msg_mask = encode_message(
            tokenizer=tokenizer,
            role=msg["role"],
            content=msg["content"],
        )
        tokens.extend(msg_tokens)
        loss_mask.extend(msg_mask)

    if add_generation_prompt:
        gen_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)
        tokens.extend(gen_tokens)
        loss_mask.extend([0] * len(gen_tokens))
    else:
        tokens.append(tokenizer.eos_token_id)
        loss_mask.append(1)

    return tokens, loss_mask


def build_generation_prompt(tokenizer, user_prompt: str) -> list[int]:
    """Build the same chat prompt format used during training for generation.
    
    Returns token_ids only (no mask needed for generation).
    """
    tokens, _ = serialize_chat_record(
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": user_prompt}],
        include_system_prompt=True,
        add_generation_prompt=True,
    )
    return tokens


def write_tokenized_split(
    source_path: str | Path,
    target_path: str | Path,
    tokenizer,
    max_seq_len: int = 8192,  # Allow longer sequences
) -> dict[str, int]:
    """Encode one JSONL split into one-doc-per-line token IDs with masks.
    
    Saves two files:
    - target_path: space-separated token IDs
    - target_path.mask: space-separated loss mask (1=train, 0=ignore)
    """
    target = Path(target_path)
    mask_path = Path(str(target_path) + ".mask")
    target.parent.mkdir(parents=True, exist_ok=True)

    documents = 0
    total_tokens = 0

    with target.open("w", encoding="utf-8") as token_file, \
         mask_path.open("w", encoding="utf-8") as mask_file:
        for record in iter_jsonl_records(source_path):
            ids, mask = serialize_chat_record(tokenizer, record["messages"])
            max_length = max_seq_len  # Use parameter instead of config
            if len(ids) > max_length:
                ids = ids[:max_length]
                mask = mask[:max_length]
            
            # Skip very short sequences
            if len(ids) < 2:
                continue
            
            # Verify mask and tokens are same length
            assert len(ids) == len(mask), \
                f"Token/mask mismatch: {len(ids)} tokens vs {len(mask)} mask values"
            
            # Write tokens and mask
            token_file.write(" ".join(map(str, ids)))
            token_file.write("\n")
            mask_file.write(" ".join(map(str, mask)))
            mask_file.write("\n")
            
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

    train_stats = write_tokenized_split(train_source, train_tokens, tokenizer, max_seq_len=24576)  # Allow up to 24K tokens
    valid_stats = write_tokenized_split(valid_source, valid_tokens, tokenizer, max_seq_len=24576)

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
