from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from tools.tokenizer import load_tokenizer

DEFAULT_PROCESSED_DIR = Path("src/processed")
DEFAULT_TRAIN_SOURCE = Path("src/fineweb_train.jsonl")
DEFAULT_VALID_SOURCE = Path("src/fineweb_valid.jsonl")
DEFAULT_TRAIN_TOKENS = DEFAULT_PROCESSED_DIR / "fineweb_train_tokens.txt"
DEFAULT_VALID_TOKENS = DEFAULT_PROCESSED_DIR / "fineweb_valid_tokens.txt"
DEFAULT_TOKENIZER_DIR = Path("tokenizer/terry_byte")

DEFAULT_DATASET_REPO = "HuggingFaceFW/fineweb"
DEFAULT_DATASET_NAME = "CC-MAIN-2025-26"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_TRAIN_TARGET_TOKENS = 500_000_000
DEFAULT_VALID_TARGET_TOKENS = 10_000_000
DEFAULT_BUFFER_SIZE = 100_000
DEFAULT_LANGUAGE = "en"
DEFAULT_MIN_LANGUAGE_SCORE = 0.95
DEFAULT_MIN_TOKEN_COUNT = 100
DEFAULT_MAX_TOKEN_COUNT = 4096


def _load_hf_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for FineWeb preparation. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return load_dataset


def iter_jsonl_records(path: str | Path) -> Iterator[dict]:
    """Yield JSONL records from a dataset split."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def serialize_text_record(
    tokenizer,
    text: str,
) -> tuple[list[int], list[int]]:
    """Convert a plain text document into one trainable token sequence."""
    normalized = text.strip()
    if not normalized:
        return [], []

    content_tokens = tokenizer.encode(normalized, add_special_tokens=False)
    tokens = [tokenizer.bos_token_id, *content_tokens, tokenizer.eos_token_id]
    loss_mask = [0, *([1] * len(content_tokens)), 1]
    return tokens, loss_mask


def serialize_record(tokenizer, record: dict) -> tuple[list[int], list[int]]:
    """Serialize a plain-text dataset record."""
    if "text" in record:
        return serialize_text_record(tokenizer, record["text"])
    raise KeyError("Expected a record with a `text` field.")


def write_tokenized_split(
    source_path: str | Path,
    target_path: str | Path,
    tokenizer,
    max_seq_len: int = 8192,
) -> dict[str, int]:
    """Encode one JSONL split into one-doc-per-line token IDs with masks."""
    target = Path(target_path)
    mask_path = Path(str(target_path) + ".mask")
    target.parent.mkdir(parents=True, exist_ok=True)

    documents = 0
    total_tokens = 0

    with target.open("w", encoding="utf-8") as token_file, mask_path.open(
        "w", encoding="utf-8"
    ) as mask_file:
        for record in iter_jsonl_records(source_path):
            ids, mask = serialize_record(tokenizer, record)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
                mask = mask[:max_seq_len]

            if len(ids) < 2:
                continue

            assert len(ids) == len(mask), (
                f"Token/mask mismatch: {len(ids)} tokens vs {len(mask)} mask values"
            )

            token_file.write(" ".join(map(str, ids)))
            token_file.write("\n")
            mask_file.write(" ".join(map(str, mask)))
            mask_file.write("\n")

            documents += 1
            total_tokens += len(ids)

    return {"documents": documents, "tokens": total_tokens}


def write_fineweb_splits(
    train_path: str | Path = DEFAULT_TRAIN_SOURCE,
    valid_path: str | Path = DEFAULT_VALID_SOURCE,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    train_target_tokens: int = DEFAULT_TRAIN_TARGET_TOKENS,
    valid_target_tokens: int = DEFAULT_VALID_TARGET_TOKENS,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    language: str = DEFAULT_LANGUAGE,
    min_language_score: float = DEFAULT_MIN_LANGUAGE_SCORE,
    min_token_count: int = DEFAULT_MIN_TOKEN_COUNT,
    max_token_count: int = DEFAULT_MAX_TOKEN_COUNT,
    seed: int = 42,
) -> dict[str, object]:
    """Stream FineWeb, filter records, and cache train/valid JSONL splits."""
    load_dataset = _load_hf_dataset()

    train = Path(train_path)
    valid = Path(valid_path)
    train.parent.mkdir(parents=True, exist_ok=True)
    valid.parent.mkdir(parents=True, exist_ok=True)

    stream_ds = load_dataset(
        dataset_repo,
        name=dataset_name,
        split=dataset_split,
        streaming=True,
    )
    stream_ds = stream_ds.shuffle(seed=seed, buffer_size=buffer_size)

    rng = random.Random(seed)
    train_tokens = 0
    valid_tokens = 0
    train_samples = 0
    valid_samples = 0

    with train.open("w", encoding="utf-8") as train_handle, valid.open(
        "w", encoding="utf-8"
    ) as valid_handle:
        for sample in stream_ds:
            if sample.get("language") != language:
                continue

            language_score = float(sample.get("language_score") or 0.0)
            if language_score < min_language_score:
                continue

            token_count = int(sample.get("token_count") or 0)
            if token_count < min_token_count or token_count > max_token_count:
                continue

            text = str(sample.get("text") or "").strip()
            if not text:
                continue

            record = {
                "text": text,
                "language": language,
                "language_score": language_score,
                "token_count": token_count,
            }

            train_remaining = max(0, train_target_tokens - train_tokens)
            valid_remaining = max(0, valid_target_tokens - valid_tokens)
            if train_remaining == 0 and valid_remaining == 0:
                break

            if train_remaining > 0 and valid_remaining > 0:
                valid_probability = valid_remaining / (train_remaining + valid_remaining)
                write_to_valid = rng.random() < valid_probability
            else:
                write_to_valid = valid_remaining > 0

            handle = valid_handle if write_to_valid else train_handle
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

            if write_to_valid:
                valid_tokens += token_count
                valid_samples += 1
            else:
                train_tokens += token_count
                train_samples += 1

            total_samples = train_samples + valid_samples
            if total_samples % 10_000 == 0:
                print(
                    f"samples={total_samples:,} | "
                    f"train_tokens={train_tokens:,} | "
                    f"valid_tokens={valid_tokens:,}"
                )

    return {
        "dataset_repo": dataset_repo,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "train_path": str(train),
        "valid_path": str(valid),
        "train_samples": train_samples,
        "valid_samples": valid_samples,
        "train_source_tokens": train_tokens,
        "valid_source_tokens": valid_tokens,
    }


def ensure_source_dataset(
    train_path: str | Path = DEFAULT_TRAIN_SOURCE,
    valid_path: str | Path = DEFAULT_VALID_SOURCE,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    train_target_tokens: int = DEFAULT_TRAIN_TARGET_TOKENS,
    valid_target_tokens: int = DEFAULT_VALID_TARGET_TOKENS,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    language: str = DEFAULT_LANGUAGE,
    min_language_score: float = DEFAULT_MIN_LANGUAGE_SCORE,
    min_token_count: int = DEFAULT_MIN_TOKEN_COUNT,
    max_token_count: int = DEFAULT_MAX_TOKEN_COUNT,
    seed: int = 42,
    force: bool = False,
) -> dict[str, object]:
    """Create cached FineWeb JSONL splits when they do not exist yet."""
    train = Path(train_path)
    valid = Path(valid_path)

    if force or not train.exists() or not valid.exists():
        return write_fineweb_splits(
            train_path=train,
            valid_path=valid,
            dataset_repo=dataset_repo,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            train_target_tokens=train_target_tokens,
            valid_target_tokens=valid_target_tokens,
            buffer_size=buffer_size,
            language=language,
            min_language_score=min_language_score,
            min_token_count=min_token_count,
            max_token_count=max_token_count,
            seed=seed,
        )

    return {
        "dataset_repo": dataset_repo,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "train_path": str(train),
        "valid_path": str(valid),
        "train_samples": None,
        "valid_samples": None,
        "train_source_tokens": None,
        "valid_source_tokens": None,
    }


def prepare_dataset_assets(
    train_source: str | Path = DEFAULT_TRAIN_SOURCE,
    valid_source: str | Path = DEFAULT_VALID_SOURCE,
    train_tokens: str | Path = DEFAULT_TRAIN_TOKENS,
    valid_tokens: str | Path = DEFAULT_VALID_TOKENS,
    tokenizer_dir: str | Path = DEFAULT_TOKENIZER_DIR,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    train_target_tokens: int = DEFAULT_TRAIN_TARGET_TOKENS,
    valid_target_tokens: int = DEFAULT_VALID_TARGET_TOKENS,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    language: str = DEFAULT_LANGUAGE,
    min_language_score: float = DEFAULT_MIN_LANGUAGE_SCORE,
    min_token_count: int = DEFAULT_MIN_TOKEN_COUNT,
    max_token_count: int = DEFAULT_MAX_TOKEN_COUNT,
    seed: int = 42,
    force: bool = False,
) -> dict[str, object]:
    """Ensure FineWeb source JSONL, tokenizer files, and tokenized splits exist."""
    source_stats = ensure_source_dataset(
        train_path=train_source,
        valid_path=valid_source,
        dataset_repo=dataset_repo,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        train_target_tokens=train_target_tokens,
        valid_target_tokens=valid_target_tokens,
        buffer_size=buffer_size,
        language=language,
        min_language_score=min_language_score,
        min_token_count=min_token_count,
        max_token_count=max_token_count,
        seed=seed,
        force=force,
    )

    tokenizer = load_tokenizer()
    tokenizer.save_pretrained(tokenizer_dir)

    train_stats = write_tokenized_split(
        train_source,
        train_tokens,
        tokenizer,
        max_seq_len=24576,
    )
    valid_stats = write_tokenized_split(
        valid_source,
        valid_tokens,
        tokenizer,
        max_seq_len=24576,
    )

    return {
        "dataset_repo": dataset_repo,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "train_source": str(train_source),
        "valid_source": str(valid_source),
        "train_tokens": str(train_tokens),
        "valid_tokens": str(valid_tokens),
        "tokenizer_dir": str(tokenizer_dir),
        "train_documents": train_stats["documents"],
        "valid_documents": valid_stats["documents"],
        "train_token_count": train_stats["tokens"],
        "valid_token_count": valid_stats["tokens"],
        "train_source_samples": source_stats["train_samples"],
        "valid_source_samples": source_stats["valid_samples"],
        "train_source_token_count": source_stats["train_source_tokens"],
        "valid_source_token_count": source_stats["valid_source_tokens"],
    }
