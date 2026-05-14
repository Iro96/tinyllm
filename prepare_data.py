from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset_builder import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_REPO,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_TOKEN_COUNT,
    DEFAULT_MIN_LANGUAGE_SCORE,
    DEFAULT_MIN_TOKEN_COUNT,
    DEFAULT_TOKENIZER_DIR,
    DEFAULT_TRAIN_SOURCE,
    DEFAULT_TRAIN_TOKENS,
    DEFAULT_TRAIN_TARGET_TOKENS,
    DEFAULT_VALID_SOURCE,
    DEFAULT_VALID_TOKENS,
    DEFAULT_VALID_TARGET_TOKENS,
    prepare_dataset_assets,
)
from tools.tokenizer import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID

# Special tokens
pad_id: int = PAD_TOKEN_ID
bos_id: int = BOS_TOKEN_ID  # <|im_start|>
eos_id: int = EOS_TOKEN_ID  # <|im_end|>


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream FineWeb, cache filtered JSONL splits, and encode them for training.",
    )
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--valid-source", type=Path, default=DEFAULT_VALID_SOURCE)
    parser.add_argument("--train-tokens", type=Path, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--valid-tokens", type=Path, default=DEFAULT_VALID_TOKENS)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--dataset-repo", type=str, default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument(
        "--train-target-tokens",
        type=int,
        default=DEFAULT_TRAIN_TARGET_TOKENS,
    )
    parser.add_argument(
        "--valid-target-tokens",
        type=int,
        default=DEFAULT_VALID_TARGET_TOKENS,
    )
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE)
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE)
    parser.add_argument(
        "--min-language-score",
        type=float,
        default=DEFAULT_MIN_LANGUAGE_SCORE,
    )
    parser.add_argument("--min-token-count", type=int, default=DEFAULT_MIN_TOKEN_COUNT)
    parser.add_argument("--max-token-count", type=int, default=DEFAULT_MAX_TOKEN_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the cached FineWeb JSONL splits before tokenization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stats = prepare_dataset_assets(
        train_source=args.train_source,
        valid_source=args.valid_source,
        train_tokens=args.train_tokens,
        valid_tokens=args.valid_tokens,
        tokenizer_dir=args.tokenizer_dir,
        dataset_repo=args.dataset_repo,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        train_target_tokens=args.train_target_tokens,
        valid_target_tokens=args.valid_target_tokens,
        buffer_size=args.buffer_size,
        language=args.language,
        min_language_score=args.min_language_score,
        min_token_count=args.min_token_count,
        max_token_count=args.max_token_count,
        seed=args.seed,
        force=args.force,
    )

    print("Prepared FineWeb dataset assets")
    print(f"pad_id={pad_id} bos_id={bos_id} eos_id={eos_id}")
    print(
        "dataset: "
        f"{stats['dataset_repo']} / {stats['dataset_name']} [{stats['dataset_split']}]"
    )
    print(f"train source: {stats['train_source']}")
    print(f"valid source: {stats['valid_source']}")
    if stats["train_source_token_count"] is None:
        print("source totals: reused cached FineWeb splits")
    else:
        print(
            "source totals: "
            f"train={stats['train_source_token_count']} "
            f"valid={stats['valid_source_token_count']}"
        )
    print(f"train tokens: {stats['train_tokens']} ({stats['train_documents']} docs)")
    print(f"valid tokens: {stats['valid_tokens']} ({stats['valid_documents']} docs)")
    print(f"tokenizer dir: {stats['tokenizer_dir']}")


if __name__ == "__main__":
    main()
