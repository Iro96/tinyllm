from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset_builder import (
    DEFAULT_TOKENIZER_DIR,
    DEFAULT_TRAIN_TOKENS,
    DEFAULT_VALID_TOKENS,
    prepare_dataset_assets,
)
from data.generate_terry_dataset import DEFAULT_TRAIN_PATH, DEFAULT_VALID_PATH
from tools.tokenizer import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID

# Special tokens
pad_id: int = PAD_TOKEN_ID
bos_id: int = BOS_TOKEN_ID  # <|im_start|>
eos_id: int = EOS_TOKEN_ID  # <|im_end|>


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Terry chat data and encode it for training.",
    )
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-source", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--train-tokens", type=Path, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--valid-tokens", type=Path, default=DEFAULT_VALID_TOKENS)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--train-samples", type=int, default=60_000)
    parser.add_argument("--valid-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the JSONL dataset before tokenization.",
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
        train_samples=args.train_samples,
        valid_samples=args.valid_samples,
        seed=args.seed,
        force=args.force,
    )

    print("Prepared Terry dataset assets")
    print(f"pad_id={pad_id} bos_id={bos_id} eos_id={eos_id}")
    print(f"train source: {stats['train_source']}")
    print(f"valid source: {stats['valid_source']}")
    print(f"train tokens: {stats['train_tokens']} ({stats['train_documents']} docs)")
    print(f"valid tokens: {stats['valid_tokens']} ({stats['valid_documents']} docs)")
    print(f"tokenizer dir: {stats['tokenizer_dir']}")


if __name__ == "__main__":
    main()
