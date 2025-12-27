import argparse
import os
from typing import List, Dict, Optional
from datasets import load_dataset
from tools.tokenizer import load_tokenizer


def _extract_text(example: dict) -> Optional[str]:
    for key in ("text", "content", "body", "article", "wiki", "code"):
        v = example.get(key)
        if isinstance(v, str) and len(v) > 20:
            return v

    for v in example.values():
        if isinstance(v, str) and len(v) > 20:
            return v

    return None


def merge_with_recipe(
    output_path: str,
    specs: List[Dict],
    recipe: Dict[str, float],
    total_tokens: int,
    seq_len: int,
    cache_dir: Optional[str] = None,
):
    tokenizer = load_tokenizer()
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")

    recipe = dict(recipe)  # do not mutate caller

    # ---------------------------
    # Group specs by category
    # ---------------------------
    category_to_specs: Dict[str, List[Dict]] = {}
    for s in specs:
        category_to_specs.setdefault(s["category"], []).append(s)

    available_cats = set(category_to_specs)
    missing = set(recipe) - available_cats

    # Rebalance missing categories
    if missing:
        print(f"[info] Missing recipe categories: {missing}")
        extra = sum(recipe[c] for c in missing)
        for c in missing:
            del recipe[c]

        if "wikipedia" in recipe:
            recipe["wikipedia"] += extra
        else:
            per = extra / len(recipe)
            for c in recipe:
                recipe[c] += per

    # Normalize recipe
    total_frac = sum(recipe.values())
    recipe = {k: v / total_frac for k, v in recipe.items()}

    # ---------------------------
    # Token budgets (not docs)
    # ---------------------------
    cat_token_targets = {
        c: int(total_tokens * frac) for c, frac in recipe.items()
    }

    # Distribute rounding remainder
    remainder = total_tokens - sum(cat_token_targets.values())
    for c in sorted(cat_token_targets):
        if remainder <= 0:
            break
        cat_token_targets[c] += 1
        remainder -= 1

    # Per-spec token targets
    spec_token_targets: Dict[str, int] = {}
    for cat, specs_in_cat in category_to_specs.items():
        total = cat_token_targets.get(cat, 0)
        if total == 0:
            continue

        base = total // len(specs_in_cat)
        rem = total % len(specs_in_cat)

        for i, s in enumerate(specs_in_cat):
            key = f"{s['repo']}::{s.get('config','')}"
            spec_token_targets[key] = base + (1 if i < rem else 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---------------------------
    # Fixed-length token packing
    # ---------------------------
    buffer: List[int] = []
    written_tokens = 0
    written_seqs = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for cat, specs_in_cat in category_to_specs.items():
            for s in specs_in_cat:
                repo = s["repo"]
                config = s.get("config")
                split = s.get("split", "train")

                key = f"{repo}::{config or ''}"
                token_budget = spec_token_targets.get(key, 0)
                if token_budget <= 0:
                    continue

                print(f"[load] {repo}:{config}:{split} → {token_budget:,} tokens ({cat})")

                ds = load_dataset(
                    repo,
                    config if config else None,
                    split=split,
                    streaming=True,
                    cache_dir=cache_dir,
                )

                consumed = 0
                for ex in ds:
                    if consumed >= token_budget:
                        break

                    text = _extract_text(ex)
                    if not text:
                        continue

                    ids = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=seq_len,
                    ).input_ids

                    if not ids:
                        continue

                    ids.append(eos)

                    for tok in ids:
                        if consumed >= token_budget:
                            break

                        buffer.append(tok)
                        consumed += 1

                        if len(buffer) == seq_len:
                            out_f.write(" ".join(map(str, buffer)) + "\n")
                            buffer.clear()
                            written_seqs += 1
                            written_tokens += seq_len

                print(f"[done] {consumed:,} tokens consumed from {repo}")

        # Drop remainder buffer (intentional, for clean packing)
        if buffer:
            print(f"[info] Dropped {len(buffer)} trailing tokens (incomplete pack)")

    print(f"[ok] wrote {written_seqs:,} sequences ({written_tokens:,} tokens)")
    print(f"[ok] output: {output_path}")


def _default_specs():
    return [
        {"repo": "Salesforce/wikitext", "config": "wikitext-103-v1", "split": "train", "category": "wikipedia"},
        {"repo": "wikimedia/wikipedia", "config": "20231101.en", "split": "train", "category": "wikipedia"},
        {"repo": "Skylion007/openwebtext", "config": None, "split": "train", "category": "web"},
        {"repo": "sentence-transformers/codesearchnet", "config": None, "split": "train", "category": "code"},
        {"repo": "math-ai/StackMathQA", "config": "stackmathqa1600k", "split": "train", "category": "math"},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/packed_tokens.txt")
    parser.add_argument("--total-tokens", type=int, default=50_000_000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    recipe = {
        "wikipedia": 0.35,
        "books": 0.25,  # rebalanced if missing
        "web": 0.20,
        "code": 0.10,
        "math": 0.10,
    }

    merge_with_recipe(
        output_path=args.output,
        specs=_default_specs(),
        recipe=recipe,
        total_tokens=args.total_tokens,
        seq_len=args.seq_len,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
