import argparse
import os
import random
from typing import List, Dict, Optional
from datasets import load_dataset
from tools.tokenizer import load_tokenizer


# ============================================================
# Helpers
# ============================================================

def strip_think_tags(text: str) -> str:
    # Keep content, remove explicit markers
    return (
        text.replace("<think>", "")
            .replace("</think>", "")
            .replace("<think/>", "")
    )


# ============================================================
# Dataset-aware text extraction
# ============================================================

def extract_by_repo(repo: str, ex: dict) -> Optional[str]:
    # --------------------------------------------------------
    # Salesforce / Wikitext
    # --------------------------------------------------------
    if repo == "Salesforce/wikitext":
        text = ex.get("text")
        if isinstance(text, str) and len(text) > 50:
            return text
        return None

    # --------------------------------------------------------
    # OpenWebText
    # --------------------------------------------------------
    if repo == "Skylion007/openwebtext":
        text = ex.get("text")
        if isinstance(text, str) and len(text) > 50:
            return text
        return None

    # --------------------------------------------------------
    # FreedomIntelligence medical-o1-reasoning-SFT
    # --------------------------------------------------------
    if repo == "FreedomIntelligence/medical-o1-reasoning-SFT":
        q = ex.get("Question")
        cot = ex.get("Complex_Cot")
        r = ex.get("Response")

        parts = []
        if isinstance(q, str):
            parts.append(f"Question:\n{q}")
        if isinstance(cot, str):
            parts.append(f"Reasoning:\n{cot}")
        if isinstance(r, str):
            parts.append(f"Answer:\n{r}")

        text = "\n\n".join(parts)
        return text if len(text) > 50 else None

    # --------------------------------------------------------
    # Facebook Natural Reasoning
    # --------------------------------------------------------
    if repo == "facebook/natural_reasoning":
        q = ex.get("question")
        ref = ex.get("reference_answer")
        responses = ex.get("responses")

        parts = []
        if isinstance(q, str):
            parts.append(f"Question:\n{q}")

        if isinstance(responses, list) and responses:
            resp = responses[0].get("response")
            if isinstance(resp, str):
                parts.append(f"Reasoning:\n{resp}")

        if isinstance(ref, str):
            parts.append(f"Answer:\n{ref}")

        text = "\n\n".join(parts)
        return text if len(text) > 50 else None

    # --------------------------------------------------------
    # CodeSearchNet
    # --------------------------------------------------------
    if repo == "sentence-transformers/codesearchnet":
        comment = ex.get("comment")
        code = ex.get("code")

        if isinstance(code, str) and len(code) > 20:
            if isinstance(comment, str) and len(comment) > 10:
                return f"{comment}\n\n{code}"
            return code
        return None

    # --------------------------------------------------------
    # StackMathQA
    # --------------------------------------------------------
    if repo == "math-ai/StackMathQA":
        q = ex.get("Q")
        a = ex.get("A")

        if isinstance(q, str) and isinstance(a, str):
            return f"Question:\n{q}\n\nAnswer:\n{a}"
        return None

    # --------------------------------------------------------
    # FineMath
    # --------------------------------------------------------
    if repo == "HuggingFaceTB/finemath":
        if ex.get("language") != "en":
            return None
        text = ex.get("text")
        if isinstance(text, str) and len(text) > 100:
            return text
        return None

    # --------------------------------------------------------
    # AI-MO NuminaMath-CoT
    # --------------------------------------------------------
    if repo == "AI-MO/NuminaMath-CoT":
        problem = ex.get("problem")
        solution = ex.get("solution")

        if isinstance(problem, str) and isinstance(solution, str):
            return f"Problem:\n{problem}\n\nSolution:\n{solution}"
        return None
    
    # --------------------------------------------------------
    # CodeFeedback-Filtered-Instruction
    # --------------------------------------------------------
    if repo == "m-a-p/CodeFeedback-Filtered-Instruction":
        query = ex.get("query")
        answer = ex.get("answer")
        lang = ex.get("lang")

        if not isinstance(query, str) or not isinstance(answer, str):
            return None

        if isinstance(lang, str):
            return f"Instruction ({lang}):\n{query}\n\nResponse:\n{answer}"

        return f"Instruction:\n{query}\n\nResponse:\n{answer}"

    # --------------------------------------------------------
    # FineWeb (English only)
    # --------------------------------------------------------
    if repo == "HuggingFaceFW/fineweb":
        if ex.get("language") != "en":
            return None

        if ex.get("language_score", 0.0) < 0.8:
            return None

        text = ex.get("text")
        if isinstance(text, str) and len(text) > 200:
            return text

        return None

    # --------------------------------------------------------
    # GlaiveAI reasoning-v1-20m
    # --------------------------------------------------------
    if repo == "glaiveai/reasoning-v1-20m":
        prompt = ex.get("prompt")
        response = ex.get("response")

        if isinstance(prompt, str) and isinstance(response, str):
            response = strip_think_tags(response)
            return f"Prompt:\n{prompt}\n\nAnswer:\n{response}"
        return None

    return None


# ============================================================
# Main merge logic
# ============================================================

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

    # Group specs by category
    category_to_specs: Dict[str, List[Dict]] = {}
    for s in specs:
        category_to_specs.setdefault(s["category"], []).append(s)

    # Normalize recipe
    total_frac = sum(recipe.values())
    recipe = {k: v / total_frac for k, v in recipe.items()}

    # Token budgets
    cat_token_targets = {
        c: int(total_tokens * frac) for c, frac in recipe.items()
    }

    spec_token_targets: Dict[str, int] = {}
    for cat, specs_in_cat in category_to_specs.items():
        total = cat_token_targets.get(cat, 0)
        base = total // len(specs_in_cat)
        rem = total % len(specs_in_cat)

        for i, s in enumerate(specs_in_cat):
            key = f"{s['repo']}::{s.get('config','')}"
            spec_token_targets[key] = base + (1 if i < rem else 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    buffer: List[int] = []
    written_tokens = 0
    written_seqs = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for cat, specs_in_cat in category_to_specs.items():
            random.shuffle(specs_in_cat)

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

                    text = extract_by_repo(repo, ex)
                    if not text:
                        continue

                    ids = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=False,
                    ).input_ids

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

        # ----------------------------------------------------
        # FIX: pad and write remainder instead of dropping
        # ----------------------------------------------------
        if buffer:
            pad = seq_len - len(buffer)
            # buffer.extend([eos] * pad)
            pad_id = tokenizer.pad_token_id or eos
            buffer.extend([pad_id] * pad)

            out_f.write(" ".join(map(str, buffer)) + "\n")
            written_seqs += 1
            written_tokens += seq_len

    print(f"[ok] wrote {written_seqs:,} sequences ({written_tokens:,} tokens)")
    print(f"[ok] output: {output_path}")


# ============================================================
# Specs
# ============================================================

def _default_specs():
    return [
        {"repo": "Salesforce/wikitext", "config": "wikitext-103-v1", "split": "train", "category": "wikipedia"},
        {"repo": "Skylion007/openwebtext", "config": None, "split": "train", "category": "web"},
        {"repo": "HuggingFaceFW/fineweb", "config": "CC-MAIN-2016-36", "split": "train", "category": "web"},

        {"repo": "FreedomIntelligence/medical-o1-reasoning-SFT", "config": "en", "split": "train", "category": "reasoning"},
        {"repo": "facebook/natural_reasoning", "config": None, "split": "train", "category": "books"},

        {"repo": "sentence-transformers/codesearchnet", "config": None, "split": "train", "category": "code"},
        {"repo": "m-a-p/CodeFeedback-Filtered-Instruction", "config": None, "split": "train", "category": "code"},

        {"repo": "math-ai/StackMathQA", "config": "stackmathqa1600k", "split": "train", "category": "math"},
        {"repo": "HuggingFaceTB/finemath", "config": "finemath-3plus", "split": "train", "category": "math"},
        {"repo": "AI-MO/NuminaMath-CoT", "config": None, "split": "train", "category": "math"},
    ]



# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="tokens/packed_tokens.txt")
    parser.add_argument("--total-tokens", type=int, default=1000_000_000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    recipe = {
        "wikipedia": 0.22,
        "books": 0.18,
        "web": 0.18,      # FineWeb needs more weight to matter
        "code": 0.20,     # Two code datasets now
        "math": 0.17,     # Still strong due to Numina + FineMath
        "reasoning": 0.05,
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
