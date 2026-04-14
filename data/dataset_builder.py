# data/dataset_builder.py

from datasets import load_dataset


def stream_wikipedia(
    tokenizer,
    max_seq_len: int,
    max_tokens: int,
    cache_dir: str,
):
    """
    Stream Wikipedia and return tokenized document chunks.
    """
    print("Streaming Wikipedia 20231101.en...")

    raw = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )

    documents = []
    token_count = 0
    chunk_len = max_seq_len - 1

    for ex in raw:
        text = ex["text"]

        if not text or not text.strip():
            continue

        ids = tokenizer(text, add_special_tokens=False).input_ids

        if len(ids) < 2:
            continue

        # chunk long documents
        for i in range(0, len(ids), chunk_len):
            chunk = ids[i : i + chunk_len]

            if len(chunk) > 1:
                chunk.append(tokenizer.eos_token_id)
                documents.append(chunk)
                token_count += len(chunk)

            if token_count >= max_tokens:
                break

        if token_count >= max_tokens:
            break

    print(f"Collected {len(documents)} documents (~{token_count} tokens)")
    return documents