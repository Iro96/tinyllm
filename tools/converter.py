import os
from pathlib import Path
from tools.tokenizer import load_tokenizer


def pack_text_files(
    input_paths,
    output_path="data/packed_tokens.txt",
    add_eos=True,
    min_chars=1,
):
    """
    Converts raw text files into a packed token ID file.

    Args:
        input_paths (list[str]): Paths to .txt files with raw text
        output_path (str): Where to write packed token IDs
        add_eos (bool): Append EOS token after each document
        min_chars (int): Skip very short documents
    """
    tokenizer = load_tokenizer()
    eos_id = tokenizer.eos_token_id

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_tokens = 0
    doc_count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for path in input_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(path)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if len(text) < min_chars:
                continue

            # Tokenize WITHOUT adding special tokens
            ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            ).input_ids

            if not ids:
                continue

            if add_eos:
                ids.append(eos_id)

            # Write as space-separated integers
            out.write(" ".join(map(str, ids)))
            out.write("\n")

            total_tokens += len(ids)
            doc_count += 1

    print(f"Packed {doc_count} documents")
    print(f"Total tokens: {total_tokens}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    # Example usage
    pack_text_files(
        input_paths=[
            "data/test_essay.txt"
        ],
        output_path="data/test_tokens.txt",
    )
