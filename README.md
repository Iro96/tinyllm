# TinyLLM

TinyLLM is a compact Transformer project for training and chatting with Terry, a tiny synthetic assistant.

## Data flow

1. Generate Terry conversations:

```bash
python data/generate_terry_dataset.py
```

This writes:

- `src/terry_daily_chat_train.jsonl`
- `src/terry_daily_chat_valid.jsonl`

1. Prepare tokenized training data:

```bash
python prepare_data.py
```

This writes:

- `src/processed/terry_train_tokens.txt`
- `src/processed/terry_valid_tokens.txt`
- `tokenizer/terry_byte/tokenizer_config.json`

1. Train:

```bash
python train.py
```

## Tokenizer

The project uses a local byte-level tokenizer with fixed special token IDs:

- `0`: `<|pad|>`
- `1`: `<|im_start|>`
- `2`: `<|im_end|>`

## Inference

```bash
python example_usage.py
```
