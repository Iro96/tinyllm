# TinyLLM

TinyLLM is a compact Transformer project for training and chatting with Terry, a tiny synthetic assistant. This model now supports **long context reasoning** with sequences up to 25K tokens using efficient sliding window attention.

## Key Features

- **Long Context Support**: Handle 10K-25K token sequences with sliding window attention
- **Memory Efficient**: Gradient checkpointing and mixed precision training
- **Reasoning Capabilities**: Improved attention mechanism for better reasoning over long contexts
- **Compact Architecture**: 256-dimensional embeddings, 8 layers, 8 attention heads

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

## Configuration

Key parameters in `config.py`:

```python
@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 8192  # Maximum sequence length
    sliding_window: int = 2048  # Local attention window
    use_sliding_window: bool = True
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
