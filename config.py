from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    ffn_mult: int = 4
    max_seq_len: int = 8192  # Increased for longer contexts
    dropout_rate: float = 0.0
    # New parameters for efficient attention
    sliding_window: int = 2048  # Local attention window
    use_sliding_window: bool = True  # Enable sliding window attention


@dataclass
class TrainConfig:
    lr: float = 1e-4  # conservative for CPU small-batch training
    weight_decay: float = 0.01
    batch_size: int = 2  # Reduced for longer sequences
    grad_accum: int = 16  # Increased accumulation for effective batch size
    device: str = "auto"
    warmup_steps: int = 500
    total_steps: int = 10000
    seed: int = 42
    train_source_path: str = "src/fineweb_train.jsonl"
    valid_source_path: str = "src/fineweb_valid.jsonl"
    train_tokens_path: str = "src/processed/fineweb_train_tokens.txt"
    valid_tokens_path: str = "src/processed/fineweb_valid_tokens.txt"
    tokenizer_dir: str = "tokenizer/terry_byte"
    dataset_repo: str = "HuggingFaceFW/fineweb"
    dataset_name: str = "CC-MAIN-2025-26"
    dataset_split: str = "train"
    train_target_tokens: int = 500_000_000
    valid_target_tokens: int = 10_000_000
    buffer_size: int = 100_000
    language: str = "en"
    min_language_score: float = 0.95
    min_token_count: int = 100
    max_token_count: int = 4096
    use_mixed_precision: bool = True  # Enable mixed precision for memory efficiency
