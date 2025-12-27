from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 160
    n_layers: int = 5
    n_heads: int = 4
    ffn_mult: int = 4
    max_seq_len: int = 1024
    dropout_rate: float = 0.1


@dataclass
class TrainConfig:
    lr: float = 1e-4  # Reduced from 3e-4 for stability
    weight_decay: float = 0.1
    batch_size: int = 2
    grad_accum: int = 16
    device: str = "auto"
    warmup_steps: int = 1000  # Added warmup steps
    total_steps: int = 20000  # Target: 200k optimizer steps (true metric)