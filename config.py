from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 160
    n_layers: int = 6
    n_heads: int = 8
    ffn_mult: int = 4
    max_seq_len: int = 1024
    dropout_rate: float = 0.1


@dataclass
class TrainConfig:
    lr: float = 1e-4  # conservative for CPU small-batch training
    weight_decay: float = 0.01
    batch_size: int = 1  # single-sample batches on CPU
    grad_accum: int = 16  # accumulate to get effective batch
    device: str = "cpu"
    warmup_steps: int = 2000
    total_steps: int = 50000