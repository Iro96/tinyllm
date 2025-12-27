# TinyLLM Inference Guide

This guide explains how to load and use the trained TinyLLM model for text generation.

## Files Overview

- **`inference.py`** - Main inference class and utilities for loading and using the model
- **`example_usage.py`** - Example script demonstrating different usage patterns
- **`checkpoint_manager.py`** - Utilities for managing and inspecting model checkpoints

## Quick Start

### Basic Usage

```python
from inference import ModelInference

# Load the latest checkpoint
model = ModelInference()

# Generate text
prompt = "The future of artificial intelligence is"
generated = model.generate(prompt, max_length=50, temperature=0.8)
print(generated)
```

### Using the Convenience Function

```python
from inference import load_model

model = load_model()
generated = model.generate("Hello world", max_length=30)
```

## Model Loading

### From Default Location

```python
model = ModelInference()  # Loads from checkpoints/last_model.pt
```

### From Custom Path

```python
model = ModelInference(checkpoint_path="path/to/your/model.pt")
```

### Specifying Device

```python
# Auto-select device (GPU if available, otherwise CPU)
model = ModelInference(device="auto")

# Force CPU
model = ModelInference(device="cpu")

# Force CUDA (if available)
model = ModelInference(device="cuda")
```

## Text Generation Methods

### 1. High-Level Generation (`generate`)

Uses the transformers library's generation utilities:

```python
generated = model.generate(
    prompt="Your prompt here",
    max_length=100,          # Total length of generated text
    temperature=0.8,         # Sampling temperature (0.0 = greedy)
    top_k=50,               # Top-k sampling
    top_p=0.9,              # Top-p (nucleus) sampling
    do_sample=True          # Enable sampling
)
```

### 2. Token-by-Token Generation (`generate_tokens`)

More control over the generation process:

```python
generated = model.generate_tokens(
    prompt="Your prompt here",
    max_new_tokens=50,       # Number of new tokens to generate
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

### 3. Next Token Probabilities

Get probabilities for the next token:

```python
probs = model.get_next_token_probabilities("Your prompt", top_k=5)
for token, prob in probs:
    print(f"Token: '{token}', Probability: {prob:.4f}")
```

## Generation Parameters

- **`temperature`**: Controls randomness (higher = more random, 0.0 = greedy)
- **`top_k`**: Limits sampling to top-k most likely tokens
- **`top_p`**: Limits sampling to tokens with cumulative probability <= top_p
- **`max_length`**: Total length of generated sequence
- **`max_new_tokens`**: Number of new tokens to generate

## Checkpoint Management

### List Available Checkpoints

```python
from checkpoint_manager import CheckpointManager

manager = CheckpointManager()
checkpoints = manager.list_checkpoints()

for ckpt in checkpoints:
    print(f"{ckpt['filename']} - Step: {ckpt['step']}, Epoch: {ckpt['epoch']}")
```

### Load Specific Checkpoint

```python
latest = manager.get_latest_checkpoint()
model = manager.load_model_from_checkpoint(latest)
```

### Compare Checkpoints

```python
manager.compare_checkpoints("checkpoints/model1.pt", "checkpoints/model2.pt")
```

## Example Usage

Run the example script:

```bash
python example_usage.py
```

This will:

1. Load the model
2. Generate text with different parameters
3. Show next token probabilities
4. Provide an interactive mode for testing

## Advanced Usage

### Custom Model Configuration

If you need to use a different model configuration:

```python
from config import ModelConfig
from model.transformer import TinyLLM

# Create custom config
config = ModelConfig(
    vocab_size=50257,
    d_model=256,
    n_layers=6,
    n_heads=8,
    max_seq_len=1024
)

# Load model with custom config
model = TinyLLM(config)
checkpoint = torch.load("path/to/checkpoint.pt")
model.load_state_dict(checkpoint["model"])
model.eval()
```

### Batch Generation

For generating multiple prompts efficiently:

```python
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
results = []

for prompt in prompts:
    generated = model.generate(prompt, max_length=50)
    results.append(generated)
```

## Troubleshooting

### CUDA Memory Issues

If you encounter CUDA memory errors:

```python
# Use CPU instead
model = ModelInference(device="cpu")

# Or reduce model size by modifying config
config = ModelConfig(d_model=128, n_layers=3)  # Smaller model
```

### Loading Errors

If checkpoints fail to load:

1. Ensure the checkpoint file exists and is not corrupted
2. Check that the model configuration matches the saved model
3. Verify device compatibility (CPU/GPU)

### Generation Quality

To improve generation quality:

1. **Adjust temperature**: Start with 0.7-0.9 for creativity
2. **Use top-k/top-p**: top_k=50, top_p=0.9 are good defaults
3. **Increase context**: Use longer prompts for better conditioning
4. **Fine-tune**: Train for more epochs or with more data

## Performance Tips

1. **Use GPU**: If available, use CUDA for faster inference
2. **Batch processing**: Process multiple prompts together when possible
3. **Model size**: Larger models generally produce better results but are slower
4. **Sequence length**: Keep prompts within the model's max_seq_len

## File Structure

```bash
.
├── inference.py              # Main inference class
├── example_usage.py          # Usage examples
├── checkpoint_manager.py     # Checkpoint utilities
├── config.py                 # Model configuration
├── model/
│   ├── transformer.py        # Model architecture
│   ├── attention.py          # Attention mechanism
│   └── ...
└── checkpoints/              # Saved models
    └── last_model.pt         # Latest checkpoint
```
