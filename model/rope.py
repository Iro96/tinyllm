import torch


def build_rope_cache(seq_len, head_dim, device):
    """
    Build RoPE cache for rotary embeddings.
    
    Args:
        seq_len: Maximum sequence length
        head_dim: Dimension of each attention head
        device: Device to place the cache on
    
    Returns:
        Tuple of (cos, sin) tensors
    """
    # Standard RoPE formula: theta_i = 10000^(-2i/head_dim)
    # This is equivalent to: inv_freq = 1 / (10000^(2i/head_dim))
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device).float() * 2 / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """
    Apply RoPE to input tensor.

    Args:
        x: (B, H, T, D)
        cos: (T, D//2)
        sin: (T, D//2)
    """
    # Expand for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out