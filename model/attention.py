import torch
import torch.nn as nn

from .rope import apply_rope, build_rope_cache


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=8192, sliding_window=2048, use_sliding_window=True):
        super().__init__()

        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self.register_buffer("causal_mask", None, persistent=False)

    def _build_sliding_window_mask(self, max_len, device):
        """Build sliding window attention mask with local window + global attention."""
        # Create base causal mask
        causal_mask = torch.tril(torch.ones(max_len, max_len, device=device)).bool()
        
        if not self.use_sliding_window:
            return causal_mask
            
        # Apply sliding window constraint
        window_mask = torch.zeros(max_len, max_len, device=device, dtype=torch.bool)
        
        for i in range(max_len):
            # Each position can attend to:
            # 1. Previous positions within the sliding window
            start = max(0, i - self.sliding_window + 1)
            window_mask[i, start:i+1] = True
            
            # 2. Global attention to key positions (every N tokens)
            # For simplicity, allow attention to positions that are multiples of sliding_window/4
            global_step = self.sliding_window // 4
            for j in range(0, i+1, global_step):
                window_mask[i, j] = True
        
        # Combine causal and sliding window masks
        return causal_mask & window_mask

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: Tensor of shape (B, T, C).
            padding_mask: Optional boolean tensor of shape (B, T).
                ``True`` marks valid tokens and ``False`` marks padding.
        """
        batch_size, seq_len, channels = x.size()
        device = x.device

        if self.cos is None or self.cos.device != device:
            self.cos, self.sin = build_rope_cache(
                self.max_seq_len,
                self.head_dim,
                device,
            )

        if self.causal_mask is None or self.causal_mask.device != device:
            if self.use_sliding_window:
                # Create sliding window causal mask
                self.causal_mask = self._build_sliding_window_mask(self.max_seq_len, device)
            else:
                self.causal_mask = torch.tril(
                    torch.ones(self.max_seq_len, self.max_seq_len, device=device)
                ).bool()

        causal_mask = self.causal_mask[:seq_len, :seq_len]

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            for tensor in qkv
        ]

        q = apply_rope(q, self.cos[:seq_len], self.sin[:seq_len])
        k = apply_rope(k, self.cos[:seq_len], self.sin[:seq_len])

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )

        if padding_mask is not None:
            key_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            query_padding_mask = padding_mask.unsqueeze(1).unsqueeze(-1)
            attn = attn.masked_fill(~key_padding_mask, float("-inf"))
            attn = attn.masked_fill(~query_padding_mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)

        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1)

        return self.out(out)
