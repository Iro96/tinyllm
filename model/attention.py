import torch
import torch.nn as nn
from .rope import apply_rope, build_rope_cache


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=2048):
        super().__init__()

        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self.register_buffer("causal_mask", None, persistent=False)

    def forward(self, x, padding_mask=None):
        """
        x: (B, T, C)
        padding_mask: (B, T)  True = valid token
        """
        B, T, C = x.size()
        device = x.device

        # ---- RoPE cache
        if self.cos is None or self.cos.device != device:
            self.cos, self.sin = build_rope_cache(
                self.max_seq_len, self.head_dim, device
            )

        # ---- causal mask cache
        if self.causal_mask is None or self.causal_mask.device != device:
            self.causal_mask = torch.tril(
                torch.ones(self.max_seq_len, self.max_seq_len, device=device)
            ).bool()

        causal_mask = self.causal_mask[:T, :T]  # (T, T)

        # ---- QKV
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            for t in qkv
        ]

        # ---- RoPE
        q = apply_rope(q, self.cos[:T], self.sin[:T])
        k = apply_rope(k, self.cos[:T], self.sin[:T])

        # ---- attention scores
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn: (B, H, T, T)

        # ---- causal mask (mask future)
        attn = attn.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )

        if padding_mask is not None:
            # ---- KEY padding mask (mask attend-to PAD)
            key_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            attn = attn.masked_fill(~key_padding_mask, float("-inf"))

            # ---- QUERY padding mask (mask PAD queries)
            query_padding_mask = padding_mask.unsqueeze(1).unsqueeze(-1)  # (B,1,T,1)
            attn = attn.masked_fill(~query_padding_mask, float("-inf"))

        # ---- softmax
        attn = torch.softmax(attn, dim=-1)

        # ---- replace NaN (if any) with zero
        attn = torch.nan_to_num(attn, nan=0.0)

        # ---- weighted sum
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # ---- zero-out PAD outputs (an toàn cuối)
        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1)

        return self.out(out)
