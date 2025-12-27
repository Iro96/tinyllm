import torch
import torch.nn as nn
from .attention import SelfAttention
from .ffn import SwiGLU
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(
            config.d_model,
            config.n_heads,
            config.max_seq_len,
        )
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(
            config.d_model,
            config.d_model * config.ffn_mult,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, T, C)
            padding_mask: (B, T) boolean, True = valid token
        """
        attn_out = self.attn(self.norm1(x), padding_mask=padding_mask)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x


class TinyLLM(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        self.config.vocab_size = vocab_size

        self.embed = nn.Embedding(self.config.vocab_size, config.d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, self.config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

    def resize_token_embeddings(self, new_vocab_size: int):
        self.config.vocab_size = new_vocab_size

        new_embed = nn.Embedding(self.config.vocab_size, self.config.d_model)
        new_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        # Copy old weights
        n = min(self.embed.weight.shape[0], new_embed.weight.shape[0])
        new_embed.weight.data[:n] = self.embed.weight.data[:n]
        new_head.weight.data[:n] = self.head.weight.data[:n]

        self.embed = new_embed
        self.head = new_head

        # Weight tying
        self.head.weight = self.embed.weight

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, T)
            padding_mask: (B, T) boolean
        """
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=None,
        eos_token_id=None,
    ):
        from torch.nn import functional as F

        if pad_token_id is None:
            raise ValueError("pad_token_id must be provided")
        if eos_token_id is None:
            raise ValueError("eos_token_id must be provided")

        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        generated = torch.full(
            (batch_size, max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        generated[:, :seq_len] = input_ids
        finished_sequences = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )

        for cur_len in range(seq_len, max_length):
            # Respect model context window
            start_idx = max(0, cur_len - self.config.max_seq_len)
            input_slice = generated[:, start_idx:cur_len]

            padding_mask = input_slice != pad_token_id

            logits = self(
                input_slice,
                padding_mask=padding_mask,
            )
            next_token_logits = logits[:, -1, :] / temperature

            if do_sample:
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    next_token_logits = self._top_p_filter(next_token_logits, top_p)

                # Top-k filtering
                if top_k > 0:
                    next_token_logits = self._top_k_filter(next_token_logits, top_k)

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(
                    next_token_logits, dim=-1, keepdim=True
                )

            # Do not update finished sequences
            next_token[finished_sequences] = pad_token_id
            generated[:, cur_len] = next_token.squeeze(-1)

            # Mark finished sequences
            finished_sequences |= next_token.squeeze(-1) == eos_token_id

            # Stop if all sequences are finished
            if torch.all(finished_sequences):
                break

        return generated

    def _top_p_filter(self, logits, top_p):
        from torch.nn import functional as F

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cum_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")
        return logits

    def _top_k_filter(self, logits, top_k):
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
        return logits
