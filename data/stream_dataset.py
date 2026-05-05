import torch
from torch.utils.data import IterableDataset
from typing import Optional
from pathlib import Path


class StreamingTokenDataset(IterableDataset):
    """
    Stream tokenized documents from disk and yield causal LM samples with loss masks.

    Each line is one full document encoded as space-separated token IDs.
    Optionally loads corresponding loss masks (1=train, 0=ignore).
    """

    def __init__(
        self,
        path: str,
        max_seq_len: int,
        min_seq_len: int = 32,
        stride: Optional[int] = None,
    ):
        self.path = path
        self.mask_path = Path(str(path) + ".mask")
        self.has_masks = self.mask_path.exists()
        
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.stride = stride or min_seq_len

    def _doc_to_samples(self, doc_tokens, doc_masks=None):
        """Generate samples from a document with optional loss mask.
        
        Args:
            doc_tokens: list of token IDs
            doc_masks: list of loss mask values (0=ignore, 1=train) or None
        
        Yields:
            Tuple of (x, y, loss_mask) tensors
            x and y are shifted by 1 for causal LM
            loss_mask[i] indicates whether y[i] should contribute to loss
        """
        n = len(doc_tokens)
        if n < self.min_seq_len + 1:
            return

        max_start = n - self.min_seq_len - 1
        for start in range(0, max_start + 1, self.stride):
            remaining = n - start - 1
            seq_len = min(self.max_seq_len, remaining)
            
            x = doc_tokens[start : start + seq_len]
            y = doc_tokens[start + 1 : start + seq_len + 1]
            
            # Extract corresponding loss mask
            if doc_masks:
                # Mask is shifted by 1 to align with y prediction targets
                loss_mask = doc_masks[start + 1 : start + seq_len + 1]
            else:
                # No mask: train on all tokens
                loss_mask = [1] * len(y)
            
            yield (
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(loss_mask, dtype=torch.long),
            )

    def parse_line(self, line: str):
        """Parse space-separated integers from a line."""
        try:
            toks = [int(x) for x in line.strip().split() if x]
            return toks
        except Exception:
            return []

    def __iter__(self):
        """Iterate documents and yield samples with loss masks."""
        if self.has_masks:
            # Load both tokens and masks in parallel
            with open(self.path, "r", encoding="utf-8") as token_file, \
                 open(self.mask_path, "r", encoding="utf-8") as mask_file:
                for token_line, mask_line in zip(token_file, mask_file):
                    doc_tokens = self.parse_line(token_line)
                    doc_masks = self.parse_line(mask_line)
                    
                    if not doc_tokens or not doc_masks:
                        continue
                    
                    # Sanity check
                    if len(doc_tokens) != len(doc_masks):
                        print(f"Warning: token/mask mismatch: {len(doc_tokens)} vs {len(doc_masks)}")
                        continue
                    
                    for sample in self._doc_to_samples(doc_tokens, doc_masks):
                        yield sample
        else:
            # Legacy mode: no masks available
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = self.parse_line(line)
                    if not doc:
                        continue
                    for sample in self._doc_to_samples(doc):
                        yield sample
