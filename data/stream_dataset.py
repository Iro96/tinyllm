import torch
from torch.utils.data import IterableDataset
from typing import Optional


class StreamingTokenDataset(IterableDataset):
    """
    Stream tokenized documents from disk and yield causal LM samples.

    Each line is one full document encoded as space-separated token IDs.
    """

    def __init__(
        self,
        path: str,
        max_seq_len: int,
        min_seq_len: int = 32,
        stride: Optional[int] = None,
    ):
        self.path = path
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.stride = stride or min_seq_len

    def _doc_to_samples(self, doc_tokens):
        n = len(doc_tokens)
        if n < self.min_seq_len + 1:
            return

        max_start = n - self.min_seq_len - 1
        for start in range(0, max_start + 1, self.stride):
            remaining = n - start - 1
            seq_len = min(self.max_seq_len, remaining)
            x = doc_tokens[start : start + seq_len]
            y = doc_tokens[start + 1 : start + seq_len + 1]
            yield (
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
            )

    def parse_line(self, line: str):
        # Expect space-separated ints
        try:
            toks = [int(x) for x in line.strip().split() if x]
            return toks
        except Exception:
            return []

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                doc = self.parse_line(line)
                if not doc:
                    continue
                for sample in self._doc_to_samples(doc):
                    yield sample
