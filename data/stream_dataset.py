import torch
from torch.utils.data import IterableDataset
from typing import Optional


class StreamingTokenDataset(IterableDataset):
    """
    IterableDataset that reads a tokenized document file (one doc per line,
    tokens as space-separated ints) and yields (x, y) samples using stride.

    This avoids loading all documents into memory.
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
        # doc_tokens: list[int]
        samples = []
        # Hard-cap to max_seq_len
        doc_tokens = doc_tokens[: self.max_seq_len]
        n = len(doc_tokens)
        if n < self.min_seq_len + 1:
            return samples

        max_start = n - self.min_seq_len - 1
        for start in range(0, max_start + 1, self.stride):
            remaining = n - start - 1
            seq_len = min(self.max_seq_len, remaining)
            x = doc_tokens[start : start + seq_len]
            y = doc_tokens[start + 1 : start + seq_len + 1]
            samples.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))

        return samples

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
