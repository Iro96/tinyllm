from torch.utils.data import Dataset
import torch


class TokenDataset(Dataset):
    """
    Document-aware causal language modeling dataset with
    minimum sequence length and strided sampling.

    Guarantees that emitted sequences never exceed max_seq_len.
    """

    def __init__(self, documents, max_seq_len, min_seq_len=32, stride=None):
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.stride = stride or min_seq_len

        # Hard-cap documents defensively (in case upstream logic is wrong)
        self.documents = [
            doc[:max_seq_len] for doc in documents if len(doc) >= min_seq_len + 1
        ]

        self.samples = []
        for doc_idx, doc in enumerate(self.documents):
            max_start = len(doc) - min_seq_len - 1
            if max_start <= 0:
                continue

            for start_idx in range(0, max_start, self.stride):
                self.samples.append((doc_idx, start_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        doc_idx, start_idx = self.samples[idx]
        doc = self.documents[doc_idx]

        # Remaining tokens available for causal shift
        remaining = len(doc) - start_idx - 1
        seq_len = min(self.max_seq_len, remaining)

        x = doc[start_idx : start_idx + seq_len]
        y = doc[start_idx + 1 : start_idx + seq_len + 1]

        # Absolute safety cap (never exceed max_seq_len)
        assert len(x) <= self.max_seq_len

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
