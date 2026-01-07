from transformers import AutoTokenizer
from config import ModelConfig

TOKENIZER_NAME = "gpt2"
PAD_TOKEN = "<|pad|>"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # MUST be done before training
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    # Avoid warnings when tokenizing very long raw documents for chunking.
    # The model's context window is still determined by ModelConfig().max_seq_len;
    # we set tokenizer.model_max_length to a large value so tokenizer won't
    # warn when encoding long texts that we'll later split into chunks.
    tokenizer.model_max_length = max(ModelConfig().max_seq_len, 10_000_000)

    return tokenizer