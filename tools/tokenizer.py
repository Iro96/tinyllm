from transformers import AutoTokenizer
from config import ModelConfig

TOKENIZER_NAME = "gpt2"
PAD_TOKEN = "<|pad|>"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # MUST be done before training
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.model_max_length = ModelConfig().max_seq_len

    return tokenizer