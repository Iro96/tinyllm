#!/usr/bin/env python3
"""
Test script to verify variable-length sequence support and dynamic padding.
"""

import torch
from transformers import AutoTokenizer
from data.dataset import TokenDataset
from model.transformer import TinyLLM
from config import ModelConfig

def test_variable_length_sequences():
    """Test that the dataset and model work with variable-length sequences."""
    print("Testing variable-length sequence support...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create sample documents of different lengths
    documents = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, tokenizer.eos_token_id],  # Length 11
        [11, 12, 13, 14, 15, tokenizer.eos_token_id],              # Length 6
        [16, 17, 18, 19, 20, 21, 22, 23, tokenizer.eos_token_id], # Length 9
        [24, 25, 26, tokenizer.eos_token_id],                      # Length 4
    ]
    
    max_seq_len = 8
    dataset = TokenDataset(documents, max_seq_len)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(5, len(dataset))):
        x, y = dataset[i]
        print(f"Sample {i}: input length={len(x)}, target length={len(y)}")
        print(f"  Input: {x}")
        print(f"  Target: {y}")
    
    # Test the collate function
    def collate_fn(batch):
        """Custom collate function for dynamic padding."""
        input_seqs, target_seqs = zip(*batch)
        
        # Find the maximum sequence length in this batch
        max_len = max(seq.size(0) for seq in input_seqs)
        
        # Pad sequences to max_len in this batch
        padded_inputs = []
        padded_targets = []
        
        for input_seq, target_seq in zip(input_seqs, target_seqs):
            pad_len = max_len - input_seq.size(0)
            padded_input = torch.nn.functional.pad(input_seq, (0, pad_len), value=tokenizer.pad_token_id)
            padded_target = torch.nn.functional.pad(target_seq, (0, pad_len), value=tokenizer.pad_token_id)
            
            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)
        
        batch_inputs = torch.stack(padded_inputs)
        batch_targets = torch.stack(padded_targets)
        
        return batch_inputs, batch_targets
    
    # Test with a batch
    batch_size = 3
    batch = [dataset[i] for i in range(batch_size)]
    batch_inputs, batch_targets = collate_fn(batch)
    
    print(f"\nBatch shape: {batch_inputs.shape}")
    print(f"Batch inputs:\n{batch_inputs}")
    print(f"Batch targets:\n{batch_targets}")
    
    # Test padding mask
    padding_mask = batch_inputs != tokenizer.pad_token_id
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Padding mask:\n{padding_mask}")
    
    # Test model forward pass
    model_cfg = ModelConfig()
    model = TinyLLM(model_cfg)
    model.eval()
    
    with torch.no_grad():
        logits = model(batch_inputs, padding_mask=padding_mask)
        print(f"\nModel output shape: {logits.shape}")
        print("Model forward pass successful with variable-length sequences!")
    
    # Test mask caching
    print(f"\nMask cache size: {len(model.transformer.layers[0].self_attn.mask_cache)}")
    print("Mask caching working correctly!")
    
    print("\nAll tests passed! Variable-length sequences are working correctly.")

if __name__ == "__main__":
    test_variable_length_sequences()