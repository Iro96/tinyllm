#!/usr/bin/env python3
"""
Test script for the TinyLLM inference system.
This script tests basic functionality to ensure everything works correctly.
"""

import torch
from tools.inference import ModelInference
from tools.checkpoint_manager import CheckpointManager

def test_model_loading():
    """Test loading the model from checkpoint."""
    print("=== Testing Model Loading ===")
    
    try:
        # Test loading from default checkpoint
        model = ModelInference()
        print("[STATUS] Model loaded successfully")
        
        # Check device
        device = next(model.model.parameters()).device
        print(f"[STATUS] Model is on device: {device}")
        
        # Check model is in eval mode
        assert not model.model.training, "Model should be in eval mode"
        print("[STATUS] Model is in eval mode")
        
        return model
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None

def test_tokenization():
    """Test tokenizer functionality."""
    print("\n=== Testing Tokenization ===")
    
    try:
        model = ModelInference()
        
        # Test encoding
        prompt = "Hello world"
        tokens = model.tokenizer.encode(prompt, return_tensors="pt")
        print(f"✓ Tokenized '{prompt}' -> {tokens.shape}")
        
        # Test decoding
        decoded = model.tokenizer.decode(tokens[0])
        print(f"✓ Decoded back: '{decoded}'")
        
        return True
    except Exception as e:
        print(f"✗ Tokenization test failed: {e}")
        return False

def test_generation():
    """Test text generation."""
    print("\n=== Testing Generation ===")
    
    try:
        model = ModelInference()
        
        # Test short generation
        prompt = "The weather is"
        generated = model.generate(prompt, max_length=20, temperature=0.0, do_sample=False)
        print(f"✓ Greedy generation: '{generated}'")
        
        # Test sampling generation
        generated_sample = model.generate(prompt, max_length=20, temperature=0.8, do_sample=True)
        print(f"✓ Sampling generation: '{generated_sample}'")
        
        # Test token-by-token generation
        generated_tokens = model.generate_tokens(prompt, max_new_tokens=10, temperature=0.8)
        print(f"✓ Token-by-token generation: '{generated_tokens}'")
        
        return True
    except Exception as e:
        print(f"✗ Generation test failed: {e}")
        return False

def test_next_token_probabilities():
    """Test next token probability calculation."""
    print("\n=== Testing Next Token Probabilities ===")
    
    try:
        model = ModelInference()
        
        prompt = "The capital of France is"
        probs = model.get_next_token_probabilities(prompt, top_k=5)
        
        print(f"✓ Next token probabilities for '{prompt}':")
        for token, prob in probs:
            print(f"  '{token}': {prob:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Next token probabilities test failed: {e}")
        return False

def test_checkpoint_manager():
    """Test checkpoint management."""
    print("\n=== Testing Checkpoint Manager ===")
    
    try:
        manager = CheckpointManager()
        
        # List checkpoints
        checkpoints = manager.list_checkpoints()
        print(f"[STATUS] Found {len(checkpoints)} checkpoint(s)")
        
        if checkpoints:
            # Get latest checkpoint
            latest = manager.get_latest_checkpoint()
            print(f"[STATUS] Latest checkpoint: {latest}")
            
            # Load model from checkpoint
            model = manager.load_model_from_checkpoint(latest)
            print("[STATUS] Model loaded from checkpoint manager")
        
        return True
    except Exception as e:
        print(f"[ERROR] Checkpoint manager test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage and device placement."""
    print("\n=== Testing Memory Usage ===")
    
    try:
        model = ModelInference()
        device = next(model.model.parameters()).device
        
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
            print(f"[STATUS] GPU memory allocated: {memory_allocated:.2f} MB")
        else:
            print("[STATUS] Using CPU (no GPU memory tracking)")
        
        # Test generation memory usage
        prompt = "A" * 100  # Long prompt
        generated = model.generate(prompt, max_length=50, temperature=0.8)
        print(f"[STATUS] Generated from long prompt successfully")
        
        return True
    except Exception as e:
        print(f"[ERROR] Memory usage test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("TinyLLM Inference Test Suite")
    print("=" * 40)
    
    tests = [
        test_model_loading,
        test_tokenization,
        test_generation,
        test_next_token_probabilities,
        test_checkpoint_manager,
        test_memory_usage
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("[COMPLETE] All tests passed!")
    else:
        print(f"[FAILED] {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)