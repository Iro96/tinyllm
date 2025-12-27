#!/usr/bin/env python3
"""
Example script demonstrating how to use the TinyLLM inference system.
This script shows different ways to load and use the trained model.
"""

from tools.inference import ModelInference, load_model

def main():
    print("=== TinyLLM Inference Example ===\n")
    
    # Method 1: Using the convenience function
    # print("1. Loading model using convenience function...")
    # model = load_model()
    
    # Method 2: Using the class directly
    print("1. Loading model manually...")
    model = ModelInference(checkpoint_path="checkpoints/last_model.pt", device="auto")
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "In the field of machine learning,",
        "The capital of France is"
    ]
    
    print("\n=== Generation Examples ===")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}: {prompt} ---")
        
        # Generate with different settings
        print("\nSampling generation (temperature=0.8):")
        generated = model.generate(prompt, max_length=60, temperature=0.8)
        print(f"Result: {generated}")
        
        print("\nGreedy generation (temperature=0.0):")
        generated_greedy = model.generate(prompt, max_length=60, temperature=0.0, do_sample=False)
        print(f"Result: {generated_greedy}")
        
        print("\nTop-k sampling (k=10):")
        generated_topk = model.generate(prompt, max_length=30, temperature=0.8, top_k=10, do_sample=True)
        print(f"Result: {generated_topk}")
    
    print("\n=== Next Token Probabilities ===")
    test_prompt = "The weather today is"
    print(f"\nAnalyzing next token probabilities for: '{test_prompt}'")
    
    probs = model.get_next_token_probabilities(test_prompt, top_k=5)
    for token, prob in probs:
        print(f"  Token: '{token}' | Probability: {prob:.4f}")
    
    print("\n=== Interactive Mode ===")
    print("Enter prompts to generate text (type 'quit' to exit):")
    
    while True:
        user_input = input("\nPrompt: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input:
            try:
                generated = model.generate(user_input, max_length=80, temperature=0.8)
                print(f"Generated: {generated}")
            except Exception as e:
                print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()