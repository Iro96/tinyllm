#!/usr/bin/env python3
"""Example script for chatting with Terry."""

from tools.inference import ModelInference


def main():
    print("=== Terry Inference Example ===\n")
    model = ModelInference(checkpoint_path="checkpoints/last_model.pt", device="auto")

    prompts = [
        "hi terry",
        "what do you think about rain on the window",
        "help me make a tiny plan for today",
        "tell me a small story",
    ]

    for index, prompt in enumerate(prompts, start=1):
        print(f"\n--- Prompt {index}: {prompt} ---")
        reply = model.generate(prompt, max_length=40, temperature=0.8)
        print(f"Terry: {reply}")

    print("\nEnter prompts to chat with Terry. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        try:
            reply = model.generate(user_input, max_length=60, temperature=0.8)
            print(f"Terry: {reply}")
        except Exception as exc:
            print(f"Error during generation: {exc}")


if __name__ == "__main__":
    main()
