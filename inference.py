import argparse
import os
import sys
import time
import torch

import config
from src.gpt_config import GPTConfig
from src.model import GPT


def parse_args():
    parser = argparse.ArgumentParser(description="GPT Interactive Inference")
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='checkpoint.pth',
        help='Path to the model checkpoint file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps' if torch.mps.is_available() else 'cpu',
        help='Device to run the model on (e.g., "cpu", "cuda"). Default is auto-detect.'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=256,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--use_kv_cache',
        action='store_true',
        default=False,
        help='Enable KV cache for faster generation (if supported by the model)'
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str = None) -> GPT:
    if device is None:
        device = "cpu"
    
    print(f"Loading model on {device}...")

    model_config = GPTConfig(
        block_size=config.block_size,
        n_layer=config.n_layers,
        n_head=config.n_head,
        n_embd=config.n_embd,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
        ffn_hidden_size=config.n_embd * 4
    )
    
    model = GPT(model_config)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print("✓ Model weights loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    return model


def print_help():
    print("\n" + "="*50)
    print("Available commands :")
    print("  /help          - Show this help message")
    print("  /config        - Show model configuration")
    print("  /quit or /exit - Exit the program")
    print("="*50 + "\n")


def main():
    print("🚀 GPT Interactive Inference")
    print("Loading model...")

    arguments = parse_args()
    
    checkpoint_path = arguments.checkpoint_path
    device = arguments.device
    max_tokens = arguments.max_tokens
    temperature = 1.0
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint file '{checkpoint_path}' not found!")
        print("Please make sure the checkpoint.pth file is in the current directory.")
        sys.exit(1)
    
    try:
        model = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)

    print(f"✓ Model loaded successfully on {device}")
    print("Type '/help' for available commands or start typing to generate text!")
    print("-" * 50)
    
    try:
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    command_parts = user_input.split()
                    command = command_parts[0].lower()
                    
                    if command in ['/quit', '/exit']:
                        print("👋 Goodbye!")
                        break
                    elif command == '/config':
                        print(f"Model configuration:")
                        print(f"  - Layers: {model.config.n_layer}")
                        print(f"  - Heads: {model.config.n_head}")
                        print(f"  - Embedding size: {model.config.n_embd}")
                        print(f"  - Vocab size: {model.config.vocab_size}")
                        print(f"  - Block size: {model.config.block_size}")
                        print(f"  - Dropout: {model.config.dropout}")
                    else:
                        print(f"✗ Unknown command: {command}")
                        print("Type '/help' for available commands.")
                    
                    continue
                
                print(f"\n🤖 Generating...")
                print("-" * 30)

                start = time.time()
                
                generated_text = model.generate(
                    user_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_kv_cache=arguments.use_kv_cache
                )

                end = time.time()

                print(generated_text)
                print("-" * 30 + f"⏱️  Generated in {end - start:.2f} seconds")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            
            except Exception as e:
                print(f"✗ Unexpected error: {e}")
                print("Continuing...")
    
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()