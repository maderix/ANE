#!/usr/bin/env python3
"""Run Qwen2.5-0.5B on ANE with proper tokenization.

Usage:
    python3 run.py "Your prompt here" [--max-tokens 50]
"""
import argparse
import ctypes
import struct
import sys
import time
from pathlib import Path

INFERENCE_DIR = Path(__file__).parent
WEIGHTS_PATH = INFERENCE_DIR / "qwen05b.bin"
MODEL_DIR = Path.home() / "models" / "Qwen2.5-0.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    # Build chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": args.prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(text)
    print(f"Prompt tokens: {len(input_ids)}")

    # Run the C binary — pass token IDs as arguments
    import subprocess
    binary = str(INFERENCE_DIR / "qwen_ane")

    # We need to modify the binary to accept token IDs as input
    # For now, print the token IDs so we can verify tokenization
    print(f"First 10 tokens: {input_ids[:10]}")
    print(f"Token text: {[tok.decode([t]) for t in input_ids[:10]]}")
    print(f"\nRunning ANE inference with {len(input_ids)} prompt tokens + {args.max_tokens} generation...")

    # Call binary with token IDs piped via stdin
    result = subprocess.run(
        [binary, str(WEIGHTS_PATH), " ".join(str(t) for t in input_ids),
         str(args.max_tokens)],
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr[:500], file=sys.stderr)

    # Parse output token IDs from binary stdout
    output_ids = []
    for line in result.stdout.split("\n"):
        if line.startswith("OUT:"):
            ids = [int(x) for x in line[4:].split() if x.isdigit()]
            output_ids.extend(ids)

    if output_ids:
        decoded = tok.decode(output_ids, skip_special_tokens=True)
        print(f"\n=== Response ===\n{decoded}")
    else:
        print("\n(No output tokens parsed — binary may need token ID input mode)")


if __name__ == "__main__":
    main()
