#!/usr/bin/env python3
import os
import json
import struct
import argparse
from sample import BPETokenizer

def main():
    parser = argparse.ArgumentParser(description="Tokenize any text file for ANE training")
    parser.add_argument("input", type=str, help="Input text file")
    parser.add_argument("--output", type=str, default="data.bin", help="Output binary file")
    parser.add_argument("--vocab", type=str, default="vocab.json", help="Path to vocab.json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        return

    print(f"Loading tokenizer from {args.vocab}...")
    tokenizer = BPETokenizer(args.vocab)

    print(f"Reading {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()

    print("Tokenizing...")
    # Add BOS token (1) at the start
    tokens = [1] + tokenizer.encode(text)
    
    print(f"Saving {len(tokens)} tokens to {args.output}...")
    with open(args.output, 'wb') as f:
        for t in tokens:
            # The ANE trainer expects uint16_t
            f.write(struct.pack('H', t))

    print("Done.")

if __name__ == "__main__":
    main()
