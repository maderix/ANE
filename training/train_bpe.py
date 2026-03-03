import os
import json
from collections import Counter

# Minimal BPE trainer for TinyStories
RAW_TEXT_PATH = "/Users/andy.huang/lab/research/ANE/training/tinystories_raw.txt"
VOCAB_PATH = "/Users/andy.huang/lab/research/ANE/training/vocab.json"
VOCAB_SIZE = 5000 # Reduced for speed of verification
SUBSET_SIZE = 200000 # 200KB limit for speed

def get_stats(ids):
    counts = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def train():
    print(f"Loading raw text (subset {SUBSET_SIZE} bytes) from {RAW_TEXT_PATH}...")
    with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read(SUBSET_SIZE)

    print("Initial byte-encoding...")
    # Start with raw bytes (0-255)
    ids = list(text.encode("utf-8"))
    
    merges = {}
    vocab = {i: bytes([i]) for i in range(256)}
    
    num_merges = VOCAB_SIZE - 256
    print(f"Training BPE for {num_merges} merges...")
    
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        if (i+1) % 100 == 0:
            print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} (count {stats[pair]})")

    # Save merges and vocab
    # We need to convert tuple keys to strings for JSON
    serializable_merges = {f"{p[0]},{p[1]}": idx for p, idx in merges.items()}
    # Convert vocab bytes to list of ints for JSON
    serializable_vocab = {idx: list(b) for idx, b in vocab.items()}
    
    with open(VOCAB_PATH, "w") as f:
        json.dump({
            "merges": serializable_merges,
            "vocab": serializable_vocab
        }, f)
    
    print(f"Vocab saved to {VOCAB_PATH}")

if __name__ == "__main__":
    train()
