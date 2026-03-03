import json
import struct

# Minimal BPE encoder for TinyStories
RAW_TEXT_PATH = "/Users/andy.huang/lab/research/ANE/training/tinystories_raw.txt"
VOCAB_PATH = "/Users/andy.huang/lab/research/ANE/training/vocab.json"
OUTPUT_PATH = "/Users/andy.huang/lab/research/ANE/training/tinystories_data00.bin"

def encode():
    print(f"Loading vocab from {VOCAB_PATH}...")
    with open(VOCAB_PATH, "r") as f:
        data = json.load(f)
        merges = {tuple(map(int, k.split(","))): idx for k, idx in data["merges"].items()}

    print(f"Loading raw text (truncated for test) from {RAW_TEXT_PATH}...")
    with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read(500000) # 500KB

    ids = list(text.encode("utf-8"))
    
    print("Applying BPE merges...")
    # Apply merges in order
    for pair, idx in merges.items():
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids

    print(f"Saving {len(ids)} tokens to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "wb") as f:
        for idx in ids:
            f.write(struct.pack("<H", idx)) # uint16 little-endian

    print("Done.")

if __name__ == "__main__":
    encode()
