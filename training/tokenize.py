#!/usr/bin/env python3
"""Extract pretokenized TinyStories data from zip.
Data format: flat uint16 token IDs (llama2.c BPE, 32K vocab).
Source: ~/tiny_stories_data_pretokenized.zip"""

import os, struct, zipfile
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for zip in local directory
ZIP_PATH = os.path.join(BASE_DIR, 'tiny_stories_data_pretokenized.zip')
if not os.path.exists(ZIP_PATH):
    # Fallback to local name if not in base dir
    ZIP_PATH = 'tiny_stories_data_pretokenized.zip'
OUTPUT_PATH = str(Path(__file__).resolve().parent / 'tinystories_data00.bin')

def main():
    if os.path.exists(OUTPUT_PATH):
        n = os.path.getsize(OUTPUT_PATH) // 2
        print(f"{OUTPUT_PATH} already exists ({n} tokens, {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")
        return

    print(f"Extracting data00.bin from {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open('data00.bin') as src, open(OUTPUT_PATH, 'wb') as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)

    n = os.path.getsize(OUTPUT_PATH) // 2
    print(f"Written {OUTPUT_PATH} ({n} tokens, {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")

    # Sanity check
    with open(OUTPUT_PATH, 'rb') as f:
        tokens = struct.unpack('<10H', f.read(20))
        print(f"First 10 tokens: {tokens}")

if __name__ == '__main__':
    main()
