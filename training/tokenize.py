#!/usr/bin/env python3
"""Extract pretokenized TinyStories data from zip.
Data format: flat uint16 token IDs (llama2.c BPE, 32K vocab).
Source: ~/tiny_stories_data_pretokenized.zip"""

import os, sys, struct, zipfile
from pathlib import Path

ZIP_PATH = os.path.expanduser('~/tiny_stories_data_pretokenized.zip')
OUTPUT_PATH = str(Path(__file__).resolve().parent / 'tinystories_data00.bin')
VOCAB_SIZE = 32000

# Configurable upper size limit for the source ZIP (default 10 GB).
# Override via environment variable: MAX_ZIP_BYTES=<bytes> python3 tokenize.py
MAX_ZIP_SIZE = int(os.environ.get('MAX_ZIP_BYTES', str(10 * 1024 * 1024 * 1024)))


def main():
    # --- Input validation (ref: docs/reports/security-audit-2026-03-02.md LOW-03) ---

    # 1. Validate ZIP exists before attempting to open
    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: ZIP-Datei nicht gefunden: {ZIP_PATH}", file=sys.stderr)
        print(f"  Erwartet: ~/tiny_stories_data_pretokenized.zip", file=sys.stderr)
        sys.exit(1)

    # 2. Validate ZIP size (guard against unexpectedly large or corrupted files)
    zip_size = os.path.getsize(ZIP_PATH)
    if zip_size > MAX_ZIP_SIZE:
        print(f"ERROR: ZIP-Datei zu gross ({zip_size/1e9:.1f} GB > {MAX_ZIP_SIZE/1e9:.0f} GB Limit).",
              file=sys.stderr)
        print(f"  Setze MAX_ZIP_BYTES=<bytes> um das Limit anzupassen.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(OUTPUT_PATH):
        n = os.path.getsize(OUTPUT_PATH) // 2
        print(f"{OUTPUT_PATH} already exists ({n} tokens, {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")
        return

    # 3. Validate ZIP contains the expected entry before extracting
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        if 'data00.bin' not in z.namelist():
            print(f"ERROR: 'data00.bin' nicht in ZIP gefunden.", file=sys.stderr)
            print(f"  ZIP-Inhalt: {z.namelist()}", file=sys.stderr)
            sys.exit(1)

        print(f"Extracting data00.bin from {ZIP_PATH}...")
        with z.open('data00.bin') as src, open(OUTPUT_PATH, 'wb') as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)

    n = os.path.getsize(OUTPUT_PATH) // 2
    print(f"Written {OUTPUT_PATH} ({n} tokens, {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")

    # 4+5. Sanity check: validate output size and token ID range
    with open(OUTPUT_PATH, 'rb') as f:
        raw = f.read(20)
        if len(raw) < 20:
            print(f"WARNING: Output-Datei sehr klein ({len(raw)} Bytes) — moeglicherweise leer.",
                  file=sys.stderr)
        else:
            tokens = struct.unpack('<10H', raw)
            invalid = [t for t in tokens if t >= VOCAB_SIZE]
            if invalid:
                print(f"WARNING: Ungueltige Token-IDs in ersten 10 Tokens: {invalid} "
                      f"(erwartet < {VOCAB_SIZE})", file=sys.stderr)
            else:
                print(f"First 10 tokens: {tokens} (alle < {VOCAB_SIZE} OK)")


if __name__ == '__main__':
    main()
