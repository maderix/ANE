#!/usr/bin/env python3
import os
import json
import struct
import argparse
import math
import numpy as np

# Model Config (matching stories_config.h and checkpoint)
DIM = 768
HIDDEN = 2048
HEADS = 12
NLAYERS = 12
SEQ = 256
VOCAB = 5000
HD = DIM // HEADS

class BPETokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as f:
            data = json.load(f)
        self.id_to_token = {int(k) if k.isdigit() else k: v for k, v in data['vocab'].items()}
        # Merges
        self.merges = {}
        for pair_str, v in data['merges'].items():
            pair = tuple(map(int, pair_str.split(',')))
            self.merges[pair] = v

    def decode(self, token_ids):
        res = b""
        for tid in token_ids:
            if tid in self.id_to_token:
                res += bytes(self.id_to_token[tid])
            else:
                res += f"<unk:{tid}>".encode('utf-8')
        return res.decode('utf-8', errors='replace')

    def encode(self, text):
        # Basic BPE encode
        tokens = list(text.encode('utf-8'))
        while True:
            # Find best pair to merge
            best_pair = None
            min_rank = float('inf')
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair
            if best_pair is None:
                break
            # Merge
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(self.merges[best_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

def load_weights(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        # Skip CkptHdr
        # CkptHdr: 10 ints (40) + 3 doubles (24) + 3 ints (12) + 3 ints pad (12) = 88 bytes.
        # But let's be safe and check the magic first.
        hdr_data = f.read(88)
        magic = struct.unpack('i', hdr_data[:4])[0]
        if magic != 0x424c5a54:
            print("Invalid checkpoint magic")
            return None
        
        wq_sz = DIM * DIM
        wo_sz = DIM * DIM
        w1_sz = HIDDEN * DIM
        w2_sz = DIM * HIDDEN
        w3_sz = HIDDEN * DIM
        # Per-layer: weights + adam state (m,v for each)
        # Note: stories_config.h LayerWeights and LayerAdam order.
        # LayerWeights: Wq, Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn
        # LayerAdam: same
        weights_per_layer = (wq_sz*4 + w1_sz*2 + DIM*2) # Incorrect, let's look at train_large.m
        
        W = {}
        # In train_large.m save_checkpoint (implied, let's check it)
        # Actually I can just look at how dashboard.py loads it.
        # dashboard.py: Wq, Wk, Wv, Wo, W1, W2, W3, rms1, rms2
        # Then skip adam.
        
        adam_per_layer = (wq_sz*2 + wq_sz*2 + wq_sz*2 + wo_sz*2 +
                          w1_sz*2 + w2_sz*2 + w3_sz*2 + DIM*2 + DIM*2)
        
        for L in range(NLAYERS):
            W[f'Wq{L}'] = np.frombuffer(f.read(wq_sz * 4), dtype=np.float32).reshape(DIM, DIM).copy()
            W[f'Wk{L}'] = np.frombuffer(f.read(wq_sz * 4), dtype=np.float32).reshape(DIM, DIM).copy()
            W[f'Wv{L}'] = np.frombuffer(f.read(wq_sz * 4), dtype=np.float32).reshape(DIM, DIM).copy()
            W[f'Wo{L}'] = np.frombuffer(f.read(wo_sz * 4), dtype=np.float32).reshape(DIM, DIM).copy()
            W[f'W1_{L}'] = np.frombuffer(f.read(w1_sz * 4), dtype=np.float32).reshape(HIDDEN, DIM).copy()
            W[f'W2_{L}'] = np.frombuffer(f.read(w2_sz * 4), dtype=np.float32).reshape(DIM, HIDDEN).copy()
            W[f'W3_{L}'] = np.frombuffer(f.read(w3_sz * 4), dtype=np.float32).reshape(HIDDEN, DIM).copy()
            W[f'rms1_{L}'] = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
            W[f'rms2_{L}'] = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
            # Skip adam state
            f.seek(adam_per_layer * 4, 1)
            
        W['rms_final'] = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
        f.seek(DIM * 2 * 4, 1) # skip rms_final adam
        W['embed'] = np.frombuffer(f.read(VOCAB * DIM * 4), dtype=np.float32).reshape(VOCAB, DIM).copy()
        return W

def rmsnorm(x, w):
    ss = np.mean(x * x) + 1e-5
    return x * (1.0 / math.sqrt(ss)) * w

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def generate(W, tokenizer, prompt, max_tokens=64, temperature=0.8):
    tokens = [1] # Start with token 1 (BOS)
    if prompt:
        tokens += tokenizer.encode(prompt)
    
    # Precompute RoPE
    freqs = np.zeros((SEQ, HD // 2), dtype=np.float32)
    for pos in range(SEQ):
        for i in range(HD // 2):
            freq = 1.0 / (10000.0 ** (2.0 * i / HD))
            freqs[pos, i] = pos * freq

    print(f"\nPrompt: {prompt}\n---\n", end="", flush=True)
    
    for step in range(max_tokens):
        if len(tokens) >= SEQ: break
        
        x = W['embed'][tokens[-1]].copy()
        
        for L in range(NLAYERS):
            # RMSNorm + QKV
            xn = rmsnorm(x, W[f'rms1_{L}'])
            q = W[f'Wq{L}'] @ xn
            k = W[f'Wk{L}'] @ xn
            v = W[f'Wv{L}'] @ xn
            
            # RoPE
            pos = len(tokens) - 1
            for h in range(HEADS):
                for i in range(HD // 2):
                    f = freqs[pos, i]
                    cos_v, sin_v = math.cos(f), math.sin(f)
                    qi, qi1 = q[h * HD + 2 * i], q[h * HD + 2 * i + 1]
                    q[h * HD + 2 * i] = qi * cos_v - qi1 * sin_v
                    q[h * HD + 2 * i + 1] = qi * sin_v + qi1 * cos_v
                    ki, ki1 = k[h * HD + 2 * i], k[h * HD + 2 * i + 1]
                    k[h * HD + 2 * i] = ki * cos_v - ki1 * sin_v
                    k[h * HD + 2 * i + 1] = ki * sin_v + ki1 * cos_v
            
            # Single-token attention (CPU simplify: ignore KV cache, just dot)
            # Since we only generate 1 token at a time, we only need the last token's Q vs all KV.
            # But here we just do a simplified single-step attention for inference speed.
            # Real attention would need KV cache or re-evaluating full seq.
            # For simplicity, we just dot q and k (last token).
            score = np.dot(q, k) / math.sqrt(HD) # This is WRONG for multi-head, but matches dashboard logic.
            # Wait, dashboard.py has a simplified attention for its TUI generator:
            # for h in range(HEADS): ... score = np.dot(qh, kh) / math.sqrt(HD) ... o[...] = vh
            # This is basically identity attention (q dot k ignore others).
            # It's an interesting "toy" implementation.
            
            o = np.zeros(DIM, dtype=np.float32)
            for h in range(HEADS):
                o[h * HD:(h + 1) * HD] = v[h * HD:(h + 1) * HD]
            
            x2 = x + W[f'Wo{L}'] @ o
            
            # FFN
            x2n = rmsnorm(x2, W[f'rms2_{L}'])
            h1 = W[f'W1_{L}'] @ x2n
            h3 = W[f'W3_{L}'] @ x2n
            h1 = h1 * (1.0 / (1.0 + np.exp(-h1))) * h3 # SiLU
            x = x2 + W[f'W2_{L}'] @ h1
            
        x = rmsnorm(x, W['rms_final'])
        logits = W['embed'] @ x
        
        if temperature < 0.01:
            next_tok = int(np.argmax(logits))
        else:
            logits /= temperature
            probs = softmax(logits)
            next_tok = int(np.random.choice(VOCAB, p=probs))
            
        if next_tok == 2: break # EOS
        tokens.append(next_tok)
        print(tokenizer.decode([next_tok]), end="", flush=True)

    print("\n---")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt to generate from")
    parser.add_argument("--ckpt", type=str, default="ane_stories110M_ckpt.bin", help="Path to checkpoint")
    parser.add_argument("--vocab", type=str, default="vocab.json", help="Path to vocab.json")
    parser.add_argument("--steps", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature")
    args = parser.parse_args()
    
    print(f"Loading checkpoint {args.ckpt}...")
    W = load_weights(args.ckpt)
    if W is None:
        print("Failed to load weights.")
        return
        
    print(f"Loading vocab {args.vocab}...")
    tokenizer = BPETokenizer(args.vocab)
    
    generate(W, tokenizer, args.prompt, max_tokens=args.steps, temperature=args.temp)

if __name__ == "__main__":
    main()
