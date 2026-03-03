#!/usr/bin/env python3
"""Convert Qwen2.5-0.5B-Instruct safetensors → flat binary for ANE inference.

Output format: config header (7 ints) + all weights in f32, layer by layer.
Matches the layout expected by qwen_ane_infer.h.

Usage:
    python3 convert_weights.py /path/to/Qwen2.5-0.5B-Instruct /path/to/output.bin
"""

import struct
import sys
import numpy as np
from pathlib import Path
from safetensors import safe_open

def convert(model_dir: str, output_path: str):
    model_dir = Path(model_dir)

    # Load safetensors
    st_files = list(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"No safetensors files in {model_dir}")
        sys.exit(1)

    tensors = {}
    for f in st_files:
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                tensors[key] = sf.get_tensor(key).float().numpy()

    print(f"Loaded {len(tensors)} tensors from {len(st_files)} files")

    # Qwen2.5-0.5B config
    dim = 896
    hidden = 4864
    n_layers = 24
    n_heads = 14
    n_kv_heads = 2
    vocab_size = 151936
    max_seq = 512

    with open(output_path, "wb") as f:
        # Config header: 7 x int32
        f.write(struct.pack("iiiiiii",
            dim, hidden, n_layers, n_heads, n_kv_heads, vocab_size, max_seq))

        # Embedding [vocab, dim]
        emb = tensors["model.embed_tokens.weight"].astype(np.float32)
        print(f"embed: {emb.shape}")
        f.write(emb.tobytes())

        # Per-layer weights
        for l in range(n_layers):
            prefix = f"model.layers.{l}"

            # Attention norm
            rms_att = tensors[f"{prefix}.input_layernorm.weight"].astype(np.float32)
            f.write(rms_att.tobytes())

            # Q, K, V projections
            wq = tensors[f"{prefix}.self_attn.q_proj.weight"].astype(np.float32)
            wk = tensors[f"{prefix}.self_attn.k_proj.weight"].astype(np.float32)
            wv = tensors[f"{prefix}.self_attn.v_proj.weight"].astype(np.float32)
            wo = tensors[f"{prefix}.self_attn.o_proj.weight"].astype(np.float32)
            f.write(wq.tobytes())
            f.write(wk.tobytes())
            f.write(wv.tobytes())
            f.write(wo.tobytes())

            # Q/K biases (Qwen has them)
            # Q/K/V biases
            qb = tensors.get(f"{prefix}.self_attn.q_proj.bias")
            kb = tensors.get(f"{prefix}.self_attn.k_proj.bias")
            vb = tensors.get(f"{prefix}.self_attn.v_proj.bias")
            f.write((qb if qb is not None else np.zeros(wq.shape[0])).astype(np.float32).tobytes())
            f.write((kb if kb is not None else np.zeros(wk.shape[0])).astype(np.float32).tobytes())
            f.write((vb if vb is not None else np.zeros(wv.shape[0])).astype(np.float32).tobytes())

            # FFN norm
            rms_ffn = tensors[f"{prefix}.post_attention_layernorm.weight"].astype(np.float32)
            f.write(rms_ffn.tobytes())

            # FFN: gate, up, down
            w_gate = tensors[f"{prefix}.mlp.gate_proj.weight"].astype(np.float32)
            w_up = tensors[f"{prefix}.mlp.up_proj.weight"].astype(np.float32)
            w_down = tensors[f"{prefix}.mlp.down_proj.weight"].astype(np.float32)
            f.write(w_gate.tobytes())
            f.write(w_up.tobytes())
            f.write(w_down.tobytes())

            print(f"  Layer {l}: Q{wq.shape} K{wk.shape} V{wv.shape} O{wo.shape} "
                  f"gate{w_gate.shape} up{w_up.shape} down{w_down.shape}")

        # Final norm
        rms_final = tensors["model.norm.weight"].astype(np.float32)
        f.write(rms_final.tobytes())

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\nWritten: {output_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 convert_weights.py <model_dir> <output.bin>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
