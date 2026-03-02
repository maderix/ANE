# ANE Training — Backpropagation on Apple Neural Engine

You might be asking, "why the FUCK would you pick GPT2?"

Have you read the art bro? Have you? Nah. I doubt it.

GPT2 had more soul in it's theoretical pinky finger than all of us combined. 

But I digress..

Training neural networks directly on Apple's Neural Engine (ANE) via reverse-engineered private APIs. No CoreML training APIs, no Metal, no GPU — pure ANE compute.

## What This Is

A from-scratch implementation of transformer training (forward + backward pass) running on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS (M4) inference accelerator that Apple does not expose for training. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

**Current results (M4, single transformer layer, dim=768, seq=512):**
- 9.3 ms/step, 11.2% ANE utilization (1.78 TFLOPS sustained)
- 6 ANE kernel dispatches per training step
- All forward and backward dx passes on ANE, dW gradients on CPU (Accelerate cblas)
- Adam optimizer, gradient accumulation, checkpoint/resume

## Architecture

The training loop uses 6 ANE kernels per step:

| Kernel | Function | Weights |
|--------|----------|---------|
| `kFwdAttn` | RMSNorm + QKV projection + SDPA + output projection | Wq, Wk, Wv, Wo, rms1, mask |
| `kFwdFFN` | RMSNorm + SwiGLU FFN (W1, W3, SiLU, W2) | W1, W2, W3, rms2 |
| `kFFNBwd` | FFN backward (W2^T + SiLU_bwd + W1^T + W3^T) | W2^T, W1^T, W3^T |
| `kSdpaBwd1` | Wo^T + SDPA backward part 1 (dV, probs, dp) | Wo^T, mask |
| `kSdpaBwd2` | SDPA backward part 2 (softmax grad, dQ, dK) | — |
| `kQKVb` | QKV backward (Wq^T + Wk^T + Wv^T → dx) | Wq^T, Wk^T, Wv^T |

CPU handles: RMSNorm backward, residual connections, loss computation, dW gradient accumulation (cblas_sgemm), Adam optimizer updates.

Key optimizations:
- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates all transpose overhead
- **vDSP vectorized RMSNorm** — 10x faster than naive (6.7ms → 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals on a serial dispatch queue
- **Deferred cblas wait** — wait pushed into next step's forward pass for maximum overlap
- **ANE RMSNorm fusion** — RMSNorm folded into forward kernels as MIL ops (reduce_sum + pow + mul)
- **Wo^T fusion** — output projection backward merged into SDPA backward kernel
- **Forward taps** — Q, K, V, attention scores, hidden states exposed via concat outputs, avoiding CPU recompute
- **exec() restart** — bypasses ~119 ANE compile limit per process

## File Structure

```
├── api_exploration.m       # Initial ANE API discovery
├── inmem_basic.m           # In-memory MIL compilation proof-of-concept
├── inmem_bench.m           # ANE dispatch latency benchmarks
├── inmem_peak.m            # Peak TFLOPS measurement (2048x2048 matmul)
├── sram_bench.m            # ANE SRAM bandwidth probing
├── sram_probe.m            # SRAM size/layout exploration
└── training/
    ├── ane_runtime.h       # ANE private API wrapper (compile, eval, IOSurface)
    ├── ane_mil_gen.h       # MIL program generation helpers
    ├── model.h             # Model weight initialization and blob builders
    ├── forward.h           # Forward pass MIL generators
    ├── backward.h          # Backward pass MIL generators
    ├── train.m             # Minimal training loop (early prototype)
    ├── tiny_train.m        # 2-layer tiny model training
    ├── train_large.m       # Main: single-layer dim=768 training (optimized)
    ├── test_*.m            # Unit tests for individual kernels
    └── Makefile
```

## Building

Requires macOS 15+ on Apple Silicon (tested on M4).

```bash
# Build the main training program
xcrun clang -O2 -framework Foundation -framework IOSurface \
  -framework CoreML -framework Accelerate -ldl -lobjc \
  -o train_large training/train_large.m

# Run
./train_large
```

No external dependencies. Uses only system frameworks + private ANE APIs resolved at runtime via `objc_msgSend`.

## How It Works

1. **MIL generation** — Objective-C code constructs MIL program text at runtime, specifying convolutions (for linear layers), matmul (for attention), softmax, element-wise ops
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + weight blobs directly to ANE programs, no disk mlmodelc needed
3. **IOSurface I/O** — Input/output tensors passed via IOSurface shared memory in `[1, channels, 1, spatial]` format (fp16)
4. **Weight embedding** — Weights baked into ANE programs as BLOBFILE constants; recompiled each batch when weights change
5. **Gradient flow** — Forward taps expose intermediates needed for backward; backward kernels compute dx (input gradients) on ANE; dW (weight gradients) computed on CPU via cblas

## Limitations

- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention is decomposed into separate Q@K^T (ANE) → mask+softmax (ANE via add+softmax) → scores@V (ANE)
- **~119 compile limit** — ANE compiler leaks resources; worked around via `exec()` restart with checkpoint
- **Single layer** — Currently trains one transformer layer; multi-layer would need pipeline scheduling
- **Synthetic data** — Currently uses random data for benchmarking; real tokenized data support is WIP

## Performance History

| Optimization | ms/step | ANE util |
|---|---|---|
| Baseline (vDSP transpose) | 33.5 | 3.1% |
| Channel-first layout | 20.3 | 5.2% |
| vDSP vectorized RMSNorm | 14.2 | 7.4% |
| GCD async cblas overlap | 11.4 | 9.2% |
| ANE RMSNorm fusion | 11.4 | 9.2% |
| Wo^T fusion (7→6 kernels) | 11.4 | 9.2% |
| Deferred cblas wait | **9.3** | **11.2%** |

## Disclaimer

This project is independent research into Apple Neural Engine architecture. It uses undocumented APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)


