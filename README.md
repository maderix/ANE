# ANE Training — Backpropagation on Apple Neural Engine

Training neural networks directly on Apple's Neural Engine (ANE) via reverse-engineered private APIs. No CoreML training APIs, no Metal, no GPU — pure ANE compute.

## Project Scope & Intent

I'm genuinely grateful for all the attention this project has received — I never expected a weekend research hack to blow up like this. Thank you to everyone who starred, forked, ran benchmarks on their own hardware, and shared the work. It means a lot.

That said, I want to set clear expectations about what this project is and isn't.

This is a **research project**, not a production framework.

The goal was to demonstrate that **training on the Apple Neural Engine — and potentially other NPUs — is possible**, and that the barrier has always been software support, not hardware capability. The ANE is a remarkably capable piece of silicon that Apple restricts to inference-only use through CoreML. This project bypasses that restriction using reverse-engineered private APIs to show what's possible when you give the hardware a chance.

### What This Project Is

- A proof of concept for ANE training via `_ANEClient` and `_ANECompiler` private APIs
- A set of benchmarks documenting real ANE performance characteristics (throughput, power, SRAM behavior)
- A reference for anyone exploring direct ANE access outside CoreML
- Research code that I update when I find something interesting

### What This Project Is Not

- A maintained framework or library
- A replacement for CoreML, MLX, llama.cpp, or any production inference stack
- A path to training large models on consumer hardware (yet)

### On The Hype

Some coverage of this project has overstated its implications. To be clear:

- Training works, but utilization is low (~5-9% of peak) with significant engineering challenges remaining
- Many element-wise operations still fall back to CPU
- This does **not** replace GPU training for anything beyond small research models today

The honest results — including all limitations — are documented in the accompanying articles:
- [Part 1: Reverse Engineering](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Part 2: Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)

### On Maintenance

I don't intend to grow this into a large community project. My focus is on original research (compiler infrastructure for edge AI optimization), and maintaining an open-source framework takes time away from that.

That said:
- I'll keep pushing updates when I discover something interesting
- Bug fixes and benchmark contributions (especially on hardware I don't own) are welcome
- Feature requests will likely go unaddressed — but feel free to fork
- PRs will be merged at a relatively slow pace, otherwise I become the bottleneck for community growth around this tech

### Fork it, build on it

This is MIT licensed for a reason. Everyone now has access to AI-assisted development tools that can adapt and extend code in hours. If this project is useful to you — take it, modify it, build something better. If you do something cool with it, I'd love to hear about it.If in future, community decides to maintain one source of truth repo, I'm in full support of that.

---

## What This Is

A from-scratch implementation of transformer training (forward + backward pass) running on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS (M4) inference accelerator that Apple does not expose for training. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

**Current results — Stories110M (12-layer, dim=768, seq=256, 109M params):**
- Static pipeline: **91 ms/step** (M3 Ultra), **106 ms/step** (M4)
- Dynamic pipeline: **110 ms/step**, no recompilation
- 72 ANE kernels per step (static), 9 shared kernels (dynamic)
- All forward and backward dx passes on ANE, dW gradients on CPU (Accelerate cblas)
- Adam optimizer, gradient accumulation, checkpoint/resume via exec() restart

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

## Training Data

Training requires pretokenized TinyStories data. To download:
```bash
cd training && bash download_data.sh
```
See [training/README.md](training/README.md) for detailed training instructions.

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
- **Compile overhead** — Static pipeline recompiles 60+ kernels every 10 steps (~3.7s); dynamic pipeline avoids this
- **Low utilization** — Training sustains ~1-2 TFLOPS out of 15.8+ peak due to CPU fallbacks and I/O overhead

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

## Community Benchmarks

Community hardware benchmark submissions live in [`benchmarks/submissions/`](benchmarks/submissions/).

- [Mac Studio (Apple M3 Ultra, 256 GB) — 2026-03-03](benchmarks/submissions/m3-ultra-mac-studio-2026-03-03/README.md)

## Disclaimer

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`). These APIs are not covered by any public stability guarantee and may change or break with any macOS update. This is independent research into Apple Neural Engine architecture, using APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)

---

*Built by a human + Claude, one weekend at a time.*

