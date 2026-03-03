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

- Training works, but utilization is low (~8-11% of peak) with significant engineering challenges remaining
- Many element-wise operations still fall back to CPU
- This does **not** replace GPU training for anything beyond small research models today

The honest results — including all limitations — are documented in the accompanying articles:
- [Part 1: Reverse Engineering](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Part 2: Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)

### Fork it, build on it

This is MIT licensed for a reason. Everyone now has access to AI-assisted development tools that can adapt and extend code in hours. If this project is useful to you — take it, modify it, build something better. If you do something cool with it, I'd love to hear about it.

---

## Community Fork

This fork extends the original project with:

- **M1/M2/M3/M4 compatibility** — MIL syntax fixes for broader Apple Silicon support (from upstream PR #6)
- **Security hardening** — stack protection, format security, input validation (upstream PRs #5, #7)
- **Bug fixes** — token sampling underflow fix, dashboard sudo hang fix (upstream PRs #17, #20)
- **Configurable paths** — training data, model, and checkpoint paths via environment variables
- **Community benchmarks** — standardized benchmark script + online dashboard for comparing results across chips
- **12-layer training** — full Stories110M (12 transformer layers, 109M params) already working

### Contributing

We welcome benchmark submissions from any Apple Silicon hardware. See [Community Benchmarks](#community-benchmarks) below for how to run and submit your results.

---

## Quick Start

**Requirements:** macOS 15+ on Apple Silicon (M1/M2/M3/M4/M5), Xcode CLI tools.

```bash
# Install Xcode CLI tools (if not already installed)
xcode-select --install

# Clone and set up
git clone https://github.com/dev-erik/ANE.git
cd ANE/training

# Download training data + model weights
make setup

# Build and run training (12-layer Stories110M)
make train_large
./train_large --steps 100
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANE_MODEL_PATH` | `../../assets/models/stories110M.bin` | Path to model weights |
| `ANE_DATA_PATH` | `../../assets/data/tinystories_data00.bin` | Path to tokenized training data |
| `ANE_CKPT_PATH` | `/tmp/ane_ckpt.bin` | Path for checkpoint files |
| `ANE_ACCUM_STEPS` | `10` | Gradient accumulation steps before weight update (max 10000) |

---

## What This Is

A from-scratch implementation of transformer training (forward + backward pass) running on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS (M4) inference accelerator that Apple does not expose for training. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

**Current results (M4 Max, 12-layer Stories110M, dim=768, seq=256):**
- 62-72 ms/step, 8-11% ANE utilization (1.3-1.7 TFLOPS sustained)
- 6 ANE kernel dispatches per layer per training step
- All forward and backward dx passes on ANE, dW gradients on CPU (Accelerate cblas)
- Adam optimizer, gradient accumulation, checkpoint/resume via process restart

## Architecture

The training loop uses 6 ANE kernels per step per layer:

| Kernel | Function | Weights |
|--------|----------|---------|
| `kFwdAttn` | RMSNorm + QKV projection + SDPA + output projection | Wq, Wk, Wv, Wo, rms1, mask |
| `kFwdFFN` | RMSNorm + SwiGLU FFN (W1, W3, SiLU, W2) | W1, W2, W3, rms2 |
| `kFFNBwd` | FFN backward (W2^T + SiLU_bwd + W1^T + W3^T) | W2^T, W1^T, W3^T |
| `kSdpaBwd1` | Wo^T + SDPA backward part 1 (dV, probs, dp) | Wo^T, mask |
| `kSdpaBwd2` | SDPA backward part 2 (softmax grad, dQ, dK) | — |
| `kQKVb` | QKV backward (Wq^T + Wk^T + Wv^T -> dx) | Wq^T, Wk^T, Wv^T |

CPU handles: RMSNorm backward, residual connections, loss computation, dW gradient accumulation (cblas_sgemm), Adam optimizer updates.

Key optimizations:
- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates all transpose overhead
- **vDSP vectorized RMSNorm** — 10x faster than naive (6.7ms to 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals on a serial dispatch queue
- **Deferred cblas wait** — wait pushed into next step's forward pass for maximum overlap
- **ANE RMSNorm fusion** — RMSNorm folded into forward kernels as MIL ops (reduce_sum + pow + mul)
- **Wo^T fusion** — output projection backward merged into SDPA backward kernel
- **Forward taps** — Q, K, V, attention scores, hidden states exposed via concat outputs, avoiding CPU recompute
- **Process restart** — bypasses ~119 ANE compile limit per process via checkpoint and re-launch

## File Structure

```
├── api_exploration.m       # Initial ANE API discovery
├── inmem_basic.m           # In-memory MIL compilation proof-of-concept
├── inmem_bench.m           # ANE dispatch latency benchmarks
├── inmem_peak.m            # Peak TFLOPS measurement
├── sram_bench.m            # ANE SRAM bandwidth probing
├── sram_probe.m            # SRAM size/layout exploration
├── scripts/
│   ├── run_benchmarks.sh           # Full benchmark suite runner
│   ├── run_community_benchmark.sh  # Standardized community benchmark (JSON output)
│   ├── gen_mlpackages.py           # Generate .mlpackage models for sram/inmem tests
│   └── aggregate_benchmarks.py     # Aggregate community JSON results
├── community_benchmarks/           # Community-submitted benchmark results (JSON)
├── web/                            # Dashboard web app (Next.js + Neon Postgres)
├── docs/
│   ├── ARCHITECTURE.md             # System architecture with diagrams
│   ├── API_REFERENCE.md            # Complete function index
│   ├── BENCHMARKS.md               # Benchmark guide
│   └── BENCHMARK_RESULTS.md        # Detailed M4 Max results
└── training/
    ├── ane_runtime.h       # ANE private API wrapper (compile, eval, IOSurface)
    ├── ane_mil_gen.h       # MIL program generation helpers
    ├── ane_classifier.h    # Classifier forward/backward MIL generators
    ├── ane_rmsnorm_bwd.h   # RMSNorm backward MIL generator
    ├── stories_config.h    # Model configuration (dims, structs, macros)
    ├── stories_io.h        # IOSurface I/O, blob builders, compile/eval helpers
    ├── stories_mil.h       # MIL generators (SDPA, FFN, QKV backward)
    ├── stories_cpu_ops.h   # CPU ops (RMSNorm, Adam, cross-entropy, embed)
    ├── model.h             # Gen1 model weight init and blob builders
    ├── forward.h           # Gen1 forward pass MIL generators
    ├── backward.h          # Gen1 backward pass MIL generators
    ├── train_large.m       # Main: 12-layer training (CPU classifier)
    ├── train_large_ane.m   # 12-layer training (ANE classifier)
    ├── train.m             # Minimal training loop (early prototype)
    ├── tiny_train.m        # 2-layer tiny model training
    ├── test_*.m            # Unit tests for individual kernels
    ├── dashboard.py        # Real-time training monitor
    ├── tokenize.py         # Training data preprocessing
    ├── download_data.sh    # Download training data + model weights
    └── Makefile            # Build system (make train_large, make test, etc.)
```

## Community Benchmarks

We collect community benchmark results across Apple Silicon chips to understand ANE performance characteristics.

### Run Benchmarks

```bash
# Run the standardized community benchmark
bash scripts/run_community_benchmark.sh

# Skip training benchmarks (if no training data)
bash scripts/run_community_benchmark.sh --skip-training

# Custom training steps
bash scripts/run_community_benchmark.sh --steps 50
```

The script will:
1. Detect your hardware (chip, memory, cores)
2. Run SRAM probe and in-memory peak benchmarks
3. Optionally run training benchmarks
4. Save results as JSON to `community_benchmarks/`
5. Ask if you'd like to submit results to the online dashboard

### Submit Results

**Option A: Automatic submission**
At the end of the benchmark run, the script will ask if you want to submit. Your results are sent anonymously to our dashboard (IP is hashed, never stored raw).

**Option B: GitHub PR**
1. Fork this repository
2. Run the benchmark script
3. Commit the JSON file from `community_benchmarks/`
4. Open a Pull Request

**Option C: GitHub Issue**
Paste the contents of your JSON results file in a new issue.

### View Results

Visit the **[ANE Community Benchmark Dashboard](https://web-lac-sigma-61.vercel.app)** to see aggregated results across all Apple Silicon chips.

### Data Privacy

- Your IP address is hashed (SHA-256) for rate limiting and duplicate detection only
- No personal information is collected or stored
- All benchmark data is public
- Rate limited to 5 submissions per hour per IP

---

## Building

Requires macOS 15+ on Apple Silicon (tested on M1 through M5).

```bash
cd training

# Build everything
make all

# Build just the training programs
make train_large train_large_ane

# Run tests
make test

# Download training data
make data

# Full setup (data + dependencies)
make setup
```

No external dependencies. Uses only system frameworks + private ANE APIs resolved at runtime via `objc_msgSend`.

## How It Works

1. **MIL generation** — Objective-C code constructs MIL program text at runtime, specifying convolutions (for linear layers), matmul (for attention), softmax, element-wise ops
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + weight blobs directly to ANE programs, no disk mlmodelc needed
3. **IOSurface I/O** — Input/output tensors passed via IOSurface shared memory in `[1, channels, 1, spatial]` format (fp16)
4. **Weight embedding** — Weights baked into ANE programs as BLOBFILE constants; recompiled each batch when weights change
5. **Gradient flow** — Forward taps expose intermediates needed for backward; backward kernels compute dx (input gradients) on ANE; dW (weight gradients) computed on CPU via cblas

## Limitations

- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention is decomposed into separate Q@K^T (ANE) then mask+softmax (ANE via add+softmax) then scores@V (ANE)
- **~119 compile limit** — ANE compiler leaks resources; worked around via process restart with checkpoint
- **Compilation overhead** — Weights baked at compile time mean recompilation every ACCUM_STEPS. Compilation is 80-85% of wall time. Investigating `_ANEChainingRequest` for potential pipeline without recompile.
- **Classifier backward regression** — ANE classifier backward is ~3x slower than CPU cblas due to matmul (not conv) being used to work around ANE's 8192 input channel limit
- **SRAM capacity** — ANE SRAM is ~24-32 MB (M4 Max). Models with weight matrices exceeding this threshold spill to DRAM with significant performance cliffs. Current Stories110M weights (~1.2 MB each) stay within SRAM.

## Performance History

| Optimization | ms/step | ANE util |
|---|---|---|
| Baseline (vDSP transpose) | 33.5 | 3.1% |
| Channel-first layout | 20.3 | 5.2% |
| vDSP vectorized RMSNorm | 14.2 | 7.4% |
| GCD async cblas overlap | 11.4 | 9.2% |
| ANE RMSNorm fusion | 11.4 | 9.2% |
| Wo^T fusion (7 to 6 kernels) | 11.4 | 9.2% |
| Deferred cblas wait | **9.3** | **11.2%** |

*Note: Above numbers are for single-layer training. Full 12-layer training runs at 62-72 ms/step.*

## Disclaimer

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`). These APIs are not covered by any public stability guarantee and may change or break with any macOS update. This is independent research into Apple Neural Engine architecture, using APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA section 1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)

---

*Originally built by [maderix](https://github.com/maderix). Community fork maintained with contributions from the ANE research community.*
