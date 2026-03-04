# ANE Training — Stories110M on Apple Neural Engine

Training a 109M-parameter Llama2-architecture transformer (Stories110M) directly on Apple's Neural Engine using private ANE APIs.

![Dashboard](dashboard.gif)

## Architecture

- **Model**: Stories110M — dim=768, hidden=2048, heads=12, layers=12, vocab=32000, seq=256
- **109.53M params** (84.95M transformer + 24.58M embedding)
- **SDPA causal mask workaround**: ANE hardware ignores attn_mask — decompose into Q@K^T (ANE conv) + mask+softmax (CPU) + scores@V (ANE conv)

## Three Training Pipelines

### 1. Static Baseline (`train_large`)
Original pipeline. Weights baked as constants in MIL kernels — recompile every 10 steps via `exec()` restart.

- 60 weight-bearing + 12 weight-free kernels = 72 per compile batch
- Classifier + softmax + RMSNorm backward on CPU
- **106.7 ms/step**, 7.6s compile per restart

### 2. Static + ANE Extras (`train_large_ane`) — PR#19
Offloads classifier forward (32K conv), softmax, final RMSNorm, and RMSNorm backward to ANE. Bridge API for C-callable ANE access.

- 86 kernels per compile batch (+24 rmsnorm_bwd, +1 classifier, +1 finalRms)
- **91.8 ms/step** (14% faster), 9.6s compile per restart
- Use `--no-ane-extras` to disable and fall back to CPU (for debugging)

### 3. Dynamic Weight Pipeline (`training_dynamic/`)
Weights passed via IOSurface spatial dimension — compile 9 kernels once at startup, no recompilation needed.

- 9 shared kernels across all 12 layers
- **111 ms/step**, 0.4s one-time compile
- No exec() restart, no compile limit issues

## Performance Comparison (20 Steps)

| | Static Baseline | PR#19 + ANE extras | PR#19 no extras | Dynamic |
|---|---|---|---|---|
| **Wall time** | **10.1s** | **11.7s** | **10.7s** | **~2.6s** |
| Compile | 7.6s (75.7%) | 9.6s (81.6%) | 7.5s (69.7%) | 0.4s (15%) |
| Train | 2.1s (21.2%) | 1.8s (15.6%) | 2.9s (27.4%) | 2.2s (85%) |
| **ms/step** | **106.7** | **91.8** | **147.0** | **111** |
| Kernels/restart | 72 | 86 | 60 | 9 (once) |
| ANE TFLOPS | 0.87 | 1.15 | 0.72 | — |
| Total TFLOPS | 1.63 | 1.90 | 1.19 | — |

**Key insights:**
- Dynamic wins on wall time for any practical run length (3.9x faster at 20 steps)
- PR#19 has the best per-step throughput (92ms) but compile overhead dominates short runs
- Static restarts every 10 steps, so dynamic's zero-recompile advantage compounds

## Files

| File | Description |
|------|-------------|
| `train_large.m` | Static baseline — 72 kernels, classifier/softmax on CPU |
| `train_large_ane.m` | PR#19 — 86 kernels, classifier/softmax/rmsnorm_bwd on ANE |
| `training_dynamic/train.m` | Dynamic pipeline — 9 kernels, weights via IOSurface |
| `training_dynamic/mil_dynamic.h` | MIL generators for dynamic weight kernels |
| `training_dynamic/config.h` | Model config (DIM=768, HIDDEN=2048, etc.) |
| `training_dynamic/io.h` | IOSurface I/O + MIL compilation helpers |
| `training_dynamic/cpu_ops.h` | CPU ops (SiLU backward, cross-entropy, Adam) |
| `stories_config.h` | Static pipeline config, structs, alloc helpers |
| `stories_io.h` | IOSurface I/O, NEON fp16 conversion, kernel compile/eval |
| `stories_mil.h` | MIL generators for static pipeline (6 kernel types) |
| `stories_cpu_ops.h` | vDSP-vectorized RMSNorm, cross-entropy, Adam |
| `ane_classifier.h` | ANE classifier fwd (32K conv), softmax kernels |
| `ane_rmsnorm_bwd.h` | ANE rmsnorm backward kernel |
| `dashboard.py` | TUI dashboard — loss curve, power/CPU/memory graphs |
| `Makefile` | Build targets |

## Usage

### 1. Download Training Data

```bash
bash download_data.sh
```

Downloads pretokenized TinyStories (Llama 2 BPE, 32K vocab) from HuggingFace. Produces `tinystories_data00.bin` (~41 MB, ~20M tokens).

All training pipelines perform token-data validation at startup:
- token file byte length must align to 16-bit token boundaries
- token file must contain at least `SEQ+1` tokens
- every token id must be within `[0, vocab_size)`

### 2. Build & Train

```bash
# Static baseline (classifier + softmax on CPU)
make train_large
./train_large stories110M.bin 256 100 1e-4
./train_large --model stories110M.bin --steps 100 --lr 1e-4

# PR#19: ANE-offloaded classifier + softmax + rmsnorm_bwd
make train_large_ane
./train_large_ane stories110M.bin 256 100 1e-4
./train_large_ane --no-ane-extras --steps 100    # disable ANE extras

# Dynamic pipeline (no recompilation)
cd training_dynamic && make train
./train --scratch              # train from random init
./train                        # resume from checkpoint
./train --steps 200 --lr 1e-4  # custom steps/lr
```

**CLI flags (all pipelines):**
- `--steps N` (default 10000)
- `--lr F` (default 3e-4)
- `--model PATH` — pretrained weights file
- `--ckpt PATH` — checkpoint file (preserved across exec() restarts)
- `--resume` — resume from checkpoint
- `--no-ane-extras` — (train_large_ane only) disable ANE classifier/softmax/rmsnorm_bwd

### 3. Monitor with Dashboard

```bash
pip install blessed psutil numpy
sudo python3 dashboard.py          # static pipeline
sudo python3 dashboard.py --dynamic # dynamic pipeline
```

### 4. Benchmarking

All programs print an **Efficiency Report** at completion:

```
=== Efficiency Report ===
Total steps:     20
Wall time:       11738 ms (11.7 s)
Compile time:    9583 ms (81.6%)
Train time:      1835 ms (15.6%)
Avg train:       91.8 ms/step
ANE TFLOPS:      1.15 sustained
```

## Key Techniques

- **NEON vectorized fp16↔fp32**: ARM NEON intrinsics for fast IOSurface data transfer
- **vDSP cross-entropy**: `vDSP_mtrans` + `vvexpf` + `vDSP_sve` — 8x faster than scalar
- **Async weight gradients**: cblas_sgemm dispatched to background queue, overlapped with ANE
- **Vocab compaction** (dynamic): 32K → 9.2K active tokens, 3.5x reduction in classifier work
- **Dynamic weight packing**: Activations + weights concatenated in IOSurface spatial dimension — one kernel serves all 12 layers
- **exec() restart**: Workaround for ANE ~119 compile limit per process
