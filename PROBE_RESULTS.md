# ANE Probe Results: M4 (macOS 26.3)

**Machine:** Apple M4 (10 cores), 32GB RAM, macOS 26.3  
**Date:** 2026-03-03  
**ANE Family:** H16 (same as M5 results in `training/m5result.md`)

## Key Discovery: Compile and Eval Run in Parallel

**This was not known before.** The M5 probes tested compile and eval sequentially.
We tested with GCD `dispatch_async` and found they fully overlap.

### probe_v2.m Results

#### TEST 1: Pure Eval Throughput
```
Conv 128x128, spatial=64
1000 evals: 189.1ms total, 0.189ms/eval
11.09 GFLOPS sustained
```

#### TEST 2: Ping-pong (Two Pre-compiled Models)
```
500 ping-pong pairs: 207.4ms (0.415ms/pair, 0.207ms/eval)
```
Near-zero overhead switching between two loaded models.

#### TEST 3: Sequential Compile (20 Models)
```
All 20 models compiled and verified ✓
Compile time: ~23-29ms each (consistent, no degradation)
All 20 models correct with different scale factors
```

#### TEST 4: Background Compile Overlap ⭐
```
Background compile: 26.8ms
Foreground evals during compile: 119 (26.8ms total)
Overlap: YES — compile and eval CAN run in parallel!
Background model verified correct ✓
```

### Summary
| Metric | Value |
|--------|-------|
| Compile time | ~25ms per kernel set |
| Eval time | 0.189ms per eval |
| Compile:eval ratio | ~130:1 |
| Parallel compile+eval | **YES** |
| Max simultaneous models | 20+ |
| Ping-pong overhead | +10% vs single model |

## Peak ANE Throughput (inmem_peak)

```
Config                         W(MB)   GFLOP   ms/eval  TFLOPS
96x conv 512ch sp64            48.0    3.22    0.429 ms   7.50
128x conv 512ch sp64           64.0    4.29    0.589 ms   7.30
256x conv 256ch sp64           32.0    2.15    0.380 ms   5.65
64x conv 512ch sp64            32.0    2.15    0.395 ms   5.43
```

Peak: **7.50 TFLOPS** (47% of 15.8 TFLOPS theoretical).

## Implications for Training

### Before (train_large.m)
- Synchronous compile: **88.6% of wall time is compilation**
- 55ms compile per batch, 0.54ms actual training
- Training throughput limited by compiler, not by ANE

### After (train_double_buffer.m)
- Async double-buffered compile: **0% compile stall**
- Background compile happens during forward/backward passes
- ~130 eval steps fit in one compile window
- Weight updates are "delayed" by one batch (standard technique in distributed training)
- Training throughput limited only by ANE eval speed

### Architecture
```
Time →
Active kernels:  [=== eval batch N ===][=== eval batch N+1 ===][=== eval batch N+2 ===]
Background:      [compile N+1 weights ][compile N+2 weights   ][compile N+3 weights   ]
                 ↑                     ↑                       ↑
                 swap ready            swap ready               swap ready
```

Two kernel sets (A and B) alternate between active evaluation and background compilation.
When the background compile finishes, pointers swap atomically at the batch boundary.
