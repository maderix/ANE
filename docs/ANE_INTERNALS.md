# ANE Internals: What We Know

A comprehensive guide to Apple's Neural Engine (ANE) based on reverse engineering, private API exploration, and community research. This extends and updates [hollance/neural-engine](https://github.com/hollance/neural-engine/tree/master/docs) with findings from direct hardware experimentation on M4 Max / macOS 15.

---

## Table of Contents

1. [How does the ANE work internally?](#1-how-does-the-ane-work-internally)
2. [Can I program the ANE directly?](#2-can-i-program-the-ane-directly)
3. [What can be compiled and run on ANE?](#3-what-can-be-compiled-and-run-on-ane)
4. [Security and safety mechanisms](#4-security-and-safety-mechanisms)
5. [Is the ANE 16-bit?](#5-is-the-ane-16-bit)
6. [ANE vs GPU vs CPU](#6-ane-vs-gpu-vs-cpu)
7. [Reverse engineering the ANE](#7-reverse-engineering-the-ane)
8. [How to verify ANE execution](#8-how-to-verify-ane-execution)
9. [References and external resources](#9-references-and-external-resources)

---

## 1. How does the ANE work internally?

> hollance/neural-engine says: "I don't think anyone outside Apple knows."

We now know substantially more.

### Hardware Architecture

The ANE is a fixed-function neural network accelerator integrated into Apple Silicon SoCs:

| Chip | ANE Cores | Peak TOPS | SRAM Budget |
|------|-----------|-----------|-------------|
| A12-A13 | 8 | 5 | ~4 MB |
| A14/M1 | 16 | 11 | ~16 MB |
| A15/M2 | 16 | 15.8 | ~24 MB |
| M4/M4 Pro/M4 Max | 16 | 38 | ~24-32 MB |

SRAM budget measured via `sram_probe.m` performance cliff detection on M4 Max:
- Peak efficiency at ~12.5 MB weights (282.6 GFLOPS/MB)
- First spill at ~32 MB (drops to 59.2 GFLOPS/MB)
- Catastrophic spilling at 128 MB (8.0 GFLOPS/MB)

The ANE operates on FP16 data exclusively. All I/O is through IOSurface shared memory buffers in `[1, C, 1, S]` channel-first FP16 layout.

### Compilation Pipeline

There are two paths from a neural network to ANE hardware execution:

**Standard CoreML path** (from [Black Hat Asia 2021, Wish Wu](https://infocondb.org/con/black-hat/black-hat-asia-2021/apple-neural-engine-internal-from-ml-algorithm-to-hw-registers)):

```
ML model (TF/PyTorch/Caffe)
  -> coremltools -> .mlmodel
  -> coremlc (CoreML compiler) -> .mlmodelc/
  -> espresso precompile -> net.plist + weights
  -> ANECompiler (in ane_compiler_service) -> model.hwx
  -> aned daemon -> H11ANEIn kernel driver (IOKit)
  -> ANE firmware -> hardware registers
```

**Direct private API path** (what this project uses):

```
MIL text + weight blobs (in memory)
  -> _ANEInMemoryModelDescriptor (ObjC object)
  -> _ANEInMemoryModel.compileWithQoS: -> ANE binary (in temp dir)
  -> _ANEInMemoryModel.loadWithQoS: -> loaded onto ANE hardware
  -> _ANEInMemoryModel.evaluateWithQoS: -> execution via aned
```

The direct path bypasses CoreML, espresso, and the `.hwx` file format entirely. It compiles MIL (Model Intermediate Language) text directly into ANE-executable binary, loads it, and runs it. This is how we achieve both training and inference on the ANE without any CoreML dependency.

### System Architecture

```
+------------------+     +------------------+     +------------------+
| User Process     |     | aned daemon      |     | Kernel           |
|                  |     |                  |     |                  |
| _ANEClient  -----+---->| ANE scheduler    +---->| H11ANEIn driver  |
| (sharedConnection)|    | (all interfaces) |     | (IOKit)          |
|                  |     |                  |     |                  |
| App gets 3 IOKit |     | Compiles models  |     | Passes model.hwx |
| interfaces:      |     | Manages loading  |     | to ANE firmware  |
|  - open          |     | Handles requests |     |                  |
|  - close         |     +------------------+     +------------------+
|  - programSend   |                                      |
|    Request       |                                      v
+------------------+                              +------------------+
                                                  | ANE Firmware     |
                                                  | (co-processor)   |
                                                  |                  |
                                                  | Parses register  |
                                                  | operations from  |
                                                  | compiled binary  |
                                                  +------------------+
```

The `aned` daemon mediates between user processes and the kernel driver. Apps only get 3 IOKit interfaces (open, close, programSendRequest). The daemon has access to all driver interfaces, which is why `_ANEClient.sharedConnection` communicates through the daemon rather than directly to the kernel.

### Execution Paths

We have benchmarked four distinct ways to trigger ANE kernel execution:

| Method | API | Latency (64x32) | Latency (768x256) |
|--------|-----|------------------|--------------------|
| Standard | `model.evaluateWithQoS:options:request:error:` | 0.175 ms | 0.205 ms |
| Real-Time | `client.evaluateRealTimeWithModel:options:request:error:` | 0.093 ms | 0.246 ms |
| processRequest | `program.processRequest:model:qos:...` | 0.131 ms | 0.185 ms |
| Direct | `client.doEvaluateDirectWithModel:options:request:qos:error:` | 0.225 ms | N/A |

**Key finding**: At production kernel dimensions (768x256, matching Stories110M), all paths converge to ~0.2 ms per kernel. The RT speedup (1.88x) observed on small 64x32 kernels does not hold at production scale. The standard path remains the most reliable.

### Resource Limits

The ANE runtime leaks internal resources during compilation. After ~119 compiles per process, subsequent compilations fail silently. The workaround is checkpoint-and-restart: save weights and optimizer state, terminate the process, and re-launch with `--resume`.

With `MAX_COMPILES=100` (conservative) and 60 weight-bearing kernels per batch (12 layers x 5 kernels), only 1 training batch fits per process lifetime.

---

## 2. Can I program the ANE directly?

> hollance/neural-engine says: "Unfortunately not. You can only use the Neural Engine through Core ML."

**Yes, you can.** The `AppleNeuralEngine.framework` contains 67+ private Objective-C classes that provide direct access to the ANE without CoreML. This project uses them for both training and inference.

### Minimal Example

The core compilation/load/execution cycle in pseudocode:

```objc
#import <dlfcn.h>
#import <objc/runtime.h>

// Load the private framework
dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

// Write MIL program as text
NSData *milData = [@"program(1.0) { ... }" dataUsingEncoding:NSUTF8StringEncoding];

// Create descriptor
id descriptor = [_ANEInMemoryModelDescriptor modelWithMILText:milData
                                                      weights:weightDict
                                                  optionsPlist:nil];

// Compile -> Load -> Run
id model = [_ANEInMemoryModel inMemoryModelWithDescriptor:descriptor];
[model compileWithQoS:21 options:nil error:&error];
[model loadWithQoS:21 options:nil error:&error];

// Create IOSurface I/O and request
id request = [_ANERequest requestWithInputs:@[inputSurface]
                               inputIndices:@[@0]
                                    outputs:@[outputSurface]
                              outputIndices:@[@0]
                              weightsBuffer:nil
                                  perfStats:nil
                             procedureIndex:0];

[model evaluateWithQoS:21 options:nil request:request error:&error];
```

A complete reusable wrapper is implemented in [`training/ane_runtime.h`](../training/ane_runtime.h) with functions:
- `ane_init()` -- load framework, resolve classes
- `ane_compile(kernel, mil_text, weight_dict)` -- compile MIL to ANE binary
- `ane_run(kernel)` -- standard execution path
- `ane_free(kernel)` -- unload and release resources

### MIL (Model Intermediate Language)

MIL is Apple's intermediate representation for neural network operations. Key facts:

- Text-based format: `program(1.0) { func main(...) { ... } }`
- Targets: `ios16`, `ios17`, `ios18` (determines available ops)
- All tensors are 4D: `[batch, channels, height, width]` or equivalently `[1, C, 1, S]`
- Convolutions (`conv`) are the workhorse: a 1x1 conv with `[out_ch, in_ch, 1, 1]` weights = matrix multiply
- Weights referenced via `BLOBFILE(path="@model_path/weights/name.bin", offset=uint64(64))`
- Weights are baked at compile time and cannot be swapped at runtime

Supported operations include: `conv`, `matmul`, `add`, `mul`, `sigmoid`, `softmax`, `reshape`, `transpose`, `concat`, `reduce_mean`, `rsqrt`, `cast`, `constexpr_affine_dequantize`, and more.

### Alternative: ANECompiler CLI

[ANETools](https://github.com/antgroup-skyward/ANETools) (from Wish Wu / Ant Group) provides command-line tools that invoke the ANECompiler module directly:

```bash
# Convert mlmodelc to ANE-compatible format
MLModelCToANECompiler input.mlmodelc output/

# Compile to hardware format
ANECompiler --target-arch ane_v5 --debug-mask 2147483647 net.plist weights/ output.hwx

# Disassemble compiled binary
ANEDisassembler output.hwx
```

The `--debug-mask` flag (set to max integer) generates intermediate files during compilation, revealing internal register operations.

---

## 3. What can be compiled and run on ANE?

Any computation expressible as a static MIL (Model Intermediate Language) dataflow graph that the E5 compiler accepts. The ANE is a fixed-function accelerator, not a general-purpose processor -- it executes predefined operation graphs, not arbitrary code.

### Verified Operations

These operations have been compiled to custom MIL programs and executed on ANE hardware with output validated against CPU reference implementations (see `test_mil_custom.m`):

| Category | Operations | Notes |
|----------|-----------|-------|
| Activations | `relu`, `gelu`, `softmax` | GELU supports EXACT, TANH_APPROXIMATION, SIGMOID_APPROXIMATION modes |
| Normalization | `layer_norm` | Epsilon type must match gamma/beta dtype |
| Attention | `scaled_dot_product_attention` | Fused Q@K^T/sqrt(d) + softmax + @V in a single op (iOS 18+) |
| Linear algebra | `linear` (const weights), `matmul` (runtime tensors) | `linear` requires compile-time constant weights; `matmul` supports runtime inputs |
| Type conversion | `cast` | fp32 <-> fp16. Required at ANE I/O boundaries |
| Elementwise | `add`, `mul`, `real_div` | Broadcasting supported |
| Shape | `reshape`, `transpose`, `concat`, `slice_by_index` | `concat` requires `interleave` param |
| Composite | Full transformer block (LN + SDPA + Residual + FFN + GELU) | Compiles and runs as a single ANE program (~0.21ms) |

### Available but Not Yet Tested

These are valid MIL operations that the E5 compiler should accept:

- `conv` -- convolutions (the upstream maderix/ANE repo uses these extensively for training)
- `reduce_sum`, `reduce_mean`, `reduce_max` -- reductions
- `gather`, `scatter` -- embedding lookups, KV cache writes
- `rsqrt`, `sqrt`, `exp`, `log`, `tanh` -- unary math
- `split`, `slice_by_size` -- tensor slicing
- `batch_norm`, `instance_norm` -- normalization variants
- Various pooling, padding, upsampling operations

### What Cannot Run on ANE

| Limitation | Detail |
|-----------|--------|
| No control flow | No loops, conditionals, or branching. MIL is a static dataflow graph. |
| No dynamic shapes | All tensor dimensions must be known at compile time. |
| No runtime weight updates | Weights are `const`, baked into the compiled binary. Changing weights requires recompilation (~10-50ms). |
| No arbitrary memory access | No pointers or indexing beyond what `gather`/`scatter` provide. |
| No custom ops | Only operations in Apple's MIL op set. No user-defined kernels at the hardware level. |
| No FP32 compute | ANE computes in FP16 only. FP32 inputs are cast to FP16 internally. |

### Implications for Training

The ANE can execute the forward pass and the matrix math of backpropagation (`matmul` for dX and dW gradients). However, training is impractical because weights are read-only constants. After computing weight gradients on ANE, the optimizer step (W -= lr * dW) must run on CPU, and the MIL program must be recompiled with updated weights before the next forward pass. This recompilation costs ~10-50ms per step, dominating training time. See [ANE_CHAINING_RESEARCH.md, Section 9](ANE_CHAINING_RESEARCH.md#9-ane-training-feasibility-analysis) for detailed analysis.

---

## 4. Security and Safety Mechanisms

The ANE has multiple layers of safety enforcement, but Apple's security model assumes access goes through CoreML. The private APIs we use bypass CoreML but still pass through the `aned` daemon and the E5 compiler.

### Compile-Time Safety

| Mechanism | What it does |
|-----------|-------------|
| MIL syntax validation | The E5 compiler rejects malformed MIL with `InvalidMILProgram` errors |
| Type checking | Tensor dtypes, shapes, and parameter types must match exactly. Mismatches cause compile errors (e.g., `layer_norm` epsilon must match gamma/beta dtype; `concat` axis must be `int32` scalar, not tensor) |
| Op validation | Unknown or unsupported operations are rejected |
| I/O matching | MIL input/output names and shapes must match the `MLModelDescription` passed to `MLE5Engine` |

### Runtime Safety

| Mechanism | What it does |
|-----------|-------------|
| Shape enforcement | Input tensors must match declared shape exactly -- `MultiArray shape doesn't match ML Program's expected shape` error on mismatch |
| Daemon mediation | ANE runs through the `aned` daemon (system service). User processes only get 3 IOKit interfaces: open, close, `programSendRequest` |
| IOSurface isolation | I/O memory is managed by the kernel via IOSurface. Cannot read/write arbitrary memory through them |
| SRAM limits | Programs exceeding the ANE SRAM budget (~24-32MB on M4 Max) are rejected or fall back to CPU/GPU |
| Compile limit | ~119 compiled programs per process before the compiler leaks enough resources to fail (resource exhaustion, not a security boundary) |

### Sandbox Interaction

The E5 runtime needs write access to `~/Library/Caches/<binary_name>/` for its ANE specialization cache. macOS app sandbox can block this, causing compilation to fail with permission errors. When running outside a sandbox (e.g., command-line tools), this directory is created automatically.

### What is NOT Protected

| Gap | Detail |
|-----|--------|
| No access control | No authentication or entitlement check for using the private APIs. Any process can call `_ANEClient.sharedConnection` |
| No rate limiting | Programs can be compiled in a loop until the ~119 limit exhausts resources |
| No MIL signing | No code signing validation on MIL text -- any syntactically valid program that passes the compiler's type checks will execute |
| No isolation between programs | Multiple programs from the same process share the ANE with no hardware-level isolation (the daemon schedules them) |

### Practical Risk Assessment

The ANE attack surface is limited because:

1. **Fixed-function hardware**: The ANE executes predefined neural network operations, not arbitrary instructions. There is no instruction pointer, no stack, and no way to jump to arbitrary code.
2. **Typed dataflow**: MIL programs operate on typed tensors with fixed shapes. There are no buffer overflows in the traditional sense -- the compiler enforces all dimensions at compile time.
3. **Daemon intermediary**: All ANE access goes through `aned`, which validates requests before forwarding to the kernel driver. Direct IOKit access to the ANE is restricted to 3 interfaces.
4. **No persistent state**: ANE programs don't persist across reboots. Compiled programs live in temp directories and caches that are cleaned by the OS.

The main risk of the private APIs is **stability**: these APIs are undocumented and may change with any macOS update, potentially breaking programs that depend on them.

---

## 5. Is the ANE 16-bit?

> hollance/neural-engine says: "It appears so."

**Confirmed.** The ANE operates in FP16 for both compute and storage:

- All IOSurface I/O must be FP16. Passing FP32 data produces zeros.
- MIL programs must use `fp16` I/O types (setting `g_fp16_io=1` in our codebase)
- F32-to-F16 conversion happens on the CPU before writing to IOSurfaces
- FP16 precision limits: values above ~65504 overflow, values below ~5.96e-8 underflow to zero

### Quantization Support

| Format | ANE Native? | Notes |
|--------|------------|-------|
| FP16 | Yes | Native compute and storage format |
| INT8 | Partial | Memory bandwidth savings only, no compute speedup. `constexpr_affine_dequantize` in MIL dequantizes to FP16 before compute |
| Q4 | No | Not supported. Requires GPU (Metal) or CPU dequantization |
| FP32 | No | Internally converted to FP16; higher precision lost |

Apple markets ANE TOPS using INT8, so the 38 TOPS figure for M4 is really ~19 TFLOPS in FP16 (each INT8 op counts as 1 TOP but FP16 ops count as 2).

---

## 6. ANE vs GPU vs CPU

Benchmarked on Qwen2.5-0.5B (dim=896, 24 layers, 494M params) on M4 Max:

### Decode Performance (single-token generation)

| Engine | Format | Weight Size | Decode t/s | Bottleneck |
|--------|--------|-------------|------------|------------|
| CPU AMX (cblas_sgemv) | F32 | 1.97 GB | ~91 t/s | Memory bandwidth |
| CPU AMX (cblas_sgemv) | F16->F32 | 658 MB disk | ~91 t/s | Memory bandwidth (F32 in RAM) |
| CPU AMX (cblas_sgemv) | Q4->F32 | 188 MB disk | ~91 t/s | Memory bandwidth (dequant at load) |
| Metal GPU (Q4 SIMD) | Q4 | 188 MB | ~10 t/s | Dispatch overhead (~400 dispatches/token) |
| LM Studio (MLX) | Q4 MLX | ~188 MB | 258-496 t/s | Optimized Metal kernels |

### Prefill Performance (batch prompt processing)

| Engine | Format | Prefill t/s | Method |
|--------|--------|-------------|--------|
| CPU AMX (cblas_sgemm) | F32 | 880-960 t/s | Batched matmul |
| CPU AMX (cblas_sgemv) | F32 | ~40 t/s | Sequential per-token |

### ANE Training Kernel Performance

| Metric | Value |
|--------|-------|
| Kernel latency | ~0.2 ms per kernel (768x256 production dims) |
| Peak TFLOPS | 11.14 (128x conv 512ch sp64) |
| Sustained training | 1.29-1.68 TFLOPS |
| ANE utilization | 8-11% of peak |

### When to use each

- **ANE**: Best for parallel FP16 operations where data stays on-chip (training kernels, fused attention). The ~119 compile limit and FP16-only restriction are significant constraints.
- **GPU (Metal)**: Best for large models (dim >= 4096) where native quantized matmul kernels (as in MLX/llama.cpp) can read Q4/Q8 data directly from GPU memory. Dispatch overhead dominates for small models.
- **CPU AMX**: Best for small/medium model decode (dim <= 896). `cblas_sgemv` uses the AMX coprocessor internally and achieves ~33% of theoretical bandwidth. Cannot be beaten by manual NEON, threading, or Metal for this model size.

---

## 7. Reverse engineering the ANE

### Prior Work

| Project | Focus | Key Contribution |
|---------|-------|-------------------|
| [hollance/neural-engine](https://github.com/hollance/neural-engine) | CoreML-level documentation | Comprehensive device list, layer compatibility, model surgery guides |
| [geohot/tinygrad ANE](https://github.com/tinygrad/tinygrad) | Driver-level reverse engineering | Initial IOKit driver analysis, ANE instruction format exploration |
| [Black Hat Asia 2021 (Wish Wu)](https://infocondb.org/con/black-hat/black-hat-asia-2021/apple-neural-engine-internal-from-ml-algorithm-to-hw-registers) | Full stack: ML to HW registers | Documented compilation pipeline, .hwx format, security attack surfaces, FaceID ANE usage. Created ANEDisassembler. [Video](https://www.youtube.com/watch?v=1wvBDUnPNEo) |
| [ANETools](https://github.com/antgroup-skyward/ANETools) | CLI compilation and disassembly | ANECompiler CLI wrapper, ANEDisassembler for .hwx files, `debug_mask` flag for intermediate output |
| [eiln/anecc](https://github.com/eiln/anecc) | Independent ANE compiler | CoreML-to-ANE compiler for Asahi Linux, alternative compilation path |
| [freedomtan/coreml_to_ane_hwx](https://github.com/freedomtan/coreml_to_ane_hwx) | CoreML to .hwx conversion | Direct converter bypassing some CoreML steps |
| [maderix/ANE](https://github.com/maderix/ANE) | Training on ANE | First neural network training on ANE via private APIs |
| [maderix Substack](https://open.substack.com/pub/maderix/p/inside-the-m4-apple-neural-engine) | M4 ANE deep-dive | Detailed M4 ANE architecture analysis, SRAM probing, kernel fusion |

### Our Discoveries: Private API Class Hierarchy

We have documented 20+ private Objective-C classes in `AppleNeuralEngine.framework`:

```
NSObject
|-- _ANEClient (singleton, daemon connection)
|   Methods: sharedConnection, evaluateWithModel:, evaluateRealTimeWithModel:,
|            doEvaluateDirectWithModel:, prepareChainingWithModel:,
|            enqueueSetsWithModel:, buffersReadyWithModel:,
|            beginRealTimeTask, endRealTimeTask
|
|-- _ANEInMemoryModelDescriptor (MIL + weights spec)
|   Factory: +modelWithMILText:weights:optionsPlist:
|
|-- _ANEInMemoryModel (compile/load/run)
|   Methods: compileWithQoS:, loadWithQoS:, evaluateWithQoS:, unloadWithQoS:
|   Props: hexStringIdentifier, programHandle (uint64), program, perfStatsMask
|
|-- _ANEModel (disk-based compiled model -- 52 instance methods)
|   Factory: +modelAtURL:key:, +modelAtURL:key:modelAttributes:
|   Methods: getUUID, inputSymbolIndicesForProcedureIndex:,
|            outputSymbolIndicesForProcedureIndex:
|   Props: mapper, program
|
|-- _ANERequest (I/O surface packaging)
|   Factory: +requestWithInputs:inputIndices:outputs:outputIndices:
|             weightsBuffer:perfStats:procedureIndex:
|
|-- _ANEIOSurfaceObject (thin IOSurface wrapper)
|   Factory: +objectWithIOSurface:
|
|-- _ANEBuffer (IOSurfaceObject + symbolIndex + source) [KEY DISCOVERY]
|   Factory: +bufferWithIOSurfaceObject:symbolIndex:source:
|   source: 0=ANE, 1=output, 2=unknown
|
|-- _ANEChainingRequest (multi-op pipeline)
|   Factory: +chainingRequestWithInputs:outputSets:lbInputSymbolId:
|             lbOutputSymbolId:procedureIndex:signalEvents:
|             transactionHandle:fwEnqueueDelay:memoryPoolId:
|   Methods: validate
|
|-- _ANEIOSurfaceOutputSets (output packaging for chaining)
|   Factory: +objectWithstatsSurRef:outputBuffer:
|   Note: requires non-NULL statsSurRef (any IOSurface works, even 64 bytes)
|
|-- _ANEInputBuffersReady (input signaling for chaining)
|   Factory: +inputBuffersWithProcedureIndex:inputBufferInfoIndex:
|             inputFreeValue:executionDelay:
|
|-- _ANEOutputSetEnqueue (output pipeline config for chaining)
|   Factory: +outputSetWithProcedureIndex:setIndex:signalValue:
|             signalNotRequired:isOpenLoop:
|
|-- _ANEProgramForEvaluation (lower-level program)
|   Factory: +programWithHandle:intermediateBufferHandle:queueDepth:
|   Methods: processRequest:model:qos:qIndex:modelStringID:options:
|             returnValue:error:
|
|-- _ANEProgramIOSurfacesMapper (symbol-to-surface mapping)
|   Factory: +mapperWithProgramHandle:, +mapperWithController:
|   Note: only works with _ANEModel, not _ANEInMemoryModel
|
|-- _ANEPerformanceStats
|   Factory: +statsWithHardwareExecutionNS:
|   Props: hwExecutionTime, performanceCounters
|
|-- _ANESharedSignalEvent (hardware signal fence)
|   Factory: +signalEventWithValue:symbolIndex:eventType:sharedEvent:
|   Requires IOSurfaceSharedEvent objects
|
|-- _ANESharedWaitEvent (hardware wait fence)
|   Factory: +waitEventWithValue:sharedEvent:
|   Requires IOSurfaceSharedEvent objects
|
|-- _ANEModelInstanceParameters, _ANEDeviceController, _ANEQoSMapper
```

Full details with experiment logs: [ANE_CHAINING_RESEARCH.md](ANE_CHAINING_RESEARCH.md)

### ChainingRequest API Status

The `_ANEChainingRequest` API is designed to pipeline multiple ANE operations without CPU round-trips. Current status:

- `_ANEChainingRequest.validate` returns **YES** (with `_ANEBuffer` inputs + `_ANEIOSurfaceOutputSets` outputs)
- `prepareChainingWithModel:` **fails** -- calls `getUUID` on `_ANEInMemoryModel` which lacks it
- Requires `_ANEModel` (disk-based compiled model) which has `getUUID` and symbol index methods
- `_ANEModel` factory methods require a `key:` parameter; the hex identifier from `_ANEInMemoryModel` is the likely key

This is the highest-priority research area. Chaining would eliminate the ~23 CPU-ANE round-trips per token in a 12-layer model, potentially enabling on-chip pipeline execution.

### model.hwx Binary Format

The `.hwx` file is the compiled hardware representation loaded by the ANE kernel driver. From Wu's Black Hat research:

- Mach-O format binary containing register operations
- Compiled from `net.plist` + weights by the ANECompiler module
- Loaded by the `H11ANEIn` kernel driver via `programCreate` interface
- ANE firmware parses it to extract register addresses and values
- Can be disassembled with [ANETools/ANEDisassembler](https://github.com/antgroup-skyward/ANETools)

Our `_ANEInMemoryModel` path bypasses `.hwx` generation -- the model goes directly from MIL to an internal binary format in a temp directory. Whether this temp directory contains an equivalent to `.hwx` is an open question (see [ANE_CHAINING_RESEARCH.md](ANE_CHAINING_RESEARCH.md) for next steps).

---

## 8. How to verify ANE execution

### Power Monitoring

```bash
sudo powermetrics --samplers ane_power -i 1000
```

Shows real-time ANE power draw. Active ANE usage typically shows 2-4W on M4 Max during training.

### Performance Statistics

```objc
model.perfStatsMask = 0xFF;
// After execution:
// model.performanceCounters -- returns nil on current macOS (limited API)
```

The `_ANEPerformanceStats` class exists and can be instantiated via `+statsWithHardwareExecutionNS:`, but the hardware counters are not populated on the current macOS/M4 combination. The `perfStatsMask` property is accepted but `performanceCounters` returns nil after execution.

### IOSurface Output Validation

Read back FP16 data from output IOSurfaces and compare against CPU reference:

```objc
_Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(surface);
IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
for (int i = 0; i < n; i++) {
    float val = (float)out[i];
    // Compare against CPU reference
}
IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
```

### ANE Compiler Debug Output

From Wu's research, the ANECompiler module has a `debug_mask` flag. Setting it to `2147483647` (max int) generates intermediate files during compilation, revealing:
- Register operation sequences
- Memory allocation decisions
- Tiling strategies
- Weight layout in SRAM

This can be applied when using the ANECompiler CLI tools from [ANETools](https://github.com/antgroup-skyward/ANETools).

---

## 9. References and External Resources

### Documentation and Research

| Resource | URL | Focus |
|----------|-----|-------|
| hollance/neural-engine | https://github.com/hollance/neural-engine | CoreML-level ANE docs |
| maderix Substack | https://open.substack.com/pub/maderix/p/inside-the-m4-apple-neural-engine | M4 ANE architecture |
| Black Hat Asia 2021 | https://infocondb.org/con/black-hat/black-hat-asia-2021/apple-neural-engine-internal-from-ml-algorithm-to-hw-registers | Full stack reverse engineering |
| BH Asia 2021 Video | https://www.youtube.com/watch?v=1wvBDUnPNEo | 30-min talk by Wish Wu |
| Apple ML Research | https://machinelearning.apple.com/research/neural-engine-transformers | Deploying transformers on ANE |
| ANE Supported Devices | https://github.com/hollance/neural-engine/blob/master/docs/supported-devices.md | Comprehensive device/chip list |

### Tools

| Tool | URL | Purpose |
|------|-----|---------|
| ANETools | https://github.com/antgroup-skyward/ANETools | ANECompiler CLI, ANEDisassembler |
| eiln/anecc | https://github.com/eiln/anecc | Independent ANE compiler (Asahi Linux) |
| freedomtan/coreml_to_ane_hwx | https://github.com/freedomtan/coreml_to_ane_hwx | CoreML to .hwx converter |
| coremltools | https://github.com/apple/coremltools | Apple's official ML model tools |

### Projects Using ANE Directly

| Project | URL | What it does |
|---------|-----|-------------|
| maderix/ANE | https://github.com/maderix/ANE | Training on ANE (this project's upstream) |
| dev-erik/ANE | https://github.com/dev-erik/ANE | This fork: inference optimization, ChainingRequest research |

### This Project's ANE Documentation

| Document | Description |
|----------|-------------|
| [ANE_INTERNALS.md](ANE_INTERNALS.md) | This file -- comprehensive ANE internals guide |
| [ANE_CHAINING_RESEARCH.md](ANE_CHAINING_RESEARCH.md) | ChainingRequest API research, experiment logs, benchmarks |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Training system architecture, kernel fusion map, data flow |
| [API_REFERENCE.md](API_REFERENCE.md) | Complete function index for all source files |
| [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) | M4 Max benchmark results (training, TFLOPS, SRAM) |
