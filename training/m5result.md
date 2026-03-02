# M5 ANE Probe Results

**Machine**: Apple M5, macOS 26.3 (Darwin 25.3.0)
**Date**: 2026-03-01
**ANE Family**: H16 (same as M4)

---

## test_weight_reload — FAIL

**Question**: Can we skip recompilation by overwriting weight blobs on disk and calling unload+load?

**Result**: **No.** Weights are baked at compile time. Overwriting `weights/weight.bin` in tmpDir and doing unload→load produces identical output — the ANE ignores the file change.

```
Kernel: 64x64 conv, spatial=32
Compile+load: 33.3ms | Unload: 0.5ms | Reload: 3.8ms
Output A (identity): [0.0100, 0.0200, 0.0300, 0.0400]
Output B (3x identity, after file overwrite + reload): [0.0100, 0.0200, 0.0300, 0.0400]
Max A-B diff: 0.000000
```

**Implication**: Cannot eliminate compilation bottleneck via file swap. Must use async recompile, raise ACCUM_STEPS, or find another path.

---

## test_perf_stats — Partial Success

**Question**: What hardware counters does `_ANEPerformanceStats` expose?

**Result**: The class exists with useful properties, but `alloc/init` returns `nil`. Must be created via factory methods that require internal buffers.

### Available Properties
| Property | Type | Description |
|----------|------|-------------|
| `hwExecutionTime` | uint64 | Hardware execution time in nanoseconds |
| `perfCounterData` | NSData | Raw performance counter data blob |
| `pStatsRawData` | NSData | Raw stats data |

### Factory Methods
- `+statsWithHardwareExecutionNS:` — create from hw execution time
- `+statsWithRequestPerformanceBuffer:statsBufferSize:` — create from raw buffer
- `+statsWithReconstructed:hardwareExecutionNS:aneStatsRawData:` — reconstruct from components
- `+driverMaskForANEFMask:` — convert ANE feature mask to driver mask

### Instance Methods
- `-performanceCounters` — returns counter object
- `-stringForPerfCounter:` — human-readable counter name
- `-emitPerfcounterSignpostsWithModelStringID:` — emit signposts for profiling

**Key Finding**: `_ANEModel` has `perfStatsMask` property. Setting this on the model before eval likely enables perf stats population in the request. The `_ANEPerformanceStats` object passed to request gets populated *by the driver* — we need to set the mask first, then read stats after eval.

---

## test_qos_sweep — All QoS Values Work

**Question**: Does QoS affect ANE frequency or latency?

**Result**: All QoS values 0-63 compile, load, and eval successfully. **No measurable latency difference** — ANE appears to run at fixed frequency regardless of QoS.

```
Kernel: 256x256 conv, spatial=64 (8.4 MFLOPS)
 QoS    Compile       Load    Eval(1) Eval(avg10)  Status
   0     13.9ms     15.6ms     0.22ms     0.11ms  OK
   1     11.6ms      1.8ms     0.17ms     0.07ms  OK
   5     11.4ms      1.7ms     0.17ms     0.07ms  OK
  10     12.0ms      1.8ms     0.18ms     0.06ms  OK
  21     11.8ms      1.7ms     0.18ms     0.08ms  OK
  33     11.5ms      1.7ms     0.17ms     0.06ms  OK
  47     10.8ms      1.7ms     0.18ms     0.06ms  OK
  63     11.3ms      1.7ms     0.17ms     0.07ms  OK
```

**Notes**:
- QoS 0 has elevated load time (15.6ms vs ~1.7ms) — possibly first-use initialization
- Compile time ~11ms, load ~1.7ms, eval ~0.07ms avg for 8.4 MFLOPS kernel
- Eval throughput: 8.4M / 0.07ms = **120 GFLOPS** for a single 256×256 conv

---

## test_ane_advanced — Key Findings

### weightsBuffer IOSurface — Does NOT Override

Passing a `weightsBuffer` IOSurface with different weights to the request **does not change output**. The compiled weights are still used.

```
Baseline (1x identity): Output[0..3] = [0.1000, 0.2000, 0.3000, 0.3999]
weightsBuffer (3x identity): Output[0..3] = [0.1000, 0.2000, 0.3000, 0.3999]
```

The `weightsBuffer` parameter likely serves a different purpose (perhaps for models that declare runtime weights vs baked constants).

### procedureIndex — All 0-15 Succeed

All procedure indices 0-15 return OK. Single-procedure models work with any index (they probably ignore non-zero indices). Multi-procedure models compiled from `_ANEChainingRequest` would use different indices for different subgraphs.

### SharedEvents — Classes Exist, Need IOSurfaceSharedEvent

- `_ANESharedEvents`, `_ANESharedSignalEvent`, `_ANESharedWaitEvent` all exist
- `alloc/init` returns nil — they need `IOSurfaceSharedEvent` objects (Metal shared events)
- `_ANESharedSignalEvent` has `symbolIndex` and `agentMask` — for GPU↔ANE sync
- Signal API: `+signalEventWithValue:symbolIndex:eventType:sharedEvent:`
- Wait API: `+waitEventWithValue:sharedEvent:eventType:`

### ChainingRequest — Exists with Loopback Support

`_ANEChainingRequest` supports chained execution:
- `inputBuffer`, `outputSets` — multiple output sets for pipeline
- `loopbackInputSymbolIndex`, `loopbackOutputSymbolIndex` — feed output back as input
- `fwEnqueueDelay` — firmware-level enqueue timing
- `memoryPoolId` — shared memory pool across chained ops
- `signalEvents` — sync with other agents

### Notable _ANEClient Methods
- `evaluateRealTimeWithModel:options:request:error:` — real-time eval path
- `loadRealTimeModel:options:qos:error:` — RT model loading
- `beginRealTimeTask` / `endRealTimeTask` — RT task bracketing
- `prepareChainingWithModel:options:chainingReq:qos:error:` — set up chaining
- `enqueueSetsWithModel:outputSet:options:qos:error:` — enqueue output sets
- `buffersReadyWithModel:inputBuffers:options:qos:error:` — signal input ready

### All ANE Classes Found (67 total)
Key unexplored classes: `_ANEDeviceController`, `_ANEQoSMapper`, `_ANEBuffer`, `_ANEIOSurfaceOutputSets`, `_ANEProgramForEvaluation`, `_ANEProgramIOSurfacesMapper`, `_ANEModelInstanceParameters`, `_ANEInputBuffersReady`, `_ANEOutputSetEnqueue`

---

## Strategic Implications

### Compilation Bottleneck (Primary)
Weight reload and weightsBuffer both fail. **Weights are irrevocably baked at compile time.** The only paths forward:
1. **Raise ACCUM_STEPS significantly** (10→100+) to amortize compile cost
2. **Async background compilation** while training continues with old weights
3. **Chaining API** (`_ANEChainingRequest`) to pipeline multiple layers in one dispatch

### Performance Monitoring
`hwExecutionTime` from `_ANEPerformanceStats` gives wall-clock ANE time per eval. To enable:
1. Set `perfStatsMask` on the `_ANEInMemoryModel` before eval
2. Pass an `_ANEPerformanceStats` to the request
3. Read `hwExecutionTime` after eval

### Real-Time Path
`_ANEClient` has a dedicated real-time evaluation path (`evaluateRealTimeWithModel:`) with RT load/unload. This may provide lower/more predictable latency.

### Chaining (Most Promising for Utilization)
`_ANEChainingRequest` with loopback could allow multiple layers to execute as a single ANE program without CPU round-trips between layers. Combined with `_ANEIOSurfaceOutputSets` and `_ANEInputBuffersReady`, this could dramatically reduce idle time between kernel dispatches.
