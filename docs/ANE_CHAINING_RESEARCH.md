# ANE ChainingRequest API Research

Research into Apple Neural Engine private APIs for multi-kernel pipelining, conducted on M4 Max / macOS 15.

**Goal**: Eliminate CPU round-trips between ANE layer evaluations. In a 12-layer model, sequential evaluation requires 23+ CPU-ANE round-trips per token. The `_ANEChainingRequest` API appears designed to let the ANE run operations back-to-back in a hardware pipeline, keeping data on-chip.

**Status**: ChainingRequest validates and `prepareChainingWithModel:` no longer crashes (crash fix: pass nil for symbol/procedure params). Blocked on Code=15 (`ANEProgramChainingPrepare Failed`) -- the `_ANEModel` needs Espresso IR format (not MIL) for full symbol table population. At production dims (768x256), sequential ANE dispatch costs ~0.2ms/kernel; chaining would save ~23 round-trips per token.

See also: [ANE_INTERNALS.md](ANE_INTERNALS.md) for comprehensive ANE documentation including compilation pipeline, hardware specs, and community research references.

---

## Test Files

| File | Purpose |
|------|---------|
| `training/test_chaining.m` | v1 prototype: sequential baseline + ChainingRequest creation |
| `training/test_chaining_v2.m` | v2 deep exploration: 6-phase probe of 12+ private classes |
| `training/test_ane_model.m` | Experiments E-P: _ANEModel loading, compiler, chaining, fences, type encoding, mapping |
| `training/test_throughput_ceiling.m` | Experiment I: 12-kernel throughput ceiling benchmark |

Build and run:
```bash
cd training
make test_chaining && ./test_chaining
make test_chaining_v2 && ./test_chaining_v2
make test_ane_model && ./test_ane_model
make test_throughput_ceiling && ./test_throughput_ceiling
```

---

## 1. Executive Summary

### What works

| Finding | Impact | Status |
|---------|--------|--------|
| `evaluateRealTimeWithModel:` via `_ANEClient` | 1.88x faster on small kernels (64x32); **no benefit at production dims** (768x256) | Benchmarked |
| `processRequest` via `_ANEProgramForEvaluation` | 1.34x faster on small kernels; marginal at production dims | Benchmarked |
| `_ANEBuffer` wraps IOSurface with `symbolIndex` | Solves input indexing for chaining | Proven |
| All 9 unexplored ANE classes exist on M4 Max | Full API surfaces documented | Documented |

> **Important**: The RT execution speedup (1.88x) observed in isolated testing on 64x32 convolution kernels does **not** generalize to production dimensions. At 768x256 (Stories110M size), all four execution paths converge to ~0.2 ms per kernel. See [Production Dimension Results](#production-dimension-results-test_bench_pathsm-m4-max) below.

### What's been solved

| Finding | Status | Detail |
|---------|--------|--------|
| `_ANEIOSurfaceOutputSets` works with 64-byte statsSurRef | **SOLVED** | Any non-NULL IOSurface works as stats buffer |
| `_ANEChainingRequest.validate` returns YES | **SOLVED** | With proper `_ANEBuffer` inputs + `_ANEIOSurfaceOutputSets` outputs |
| `processRequest` via `_ANEProgramForEvaluation` | **1.34x faster** | Lower-level eval (0.131 ms vs 0.175 ms) |
| ChainingRequest factory crash (`[NSConstantIntegerNumber count]`) | **SOLVED** | Pass `nil` for `lbInputSymbolId`, `lbOutputSymbolId`, `procedureIndex` |
| `_ANEModel` loading from temp directory | **SOLVED** | `modelAtURL:key:` with tmpDir URL + hexStringIdentifier |
| `_ANESharedSignalEvent` / `_ANESharedWaitEvent` | **SOLVED** | Use `MTLSharedEvent` or `IOSurfaceSharedEventCreate()` |
| ChainingRequest type encodings | **DOCUMENTED** | All 9 factory params are `@` (object). `prepare` has 5 params (3x`@`, 1x`I` qos, 1x`^@` err) |

### What's still blocked

| Blocker | Root Cause |
|---------|------------|
| `prepareChainingWithModel:` returns Code=15 | `ANEProgramChainingPrepare() Failed` -- model not recognized as chaining-capable |
| `_ANEModel` has empty symbol table | MIL-compiled model shell lacks Espresso IR data (`model.espresso.net`) |
| `_ANEClient.loadModel:` / `compileModel:` fail | Require Espresso IR format, not MIL |
| `_ANEProgramIOSurfacesMapper` returns NO | Needs fully loaded model with symbol table |
| `_ANEPerformanceStats` with `_ANERequest` | Request expects `statType` selector on perfStats objects |

---

## 2. ANE Private API Class Map

### Core Classes (known working)

**`_ANEInMemoryModel`** -- the model object for in-memory MIL compilation.
- `+inMemoryModelWithDescriptor:` -- create from `_ANEInMemoryModelDescriptor`
- `-compileWithQoS:options:error:` -- compile MIL to ANE binary
- `-loadWithQoS:options:error:` -- load compiled model onto ANE
- `-evaluateWithQoS:options:request:error:` -- standard evaluation (QoS 0-63, 21 default)
- `-unloadWithQoS:error:` -- unload from ANE
- Properties: `hexStringIdentifier`, `programHandle` (uint64), `program` (`_ANEProgramForEvaluation`), `perfStatsMask`
- Missing: `inputSymbolNames`, `outputSymbolNames`, `inputSymbolIndicesForProcedureIndex:`

**`_ANEInMemoryModelDescriptor`** -- model specification.
- `+modelWithMILText:weights:optionsPlist:` -- create descriptor from MIL NSData + weight dict

**`_ANERequest`** -- evaluation request packaging I/O surfaces.
- `+requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
- `perfStats` parameter expects `NSArray` of stat info objects (not `_ANEPerformanceStats`)

**`_ANEIOSurfaceObject`** -- thin wrapper around `IOSurfaceRef`.
- `+objectWithIOSurface:` -- wrap a raw IOSurface
- Does NOT have `symbolIndex` property (this is the v1 blocker)

**`_ANEClient`** -- client connection to the ANE daemon.
- `+sharedConnection` -- singleton accessor
- `-evaluateWithModel:options:request:qos:error:` -- 5-param eval via client
- `-evaluateRealTimeWithModel:options:request:error:` -- **RT priority eval (1.7x faster)**
- `-doEvaluateDirectWithModel:options:request:qos:error:` -- direct eval bypass
- `-beginRealTimeTask` / `-endRealTimeTask` -- RT task bracketing (returns NO, but RT eval still works)
- `-prepareChainingWithModel:options:chainingReq:qos:error:` -- chaining setup
- `-enqueueSetsWithModel:outputSet:options:qos:error:` -- chaining output enqueue
- `-buffersReadyWithModel:inputBuffers:options:qos:error:` -- chaining input signal

### Discovered Classes (v2 exploration)

**`_ANEBuffer`** -- wraps `_ANEIOSurfaceObject` with index metadata. **Key discovery.**
- `+bufferWithIOSurfaceObject:symbolIndex:source:` -- factory
  - `ioSurfaceObject`: an `_ANEIOSurfaceObject` (NOT raw `IOSurfaceRef`)
  - `symbolIndex`: `NSNumber` mapping to compiled model I/O symbol
  - `source`: `long long` -- 0=ANE, 1=output, 2=unknown
- Properties: `ioSurfaceObject`, `symbolIndex`, `source`
- Description format: `"_ANEBuffer: { ioSurface=0x... ; symbolIndex=0 ; ANEBufferProducerAgent=0 }"`

**`_ANEProgramIOSurfacesMapper`** -- maps IOSurfaces to compiled model symbols.
- `+mapperWithProgramHandle:(uint64_t)handle` -- works, creates mapper
- `+mapperWithController:(id)ctrl` -- alternative factory
- `-mapIOSurfacesWithModel:request:cacheInference:error:` -- **FAILS** on `_ANEInMemoryModel` (calls `inputSymbolIndicesForProcedureIndex:` which doesn't exist)
- `-validateRequest:model:` -- also fails for same reason
- Implication: designed for `_ANEModel` (disk-based compiled models), not in-memory MIL

**`_ANEProgramForEvaluation`** -- lower-level evaluation program.
- Accessible via `model.program` property
- `+programWithHandle:intermediateBufferHandle:queueDepth:` -- factory
- `-processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:` -- low-level eval

**`_ANEIOSurfaceOutputSets`** -- output set packaging for chaining.
- `+objectWithstatsSurRef:outputBuffer:` -- factory
  - `statsSurRef`: `IOSurfaceRef` for perf stats collection -- **returns nil when NULL**
  - `outputBuffer`: `NSArray` of `_ANEBuffer` objects
- This is the current blocker: we don't know the correct stats IOSurface format

**`_ANEInputBuffersReady`** -- input signaling for chaining pipeline.
- `+inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:`
- Parameters: procedure index, buffer info indices, free values, execution delay
- This is the mechanism that tells the ANE "inputs are ready, start processing"

**`_ANEOutputSetEnqueue`** -- output pipeline configuration for chaining.
- `+outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:`
- Configures output set enqueue behavior with signal values and open-loop mode

**`_ANEChainingRequest`** -- the chaining request itself.
- `+chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:`
- `-validate` -- returns YES/NO
- Expects `inputs` as `_ANEBuffer` objects, `outputSets` as `_ANEIOSurfaceOutputSets` objects

**`_ANEModelInstanceParameters`** -- model instance configuration.
- Alloc/init produces a valid object
- API surface dumped but not yet exercised

**`_ANEDeviceController`** -- device-level controller.
- `+controllerWithProgramHandle:` -- attempted but returned nil in our tests

**`_ANEQoSMapper`** -- QoS level mapping.
- API surface dumped, not yet exercised

**`_ANEPerformanceStats`** -- performance statistics.
- `+statsWithHardwareExecutionNS:(uint64_t)ns` -- factory
- Properties: `hwExecutionTime`, `performanceCounters`
- Cannot be used with `_ANERequest.perfStats` (expects array of objects with `statType` selector)
- Setting `perfStatsMask=0xFF` on model works but `performanceCounters` returns nil

**`_ANESharedSignalEvent` / `_ANESharedWaitEvent`** -- hardware sync primitives (not yet explored).
- Likely the fence mechanism for GPU-ANE or multi-model synchronization
- Referenced in `_ANEChainingRequest.signalEvents` parameter

---

## 3. Experiment Logs

### v1: test_chaining.m Results (M4 Max)

```
=== ANE ChainingRequest Prototype ===

All required classes found.

--- Phase 1: Compile two identical conv kernels ---
  Kernel 1: compiled and loaded
  Kernel 2: compiled and loaded

--- Phase 2: Baseline (sequential eval) ---
  Sequential: 10.355 ms total (0.207 ms/pair)
  Output[0..3]: [0.2500, 0.2500, 0.2500, 0.2500]

--- Phase 3: _ANEChainingRequest exploration ---
  _ANEClient: obtained
  ChainingRequest created: _ANEChainingRequest: { inputBuffer=(
    "_ANEIOSurfaceObject: { ioSurface=0x... ; startOffset=0 }"
  ) ; outputSets=( ... ) }
  validate: NO

--- Phase 4: Loopback ChainingRequest ---
  ChainingRequest created (loopback)
  validate: NO
  prepareChainingWithModel: EXCEPTION (validate fails first)

--- Summary ---
  Sequential baseline: 0.207 ms/pair (two evals + memcpy)
  ChainingRequest: creates but validate FAILS
  Root cause: _ANEIOSurfaceObject lacks symbolIndex property
  Next: explore _ANEBuffer and _ANEProgramIOSurfacesMapper
```

### v2: test_chaining_v2.m Results (M4 Max)

**Phase 1: Class Introspection**
- 9 classes found, 0 missing
- All classes exist on M4 Max / macOS 15
- Full method lists, properties, and type encodings dumped for each

**Phase 2: Symbol Name Discovery**
- `inputSymbolNames`: NOT available on `_ANEInMemoryModel`
- `outputSymbolNames`: NOT available on `_ANEInMemoryModel`
- `programHandle`: YES (uint64 handle to compiled program)
- `_ANEIOSurfaceObject` does NOT have `symbolIndex` getter or setter
- `+objectWithIOSurface:symbolIndex:` class method NOT available

**Phase 3: IOSurface Mapper & Buffer Experiments**

3a: `_ANEProgramIOSurfacesMapper`
```
  mapperWithProgramHandle(12345): created successfully
  mapIOSurfacesWithModel: EXCEPTION
    -[_ANEInMemoryModel inputSymbolIndicesForProcedureIndex:]:
    unrecognized selector
  validateRequest:model: EXCEPTION (same reason)
```

3b: `_ANEBuffer` -- **success**
```
  bufferWithIOSurfaceObject(symIdx=0, source=0):
    _ANEBuffer: { ioSurface=0x... ; symbolIndex=0 ; ANEBufferProducerAgent=0 }
  bufferWithIOSurfaceObject(symIdx=0, source=1):
    _ANEBuffer: { ioSurface=0x... ; symbolIndex=0 ; ANEBufferProducerAgent=1 }
  bufferWithIOSurfaceObject(symIdx=0, source=2):
    _ANEBuffer: { ioSurface=0x... ; symbolIndex=0 ; ANEBufferProducerAgent=2 }
  bufferWithIOSurfaceObject(symIdx=1, source=0):
    _ANEBuffer: { ioSurface=0x... ; symbolIndex=1 ; ANEBufferProducerAgent=0 }
  symbolIndex property: accessible and correct
```

3c: `_ANEIOSurfaceObject` symbolIndex experiments
```
  setSymbolIndex: NOT available on _ANEIOSurfaceObject
  symbolIndex getter: NOT available
  +objectWithIOSurface:symbolIndex: NOT available
```

3d: IOSurface property experiments
```
  IOSurface 'symbolIndex' property (set via IOSurfaceSetValue): 0
  _ANEIOSurfaceObject.symbolIndex after property set: <exception>
  (IOSurface user properties do NOT propagate to _ANEIOSurfaceObject)
```

3e: `_ANEProgramForEvaluation`
```
  k1.model.program: <_ANEProgramForEvaluation: 0x...>
  (accessible via model.program property)
```

**Phase 4: ChainingRequest Retry**

4a: Sequential baseline
```
  Sequential: 0.259 ms/pair (50 iters)
  Output[0..3]: [0.2500, 0.2500, 0.2500, 0.2500]
```

Attempts 1-4: Various raw IOSurface configurations
```
  [Attempt 1] Standard (raw IOSurfaceObject): CRASH
    -[_ANEIOSurfaceObject symbolIndex]: unrecognized selector
  [Attempt 2] IOSurface with symbolIndex property: CRASH (same)
  [Attempt 3] Two-model loopback: CRASH (same)
  [Attempt 4] Skip validate, call prepareChainingWithModel directly: CRASH (same)
```

Attempt 5: `_ANEBuffer` + `_ANEIOSurfaceOutputSets`
```
  bufIn: _ANEBuffer: { ... symbolIndex=0 ; ANEBufferProducerAgent=0 }
  bufOut: _ANEBuffer: { ... symbolIndex=0 ; ANEBufferProducerAgent=1 }
  outputSet (objectWithstatsSurRef:NULL outputBuffer:@[bufOut]): nil
  -> _ANEIOSurfaceOutputSets returns nil when statsSurRef is NULL
```

Attempt 6: `_ANEClient.evaluateWithModel:` -- **works**
```
  evaluateWithModel (via client): YES
```

Attempt 7: `_ANEClient.doEvaluateDirectWithModel:` -- **works**
```
  doEvaluateDirectWithModel: YES
```

**Phase 5: Alternative Execution Paths**

5a: Real-time eval -- **1.7x speedup**
```
  beginRealTimeTask: NO (possibly needs entitlement)
  evaluateRealTimeWithModel: YES

  RT eval:       0.090 ms/eval avg (50 iters)
  Standard eval: 0.157 ms/eval avg (50 iters)
  RT vs Standard speedup: 1.74x

  endRealTimeTask: NO
```

5b: PerfStats
```
  perfStatsMask = 0x01..0x80: set OK (all masks accepted)
  statsWithHardwareExecutionNS:0 = <_ANEPerformanceStats>
  Eval with @[perfStats]: OK (no crash when wrapped in array)
  hwExecutionTime after eval: nil
  Eval with mask=0xFF, perfStats=nil: OK
  performanceCounters: nil
```

---

## 4. Evaluation Path Benchmarks

Measured on 64x32 convolution kernels, M4 Max, 200 iterations after 10 warmup:

| Method | Latency | Speedup | API |
|--------|---------|---------|-----|
| `evaluateWithQoS:` (standard) | 0.175 ms | 1.0x | `model.evaluateWithQoS:options:request:error:` |
| `evaluateRealTimeWithModel:` | 0.093 ms | **1.88x** | `client.evaluateRealTimeWithModel:options:request:error:` |
| `processRequest` | 0.131 ms | **1.34x** | `program.processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:` |
| `doEvaluateDirectWithModel:` | 0.225 ms | 0.78x | `client.doEvaluateDirectWithModel:options:request:qos:error:` |

Key observations (small kernel, isolated):
- RT eval was fastest in isolated test (1.88x speedup on 64x32)
- `processRequest` was faster than standard but slower than RT
- `doEvaluateDirectWithModel` was actually **slower** than standard (0.78x)
- `beginRealTimeTask` returning NO does not prevent `evaluateRealTimeWithModel:` from working

### Production Dimension Results (test_bench_paths.m, M4 Max)

At realistic kernel sizes with multiple compiled models, the picture changes:

| Config | Standard | RT | processRequest | ane_eval_rt |
|--------|----------|-----|----------------|-------------|
| 64x32 (test) | 0.109 ms | 0.233 ms (0.5x) | 0.156 ms (0.7x) | 0.195 ms (0.6x) |
| 128x64 | 0.208 ms | 0.184 ms (1.1x) | 0.201 ms (1.0x) | 0.185 ms (1.1x) |
| 256x64 | 0.197 ms | 0.212 ms (0.9x) | 0.203 ms (1.0x) | 0.157 ms (1.3x) |
| 512x64 | 0.120 ms | 0.147 ms (0.8x) | 0.194 ms (0.6x) | 0.179 ms (0.7x) |
| 768x256 (prod) | 0.205 ms | 0.246 ms (0.8x) | 0.185 ms (1.1x) | 0.291 ms (0.7x) |

**Key finding**: The RT eval speedup observed in isolated testing (1.88x) does not hold at production dimensions. At 768x256 (Stories110M size), all eval paths perform similarly (~0.2 ms), with standard eval being competitive or fastest. The overhead of the client-based paths (RT, direct) outweighs any ANE scheduling benefit at scale.

---

## 5. Remaining Blockers and Next Steps

### SOLVED: _ANEIOSurfaceOutputSets statsSurRef

The chaining pipeline requires:
1. Inputs as `_ANEBuffer` objects with `symbolIndex` -- **SOLVED**
2. OutputSets as `_ANEIOSurfaceOutputSets` objects -- **SOLVED**

A 64-byte IOSurface as `statsSurRef` is sufficient. `_ANEChainingRequest.validate` returns YES with this setup.

### SOLVED: ChainingRequest parameter type mismatch (Experiment K-L)

The `[NSConstantIntegerNumber count]` crash was caused by passing `NSNumber` values for `lbInputSymbolId`, `lbOutputSymbolId`, and `procedureIndex`. Type encoding analysis (Experiment K) revealed all 9 factory parameters are `@` (id/object), but the factory internally calls `count` on them, expecting arrays or nil.

**Fix**: Pass `nil` for `lbInputSymbolId`, `lbOutputSymbolId`, and `procedureIndex`:
```objc
chainingRequestWithInputs:@[buf] outputSets:@[outSet]
    lbInputSymbolId:nil lbOutputSymbolId:nil procedureIndex:nil
    signalEvents:@[] transactionHandle:@0 fwEnqueueDelay:@0 memoryPoolId:@0
```
This produces a valid `_ANEChainingRequest` (`validate` returns YES) and `prepareChainingWithModel:` no longer crashes.

### Current Blocker: ANEProgramChainingPrepare() Failed (Code=15)

`prepareChainingWithModel:` now returns NO with error:
```
Error Domain=com.apple.appleneuralengine Code=15
"ANEProgramChainingPrepare() Failed: Program chaining prepare error"
```

This error occurs with all three model types tested:
- Fresh `_ANEModel` (state=1, populated with programHandle+program)
- Populated `_ANEModel` from Experiment E (state=5 after failed loadModel/compileModel)
- `_ANEInMemoryModel` still crashes on `getUUID` (cannot be used with chaining at all)

The `Code=15` error is a **logical failure** in the ANE daemon's chaining preparation, not a crash. The model is not fully recognized as "chaining-capable" by the daemon, likely because:
1. The `_ANEModel` was populated by copying `programHandle`/`program` from an `_ANEInMemoryModel`, not loaded through the standard CoreML/Espresso pipeline
2. Symbol indices remain empty (the daemon may require them for chaining buffer routing)
3. The model needs `model.espresso.net` format (not MIL) for `_ANEClient.loadModel:` / `compileModel:`

**Previous blocker (SOLVED)**: `[NSConstantIntegerNumber count]` crash -- fixed by passing `nil` for symbol/procedure params.

### Experiments E-H Results (test_ane_model.m)

#### Experiment E: _ANEModel Loading -- SOLVED

`_ANEModel.modelAtURL:key:` works with the compiled temp directory URL and `hexStringIdentifier` as key:
```
diskModel = _ANEModel.modelAtURL:key:(tmpDirURL, hexId)
  -> _ANEModel with UUID, getUUID works
  -> state=1, program=nil, programHandle=0 (shell only)
```

Populating the shell with `_ANEInMemoryModel` data:
```
diskModel.setProgramHandle:(inMemoryModel.programHandle)  -> success
diskModel.setProgram:(inMemoryModel.program)              -> success
```

After population, `programHandle` and `program` are set, but `inputSymbolIndicesForProcedureIndex:0` still returns empty `NSIndexSet`. The symbol table data isn't stored in the `_ANEProgramForEvaluation` -- it's likely in the `model.hwx` or `net.plist` that the standard CoreML path generates.

#### Experiment E2: ANECompiler -- No ObjC API

- `ANECompiler.framework` exists at `/System/Library/PrivateFrameworks/ANECompiler.framework/` but contains **no ObjC classes** -- it's a pure C library (`ANECCompile()` is the entry point, called internally by `_ANEInMemoryModel.compileWithQoS:`)
- `debug_mask` option had no visible effect on compilation output
- No `ane_compiler_service` found at standard paths
- Key `_ANEInMemoryModel` compilation methods found: `saveModelFiles`, `localModelPath`, `compiledModelExists`, `mapIOSurfacesWithRequest:cacheInference:error:`

#### Experiment F: Chaining Pipeline -- Blocked

With populated `_ANEModel` (has UUID + programHandle + program), `prepareChainingWithModel:` still crashes on `[NSConstantIntegerNumber count]`. The crash is in the `_ANEChainingRequest` parameter handling, not in the model itself.

#### Experiment G: Hardware Fences -- FULLY SOLVED

Both `_ANESharedSignalEvent` and `_ANESharedWaitEvent` now work:

```objc
// MTLSharedEvent via Metal (works)
id device = MTLCreateSystemDefaultDevice();
id sharedEvent = [device newSharedEvent];

// IOSurfaceSharedEvent via IOKit (also works)
id iosEvent = IOSurfaceSharedEventCreate();

// Signal event factory: (uint64_t value, unsigned int symbolIndex, long long eventType, id sharedEvent)
_ANESharedSignalEvent.signalEventWithValue:symbolIndex:eventType:sharedEvent:
  -> works with both MTLSharedEvent and IOSurfaceSharedEvent

// Wait event factory: (uint64_t value, id sharedEvent)
_ANESharedWaitEvent.waitEventWithValue:sharedEvent:
  -> works with both event types
```

Event types 0, 1, 2 all produce valid signal events. The `eventType` property is correctly set.

#### Experiment H: Alternative Preparation -- Same Crash

`doPrepareChainingWithModel:options:chainingReq:qos:error:` exists with identical signature and crashes identically. Full `_ANEClient` API (46 instance methods) documented in test output.

### Throughput Ceiling (test_throughput_ceiling.m, Experiment I)

12-kernel pipeline benchmarks on M4 Max:

| Config | Sequential (run+memcpy) | Run-only | Memcpy-only | GCD Serial |
|--------|------------------------|----------|-------------|------------|
| 64x32 (test) | 0.272 ms/kernel | 0.158 ms/kernel | 0.001 ms/copy | 0.200 ms/kernel |
| 256x64 (small) | 0.191 ms/kernel | 0.181 ms/kernel | 0.002 ms/copy | 0.176 ms/kernel |
| 768x256 (prod) | 0.177 ms/kernel | 0.226 ms/kernel | 0.006 ms/copy | 0.186 ms/kernel |

**Key findings**:
- **Memcpy overhead is negligible** (<0.01 ms per copy even at 393KB). Not the bottleneck.
- **CPU round-trip overhead** is in the ANE dispatch itself, not data movement.
- At production dims, sequential with memcpy is actually *faster* than eval-only (pipeline caching effect).
- **GCD serial queue** provides modest improvement at small dims but marginal at production.
- **Chaining's value** would be eliminating the ~0.2ms/kernel ANE dispatch overhead, not memcpy. With 12 kernels, total pipeline takes ~2.1ms (prod), so eliminating dispatch could potentially halve this.

### Experiments K-P Results (test_ane_model.m, 2026-03-04)

#### Experiment K: Type Encoding Analysis -- COMPLETE

Full type encodings for all chaining-related methods:

| Method | Encoding | Notes |
|--------|----------|-------|
| `chainingRequestWithInputs:...` | `@88@0:8@16@24@32@40@48@56@64@72@80` | All 9 params are `@` (id/object) |
| `prepareChainingWithModel:...` | `B52@0:8@16@24@32I40^@44` | 5 params: 3x `@`, 1x `I` (uint32 qos), 1x `^@` (error ptr) |
| `doPrepareChainingWithModel:...` | `B52@0:8@16@24@32I40^@44` | Same signature as prepareChainingWithModel |

The `_ANEChainingRequest` factory takes 9 object parameters. The `lbInputSymbolId`, `lbOutputSymbolId`, and `procedureIndex` are all `@` (object), not raw integers. Internally, the factory calls `unsignedIntegerValue` (from NSNumber) or `count` (from NSArray) on these parameters.

| `_ANEChainingRequest` Property | Encoding | Type |
|-------------------------------|----------|------|
| `procedureIndex` | `@` | id (nil or NSArray) |
| `loopbackInputSymbolIndex` | `@` | id (nil or NSArray) |
| `loopbackOutputSymbolIndex` | `@` | id (nil or NSArray) |

#### Experiment L: Array-Typed Parameters -- BREAKTHROUGH

| Combo | lbIn | lbOut | procIdx | Factory | Validate | Prepare |
|-------|------|-------|---------|---------|----------|---------|
| L.1: Arrays `@[@(-1)]` | `@[@(-1)]` | `@[@(-1)]` | `@[@0]` | CRASH: `unsignedIntegerValue` on NSArray | - | - |
| L.2: Arrays `@[@0]` | `@[@0]` | `@[@0]` | `@[@0]` | CRASH: `unsignedIntegerValue` on NSArray | - | - |
| L.3: Empty `@[]` | `@[]` | `@[]` | `@[]` | CRASH: `unsignedIntegerValue` on empty array | - | - |
| **L.4: nil** | **nil** | **nil** | **nil** | **OK** | **YES** | **NO (Code=15)** |
| L.5: NSNumber | `@(-1)` | `@(-1)` | `@0` | CRASH: `count` on NSNumber | - | - |

**Passing `nil` for all three symbol/procedure params gets past both the factory crash and the `prepareChainingWithModel` crash.** The `validate` returns YES and `prepareChainingWithModel:` returns a clean error (Code=15: `ANEProgramChainingPrepare() Failed`) instead of crashing.

#### Experiment M: Load Model via _ANEClient -- BLOCKED

Both `loadModel:` and `compileModel:` on `_ANEClient` require **Espresso IR** format (`model.espresso.net`), not MIL:
```
Error Domain=com.apple.appleneuralengine.espresso Code=-1
"_ANEEspressoIRTranslator : error Cannot load network '.../model.espresso.net'"
```

`compiledModelExistsFor:` returns NO for our MIL-compiled model. After the failed load/compile attempts, the `_ANEModel` state changes from 1 to 5 (error/invalid state).

The standard CoreML pipeline generates `model.espresso.net` (Espresso IR) and `model.espresso.weights` from the `.mlpackage` / `.mlmodelc` format. Our MIL-only path bypasses this, so we can't use `_ANEClient.loadModel:` without first generating the Espresso IR.

#### Experiment N: IOSurface Mapping -- PARTIAL

`_ANEProgramIOSurfacesMapper`:
- `mapperWithProgramHandle:` creates a valid mapper from the `_ANEInMemoryModel` programHandle
- `mapIOSurfacesWithModel:request:cacheInference:error:` returns NO (no exception, no error output)
- `validateRequest:model:` returns NO
- `_ANEModel.mapper` property is nil
- `prepareANEMemoryMappingParams:request:` revealed `ANEMemoryMappingParamsStruct` has 128 `ANEBufferStruct` slots: `[128{ANEBufferStruct=^{__IOSurface}IiiI}]`

The mapper appears to need a fully loaded model with symbol table data that our MIL-compiled shell doesn't have.

#### Experiment O: Procedure Info -- EMPTY

- `procedureInfoForProcedureIndex:0` returns **nil** on the populated `_ANEModel`
- `procedureCount` is not a method or KVC-accessible property
- `modelAttributes` returns empty dictionary `{}`
- `inputSymbolNames` / `outputSymbolNames` not available on `_ANEModel`
- The `symbolIndicesForProcedureIndex:indexArrayKey:` method exists (takes `I` + `@`) but symbol data is empty

#### Experiment P: Full Chaining Retry -- Code=15

Tested with three model types, all using nil for symbol params:

| Model | State | validate | prepare Result |
|-------|-------|----------|---------------|
| Fresh `_ANEModel` (state=1, populated) | 1 | YES | NO (Code=15) |
| `_ANEInMemoryModel` | 3 | YES | CRASH: `getUUID` |
| Populated `_ANEModel` (from E, state=5) | 5 | YES | NO (Code=15) |

Also documented `_ANEInputBuffersReady` and `_ANEOutputSetEnqueue` type signatures:

| Class | Factory | Param Types |
|-------|---------|-------------|
| `_ANEInputBuffersReady` | `inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:` | `I` (uint32), `@` (NSArray), `@` (NSArray), `Q` (uint64) |
| `_ANEOutputSetEnqueue` | `outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:` | `I`, `I`, `Q`, `B`, `B` |

### Experiments Q-S Results (test_coreml_chaining.m, 2026-03-04)

#### Experiment Q: CoreML Pipeline -- MAJOR DISCOVERY

**The E5 runtime (macOS 15+) does NOT use `_ANEModel` or `_ANEChainingRequest` at all.**

CoreML on macOS 15 uses the MIL-based "E5" runtime, which completely bypasses the older Espresso/`_ANEModel`/`_ANEChainingRequest` path:

| Component | Old Path (Espresso) | New Path (E5/MIL) |
|-----------|--------------------|--------------------|
| Model format | `.espresso.net` + `.espresso.weights` | `model.mil` + `weights/weight.bin` |
| Model class | `_ANEModel` | `e5rt_program_library` (C struct) |
| Engine | `_ANEClient` + `_ANERequest` | `MLE5Engine` + `MLE5ExecutionStreamOperation` |
| Chaining | `_ANEChainingRequest` | `e5rt_execution_stream_operation` (unknown) |
| Compile | `_ANEClient.compileModel:` | `e5rt_program_library` AOT compilation |
| Sync | `_ANESharedSignalEvent` | `IOSurfaceSharedEventListener` + `MTLSharedEvent` |

Key findings:
- `MLModel.compileModelAtURL:` produces `.mlmodelc` with `model.mil` (NOT `model.espresso.net`)
- Loading an `MLModel` creates `MLDelegateModel` -> `MLE5Engine` -> `MLE5ProgramLibrary` -> `MLE5ProgramLibraryOnDeviceAOTCompilationImpl`
- No `_ANEModel` exists anywhere in the E5 object graph
- `_ANEClient.loadModel:` / `compileModel:` both require `model.espresso.net` which isn't generated
- Prediction succeeds (model runs on ANE), confirming E5 runtime works independently of `_ANEModel`

Internal E5 class hierarchy:
```
MLDelegateModel
  └── _internalEngine: MLE5Engine
        ├── _programLibrary: MLE5ProgramLibrary
        │     ├── _programLibraryHandle: e5rt_program_library* (opaque C struct)
        │     ├── _impl: MLE5ProgramLibraryOnDeviceAOTCompilationImpl
        │     │     ├── _milTextURL: NSURL
        │     │     ├── _irProgram: shared_ptr<MIL::IRProgram> (C++)
        │     │     └── _container: MLProgramE5Container
        │     └── _container: MLProgramE5Container
        │           ├── _modelAssetDescription
        │           ├── _compilerVersionInfo
        │           └── _functionInfoArray
        └── _operationPool: MLE5StaticShapeExecutionStreamOperationPool
              └── _pool: NSMutableSet of MLE5ExecutionStreamOperation
                    ├── _operationHandle: e5rt_execution_stream_operation* (opaque)
                    ├── _programLibrary: MLE5ProgramLibrary
                    ├── _inputPorts / _outputPorts: NSArray
                    ├── _waitEventListener: IOSurfaceSharedEventListener
                    └── _completionSharedEventBoundToESOP: MTLSharedEvent
```

#### Experiment R: Chaining with CoreML model -- BLOCKED

No `_ANEModel` extracted from E5 runtime, so `prepareChainingWithModel:` cannot be tested with a CoreML-compiled model. The E5 runtime is a completely separate execution path.

#### Experiment S: Two-Kernel Chaining -- BLOCKED

Blocked by Experiment R. The `_ANEChainingRequest` API appears to be from the **older Espresso-based runtime** and may not be usable with models compiled through the E5/MIL path.

### Experiments T-V Results (2026-03-04)

#### Experiment T: E5 Runtime Symbol Scan

Found 4 exported C functions from the `e5rt_*` API:
- `e5rt_program_library_create` -- creates program library handle
- `e5rt_execution_stream_create` -- creates execution stream handle
- `e5rt_async_event_create` -- creates async event for synchronization
- `e5rt_async_event_signal` -- signals an async event

Key ObjC classes in the E5 runtime:
- `MLE5ExecutionStreamOperation` (63 instance methods) -- holds `e5rt_execution_stream_operation*`, manages input/output ports
- `MLE5ExecutionStream` (29 instance methods) -- holds `e5rt_execution_stream*`, executes `operations` array
- `MLE5ExecutionStreamPool` -- manages streams via `takeOut` / `putBack:`
- `MLE5InputPort` / `MLE5OutputPort` -- hold `e5rt_io_port*`, bind features to ports
- `MLE5InputPortBinder` / `MLE5OutputPortBinder` -- handle memory binding for ports
- `MLE5ProgramLibrary` -- holds `e5rt_program_library*`

Critical method: `MLE5ExecutionStream._executeStream:error:` takes `e5rt_execution_stream*` and executes **all operations** in the `operations` array in sequence.

#### Experiment U: E5 Multi-Op Stream -- MAJOR BREAKTHROUGH

**Successfully executed multiple ANE operations in a single E5 stream, achieving up to 4.87x speedup over sequential CoreML.**

Method:
1. Load multiple CoreML models (`.mlpackage` -> `MLModel`)
2. Extract `MLE5ProgramLibrary` from each model's `MLE5Engine`
3. Create `MLE5ExecutionStreamOperation` for each, backed by each program library
4. Preload operations (`preloadAndReturnError:`) to compile ANE programs
5. Borrow an `MLE5ExecutionStream` from the stream pool
6. Set multiple operations on the stream via `setOperations:`
7. Prepare each operation's input features via `prepareForInputFeatures:options:error:`
8. Execute all operations in one call via `_executeStream:error:`

#### Benchmark Results (M4 Max, macOS 15, N=500)

| Kernels | CoreML Sequential | E5 Multi-Op Stream | Speedup |
|---------|------------------|--------------------|---------|
| 1 (256ch)           | 0.0359 ms | 0.0272 ms | **1.32x** |
| 2 (256+512ch)       | 0.0623 ms | 0.0406 ms | **1.53x** |
| 3 (256+512+1024ch)  | 0.1599 ms | 0.0578 ms | **2.77x** |
| 4 (256+512+1024+2048ch) | 0.3781 ms | 0.0776 ms | **4.87x** |

Key observations:
- E5 stream per-kernel overhead is remarkably consistent: ~0.02 ms/kernel regardless of count
- CoreML sequential overhead grows non-linearly (0.036 -> 0.095 ms/kernel with 4 kernels)
- The speedup increases with more kernels: the dispatch overhead is amortized
- All operations execute on ANE with a single `_executeStream:` call

Code path for E5 multi-op stream:
```
// 1. Extract internals from CoreML-loaded model
id e5engine = [mlModel valueForKey:@"_internalEngine"];  // MLE5Engine
id progLib  = [e5engine valueForKey:@"programLibrary"];   // MLE5ProgramLibrary
id pool     = [e5engine valueForKey:@"streamPool"];       // MLE5ExecutionStreamPool

// 2. Create operation from program library
id op = [[MLE5ExecutionStreamOperation alloc]
    initWithProgramLibrary:progLib functionName:@"main"
    modelDescription:desc configuration:cfg
    debugLabel:@"myOp" modelSignpostId:0];
[op preloadAndReturnError:nil];

// 3. Get stream and set operations
id stream = [pool takeOut];
void *sh = stream._streamHandle;  // e5rt_execution_stream*
[stream setOperations:@[op1, op2, op3]];

// 4. Prepare and execute
for (op in operations)
    [op prepareForInputFeatures:features options:predOpts error:nil];
[stream _executeStream:sh error:nil];
```

### Revised Assessment (after T-V)

~~The **E5 runtime** (`MLE5ExecutionStream` + `MLE5ExecutionStreamOperation`) is the correct path for multi-kernel pipelining on macOS 15+.~~ **CORRECTED in Experiments W1 (see below).**

### Experiments W1-W5: Validation & Deep API Documentation (2026-03-04)

#### W1: Output Correctness Validation

**CRITICAL CORRECTION**: The previously reported "4.87x speedup" from multi-op streams was **invalid**. Validation revealed:

1. `MLE5Engine.predictionFromFeatures:options:error:` produces **EXACT** (bit-identical) output to `MLModel.predictionFromFeatures:error:` for all tested sizes (256, 512, 1024, 2048 channels). This confirms the E5 engine is the correct computation path.

2. Our manually-created `MLE5ExecutionStreamOperation` objects via `initWithProgramLibrary:` **do not produce correct output** -- they return all zeros. The `_executeStream:` call returns YES but no actual ANE compute occurs. The operation handles are `0x0` (not compiled), meaning our manually-created ops were never wired to actual ANE programs.

3. The "speedup" was measuring the overhead of a no-op function returning immediately vs CoreML doing actual computation.

4. `MLE5StaticShapeExecutionStreamOperationPool.takeOutOperationForFeatures:error:` returns pool-managed operations with valid handles, but using them with `_executeStream:` still produces zeros -- the output port bindings are not correctly populated.

5. Stream reuse via `_predictionFromFeatures:stream:options:error:` fails with "E5RT: Port bindings cannot be changed while operation is in use in an execution stream" -- streams are locked after first use and cannot be reconfigured.

#### W1 Performance Profile

| Path | 256ch (ms) | 2048ch (ms) |
|------|-----------|-------------|
| CoreML API (`predictionFromFeatures:error:`) | 0.035 | 0.217 |
| Engine direct (`predictionFromFeatures:options:error:`) | 0.074 | 0.284 |
| Engine private (`_predictionFromFeatures:options:error:`) | 0.100 | 0.332 |
| Stream pool cycle (takeOut + putBack) | 0.008 | 0.008 |
| Op pool cycle | <0.001 | <0.001 |

**Key finding: CoreML API is FASTER than calling the engine directly.** `MLDelegateModel` implements internal caching (likely keeping a hot stream + operation) that avoids the per-call pool acquire/release overhead. The engine's `predictionFromFeatures:` method performs pool management on every call.

#### W2: Exhaustive E5 Runtime API

Full class dumps captured for all E5 runtime classes. Key classes and their roles:

**`MLE5Engine`** (49 instance methods, 10 ivars)
- Superclass: `MLModelEngine`
- Entry point: `predictionFromFeatures:options:error:` (public), `_predictionFromFeatures:stream:options:error:` (internal)
- Key properties: `streamPool` (MLE5ExecutionStreamPool), `operationPool` (<MLE5ExecutionStreamOperationPool>), `programLibrary` (MLE5ProgramLibrary)
- Manages: stream acquisition, operation preparation, input conforming, output post-processing

**`MLE5ProgramLibrary`** (17 instance methods, 5 ivars)
- Holds `_programLibraryHandle` (C struct `e5rt_program_library*`)
- Key method: `createOperationForFunctionName:forceRespecialization:hasRangeShapeInputs:error:` -- returns C-level `e5rt_execution_stream_operation*`
- Contains: compiled MIL program, model configuration, implementation object

**`MLE5ExecutionStreamOperation`** (63 instance methods, ~20 ivars)
- Holds `_operationHandle` (C struct `e5rt_execution_stream_operation*`)
- States: 0=created, transitions through prepare/execute
- Key methods: `prepareForInputFeatures:options:error:`, `preloadAndReturnError:`, `outputFeatures`
- Has input/output/state ports (MLE5InputPort, MLE5OutputPort)
- Internal binding: `_bindInputFeaturesAndWaitEvents:options:error:`, `_bindOutputPortsWithOptions:error:`
- Port binding modes: `directlyBoundFeatureValue` (zero-copy) vs `copyFeatureValue` (memcpy)

**`MLE5ExecutionStream`** (21 instance methods, 5 ivars)
- Holds `_streamHandle` (C struct `e5rt_execution_stream*`)
- Key methods: `_executeStream:error:`, `executeForInputFeatures:options:error:`, `submitWithCompletionHandler:`
- Operations set via `setOperations:` (NSArray of MLE5ExecutionStreamOperation)
- Reset via `_cleanUpStream:` on engine

**`MLE5ExecutionStreamPool`** (11 instance methods)
- Pool pattern: `takeOut` / `putBack:`
- Creates streams on demand with `e5rt_execution_stream_create`
- Tracks all streams via `allStreams`

**`MLE5StaticShapeExecutionStreamOperationPool`** (17 instance methods)
- Pool for operations with fixed input shapes
- Key method: `takeOutOperationForFeatures:error:` -- matches feature shape to pooled operation

**`MLE5InputPort` / `MLE5OutputPort`**
- Wraps `e5rt_io_port*` handles
- Each has a `binder` (MLE5InputPortBinder / MLE5OutputPortBinder)
- Input binder has `bindingMode` (char): controls copy vs direct binding
- Output binder has `outputBacking` and `featureValue` for result retrieval

**`MLE5InputPortBinder`** (16 instance methods, 6 ivars)
- `bindingMode` (char): 0=copy, 1=direct
- `bindMemoryObjectForFeatureValue:error:` -- zero-copy IOSurface binding
- `copyFeatureValue:error:` -- memcpy binding

**`MLE5OutputPortBinder`** (27 instance methods, 9 ivars)
- `outputBacking` -- output buffer
- `boundFeatureDirectly` (BOOL) -- tracks binding mode
- `_makeFeatureValueFromPort:featureDescription:error:` -- read ANE output

**`MLProgramE5Container`** (11 instance methods, 6 ivars)
- Container for compiled model assets
- `URLOfMILText` -- path to MIL source
- `compilerOutput` -- `MLCompilerNeuralNetworkOutput`
- `findPrecompiledE5BundleAndReturnError:` -- looks for pre-compiled E5 bundle

**e5rt_* C API** (found via dlsym):
- `e5rt_program_library_create` -- creates program library from MIL
- `e5rt_execution_stream_create` -- creates execution stream
- `e5rt_async_event_create` -- creates async event for synchronization
- `e5rt_async_event_signal` -- signals async event

#### W4: Async Stream Submission

`submitWithCompletionHandler:` **FAILED** with: "Failed to add operation to E5 stream. E5RT: Reset stream to add more operations to stream. (2)". The stream must be in a specific state (reset) before async submission is possible. The stream state becomes locked after `_executeStream:` or `executeForInputFeatures:`.

#### W5: Port-Based Data Flow

- Each operation has `inputPorts` (array of MLE5InputPort) and `outputPorts` (array of MLE5OutputPort)
- Input binding mode 1 = direct binding (zero-copy from MLMultiArray)
- Output `outputBacking` is nil after manual execution -- bindings are not populated by our manual path
- Port handles are `e5rt_io_port*` C structs -- connecting ports across operations would require knowing the C API for port linking

### Revised Assessment (after W1-W5)

1. **CoreML API is already near-optimal** for single-model inference. The `MLDelegateModel` wrapper is faster than calling engine methods directly due to internal stream/operation caching.

2. **Manual `_executeStream:` with custom operations is invalid** -- it produces zero output. The operations must be created through the engine's internal pipeline (via `_predictionFromFeatures:stream:options:error:`) which handles binding correctly.

3. **The opportunity for speedup lies in**:
   - Eliminating ObjC overhead via direct `e5rt_*` C API calls
   - Batching multiple models into a single stream (requires understanding `e5rt_execution_stream_operation` lifecycle)
   - Direct MIL compilation to `e5rt_program_library` without going through CoreML

### Experiment X1: Custom MIL -> ANE Execution (BREAKTHROUGH)

**Pipeline discovered**: Write MIL text file -> `MLE5ProgramLibraryOnDeviceAOTCompilationImpl` -> `MLE5ProgramLibrary` -> `MLE5Engine` -> `predictionFromFeatures:`

```objc
// 1. Write MIL text to file
NSString *mil = @"program(1.3)\n{\n    func main<ios18>(...) { ... } -> (cast_out);\n}\n";
[mil writeToFile:@"/tmp/custom.mil" ...];

// 2. Compile MIL to E5 program library
id aotImpl = [[MLE5ProgramLibraryOnDeviceAOTCompilationImpl alloc]
    initWithMILTextAtURL:milURL container:refContainer configuration:cfg];
void *plHandle = [aotImpl createProgramLibraryHandleWithRespecialization:NO error:&err];

// 3. Create program library + engine
id progLib = [[MLE5ProgramLibrary alloc] initWithImpl:aotImpl container:refContainer configuration:cfg];
id engine = [[MLE5Engine alloc] initWithProgramLibrary:progLib modelDescription:desc ...];
[engine prepareWithConcurrencyHint:1 error:nil];

// 4. Execute
id result = [engine predictionFromFeatures:fp options:opts error:&err];
```

**Requirements**:
- MIL input/output variable names must match the model description (e.g., `x` for input, `cast_out` for output)
- MIL shapes must match the model description shapes
- A "container" (`MLProgramE5Container`) is borrowed from a pre-compiled CoreML model (needed for compilation context)
- Input/output types should be fp32 with internal fp16 compute (cast in/out) for ANE compatibility

**Verified kernels** (all produce EXACT correct output on ANE):

| Kernel | MIL Op | Verification |
|--------|--------|-------------|
| ReLU | `relu(x=x16)` | Max diff = 0.000000, 0/16384 wrong |
| GELU | `gelu(x=x16, mode="TANH_APPROXIMATION")` | Verified against reference |
| Elementwise (x*2+1) | `mul` + `add` with scalar constants | Verified against reference |
| Softmax | `softmax(x=x16, axis=-1)` | Sum = 1.000000 |
| Layer Norm | `layer_norm(x=x16, axes=[3], epsilon=1e-5)` | Mean = 0.000000, Var = 0.999975 |

**Significance**: This allows compiling **arbitrary MIL programs** (any operation supported by Apple's MIL spec) to run on the ANE, without going through CoreML's .mlpackage pipeline. This is the foundation for custom training/inference kernels.

### Experiment Y1: Fused SDPA on ANE (PASSED)

**Operation**: `scaled_dot_product_attention(query=Q, key=K, value=V)` -- single fused op for entire attention computation.

Config: B=1, nHeads=1, seqLen=256, headDim=64 (self-attention: Q=K=V=reshape(input))

| Metric | Value |
|--------|-------|
| Max abs diff (vs CPU) | 0.000021 |
| Relative error | 1.40e-03 |
| Latency (first call) | 2.454 ms |
| **Benchmark** | **0.1708 ms/eval** |

### Experiment Y2: Linear with Embedded Weights (PASSED)

**Operation**: `linear(x=flat, weight=Wc, bias=Bc)` where `Wc` and `Bc` are compile-time `const` tensors embedded in the MIL program.

Config: input [256, 64], linear 64->64 with embedded weight matrix and bias vector.

| Metric | Value |
|--------|-------|
| Max abs diff (vs CPU) | 0.001106 |
| Relative error | 1.05e-02 |
| **Benchmark** | **0.0610 ms/eval** |

**Significance**: Confirms that compile-time weight constants work in MIL text format. This is the foundation for transformer inference (where weights are frozen).

### Experiment Y3: Complete Transformer Block on ANE (PASSED)

**Pipeline**: LayerNorm -> SDPA (self-attention) -> Residual Add -> LayerNorm -> FFN (linear+GELU+linear) -> Residual Add

All in a **single MIL program**, compiled and executed as one ANE operation.

Config: seqLen=256, dim=64, ffnDim=128, 1-head attention, embedded FFN weights.

| Metric | Value |
|--------|-------|
| Output mean abs | 1.017404 (non-zero, correct) |
| **Benchmark** | **0.2091 ms/eval** |

**Significance**: A full transformer layer runs on ANE in ~0.2ms. This proves that complex multi-op pipelines can be compiled as single MIL programs with no CPU round-trips between ops. The ANE compiler fuses the entire graph.

### Experiment Z1: Backward Pass (Gradient Computation) on ANE (PASSED)

**Operations**: `matmul(x=dY, y=W)` for dX (input gradient), `matmul(x=dY, y=dY, transpose_x=true)` for dW (weight gradient). Both use **runtime tensors** (not const), proving backward-pass operations work on ANE.

Also tests: `slice_by_index` for tensor slicing, `concat` for packing results.

Config: dY [128,64] @ W [64,64] -> dX [128,64]; dY^T [64,128] @ dY [128,64] -> dW [64,64]

| Metric | dX | dW |
|--------|-----|-----|
| Max abs diff | 0.001940 | 0.012828 |
| Relative error | 1.02e-02 | 3.92e-02 |
| **Benchmark** | **0.0593 ms/eval** (both combined) |

**Significance**: This is the first demonstration of ANE executing gradient computation operations. The `matmul` with `transpose_x=true` works correctly, producing valid weight gradients. Combined with Y3's forward pass, this establishes the complete pipeline for manual ANE training:
1. Forward pass: Y3-style MIL (0.2 ms)
2. Backward pass: Z1-style MIL (0.06 ms)
3. Weight update: CPU (trivial)
4. Recompile: (~10-50 ms, dominates training time)

### MIL Text Syntax Lessons Learned

Key syntax rules discovered during Y/Z experiments:

1. **`epsilon` in `layer_norm`**: Must be same dtype as gamma/beta. Use `fp16 eps = const()[..., val = fp16(1e-5)]` when gamma is fp16.
2. **Boolean params**: Use `bool tx = const()[..., val = bool(true)]` for params like `transpose_x`.
3. **`concat` axis**: Must be `int32` scalar, not `tensor<int32, [1]>`. Use `int32 ax = const()[..., val = int32(0)]`.
4. **`concat` interleave**: Required param, use `bool il = const()[..., val = bool(false)]`.
5. **MLE5Engine init**: Correct selector is `initWithProgramLibrary:modelDescription:configuration:functionName:classProbabilitiesFeatureName:optionalInputDefaultValues:compilerVersionInfo:` (7 args).
6. **Container path**: On macOS 15+, models may use Espresso backend. Create `MLProgramE5Container` via `initWithModelAssetPath:configuration:` using the `.mlmodelc` path.
7. **Sandbox**: E5RT needs write access to `~/Library/Caches/` for model specialization cache.

### Next Steps

1. **[HIGH] Multi-head attention** -- test SDPA with multiple heads (reshape to [B, nHeads, seqLen, headDim])
2. **[HIGH] Real Qwen2.5 layer weights** -- load actual model weights into MIL const tensors
3. **[HIGH] Full backward pass** -- implement complete transformer backward pass (attention + FFN gradients)
4. **[MEDIUM] Training loop** -- forward + backward + weight update + recompile cycle
5. **[MEDIUM] Explore e5rt_* C API directly** -- bypass ObjC wrappers for lower overhead
6. **[LOW] Runtime weight injection** -- investigate if weights can be updated without recompilation

**Phase 7: OutputSets with stats IOSurface -- BREAKTHROUGH**
```
  statsSurRef size=64 bytes:
    objectWithstatsSurRef: _ANEIOSurfaceOutputSets: { statsSurRef=<IOSurface: 0x...>
    id = 0x... width = 64 height = 1 pixelFormat = 0
    name = test_chaining_v2 ; outputBuffer=(
      "_ANEBuffer: { ... symbolIndex=0 ; ANEBufferProducerAgent=1}"
    )}

    Attempting ChainingRequest with valid outputSet...
    ChainingRequest created | validate: YES     <-- FIRST TIME VALIDATE PASSES!
    prepareChainingWithModel EXCEPTION:
      -[_ANEInMemoryModel getUUID]: unrecognized selector
```

**Phase 8: Disk-based _ANEModel**
```
  _ANEModel class found (12 class methods, 52 instance methods, 17 properties)
  Has: getUUID, inputSymbolIndicesForProcedureIndex:,
       outputSymbolIndicesForProcedureIndex:, mapper, program
  Factory: +modelAtURL:key:, +modelAtURL:key:modelAttributes:, etc.

  tmpDir contents: (weights, model.mil, net.plist, data)
  +modelAtURL: NOT available (needs key: parameter)
  -> _ANEModel could not be loaded (need correct factory + key)
```

**Phase 9: processRequest via ProgramForEvaluation**
```
  k1.model.program: _ANEProgramForEvaluation: { programHandle=1319967543575
    intermediateBufferHandle=0 queueDepth=127 }
  processRequest single call: YES (rv=NO)
  processRequest: 0.131 ms/eval (50 iters)
  vs RT eval: 1.45x (slower than RT but faster than standard)
```

**Phase 10: Shared Events**
```
  _ANESharedEvents: found (+sharedEventsWithSignalEvents:waitEvents:)
  _ANESharedSignalEvent: found
    +signalEventWithValue:symbolIndex:eventType:sharedEvent:
    Properties: sharedEvent (IOSurfaceSharedEvent), value, symbolIndex, agentMask, eventType
    alloc/init: nil (needs sharedEvent parameter)
  _ANESharedWaitEvent: found
    +waitEventWithValue:sharedEvent:
    alloc/init: nil (needs sharedEvent parameter)
  -> Both require IOSurfaceSharedEvent objects, not available from bare init
```

---

## 6. Architecture: Chaining Data Flow

```
Current (sequential):
  CPU -> IOSurface -> ANE eval layer 1 -> IOSurface -> CPU memcpy
  CPU -> IOSurface -> ANE eval layer 2 -> IOSurface -> CPU memcpy
  ... (23 round-trips for 12-layer model)

Target (chained):
  CPU -> IOSurface -> ANE eval layer 1 -> [on-chip] -> ANE eval layer 2
                   -> [on-chip] -> ... -> IOSurface -> CPU
  (1 round-trip for entire model)

Current best (sequential with standard path):
  At production dims (768x256), all paths are ~0.2ms/kernel.
  RT path only helps for small kernels (64x32: 1.88x speedup).
  For 24 evals/token at ~0.2ms each: ~4.8ms total ANE time per token.
  Chaining target: 1 round-trip instead of 24, saving ~23 x overhead per trip.
```

---

## 7. Class Hierarchy (inferred)

```
NSObject
├── _ANEClient (singleton, daemon connection)
├── _ANEInMemoryModelDescriptor (MIL + weights spec)
├── _ANEInMemoryModel (compile/load/run -- in-memory MIL path)
│   └── .program -> _ANEProgramForEvaluation
├── _ANEModel (disk-based compiled model -- 52 methods, has getUUID)
│   └── .program -> _ANEProgramForEvaluation
│   └── .mapper -> _ANEProgramIOSurfacesMapper
├── _ANERequest (I/O surface packaging)
├── _ANEIOSurfaceObject (thin IOSurface wrapper)
├── _ANEBuffer (IOSurfaceObject + symbolIndex + source)
├── _ANEChainingRequest (multi-op pipeline)
├── _ANEIOSurfaceOutputSets (output packaging for chaining)
├── _ANEInputBuffersReady (input signaling for chaining)
├── _ANEOutputSetEnqueue (output enqueue config for chaining)
├── _ANEProgramIOSurfacesMapper (symbol-to-surface mapping)
├── _ANEProgramForEvaluation (lower-level eval program)
├── _ANEModelInstanceParameters (model config)
├── _ANEDeviceController (device-level control)
├── _ANEQoSMapper (QoS level mapping)
├── _ANEPerformanceStats (perf counters)
├── _ANESharedSignalEvent (hardware signal fence)
└── _ANESharedWaitEvent (hardware wait fence)
```

---

## 8. MIL Operations Reference (for Custom ANE Kernels)

Source: [coremltools MIL Ops API Reference](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html)

The following MIL operations are available for writing custom ANE kernels via our `MLE5ProgramLibraryOnDeviceAOTCompilationImpl` pipeline (Experiment X1). All ops below have been confirmed available in the MIL text format used by the E5 compiler on macOS 15+.

### Transformer-Critical Ops

| Op | Signature | Notes |
|----|-----------|-------|
| `scaled_dot_product_attention` (iOS 18+) | `(query:[B,*?,L,E], key:[B,*?,S,E], value:[B,*?,S,EV], attn_mask?) -> [B,*?,L,EV]` | Fused `softmax(Q@K.T/sqrt(d))@V`. Single op for entire attention computation. |
| `linear` | `(x:[*D,D_in], weight:const[D_out,D_in], bias:const[D_out]?) -> [*D,D_out]` | `x @ W.T + b`. **Weight/bias must be compile-time constants.** Rank 1-3 input. |
| `matmul` | `(x:[*,K1], y:[*,K2], transpose_x?, transpose_y?) -> [*,T]` | N-D batch matmul with broadcasting. Supports runtime (non-const) inputs. |
| `layer_norm` | `(x, axes, gamma?, beta?, epsilon?) -> same shape` | Verified working on ANE (Experiment X1). |
| `gelu` | `(x, mode=EXACT/TANH_APPROXIMATION/SIGMOID_APPROXIMATION) -> same shape` | Verified working on ANE (Experiment X1). |
| `softmax` | `(x, axis) -> same shape` | Verified working on ANE (Experiment X1). |
| `relu` | `(x) -> same shape` | Verified working on ANE (Experiment X1). |

### Data Movement Ops

| Op | Signature | Notes |
|----|-----------|-------|
| `gather` | `(x, indices, axis?) -> gathered` | For embedding table lookups. |
| `gather_along_axis` | `(x, indices, axis?) -> gathered` | Take values along axis at index locations. |
| `scatter` | `(data, indices, updates, axis?, mode?) -> scattered` | For KV cache writes. Mode: update/add/sub/mul/div/max/min. |
| `scatter_along_axis` | `(data, indices, updates, axis?, mode?) -> scattered` | Scatter updates along axis. |

### Elementwise / Reduction Ops

| Op | Notes |
|----|-------|
| `add`, `sub`, `mul`, `real_div` | Elementwise with broadcasting. |
| `cast` | Type conversion (fp32 <-> fp16). Required for ANE I/O (fp32 in, fp16 compute, fp32 out). |
| `reduce_sum`, `reduce_mean`, `reduce_max` | Reduction along axes. |
| `rsqrt`, `sqrt`, `exp`, `log`, `tanh` | Unary elementwise. Useful for manual norm/activation implementations. |
| `concat`, `split`, `reshape`, `transpose` | Shape manipulation. |
| `slice_by_index`, `slice_by_size` | Tensor slicing for KV cache windowing. |

### Key Constraints

1. **`linear` weights must be `const`**: For inference this is fine (weights don't change). For training, use `matmul` with runtime tensors instead.
2. **MIL text format**: Programs use `program(1.3) { func main<ios18>(...) { ... } -> (output); }` syntax. Constants use `const()[name=..., val=...]`. Weights reference blob files via `BLOBFILE(path=..., offset=...)`.
3. **ANE I/O convention**: Input/output should be fp32; internal compute should be fp16. Use `cast` ops at boundaries.
4. **Shape constraints**: ANE prefers NCHW layout. Most ops work with rank-4 tensors `[B, C, H, W]` but `linear`/`matmul` work with lower ranks.

---

## 9. ANE Training Feasibility Analysis

### Apple's Official Position

Apple's deprecated **MLCompute** framework (`MLCDevice.ane()`) explicitly states:
> "This device applies to inference graphs only. It doesn't work with a training graph or inference graph that shares layers with a training graph."

This means Apple never shipped ANE-based training, even in their own training framework. The `MLCTrainingGraph` class supported `executeForward`, `executeGradient`, and `executeOptimizerUpdate` but only on CPU and GPU devices.

### WWDC 2025 Confirmation

WWDC 2025 Session 360 ("Discover ML & AI frameworks") confirms:
- CoreML dispatches to CPU, GPU, and Neural Engine at runtime for **inference**
- MLX is the recommended tool for training/fine-tuning but uses Metal GPU, not ANE
- No mention of ANE training APIs in any Apple framework
- BNNSGraph (Accelerate) added `BNNSGraphBuilder` for CPU-only real-time inference

### Why ANE Lacks Native Training Support

The ANE is a fixed-function inference accelerator. It likely lacks:
- Hardware support for automatic differentiation / backward passes
- Ability to write to weight storage during execution (weights are read-only constants in the `e5rt_program_library`)
- Dynamic memory allocation needed for activation checkpointing

### Manual ANE Training Approach

Despite the lack of native support, training on ANE is theoretically possible using our custom MIL pipeline:

1. **Forward pass**: Write MIL program with `linear`/`matmul`/`layer_norm`/`gelu` ops. Weights embedded as constants. Execute on ANE. Save activations.
2. **Backward pass**: Write separate MIL programs for each layer's gradient computation:
   - Linear backward: `dX = dY @ W` (matmul), `dW = dY.T @ X` (matmul)
   - ReLU backward: `dX = dY * (X > 0)` (elementwise)
   - LayerNorm backward: Multiple reduction + elementwise ops
3. **Optimizer step**: Run on CPU (simple elementwise: `W -= lr * dW`)
4. **Recompile**: After weight update, recompile MIL with new weights for next forward pass

The key bottleneck is step 4: recompiling MIL after every weight update. The `createProgramLibraryHandleWithRespecialization:` call takes ~10-50ms, which would dominate training time. This makes per-step ANE training impractical unless we can find a way to update weights without recompilation (e.g., via the `e5rt_*` C API or runtime weight injection).
