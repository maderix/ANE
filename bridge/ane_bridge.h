// ane_bridge.h — C-callable bridge to ANE private APIs for Python ctypes
// Wraps _ANEInMemoryModel via private AppleNeuralEngine.framework
//
// Two compilation modes:
//
//   BLOBFILE (upstream compatible):
//     ane_bridge_compile() / ane_bridge_compile_multi_weights()
//     Weights compiled into MIL as constants. Requires recompile when weights
//     change — hits ANE compile limit (~119), needs exec() restart per batch.
//
//   Dynamic IOSurface (our approach):
//     ane_bridge_compile_dyn()
//     Weights declared as runtime tensor function parameters. Compile ONCE at
//     startup, update weights via ane_bridge_write_weight() (0.002ms per call).
//     No exec() restart, no compile limit during training.
//
// Extras (our additions):
//   ane_bridge_begin/end_realtime() — 90.6% p99 jitter reduction
//   ane_bridge_copy_io()            — direct IOSurface-to-IOSurface, no CPU
//   Compile cache                   — ~700ms vs ~3800ms on cache hit

#ifndef ANE_BRIDGE_H
#define ANE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque kernel handle
typedef struct ANEKernelHandle ANEKernelHandle;

// Initialize ANE runtime (load private framework, resolve classes)
// Returns 0 on success, -1 on failure
int ane_bridge_init(void);

// ---------------------------------------------------------------------------
// BLOBFILE compile (upstream compatible)
// Weights compiled into the MIL program as constants.
// ---------------------------------------------------------------------------

// Compile a MIL program with a single weight blob
ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes);

// Compile with multiple named weight files
ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// ---------------------------------------------------------------------------
// Dynamic weight compile (our approach — compile once, update per Adam step)
// Weights declared as runtime tensor function parameters backed by IOSurfaces.
//
//   n_inputs:      number of activation input tensors
//   input_sizes:   byte sizes (fp16) for each activation input
//   n_weights:     number of dynamic weight tensors
//   weight_sizes:  byte sizes (fp16) for each weight IOSurface
//   output_size:   byte size (fp16) of the single output tensor
//
// MIL function signature must match: func main<ios18>(x0, x1, ..., w0, w1, ...)
// where activation inputs come first, weight inputs follow.
// ---------------------------------------------------------------------------
ANEKernelHandle *ane_bridge_compile_dyn(
    const char *mil_text, size_t mil_len,
    int n_inputs, const size_t *input_sizes,
    int n_weights, const size_t *weight_sizes,
    size_t output_size);

// ---------------------------------------------------------------------------
// Eval and I/O
// ---------------------------------------------------------------------------

// Evaluate (run) a compiled kernel on ANE
bool ane_bridge_eval(ANEKernelHandle *kernel);

// Write data to activation input tensor (fp16 or raw bytes)
void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes);

// Read data from output tensor (fp16 or raw bytes)
void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes);

// ---------------------------------------------------------------------------
// Dynamic weight I/O (our approach)
// ---------------------------------------------------------------------------

// Write fp16 data directly to weight IOSurface (~0.002ms per call)
// idx: weight index (0..n_weights-1)
void ane_bridge_write_weight(ANEKernelHandle *kernel, int idx,
                              const void *fp16_data, size_t bytes);

// Write fp32 data to weight IOSurface with automatic fp32→fp16 conversion
// count: number of float elements (bytes = count * 2 fp16)
void ane_bridge_write_weight_f32(ANEKernelHandle *kernel, int idx,
                                  const float *fp32_data, size_t count);

// ---------------------------------------------------------------------------
// Direct IOSurface copy — no CPU round-trip between chained kernels
// Copies src kernel's output[src_out_idx] → dst kernel's input[dst_in_idx]
// Zero-copy: just memcpy between IOSurface base addresses
// ---------------------------------------------------------------------------
void ane_bridge_copy_io(ANEKernelHandle *src, int src_out_idx,
                         ANEKernelHandle *dst, int dst_in_idx);

// ---------------------------------------------------------------------------
// Real-time task — 90.6% p99 jitter reduction
// Wrap a sequence of evals with begin/end to prevent ANE scheduler preemption.
// Proven: plain p99=35.2ms → with RT task p99=3.3ms
// Requires at least one kernel to have been compiled and loaded.
// ---------------------------------------------------------------------------
void ane_bridge_begin_realtime(void);
void ane_bridge_end_realtime(void);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void ane_bridge_free(ANEKernelHandle *kernel);

// Compile count (useful for tracking exec() restart budget in BLOBFILE mode)
int  ane_bridge_get_compile_count(void);
void ane_bridge_reset_compile_count(void);

// ---------------------------------------------------------------------------
// Weight blob helpers (BLOBFILE mode)
// Builds the 128-byte ANE blob header + fp16 weights for use with
// ane_bridge_compile / ane_bridge_compile_multi_weights.
// ---------------------------------------------------------------------------
uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len);
uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
                                                   size_t *out_len);
void ane_bridge_free_blob(void *ptr);

#ifdef __cplusplus
}
#endif

#endif // ANE_BRIDGE_H
