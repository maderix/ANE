// ane_bridge.h — C-callable bridge to ANE private APIs for Python ctypes
// Wraps _ANEInMemoryModel via private AppleNeuralEngine.framework

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

// Compile a MIL program with weight blobs into an ANE kernel
// mil_text: UTF-8 MIL program text
// mil_len: length of MIL text
// weight_data: raw weight blob (can be NULL)
// weight_len: length of weight blob
// n_inputs: number of input tensors
// input_sizes: array of byte sizes for each input
// n_outputs: number of output tensors
// output_sizes: array of byte sizes for each output
// Returns kernel handle or NULL on failure
ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes);

// Compile with multiple named weight files (for transformer kernels)
// weight_names: array of weight file paths (e.g. "@model_path/weights/wq.bin")
// weight_datas: array of weight data pointers
// weight_lens: array of weight data lengths
// n_weights: number of weight files
ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes);

// Evaluate (run) a compiled kernel on ANE
// Returns true on success
bool ane_bridge_eval(ANEKernelHandle *kernel);

// Write data to kernel input tensor
void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes);

// Read data from kernel output tensor
void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes);

// Free a compiled kernel and all associated resources
void ane_bridge_free(ANEKernelHandle *kernel);

// Get compile count (for exec() restart budgeting)
int ane_bridge_get_compile_count(void);

// Reset compile count
void ane_bridge_reset_compile_count(void);

// Build a weight blob in ANE format (128-byte header + fp16 data)
// src: float32 weights [rows x cols]
// Returns allocated buffer and sets out_len. Caller must free().
uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len);

// Build a transposed weight blob in ANE format
uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
                                                   size_t *out_len);

// Build an int8 weight blob in ANE format (64-byte header + int8 data per chunk)
// src: int8 weights [rows x cols], scale: dequantization scale, zero_point: int8 zero
// For use with constexpr_affine_dequantize in MIL
// Returns allocated buffer and sets out_len. Caller must free().
uint8_t *ane_bridge_build_weight_blob_int8(const int8_t *src, int rows, int cols,
                                            size_t *out_len);

// Quantize float32 weights to int8 and build ANE blob in one step
// Computes per-channel (axis=0) scale = max(abs(row)) / 127
// Returns allocated buffer, sets out_len and out_scale. Caller must free().
uint8_t *ane_bridge_build_weight_blob_quantized(const float *src, int rows, int cols,
                                                 float *out_scale, size_t *out_len);

// Free a blob allocated by ane_bridge_build_weight_blob*
void ane_bridge_free_blob(void *ptr);

#ifdef __cplusplus
}
#endif

#endif // ANE_BRIDGE_H
