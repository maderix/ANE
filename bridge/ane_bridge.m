// ane_bridge.m — Objective-C implementation of ANE bridge for Python ctypes
// Wraps _ANEInMemoryModel private APIs into C-callable functions
//
// Two modes: BLOBFILE (upstream compatible) and dynamic IOSurface (our approach).
// See ane_bridge.h for full API documentation.

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <arm_neon.h>
#include "ane_bridge.h"

// --- Private class references ---
static Class g_ANEDesc  = nil;
static Class g_ANEInMem = nil;
static Class g_ANEReq   = nil;
static Class g_ANEIO    = nil;
static bool  g_initialized = false;
static int   g_compile_count = 0;

// _ANEClient for beginRealTimeTask — retrieved from first loaded model
static id g_rt_client = nil;

// --- Kernel handle ---
struct ANEKernelHandle {
    id model;                   // _ANEInMemoryModel
    IOSurfaceRef *ioInputs;     // activation input surfaces
    IOSurfaceRef *ioOutputs;    // output surfaces
    IOSurfaceRef *ioWeights;    // dynamic weight surfaces (NULL for BLOBFILE mode)
    id request;                 // _ANERequest
    NSString *tmpDir;
    int nInputs, nOutputs, nWeights;
    size_t *inputBytes;
    size_t *outputBytes;
    size_t *weightBytes;
};

// --- Helpers ---

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:          @(bytes),
        (id)kIOSurfaceHeight:         @1,
        (id)kIOSurfaceBytesPerElement:@1,
        (id)kIOSurfaceBytesPerRow:    @(bytes),
        (id)kIOSurfaceAllocSize:      @(bytes),
        (id)kIOSurfacePixelFormat:    @0
    });
}

static id wrap_surface(IOSurfaceRef s) {
    return ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
        g_ANEIO, @selector(objectWithIOSurface:), s);
}

// Compile cache: ~/.ane_cache/<hexId>/
// Saves ~3100ms on cache hit (700ms vs 3800ms for 74 kernels).
static BOOL try_cache_restore(id mdl, NSString *td, NSFileManager *fm) {
    NSString *hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *cacheDir = [NSHomeDirectory() stringByAppendingPathComponent:
        [@".ane_cache/" stringByAppendingString:hx]];
    NSString *cachedPlist = [cacheDir stringByAppendingPathComponent:@"net.plist"];
    if (![fm fileExistsAtPath:cachedPlist]) return NO;
    [fm copyItemAtPath:cachedPlist toPath:[td stringByAppendingPathComponent:@"net.plist"] error:nil];
    // BLOBFILE models also produce a `data` file; dynamic-weight models do not
    NSString *cachedData = [cacheDir stringByAppendingPathComponent:@"data"];
    if ([fm fileExistsAtPath:cachedData])
        [fm copyItemAtPath:cachedData toPath:[td stringByAppendingPathComponent:@"data"] error:nil];
    return YES;
}

static void save_to_cache(id mdl, NSString *td, NSFileManager *fm) {
    NSString *hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *cacheDir = [NSHomeDirectory() stringByAppendingPathComponent:
        [@".ane_cache/" stringByAppendingString:hx]];
    [fm createDirectoryAtPath:cacheDir withIntermediateDirectories:YES attributes:nil error:nil];
    [fm copyItemAtPath:[td stringByAppendingPathComponent:@"net.plist"]
                toPath:[cacheDir stringByAppendingPathComponent:@"net.plist"] error:nil];
    // Copy data only if present (BLOBFILE models)
    NSString *tdData = [td stringByAppendingPathComponent:@"data"];
    if ([fm fileExistsAtPath:tdData])
        [fm copyItemAtPath:tdData toPath:[cacheDir stringByAppendingPathComponent:@"data"] error:nil];
}

static BOOL compile_and_load(id mdl, NSString *td, NSFileManager *fm) {
    NSError *e = nil;
    BOOL fromCache = try_cache_restore(mdl, td, fm);

    if (!fromCache) {
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ane_bridge: compile failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            return NO;
        }
        save_to_cache(mdl, td, fm);
        g_compile_count++;
    }

    e = nil;
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!loaded) {
        usleep(100000);
        e = nil;
        loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    }
    if (!loaded) {
        fprintf(stderr, "ane_bridge: load failed: %s\n",
                e ? [[e description] UTF8String] : "unknown");
        return NO;
    }

    // Cache _ANEClient for real-time task API
    if (!g_rt_client) {
        Ivar iv = class_getInstanceVariable(object_getClass(mdl), "_sharedConnection");
        if (iv) g_rt_client = object_getIvar(mdl, iv);
    }

    return YES;
}

// Build _ANERequest from arrays of input and output surfaces
static id build_request(IOSurfaceRef *inputs, int nIn,
                         IOSurfaceRef *outputs, int nOut) {
    NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:nIn];
    NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:nIn];
    for (int i = 0; i < nIn; i++) {
        [wIns addObject:wrap_surface(inputs[i])];
        [iIdx addObject:@(i)];
    }
    NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:nOut];
    NSMutableArray *oIdx  = [NSMutableArray arrayWithCapacity:nOut];
    for (int i = 0; i < nOut; i++) {
        [wOuts addObject:wrap_surface(outputs[i])];
        [oIdx  addObject:@(i)];
    }
    return ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
        g_ANEReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        wIns, iIdx, wOuts, oIdx, nil, nil, @0);
}

// NEON fp32 → fp16 conversion
static void cvt_f32_f16(const float *src, uint16_t *dst, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t f = vld1q_f32(src + i);
        vst1_u16(dst + i, vreinterpret_u16_f16(vcvt_f16_f32(f)));
    }
    for (; i < count; i++)
        dst[i] = vreinterpret_u16_f16(vcvt_f16_f32(vdupq_n_f32(src[i])))[0];
}

// --- Public API ---

int ane_bridge_init(void) {
    if (g_initialized) return 0;
    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!handle) { fprintf(stderr, "ane_bridge: failed to load ANE framework\n"); return -1; }

    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        fprintf(stderr, "ane_bridge: failed to resolve ANE private classes\n");
        return -1;
    }
    g_initialized = true;
    return 0;
}

// ---------------------------------------------------------------------------
// BLOBFILE compile (upstream compatible)
// ---------------------------------------------------------------------------

ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes)
{
    @autoreleasepool {
        if (!g_initialized) { fprintf(stderr, "ane_bridge: not initialized\n"); return NULL; }

        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            NSData   *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            wdict[name] = @{@"offset": @0, @"data": data};
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict.count > 0 ? wdict : nil, nil);
        if (!desc) { fprintf(stderr, "ane_bridge: modelWithMILText failed\n"); return NULL; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);

        NSString *hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        for (int i = 0; i < n_weights; i++) {
            NSString *rel = [NSString stringWithUTF8String:weight_names[i]];
            if ([rel hasPrefix:@"@model_path/"]) rel = [rel substringFromIndex:12];
            NSString *full = [td stringByAppendingPathComponent:rel];
            [fm createDirectoryAtPath:[full stringByDeletingLastPathComponent]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [[NSData dataWithBytes:weight_datas[i] length:weight_lens[i]] writeToFile:full atomically:YES];
        }

        if (!compile_and_load(mdl, td, fm)) { [fm removeItemAtPath:td error:nil]; return NULL; }

        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        k->model      = mdl;
        k->tmpDir     = td;
        k->nInputs    = n_inputs;
        k->nOutputs   = n_outputs;
        k->nWeights   = 0;
        k->ioWeights  = NULL;
        k->weightBytes = NULL;
        k->inputBytes  = (size_t *)malloc(n_inputs  * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        memcpy(k->inputBytes,  input_sizes,  n_inputs  * sizeof(size_t));
        memcpy(k->outputBytes, output_sizes, n_outputs * sizeof(size_t));

        k->ioInputs  = (IOSurfaceRef *)malloc(n_inputs  * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(n_outputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs;  i++) k->ioInputs[i]  = make_surface(input_sizes[i]);
        for (int i = 0; i < n_outputs; i++) k->ioOutputs[i] = make_surface(output_sizes[i]);

        k->request = build_request(k->ioInputs, n_inputs, k->ioOutputs, n_outputs);
        return k;
    }
}

ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes) {
    if (weight_data && weight_len > 0) {
        const char *name = "@model_path/weights/weight.bin";
        return ane_bridge_compile_multi_weights(mil_text, mil_len,
            &name, &weight_data, &weight_len, 1,
            n_inputs, input_sizes, n_outputs, output_sizes);
    }
    return ane_bridge_compile_multi_weights(mil_text, mil_len,
        NULL, NULL, NULL, 0,
        n_inputs, input_sizes, n_outputs, output_sizes);
}

// ---------------------------------------------------------------------------
// Dynamic IOSurface compile (our approach — compile ONCE, write per Adam step)
//
// MIL program must declare weights as function parameters after activation inputs:
//   func main<ios18>(tensor<fp16,...> x0, tensor<fp16,...> w0, tensor<fp16,...> w1) { ... }
//
// The _ANERequest bundles them as: inputs=[x0, w0, w1, ...], outputs=[out]
// Weight IOSurfaces persist between evals — update via ane_bridge_write_weight().
// ---------------------------------------------------------------------------
ANEKernelHandle *ane_bridge_compile_dyn(
    const char *mil_text, size_t mil_len,
    int n_inputs, const size_t *input_sizes,
    int n_weights, const size_t *weight_sizes,
    size_t output_size)
{
    @autoreleasepool {
        if (!g_initialized) { fprintf(stderr, "ane_bridge: not initialized\n"); return NULL; }

        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];

        // Dynamic weights: pass empty weight dict (no BLOBFILE)
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);
        if (!desc) { fprintf(stderr, "ane_bridge: modelWithMILText failed\n"); return NULL; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);

        NSString *hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        if (!compile_and_load(mdl, td, fm)) { [fm removeItemAtPath:td error:nil]; return NULL; }

        // Allocate kernel handle
        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        k->model    = mdl;
        k->tmpDir   = td;
        k->nInputs  = n_inputs;
        k->nOutputs = 1;
        k->nWeights = n_weights;

        k->inputBytes  = (size_t *)malloc(n_inputs  * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(1         * sizeof(size_t));
        k->weightBytes = (size_t *)malloc(n_weights * sizeof(size_t));
        memcpy(k->inputBytes,  input_sizes,  n_inputs  * sizeof(size_t));
        memcpy(k->weightBytes, weight_sizes, n_weights * sizeof(size_t));
        k->outputBytes[0] = output_size;

        // Create IOSurfaces for activations, weights, and output
        k->ioInputs  = (IOSurfaceRef *)malloc(n_inputs  * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(1         * sizeof(IOSurfaceRef));
        k->ioWeights = (IOSurfaceRef *)malloc(n_weights * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs;  i++) k->ioInputs[i]  = make_surface(input_sizes[i]);
        for (int i = 0; i < n_weights; i++) k->ioWeights[i] = make_surface(weight_sizes[i]);
        k->ioOutputs[0] = make_surface(output_size);

        // Build request: inputs = [x0, x1, ..., w0, w1, ...], outputs = [out]
        // Weights follow activation inputs at higher indices — matches MIL param order.
        int total_inputs = n_inputs + n_weights;
        NSMutableArray *allInputs = [NSMutableArray arrayWithCapacity:total_inputs];
        NSMutableArray *allIdx    = [NSMutableArray arrayWithCapacity:total_inputs];
        for (int i = 0; i < n_inputs;  i++) { [allInputs addObject:wrap_surface(k->ioInputs[i])];  [allIdx addObject:@(i)]; }
        for (int i = 0; i < n_weights; i++) { [allInputs addObject:wrap_surface(k->ioWeights[i])]; [allIdx addObject:@(n_inputs + i)]; }

        id wOut = wrap_surface(k->ioOutputs[0]);
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            allInputs, allIdx, @[wOut], @[@0], nil, nil, @0);

        return k;
    }
}

// ---------------------------------------------------------------------------
// Eval and I/O
// ---------------------------------------------------------------------------

bool ane_bridge_eval(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel || !kernel->model) return false;
        NSError *e = nil;
        return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            kernel->model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, kernel->request, &e);
    }
}

void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes) {
    if (!kernel || idx < 0 || idx >= kernel->nInputs) return;
    IOSurfaceLock(kernel->ioInputs[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(kernel->ioInputs[idx]), data, bytes);
    IOSurfaceUnlock(kernel->ioInputs[idx], 0, NULL);
}

void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes) {
    if (!kernel || idx < 0 || idx >= kernel->nOutputs) return;
    IOSurfaceLock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(kernel->ioOutputs[idx]), bytes);
    IOSurfaceUnlock(kernel->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
}

// ---------------------------------------------------------------------------
// Dynamic weight I/O
// ---------------------------------------------------------------------------

void ane_bridge_write_weight(ANEKernelHandle *kernel, int idx,
                              const void *fp16_data, size_t bytes) {
    if (!kernel || !kernel->ioWeights || idx < 0 || idx >= kernel->nWeights) return;
    IOSurfaceLock(kernel->ioWeights[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(kernel->ioWeights[idx]), fp16_data, bytes);
    IOSurfaceUnlock(kernel->ioWeights[idx], 0, NULL);
}

void ane_bridge_write_weight_f32(ANEKernelHandle *kernel, int idx,
                                  const float *fp32_data, size_t count) {
    if (!kernel || !kernel->ioWeights || idx < 0 || idx >= kernel->nWeights) return;
    IOSurfaceLock(kernel->ioWeights[idx], 0, NULL);
    cvt_f32_f16(fp32_data, (uint16_t *)IOSurfaceGetBaseAddress(kernel->ioWeights[idx]), count);
    IOSurfaceUnlock(kernel->ioWeights[idx], 0, NULL);
}

// ---------------------------------------------------------------------------
// Direct IOSurface copy (no CPU round-trip between chained kernels)
// ---------------------------------------------------------------------------

void ane_bridge_copy_io(ANEKernelHandle *src, int src_out_idx,
                         ANEKernelHandle *dst, int dst_in_idx) {
    if (!src || !dst) return;
    if (src_out_idx < 0 || src_out_idx >= src->nOutputs) return;
    if (dst_in_idx  < 0 || dst_in_idx  >= dst->nInputs)  return;

    IOSurfaceRef srf = src->ioOutputs[src_out_idx];
    IOSurfaceRef drf = dst->ioInputs[dst_in_idx];
    size_t bytes = IOSurfaceGetAllocSize(srf);

    IOSurfaceLock(srf, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(drf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(drf), IOSurfaceGetBaseAddress(srf), bytes);
    IOSurfaceUnlock(drf, 0, NULL);
    IOSurfaceUnlock(srf, kIOSurfaceLockReadOnly, NULL);
}

// ---------------------------------------------------------------------------
// Real-time task — 90.6% p99 jitter reduction
// ---------------------------------------------------------------------------

void ane_bridge_begin_realtime(void) {
    if (!g_rt_client) return;
    ((void(*)(id,SEL))objc_msgSend)(g_rt_client, @selector(beginRealTimeTask));
}

void ane_bridge_end_realtime(void) {
    if (!g_rt_client) return;
    ((void(*)(id,SEL))objc_msgSend)(g_rt_client, @selector(endRealTimeTask));
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void ane_bridge_free(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel) return;
        NSError *e = nil;
        if (kernel->model) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                kernel->model, @selector(unloadWithQoS:error:), 21, &e);
        }
        for (int i = 0; i < kernel->nInputs;  i++) if (kernel->ioInputs[i])  CFRelease(kernel->ioInputs[i]);
        for (int i = 0; i < kernel->nOutputs; i++) if (kernel->ioOutputs[i]) CFRelease(kernel->ioOutputs[i]);
        for (int i = 0; i < kernel->nWeights; i++) if (kernel->ioWeights[i]) CFRelease(kernel->ioWeights[i]);
        if (kernel->tmpDir) [[NSFileManager defaultManager] removeItemAtPath:kernel->tmpDir error:nil];

        free(kernel->ioInputs);
        free(kernel->ioOutputs);
        free(kernel->ioWeights);
        free(kernel->inputBytes);
        free(kernel->outputBytes);
        free(kernel->weightBytes);

        kernel->model   = nil;
        kernel->request = nil;
        kernel->tmpDir  = nil;
        free(kernel);
    }
}

int ane_bridge_get_compile_count(void)  { return g_compile_count; }
void ane_bridge_reset_compile_count(void) { g_compile_count = 0; }

// ---------------------------------------------------------------------------
// Weight blob helpers (BLOBFILE mode)
// ---------------------------------------------------------------------------

uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len) {
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;
    cvt_f32_f16(src, (uint16_t *)(buf + 128), rows * cols);
    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_transposed(const float *src, int rows, int cols,
                                                   size_t *out_len) {
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;
    uint16_t *fp16 = (uint16_t *)(buf + 128);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            float32x4_t f = vdupq_n_f32(src[i * cols + j]);
            fp16[j * rows + i] = vreinterpret_u16_f16(vcvt_f16_f32(f))[0];
        }
    *out_len = total;
    return buf;
}

void ane_bridge_free_blob(void *ptr) { free(ptr); }
