// ane_bridge.m — Objective-C implementation of ANE bridge for Python ctypes
// Wraps _ANEInMemoryModel private APIs into C-callable functions

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include "ane_bridge.h"

// --- Private class references ---
static Class g_ANEDesc = nil;
static Class g_ANEInMem = nil;
static Class g_ANEReq = nil;
static Class g_ANEIO = nil;
static bool g_initialized = false;
static int g_compile_count = 0;

// --- Kernel handle struct ---
struct ANEKernelHandle {
    id model;               // _ANEInMemoryModel
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    id request;             // _ANERequest
    NSString *tmpDir;
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
};

// --- Public API ---

int ane_bridge_init(void) {
    if (g_initialized) return 0;

    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "ane_bridge: Failed to load AppleNeuralEngine.framework\n");
        return -1;
    }

    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        fprintf(stderr, "ane_bridge: Failed to resolve ANE private classes\n");
        return -1;
    }

    g_initialized = true;
    g_compile_count = 0;
    return 0;
}

static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

ANEKernelHandle *ane_bridge_compile_multi_weights(
    const char *mil_text, size_t mil_len,
    const char **weight_names, const uint8_t **weight_datas,
    const size_t *weight_lens, int n_weights,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes)
{
    @autoreleasepool {
        if (!g_initialized) {
            fprintf(stderr, "ane_bridge: Not initialized\n");
            return NULL;
        }

        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];
        NSError *e = nil;

        // Build weight dictionary
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            wdict[name] = @{@"offset": @0, @"data": data};
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict.count > 0 ? wdict : @{}, nil);
        if (!desc) {
            fprintf(stderr, "ane_bridge: modelWithMILText failed\n");
            return NULL;
        }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            fprintf(stderr, "ane_bridge: inMemoryModelWithDescriptor failed\n");
            return NULL;
        }

        // Pre-populate temp dir
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            // Extract filename from path like "@model_path/weights/wq.bin" -> "weights/wq.bin"
            NSString *relPath = name;
            if ([name hasPrefix:@"@model_path/"]) {
                relPath = [name substringFromIndex:12];
            }
            NSString *fullPath = [td stringByAppendingPathComponent:relPath];
            NSString *dir = [fullPath stringByDeletingLastPathComponent];
            [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
            NSData *data = [NSData dataWithBytes:weight_datas[i] length:weight_lens[i]];
            [data writeToFile:fullPath atomically:YES];
        }

        // Compile
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ane_bridge: ANE compile failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        // Load (with one retry after a brief pause for ANE slot reclamation)
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!loaded) {
            fprintf(stderr, "ane_bridge: ANE load failed (retrying in 100ms): %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            usleep(100000); // 100ms
            e = nil;
            loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        }
        if (!loaded) {
            fprintf(stderr, "ane_bridge: ANE load failed after retry: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        g_compile_count++;

        // Create kernel handle
        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        k->model = mdl;
        k->tmpDir = td;
        k->nInputs = n_inputs;
        k->nOutputs = n_outputs;
        k->inputBytes = (size_t *)malloc(n_inputs * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        memcpy(k->inputBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(k->outputBytes, output_sizes, n_outputs * sizeof(size_t));

        // Create IOSurfaces
        k->ioInputs = (IOSurfaceRef *)malloc(n_inputs * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(n_outputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs; i++)
            k->ioInputs[i] = create_surface(input_sizes[i]);
        for (int i = 0; i < n_outputs; i++)
            k->ioOutputs[i] = create_surface(output_sizes[i]);

        // Build request
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        return k;
    }
}

ANEKernelHandle *ane_bridge_compile(const char *mil_text, size_t mil_len,
                                     const uint8_t *weight_data, size_t weight_len,
                                     int n_inputs, const size_t *input_sizes,
                                     int n_outputs, const size_t *output_sizes) {
    if (weight_data && weight_len > 0) {
        const char *name = "@model_path/weights/weight.bin";
        return ane_bridge_compile_multi_weights(
            mil_text, mil_len,
            &name, &weight_data, &weight_len, 1,
            n_inputs, input_sizes,
            n_outputs, output_sizes);
    } else {
        return ane_bridge_compile_multi_weights(
            mil_text, mil_len,
            NULL, NULL, NULL, 0,
            n_inputs, input_sizes,
            n_outputs, output_sizes);
    }
}

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

void ane_bridge_free(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel) return;
        NSError *e = nil;
        if (kernel->model) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                kernel->model, @selector(unloadWithQoS:error:), 21, &e);
        }
        for (int i = 0; i < kernel->nInputs; i++)
            if (kernel->ioInputs[i]) CFRelease(kernel->ioInputs[i]);
        for (int i = 0; i < kernel->nOutputs; i++)
            if (kernel->ioOutputs[i]) CFRelease(kernel->ioOutputs[i]);
        if (kernel->tmpDir) {
            [[NSFileManager defaultManager] removeItemAtPath:kernel->tmpDir error:nil];
        }
        free(kernel->ioInputs);
        free(kernel->ioOutputs);
        free(kernel->inputBytes);
        free(kernel->outputBytes);
        
        // Explicitly nil Objective-C objects to trigger ARC release before freeing struct
        kernel->model = nil;
        kernel->request = nil;
        kernel->tmpDir = nil;
        
        free(kernel);
    }
}

int ane_bridge_get_compile_count(void) {
    return g_compile_count;
}

void ane_bridge_reset_compile_count(void) {
    g_compile_count = 0;
}

uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols,
                                       size_t *out_len) {
    int wsize = rows * cols * 2; // fp16
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    // ANE blob header
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;

    // Convert float32 -> float16
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < rows * cols; i++) {
        fp16[i] = (_Float16)src[i];
    }

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

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fp16[j * rows + i] = (_Float16)src[i * cols + j];

    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_int8(const int8_t *src, int rows, int cols,
                                            size_t *out_len) {
    int wsize = rows * cols;  // 1 byte per int8 element
    int total = 64 + wsize;   // 64-byte header + data
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    // ANE int8 blob header
    buf[0] = 0xEF; buf[1] = 0xBE; buf[2] = 0xAD; buf[3] = 0xDE;
    buf[4] = 0x01;
    buf[10] = 0x08;  // 8-bit element marker

    memcpy(buf + 64, src, wsize);
    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_weight_blob_quantized(const float *src, int rows, int cols,
                                                 float *out_scale, size_t *out_len) {
    // Find global max abs for symmetric quantization
    float max_abs = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        float a = src[i] < 0 ? -src[i] : src[i];
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs / 127.0f;
    if (scale == 0.0f) scale = 1.0f;

    // Quantize to int8
    int wsize = rows * cols;
    int8_t *qdata = (int8_t *)malloc(wsize);
    for (int i = 0; i < wsize; i++) {
        float v = src[i] / scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        qdata[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }

    uint8_t *blob = ane_bridge_build_weight_blob_int8(qdata, rows, cols, out_len);
    free(qdata);
    *out_scale = scale;
    return blob;
}
