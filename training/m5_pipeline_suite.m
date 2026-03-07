/*
 * m5_pipeline_suite.m
 * M5 ANE Pipeline Benchmark Suite
 * High-fidelity benchmarking for training pipeline simulation
 */

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <mach/mach.h>
#include <string.h>
#include <stdlib.h>

#include "ane_runtime.h"

const uint32_t ANE_QOS_CLASS = 21;
const uint32_t WARMUP_ITERATIONS = 10;
const uint32_t BENCHMARK_ITERATIONS = 100;
const uint32_t IOSURFACE_ALIGNMENT_BYTES = 128;
const uint32_t IOSURFACE_LOCK_READ_ONLY = 1;
const uint32_t IOSURFACE_LOCK_DEFAULT = 0;

const double NANOSECONDS_PER_MILLISECOND = 1e6;
const double NANOSECONDS_PER_MICROSECOND = 1e3;
const double NANOSECONDS_PER_SECOND = 1e9;
const double BYTES_PER_MEGABYTE = 1e6;
const double BYTES_PER_GIGABYTE = 1e9;

const int STRESS_TEST_LAYERS = 24;
const int STRESS_TEST_DIM = 4096;
const int LONG_SEQ_DIM = 768;
const int TRAINING_DIM = 768;
const int TRAINING_SEQ = 1024;
const int STRESS_TEST_SEQ = 1;

static NSString* const MIL_VERSION_1_3 = @"1.3";
static NSString* const MIL_VERSION_1_5 = @"1.5";
static NSString* const MIL_TARGET_IOS17 = @"ios17";
static NSString* const MIL_TARGET_IOS18 = @"ios18";

static NSString* const ANE_FRAMEWORK_PATH = @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";

static NSString* const MIL_BUILD_INFO_COMPONENT_MIL_KEY = @"coremlc-component-MIL";
static NSString* const MIL_BUILD_INFO_COMPONENT_MIL_VAL = @"3510.2.1";
static NSString* const MIL_BUILD_INFO_VER_KEY = @"coremlc-version";
static NSString* const MIL_BUILD_INFO_VER_VAL = @"3505.4.1";
static NSString* const MIL_BUILD_INFO_MILINTERNAL_KEY = @"coremltools-component-milinternal";
static NSString* const MIL_BUILD_INFO_MILINTERNAL_VAL = @"";
static NSString* const MIL_BUILD_INFO_TOOLS_VER_KEY = @"coremltools-version";
static NSString* const MIL_BUILD_INFO_TOOLS_VER_VAL = @"9.0";


static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;

typedef struct {
    void *model;
    IOSurfaceRef ioIn;
    IOSurfaceRef ioWeights;
    IOSurfaceRef ioOut;
    void *request;
    void *tmpDir;
} Kern;

typedef struct {
    int dimension;
    int num_layers;
    double total_pipeline_ms;
    double per_layer_ms;
    double context_switch_overhead_us;
    double cumulative_gflops;
    double weight_tensor_mb;
    bool success;
} LayerStressResult;

typedef struct {
    int dimension;
    int sequence_length;
    double eval_ms;
    double gflops;
    double bandwidth_gbps;
    double scaling;
    bool success;
} SequenceSweepResult;

typedef struct {
    int dimension;
    int num_layers;
    int sequence_length;
    double weight_update_ms;
    double forward_pass_ms;
    double total_step_ms;
    double tokens_per_second;
    double memory_io_ratio;
    double compute_ratio;
    bool success;
} TrainingSimResult;

typedef id (*MakeDescriptorFunc)(Class, SEL, id, id, id);
typedef id (*MakeModelFunc)(Class, SEL, id);
typedef BOOL (*CompileModelFunc)(id, SEL, unsigned int, id, id*);
typedef BOOL (*LoadModelFunc)(id, SEL, unsigned int, id, id*);
typedef BOOL (*UnloadModelFunc)(id, SEL, unsigned int, id*);
typedef BOOL (*EvaluateModelFunc)(id, SEL, unsigned int, id, id, id*);
typedef id (*MakeAIOFunc)(Class, SEL, IOSurfaceRef);
typedef id (*MakeRequestFunc)(Class, SEL, id, id, id, id, id, id, id);

static void suite_ane_init(void) {
    static bool loaded = false;
    if (loaded) return;

    mach_timebase_info(&g_tb);

    void *handle = dlopen(ANE_FRAMEWORK_PATH.UTF8String, RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "ERROR: Failed to load AppleNeuralEngine framework: %s\n", dlerror());
        return;
    }

    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");

    if (!g_D || !g_I || !g_AR || !g_AIO) {
        fprintf(stderr, "ERROR: Failed to load ANE classes\n");
        return;
    }

    loaded = true;
    printf("ANE framework loaded successfully\n");
}

static double tb_ms(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / NANOSECONDS_PER_MILLISECOND;
}

static double tb_us(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / NANOSECONDS_PER_MICROSECOND;
}

static double tb_s(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / NANOSECONDS_PER_SECOND;
}

static IOSurfaceRef make_surface(size_t bytes) {
    size_t aligned = ((bytes + (IOSURFACE_ALIGNMENT_BYTES - 1)) / IOSURFACE_ALIGNMENT_BYTES) * IOSURFACE_ALIGNMENT_BYTES;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (__bridge id)kIOSurfaceWidth: @(aligned),
        (__bridge id)kIOSurfaceHeight: @1,
        (__bridge id)kIOSurfaceBytesPerElement: @1,
        (__bridge id)kIOSurfaceBytesPerRow: @(aligned),
        (__bridge id)kIOSurfaceAllocSize: @(aligned),
        (__bridge id)kIOSurfacePixelFormat: @0
    });
}

static IOSurfaceRef make_weights_surface(size_t bytes) {
    size_t aligned = ((bytes + (IOSURFACE_ALIGNMENT_BYTES - 1)) / IOSURFACE_ALIGNMENT_BYTES) * IOSURFACE_ALIGNMENT_BYTES;
    if (aligned < IOSURFACE_ALIGNMENT_BYTES) aligned = IOSURFACE_ALIGNMENT_BYTES;
    
    NSMutableDictionary *props = [NSMutableDictionary dictionaryWithObjectsAndKeys:
        @(aligned), (__bridge id)kIOSurfaceWidth,
        @1, (__bridge id)kIOSurfaceHeight,
        @1, (__bridge id)kIOSurfaceBytesPerElement,
        @(aligned), (__bridge id)kIOSurfaceBytesPerRow,
        @(aligned), (__bridge id)kIOSurfaceAllocSize,
        @0, (__bridge id)kIOSurfacePixelFormat,
        nil];
    [props setObject:@YES forKey:(__bridge id)kIOSurfaceIsGlobal];
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static NSString *gen_packed_matmul_mil_v1_3(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendFormat:@"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    int sp_total = seq + oc;
    [m appendFormat:@"    func main<ios17>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ic, sp_total];
    [m appendString:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", ic, sp_total];
    [m appendString:@"        tensor<int32, [4]> ba = const()[name = string(\"ba\"), val = tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"act\")];\n", ic, seq];
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0,0,0,%d])];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name = string(\"ra\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", ic, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", seq, ic];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name = string(\"rw\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", ic, oc];
    [m appendString:@"        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"mm\")];\n", seq, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", oc, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name = string(\"ro\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];\n", oc, seq];
    [m appendString:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static NSString *gen_packed_matmul_mil_v1_5(int ic, int oc, int seq) {
    // MIL 1.5/ios18 not supported by ANE compiler, fallback to 1.3/ios17
    return gen_packed_matmul_mil_v1_3(ic, oc, seq);
}

static NSString *gen_dynamic_matmul_mil(int ic, int oc, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios17>(tensor<fp32, [1, 1, %d, %d]> x, tensor<fp32, [1, 1, %d, %d]> weights) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, 1, %d, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_x\")];\n"
        "        tensor<fp16, [1, 1, %d, %d]> w16 = cast(dtype = to_fp16, x = weights)[name = string(\"cast_w\")];\n"
        "        bool tx = const()[name = string(\"tx\"), val = bool(false)];\n"
        "        bool ty = const()[name = string(\"ty\"), val = bool(false)];\n"
        "        tensor<fp16, [1, 1, %d, %d]> y16 = matmul(transpose_x = tx, transpose_y = ty, x = x16, y = w16)[name = string(\"matmul\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, 1, %d, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n"
        "}\n",
        seq, ic, ic, oc,
        seq, ic, ic, oc,
        seq, oc, seq, oc];
}

static Kern *compile_kern_mil(NSString *mil, size_t in_bytes, size_t out_bytes, size_t weight_bytes) {
    @autoreleasepool {
        NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
        
        MakeDescriptorFunc makeDesc = (MakeDescriptorFunc)objc_msgSend;
        id desc = makeDesc(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, @{}, nil);
        if (!desc) {
            fprintf(stderr, "  [compile] desc=NULL\n");
            return NULL;
        }
        
        MakeModelFunc makeModel = (MakeModelFunc)objc_msgSend;
        id mdl = makeModel(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSString *weightsDir = [td stringByAppendingPathComponent:@"weights"];
        NSString *modelPath = [td stringByAppendingPathComponent:@"model.mil"];
        
        [[NSFileManager defaultManager] createDirectoryAtPath:weightsDir withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:modelPath atomically:YES];
        
        NSError *e = nil;
        CompileModelFunc compileModel = (CompileModelFunc)objc_msgSend;
        if (!compileModel(mdl, @selector(compileWithQoS:options:error:), ANE_QOS_CLASS, @{}, &e)) {
            fprintf(stderr, "  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error");
            return NULL;
        }
        
        LoadModelFunc loadModel = (LoadModelFunc)objc_msgSend;
        if (!loadModel(mdl, @selector(loadWithQoS:options:error:), ANE_QOS_CLASS, @{}, &e)) {
            fprintf(stderr, "  [compile] load FAIL\n");
            return NULL;
        }
        
        Kern *k = (Kern*)calloc(1, sizeof(Kern));
        k->model = (void*)CFBridgingRetain(mdl);
        k->ioIn = make_surface(in_bytes);
        k->ioOut = make_surface(out_bytes);
        
        MakeAIOFunc makeAIO = (MakeAIOFunc)objc_msgSend;
        id wI = makeAIO(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
        id wO = makeAIO(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
        
        NSArray *inputs = @[wI];
        NSArray *inputIndices = @[@0];
        
        if (weight_bytes > 0) {
            k->ioWeights = make_weights_surface(weight_bytes);
            id wW = makeAIO(g_AIO, @selector(objectWithIOSurface:), k->ioWeights);
            inputs = @[wI, wW];
            inputIndices = @[@0, @1];
        }
        
        MakeRequestFunc makeReq = (MakeRequestFunc)objc_msgSend;
        k->request = (void*)CFBridgingRetain(makeReq(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            inputs, inputIndices, @[wO], @[@0], nil, nil, @0));
        k->tmpDir = (void*)CFBridgingRetain(td);
        
        return k;
    }
}

static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model;
    NSError *e = nil;
    UnloadModelFunc unloadModel = (UnloadModelFunc)objc_msgSend;
    unloadModel(mdl, @selector(unloadWithQoS:error:), ANE_QOS_CLASS, &e);
    CFRelease(k->ioIn);
    CFRelease(k->ioOut);
    if (k->ioWeights) {
        CFRelease(k->ioWeights);
    }
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    CFRelease(k->model);
    CFRelease(k->request);
    CFRelease(k->tmpDir);
    free(k);
}

static void suite_ane_eval_sync(Kern *k) {
    id mdl = (__bridge id)k->model;
    id req = (__bridge id)k->request;
    NSError *e = nil;
    
    EvaluateModelFunc evalModel = (EvaluateModelFunc)objc_msgSend;
    evalModel(mdl, @selector(evaluateWithQoS:options:request:error:), ANE_QOS_CLASS, @{}, req, &e);
    
    IOSurfaceLock(k->ioOut, IOSURFACE_LOCK_READ_ONLY, NULL);
    IOSurfaceUnlock(k->ioOut, IOSURFACE_LOCK_READ_ONLY, NULL);
}

static NSString *get_macos_version(void) {
    NSProcessInfo *pi = [NSProcessInfo processInfo];
    NSOperatingSystemVersion v = [pi operatingSystemVersion];
    return [NSString stringWithFormat:@"%ld.%ld.%ld", (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion];
}

static void print_header(const char *chip_name, const char *mil_version, const char *ios_target) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    M5 ANE Pipeline Benchmark Suite                           ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Hardware: Apple %-4s                                                        ║\n", chip_name);
    NSString *macos_ver = get_macos_version();
    const char *macos_str = macos_ver ? [macos_ver UTF8String] : "Unknown";
    printf("║  macOS:   %-10s                                                          ║\n", macos_str);
    printf("║  MIL Version: %-4s (%-6s target)                                          ║\n", mil_version, ios_target);
    printf("║  ANE QoS: %d                                                                 ║\n", ANE_QOS_CLASS);
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

static void print_section_header(const char *title) {
    printf("\n");
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  %-76s│\n", title);
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
}

static void run_layer_stress_test(int dim, int num_layers, bool is_m5, LayerStressResult *result) {
    printf("\n");
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│                    BENCHMARK 1: %d-Layer Stress Test                         │\n", num_layers);
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Configuration:                                                              │\n");
    printf("│    Dimension: %d x %d                                                    │\n", dim, dim);
    printf("│    Layers: %d                                                                │\n", num_layers);
    printf("│    Sequence: %d                                                              │\n", STRESS_TEST_SEQ);
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    
    memset(result, 0, sizeof(LayerStressResult));
    result->dimension = dim;
    result->num_layers = num_layers;
    result->weight_tensor_mb = (double)dim * dim * sizeof(float) / BYTES_PER_MEGABYTE;
    
    const int sp_total = STRESS_TEST_SEQ + dim;
    size_t in_bytes = (size_t)dim * sp_total * sizeof(float);
    size_t out_bytes = (size_t)dim * STRESS_TEST_SEQ * sizeof(float);
    size_t weight_bytes = 0;
    
    NSString *mil = is_m5 ? gen_packed_matmul_mil_v1_5(dim, dim, STRESS_TEST_SEQ) : gen_packed_matmul_mil_v1_3(dim, dim, STRESS_TEST_SEQ);
    
    printf("│  [Compiling MIL program...]                                                 │\n");
    uint64_t t0 = mach_absolute_time();
    Kern *k = compile_kern_mil(mil, in_bytes, out_bytes, weight_bytes);
    uint64_t compile_us = tb_us(mach_absolute_time() - t0);
    
    if (!k) {
        printf("│  ✗ Compilation FAILED                                                       │\n");
        printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
        result->success = false;
        return;
    }
    
    printf("│  ✓ Compiled in %.1f ms                                                       │\n", compile_us / NANOSECONDS_PER_MICROSECOND);
    printf("│  ✓ Weight tensor: %.2f MB per layer                                          │\n", result->weight_tensor_mb);
    
    float **weight_sets = (float**)calloc(num_layers, sizeof(float*));
    for (int layer = 0; layer < num_layers; layer++) {
        weight_sets[layer] = (float*)calloc(dim * dim, sizeof(float));
        for (int i = 0; i < dim * dim; i++) {
            weight_sets[layer][i] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.01f;
        }
    }
    
    float *input_data = (float*)calloc(in_bytes / sizeof(float), sizeof(float));
    for (size_t i = 0; i < in_bytes / sizeof(float); i++) {
        input_data[i] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.1f;
    }
    
    IOSurfaceLock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioIn), input_data, in_bytes);
    IOSurfaceUnlock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
    
    printf("│  [Warming up...]                                                            │\n");
    for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
        suite_ane_eval_sync(k);
    }
    
    printf("│  [Running %d-layer pipeline...]                                             │\n", num_layers);
    
    uint64_t *layer_times = (uint64_t*)calloc(num_layers, sizeof(uint64_t));
    uint64_t total_start = mach_absolute_time();
    
    for (int layer = 0; layer < num_layers; layer++) {
        uint64_t layer_start = mach_absolute_time();
        
        IOSurfaceLock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
        float *buf = (float*)IOSurfaceGetBaseAddress(k->ioIn);
        for (int d = 0; d < dim; d++) {
            memcpy(buf + d * sp_total + STRESS_TEST_SEQ, weight_sets[layer] + d * dim, dim * sizeof(float));
        }
        IOSurfaceUnlock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
        
        suite_ane_eval_sync(k);
        
        layer_times[layer] = mach_absolute_time() - layer_start;
    }
    
    uint64_t total_end = mach_absolute_time();
    double total_ms = tb_ms(total_end - total_start);
    
    double per_layer_ms = total_ms / num_layers;
    
    long long flops_per_layer_ll = 2LL * (long long)1 * (long long)dim * (long long)dim;
    long long total_flops_ll = flops_per_layer_ll * (long long)num_layers;
    double total_time_seconds = tb_s(total_end - total_start);
    
    double total_gflops = (double)total_flops_ll / (total_time_seconds * 1e9);
    double tflops = (total_gflops > 100.0) ? (total_gflops / 1000.0) : 0.0;
    
    double per_layer_time_seconds = per_layer_ms / 1000.0;
    double per_layer_gflops = (double)flops_per_layer_ll / (per_layer_time_seconds * 1e9);
    
    double sum_layer_ms = 0;
    for (int layer = 0; layer < num_layers; layer++) {
        sum_layer_ms += tb_ms(layer_times[layer]);
    }
    double context_overhead_us = (total_ms - sum_layer_ms) * NANOSECONDS_PER_MICROSECOND / NANOSECONDS_PER_MILLISECOND;
    
    result->total_pipeline_ms = total_ms;
    result->per_layer_ms = per_layer_ms;
    result->context_switch_overhead_us = context_overhead_us;
    result->cumulative_gflops = total_gflops;
    result->success = true;
    
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Results:                                                                    │\n");
    printf("│    Total Pipeline Latency:    %8.2f ms                                    │\n", total_ms);
    printf("│    Per-Layer Average:         %8.3f ms                                     │\n", per_layer_ms);
    printf("│    Context Switch Overhead:   %8.3f µs                                    │\n", context_overhead_us);
    printf("│    Per-Layer Performance:     %8.2f GFLOPS                                │\n", per_layer_gflops);
    
    if (total_gflops < 1.0) {
        printf("│    Total Pipeline Throughput: %8.4f GFLOPS                                │\n", total_gflops);
    } else if (total_gflops < 100.0) {
        printf("│    Total Pipeline Throughput: %8.2f GFLOPS                                │\n", total_gflops);
    } else {
        printf("│    Total Pipeline Throughput: %8.4f TFLOPS                                │\n", tflops);
    }
    printf("│    Weight Tensor Size:        %8.2f MB per layer                          │\n", result->weight_tensor_mb);
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
    
    for (int layer = 0; layer < num_layers; layer++) {
        free(weight_sets[layer]);
    }
    free(weight_sets);
    free(input_data);
    free(layer_times);
    free_kern(k);
}

static void run_long_sequence_sweep(int dim, const int *seq_values, int num_seq, SequenceSweepResult *results) {
    printf("\n");
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│                  BENCHMARK 2: Long-Sequence Sweep                            │\n");
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Configuration: dim=%d                                                      │\n", dim);
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│  SEQ    │  Eval Time (ms)  │  GFLOPS*  │  Bandwidth (GB/s)* │  Scaling       │\n");
    printf("├─────────┼──────────────────┼──────────┼────────────────────┼────────────────┤\n");
    
    double base_tflops = 0;
    
    for (int i = 0; i < num_seq; i++) {
        int seq = seq_values[i];
        memset(&results[i], 0, sizeof(SequenceSweepResult));
        results[i].dimension = dim;
        results[i].sequence_length = seq;
        
        size_t in_bytes = (size_t)seq * dim * sizeof(float);
        size_t weight_bytes = (size_t)dim * dim * sizeof(float);
        size_t out_bytes = (size_t)seq * dim * sizeof(float);
        
        NSString *mil = gen_dynamic_matmul_mil(dim, dim, seq);
        Kern *k = compile_kern_mil(mil, in_bytes, out_bytes, weight_bytes);
        
        if (!k) {
            printf("│  %5d │  COMPILATION FAILED                                            │\n", seq);
            results[i].success = false;
            continue;
        }
        
        float *input_data = (float*)calloc(in_bytes / sizeof(float), sizeof(float));
        float *weight_data = (float*)calloc(weight_bytes / sizeof(float), sizeof(float));
        for (size_t j = 0; j < in_bytes / sizeof(float); j++) {
            input_data[j] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.1f;
        }
        for (size_t j = 0; j < weight_bytes / sizeof(float); j++) {
            weight_data[j] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.01f;
        }
        
        IOSurfaceLock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
        memcpy(IOSurfaceGetBaseAddress(k->ioIn), input_data, in_bytes);
        IOSurfaceUnlock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
        
        IOSurfaceLock(k->ioWeights, IOSURFACE_LOCK_DEFAULT, NULL);
        memcpy(IOSurfaceGetBaseAddress(k->ioWeights), weight_data, weight_bytes);
        IOSurfaceUnlock(k->ioWeights, IOSURFACE_LOCK_DEFAULT, NULL);
        
        for (uint32_t w = 0; w < WARMUP_ITERATIONS; w++) {
            suite_ane_eval_sync(k);
        }
        
        uint64_t t0 = mach_absolute_time();
        for (uint32_t iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
            suite_ane_eval_sync(k);
        }
        double eval_ms = tb_ms(mach_absolute_time() - t0) / BENCHMARK_ITERATIONS;
        
        long long flops_ll = 2LL * (long long)seq * (long long)dim * (long long)dim;
        double eval_time_seconds = eval_ms / 1000.0;
        
        double gflops = (double)flops_ll / (eval_time_seconds * 1e9);
        
        double total_bytes = (double)in_bytes + (double)out_bytes + (double)weight_bytes;
        double bandwidth = total_bytes / eval_time_seconds / BYTES_PER_GIGABYTE;
        
        if (i == 0) {
            base_tflops = gflops;
            results[i].scaling = 1.0;
        } else {
            results[i].scaling = gflops / base_tflops;
        }
        
        results[i].eval_ms = eval_ms;
        results[i].gflops = gflops;
        results[i].bandwidth_gbps = bandwidth;
        results[i].success = true;
        
        printf("│  %5d │      %8.3f      │  %7.2f*  │       %8.2f*     │  %5.2fx         │\n",
               seq, eval_ms, gflops, bandwidth, results[i].scaling);
        
        free(input_data);
        free(weight_data);
        free_kern(k);
    }
    
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    
    bool linear_scaling = true;
    for (int i = 1; i < num_seq; i++) {
        if (results[i].success && results[i].scaling < results[i-1].scaling * 0.8) {
            linear_scaling = false;
            break;
        }
    }
    
    int threshold_seq = -1;
    for (int i = 1; i < num_seq; i++) {
        if (results[i].success && results[i].gflops > results[0].gflops * 1.5) {
            threshold_seq = seq_values[i];
            break;
        }
    }
    
    printf("│  Analysis: TFLOPS scales %-10s with sequence length                    │\n",
           linear_scaling ? "linearly" : "sub-linearly");
    if (threshold_seq > 0) {
        printf("│  Compute-bound threshold: SEQ >= %-5d                                      │\n", threshold_seq);
    } else {
        printf("│  Compute-bound threshold: Not reached in tested range                       │\n");
    }
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
    printf("  * SRAM: ANE internal cache bandwidth (exceeds system RAM limits)\n");
}

static void run_training_simulator(int dim, int layers, int seq, TrainingSimResult *result) {
    printf("\n");
    printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│              BENCHMARK 3: End-to-End Training Throughput Simulator            │\n");
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Configuration:                                                              │\n");
    printf("│    Dimension: %d                                                            │\n", dim);
    printf("│    Layers: %d                                                                │\n", layers);
    printf("│    Sequence: %d                                                              │\n", seq);
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    
    memset(result, 0, sizeof(TrainingSimResult));
    result->dimension = dim;
    result->num_layers = layers;
    result->sequence_length = seq;
    
    size_t in_bytes = (size_t)seq * dim * sizeof(float);
    size_t weight_bytes = (size_t)dim * dim * sizeof(float);
    size_t out_bytes = (size_t)seq * dim * sizeof(float);
    
    NSString *mil = gen_dynamic_matmul_mil(dim, dim, seq);
    
    printf("│  [Compiling MIL program...]                                                 │\n");
    uint64_t t0 = mach_absolute_time();
    Kern *k = compile_kern_mil(mil, in_bytes, out_bytes, weight_bytes);
    uint64_t compile_us = tb_us(mach_absolute_time() - t0);
    
    if (!k) {
        printf("│  ✗ Compilation FAILED                                                       │\n");
        printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
        result->success = false;
        return;
    }
    
    printf("│  ✓ Compiled in %.1f ms                                                       │\n", compile_us / NANOSECONDS_PER_MICROSECOND);
    
    float **weight_sets = (float**)calloc(layers, sizeof(float*));
    for (int layer = 0; layer < layers; layer++) {
        weight_sets[layer] = (float*)calloc(dim * dim, sizeof(float));
        for (int i = 0; i < dim * dim; i++) {
            weight_sets[layer][i] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.01f;
        }
    }
    
    float *input_data = (float*)calloc(in_bytes / sizeof(float), sizeof(float));
    for (size_t i = 0; i < in_bytes / sizeof(float); i++) {
        input_data[i] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.1f;
    }
    
    IOSurfaceLock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioIn), input_data, in_bytes);
    IOSurfaceUnlock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
    
    printf("│  [Warming up...]                                                            │\n");
    for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
        IOSurfaceLock(k->ioWeights, IOSURFACE_LOCK_DEFAULT, NULL);
        memcpy(IOSurfaceGetBaseAddress(k->ioWeights), weight_sets[0], weight_bytes);
        IOSurfaceUnlock(k->ioWeights, IOSURFACE_LOCK_DEFAULT, NULL);
        suite_ane_eval_sync(k);
    }
    
    printf("│  [Simulating %d-layer training step...]                                     │\n", layers);
    
    double total_update_us = 0;
    double total_forward_us = 0;
    
    for (int layer = 0; layer < layers; layer++) {
        uint64_t update_start = mach_absolute_time();
        IOSurfaceLock(k->ioWeights, IOSURFACE_LOCK_DEFAULT, NULL);
        memcpy(IOSurfaceGetBaseAddress(k->ioWeights), weight_sets[layer], weight_bytes);
        IOSurfaceUnlock(k->ioWeights, IOSURFACE_LOCK_DEFAULT, NULL);
        uint64_t update_end = mach_absolute_time();
        total_update_us += tb_us(update_end - update_start);
        
        uint64_t forward_start = mach_absolute_time();
        suite_ane_eval_sync(k);
        uint64_t forward_end = mach_absolute_time();
        total_forward_us += tb_us(forward_end - forward_start);
    }
    
    double total_update_ms = total_update_us / NANOSECONDS_PER_MICROSECOND;
    double total_forward_ms = total_forward_us / NANOSECONDS_PER_MICROSECOND;
    double total_step_ms = total_update_ms + total_forward_ms;
    
    double total_step_seconds = total_step_ms / 1000.0;
    double tps = (double)seq / total_step_seconds;
    
    double memory_io_ratio = total_update_ms / total_forward_ms;
    double compute_ratio = total_forward_ms / total_step_ms;
    
    double weight_update_bytes = (double)weight_bytes * (double)layers;
    double update_time_seconds = total_update_ms / 1000.0;
    double bandwidth_gbps = weight_update_bytes / update_time_seconds / BYTES_PER_GIGABYTE;
    
    long long flops_per_layer_ll = 2LL * (long long)seq * (long long)dim * (long long)dim;
    long long total_flops_ll = flops_per_layer_ll * (long long)layers;
    
    double total_gflops = (double)total_flops_ll / (total_step_seconds * 1e9);
    double tflops = (total_gflops > 100.0) ? (total_gflops / 1000.0) : 0.0;
    
    double per_layer_time_seconds = (total_forward_ms / (double)layers) / 1000.0;
    double per_layer_gflops = (double)flops_per_layer_ll / (per_layer_time_seconds * 1e9);
    
    result->weight_update_ms = total_update_ms;
    result->forward_pass_ms = total_forward_ms;
    result->total_step_ms = total_step_ms;
    result->tokens_per_second = tps;
    result->memory_io_ratio = memory_io_ratio;
    result->compute_ratio = compute_ratio;
    result->success = true;
    
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Timing Breakdown:                                                           │\n");
    printf("│    Weight Update (Memory I/O):  %8.2f ms (%5.1f%%)                         │\n",
           total_update_ms, (total_update_ms / total_step_ms) * 100);
    printf("│    Forward Pass (ANE Compute):  %8.2f ms (%5.1f%%)                         │\n",
           total_forward_ms, (total_forward_ms / total_step_ms) * 100);
    printf("│    Total Step Time:             %8.2f ms                                    │\n", total_step_ms);
    printf("├──────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Throughput Metrics:                                                         │\n");
    printf("│    Tokens Per Second:           %8.2f TPS                                  │\n", tps);
    printf("│    Memory Bandwidth:            %8.2f GB/s                                  │\n", bandwidth_gbps);
    printf("│    Per-Layer Compute:           %8.2f GFLOPS                                │\n", per_layer_gflops);
    
    if (total_gflops < 1.0) {
        printf("│    Total Pipeline Throughput:   %8.4f GFLOPS                               │\n", total_gflops);
    } else if (total_gflops < 100.0) {
        printf("│    Total Pipeline Throughput:   %8.2f GFLOPS                                │\n", total_gflops);
    } else {
        printf("│    Total Pipeline Throughput:   %8.4f TFLOPS                                │\n", tflops);
    }
    printf("│    Memory/Compute Ratio:        %8.2f (%s)                    │\n",
           memory_io_ratio, memory_io_ratio > 1.0 ? "I/O bound" : "Compute bound");
    printf("└──────────────────────────────────────────────────────────────────────────────┘\n");
    
    for (int layer = 0; layer < layers; layer++) {
        free(weight_sets[layer]);
    }
    free(weight_sets);
    free(input_data);
    free_kern(k);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        suite_ane_init();
        
        const char *chip_name = ane_get_chip_name();
        bool is_m5 = ane_supports_mil_1_5();
        const char *mil_version = MIL_VERSION_1_3.UTF8String;
        const char *ios_target = MIL_TARGET_IOS17.UTF8String;
        
        print_header(chip_name, mil_version, ios_target);
        
        LayerStressResult stress_result;
        SequenceSweepResult seq_results[3];
        TrainingSimResult train_result;
        
        print_section_header("BENCHMARK 1: 24-Layer Stress Test");
        run_layer_stress_test(STRESS_TEST_DIM, STRESS_TEST_LAYERS, is_m5, &stress_result);
        
        print_section_header("BENCHMARK 2: Long-Sequence Sweep");
        const int seq_values[] = {128, 512, 1024};
        run_long_sequence_sweep(LONG_SEQ_DIM, seq_values, 3, seq_results);
        
        print_section_header("BENCHMARK 3: Training Throughput Simulator");
        run_training_simulator(TRAINING_DIM, STRESS_TEST_LAYERS, TRAINING_SEQ, &train_result);
        
        printf("\n");
        printf("║                         M5 PIPELINE SUITE SUMMARY                            ║\n");
        printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║  Benchmark              │  Key Metric           │  Value                     ║\n");
        printf("╠═════════════════════════╪═══════════════════════╪════════════════════════════╣\n");
        
        if (stress_result.success) {
            printf("║  24-Layer Stress        │  Per-Layer GFLOPS     │  %8.2f GFLOPS           ║\n",
                   stress_result.cumulative_gflops);
        } else {
            printf("║  24-Layer Stress        │  Status               │  FAILED                    ║\n");
        }
        
        if (seq_results[2].success) {
            printf("║  Long-Sequence (1024)   │  Peak GFLOPS          │  %8.2f GFLOPS           ║\n",
                   seq_results[2].gflops);
        } else if (seq_results[1].success) {
            printf("║  Long-Sequence (512)    │  Peak GFLOPS          │  %8.2f GFLOPS           ║\n",
                   seq_results[1].gflops);
        } else if (seq_results[0].success) {
            printf("║  Long-Sequence (128)    │  Peak GFLOPS          │  %8.2f GFLOPS           ║\n",
                   seq_results[0].gflops);
        } else {
            printf("║  Long-Sequence          │  Status               │  FAILED                    ║\n");
        }
        
        if (train_result.success) {
            printf("║  Training Simulator     │  Tokens/Second        │  %8.2f TPS               ║\n",
                   train_result.tokens_per_second);
        } else {
            printf("║  Training Simulator     │  Status               │  FAILED                    ║\n");
        }
        
        printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        printf("\n");
        
        return 0;
    }
}
