/*
 * m5_performance_suite.m
 * Dual-track ANE capability benchmark.
 * Evaluates dynamic weight limits strictly under compatible MIL 1.3 targets.
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

typedef NS_ENUM(NSInteger, BenchmarkMode) {
    BENCHMARK_MODE_PACKED_V1_3 = 0,
    BENCHMARK_MODE_DUAL_INPUT_V1_3 = 1
};

const uint32_t ANE_QOS_CLASS = 21;
const uint32_t WARMUP_ITERATIONS = 10;
const uint32_t BENCHMARK_ITERATIONS = 1000;
const uint32_t IOSURFACE_ALIGNMENT_BYTES = 128;
const uint32_t IOSURFACE_LOCK_READ_ONLY = 1;
const uint32_t IOSURFACE_LOCK_DEFAULT = 0;

const double NANOSECONDS_PER_MILLISECOND = 1e6;
const double NANOSECONDS_PER_MICROSECOND = 1e3;
const double BYTES_PER_MEGABYTE = 1e6;
const double FLOPS_PER_TERAFLOP_CONVERSION = 1000.0;
const double DEFAULT_LATENCY_MAX_INIT = 1e9;
const double FLOP_MULTIPLIER_MATMUL = 2.0;

static NSString* const MIL_VERSION_REQUIRED_1_3 = @"1.3";
static NSString* const MIL_TARGET_REQUIRED_IOS17 = @"ios17";

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
    BenchmarkMode mode;
    bool compile_success;
    double pure_eval_ms;
    double update_latency_ms;
    double total_throughput_gflops;
    double peak_gflops;
    size_t weight_size_bytes;
} BenchmarkResult;

static void suite_ane_init(void) {
    static bool loaded = false;
    if (loaded) return;

    mach_timebase_info(&g_tb);

    void *handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
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

static IOSurfaceRef make_surface(size_t bytes) {
    size_t aligned = ((bytes + (IOSURFACE_ALIGNMENT_BYTES - 1)) / IOSURFACE_ALIGNMENT_BYTES) * IOSURFACE_ALIGNMENT_BYTES;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(aligned),
        (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,
        (id)kIOSurfaceBytesPerRow:@(aligned),
        (id)kIOSurfaceAllocSize:@(aligned),
        (id)kIOSurfacePixelFormat:@0
    });
}

static IOSurfaceRef make_weights_surface(size_t bytes) {
    size_t aligned = ((bytes + (IOSURFACE_ALIGNMENT_BYTES - 1)) / IOSURFACE_ALIGNMENT_BYTES) * IOSURFACE_ALIGNMENT_BYTES;
    NSMutableDictionary *props = [NSMutableDictionary dictionaryWithObjectsAndKeys:
        @(aligned), (id)kIOSurfaceWidth,
        @1, (id)kIOSurfaceHeight,
        @1, (id)kIOSurfaceBytesPerElement,
        @(aligned), (id)kIOSurfaceBytesPerRow,
        @(aligned), (id)kIOSurfaceAllocSize,
        @0, (id)kIOSurfacePixelFormat,
        nil];
    [props setObject:@YES forKey:(id)kIOSurfaceIsGlobal];
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static NSString *gen_packed_matmul_mil_v1_3(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendFormat:@"program(%@)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n", MIL_VERSION_REQUIRED_1_3];
    int sp_total = seq + oc;
    [m appendFormat:@"    func main<%@>(tensor<fp32, [1, %d, 1, %d]> x) {\n", MIL_TARGET_REQUIRED_IOS17, ic, sp_total];
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

static NSString *gen_dual_input_matmul_mil_v1_3(int ic, int oc, int seq) {
    return [NSString stringWithFormat:
        @"program(%@)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<%@>(tensor<fp32, [1, 1, %d, %d]> x, tensor<fp32, [1, 1, %d, %d]> weights) {\n"
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
        MIL_VERSION_REQUIRED_1_3, MIL_TARGET_REQUIRED_IOS17,
        seq, ic, ic, oc,
        seq, ic, ic, oc,
        seq, oc, seq, oc];
}

static Kern *compile_kern_mil(NSString *mil, size_t in_bytes, size_t out_bytes, size_t weight_bytes) {
    @autoreleasepool {
        NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, @{}, nil);
        if (!desc) {
            fprintf(stderr, "  [compile] desc=NULL\n");
            return NULL;
        }
        
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSString *weightsDir = [td stringByAppendingPathComponent:@"weights"];
        NSString *modelPath = [td stringByAppendingPathComponent:@"model.mil"];
        
        [[NSFileManager defaultManager] createDirectoryAtPath:weightsDir withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:modelPath atomically:YES];
        
        NSError *e = nil;
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), ANE_QOS_CLASS, @{}, &e)) {
            fprintf(stderr, "  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error");
            return NULL;
        }
        
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), ANE_QOS_CLASS, @{}, &e)) {
            fprintf(stderr, "  [compile] load FAIL\n");
            return NULL;
        }
        
        Kern *k = (Kern*)calloc(1, sizeof(Kern));
        k->model = (void*)CFBridgingRetain(mdl);
        k->ioIn = make_surface(in_bytes);
        k->ioOut = make_surface(out_bytes);
        
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
        
        NSArray *inputs = @[wI];
        NSArray *inputIndices = @[@0];
        
        if (weight_bytes > 0) {
            k->ioWeights = make_weights_surface(weight_bytes);
            id wW = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioWeights);
            inputs = @[wI, wW];
            inputIndices = @[@0, @1];
        }
        
        k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
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
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), ANE_QOS_CLASS, &e);
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
    
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), ANE_QOS_CLASS, @{}, req, &e);
    
    IOSurfaceLock(k->ioOut, IOSURFACE_LOCK_READ_ONLY, NULL);
    IOSurfaceUnlock(k->ioOut, IOSURFACE_LOCK_READ_ONLY, NULL);
}

static void run_dimension_benchmark(int dim, BenchmarkMode mode, BenchmarkResult *result) {
    const char *mode_name = (mode == BENCHMARK_MODE_PACKED_V1_3) ? "PACKED V1.3" : "DUAL-INPUT V1.3";
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Dimension: %4d x %-4d | Mode: %-26s ║\n", dim, dim, mode_name);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    memset(result, 0, sizeof(BenchmarkResult));
    result->dimension = dim;
    result->mode = mode;
    result->weight_size_bytes = (size_t)dim * dim * sizeof(float);
    
    const int seq = 1;
    size_t in_bytes = 0;
    size_t weight_bytes = 0;
    size_t out_bytes = (size_t)dim * seq * sizeof(float);
    NSString *mil = nil;
    
    if (mode == BENCHMARK_MODE_PACKED_V1_3) {
        const int sp_total = seq + dim;
        in_bytes = (size_t)dim * sp_total * sizeof(float);
        mil = gen_packed_matmul_mil_v1_3(dim, dim, seq);
    } else {
        in_bytes = (size_t)seq * dim * sizeof(float);
        weight_bytes = result->weight_size_bytes;
        mil = gen_dual_input_matmul_mil_v1_3(dim, dim, seq);
    }
    
    printf("  [Compiling MIL program...]\n");
    uint64_t t0 = mach_absolute_time();
    Kern *k = compile_kern_mil(mil, in_bytes, out_bytes, weight_bytes);
    uint64_t compile_us = tb_us(mach_absolute_time() - t0);
    
    if (!k) {
        printf("  ✗ Compilation FAILED\n");
        result->compile_success = false;
        return;
    }
    
    result->compile_success = true;
    printf("  ✓ Compiled in %.1f ms\n", compile_us / NANOSECONDS_PER_MICROSECOND);
    printf("  ✓ Weight tensor: %.2f MB\n", result->weight_size_bytes / BYTES_PER_MEGABYTE);
    
    float *input_data = (float*)calloc(in_bytes / sizeof(float), sizeof(float));
    for (size_t i = 0; i < in_bytes / sizeof(float); i++) {
        input_data[i] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.1f;
    }
    
    IOSurfaceLock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioIn), input_data, in_bytes);
    IOSurfaceUnlock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
    
    float *new_weights = (float*)calloc(dim * dim, sizeof(float));
    for (int i = 0; i < dim * dim; i++) {
        new_weights[i] = ((float)arc4random() / UINT32_MAX - 0.5f) * 0.01f;
    }

    if (mode == BENCHMARK_MODE_DUAL_INPUT_V1_3) {
        IOSurfaceLock(k->ioWeights, IOSURFACE_LOCK_DEFAULT, NULL);
        memcpy(IOSurfaceGetBaseAddress(k->ioWeights), new_weights, weight_bytes);
        IOSurfaceUnlock(k->ioWeights, IOSURFACE_LOCK_DEFAULT, NULL);
    }
    
    printf("  [Warming up...]\n");
    for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
        suite_ane_eval_sync(k);
    }
    
    printf("  [Benchmarking pure ANE evaluation...]\n");
    t0 = mach_absolute_time();
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
        suite_ane_eval_sync(k);
    }
    double pure_eval_ms = tb_ms(mach_absolute_time() - t0) / BENCHMARK_ITERATIONS;
    
    double flops = FLOP_MULTIPLIER_MATMUL * dim * dim;
    double peak_gflops = flops / (pure_eval_ms * NANOSECONDS_PER_MILLISECOND);
    
    result->pure_eval_ms = pure_eval_ms;
    result->peak_gflops = peak_gflops;
    
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │  Pure ANE Eval:  %8.3f ms                            │\n", pure_eval_ms);
    printf("  │  Peak Throughput: %8.2f GFLOP/s (%.2f TFLOPS)        │\n", peak_gflops, peak_gflops / FLOPS_PER_TERAFLOP_CONVERSION);
    printf("  └─────────────────────────────────────────────────────────┘\n");
    
    printf("  [Benchmarking weight update latency...]\n");
    t0 = mach_absolute_time();
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
        if (mode == BENCHMARK_MODE_PACKED_V1_3) {
            IOSurfaceLock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
            float *p = (float*)IOSurfaceGetBaseAddress(k->ioIn);
            const int sp_total = seq + dim;
            for (int d = 0; d < dim; d++) {
                memcpy(p + d * sp_total + seq, new_weights + d * dim, dim * sizeof(float));
            }
            IOSurfaceUnlock(k->ioIn, IOSURFACE_LOCK_DEFAULT, NULL);
        } else {
            IOSurfaceLock(k->ioWeights, IOSURFACE_LOCK_DEFAULT, NULL);
            memcpy(IOSurfaceGetBaseAddress(k->ioWeights), new_weights, weight_bytes);
            IOSurfaceUnlock(k->ioWeights, IOSURFACE_LOCK_DEFAULT, NULL);
        }
        
        suite_ane_eval_sync(k);
    }
    double total_ms = tb_ms(mach_absolute_time() - t0) / BENCHMARK_ITERATIONS;
    
    double update_latency_ms = total_ms - pure_eval_ms;
    double total_throughput = flops / (total_ms * NANOSECONDS_PER_MILLISECOND);
    
    result->update_latency_ms = update_latency_ms;
    result->total_throughput_gflops = total_throughput;
    
    double bandwidth_gbps = result->weight_size_bytes / (update_latency_ms * NANOSECONDS_PER_MILLISECOND);
    
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │  Update Latency:  %8.3f ms (%.1f µs)              │\n", update_latency_ms, update_latency_ms * NANOSECONDS_PER_MICROSECOND);
    printf("  │  Memory Bandwidth: %8.2f GB/s                      │\n", bandwidth_gbps);
    printf("  │  Total Throughput: %8.2f GFLOP/s                   │\n", total_throughput);
    printf("  └─────────────────────────────────────────────────────────┘\n");
    
    free(input_data);
    free(new_weights);
    free_kern(k);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        suite_ane_init();
        
        const char *chip_name = ane_get_chip_name();
        
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════════╗\n");
        printf("║         ANE Performance Suite - Apple Neural Engine Benchmark        ║\n");
        printf("║                      Hardware Detection: %-10s          ║\n", chip_name);
        printf("╚══════════════════════════════════════════════════════════════════════╝\n");
        printf("\n");
        
        const int dims[] = {128, 256, 512, 1024, 2048, 4096};
        const int num_dims = sizeof(dims) / sizeof(dims[0]);
        
        BenchmarkResult results_packed[16];
        BenchmarkResult results_dual[16];
        int max_working_dim = 0;
        double max_gflops = 0;
        double min_update_latency = DEFAULT_LATENCY_MAX_INIT;
        
        printf("\n>>> PASS 1: MIL 1.3 Packed Input (Max Bandwidth Sweep) <<<\n");
        for (int i = 0; i < num_dims; i++) {
            run_dimension_benchmark(dims[i], BENCHMARK_MODE_PACKED_V1_3, &results_packed[i]);
            if (results_packed[i].compile_success && results_packed[i].peak_gflops > max_gflops) {
                max_gflops = results_packed[i].peak_gflops;
            }
        }
        
        printf("\n>>> PASS 2: MIL 1.3 Dual Input (Standard Protocol Sweep) <<<\n");
        for (int i = 0; i < num_dims; i++) {
            run_dimension_benchmark(dims[i], BENCHMARK_MODE_DUAL_INPUT_V1_3, &results_dual[i]);
            if (results_dual[i].compile_success) {
                if (results_dual[i].dimension > max_working_dim) max_working_dim = results_dual[i].dimension;
                if (results_dual[i].update_latency_ms < min_update_latency && results_dual[i].dimension >= 1024) {
                    min_update_latency = results_dual[i].update_latency_ms;
                }
            }
        }
        
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║                             BENCHMARK SUMMARY                                ║\n");
        printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        printf("┌─────────────┬───────────────────────────┬───────────────────────────┐\n");
        printf("│ Dimension   │ PACKED v1.3 (Throughput)  │ DUAL v1.3 (Update Latency)│\n");
        printf("├─────────────┼───────────────────────────┼───────────────────────────┤\n");
        
        for (int i = 0; i < num_dims; i++) {
            BenchmarkResult *r1 = &results_packed[i];
            BenchmarkResult *r2 = &results_dual[i];
            
            char r1_str[32] = "FAIL";
            if (r1->compile_success) sprintf(r1_str, "%.2f TFLOPS", r1->peak_gflops / FLOPS_PER_TERAFLOP_CONVERSION);
            
            char r2_str[32] = "FAIL";
            if (r2->compile_success) sprintf(r2_str, "%.3f ms", r2->update_latency_ms);
            
            printf("│ %4d x %-4d │ %-25s │ %-25s │\n", dims[i], dims[i], r1_str, r2_str);
        }
        printf("└─────────────┴───────────────────────────┴───────────────────────────┘\n");
        
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════════╗\n");
        printf("║                 %-6s ANE CHARACTERIZATION RESULTS                    ║\n", chip_name);
        printf("╠══════════════════════════════════════════════════════════════════════╣\n");
        printf("║  Max Dynamic Dimension:     %8d x %-8d                   ║\n", max_working_dim, max_working_dim);
        printf("║  Peak Throughput (1.3):     %8.2f TFLOPS                        ║\n", max_gflops / FLOPS_PER_TERAFLOP_CONVERSION);
        printf("║  Std Update Latency (1.3):  %8.2f ms                            ║\n", min_update_latency < DEFAULT_LATENCY_MAX_INIT ? min_update_latency : 0);
        printf("║  Max Weight Tensor Size:    %8.2f MB                            ║\n",
               (double)max_working_dim * max_working_dim * sizeof(float) / BYTES_PER_MEGABYTE);
        printf("╚══════════════════════════════════════════════════════════════════════╝\n");
        printf("\n");
        
        return 0;
    }
}