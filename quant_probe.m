// quant_probe.m — Probe whether ANE executes int8/int4 quantized ops natively
// Tests: (1) fp16 baseline conv, (2) int8 via constexpr_affine_dequantize,
//        (3) int4 via constexpr_affine_dequantize, (4) raw int8 conv weight,
//        (5) uint8 palettized via constexpr_lut_to_dense
// If ANE hardware does native quantized execution (not just dequant-to-fp16),
// we expect 2-4x speedup over fp16 at same dimensions.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ── ANE private API boilerplate ──────────────────────────────────────────────
static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
           "AppleNeuralEngine", RTLD_NOW);
    g_D   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I   = NSClassFromString(@"_ANEInMemoryModel");
    g_AR  = NSClassFromString(@"_ANERequest");
    g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
}

static double tb_ms(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

static IOSurfaceRef make_surface(size_t bytes) {
    if (bytes < 49152) bytes = 49152;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0
    });
}

// ── Weight blob builders ─────────────────────────────────────────────────────

// FP16 blob: global header (64B) + chunk header (64B) + fp16 data
static NSData *build_fp16_blob(int oc, int ic) {
    NSUInteger wsize = (NSUInteger)oc * ic * 2;
    NSUInteger total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf + 72) = (uint32_t)wsize;
    *(uint32_t*)(buf + 80) = 128;
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (NSUInteger i = 0; i < (NSUInteger)oc * ic; i++)
        fp16[i] = (_Float16)(((float)arc4random() / UINT32_MAX - 0.5f) * 0.1f);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// INT8 blob: same header structure, but int8 data (1 byte per weight)
static NSData *build_int8_blob(int oc, int ic) {
    NSUInteger wsize = (NSUInteger)oc * ic;
    NSUInteger total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf + 72) = (uint32_t)wsize;
    *(uint32_t*)(buf + 80) = 128;
    int8_t *i8 = (int8_t*)(buf + 128);
    for (NSUInteger i = 0; i < (NSUInteger)oc * ic; i++)
        i8[i] = (int8_t)(arc4random() % 256 - 128);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// INT4 blob: packed nibbles (2 weights per byte), row-major [oc, ic/2]
static NSData *build_int4_blob(int oc, int ic) {
    NSUInteger wsize = (NSUInteger)oc * ic / 2;
    NSUInteger total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf + 72) = (uint32_t)wsize;
    *(uint32_t*)(buf + 80) = 128;
    uint8_t *packed = buf + 128;
    for (NSUInteger i = 0; i < wsize; i++)
        packed[i] = (uint8_t)(arc4random() & 0xFF);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Scale+ZP blob for affine dequantize: fp16 scale per output channel + int8 zp
static NSData *build_scale_blob(int oc) {
    NSUInteger wsize = (NSUInteger)oc * 2; // fp16 per channel
    NSUInteger total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf + 72) = (uint32_t)wsize;
    *(uint32_t*)(buf + 80) = 128;
    _Float16 *s = (_Float16*)(buf + 128);
    for (int i = 0; i < oc; i++)
        s[i] = (_Float16)(0.01f);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSData *build_zp_int8_blob(int oc) {
    NSUInteger wsize = (NSUInteger)oc;
    NSUInteger total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf + 72) = (uint32_t)wsize;
    *(uint32_t*)(buf + 80) = 128;
    // zero-points all 0
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSData *build_zp_int4_blob(int oc) {
    // For int4, zero points are also int4 packed or per-channel uint8
    // Use uint8 zero-point (one per output channel)
    NSUInteger wsize = (NSUInteger)oc;
    NSUInteger total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf + 72) = (uint32_t)wsize;
    *(uint32_t*)(buf + 80) = 128;
    memset(buf + 128, 8, wsize); // zero-point = 8 for uint4 center
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// ── Compile + eval helpers ───────────────────────────────────────────────────

typedef struct { id model; NSString *td; bool ok; } Kern;

static Kern try_compile(NSString *mil, NSDictionary *wd) {
    Kern k = {nil, nil, false};
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSError *e = nil;

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_D, @selector(modelWithMILText:weights:optionsPlist:), md, wd ?: @{}, nil);
    if (!desc) { printf("    descriptor=NULL\n"); return k; }

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in wd) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [wd[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }

    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        NSString *desc_str = [e localizedDescription] ?: @"unknown";
        if ([desc_str length] > 200) desc_str = [desc_str substringToIndex:200];
        printf("    compile FAIL: %s\n", [desc_str UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return k;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("    load FAIL\n");
        [fm removeItemAtPath:td error:nil];
        return k;
    }
    k.model = mdl; k.td = td; k.ok = true;
    return k;
}

static void kern_free(Kern *k) {
    if (!k->ok) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        k->model, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:k->td error:nil];
    k->ok = false;
}

// Benchmark: returns ms/eval, or -1 on failure
static double bench_kern(Kern *k, size_t inBytes, size_t outBytes, int warmup, int iters) {
    if (!k->ok) return -1;
    IOSurfaceRef ioIn  = make_surface(inBytes);
    IOSurfaceRef ioOut = make_surface(outBytes);

    id wIn  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    id req  = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

    NSError *e = nil;
    for (int i = 0; i < warmup; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        if (!ok) { CFRelease(ioIn); CFRelease(ioOut); return -1; }
    }
    double ms = tb_ms(mach_absolute_time() - t0) / iters;

    CFRelease(ioIn); CFRelease(ioOut);
    return ms;
}

// ── MIL generators ───────────────────────────────────────────────────────────

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

// Test 1: FP16 baseline conv (baked weights)
static NSString *gen_fp16_conv(int ic, int oc, int sp) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to16, x=x)[name=string(\"cx\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x16)[name=string(\"cv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to32, x=y16)[name=string(\"co\")];\n"
        "    } -> (y);\n}\n",
        MIL_HDR, ic, sp, ic, sp, oc, ic, oc, ic, oc, sp, oc, sp];
}

// Test 2: INT8 weights via constexpr_affine_dequantize → fp16 conv
// This is how coremltools emits int8 quantized models
// dequant formula: fp16_weight = scale * (int8_weight - zero_point)
static NSString *gen_int8_dequant_conv(int ic, int oc, int sp) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to16, x=x)[name=string(\"cx\")];\n"
        "        tensor<int8, [%d, %d, 1, 1]> Wq = const()[name=string(\"Wq\"), "
        "val=tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [%d]> sc = const()[name=string(\"sc\"), "
        "val=tensor<fp16, [%d]>(BLOBFILE(path=string(\"@model_path/weights/scale.bin\"), offset=uint64(64)))];\n"
        "        tensor<int8, [%d]> zp = const()[name=string(\"zp\"), "
        "val=tensor<int8, [%d]>(BLOBFILE(path=string(\"@model_path/weights/zp.bin\"), offset=uint64(64)))];\n"
        "        int32 ax = const()[name=string(\"ax\"), val=int32(0)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize(axis=ax, zero_point=zp, quantized_data=Wq, scale=sc)[name=string(\"dq\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x16)[name=string(\"cv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to32, x=y16)[name=string(\"co\")];\n"
        "    } -> (y);\n}\n",
        MIL_HDR, ic, sp, ic, sp,
        oc, ic, oc, ic,
        oc, oc,
        oc, oc,
        oc, ic,
        oc, sp,
        oc, sp];
}

// Test 3: INT4 (uint4) weights via constexpr_affine_dequantize → fp16 conv
// uint4 packed: 2 values per byte, axis=1 dequantize
static NSString *gen_int4_dequant_conv(int ic, int oc, int sp) {
    // For uint4, quantized_data shape is [oc, ic/2, 1, 1] packed
    // But MIL may want the logical shape [oc, ic, 1, 1] with uint4 type
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to16, x=x)[name=string(\"cx\")];\n"
        "        tensor<uint4, [%d, %d, 1, 1]> Wq = const()[name=string(\"Wq\"), "
        "val=tensor<uint4, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [%d]> sc = const()[name=string(\"sc\"), "
        "val=tensor<fp16, [%d]>(BLOBFILE(path=string(\"@model_path/weights/scale.bin\"), offset=uint64(64)))];\n"
        "        tensor<uint4, [%d]> zp = const()[name=string(\"zp\"), "
        "val=tensor<uint4, [%d]>(BLOBFILE(path=string(\"@model_path/weights/zp.bin\"), offset=uint64(64)))];\n"
        "        int32 ax = const()[name=string(\"ax\"), val=int32(0)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize(axis=ax, zero_point=zp, quantized_data=Wq, scale=sc)[name=string(\"dq\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x16)[name=string(\"cv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to32, x=y16)[name=string(\"co\")];\n"
        "    } -> (y);\n}\n",
        MIL_HDR, ic, sp, ic, sp,
        oc, ic, oc, ic,
        oc, oc,
        oc, oc,
        oc, ic,
        oc, sp,
        oc, sp];
}

// Test 4: Block-wise int4 quantization (constexpr_blockwise_shift_scale)
// This is the more modern approach used in coremltools 8+
static NSString *gen_int4_blockwise_conv(int ic, int oc, int sp, int block_size) {
    int n_blocks = ic / block_size;
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to16, x=x)[name=string(\"cx\")];\n"
        "        tensor<uint4, [%d, %d, 1, 1]> Wq = const()[name=string(\"Wq\"), "
        "val=tensor<uint4, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> ss = const()[name=string(\"ss\"), "
        "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/scale.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_blockwise_shift_scale(data=Wq, scale=ss)[name=string(\"dq\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x16)[name=string(\"cv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to32, x=y16)[name=string(\"co\")];\n"
        "    } -> (y);\n}\n",
        MIL_HDR, ic, sp, ic, sp,
        oc, ic, oc, ic,
        oc, n_blocks, oc, n_blocks,
        oc, ic,
        oc, sp,
        oc, sp];
}

// Test 5: Palettized (LUT) weights via constexpr_lut_to_dense (iOS16)
// 4-bit indices packed into bytes, 16-entry fp16 lookup table
// indices: packed byte tensor of size ceil(4 * oc * ic / 8) = oc*ic/2 bytes
// lut: [1, 1, 16] for shared LUT across all channels
// shape: [oc, ic, 1, 1] output shape
static NSString *gen_lut4_conv(int ic, int oc, int sp) {
    int packed_bytes = oc * ic / 2; // 4-bit, 2 per byte
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to16, x=x)[name=string(\"cx\")];\n"
        "        tensor<uint8, [%d]> idx = const()[name=string(\"idx\"), "
        "val=tensor<uint8, [%d]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1, 1, 16]> lut = const()[name=string(\"lut\"), "
        "val=tensor<fp16, [1, 1, 16]>(BLOBFILE(path=string(\"@model_path/weights/lut.bin\"), offset=uint64(64)))];\n"
        "        tensor<int32, [4]> shp = const()[name=string(\"shp\"), val=tensor<int32, [4]>([%d, %d, 1, 1])];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_lut_to_dense(indices=idx, lut=lut, shape=shp)[name=string(\"dq\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x16)[name=string(\"cv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to32, x=y16)[name=string(\"co\")];\n"
        "    } -> (y);\n}\n",
        MIL_HDR, ic, sp, ic, sp,
        packed_bytes, packed_bytes,
        oc, ic,
        oc, ic,
        oc, sp,
        oc, sp];
}

// LUT blob: fp16 lookup table [1, 1, 16] — shared across all channels
static NSData *build_lut_blob(int oc) {
    (void)oc; // shared LUT, oc not needed
    NSUInteger wsize = 1 * 1 * 16 * 2; // [1,1,16] fp16
    NSUInteger total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf + 72) = (uint32_t)wsize;
    *(uint32_t*)(buf + 80) = 128;
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (int j = 0; j < 16; j++)
        fp16[j] = (_Float16)((j - 8) * 0.01f);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Packed 4-bit index blob for LUT: oc*ic/2 bytes (2 indices per byte)
static NSData *build_lut_index_blob(int oc, int ic) {
    NSUInteger wsize = (NSUInteger)oc * ic / 2;
    NSUInteger total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf + 72) = (uint32_t)wsize;
    *(uint32_t*)(buf + 80) = 128;
    uint8_t *packed = buf + 128;
    for (NSUInteger i = 0; i < wsize; i++)
        packed[i] = (uint8_t)(arc4random() & 0xFF); // random 4-bit pairs
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Blockwise scale blob: [oc, n_blocks, 1, 1] fp16
static NSData *build_blockwise_scale_blob(int oc, int n_blocks) {
    NSUInteger wsize = (NSUInteger)oc * n_blocks * 2;
    NSUInteger total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf + 72) = (uint32_t)wsize;
    *(uint32_t*)(buf + 80) = 128;
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (NSUInteger i = 0; i < (NSUInteger)oc * n_blocks; i++)
        fp16[i] = (_Float16)(0.01f);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// ── Main ─────────────────────────────────────────────────────────────────────

static void run_test(const char *name, int ic, int oc, int sp,
                     NSString *mil, NSDictionary *wd) {
    printf("\n  [%s] %dx%d sp=%d\n", name, oc, ic, sp);
    Kern k = try_compile(mil, wd);
    if (!k.ok) {
        printf("    RESULT: COMPILE FAILED\n");
        return;
    }
    printf("    compile+load: OK\n");

    size_t inBytes  = (size_t)ic * sp * 4;
    size_t outBytes = (size_t)oc * sp * 4;
    double ms = bench_kern(&k, inBytes, outBytes, 10, 100);
    if (ms < 0) {
        printf("    RESULT: EVAL FAILED\n");
    } else {
        double gflops = 2.0 * oc * ic * sp / 1e9;
        double tflops = gflops / ms;
        printf("    %.3f ms/eval  (%.2f GFLOP → %.3f TFLOPS)\n", ms, gflops, tflops);
    }
    kern_free(&k);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        ane_init();

        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║  ANE Quantization Probe — int8 / int4 / LUT on Neural Engine  ║\n");
        printf("╚══════════════════════════════════════════════════════════════╝\n");
        printf("\nGoal: Determine if ANE executes quantized ops natively or just\n");
        printf("dequantizes to fp16. Native execution → 2-4x speedup over fp16.\n");
        printf("Dequant-only → same speed (compute-bound) but smaller weight blobs.\n\n");

        // Test dimensions - representative of transformer layers
        typedef struct { int ic, oc, sp; const char *desc; } Cfg;
        Cfg cfgs[] = {
            {768,  768,  64,  "Stories110M attn proj"},
            {768,  2048, 64,  "Stories110M FFN up"},
            {2048, 768,  64,  "Stories110M FFN down"},
            {1024, 1024, 64,  "1K square"},
            {2048, 2048, 64,  "2K square (stress)"},
        };
        int ncfg = sizeof(cfgs) / sizeof(cfgs[0]);

        for (int ci = 0; ci < ncfg; ci++) {
            int ic = cfgs[ci].ic, oc = cfgs[ci].oc, sp = cfgs[ci].sp;
            printf("\n━━━ %s (%dx%d, seq=%d) ━━━\n", cfgs[ci].desc, oc, ic, sp);

            // ── Test 1: FP16 baseline ──
            {
                NSString *mil = gen_fp16_conv(ic, oc, sp);
                NSData *wb = build_fp16_blob(oc, ic);
                NSDictionary *wd = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}};
                run_test("FP16 baseline", ic, oc, sp, mil, wd);
            }

            // ── Test 2: INT8 affine dequantize ──
            {
                NSString *mil = gen_int8_dequant_conv(ic, oc, sp);
                NSData *wb  = build_int8_blob(oc, ic);
                NSData *scb = build_scale_blob(oc);
                NSData *zpb = build_zp_int8_blob(oc);
                NSDictionary *wd = @{
                    @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb},
                    @"@model_path/weights/scale.bin":  @{@"offset": @0, @"data": scb},
                    @"@model_path/weights/zp.bin":     @{@"offset": @0, @"data": zpb}
                };
                run_test("INT8 affine dequant", ic, oc, sp, mil, wd);
            }

            // ── Test 3: INT4 (uint4) affine dequantize ──
            {
                NSString *mil = gen_int4_dequant_conv(ic, oc, sp);
                NSData *wb  = build_int4_blob(oc, ic);
                NSData *scb = build_scale_blob(oc);
                NSData *zpb = build_zp_int4_blob(oc);
                NSDictionary *wd = @{
                    @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb},
                    @"@model_path/weights/scale.bin":  @{@"offset": @0, @"data": scb},
                    @"@model_path/weights/zp.bin":     @{@"offset": @0, @"data": zpb}
                };
                run_test("UINT4 affine dequant", ic, oc, sp, mil, wd);
            }

            // ── Test 4: INT4 blockwise (block_size=32) ──
            if (ic % 32 == 0) {
                int block_size = 32;
                int n_blocks = ic / block_size;
                NSString *mil = gen_int4_blockwise_conv(ic, oc, sp, block_size);
                NSData *wb  = build_int4_blob(oc, ic);
                NSData *scb = build_blockwise_scale_blob(oc, n_blocks);
                NSDictionary *wd = @{
                    @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb},
                    @"@model_path/weights/scale.bin":  @{@"offset": @0, @"data": scb}
                };
                run_test("UINT4 blockwise(32)", ic, oc, sp, mil, wd);
            }

            // ── Test 5: LUT (4-bit palettized) ──
            {
                NSString *mil = gen_lut4_conv(ic, oc, sp);
                NSData *wb  = build_lut_index_blob(oc, ic);
                NSData *lut = build_lut_blob(oc);
                NSDictionary *wd = @{
                    @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb},
                    @"@model_path/weights/lut.bin":    @{@"offset": @0, @"data": lut}
                };
                run_test("LUT4 palettized", ic, oc, sp, mil, wd);
            }
        }

        // ── Summary interpretation ──
        printf("\n\n╔══════════════════════════════════════════════════════╗\n");
        printf("║  Interpretation Guide                                 ║\n");
        printf("╠══════════════════════════════════════════════════════╣\n");
        printf("║  If int8 ≈ 2x fp16 TFLOPS → native int8 execution    ║\n");
        printf("║  If int4 ≈ 4x fp16 TFLOPS → native int4 execution    ║\n");
        printf("║  If int8 ≈ fp16 TFLOPS    → dequant-to-fp16 only     ║\n");
        printf("║  If COMPILE FAIL          → type not supported in MIL ║\n");
        printf("║  If EVAL FAIL             → compiles but ANE rejects  ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n");
    }
    return 0;
}
