// ane_int8_bench.m — INT8 W8A8 benchmark on ANE via _ANEInMemoryModel
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o ane_int8_bench ane_int8_bench.m
// Usage: ./ane_int8_bench
//
// Tests FP16 vs W8A8 (int8 weights + int8 activation caching) throughput.
// Key MIL ops: constexpr_affine_dequantize, quantize, dequantize
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Weight blob for int8 weights (1 byte per element)
static NSData *buildWeightBlobInt8(int ch, int depth) {
    NSUInteger wsize = ch * ch * 1;
    NSUInteger chunkSize = 64 + wsize;
    NSUInteger total = 64 + chunkSize * depth;
    uint8_t *buf = calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    for (int i = 0; i < depth; i++) {
        uint8_t *chunk = buf + 64 + i * chunkSize;
        chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE;
        chunk[4]=0x01; chunk[10]=0x08;
        int8_t *data = (int8_t*)(chunk + 64);
        for (NSUInteger j = 0; j < wsize; j++) data[j] = (int8_t)(arc4random() % 256 - 128);
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Weight blob for fp16 weights (2 bytes per element)
static NSData *buildWeightBlobFP16(int ch, int depth) {
    NSUInteger wsize = ch * ch * 2;
    NSUInteger chunkSize = 64 + wsize;
    NSUInteger total = 64 + chunkSize * depth;
    uint8_t *buf = calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    for (int i = 0; i < depth; i++) {
        uint8_t *chunk = buf + 64 + i * chunkSize;
        chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE;
        chunk[4]=0x01; chunk[10]=0x10;
        _Float16 *data = (_Float16*)(chunk + 64);
        for (NSUInteger j = 0; j < (NSUInteger)(ch*ch); j++) data[j] = (_Float16)(((float)(arc4random()%1000) - 500.0f) * 0.001f);
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Generate W8A8 INT8 MIL: conv with int8 weights + quantize/dequantize between layers
static NSString *genMILInt8(int ch, int sp, int depth) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, %d, %d]> x) {\n", ch, sp, sp];
    // Conv constants
    [m appendString:@"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"];
    // Quantize/dequantize scale
    [m appendString:@"        fp16 q_scale = const()[name = string(\"q_scale\"), val = fp16(0x1p-3)];\n"
                    @"        string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"
                    @"        fp16 dq_scale = const()[name = string(\"dq_scale\"), val = fp16(0x1p-3)];\n"];

    NSUInteger cs = 64 + ch * ch * 1;  // int8 chunk size
    NSString *prev = @"x";
    for (int i = 0; i < depth; i++) {
        // constexpr_affine_dequantize: int8 weights → fp16 at compile time
        [m appendFormat:
            @"        tensor<fp16, [%d, %d, 1, 1]> W%d = constexpr_affine_dequantize()"
            @"[axis = int32(0), name = string(\"W%d\"), "
            @"quantized_data = tensor<int8, [%d, %d, 1, 1]>"
            @"(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu))), "
            @"scale = fp16(0x1p-3), zero_point = int8(0)];\n",
            ch, ch, i, i, ch, ch, (unsigned long)(64 + i * cs)];
        // conv
        NSString *conv_out = [NSString stringWithFormat:@"c%d", i];
        [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> %@ = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W%d, x = %@)[name = string(\"%@\")];\n",
            ch, sp, sp, conv_out, i, prev, conv_out];

        if (i < depth - 1) {
            // quantize: fp16 → int8
            NSString *q_out = [NSString stringWithFormat:@"q%d", i];
            [m appendFormat:@"        tensor<int8, [1, %d, %d, %d]> %@ = quantize(input = %@, output_dtype = q_dtype, scale = q_scale)[name = string(\"%@\")];\n",
                ch, sp, sp, q_out, conv_out, q_out];
            // dequantize: int8 → fp16
            NSString *dq_out = [NSString stringWithFormat:@"dq%d", i];
            [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> %@ = dequantize(input = %@, scale = dq_scale)[name = string(\"%@\")];\n",
                ch, sp, sp, dq_out, q_out, dq_out];
            prev = dq_out;
        } else {
            prev = conv_out;
        }
    }
    [m appendFormat:@"    } -> (%@);\n}\n", prev];
    return m;
}

// Generate FP16 baseline MIL: pure fp16 conv chain
static NSString *genMILFP16(int ch, int sp, int depth) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, %d, %d]> x) {\n", ch, sp, sp];
    [m appendString:@"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"];

    NSUInteger cs = 64 + ch * ch * 2;  // fp16 chunk size
    NSString *prev = @"x";
    for (int i = 0; i < depth; i++) {
        // fp16 weights from blob
        [m appendFormat:
            @"        tensor<fp16, [%d, %d, 1, 1]> W%d = const()"
            @"[name = string(\"W%d\"), "
            @"val = tensor<fp16, [%d, %d, 1, 1]>"
            @"(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n",
            ch, ch, i, i, ch, ch, (unsigned long)(64 + i * cs)];
        NSString *conv_out = [NSString stringWithFormat:@"c%d", i];
        [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> %@ = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W%d, x = %@)[name = string(\"%@\")];\n",
            ch, sp, sp, conv_out, i, prev, conv_out];
        prev = conv_out;
    }
    [m appendFormat:@"    } -> (%@);\n}\n", prev];
    return m;
}

static double benchModel(NSString *milStr, NSData *wb, int ch, int sp, const char *label) {
    @autoreleasepool {
        NSError *e = nil;
        NSData *milData = [milStr dataUsingEncoding:NSUTF8StringEncoding];
        Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class I = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,
            @selector(modelWithMILText:weights:optionsPlist:), milData,
            @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}}, nil);
        if (!desc) { printf("  %s: desc FAIL\n", label); return -1; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(I, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { printf("  %s: mdl FAIL\n", label); return -2; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 0, @{}, &e)) {
            printf("  %s: compile FAIL: %s\n", label, e ? [[e description] UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return -3;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 0, @{}, &e)) {
            printf("  %s: load FAIL\n", label);
            [fm removeItemAtPath:td error:nil];
            return -4;
        }

        NSUInteger bytes = (NSUInteger)ch * sp * sp * 2;  // fp16 I/O
        IOSurfaceRef ioI = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
            (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
            (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
        IOSurfaceRef ioO = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
            (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
            (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});

        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioI);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioO);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        // Warmup
        for (int i = 0; i < 10; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 0, @{}, req, &e);

        int iters = 50;
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 0, @{}, req, &e);
        double ms = ticksToMs(mach_absolute_time() - t0) / iters;

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 0, &e);
        CFRelease(ioI); CFRelease(ioO);
        [fm removeItemAtPath:td error:nil];
        return ms;
    }
}

int main(void) {
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

    // Query HW info
    Class DI = NSClassFromString(@"_ANEDeviceInfo");
    const char *ane_type = "unknown";
    if (DI) {
        id subType = ((id(*)(Class,SEL))objc_msgSend)(DI, @selector(aneSubType));
        if (subType) ane_type = [[subType description] UTF8String];
    }

    printf("=== ANE INT8 W8A8 Benchmark (M4, %s) ===\n\n", ane_type);
    printf("%-30s %7s %7s %9s %7s %7s\n", "Config", "W(MB)", "GOP", "ms/eval", "TOPS", "Ratio");
    printf("--------------------------------------------------------------------------------\n");

    typedef struct { int ch; int sp; int depth; } Config;
    Config configs[] = {
        {512, 64, 128},
        {512, 64, 64},
        {256, 64, 256},
        {256, 64, 128},
        {384, 64, 128},
    };
    int ncfg = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < ncfg; ci++) {
        int ch = configs[ci].ch, sp = configs[ci].sp, depth = configs[ci].depth;
        double gop = 2.0 * ch * ch * sp * sp * depth / 1e9;

        // FP16
        double w_fp16 = (double)ch * ch * 2 * depth / 1024 / 1024;
        NSString *milFP16 = genMILFP16(ch, sp, depth);
        NSData *wbFP16 = buildWeightBlobFP16(ch, depth);
        char lbl[64];
        snprintf(lbl, 64, "FP16 %dx conv %dch", depth, ch);
        double ms_fp16 = benchModel(milFP16, wbFP16, ch, sp, lbl);

        // INT8 W8A8
        double w_int8 = (double)ch * ch * 1 * depth / 1024 / 1024;
        NSString *milInt8 = genMILInt8(ch, sp, depth);
        NSData *wbInt8 = buildWeightBlobInt8(ch, depth);
        snprintf(lbl, 64, "W8A8 %dx conv %dch", depth, ch);
        double ms_int8 = benchModel(milInt8, wbInt8, ch, sp, lbl);

        if (ms_fp16 > 0 && ms_int8 > 0) {
            double tops_fp16 = gop / ms_fp16;
            double tops_int8 = gop / ms_int8;
            double ratio = ms_fp16 / ms_int8;
            printf("FP16 %-25s %6.1f  %6.2f  %7.3f ms %6.2f\n",
                   [NSString stringWithFormat:@"%dx conv %dch %dx%d", depth, ch, sp, sp].UTF8String,
                   w_fp16, gop, ms_fp16, tops_fp16);
            printf("W8A8 %-25s %6.1f  %6.2f  %7.3f ms %6.2f  %.2fx\n",
                   [NSString stringWithFormat:@"%dx conv %dch %dx%d", depth, ch, sp, sp].UTF8String,
                   w_int8, gop, ms_int8, tops_int8, ratio);
            printf("\n");
        } else {
            printf("  %dx conv %dch: FP16=%.1f INT8=%.1f (FAIL)\n", depth, ch, ms_fp16, ms_int8);
        }
    }

    printf("=== Done ===\n");
    return 0;
}
