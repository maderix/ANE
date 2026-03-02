// test_weight_reload.m — Can we skip recompilation by rewriting weight blobs on disk?
// Compile a conv kernel with weights A, eval, verify output.
// Overwrite weights/weight.bin in tmpDir with weights B.
// unloadWithQoS: then loadWithQoS: (no recompile).
// Eval again — if output matches B @ x, compilation bottleneck is eliminated.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Build weight blob matching inmem_peak format (single chunk)
static NSData *build_weight_blob(_Float16 *w, int rows, int cols) {
    int ws = rows * cols * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t*)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t*)(b+72) = ws;
    *(uint32_t*)(b+80) = 128;
    memcpy(b + 128, w, ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// Generate MIL for a simple conv: fp32 in → cast fp16 → conv → cast fp32 out
static NSString *gen_mil(int ch, int sp) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
        "[name=string(\"conv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
        "    } -> (y);\n"
        "}\n", ch, sp, ch, sp, ch, ch, ch, ch, ch, sp, ch, sp];
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        if (!g_D || !g_I || !g_AR || !g_AIO) {
            printf("FAIL: ANE classes not found\n");
            return 1;
        }

        // Use 64-channel conv, spatial=32 (known to work on ANE)
        int CH = 64, SP = 32;

        // Weight set A: scaled identity (1.0 on diagonal)
        _Float16 *weightsA = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) weightsA[i*CH+i] = (_Float16)1.0f;

        // Weight set B: 3x identity
        _Float16 *weightsB = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) weightsB[i*CH+i] = (_Float16)3.0f;

        NSData *wdataA = build_weight_blob(weightsA, CH, CH);
        NSString *mil = gen_mil(CH, SP);
        NSDictionary *weights = @{
            @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wdataA}
        };
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

        // === Compile with weights A ===
        printf("=== Step 1: Compile with weights A (identity) ===\n");
        printf("  Kernel: %dx%d conv, spatial=%d\n", CH, CH, SP);
        uint64_t t0 = mach_absolute_time();
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), milData, weights, nil);
        if (!desc) { printf("FAIL: desc=NULL\n"); return 1; }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wdataA writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) { printf("FAIL: compile: %s\n", [[e description] UTF8String]); return 1; }
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) { printf("FAIL: load: %s\n", [[e description] UTF8String]); return 1; }
        double compile_ms = tb_ms(mach_absolute_time() - t0);
        printf("  Compile+load: %.1fms\n", compile_ms);
        printf("  tmpDir: %s\n", [td UTF8String]);

        // Build request and IOSurfaces (fp32 I/O)
        int inBytes = CH * SP * 4;  // fp32
        int outBytes = CH * SP * 4;
        IOSurfaceRef ioIn = make_surface(inBytes);
        IOSurfaceRef ioOut = make_surface(outBytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        // Write input: channel c, spatial s = (c*SP + s + 1) * 0.01
        IOSurfaceLock(ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int c = 0; c < CH; c++)
            for (int s = 0; s < SP; s++)
                inp[c*SP+s] = (float)(c*SP + s + 1) * 0.01f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Eval with weights A
        printf("\n=== Step 2: Eval with weights A ===\n");
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        if (!ok) { printf("FAIL: eval: %s\n", e ? [[e description] UTF8String] : "?"); return 1; }

        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        float *outA = (float*)IOSurfaceGetBaseAddress(ioOut);
        printf("  Output A[0..3]: [%.4f, %.4f, %.4f, %.4f]\n", outA[0], outA[1], outA[2], outA[3]);
        printf("  Output A[%d..%d]: [%.4f, %.4f, %.4f, %.4f]\n", CH*SP-4, CH*SP-1,
               outA[CH*SP-4], outA[CH*SP-3], outA[CH*SP-2], outA[CH*SP-1]);
        // Save copy
        float *outA_copy = (float*)malloc(outBytes);
        memcpy(outA_copy, outA, outBytes);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        // === Step 3: Overwrite weight file with B, unload+load ===
        printf("\n=== Step 3: Overwrite weight.bin with B (3x identity), unload+load ===\n");
        NSData *wdataB = build_weight_blob(weightsB, CH, CH);
        NSString *weightPath = [td stringByAppendingPathComponent:@"weights/weight.bin"];
        [wdataB writeToFile:weightPath atomically:YES];
        printf("  Wrote new weight.bin\n");

        // Unload
        t0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
        double unload_ms = tb_ms(mach_absolute_time() - t0);
        printf("  Unload: %s (%.2fms)\n", ok ? "OK" : "FAIL", unload_ms);

        // Reload (no compile!)
        t0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        double reload_ms = tb_ms(mach_absolute_time() - t0);
        printf("  Load (no recompile): %s (%.2fms)\n", ok ? "OK" : [[e description] UTF8String], reload_ms);

        if (!ok) {
            printf("\n*** Load-after-overwrite FAILED — trying recompile+load ***\n");
            t0 = mach_absolute_time();
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            printf("  Re-compile: %s (%.2fms)\n", ok ? "OK" : "FAIL", tb_ms(mach_absolute_time() - t0));
            t0 = mach_absolute_time();
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            printf("  Re-load: %s (%.2fms)\n", ok ? "OK" : "FAIL", tb_ms(mach_absolute_time() - t0));
        }

        // Build new request (re-use same surfaces)
        wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        // Re-write same input
        IOSurfaceLock(ioIn, 0, NULL);
        inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int c = 0; c < CH; c++)
            for (int s = 0; s < SP; s++)
                inp[c*SP+s] = (float)(c*SP + s + 1) * 0.01f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Eval with (possibly reloaded) weights B
        printf("\n=== Step 4: Eval after reload ===\n");
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        if (!ok) { printf("FAIL: eval after reload: %s\n", e ? [[e description] UTF8String] : "?"); return 1; }

        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        float *outB = (float*)IOSurfaceGetBaseAddress(ioOut);
        printf("  Output B[0..3]: [%.4f, %.4f, %.4f, %.4f]\n", outB[0], outB[1], outB[2], outB[3]);
        printf("  Output B[%d..%d]: [%.4f, %.4f, %.4f, %.4f]\n", CH*SP-4, CH*SP-1,
               outB[CH*SP-4], outB[CH*SP-3], outB[CH*SP-2], outB[CH*SP-1]);

        // Check: did the output change?
        bool changed = false;
        float max_diff = 0;
        for (int i = 0; i < CH*SP; i++) {
            float d = fabsf(outB[i] - outA_copy[i]);
            if (d > max_diff) max_diff = d;
            if (d > 0.001f) changed = true;
        }
        // Expected: output B should be 3x output A
        bool correct_3x = true;
        float max_3x_err = 0;
        for (int i = 0; i < CH*SP; i++) {
            float expected = outA_copy[i] * 3.0f;
            float err = fabsf(outB[i] - expected);
            if (err > max_3x_err) max_3x_err = err;
            if (err > 0.1f) correct_3x = false;
        }
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        printf("\n=== RESULT ===\n");
        printf("  Max A-B diff: %.6f\n", max_diff);
        printf("  Max 3x error: %.6f\n", max_3x_err);
        printf("  Compile+load: %.1fms | Unload: %.1fms | Reload: %.1fms\n", compile_ms, unload_ms, reload_ms);

        if (changed && correct_3x) {
            printf("\nSUCCESS: Weight reload works! Output matches 3x identity.\n");
            printf("  Speedup: compile=%.1fms vs reload=%.1fms (%.1fx faster)\n",
                   compile_ms, unload_ms + reload_ms, compile_ms / (unload_ms + reload_ms));
            printf(">>> Compilation bottleneck can be eliminated <<<\n");
        } else if (changed && !correct_3x) {
            printf("\nPARTIAL: Output changed but doesn't match expected 3x.\n");
        } else {
            printf("\nFAIL: Output did NOT change. Weight reload does not work.\n");
            printf("  ANE cached the compiled model — weights baked at compile time.\n");
            printf(">>> Need alternative: weightsBuffer IOSurface or async recompile <<<\n");
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
        [fm removeItemAtPath:td error:nil];
        CFRelease(ioIn); CFRelease(ioOut);
        free(outA_copy); free(weightsA); free(weightsB);
    }
    return 0;
}
