// test_mil_custom.m — Experiments Y1-Y3, Z1: Custom MIL -> ANE Execution
// Build: make test_mil_custom && ./test_mil_custom
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

#pragma mark - MIL Compilation Pipeline

static id compileAndCreateEngine(NSString *milText, NSString *label,
                                  id container, MLModelConfiguration *cfg,
                                  MLModelDescription *desc, NSError **outErr) {
    NSString *milPath = [NSString stringWithFormat:@"/tmp/%@.mil", label];
    [milText writeToFile:milPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
    NSURL *milURL = [NSURL fileURLWithPath:milPath];

    Class aotCls = NSClassFromString(@"MLE5ProgramLibraryOnDeviceAOTCompilationImpl");
    if (!aotCls) {
        if (outErr) *outErr = [NSError errorWithDomain:@"MIL" code:1
            userInfo:@{NSLocalizedDescriptionKey: @"AOT class not found"}];
        return nil;
    }

    id aotImpl = ((id(*)(id,SEL,id,id,id))objc_msgSend)(
        [aotCls alloc],
        NSSelectorFromString(@"initWithMILTextAtURL:container:configuration:"),
        milURL, container, cfg);
    if (!aotImpl) {
        if (outErr) *outErr = [NSError errorWithDomain:@"MIL" code:2
            userInfo:@{NSLocalizedDescriptionKey: @"AOT init failed"}];
        return nil;
    }

    NSError *plErr = nil;
    void *plHandle = ((void*(*)(id,SEL,BOOL,NSError**))objc_msgSend)(
        aotImpl,
        NSSelectorFromString(@"createProgramLibraryHandleWithRespecialization:error:"),
        NO, &plErr);
    if (!plHandle) {
        printf("  [%s] PL handle failed: %s\n", [label UTF8String],
               plErr ? [[plErr description] UTF8String] : "unknown");
        if (outErr) *outErr = plErr;
        return nil;
    }

    Class plCls = NSClassFromString(@"MLE5ProgramLibrary");
    id progLib = ((id(*)(id,SEL,id,id,id))objc_msgSend)(
        [plCls alloc],
        NSSelectorFromString(@"initWithImpl:container:configuration:"),
        aotImpl, container, cfg);
    if (!progLib) {
        if (outErr) *outErr = [NSError errorWithDomain:@"MIL" code:4
            userInfo:@{NSLocalizedDescriptionKey: @"ProgramLibrary init failed"}];
        return nil;
    }

    Class engCls = NSClassFromString(@"MLE5Engine");

    // Find the correct init selector
    static dispatch_once_t once;
    static SEL engInitSel = NULL;
    dispatch_once(&once, ^{
        unsigned int mc;
        Method *ims = class_copyMethodList(engCls, &mc);
        printf("  MLE5Engine init selectors:\n");
        for (unsigned int i = 0; i < mc; i++) {
            const char *sel = sel_getName(method_getName(ims[i]));
            if (strstr(sel, "init")) {
                printf("    - %s  [%s]\n", sel, method_getTypeEncoding(ims[i]));
                if (strstr(sel, "ProgramLibrary") && strstr(sel, "modelDescription"))
                    engInitSel = method_getName(ims[i]);
            }
        }
        free(ims);
    });

    if (!engInitSel) {
        if (outErr) *outErr = [NSError errorWithDomain:@"MIL" code:5
            userInfo:@{NSLocalizedDescriptionKey: @"No MLE5Engine init selector found"}];
        return nil;
    }

    printf("  Using init: %s\n", sel_getName(engInitSel));

    // Count colons to determine argument count
    const char *selName = sel_getName(engInitSel);
    int argCount = 0;
    for (const char *p = selName; *p; p++) if (*p == ':') argCount++;

    id engine = nil;
    if (argCount == 7) {
        // initWithProgramLibrary:modelDescription:configuration:functionName:
        //   classProbabilitiesFeatureName:optionalInputDefaultValues:compilerVersionInfo:
        engine = ((id(*)(id,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            [engCls alloc], engInitSel, progLib, desc, cfg,
            @"main", nil, nil, nil);
    } else if (argCount == 5) {
        engine = ((id(*)(id,SEL,id,id,id,id,id))objc_msgSend)(
            [engCls alloc], engInitSel, progLib, desc, cfg, nil, label);
    } else if (argCount == 6) {
        engine = ((id(*)(id,SEL,id,id,id,id,id,id))objc_msgSend)(
            [engCls alloc], engInitSel, progLib, desc, cfg, nil, nil, label);
    } else {
        printf("  Unexpected arg count %d for MLE5Engine init\n", argCount);
    }

    if (!engine) {
        if (outErr) *outErr = [NSError errorWithDomain:@"MIL" code:5
            userInfo:@{NSLocalizedDescriptionKey: @"Engine init failed"}];
        return nil;
    }

    NSError *prepErr = nil;
    BOOL prepOk = ((BOOL(*)(id,SEL,long long,NSError**))objc_msgSend)(
        engine, NSSelectorFromString(@"prepareWithConcurrencyHint:error:"),
        (long long)1, &prepErr);
    if (!prepOk) {
        printf("  [%s] Prepare failed: %s\n", [label UTF8String],
               prepErr ? [[prepErr description] UTF8String] : "unknown");
        if (outErr) *outErr = prepErr;
        return nil;
    }

    return engine;
}

static id<MLFeatureProvider> runEngine(id engine, id<MLFeatureProvider> features,
                                       MLPredictionOptions *opts, NSError **outErr) {
    return ((id(*)(id,SEL,id,id,NSError**))objc_msgSend)(
        engine, NSSelectorFromString(@"predictionFromFeatures:options:error:"),
        features, opts, outErr);
}

#pragma mark - Numeric Helpers

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static float mean_abs(const float *a, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += fabsf(a[i]);
    return s / n;
}

static void fill_random(float *buf, int n, float scale) {
    for (int i = 0; i < n; i++)
        buf[i] = ((float)arc4random() / (float)UINT32_MAX - 0.5f) * 2.0f * scale;
}

static void print_first(const char *label, const float *buf, int total) {
    int n = total < 8 ? total : 8;
    printf("  %s: [", label);
    for (int i = 0; i < n; i++)
        printf("%s%.4f", i ? ", " : "", buf[i]);
    printf("]\n");
}

#pragma mark - CPU Reference Implementations

static void cpu_sdpa(const float *Q, const float *K, const float *V,
                     float *out, int seqLen, int headDim) {
    float scale = 1.0f / sqrtf((float)headDim);
    float *scores = (float *)calloc(seqLen * seqLen, sizeof(float));

    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < seqLen; j++) {
            float dot = 0;
            for (int d = 0; d < headDim; d++)
                dot += Q[i * headDim + d] * K[j * headDim + d];
            scores[i * seqLen + j] = dot * scale;
        }
    }
    for (int i = 0; i < seqLen; i++) {
        float maxv = scores[i * seqLen];
        for (int j = 1; j < seqLen; j++)
            if (scores[i * seqLen + j] > maxv) maxv = scores[i * seqLen + j];
        float sum = 0;
        for (int j = 0; j < seqLen; j++) {
            scores[i * seqLen + j] = expf(scores[i * seqLen + j] - maxv);
            sum += scores[i * seqLen + j];
        }
        for (int j = 0; j < seqLen; j++)
            scores[i * seqLen + j] /= sum;
    }
    for (int i = 0; i < seqLen; i++) {
        for (int d = 0; d < headDim; d++) {
            float acc = 0;
            for (int j = 0; j < seqLen; j++)
                acc += scores[i * seqLen + j] * V[j * headDim + d];
            out[i * headDim + d] = acc;
        }
    }
    free(scores);
}

#pragma mark - Container Discovery

static id findE5Container(MLModel *model, NSURL *compiledURL, MLModelConfiguration *cfg) {
    // Try standard paths first
    @try {
        id eng = [model valueForKey:@"_internalEngine"];
        if ([NSStringFromClass([eng class]) containsString:@"MLE5"]) {
            id pl = [eng valueForKey:@"programLibrary"];
            if (pl) {
                id c = nil;
                @try { c = [pl valueForKey:@"_container"]; } @catch(id e) { (void)e; }
                if (!c) {
                    @try {
                        id impl = [pl valueForKey:@"_impl"];
                        if (impl) c = [impl valueForKey:@"_container"];
                    } @catch(id e) { (void)e; }
                }
                if (c) return c;
            }
        }

        // MLMultiFunctionProgramEngine path
        if ([NSStringFromClass([eng class]) isEqualToString:@"MLMultiFunctionProgramEngine"]) {
            NSDictionary *map = [eng valueForKey:@"_functionNameToEngineMap"];
            for (id key in map) {
                id sub = map[key];
                if ([NSStringFromClass([sub class]) containsString:@"MLE5"]) {
                    id pl = [sub valueForKey:@"programLibrary"];
                    if (pl) {
                        id c = nil;
                        @try { c = [pl valueForKey:@"_container"]; } @catch(id e) { (void)e; }
                        if (!c) {
                            @try {
                                id impl = [pl valueForKey:@"_impl"];
                                if (impl) c = [impl valueForKey:@"_container"];
                            } @catch(id e) { (void)e; }
                        }
                        if (c) return c;
                    }
                }
            }
        }
    } @catch(id e) { (void)e; }

    // Create MLProgramE5Container directly from compiled model
    Class e5Cls = NSClassFromString(@"MLProgramE5Container");
    if (!e5Cls) return nil;

    // Find model.mil path inside the compiled model
    NSString *compiledPath = [compiledURL path];
    NSString *milPath = [compiledPath stringByAppendingPathComponent:@"model.mil"];
    if (![[NSFileManager defaultManager] fileExistsAtPath:milPath]) {
        printf("  No model.mil at %s\n", [milPath UTF8String]);

        // List contents
        NSArray *contents = [[NSFileManager defaultManager]
            contentsOfDirectoryAtPath:compiledPath error:nil];
        printf("  Compiled model contents: %s\n", [[contents description] UTF8String]);
    }

    // Try to create E5 container with the model asset description from NN container
    @try {
        id eng = [model valueForKey:@"_internalEngine"];
        id nnContainer = [eng valueForKey:@"_container"];
        if (nnContainer) {
            // Get model file path
            NSString *modelFilePath = nil;
            @try { modelFilePath = [nnContainer valueForKey:@"_modelFilePath"]; }
            @catch(id e) { (void)e; }

            if (modelFilePath) {
                printf("  Model file path: %s\n", [modelFilePath UTF8String]);

                // Try to create E5 container with this path
                @try {
                    id c = ((id(*)(id,SEL,id,id))objc_msgSend)(
                        [e5Cls alloc],
                        NSSelectorFromString(@"initWithModelAssetPath:configuration:"),
                        modelFilePath, cfg);
                    if (c) return c;
                } @catch(id e) { (void)e; }
            }

            // Try initWithModelAssetDescription
            @try {
                id assetDesc = nil;
                @try { assetDesc = [nnContainer valueForKey:@"_modelAssetDescription"]; }
                @catch(id e) { (void)e; }
                if (!assetDesc) {
                    @try { assetDesc = [nnContainer valueForKey:@"modelAssetDescription"]; }
                    @catch(id e) { (void)e; }
                }
                if (assetDesc) {
                    printf("  Asset description: %s\n",
                           [NSStringFromClass([assetDesc class]) UTF8String]);
                    id c = ((id(*)(id,SEL,id,id))objc_msgSend)(
                        [e5Cls alloc],
                        NSSelectorFromString(@"initWithModelAssetDescription:configuration:"),
                        assetDesc, cfg);
                    if (c) return c;
                }
            } @catch(id e) { (void)e; }
        }
    } @catch(id e) { (void)e; }

    // Dump E5Container init methods
    unsigned int mc;
    Method *ims = class_copyMethodList(e5Cls, &mc);
    printf("  MLProgramE5Container init methods:\n");
    for (unsigned int i = 0; i < mc; i++) {
        const char *sel = sel_getName(method_getName(ims[i]));
        if (strstr(sel, "init"))
            printf("    - %s\n", sel);
    }
    free(ims);

    return nil;
}

#pragma mark - Main

int main(int argc, const char *argv[]) {
    (void)argc; (void)argv;
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        printf("================================================================\n");
        printf("  Custom MIL -> ANE: Experiments Y1, Y2, Y3, Z1\n");
        printf("================================================================\n\n");

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
               "AppleNeuralEngine", RTLD_NOW);

        NSString *pkgPath = @"/tmp/ane_sram_256ch_64sp.mlpackage";
        if (![[NSFileManager defaultManager] fileExistsAtPath:pkgPath]) {
            printf("FATAL: %s not found. Run: python3 scripts/gen_mlpackages.py\n",
                   [pkgPath UTF8String]);
            return 1;
        }

        NSError *err = nil;
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsAll;
        MLPredictionOptions *opts = [[MLPredictionOptions alloc] init];

        NSURL *compiled = [MLModel compileModelAtURL:
            [NSURL fileURLWithPath:pkgPath] error:&err];
        if (err) { printf("FATAL: compile: %s\n", [[err description] UTF8String]); return 1; }

        MLModel *refModel = [MLModel modelWithContentsOfURL:compiled
                                              configuration:cfg error:&err];
        if (err) { printf("FATAL: load: %s\n", [[err description] UTF8String]); return 1; }
        printf("  Ref model: %s\n", [NSStringFromClass([refModel class]) UTF8String]);

        MLModelDescription *refDesc = [refModel modelDescription];

        // Find or create E5 container
        id refContainer = findE5Container(refModel, compiled, cfg);
        if (refContainer) {
            printf("  Container: %s\n\n", [NSStringFromClass([refContainer class]) UTF8String]);
        } else {
            printf("  No E5 container found. Trying nil container...\n\n");
        }

        int ch = 256, sp = 64;
        int nElems = ch * sp;
        NSString *inName = [[[refDesc inputDescriptionsByName] allKeys] firstObject];
        NSString *outName = [[[refDesc outputDescriptionsByName] allKeys] firstObject];
        printf("  I/O: %s -> %s, shape [1,%d,1,%d]\n\n", [inName UTF8String],
               [outName UTF8String], ch, sp);

        // ============================================================
        // Y1: Scaled Dot-Product Attention
        // ============================================================
        printf("================================================================\n");
        printf("  Y1: scaled_dot_product_attention on ANE\n");
        printf("================================================================\n\n");

        {
            int seqLen = ch, headDim = sp;

            NSString *sdpaMIL = [NSString stringWithFormat:
                @"program(1.3)\n"
                "{\n"
                "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
                "        string c16 = const()[name = string(\"c16\"), val = string(\"fp16\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = c16, x = x)[name = string(\"x16\")];\n"
                "        tensor<int32, [4]> sr = const()[name = string(\"sr\"), val = tensor<int32, [4]>([1, 1, %d, %d])];\n"
                "        tensor<fp16, [1, 1, %d, %d]> q = reshape(x = x16, shape = sr)[name = string(\"q\")];\n"
                "        tensor<fp16, [1, 1, %d, %d]> k = reshape(x = x16, shape = sr)[name = string(\"k\")];\n"
                "        tensor<fp16, [1, 1, %d, %d]> v = reshape(x = x16, shape = sr)[name = string(\"v\")];\n"
                "        tensor<fp16, [1, 1, %d, %d]> attn = scaled_dot_product_attention(query = q, key = k, value = v)[name = string(\"attn\")];\n"
                "        tensor<int32, [4]> or = const()[name = string(\"or\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
                "        tensor<fp16, [1, %d, 1, %d]> rs = reshape(x = attn, shape = or)[name = string(\"rs\")];\n"
                "        string c32 = const()[name = string(\"c32\"), val = string(\"fp32\")];\n"
                "        tensor<fp32, [1, %d, 1, %d]> cast_out = cast(dtype = c32, x = rs)[name = string(\"cast_out\")];\n"
                "    } -> (cast_out);\n"
                "}\n",
                ch, sp, ch, sp,
                seqLen, headDim, seqLen, headDim, seqLen, headDim, seqLen, headDim,
                seqLen, headDim,
                ch, sp, ch, sp,
                ch, sp];

            printf("  Self-attention: B=1, nHeads=1, seqLen=%d, headDim=%d\n\n", seqLen, headDim);

            err = nil;
            id engine = compileAndCreateEngine(sdpaMIL, @"y1_sdpa", refContainer, cfg, refDesc, &err);

            if (!engine) {
                printf("  Y1 FAILED: %s\n\n", err ? [[err description] UTF8String] : "unknown");
            } else {
                printf("  Y1: Engine created\n");
                MLMultiArray *inputArr = [[MLMultiArray alloc]
                    initWithShape:@[@1, @(ch), @1, @(sp)]
                    dataType:MLMultiArrayDataTypeFloat32 error:nil];
                float *inPtr = (float *)[inputArr dataPointer];
                fill_random(inPtr, nElems, 0.5f);

                MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{inName: inputArr} error:nil];

                NSError *runErr = nil;
                uint64_t t0 = mach_absolute_time();
                id<MLFeatureProvider> result = runEngine(engine, fp, opts, &runErr);
                double ms = tb_ms(mach_absolute_time() - t0);

                if (runErr || !result) {
                    printf("  Y1 prediction FAILED: %s\n\n",
                           runErr ? [[runErr description] UTF8String] : "nil");
                } else {
                    MLMultiArray *outArr = [result featureValueForName:outName].multiArrayValue;
                    if (!outArr) {
                        printf("  Y1 output nil\n\n");
                    } else {
                        float *outPtr = (float *)[outArr dataPointer];
                        print_first("ANE out", outPtr, nElems);
                        printf("  Time: %.3f ms\n", ms);

                        float *cpuOut = (float *)calloc(nElems, sizeof(float));
                        cpu_sdpa(inPtr, inPtr, inPtr, cpuOut, seqLen, headDim);
                        print_first("CPU ref", cpuOut, nElems);

                        float mad = max_abs_diff(outPtr, cpuOut, nElems);
                        printf("  Max diff: %.6f, Rel: %.2e\n",
                               mad, mad / (mean_abs(cpuOut, nElems) + 1e-10f));
                        printf("  %s\n\n", mad < 0.02f ? "*** Y1 PASSED ***" :
                               (mad < 0.1f ? "Y1 WARNING" : "Y1 FAILED"));

                        int N = 100;
                        t0 = mach_absolute_time();
                        for (int i = 0; i < N; i++) runEngine(engine, fp, opts, nil);
                        printf("  Bench: %.4f ms/eval (%d iters)\n\n",
                               tb_ms(mach_absolute_time() - t0) / N, N);
                        free(cpuOut);
                    }
                }
            }
        }

        // ============================================================
        // Y2: Linear with Embedded Weights
        // ============================================================
        printf("================================================================\n");
        printf("  Y2: linear op with embedded weights on ANE\n");
        printf("================================================================\n\n");

        {
            int inDim = sp, outDim = sp;

            float *W = (float *)malloc(outDim * inDim * sizeof(float));
            float *B = (float *)malloc(outDim * sizeof(float));
            fill_random(W, outDim * inDim, 0.1f);
            fill_random(B, outDim, 0.01f);

            NSMutableString *wLit = [NSMutableString stringWithString:@"["];
            for (int i = 0; i < outDim; i++) {
                if (i > 0) [wLit appendString:@", "];
                [wLit appendString:@"["];
                for (int j = 0; j < inDim; j++) {
                    if (j > 0) [wLit appendString:@", "];
                    [wLit appendFormat:@"%.8e", W[i * inDim + j]];
                }
                [wLit appendString:@"]"];
            }
            [wLit appendString:@"]"];

            NSMutableString *bLit = [NSMutableString stringWithString:@"["];
            for (int j = 0; j < outDim; j++) {
                if (j > 0) [bLit appendString:@", "];
                [bLit appendFormat:@"%.8e", B[j]];
            }
            [bLit appendString:@"]"];

            NSString *linearMIL = [NSString stringWithFormat:
                @"program(1.3)\n"
                "{\n"
                "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
                "        string c16 = const()[name = string(\"c16\"), val = string(\"fp16\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = c16, x = x)[name = string(\"x16\")];\n"
                "        tensor<int32, [2]> rs = const()[name = string(\"rs\"), val = tensor<int32, [2]>([%d, %d])];\n"
                "        tensor<fp16, [%d, %d]> flat = reshape(x = x16, shape = rs)[name = string(\"flat\")];\n"
                "        tensor<fp16, [%d, %d]> Wc = const()[name = string(\"Wc\"), val = tensor<fp16, [%d, %d]>(%@)];\n"
                "        tensor<fp16, [%d]> Bc = const()[name = string(\"Bc\"), val = tensor<fp16, [%d]>(%@)];\n"
                "        tensor<fp16, [%d, %d]> lin = linear(x = flat, weight = Wc, bias = Bc)[name = string(\"lin\")];\n"
                "        tensor<int32, [4]> rs2 = const()[name = string(\"rs2\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
                "        tensor<fp16, [1, %d, 1, %d]> rso = reshape(x = lin, shape = rs2)[name = string(\"rso\")];\n"
                "        string c32 = const()[name = string(\"c32\"), val = string(\"fp32\")];\n"
                "        tensor<fp32, [1, %d, 1, %d]> cast_out = cast(dtype = c32, x = rso)[name = string(\"cast_out\")];\n"
                "    } -> (cast_out);\n"
                "}\n",
                ch, sp, ch, sp,
                ch, sp, ch, sp,
                outDim, inDim, outDim, inDim, wLit,
                outDim, outDim, bLit,
                ch, outDim,
                ch, sp, ch, sp,
                ch, sp];

            printf("  Config: [%d,%d] linear %d->%d with embedded W+b\n\n", ch, sp, inDim, outDim);

            err = nil;
            id engine = compileAndCreateEngine(linearMIL, @"y2_linear", refContainer, cfg, refDesc, &err);

            if (!engine) {
                printf("  Y2 FAILED: %s\n\n", err ? [[err description] UTF8String] : "unknown");
            } else {
                printf("  Y2: Engine created\n");
                MLMultiArray *inputArr = [[MLMultiArray alloc]
                    initWithShape:@[@1, @(ch), @1, @(sp)]
                    dataType:MLMultiArrayDataTypeFloat32 error:nil];
                float *inPtr = (float *)[inputArr dataPointer];
                fill_random(inPtr, nElems, 0.5f);

                MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{inName: inputArr} error:nil];

                NSError *runErr = nil;
                uint64_t t0 = mach_absolute_time();
                id<MLFeatureProvider> result = runEngine(engine, fp, opts, &runErr);
                double ms = tb_ms(mach_absolute_time() - t0);

                if (runErr || !result) {
                    printf("  Y2 prediction FAILED: %s\n\n",
                           runErr ? [[runErr description] UTF8String] : "nil");
                } else {
                    MLMultiArray *outArr = [result featureValueForName:outName].multiArrayValue;
                    if (outArr) {
                        float *outPtr = (float *)[outArr dataPointer];
                        print_first("ANE out", outPtr, nElems);
                        printf("  Time: %.3f ms\n", ms);

                        // CPU: x[ch,sp] @ W^T[sp,sp] + b[sp]
                        float *cpuOut = (float *)calloc(nElems, sizeof(float));
                        for (int i = 0; i < ch; i++) {
                            for (int j = 0; j < outDim; j++) {
                                float acc = 0;
                                for (int k = 0; k < inDim; k++)
                                    acc += inPtr[i * inDim + k] * W[j * inDim + k];
                                cpuOut[i * outDim + j] = acc + B[j];
                            }
                        }
                        print_first("CPU ref", cpuOut, nElems);

                        float mad = max_abs_diff(outPtr, cpuOut, nElems);
                        printf("  Max diff: %.6f, Rel: %.2e\n",
                               mad, mad / (mean_abs(cpuOut, nElems) + 1e-10f));
                        printf("  %s\n\n", mad < 0.05f ? "*** Y2 PASSED ***" :
                               (mad < 0.5f ? "Y2 WARNING" : "Y2 FAILED"));

                        int N = 100;
                        t0 = mach_absolute_time();
                        for (int i = 0; i < N; i++) runEngine(engine, fp, opts, nil);
                        printf("  Bench: %.4f ms/eval (%d iters)\n\n",
                               tb_ms(mach_absolute_time() - t0) / N, N);
                        free(cpuOut);
                    }
                }
            }
            free(W); free(B);
        }

        // ============================================================
        // Y3: Transformer Block (Attention + FFN)
        // ============================================================
        printf("================================================================\n");
        printf("  Y3: Transformer Block (LN + SDPA + Residual + LN + FFN + Residual)\n");
        printf("================================================================\n\n");

        {
            int seqLen = ch, dim = sp, ffnDim = 128;

            float *w1 = (float *)malloc(ffnDim * dim * sizeof(float));
            float *b1 = (float *)malloc(ffnDim * sizeof(float));
            float *w2 = (float *)malloc(dim * ffnDim * sizeof(float));
            float *b2 = (float *)malloc(dim * sizeof(float));
            fill_random(w1, ffnDim * dim, 0.05f);
            fill_random(b1, ffnDim, 0.01f);
            fill_random(w2, dim * ffnDim, 0.05f);
            fill_random(b2, dim, 0.01f);

            // Build weight string literals
            NSMutableString *(^buildMat)(float*, int, int) = ^(float *m, int rows, int cols) {
                NSMutableString *s = [NSMutableString stringWithString:@"["];
                for (int i = 0; i < rows; i++) {
                    if (i > 0) [s appendString:@", "];
                    [s appendString:@"["];
                    for (int j = 0; j < cols; j++) {
                        if (j > 0) [s appendString:@", "];
                        [s appendFormat:@"%.8e", m[i * cols + j]];
                    }
                    [s appendString:@"]"];
                }
                [s appendString:@"]"];
                return s;
            };

            NSMutableString *(^buildVec)(float*, int) = ^(float *v, int n) {
                NSMutableString *s = [NSMutableString stringWithString:@"["];
                for (int i = 0; i < n; i++) {
                    if (i > 0) [s appendString:@", "];
                    [s appendFormat:@"%.8e", v[i]];
                }
                [s appendString:@"]"];
                return s;
            };

            NSMutableString *(^buildOnes)(int) = ^(int n) {
                NSMutableString *s = [NSMutableString stringWithString:@"["];
                for (int i = 0; i < n; i++) {
                    if (i > 0) [s appendString:@", "];
                    [s appendString:@"1.0"];
                }
                [s appendString:@"]"];
                return s;
            };

            NSMutableString *(^buildZeros)(int) = ^(int n) {
                NSMutableString *s = [NSMutableString stringWithString:@"["];
                for (int i = 0; i < n; i++) {
                    if (i > 0) [s appendString:@", "];
                    [s appendString:@"0.0"];
                }
                [s appendString:@"]"];
                return s;
            };

            NSString *tfMIL = [NSString stringWithFormat:
                @"program(1.3)\n"
                "{\n"
                "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
                "        string c16 = const()[name = string(\"c16\"), val = string(\"fp16\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = c16, x = x)[name = string(\"x16\")];\n"
                "        tensor<int32, [2]> r2 = const()[name = string(\"r2\"), val = tensor<int32, [2]>([%d, %d])];\n"
                "        tensor<fp16, [%d, %d]> flat = reshape(x = x16, shape = r2)[name = string(\"flat\")];\n"
                // LN1
                "        tensor<fp16, [%d]> g1 = const()[name = string(\"g1\"), val = tensor<fp16, [%d]>(%@)];\n"
                "        tensor<fp16, [%d]> b1 = const()[name = string(\"b1\"), val = tensor<fp16, [%d]>(%@)];\n"
                "        tensor<int32, [1]> la = const()[name = string(\"la\"), val = tensor<int32, [1]>([-1])];\n"
                "        fp16 eps = const()[name = string(\"eps\"), val = fp16(1e-5)];\n"
                "        tensor<fp16, [%d, %d]> ln1 = layer_norm(x = flat, axes = la, gamma = g1, beta = b1, epsilon = eps)[name = string(\"ln1\")];\n"
                // SDPA
                "        tensor<int32, [4]> sr = const()[name = string(\"sr\"), val = tensor<int32, [4]>([1, 1, %d, %d])];\n"
                "        tensor<fp16, [1, 1, %d, %d]> q = reshape(x = ln1, shape = sr)[name = string(\"q\")];\n"
                "        tensor<fp16, [1, 1, %d, %d]> k = reshape(x = ln1, shape = sr)[name = string(\"k\")];\n"
                "        tensor<fp16, [1, 1, %d, %d]> v = reshape(x = ln1, shape = sr)[name = string(\"v\")];\n"
                "        tensor<fp16, [1, 1, %d, %d]> at = scaled_dot_product_attention(query = q, key = k, value = v)[name = string(\"at\")];\n"
                "        tensor<fp16, [%d, %d]> af = reshape(x = at, shape = r2)[name = string(\"af\")];\n"
                // Residual 1
                "        tensor<fp16, [%d, %d]> r1 = add(x = flat, y = af)[name = string(\"r1\")];\n"
                // LN2
                "        tensor<fp16, [%d]> g2 = const()[name = string(\"g2\"), val = tensor<fp16, [%d]>(%@)];\n"
                "        tensor<fp16, [%d]> b2 = const()[name = string(\"b2\"), val = tensor<fp16, [%d]>(%@)];\n"
                "        tensor<fp16, [%d, %d]> ln2 = layer_norm(x = r1, axes = la, gamma = g2, beta = b2, epsilon = eps)[name = string(\"ln2\")];\n"
                // FFN
                "        tensor<fp16, [%d, %d]> W1 = const()[name = string(\"W1\"), val = tensor<fp16, [%d, %d]>(%@)];\n"
                "        tensor<fp16, [%d]> B1 = const()[name = string(\"B1\"), val = tensor<fp16, [%d]>(%@)];\n"
                "        tensor<fp16, [%d, %d]> f1 = linear(x = ln2, weight = W1, bias = B1)[name = string(\"f1\")];\n"
                "        tensor<fp16, [%d, %d]> ga = gelu(x = f1, mode = string(\"TANH_APPROXIMATION\"))[name = string(\"ga\")];\n"
                "        tensor<fp16, [%d, %d]> W2 = const()[name = string(\"W2\"), val = tensor<fp16, [%d, %d]>(%@)];\n"
                "        tensor<fp16, [%d]> B2 = const()[name = string(\"B2\"), val = tensor<fp16, [%d]>(%@)];\n"
                "        tensor<fp16, [%d, %d]> f2 = linear(x = ga, weight = W2, bias = B2)[name = string(\"f2\")];\n"
                // Residual 2
                "        tensor<fp16, [%d, %d]> r2o = add(x = r1, y = f2)[name = string(\"r2o\")];\n"
                // Output
                "        tensor<int32, [4]> r4 = const()[name = string(\"r4\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
                "        tensor<fp16, [1, %d, 1, %d]> o16 = reshape(x = r2o, shape = r4)[name = string(\"o16\")];\n"
                "        string c32 = const()[name = string(\"c32\"), val = string(\"fp32\")];\n"
                "        tensor<fp32, [1, %d, 1, %d]> cast_out = cast(dtype = c32, x = o16)[name = string(\"cast_out\")];\n"
                "    } -> (cast_out);\n"
                "}\n",
                ch, sp, ch, sp,
                seqLen, dim, seqLen, dim,
                dim, dim, buildOnes(dim),
                dim, dim, buildZeros(dim),
                seqLen, dim,
                seqLen, dim, seqLen, dim, seqLen, dim, seqLen, dim,
                seqLen, dim,
                seqLen, dim,
                seqLen, dim,
                dim, dim, buildOnes(dim),
                dim, dim, buildZeros(dim),
                seqLen, dim,
                ffnDim, dim, ffnDim, dim, buildMat(w1, ffnDim, dim),
                ffnDim, ffnDim, buildVec(b1, ffnDim),
                seqLen, ffnDim,
                seqLen, ffnDim,
                dim, ffnDim, dim, ffnDim, buildMat(w2, dim, ffnDim),
                dim, dim, buildVec(b2, dim),
                seqLen, dim,
                seqLen, dim,
                ch, sp, ch, sp,
                ch, sp];

            printf("  Pipeline: LN->SDPA->Res->LN->FFN(%d->%d->%d)->Res\n\n", dim, ffnDim, dim);

            err = nil;
            id engine = compileAndCreateEngine(tfMIL, @"y3_transformer",
                refContainer, cfg, refDesc, &err);

            if (!engine) {
                printf("  Y3 FAILED: %s\n\n", err ? [[err description] UTF8String] : "unknown");
            } else {
                printf("  Y3: Engine created!\n");
                MLMultiArray *inputArr = [[MLMultiArray alloc]
                    initWithShape:@[@1, @(ch), @1, @(sp)]
                    dataType:MLMultiArrayDataTypeFloat32 error:nil];
                float *inPtr = (float *)[inputArr dataPointer];
                fill_random(inPtr, nElems, 0.5f);

                MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{inName: inputArr} error:nil];

                NSError *runErr = nil;
                uint64_t t0 = mach_absolute_time();
                id<MLFeatureProvider> result = runEngine(engine, fp, opts, &runErr);
                double ms = tb_ms(mach_absolute_time() - t0);

                if (runErr || !result) {
                    printf("  Y3 prediction FAILED: %s\n\n",
                           runErr ? [[runErr description] UTF8String] : "nil");
                } else {
                    MLMultiArray *outArr = [result featureValueForName:outName].multiArrayValue;
                    if (outArr) {
                        float *outPtr = (float *)[outArr dataPointer];
                        print_first("ANE out", outPtr, nElems);
                        printf("  Time: %.3f ms\n", ms);
                        float m = mean_abs(outPtr, nElems);
                        printf("  Non-zero: %s (mean_abs=%.6f)\n", m > 1e-6f ? "YES" : "NO", m);
                        printf("  %s\n\n", m > 1e-6f ? "*** Y3 PASSED ***" : "Y3 FAILED");

                        int N = 100;
                        t0 = mach_absolute_time();
                        for (int i = 0; i < N; i++) runEngine(engine, fp, opts, nil);
                        printf("  Bench: %.4f ms/eval (%d iters)\n\n",
                               tb_ms(mach_absolute_time() - t0) / N, N);
                    }
                }
            }
            free(w1); free(b1); free(w2); free(b2);
        }

        // ============================================================
        // Z1: Linear Backward Pass (Gradient Computation)
        // ============================================================
        printf("================================================================\n");
        printf("  Z1: Backward Pass (matmul with runtime tensors) on ANE\n");
        printf("================================================================\n\n");

        {
            int M = 128, K = 64, N = 64;

            NSString *bwdMIL = [NSString stringWithFormat:
                @"program(1.3)\n"
                "{\n"
                "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
                "        string c16 = const()[name = string(\"c16\"), val = string(\"fp16\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = c16, x = x)[name = string(\"x16\")];\n"
                "        tensor<int32, [2]> r2 = const()[name = string(\"r2\"), val = tensor<int32, [2]>([%d, %d])];\n"
                "        tensor<fp16, [%d, %d]> flat = reshape(x = x16, shape = r2)[name = string(\"flat\")];\n"
                // Slice dY [0:128, :]
                "        tensor<int32, [2]> db = const()[name = string(\"db\"), val = tensor<int32, [2]>([0, 0])];\n"
                "        tensor<int32, [2]> de = const()[name = string(\"de\"), val = tensor<int32, [2]>([%d, %d])];\n"
                "        tensor<fp16, [%d, %d]> dY = slice_by_index(x = flat, begin = db, end = de)[name = string(\"dY\")];\n"
                // Slice W [128:192, :]
                "        tensor<int32, [2]> wb = const()[name = string(\"wb\"), val = tensor<int32, [2]>([%d, 0])];\n"
                "        tensor<int32, [2]> we = const()[name = string(\"we\"), val = tensor<int32, [2]>([%d, %d])];\n"
                "        tensor<fp16, [%d, %d]> W = slice_by_index(x = flat, begin = wb, end = we)[name = string(\"W\")];\n"
                // Slice pad [192:256, :]
                "        tensor<int32, [2]> pb = const()[name = string(\"pb\"), val = tensor<int32, [2]>([%d, 0])];\n"
                "        tensor<int32, [2]> pe = const()[name = string(\"pe\"), val = tensor<int32, [2]>([%d, %d])];\n"
                "        tensor<fp16, [%d, %d]> pad = slice_by_index(x = flat, begin = pb, end = pe)[name = string(\"pad\")];\n"
                // dX = dY @ W
                "        bool txf = const()[name = string(\"txf\"), val = bool(false)];\n"
                "        bool tyf = const()[name = string(\"tyf\"), val = bool(false)];\n"
                "        bool txt = const()[name = string(\"txt\"), val = bool(true)];\n"
                "        tensor<fp16, [%d, %d]> dX = matmul(x = dY, y = W, transpose_x = txf, transpose_y = tyf)[name = string(\"dX\")];\n"
                // dW = dY^T @ dY
                "        tensor<fp16, [%d, %d]> dW = matmul(x = dY, y = dY, transpose_x = txt, transpose_y = tyf)[name = string(\"dW\")];\n"
                // Concat [dX, dW, pad]
                "        int32 ax = const()[name = string(\"ax\"), val = int32(0)];\n"
                "        bool il = const()[name = string(\"il\"), val = bool(false)];\n"
                "        tensor<fp16, [%d, %d]> pk = concat(values = (dX, dW, pad), axis = ax, interleave = il)[name = string(\"pk\")];\n"
                "        tensor<int32, [4]> r4 = const()[name = string(\"r4\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
                "        tensor<fp16, [1, %d, 1, %d]> o16 = reshape(x = pk, shape = r4)[name = string(\"o16\")];\n"
                "        string c32 = const()[name = string(\"c32\"), val = string(\"fp32\")];\n"
                "        tensor<fp32, [1, %d, 1, %d]> cast_out = cast(dtype = c32, x = o16)[name = string(\"cast_out\")];\n"
                "    } -> (cast_out);\n"
                "}\n",
                ch, sp, ch, sp,
                ch, sp, ch, sp,
                M, K, M, K,
                M, M + K, K, K, K,
                M + K, ch, sp, ch - M - K, sp,
                M, N,
                K, K,
                ch, sp,
                ch, sp, ch, sp,
                ch, sp];

            printf("  dX = dY[%d,%d] @ W[%d,%d] -> [%d,%d]\n", M, K, K, N, M, N);
            printf("  dW = dY^T @ dY -> [%d,%d]\n\n", K, K);

            err = nil;
            id engine = compileAndCreateEngine(bwdMIL, @"z1_backward",
                refContainer, cfg, refDesc, &err);

            if (!engine) {
                printf("  Z1 FAILED: %s\n\n", err ? [[err description] UTF8String] : "unknown");
            } else {
                printf("  Z1: Engine created\n");
                MLMultiArray *inputArr = [[MLMultiArray alloc]
                    initWithShape:@[@1, @(ch), @1, @(sp)]
                    dataType:MLMultiArrayDataTypeFloat32 error:nil];
                float *inPtr = (float *)[inputArr dataPointer];
                fill_random(inPtr, nElems, 0.3f);

                MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{inName: inputArr} error:nil];

                NSError *runErr = nil;
                uint64_t t0 = mach_absolute_time();
                id<MLFeatureProvider> result = runEngine(engine, fp, opts, &runErr);
                double ms = tb_ms(mach_absolute_time() - t0);

                if (runErr || !result) {
                    printf("  Z1 prediction FAILED: %s\n\n",
                           runErr ? [[runErr description] UTF8String] : "nil");
                } else {
                    MLMultiArray *outArr = [result featureValueForName:outName].multiArrayValue;
                    if (outArr) {
                        float *outPtr = (float *)[outArr dataPointer];

                        // CPU: dX = dY @ W
                        float *dY_cpu = inPtr;
                        float *W_cpu = inPtr + M * K;
                        float *dX_cpu = (float *)calloc(M * N, sizeof(float));
                        for (int i = 0; i < M; i++)
                            for (int j = 0; j < N; j++) {
                                float a = 0;
                                for (int k = 0; k < K; k++)
                                    a += dY_cpu[i*K+k] * W_cpu[k*N+j];
                                dX_cpu[i*N+j] = a;
                            }

                        // CPU: dW = dY^T @ dY
                        float *dW_cpu = (float *)calloc(K * K, sizeof(float));
                        for (int i = 0; i < K; i++)
                            for (int j = 0; j < K; j++) {
                                float a = 0;
                                for (int m = 0; m < M; m++)
                                    a += dY_cpu[m*K+i] * dY_cpu[m*K+j];
                                dW_cpu[i*K+j] = a;
                            }

                        print_first("ANE dX", outPtr, M * N);
                        print_first("CPU dX", dX_cpu, M * N);
                        float mad_dx = max_abs_diff(outPtr, dX_cpu, M * N);
                        printf("  dX diff: %.6f, Rel: %.2e\n",
                               mad_dx, mad_dx / (mean_abs(dX_cpu, M*N) + 1e-10f));

                        print_first("ANE dW", outPtr + M*N, K*K);
                        print_first("CPU dW", dW_cpu, K*K);
                        float mad_dw = max_abs_diff(outPtr + M*N, dW_cpu, K * K);
                        printf("  dW diff: %.6f, Rel: %.2e\n",
                               mad_dw, mad_dw / (mean_abs(dW_cpu, K*K) + 1e-10f));
                        printf("  Time: %.3f ms\n", ms);
                        printf("  %s\n\n",
                               (mad_dx < 0.5f && mad_dw < 1.0f)
                               ? "*** Z1 PASSED ***" : "Z1: differences (fp16 precision)");

                        int NN = 100;
                        t0 = mach_absolute_time();
                        for (int i = 0; i < NN; i++) runEngine(engine, fp, opts, nil);
                        printf("  Bench: %.4f ms/eval (%d iters)\n\n",
                               tb_ms(mach_absolute_time() - t0) / NN, NN);

                        free(dX_cpu); free(dW_cpu);
                    }
                }
            }
        }

        printf("================================================================\n");
        printf("  DONE\n");
        printf("================================================================\n");
    }
    return 0;
}
