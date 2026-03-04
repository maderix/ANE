// test_e5_validate.m — Experiments W1-W5: E5 Runtime Validation & Deep API Exploration
// Build: make test_e5_validate && ./test_e5_validate
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

#pragma mark - Helpers

static void dump_all_methods(Class cls, const char *label) {
    if (!cls) { printf("  %s: NOT FOUND\n", label); return; }
    printf("\n--- %s ---\n", label);

    unsigned int mc;
    Method *cm = class_copyMethodList(object_getClass(cls), &mc);
    if (mc > 0) {
        printf("  Class methods (%u):\n", mc);
        for (unsigned int i = 0; i < mc; i++) {
            const char *sel = sel_getName(method_getName(cm[i]));
            const char *enc = method_getTypeEncoding(cm[i]);
            printf("    + %s  [%s]\n", sel, enc ? enc : "?");
        }
    }
    free(cm);

    Method *im = class_copyMethodList(cls, &mc);
    if (mc > 0) {
        printf("  Instance methods (%u):\n", mc);
        for (unsigned int i = 0; i < mc; i++) {
            const char *sel = sel_getName(method_getName(im[i]));
            const char *enc = method_getTypeEncoding(im[i]);
            printf("    - %s  [%s]\n", sel, enc ? enc : "?");
        }
    }
    free(im);

    unsigned int pc;
    objc_property_t *props = class_copyPropertyList(cls, &pc);
    if (pc > 0) {
        printf("  Properties (%u):\n", pc);
        for (unsigned int i = 0; i < pc; i++)
            printf("    %s  [%s]\n", property_getName(props[i]),
                   property_getAttributes(props[i]));
    }
    free(props);

    unsigned int ic;
    Ivar *ivars = class_copyIvarList(cls, &ic);
    if (ic > 0) {
        printf("  Ivars (%u):\n", ic);
        for (unsigned int i = 0; i < ic; i++) {
            const char *n = ivar_getName(ivars[i]);
            const char *t = ivar_getTypeEncoding(ivars[i]);
            printf("    %s  type=%s\n", n, t ? t : "?");
        }
    }
    free(ivars);

    Class super = class_getSuperclass(cls);
    if (super && super != [NSObject class])
        printf("  Superclass: %s\n", class_getName(super));
}

static float max_abs_diff(float *a, float *b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static float mean_abs(float *a, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += fabsf(a[i]);
    return s / n;
}

#pragma mark - Main

int main(int argc, const char *argv[]) {
    (void)argc; (void)argv;
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        printf("================================================================\n");
        printf("  E5 Runtime: Validation & Exhaustive API Documentation\n");
        printf("================================================================\n\n");

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
               "AppleNeuralEngine", RTLD_NOW);

        // ============================================================
        // W2: Exhaustive API Documentation (dump first so we have it)
        // ============================================================
        printf("================================================================\n");
        printf("  W2: Exhaustive E5 Runtime API Documentation\n");
        printf("================================================================\n");

        const char *classNames[] = {
            "MLE5Engine",
            "MLE5ProgramLibrary",
            "MLE5ProgramLibraryOnDeviceAOTCompilationImpl",
            "MLE5ProgramLibraryE5BundleImpl",
            "MLE5ExecutionStreamOperation",
            "MLE5ExecutionStream",
            "MLE5ExecutionStreamPool",
            "MLE5StaticShapeExecutionStreamOperationPool",
            "MLE5RangeShapeExecutionStreamOperationPool",
            "MLE5EnumeratedShapeExecutionStreamOperationPool",
            "MLE5ExecutionStreamOperationPoolFactory",
            "MLE5InputPort",
            "MLE5OutputPort",
            "MLE5InputPortBinder",
            "MLE5OutputPortBinder",
            "MLProgramE5Container",
            NULL
        };
        for (int i = 0; classNames[i]; i++) {
            Class cls = NSClassFromString(
                [NSString stringWithUTF8String:classNames[i]]);
            dump_all_methods(cls, classNames[i]);
        }

        printf("\n--- e5rt_* C API Symbols ---\n");
        const char *cFuncs[] = {
            "e5rt_program_library_create",
            "e5rt_program_library_destroy",
            "e5rt_program_library_compile",
            "e5rt_program_library_get_function",
            "e5rt_program_library_load_function",
            "e5rt_execution_stream_create",
            "e5rt_execution_stream_destroy",
            "e5rt_execution_stream_submit",
            "e5rt_execution_stream_wait",
            "e5rt_execution_stream_execute",
            "e5rt_execution_stream_sync",
            "e5rt_execution_stream_operation_create",
            "e5rt_execution_stream_operation_destroy",
            "e5rt_execution_stream_operation_set_input",
            "e5rt_execution_stream_operation_set_output",
            "e5rt_execution_stream_operation_execute",
            "e5rt_async_event_create",
            "e5rt_async_event_destroy",
            "e5rt_async_event_signal",
            "e5rt_async_event_wait",
            "e5rt_buffer_create",
            "e5rt_buffer_destroy",
            "e5rt_io_port_create",
            "e5rt_io_port_bind",
            "e5rt_context_create",
            "e5rt_init",
            "e5rt_get_version",
            NULL
        };
        for (int i = 0; cFuncs[i]; i++) {
            void *sym = dlsym(RTLD_DEFAULT, cFuncs[i]);
            if (sym) printf("  FOUND: %s at %p\n", cFuncs[i], sym);
        }
        fflush(stdout);

        // ============================================================
        // W1: Output Validation
        // ============================================================
        printf("\n================================================================\n");
        printf("  W1: Output Correctness Validation\n");
        printf("================================================================\n\n");

        int ch = 256, sp = 64;
        NSString *pkgPath = [NSString stringWithFormat:
            @"/tmp/ane_sram_%dch_%dsp.mlpackage", ch, sp];
        if (![[NSFileManager defaultManager] fileExistsAtPath:pkgPath]) {
            printf("  FATAL: %s not found. Run gen_mlpackages.py\n",
                   [pkgPath UTF8String]);
            return 1;
        }

        NSError *err = nil;
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsAll;
        MLPredictionOptions *predOpts = [[MLPredictionOptions alloc] init];
        Class opCls = NSClassFromString(@"MLE5ExecutionStreamOperation");

        NSURL *compiled = [MLModel compileModelAtURL:
            [NSURL fileURLWithPath:pkgPath] error:&err];
        if (err) { printf("  Compile FAILED\n"); return 1; }
        err = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:compiled
                                           configuration:cfg error:&err];
        if (err) { printf("  Load FAILED\n"); return 1; }

        int nElems = 1 * ch * 1 * sp;
        MLMultiArray *inputArr = [[MLMultiArray alloc]
            initWithShape:@[@1, @(ch), @1, @(sp)]
            dataType:MLMultiArrayDataTypeFloat32 error:nil];

        float *inPtr = (float *)[inputArr dataPointer];
        for (int i = 0; i < nElems; i++)
            inPtr[i] = sinf((float)i * 0.01f) * 0.5f;

        NSString *inName = [[[[model modelDescription] inputDescriptionsByName]
            allKeys] firstObject];
        NSString *outName = [[[[model modelDescription] outputDescriptionsByName]
            allKeys] firstObject];
        MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{inName: inputArr} error:nil];

        printf("  Input: %s [1,%d,1,%d], first 5: [%.4f %.4f %.4f %.4f %.4f]\n",
               [inName UTF8String], ch, sp,
               inPtr[0], inPtr[1], inPtr[2], inPtr[3], inPtr[4]);
        printf("  Output: %s\n", [outName UTF8String]);
        fflush(stdout);

        // --- Reference: CoreML sequential prediction ---
        printf("\n  --- W1.1: CoreML reference prediction ---\n");
        err = nil;
        id<MLFeatureProvider> refResult = [model predictionFromFeatures:fp error:&err];
        if (err) { printf("  Prediction FAILED\n"); return 1; }

        MLMultiArray *refOut = [refResult featureValueForName:outName].multiArrayValue;
        float *refPtr = (float *)[refOut dataPointer];
        int outElems = 1;
        for (int d = 0; d < (int)refOut.shape.count; d++)
            outElems *= [refOut.shape[d] intValue];
        printf("  Output shape: [");
        for (int d = 0; d < (int)refOut.shape.count; d++)
            printf("%s%d", d ? "," : "", [refOut.shape[d] intValue]);
        printf("] (%d elements)\n", outElems);
        printf("  First 5 ref: [%.6f %.6f %.6f %.6f %.6f]\n",
               refPtr[0], refPtr[1], refPtr[2], refPtr[3], refPtr[4]);
        printf("  Mean |ref|: %.6f\n", mean_abs(refPtr, outElems));
        fflush(stdout);

        // --- E5 stream prediction ---
        printf("\n  --- W1.2: E5 stream prediction ---\n");

        id e5engine = nil;
        @try { e5engine = [model valueForKey:@"_internalEngine"]; }
        @catch (NSException *e) { (void)e; }
        id progLib = nil;
        @try { progLib = [e5engine valueForKey:@"programLibrary"]; }
        @catch (NSException *e) { (void)e; }
        id streamPool = nil;
        @try { streamPool = [e5engine valueForKey:@"streamPool"]; }
        @catch (NSException *e) { (void)e; }

        id op = ((id(*)(id,SEL,id,id,id,id,id,unsigned long long))objc_msgSend)(
            [opCls alloc],
            @selector(initWithProgramLibrary:functionName:modelDescription:
                configuration:debugLabel:modelSignpostId:),
            progLib, @"main", [model modelDescription], cfg,
            @"validate_op", (unsigned long long)0);

        NSError *plErr = nil;
        BOOL plOk = ((BOOL(*)(id,SEL,NSError**))objc_msgSend)(
            op, @selector(preloadAndReturnError:), &plErr);
        printf("  preload: %s\n", plOk ? "YES" : "NO");
        if (plErr) printf("  Error: %s\n", [[plErr description] UTF8String]);
        fflush(stdout);

        id stream = [streamPool performSelector:@selector(takeOut)];
        Ivar shIvar = class_getInstanceVariable([stream class], "_streamHandle");
        void *sh = (__bridge void *)object_getIvar(stream, shIvar);
        printf("  stream: %p, handle: %p\n", (__bridge void *)stream, sh);

        [stream setValue:@[op] forKey:@"operations"];

        NSError *prepErr = nil;
        BOOL prepOk = ((BOOL(*)(id,SEL,id,id,NSError**))objc_msgSend)(
            op, @selector(prepareForInputFeatures:options:error:),
            fp, predOpts, &prepErr);
        printf("  prepare: %s\n", prepOk ? "YES" : "NO");
        if (prepErr) printf("  Error: %s\n", [[prepErr description] UTF8String]);
        fflush(stdout);

        NSError *execErr = nil;
        BOOL execOk = ((BOOL(*)(id,SEL,void*,NSError**))objc_msgSend)(
            stream, @selector(_executeStream:error:), sh, &execErr);
        printf("  execute: %s\n", execOk ? "YES" : "NO");
        if (execErr) printf("  Error: %s\n", [[execErr description] UTF8String]);
        fflush(stdout);

        // Read output from the operation
        printf("\n  --- W1.3: Read E5 output features ---\n");
        fflush(stdout);
        id e5Result = nil;
        @try {
            e5Result = [op valueForKey:@"outputFeatures"];
            printf("  outputFeatures: %s\n",
                   e5Result ? [NSStringFromClass([e5Result class]) UTF8String]
                            : "nil");
        } @catch (NSException *ex) {
            printf("  outputFeatures EXCEPTION: %s\n",
                   [[ex reason] UTF8String]);
        }

        if (e5Result && [e5Result conformsToProtocol:@protocol(MLFeatureProvider)]) {
            MLMultiArray *e5Out = [(id<MLFeatureProvider>)e5Result
                featureValueForName:outName].multiArrayValue;
            if (e5Out) {
                float *e5Ptr = (float *)[e5Out dataPointer];
                printf("  E5 first 5: [%.6f %.6f %.6f %.6f %.6f]\n",
                       e5Ptr[0], e5Ptr[1], e5Ptr[2], e5Ptr[3], e5Ptr[4]);
                printf("  Mean |e5|: %.6f\n", mean_abs(e5Ptr, outElems));

                float mad = max_abs_diff(refPtr, e5Ptr, outElems);
                printf("  Max abs diff: %.8f\n", mad);
                printf("  Relative error: %.2e\n",
                       mad / (mean_abs(refPtr, outElems) + 1e-10f));

                if (mad < 1e-3f) {
                    printf("  *** VALIDATION PASSED: outputs match ***\n");
                } else if (mad < 1e-1f) {
                    printf("  VALIDATION WARNING: small differences (FP16 expected)\n");
                } else {
                    printf("  VALIDATION FAILED: outputs diverge!\n");
                }
            } else {
                printf("  E5 output array is nil for key '%s'\n",
                       [outName UTF8String]);

                NSArray *ofNames = [(id<MLFeatureProvider>)e5Result
                    featureNames].allObjects;
                printf("  Available features: %s\n",
                       [[ofNames description] UTF8String]);
            }
        } else {
            printf("  Cannot read output features\n");
        }

        // Also read output via outputPorts
        printf("\n  --- W1.4: Read via output ports ---\n");
        fflush(stdout);
        @try {
            id outPorts = [op valueForKey:@"outputPorts"];
            printf("  outputPorts: %s (count=%lu)\n",
                   outPorts ? [NSStringFromClass([outPorts class]) UTF8String]
                            : "nil",
                   outPorts ? (unsigned long)[(NSArray *)outPorts count] : 0);

            if (outPorts && [(NSArray *)outPorts count] > 0) {
                for (NSUInteger pi = 0; pi < [(NSArray *)outPorts count]; pi++) {
                    id port = [(NSArray *)outPorts objectAtIndex:pi];
                    printf("    Port[%lu]: %s\n", (unsigned long)pi,
                           [[port description] UTF8String]);
                    @try {
                        id portName = [port valueForKey:@"name"];
                        printf("      name: %s\n",
                               portName ? [(NSString *)portName UTF8String] : "nil");
                    } @catch (NSException *ex) { (void)ex; }
                    @try {
                        id portFD = [port valueForKey:@"featureDescription"];
                        printf("      featureDescription: %s\n",
                               portFD ? [[portFD description] UTF8String] : "nil");
                    } @catch (NSException *ex) { (void)ex; }
                    @try {
                        id binder = [port valueForKey:@"binder"];
                        printf("      binder: %s\n",
                               binder ? [NSStringFromClass([binder class])
                                            UTF8String] : "nil");
                        if (binder) {
                            @try {
                                id fv = [binder valueForKey:@"featureValue"];
                                printf("      featureValue: %s\n",
                                       fv ? [NSStringFromClass([fv class])
                                                UTF8String] : "nil");
                                if (fv) {
                                    MLMultiArray *ma = [(MLFeatureValue *)fv
                                        multiArrayValue];
                                    if (ma) {
                                        float *ptr = (float *)[ma dataPointer];
                                        printf("      first 5: [%.6f %.6f %.6f"
                                               " %.6f %.6f]\n",
                                               ptr[0], ptr[1], ptr[2],
                                               ptr[3], ptr[4]);
                                        float mad2 = max_abs_diff(refPtr, ptr,
                                            outElems);
                                        printf("      Max abs diff vs ref: %.8f\n",
                                               mad2);
                                    }
                                }
                            } @catch (NSException *ex) {
                                printf("      featureValue EXCEPTION: %s\n",
                                       [[ex reason] UTF8String]);
                            }
                        }
                    } @catch (NSException *ex) { (void)ex; }
                }
            }
        } @catch (NSException *ex) {
            printf("  outputPorts EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // Also read input ports
        printf("\n  --- W1.5: Inspect input ports ---\n");
        fflush(stdout);
        @try {
            id inPorts = [op valueForKey:@"inputPorts"];
            printf("  inputPorts: %s (count=%lu)\n",
                   inPorts ? [NSStringFromClass([inPorts class]) UTF8String]
                           : "nil",
                   inPorts ? (unsigned long)[(NSArray *)inPorts count] : 0);
            if (inPorts) {
                for (NSUInteger pi = 0; pi < [(NSArray *)inPorts count]; pi++) {
                    id port = [(NSArray *)inPorts objectAtIndex:pi];
                    printf("    Port[%lu]: %s\n", (unsigned long)pi,
                           [[port description] UTF8String]);
                    @try {
                        printf("      name: %s\n",
                               [[(id)[port valueForKey:@"name"] description]
                                   UTF8String]);
                        printf("      portHandle: %p\n",
                               (__bridge void *)[port valueForKey:@"portHandle"]);
                    } @catch (NSException *ex) { (void)ex; }
                    @try {
                        id binder = [port valueForKey:@"binder"];
                        if (binder) {
                            printf("      binder: %s\n",
                                   [NSStringFromClass([binder class]) UTF8String]);
                            printf("      bindingMode: %d\n",
                                   ((char(*)(id,SEL))objc_msgSend)(
                                       binder, @selector(bindingMode)));
                            id dfv = nil;
                            @try {
                                dfv = [binder valueForKey:@"directlyBoundFeatureValue"];
                            } @catch (NSException *ex) { (void)ex; }
                            printf("      directlyBound: %s\n",
                                   dfv ? "YES" : "NO");
                        }
                    } @catch (NSException *ex) { (void)ex; }
                }
            }
        } @catch (NSException *ex) {
            printf("  inputPorts EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // Return stream
        [stream setValue:@[op] forKey:@"operations"];
        ((void(*)(id,SEL,id))objc_msgSend)(
            streamPool, @selector(putBack:), stream);

        // ============================================================
        // W1.6: Multi-op output validation
        // ============================================================
        printf("\n  --- W1.6: Multi-op output validation ---\n");
        fflush(stdout);

        {
            NSString *pkg2Path = @"/tmp/ane_sram_512ch_64sp.mlpackage";
            err = nil;
            NSURL *c2 = [MLModel compileModelAtURL:
                [NSURL fileURLWithPath:pkg2Path] error:&err];
            if (err) { printf("  Compile2 FAILED\n"); goto skip_multiop; }
            err = nil;
            MLModel *model2 = [MLModel modelWithContentsOfURL:c2
                                                 configuration:cfg error:&err];
            if (err) { printf("  Load2 FAILED\n"); goto skip_multiop; }
            int ch2 = 512;
            int nElems2 = 1 * ch2 * 1 * sp;
            MLMultiArray *inputArr2 = [[MLMultiArray alloc]
                initWithShape:@[@1, @(ch2), @1, @(sp)]
                dataType:MLMultiArrayDataTypeFloat32 error:nil];
            float *in2Ptr = (float *)[inputArr2 dataPointer];
            for (int i = 0; i < nElems2; i++)
                in2Ptr[i] = cosf((float)i * 0.02f) * 0.3f;

            NSString *in2Name = [[[[model2 modelDescription] inputDescriptionsByName]
                allKeys] firstObject];
            NSString *out2Name = [[[[model2 modelDescription] outputDescriptionsByName]
                allKeys] firstObject];
            MLDictionaryFeatureProvider *fp2 = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{in2Name: inputArr2} error:nil];

            // Reference predictions
            err = nil;
            id<MLFeatureProvider> ref1 = [model predictionFromFeatures:fp error:&err];
            err = nil;
            id<MLFeatureProvider> ref2 = [model2 predictionFromFeatures:fp2 error:&err];
            float *ref1Ptr = (float *)[[ref1 featureValueForName:outName].multiArrayValue dataPointer];
            float *ref2Ptr = (float *)[[ref2 featureValueForName:out2Name].multiArrayValue dataPointer];

            // E5 multi-op stream
            id e5_2 = nil;
            @try { e5_2 = [model2 valueForKey:@"_internalEngine"]; }
            @catch (NSException *e) { (void)e; }
            id pLib2 = nil;
            @try { pLib2 = [e5_2 valueForKey:@"programLibrary"]; }
            @catch (NSException *e) { (void)e; }

            id op1 = ((id(*)(id,SEL,id,id,id,id,id,unsigned long long))objc_msgSend)(
                [opCls alloc],
                @selector(initWithProgramLibrary:functionName:modelDescription:
                    configuration:debugLabel:modelSignpostId:),
                progLib, @"main", [model modelDescription], cfg,
                @"val_op1", (unsigned long long)0);
            id op2 = ((id(*)(id,SEL,id,id,id,id,id,unsigned long long))objc_msgSend)(
                [opCls alloc],
                @selector(initWithProgramLibrary:functionName:modelDescription:
                    configuration:debugLabel:modelSignpostId:),
                pLib2, @"main", [model2 modelDescription], cfg,
                @"val_op2", (unsigned long long)0);

            ((BOOL(*)(id,SEL,NSError**))objc_msgSend)(op1, @selector(preloadAndReturnError:), nil);
            ((BOOL(*)(id,SEL,NSError**))objc_msgSend)(op2, @selector(preloadAndReturnError:), nil);

            id stream2 = [streamPool performSelector:@selector(takeOut)];
            Ivar shIvar2 = class_getInstanceVariable([stream2 class], "_streamHandle");
            void *sh2 = (__bridge void *)object_getIvar(stream2, shIvar2);

            [stream2 setValue:@[op1, op2] forKey:@"operations"];

            ((BOOL(*)(id,SEL,id,id,NSError**))objc_msgSend)(
                op1, @selector(prepareForInputFeatures:options:error:),
                fp, predOpts, nil);
            ((BOOL(*)(id,SEL,id,id,NSError**))objc_msgSend)(
                op2, @selector(prepareForInputFeatures:options:error:),
                fp2, predOpts, nil);

            NSError *mErr = nil;
            BOOL mOk = ((BOOL(*)(id,SEL,void*,NSError**))objc_msgSend)(
                stream2, @selector(_executeStream:error:), sh2, &mErr);
            printf("  Multi-op execute: %s\n", mOk ? "YES" : "NO");
            if (mErr) printf("  Error: %s\n", [[mErr description] UTF8String]);
            fflush(stdout);

            if (mOk) {
                // Read outputs
                @try {
                    id out1 = [op1 valueForKey:@"outputFeatures"];
                    id out2 = [op2 valueForKey:@"outputFeatures"];

                    if (out1 && out2) {
                        MLMultiArray *ma1 = [(id<MLFeatureProvider>)out1
                            featureValueForName:outName].multiArrayValue;
                        MLMultiArray *ma2 = [(id<MLFeatureProvider>)out2
                            featureValueForName:out2Name].multiArrayValue;

                        if (ma1 && ma2) {
                            float *p1 = (float *)[ma1 dataPointer];
                            float *p2 = (float *)[ma2 dataPointer];

                            float mad1 = max_abs_diff(ref1Ptr, p1, outElems);
                            float mad2 = max_abs_diff(ref2Ptr, p2, nElems2);

                            printf("  Op1 max diff: %.8f  (mean_ref=%.6f)\n",
                                   mad1, mean_abs(ref1Ptr, outElems));
                            printf("  Op2 max diff: %.8f  (mean_ref=%.6f)\n",
                                   mad2, mean_abs(ref2Ptr, nElems2));

                            if (mad1 < 1e-3f && mad2 < 1e-3f) {
                                printf("  *** MULTI-OP VALIDATION PASSED ***\n");
                            } else {
                                printf("  MULTI-OP VALIDATION: differences detected\n");
                            }
                        } else {
                            printf("  Could not extract MLMultiArray from outputs\n");
                        }
                    } else {
                        printf("  outputFeatures nil for op1 or op2\n");
                    }
                } @catch (NSException *ex) {
                    printf("  Output read EXCEPTION: %s\n",
                           [[ex reason] UTF8String]);
                }
            }

            [stream2 setValue:@[op1] forKey:@"operations"];
            ((void(*)(id,SEL,id))objc_msgSend)(
                streamPool, @selector(putBack:), stream2);
        }
skip_multiop:

        // ============================================================
        // W4: Async stream submission
        // ============================================================
        printf("\n================================================================\n");
        printf("  W4: Async Stream Submission\n");
        printf("================================================================\n\n");
        fflush(stdout);

        {
            id asyncStream = [streamPool performSelector:@selector(takeOut)];
            Ivar ashIvar = class_getInstanceVariable([asyncStream class], "_streamHandle");
            void *ash = (__bridge void *)object_getIvar(asyncStream, ashIvar);

            id asyncOp = ((id(*)(id,SEL,id,id,id,id,id,unsigned long long))
                objc_msgSend)([opCls alloc],
                @selector(initWithProgramLibrary:functionName:modelDescription:
                    configuration:debugLabel:modelSignpostId:),
                progLib, @"main", [model modelDescription], cfg,
                @"async_op", (unsigned long long)0);
            ((BOOL(*)(id,SEL,NSError**))objc_msgSend)(
                asyncOp, @selector(preloadAndReturnError:), nil);
            [asyncStream setValue:@[asyncOp] forKey:@"operations"];

            ((BOOL(*)(id,SEL,id,id,NSError**))objc_msgSend)(
                asyncOp, @selector(prepareForInputFeatures:options:error:),
                fp, predOpts, nil);

            // Try async submission
            __block BOOL asyncDone = NO;
            __block double asyncMs = 0;
            uint64_t asyncT0 = mach_absolute_time();

            @try {
                // prepareAsyncSubmissionForInputFeatures
                NSError *asyncPrepErr = nil;
                BOOL asyncPrepOk = ((BOOL(*)(id,SEL,id,id,NSError**))
                    objc_msgSend)(asyncStream,
                    @selector(prepareAsyncSubmissionForInputFeatures:options:error:),
                    fp, predOpts, &asyncPrepErr);
                printf("  prepareAsyncSubmission: %s\n",
                       asyncPrepOk ? "YES" : "NO");
                if (asyncPrepErr) printf("  Error: %s\n",
                    [[asyncPrepErr description] UTF8String]);
                fflush(stdout);

                if (asyncPrepOk) {
                    ((void(*)(id,SEL,void(^)(void)))objc_msgSend)(
                        asyncStream, @selector(submitWithCompletionHandler:),
                        ^{
                            asyncMs = tb_ms(mach_absolute_time() - asyncT0);
                            asyncDone = YES;
                        });
                    printf("  Submitted async, waiting...\n");
                    fflush(stdout);

                    for (int w = 0; w < 100 && !asyncDone; w++)
                        usleep(1000);

                    printf("  Async completed: %s (%.3f ms)\n",
                           asyncDone ? "YES" : "TIMEOUT", asyncMs);
                    fflush(stdout);

                    if (asyncDone) {
                        // Benchmark async vs sync
                        int N = 200;

                        // Sync benchmark
                        uint64_t t0 = mach_absolute_time();
                        for (int i = 0; i < N; i++) {
                            ((BOOL(*)(id,SEL,id,id,NSError**))objc_msgSend)(
                                asyncOp,
                                @selector(prepareForInputFeatures:options:error:),
                                fp, predOpts, nil);
                            ((BOOL(*)(id,SEL,void*,NSError**))objc_msgSend)(
                                asyncStream,
                                @selector(_executeStream:error:), ash, nil);
                        }
                        double syncMs = tb_ms(mach_absolute_time() - t0) / N;

                        // Async benchmark
                        t0 = mach_absolute_time();
                        for (int i = 0; i < N; i++) {
                            ((BOOL(*)(id,SEL,id,id,NSError**))objc_msgSend)(
                                asyncOp,
                                @selector(prepareForInputFeatures:options:error:),
                                fp, predOpts, nil);
                            ((BOOL(*)(id,SEL,id,id,NSError**))objc_msgSend)(
                                asyncStream,
                                @selector(prepareAsyncSubmissionForInputFeatures:
                                    options:error:),
                                fp, predOpts, nil);

                            __block BOOL done = NO;
                            ((void(*)(id,SEL,void(^)(void)))objc_msgSend)(
                                asyncStream,
                                @selector(submitWithCompletionHandler:),
                                ^{ done = YES; });
                            while (!done) usleep(100);
                        }
                        double asyncBenchMs = tb_ms(mach_absolute_time() - t0) / N;

                        printf("  Sync: %.4f ms/eval\n", syncMs);
                        printf("  Async (wait): %.4f ms/eval\n", asyncBenchMs);
                    }
                }
            } @catch (NSException *ex) {
                printf("  Async EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            [asyncStream setValue:@[asyncOp] forKey:@"operations"];
            ((void(*)(id,SEL,id))objc_msgSend)(
                streamPool, @selector(putBack:), asyncStream);
        }

        // ============================================================
        // W5: Port-Based Data Flow
        // ============================================================
        printf("\n================================================================\n");
        printf("  W5: Port-Based Data Flow Investigation\n");
        printf("================================================================\n\n");
        fflush(stdout);

        {
            id portOp = ((id(*)(id,SEL,id,id,id,id,id,unsigned long long))
                objc_msgSend)([opCls alloc],
                @selector(initWithProgramLibrary:functionName:modelDescription:
                    configuration:debugLabel:modelSignpostId:),
                progLib, @"main", [model modelDescription], cfg,
                @"port_op", (unsigned long long)0);
            ((BOOL(*)(id,SEL,NSError**))objc_msgSend)(
                portOp, @selector(preloadAndReturnError:), nil);

            // Inspect ports before prepare
            printf("  --- Before prepare ---\n");
            @try {
                id inP = [portOp valueForKey:@"inputPorts"];
                id outP = [portOp valueForKey:@"outputPorts"];
                id stP = [portOp valueForKey:@"statePorts"];
                printf("  inputPorts: %lu, outputPorts: %lu, statePorts: %lu\n",
                       inP ? (unsigned long)[(NSArray *)inP count] : 0,
                       outP ? (unsigned long)[(NSArray *)outP count] : 0,
                       stP ? (unsigned long)[(NSArray *)stP count] : 0);

                if (inP) {
                    for (id p in (NSArray *)inP) {
                        printf("    in: %s  portHandle=%p  name=%s\n",
                               [NSStringFromClass([p class]) UTF8String],
                               (__bridge void *)[p valueForKey:@"portHandle"],
                               [[(id)[p valueForKey:@"name"] description] UTF8String]);
                    }
                }
                if (outP) {
                    for (id p in (NSArray *)outP) {
                        printf("    out: %s  portHandle=%p  name=%s\n",
                               [NSStringFromClass([p class]) UTF8String],
                               (__bridge void *)[p valueForKey:@"portHandle"],
                               [[(id)[p valueForKey:@"name"] description] UTF8String]);
                        @try {
                            id fd = [p valueForKey:@"featureDescription"];
                            if (fd) printf("         featureDesc: %s\n",
                                           [[fd description] UTF8String]);
                        } @catch (NSException *ex) { (void)ex; }
                    }
                }
            } @catch (NSException *ex) {
                printf("  Port inspection EXCEPTION: %s\n",
                       [[ex reason] UTF8String]);
            }

            // Prepare and inspect after
            ((BOOL(*)(id,SEL,id,id,NSError**))objc_msgSend)(
                portOp, @selector(prepareForInputFeatures:options:error:),
                fp, predOpts, nil);

            printf("\n  --- After prepare ---\n");
            @try {
                id inP = [portOp valueForKey:@"inputPorts"];
                if (inP) {
                    for (id p in (NSArray *)inP) {
                        id binder = [p valueForKey:@"binder"];
                        BOOL directBound = ((BOOL(*)(id,SEL))objc_msgSend)(
                            p, @selector(boundFeatureDirectly));
                        printf("    in: name=%s  directBound=%s  binder=%s\n",
                               [[(id)[p valueForKey:@"name"] description] UTF8String],
                               directBound ? "YES" : "NO",
                               binder ? [NSStringFromClass([binder class])
                                            UTF8String] : "nil");
                        if (binder) {
                            char mode = ((char(*)(id,SEL))objc_msgSend)(
                                binder, @selector(bindingMode));
                            printf("         bindingMode=%d\n", (int)mode);
                        }
                    }
                }
                id outP = [portOp valueForKey:@"outputPorts"];
                if (outP) {
                    for (id p in (NSArray *)outP) {
                        BOOL directBound = ((BOOL(*)(id,SEL))objc_msgSend)(
                            p, @selector(boundFeatureDirectly));
                        BOOL obDirectBound = ((BOOL(*)(id,SEL))objc_msgSend)(
                            p, @selector(outputBackingWasDirectlyBound));
                        printf("    out: name=%s  directBound=%s"
                               "  outputBackingDirectBound=%s\n",
                               [[(id)[p valueForKey:@"name"] description] UTF8String],
                               directBound ? "YES" : "NO",
                               obDirectBound ? "YES" : "NO");
                        id binder = [p valueForKey:@"binder"];
                        if (binder) {
                            printf("         binder: %s\n",
                                   [NSStringFromClass([binder class]) UTF8String]);
                            @try {
                                id ob = [binder valueForKey:@"outputBacking"];
                                printf("         outputBacking: %s\n",
                                       ob ? [NSStringFromClass([ob class])
                                                UTF8String] : "nil");
                            } @catch (NSException *ex) { (void)ex; }
                        }
                    }
                }
            } @catch (NSException *ex) {
                printf("  Post-prepare EXCEPTION: %s\n",
                       [[ex reason] UTF8String]);
            }
        }

        // ============================================================
        // Summary
        // ============================================================
        printf("\n================================================================\n");
        printf("  SUMMARY\n");
        printf("================================================================\n");
        printf("  W1: Output validation          -- see above\n");
        printf("  W2: API documentation           -- complete (all classes dumped)\n");
        printf("  W4: Async submission            -- see above\n");
        printf("  W5: Port data flow              -- see above\n");
        printf("================================================================\n");
        printf("\nDone.\n");
    }
    return 0;
}
