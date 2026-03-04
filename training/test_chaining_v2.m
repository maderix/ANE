// test_chaining_v2.m — Deep exploration of _ANEChainingRequest and related APIs
// Phases:
//   1. Dump unexplored ANE classes (mapper, buffer, output sets, etc.)
//   2. Query compiled model for symbol names and I/O mapping
//   3. Try _ANEProgramIOSurfacesMapper and _ANEBuffer for indexed IOSurfaces
//   4. Retry ChainingRequest with indexed surfaces
//   5. Test real-time eval path and perfStatsMask
//   6. Print structured summary
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static int g_fp16_io = 0;

#pragma mark — Helpers

static void dump_class(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("  %s: NOT FOUND\n", name); return; }
    printf("\n=== %s ===\n", name);

    unsigned int count;
    Method *methods = class_copyMethodList(object_getClass(cls), &count);
    if (count) printf("  Class methods (%u):\n", count);
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    + %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);

    methods = class_copyMethodList(cls, &count);
    if (count) printf("  Instance methods (%u):\n", count);
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    - %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);

    unsigned int pcount;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    if (pcount) printf("  Properties (%u):\n", pcount);
    for (unsigned int i = 0; i < pcount; i++) {
        const char *pname = property_getName(props[i]);
        const char *pattr = property_getAttributes(props[i]);
        printf("    @property %s  [%s]\n", pname, pattr ? pattr : "?");
    }
    free(props);
}

static void try_alloc_init(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) return;
    @try {
        id obj = [[cls alloc] init];
        printf("  %s alloc/init: %s\n", name,
               obj ? [[obj description] UTF8String] : "nil");
    } @catch (NSException *ex) {
        printf("  %s alloc/init EXCEPTION: %s\n", name, [[ex reason] UTF8String]);
    }
}

static void dump_all_properties(id obj, Class cls) {
    if (!obj) return;
    unsigned int pcount;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    for (unsigned int i = 0; i < pcount; i++) {
        const char *pname = property_getName(props[i]);
        @try {
            id val = [obj valueForKey:[NSString stringWithUTF8String:pname]];
            printf("    %s = %s\n", pname, val ? [[val description] UTF8String] : "nil");
        } @catch (NSException *ex) {
            printf("    %s = <exception: %s>\n", pname, [[ex reason] UTF8String]);
        }
    }
    free(props);
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

typedef struct { id model; IOSurfaceRef ioIn, ioOut; NSString *tmpDir; } CompiledKernel;

static NSString *gen_conv_mil(int ch, int sp) {
    if (g_fp16_io) {
        return [NSString stringWithFormat:
            @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
            "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
            "        tensor<string, []> pt = const()[name=tensor<string, []>(\"pt\"), val=tensor<string, []>(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name=tensor<string, []>(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, [4]> pd = const()[name=tensor<string, []>(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
            "        tensor<int32, [2]> dl = const()[name=tensor<string, []>(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, []> gr = const()[name=tensor<string, []>(\"gr\"), val=tensor<int32, []>(1)];\n"
            "        tensor<fp16, [%d,%d,1,1]> W = const()[name=tensor<string, []>(\"W\"), "
            "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/weight.bin\"), offset=tensor<uint64, []>(64)))];\n"
            "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
            "[name=tensor<string, []>(\"conv\")];\n"
            "    } -> (y);\n}\n", ch, sp, ch, ch, ch, ch, ch, sp];
    }
    return [NSString stringWithFormat:
        @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
        "    func main<ios16>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        tensor<string, []> pt = const()[name=tensor<string, []>(\"pt\"), val=tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=tensor<string, []>(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=tensor<string, []>(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=tensor<string, []>(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, []> gr = const()[name=tensor<string, []>(\"gr\"), val=tensor<int32, []>(1)];\n"
        "        tensor<string, []> to16 = const()[name=tensor<string, []>(\"to16\"), val=tensor<string, []>(\"fp16\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=tensor<string, []>(\"cin\")];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=tensor<string, []>(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/weight.bin\"), offset=tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
        "[name=tensor<string, []>(\"conv\")];\n"
        "        tensor<string, []> to32 = const()[name=tensor<string, []>(\"to32\"), val=tensor<string, []>(\"fp32\")];\n"
        "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=tensor<string, []>(\"cout\")];\n"
        "    } -> (y);\n}\n", ch, sp, ch, sp, ch, ch, ch, ch, ch, sp, ch, sp];
}

static CompiledKernel compile_kernel(Class gD, Class gI, int ch, int sp, NSData *wdata) {
    CompiledKernel k = {0};
    NSFileManager *fm = [NSFileManager defaultManager];

    NSString *mil = gen_conv_mil(ch, sp);
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(gD,
        @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(gI, @selector(inMemoryModelWithDescriptor:), desc);

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        if (!g_fp16_io) {
            printf("  fp32 compile failed, retrying with fp16 I/O\n");
            g_fp16_io = 1;
            [fm removeItemAtPath:td error:nil];
            return compile_kernel(gD, gI, ch, sp, wdata);
        }
        printf("  Compile failed: %s\n", e ? [[e description] UTF8String] : "unknown");
        return k;
    }

    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);

    int bpe = g_fp16_io ? 2 : 4;
    k.model = mdl;
    k.ioIn = make_surface(ch * sp * bpe);
    k.ioOut = make_surface(ch * sp * bpe);
    k.tmpDir = td;
    return k;
}

#pragma mark — Result tracking

typedef struct {
    bool phase1_done;
    int classes_found;
    int classes_missing;

    bool phase2_done;
    bool has_input_symbols;
    bool has_output_symbols;
    bool has_program_handle;

    bool phase3_done;
    bool mapper_works;
    bool buffer_works;
    bool got_symbol_index;

    bool phase4_done;
    bool validate_passed;
    bool chaining_executed;
    double sequential_ms;
    double chained_ms;

    bool phase5_done;
    bool realtime_eval_works;
    bool perfstats_works;
    uint64_t hw_exec_time_ns;

    bool phase7_done;
    bool outputsets_with_stats_works;
    bool chaining_with_stats_works;

    bool phase8_done;
    bool disk_model_loads;
    bool disk_model_has_symbols;

    bool phase9_done;
    bool process_request_works;
    double process_request_ms;

    bool phase10_done;
    bool shared_events_exist;

    double rt_eval_ms;
    double std_eval_ms;
    double direct_eval_ms;
} Results;

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        Results R = {0};

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  ANE ChainingRequest Deep Exploration v2               ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        Class gD     = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class gI     = NSClassFromString(@"_ANEInMemoryModel");
        Class gAR    = NSClassFromString(@"_ANERequest");
        Class gAIO   = NSClassFromString(@"_ANEIOSurfaceObject");
        Class gClient= NSClassFromString(@"_ANEClient");
        Class gChain = NSClassFromString(@"_ANEChainingRequest");

        if (!gD || !gI || !gAR || !gAIO) {
            printf("FATAL: Core ANE classes not found\n");
            return 1;
        }

        // =====================================================================
        // PHASE 1: Dump all unexplored ANE classes
        // =====================================================================
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  PHASE 1: Class Introspection (unexplored classes)\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        const char *explore_classes[] = {
            "_ANEProgramIOSurfacesMapper",
            "_ANEBuffer",
            "_ANEProgramForEvaluation",
            "_ANEIOSurfaceOutputSets",
            "_ANEInputBuffersReady",
            "_ANEOutputSetEnqueue",
            "_ANEModelInstanceParameters",
            "_ANEDeviceController",
            "_ANEQoSMapper",
            NULL
        };

        for (int i = 0; explore_classes[i]; i++) {
            Class cls = NSClassFromString([NSString stringWithUTF8String:explore_classes[i]]);
            if (cls) R.classes_found++;
            else R.classes_missing++;
            dump_class(explore_classes[i]);
        }

        printf("\n  --- Alloc/init tests ---\n");
        for (int i = 0; explore_classes[i]; i++) {
            try_alloc_init(explore_classes[i]);
        }

        printf("\n  --- Also dump _ANEIOSurfaceObject (for symbolIndex) ---\n");
        dump_class("_ANEIOSurfaceObject");
        dump_class("_ANEChainingRequest");
        dump_class("_ANEClient");

        R.phase1_done = true;
        printf("\n  Phase 1 complete: %d classes found, %d missing\n",
               R.classes_found, R.classes_missing);

        // =====================================================================
        // Compile test kernel (shared by subsequent phases)
        // =====================================================================
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  Compiling test kernels...\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        int CH = 64, SP = 32;

        _Float16 *w = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) w[i*CH+i] = (_Float16)0.5f;
        int ws = CH*CH*2, tot = 128+ws;
        uint8_t *blob = (uint8_t*)calloc(tot, 1);
        blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
        *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
        memcpy(blob+128, w, ws);
        NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];
        free(w);

        CompiledKernel k1 = compile_kernel(gD, gI, CH, SP, wdata);
        CompiledKernel k2 = compile_kernel(gD, gI, CH, SP, wdata);

        if (!k1.model || !k2.model) {
            printf("FATAL: Failed to compile test kernels\n");
            return 1;
        }
        printf("  Kernel 1: compiled and loaded (fp16_io=%d)\n", g_fp16_io);
        printf("  Kernel 2: compiled and loaded\n");

        int bpe = g_fp16_io ? 2 : 4;
        int ioBytes = CH * SP * bpe;

        IOSurfaceLock(k1.ioIn, 0, NULL);
        if (g_fp16_io) {
            _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(k1.ioIn);
            for (int i = 0; i < CH*SP; i++) inp[i] = (_Float16)1.0f;
        } else {
            float *inp = (float*)IOSurfaceGetBaseAddress(k1.ioIn);
            for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f;
        }
        IOSurfaceUnlock(k1.ioIn, 0, NULL);

        // =====================================================================
        // PHASE 2: Symbol Name Discovery
        // =====================================================================
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  PHASE 2: Symbol Name Discovery\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        const char *model_keys[] = {
            "inputSymbolNames", "outputSymbolNames",
            "programHandle", "intermediateBufferHandle",
            "hexStringIdentifier", "modelDescription",
            "inputFeatureNames", "outputFeatureNames",
            "perfStatsMask",
            "numberOfInputs", "numberOfOutputs",
            "compiledModelPath", "compiledModelURL",
            "modelPath", "modelURL",
            NULL
        };

        printf("\n  --- _ANEInMemoryModel properties (k1) ---\n");
        for (int i = 0; model_keys[i]; i++) {
            NSString *key = [NSString stringWithUTF8String:model_keys[i]];
            @try {
                id val = [k1.model valueForKey:key];
                const char *desc = val ? [[val description] UTF8String] : "nil";
                size_t len = strlen(desc);
                if (len > 200) {
                    printf("    %s = %.200s... (truncated, %zu chars)\n", model_keys[i], desc, len);
                } else {
                    printf("    %s = %s\n", model_keys[i], desc);
                }
                if (strcmp(model_keys[i], "inputSymbolNames") == 0 && val) R.has_input_symbols = true;
                if (strcmp(model_keys[i], "outputSymbolNames") == 0 && val) R.has_output_symbols = true;
                if (strcmp(model_keys[i], "programHandle") == 0 && val) R.has_program_handle = true;
            } @catch (NSException *ex) {
                printf("    %s = <KVC exception: %s>\n", model_keys[i], [[ex reason] UTF8String]);
            }
        }

        printf("\n  --- Full property dump of _ANEInMemoryModel ---\n");
        {
            unsigned int pcount;
            objc_property_t *props = class_copyPropertyList(gI, &pcount);
            printf("  (%u properties declared on class)\n", pcount);
            for (unsigned int i = 0; i < pcount; i++) {
                const char *pname = property_getName(props[i]);
                const char *pattr = property_getAttributes(props[i]);
                printf("    @property %s  [%s]\n", pname, pattr ? pattr : "?");
                @try {
                    id val = [k1.model valueForKey:[NSString stringWithUTF8String:pname]];
                    const char *desc = val ? [[val description] UTF8String] : "nil";
                    size_t len = strlen(desc);
                    if (len > 200) {
                        printf("      value = %.200s... (truncated)\n", desc);
                    } else {
                        printf("      value = %s\n", desc);
                    }
                } @catch (NSException *ex) {
                    printf("      value = <exception: %s>\n", [[ex reason] UTF8String]);
                }
            }
            free(props);
        }

        printf("\n  --- Walk superclasses for inherited properties ---\n");
        {
            Class c = gI;
            while (c) {
                const char *cname = class_getName(c);
                if (strstr(cname, "ANE")) {
                    unsigned int pcount;
                    objc_property_t *props = class_copyPropertyList(c, &pcount);
                    if (pcount > 0) {
                        printf("  %s (%u props):\n", cname, pcount);
                        for (unsigned int i = 0; i < pcount; i++) {
                            const char *pname = property_getName(props[i]);
                            printf("    @property %s\n", pname);
                        }
                    }
                    free(props);
                }
                c = class_getSuperclass(c);
            }
        }

        printf("\n  --- _ANEIOSurfaceObject introspection ---\n");
        {
            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO,
                @selector(objectWithIOSurface:), k1.ioIn);
            printf("  wI description: %s\n", [[wI description] UTF8String]);

            unsigned int pcount;
            objc_property_t *props = class_copyPropertyList(gAIO, &pcount);
            for (unsigned int i = 0; i < pcount; i++) {
                const char *pname = property_getName(props[i]);
                @try {
                    id val = [wI valueForKey:[NSString stringWithUTF8String:pname]];
                    printf("    %s = %s\n", pname, val ? [[val description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("    %s = <exception: %s>\n", pname, [[ex reason] UTF8String]);
                }
            }
            free(props);

            @try {
                id symIdx = [wI valueForKey:@"symbolIndex"];
                printf("  symbolIndex (KVC): %s\n", symIdx ? [[symIdx description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("  symbolIndex (KVC): <exception: %s>\n", [[ex reason] UTF8String]);
            }
        }

        R.phase2_done = true;
        printf("\n  Phase 2 complete: inputSymbols=%s outputSymbols=%s programHandle=%s\n",
               R.has_input_symbols ? "YES" : "NO",
               R.has_output_symbols ? "YES" : "NO",
               R.has_program_handle ? "YES" : "NO");

        // Create IOSurface wrapper objects (shared across Phases 3-5)
        id wI1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), k1.ioIn);
        id wO1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), k1.ioOut);
        id wI2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), k2.ioIn);
        id wO2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), k2.ioOut);

        id req1 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(gAR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI1], @[@0], @[wO1], @[@0], nil, nil, @0);
        id req2 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(gAR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI2], @[@0], @[wO2], @[@0], nil, nil, @0);

        NSError *e = nil;

        // =====================================================================
        // PHASE 3: IOSurface Mapper & _ANEBuffer
        // =====================================================================
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  PHASE 3: IOSurface Mapper & Buffer Experiments\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        uint64_t progHandle = 0;
        @try {
            id ph = [k1.model valueForKey:@"programHandle"];
            if (ph) progHandle = [ph unsignedLongLongValue];
        } @catch (NSException *ex) { (void)ex; }
        printf("  k1 programHandle = %llu\n", progHandle);

        // 3a: Try _ANEProgramIOSurfacesMapper
        // API: +mapperWithController:(id)ctrl  |  +mapperWithProgramHandle:(uint64_t)handle
        Class gMapper = NSClassFromString(@"_ANEProgramIOSurfacesMapper");
        if (gMapper) {
            printf("\n  --- 3a: _ANEProgramIOSurfacesMapper ---\n");

            // Try mapperWithProgramHandle: (takes uint64_t)
            id mapper = nil;
            if (progHandle) {
                @try {
                    mapper = ((id(*)(Class,SEL,uint64_t))objc_msgSend)(gMapper,
                        @selector(mapperWithProgramHandle:), progHandle);
                    printf("  mapperWithProgramHandle(%llu): %s\n", progHandle,
                           mapper ? [[mapper description] UTF8String] : "nil");
                    if (mapper) {
                        R.mapper_works = true;
                        dump_all_properties(mapper, gMapper);
                    }
                } @catch (NSException *ex) {
                    printf("  mapperWithProgramHandle EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }

            // Try mapperWithController: using model's sharedConnection or DeviceController
            if (!mapper) {
                @try {
                    id devCtrl = nil;
                    Class gDevCtrl = NSClassFromString(@"_ANEDeviceController");
                    if (gDevCtrl) {
                        devCtrl = ((id(*)(Class,SEL,uint64_t))objc_msgSend)(gDevCtrl,
                            @selector(controllerWithProgramHandle:), progHandle);
                        printf("  _ANEDeviceController.controllerWithProgramHandle: %s\n",
                               devCtrl ? [[devCtrl description] UTF8String] : "nil");
                    }
                    if (devCtrl) {
                        mapper = ((id(*)(Class,SEL,id))objc_msgSend)(gMapper,
                            @selector(mapperWithController:), devCtrl);
                        printf("  mapperWithController: %s\n",
                               mapper ? [[mapper description] UTF8String] : "nil");
                        if (mapper) {
                            R.mapper_works = true;
                            dump_all_properties(mapper, gMapper);
                        }
                    }
                } @catch (NSException *ex) {
                    printf("  mapperWithController EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }

            // Try mapIOSurfacesWithModel:request:cacheInference:error: if we have a mapper
            if (mapper) {
                printf("\n  Trying mapIOSurfacesWithModel...\n");
                id reqMap = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(gAR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI1], @[@0], @[wO1], @[@0], nil, nil, @0);
                @try {
                    NSError *mapErr = nil;
                    BOOL mapOk = ((BOOL(*)(id,SEL,id,id,BOOL,NSError**))objc_msgSend)(
                        mapper, @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
                        k1.model, reqMap, NO, &mapErr);
                    printf("  mapIOSurfacesWithModel: %s\n", mapOk ? "YES" : "NO");
                    if (!mapOk && mapErr) printf("    error: %s\n", [[mapErr description] UTF8String]);
                } @catch (NSException *ex) {
                    printf("  mapIOSurfacesWithModel EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }

                // Also try validateRequest:model:
                @try {
                    BOOL validReq = ((BOOL(*)(id,SEL,id,id))objc_msgSend)(
                        mapper, @selector(validateRequest:model:), reqMap, k1.model);
                    printf("  validateRequest:model: %s\n", validReq ? "YES" : "NO");
                } @catch (NSException *ex) {
                    printf("  validateRequest:model: EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }
        } else {
            printf("\n  _ANEProgramIOSurfacesMapper: NOT FOUND\n");
        }

        // 3b: Try _ANEBuffer
        // API: +bufferWithIOSurfaceObject:(id)ioSurfObj symbolIndex:(id)symIdx source:(long long)src
        Class gBuffer = NSClassFromString(@"_ANEBuffer");
        if (gBuffer) {
            printf("\n  --- 3b: _ANEBuffer ---\n");

            id wBufTest = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO,
                @selector(objectWithIOSurface:), k1.ioIn);

            for (long long src = 0; src <= 2; src++) {
                @try {
                    id buf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(gBuffer,
                        @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wBufTest, @0, src);
                    printf("  bufferWithIOSurfaceObject(symIdx=0, source=%lld): %s\n",
                           src, buf ? [[buf description] UTF8String] : "nil");
                    if (buf) {
                        R.buffer_works = true;
                        dump_all_properties(buf, gBuffer);
                        @try {
                            id symIdx = [buf valueForKey:@"symbolIndex"];
                            printf("    symbolIndex = %s\n", symIdx ? [[symIdx description] UTF8String] : "nil");
                            if (symIdx) R.got_symbol_index = true;
                        } @catch (NSException *ex) {
                            printf("    symbolIndex: <exception>\n");
                        }
                    }
                } @catch (NSException *ex) {
                    printf("  bufferWithIOSurfaceObject(source=%lld) EXCEPTION: %s\n",
                           src, [[ex reason] UTF8String]);
                }
            }

            @try {
                id buf1 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(gBuffer,
                    @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                    wBufTest, @1, (long long)0);
                printf("  bufferWithIOSurfaceObject(symIdx=1, source=0): %s\n",
                       buf1 ? [[buf1 description] UTF8String] : "nil");
                if (buf1) {
                    R.buffer_works = true;
                    dump_all_properties(buf1, gBuffer);
                }
            } @catch (NSException *ex) {
                printf("  bufferWithIOSurfaceObject(symIdx=1) EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("\n  _ANEBuffer: NOT FOUND\n");
        }

        // 3c: Try _ANEIOSurfaceObject with symbolIndex setter
        printf("\n  --- 3c: _ANEIOSurfaceObject symbolIndex experiment ---\n");
        {
            id wTest = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO,
                @selector(objectWithIOSurface:), k1.ioIn);

            if ([wTest respondsToSelector:@selector(setSymbolIndex:)]) {
                printf("  setSymbolIndex: is available!\n");
                @try {
                    ((void(*)(id,SEL,NSUInteger))objc_msgSend)(wTest, @selector(setSymbolIndex:), 0);
                    printf("  setSymbolIndex:0 succeeded\n");
                    R.got_symbol_index = true;
                } @catch (NSException *ex) {
                    printf("  setSymbolIndex:0 EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            } else {
                printf("  setSymbolIndex: NOT available on _ANEIOSurfaceObject\n");
            }

            if ([wTest respondsToSelector:NSSelectorFromString(@"symbolIndex")]) {
                printf("  symbolIndex getter: available\n");
                @try {
                    NSUInteger idx = ((NSUInteger(*)(id,SEL))objc_msgSend)(wTest,
                        NSSelectorFromString(@"symbolIndex"));
                    printf("  symbolIndex = %lu\n", (unsigned long)idx);
                } @catch (NSException *ex) {
                    printf("  symbolIndex getter EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            } else {
                printf("  symbolIndex getter: NOT available\n");
            }

            SEL selObjWithSurfaceIdx = NSSelectorFromString(@"objectWithIOSurface:symbolIndex:");
            if ([gAIO respondsToSelector:selObjWithSurfaceIdx]) {
                printf("  +objectWithIOSurface:symbolIndex: is available!\n");
                @try {
                    id wIndexed = ((id(*)(Class,SEL,IOSurfaceRef,NSUInteger))objc_msgSend)(
                        gAIO, selObjWithSurfaceIdx, k1.ioIn, (NSUInteger)0);
                    printf("  result: %s\n", wIndexed ? [[wIndexed description] UTF8String] : "nil");
                    if (wIndexed) R.got_symbol_index = true;
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            } else {
                printf("  +objectWithIOSurface:symbolIndex: NOT available\n");
            }
        }

        // 3d: Try setting symbolIndex on IOSurface itself (as IOSurface property)
        printf("\n  --- 3d: IOSurface property experiments ---\n");
        {
            IOSurfaceLock(k1.ioIn, 0, NULL);
            IOSurfaceSetValue(k1.ioIn, CFSTR("symbolIndex"), (__bridge CFTypeRef)@0);
            IOSurfaceUnlock(k1.ioIn, 0, NULL);

            CFTypeRef val = IOSurfaceCopyValue(k1.ioIn, CFSTR("symbolIndex"));
            printf("  IOSurface 'symbolIndex' property: %s\n",
                   val ? [(__bridge id)val description].UTF8String : "nil");
            if (val) CFRelease(val);

            id wWithProp = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO,
                @selector(objectWithIOSurface:), k1.ioIn);
            @try {
                id symIdx = [wWithProp valueForKey:@"symbolIndex"];
                printf("  _ANEIOSurfaceObject.symbolIndex after IOSurface property set: %s\n",
                       symIdx ? [[symIdx description] UTF8String] : "nil");
                if (symIdx) R.got_symbol_index = true;
            } @catch (NSException *ex) {
                printf("  _ANEIOSurfaceObject.symbolIndex: <exception: %s>\n", [[ex reason] UTF8String]);
            }
        }

        // 3e: Try _ANEProgramForEvaluation
        // API: +programWithHandle:(uint64_t)handle intermediateBufferHandle:(uint64_t)ibh queueDepth:(char)qd
        //      +programWithController:(id)ctrl intermediateBufferHandle:(uint64_t)ibh queueDepth:(char)qd
        Class gProgEval = NSClassFromString(@"_ANEProgramForEvaluation");
        if (gProgEval) {
            printf("\n  --- 3e: _ANEProgramForEvaluation ---\n");

            // The model already has a .program property -- read it directly
            @try {
                id existingProg = [k1.model valueForKey:@"program"];
                printf("  k1.model.program: %s\n",
                       existingProg ? [[existingProg description] UTF8String] : "nil");
                if (existingProg) {
                    dump_all_properties(existingProg, gProgEval);
                }
            } @catch (NSException *ex) {
                printf("  k1.model.program: <exception: %s>\n", [[ex reason] UTF8String]);
            }

            // Try programWithHandle:intermediateBufferHandle:queueDepth:
            uint64_t ibHandle = 0;
            @try {
                id ibh = [k1.model valueForKey:@"intermediateBufferHandle"];
                if (ibh) ibHandle = [ibh unsignedLongLongValue];
            } @catch (NSException *ex) { (void)ex; }

            @try {
                id prog = ((id(*)(Class,SEL,uint64_t,uint64_t,char))objc_msgSend)(gProgEval,
                    @selector(programWithHandle:intermediateBufferHandle:queueDepth:),
                    progHandle, ibHandle, (char)1);
                printf("  programWithHandle(%llu, %llu, 1): %s\n",
                       progHandle, ibHandle,
                       prog ? [[prog description] UTF8String] : "nil");
                if (prog) dump_all_properties(prog, gProgEval);
            } @catch (NSException *ex) {
                printf("  programWithHandle EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }

        // 3f: Try _ANEIOSurfaceOutputSets and _ANEInputBuffersReady
        const char *chain_helper_classes[] = {
            "_ANEIOSurfaceOutputSets",
            "_ANEInputBuffersReady",
            "_ANEOutputSetEnqueue",
            NULL
        };
        for (int ci = 0; chain_helper_classes[ci]; ci++) {
            Class cls = NSClassFromString([NSString stringWithUTF8String:chain_helper_classes[ci]]);
            if (!cls) continue;
            printf("\n  --- 3f: %s instantiation ---\n", chain_helper_classes[ci]);

            unsigned int mc = 0;
            Method *ms = class_copyMethodList(object_getClass(cls), &mc);
            for (unsigned int i = 0; i < mc; i++) {
                SEL s = method_getName(ms[i]);
                printf("    + %s\n", sel_getName(s));
            }
            free(ms);

            @try {
                id obj = [[cls alloc] init];
                printf("  alloc/init: %s\n", obj ? [[obj description] UTF8String] : "nil");
                if (obj) dump_all_properties(obj, cls);
            } @catch (NSException *ex) {
                printf("  alloc/init EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }

        R.phase3_done = true;
        printf("\n  Phase 3 complete: mapper=%s buffer=%s symbolIndex=%s\n",
               R.mapper_works ? "YES" : "NO",
               R.buffer_works ? "YES" : "NO",
               R.got_symbol_index ? "YES" : "NO");

        // =====================================================================
        // PHASE 4: ChainingRequest with (potentially) indexed surfaces
        // =====================================================================
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  PHASE 4: ChainingRequest Retry\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // 4a: Sequential baseline
        printf("\n  --- 4a: Sequential baseline ---\n");
        int WARMUP = 5, ITERS = 50;
        for (int i = 0; i < WARMUP; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                k1.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req1, &e);
            IOSurfaceLock(k1.ioOut, 0, NULL);
            memcpy(IOSurfaceGetBaseAddress(k2.ioIn), IOSurfaceGetBaseAddress(k1.ioOut), ioBytes);
            IOSurfaceUnlock(k1.ioOut, 0, NULL);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                k2.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req2, &e);
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < ITERS; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                k1.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req1, &e);
            IOSurfaceLock(k1.ioOut, 0, NULL);
            memcpy(IOSurfaceGetBaseAddress(k2.ioIn), IOSurfaceGetBaseAddress(k1.ioOut), ioBytes);
            IOSurfaceUnlock(k1.ioOut, 0, NULL);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                k2.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req2, &e);
        }
        R.sequential_ms = tb_ms(mach_absolute_time() - t0) / ITERS;
        printf("  Sequential: %.3f ms/pair (%d iters)\n", R.sequential_ms, ITERS);

        IOSurfaceLock(k2.ioOut, kIOSurfaceLockReadOnly, NULL);
        if (g_fp16_io) {
            _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k2.ioOut);
            printf("  Output[0..3]: [%.4f, %.4f, %.4f, %.4f]\n",
                   (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
        } else {
            float *out = (float*)IOSurfaceGetBaseAddress(k2.ioOut);
            printf("  Output[0..3]: [%.4f, %.4f, %.4f, %.4f]\n", out[0], out[1], out[2], out[3]);
        }
        IOSurfaceUnlock(k2.ioOut, kIOSurfaceLockReadOnly, NULL);

        // 4b: ChainingRequest attempts
        printf("\n  --- 4b: ChainingRequest attempts ---\n");

        id client = nil;
        if (gClient) {
            client = [gClient performSelector:@selector(sharedConnection)];
            printf("  _ANEClient: %s\n", client ? "obtained" : "FAILED");
        }

        if (gChain && client) {
            // Attempt 1: standard (same as v1)
            printf("\n  [Attempt 1] Standard ChainingRequest (raw IOSurface objects)\n");
            @try {
                id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(gChain,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[wI1], @[@[wO1]], @[@0], @[@0], @0, @[], @0, @0, @0);

                if (chainReq) {
                    BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainReq, @selector(validate));
                    printf("    created: YES | validate: %s\n", valid ? "YES" : "NO");
                    R.validate_passed = valid;

                    if (valid && client) {
                        @try {
                            BOOL prep = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                                k1.model, @{}, chainReq, 21, &e);
                            printf("    prepareChainingWithModel: %s\n", prep ? "YES" : "NO");
                            if (!prep && e) printf("      error: %s\n", [[e description] UTF8String]);
                        } @catch (NSException *ex) {
                            printf("    prepareChainingWithModel EXCEPTION: %s\n", [[ex reason] UTF8String]);
                        }
                    }
                } else {
                    printf("    created: NO\n");
                }
            } @catch (NSException *ex) {
                printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            // Attempt 2: with IOSurface property "symbolIndex"
            printf("\n  [Attempt 2] IOSurface with symbolIndex property\n");
            @try {
                IOSurfaceRef sIn = make_surface(ioBytes);
                IOSurfaceRef sOut = make_surface(ioBytes);

                IOSurfaceLock(sIn, 0, NULL);
                if (g_fp16_io) {
                    _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(sIn);
                    for (int i = 0; i < CH*SP; i++) inp[i] = (_Float16)1.0f;
                } else {
                    float *inp = (float*)IOSurfaceGetBaseAddress(sIn);
                    for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f;
                }
                IOSurfaceSetValue(sIn, CFSTR("symbolIndex"), (__bridge CFTypeRef)@0);
                IOSurfaceUnlock(sIn, 0, NULL);

                IOSurfaceLock(sOut, 0, NULL);
                IOSurfaceSetValue(sOut, CFSTR("symbolIndex"), (__bridge CFTypeRef)@0);
                IOSurfaceUnlock(sOut, 0, NULL);

                id wIn2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), sIn);
                id wOut2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), sOut);

                id chainReq2 = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(gChain,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[wIn2], @[@[wOut2]], @[@0], @[@0], @0, @[], @0, @0, @0);

                if (chainReq2) {
                    BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainReq2, @selector(validate));
                    printf("    created: YES | validate: %s\n", valid ? "YES" : "NO");
                    if (valid) R.validate_passed = true;

                    if (valid) {
                        @try {
                            BOOL prep = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                                k1.model, @{}, chainReq2, 21, &e);
                            printf("    prepareChainingWithModel: %s\n", prep ? "YES" : "NO");
                            if (prep) R.chaining_executed = true;
                        } @catch (NSException *ex) {
                            printf("    prepareChainingWithModel EXCEPTION: %s\n", [[ex reason] UTF8String]);
                        }
                    }
                }

                CFRelease(sIn); CFRelease(sOut);
            } @catch (NSException *ex) {
                printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            // Attempt 3: two-model loopback with output sets
            printf("\n  [Attempt 3] Two-model loopback with multiple output sets\n");
            @try {
                IOSurfaceRef sMid = make_surface(ioBytes);
                id wMid = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), sMid);

                id chainLoop = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(gChain,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[wI1],
                    @[@[wMid], @[wO2]],
                    @[@0],
                    @[@0],
                    @0, @[], @0, @0, @0);

                if (chainLoop) {
                    BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainLoop, @selector(validate));
                    printf("    created: YES | validate: %s\n", valid ? "YES" : "NO");

                    @try {
                        BOOL prep = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                            k1.model, @{}, chainLoop, 21, &e);
                        printf("    prepareChainingWithModel: %s\n", prep ? "YES" : "NO");
                        if (!prep && e) printf("      error: %s\n", [[e description] UTF8String]);

                        if (prep) {
                            R.chaining_executed = true;

                            uint64_t tc0 = mach_absolute_time();
                            for (int i = 0; i < ITERS; i++) {
                                BOOL enq = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                    client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                                    k1.model, @[wMid], @{}, 21, &e);
                                (void)enq;
                                BOOL buf = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                    client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                                    k1.model, @[wI1], @{}, 21, &e);
                                (void)buf;
                            }
                            R.chained_ms = tb_ms(mach_absolute_time() - tc0) / ITERS;
                            printf("    Chained: %.3f ms/pair (%d iters)\n", R.chained_ms, ITERS);
                            printf("    Speedup: %.2fx vs sequential\n", R.sequential_ms / R.chained_ms);
                        }
                    } @catch (NSException *ex) {
                        printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    }
                }
                CFRelease(sMid);
            } @catch (NSException *ex) {
                printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            // Attempt 4: force validate bypass via prepareChainingWithModel directly
            printf("\n  [Attempt 4] Skip validate, call prepareChainingWithModel directly\n");
            @try {
                id chainDirect = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(gChain,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[wI1], @[@[wO1]], @[@0], @[@0], @0, @[], @0, @0, @0);

                if (chainDirect) {
                    BOOL prep = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                        k1.model, @{}, chainDirect, 21, &e);
                    printf("    prepareChainingWithModel (no validate): %s\n", prep ? "YES" : "NO");
                    if (!prep && e) printf("      error: %s\n", [[e description] UTF8String]);

                    if (prep) {
                        R.chaining_executed = true;

                        @try {
                            BOOL enq = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                                k1.model, @[wO1], @{}, 21, &e);
                            printf("    enqueueSets: %s\n", enq ? "YES" : "NO");
                            if (!enq && e) printf("      error: %s\n", [[e description] UTF8String]);
                        } @catch (NSException *ex) {
                            printf("    enqueueSets EXCEPTION: %s\n", [[ex reason] UTF8String]);
                        }

                        @try {
                            BOOL buf = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                                k1.model, @[wI1], @{}, 21, &e);
                            printf("    buffersReady: %s\n", buf ? "YES" : "NO");
                            if (!buf && e) printf("      error: %s\n", [[e description] UTF8String]);
                        } @catch (NSException *ex) {
                            printf("    buffersReady EXCEPTION: %s\n", [[ex reason] UTF8String]);
                        }
                    }
                }
            } @catch (NSException *ex) {
                printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
            // Attempt 5: Use _ANEBuffer inputs + _ANEIOSurfaceOutputSets for outputSets
            printf("\n  [Attempt 5] ChainingRequest with _ANEBuffer + _ANEIOSurfaceOutputSets\n");
            {
                Class gOutSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
                if (gBuffer && gOutSets) {
                    @try {
                        id bufIn = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(gBuffer,
                            @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                            wI1, @0, (long long)0);
                        id bufOut = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(gBuffer,
                            @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                            wO1, @0, (long long)1);
                        printf("    bufIn: %s\n", bufIn ? [[bufIn description] UTF8String] : "nil");
                        printf("    bufOut: %s\n", bufOut ? [[bufOut description] UTF8String] : "nil");

                        // Create _ANEIOSurfaceOutputSets: +objectWithstatsSurRef:outputBuffer:
                        // statsSurRef can be NULL, outputBuffer is NSArray of _ANEBuffer
                        id outSet = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(gOutSets,
                            @selector(objectWithstatsSurRef:outputBuffer:),
                            NULL, @[bufOut]);
                        printf("    outputSet: %s\n", outSet ? [[outSet description] UTF8String] : "nil");

                        if (bufIn && outSet) {
                            id chainBuf = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(gChain,
                                @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                                @[bufIn], @[outSet], @[@0], @[@0], @0, @[], @0, @0, @0);

                            if (chainBuf) {
                                BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainBuf, @selector(validate));
                                printf("    created: YES | validate: %s\n", valid ? "YES" : "NO");
                                if (valid) R.validate_passed = true;

                                @try {
                                    BOOL prep = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                        client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                                        k1.model, @{}, chainBuf, 21, &e);
                                    printf("    prepareChainingWithModel: %s\n", prep ? "YES" : "NO");
                                    if (!prep && e) printf("      error: %s\n", [[e description] UTF8String]);
                                    if (prep) R.chaining_executed = true;
                                } @catch (NSException *ex) {
                                    printf("    prepareChainingWithModel EXCEPTION: %s\n", [[ex reason] UTF8String]);
                                }
                            } else {
                                printf("    ChainingRequest creation: nil\n");
                            }
                        }
                    } @catch (NSException *ex) {
                        printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    }
                } else {
                    printf("    _ANEBuffer or _ANEIOSurfaceOutputSets not available\n");
                }
            }

            // Attempt 6: Use _ANEClient.evaluateWithModel (5-param variant)
            printf("\n  [Attempt 6] _ANEClient.evaluateWithModel:options:request:qos:error:\n");
            @try {
                BOOL clientEval = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, @selector(evaluateWithModel:options:request:qos:error:),
                    k1.model, @{}, req1, 21, &e);
                printf("    evaluateWithModel (via client): %s\n", clientEval ? "YES" : "NO");
                if (!clientEval && e) printf("      error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            // Attempt 7: Use _ANEClient.doEvaluateDirectWithModel
            printf("\n  [Attempt 7] _ANEClient.doEvaluateDirectWithModel\n");
            @try {
                BOOL directEval = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    k1.model, @{}, req1, 21, &e);
                printf("    doEvaluateDirectWithModel: %s\n", directEval ? "YES" : "NO");
                if (!directEval && e) printf("      error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("  Skipped: gChain=%s client=%s\n",
                   gChain ? "YES" : "NO", client ? "YES" : "NO");
        }

        R.phase4_done = true;

        // =====================================================================
        // PHASE 5: Alternative Execution Paths
        // =====================================================================
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  PHASE 5: Alternative Execution Paths\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // 5a: Real-time eval
        printf("\n  --- 5a: Real-time eval path ---\n");
        if (client) {
            @try {
                printf("  Calling beginRealTimeTask...\n");
                BOOL rtBegin = ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(beginRealTimeTask));
                printf("  beginRealTimeTask: %s\n", rtBegin ? "YES" : "NO");

                printf("  Calling evaluateRealTimeWithModel...\n");
                BOOL rtOk = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    k1.model, @{}, req1, &e);
                printf("  evaluateRealTimeWithModel: %s\n", rtOk ? "YES" : "NO");
                if (!rtOk && e) printf("    error: %s\n", [[e description] UTF8String]);
                R.realtime_eval_works = rtOk;

                if (rtOk) {
                    double rt_times[ITERS];
                    for (int i = 0; i < WARMUP; i++) {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, @selector(evaluateRealTimeWithModel:options:request:error:),
                            k1.model, @{}, req1, &e);
                    }
                    for (int i = 0; i < ITERS; i++) {
                        uint64_t ti = mach_absolute_time();
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, @selector(evaluateRealTimeWithModel:options:request:error:),
                            k1.model, @{}, req1, &e);
                        rt_times[i] = tb_ms(mach_absolute_time() - ti);
                    }
                    double rt_sum = 0;
                    for (int i = 0; i < ITERS; i++) rt_sum += rt_times[i];
                    printf("  RT eval: %.3f ms/eval avg (%d iters)\n", rt_sum/ITERS, ITERS);

                    double std_times[ITERS];
                    for (int i = 0; i < ITERS; i++) {
                        uint64_t ti = mach_absolute_time();
                        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            k1.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req1, &e);
                        std_times[i] = tb_ms(mach_absolute_time() - ti);
                    }
                    double std_sum = 0;
                    for (int i = 0; i < ITERS; i++) std_sum += std_times[i];
                    printf("  Standard eval: %.3f ms/eval avg (%d iters)\n", std_sum/ITERS, ITERS);
                    printf("  RT vs Standard speedup: %.2fx\n", (std_sum/ITERS) / (rt_sum/ITERS));
                }

                BOOL rtEnd = ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
                printf("  endRealTimeTask: %s\n", rtEnd ? "YES" : "NO");
            } @catch (NSException *ex) {
                printf("  Real-time eval EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }

        // 5b: PerfStats with perfStatsMask
        printf("\n  --- 5b: PerfStats with perfStatsMask ---\n");
        {
            Class perfClass = NSClassFromString(@"_ANEPerformanceStats");
            if (perfClass) {
                @try {
                    printf("  Setting perfStatsMask on model...\n");
                    for (unsigned int mask = 1; mask <= 0xFF; mask <<= 1) {
                        @try {
                            [k1.model setValue:@(mask) forKey:@"perfStatsMask"];
                            printf("    perfStatsMask = 0x%02X: set OK\n", mask);
                        } @catch (NSException *ex) {
                            printf("    perfStatsMask = 0x%02X: <exception: %s>\n", mask, [[ex reason] UTF8String]);
                            break;
                        }
                    }
                } @catch (NSException *ex) {
                    printf("  perfStatsMask setter: <exception: %s>\n", [[ex reason] UTF8String]);
                }

                id perfStats = nil;
                @try {
                    perfStats = ((id(*)(Class,SEL,uint64_t))objc_msgSend)(perfClass,
                        @selector(statsWithHardwareExecutionNS:), (uint64_t)0);
                    printf("  statsWithHardwareExecutionNS:0 = %s\n",
                           perfStats ? [[perfStats description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  statsWithHardwareExecutionNS: EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }

                if (!perfStats) {
                    @try {
                        perfStats = [[perfClass alloc] init];
                        printf("  alloc/init fallback = %s\n",
                               perfStats ? [[perfStats description] UTF8String] : "nil");
                    } @catch (NSException *ex) {
                        printf("  alloc/init EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    }
                }

                if (perfStats) {
                    // perfStats param expects NSArray (request calls [perfStats count])
                    // Try wrapping in array, and also try with nil + perfStatsMask
                    printf("  Test A: perfStats as NSArray wrapper\n");
                    @try {
                        id reqPerfA = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(gAR,
                            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                            @[wI1], @[@0], @[wO1], @[@0], nil, @[perfStats], @0);
                        if (reqPerfA) {
                            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                                k1.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqPerfA, &e);
                            printf("    Eval with @[perfStats]: %s\n", ok ? "OK" : "FAIL");
                            if (ok) {
                                printf("    PerfStats after eval:\n");
                                dump_all_properties(perfStats, perfClass);
                                @try {
                                    id hwTime = [perfStats valueForKey:@"hwExecutionTime"];
                                    printf("    hwExecutionTime = %s\n",
                                           hwTime ? [[hwTime description] UTF8String] : "nil");
                                    if (hwTime) {
                                        R.hw_exec_time_ns = [hwTime unsignedLongLongValue];
                                        R.perfstats_works = (R.hw_exec_time_ns > 0);
                                    }
                                } @catch (NSException *ex) {
                                    printf("    hwExecutionTime: <exception>\n");
                                }
                            }
                        }
                    } @catch (NSException *ex) {
                        printf("    Test A EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    }

                    // Test B: perfStatsMask set, but perfStats=nil in request
                    printf("  Test B: perfStatsMask=0xFF, perfStats=nil\n");
                    @try {
                        [k1.model setValue:@(0xFF) forKey:@"perfStatsMask"];
                        id reqPerfB = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(gAR,
                            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                            @[wI1], @[@0], @[wO1], @[@0], nil, nil, @0);
                        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            k1.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqPerfB, &e);
                        printf("    Eval with mask=0xFF, perfStats=nil: %s\n", ok ? "OK" : "FAIL");
                        uint64_t psMask = [[k1.model valueForKey:@"perfStatsMask"] unsignedIntValue];
                        printf("    perfStatsMask after eval: 0x%llX\n", (unsigned long long)psMask);
                    } @catch (NSException *ex) {
                        printf("    Test B EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    }

                    @try {
                        id counters = [perfStats performSelector:@selector(performanceCounters)];
                        printf("  performanceCounters: %s\n",
                               counters ? [[counters description] UTF8String] : "nil");
                    } @catch (NSException *ex) {
                        printf("  performanceCounters: <exception: %s>\n", [[ex reason] UTF8String]);
                    }
                }
            } else {
                printf("  _ANEPerformanceStats: NOT FOUND\n");
            }
        }

        R.phase5_done = true;

        // =====================================================================
        // PHASE 7 (Exp A): _ANEIOSurfaceOutputSets with non-NULL statsSurRef
        // =====================================================================
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  PHASE 7: OutputSets with stats IOSurface\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        {
            Class gOutSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
            Class gBuf = NSClassFromString(@"_ANEBuffer");
            if (gOutSets && gBuf) {
                id bufOut7 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(gBuf,
                    @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                    wO1, @0, (long long)1);

                size_t stats_sizes[] = {64, 256, 1024, 4096, 16384};
                for (int si = 0; si < 5; si++) {
                    IOSurfaceRef statsSurf = make_surface(stats_sizes[si]);
                    printf("\n  statsSurRef size=%zu bytes:\n", stats_sizes[si]);

                    @try {
                        id outSet = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(gOutSets,
                            @selector(objectWithstatsSurRef:outputBuffer:),
                            statsSurf, @[bufOut7]);
                        printf("    objectWithstatsSurRef: %s\n",
                               outSet ? [[outSet description] UTF8String] : "nil");

                        if (outSet) {
                            R.outputsets_with_stats_works = true;
                            dump_all_properties(outSet, gOutSets);

                            printf("\n    Attempting ChainingRequest with valid outputSet...\n");
                            id bufIn7 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(gBuf,
                                @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                                wI1, @0, (long long)0);

                            @try {
                                id chain7 = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(gChain,
                                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                                    @[bufIn7], @[outSet], @[@0], @[@0], @0, @[], @0, @0, @0);

                                if (chain7) {
                                    BOOL valid7 = ((BOOL(*)(id,SEL))objc_msgSend)(chain7, @selector(validate));
                                    printf("    ChainingRequest created | validate: %s\n", valid7 ? "YES" : "NO");
                                    if (valid7) R.chaining_with_stats_works = true;

                                    @try {
                                        BOOL prep7 = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                            client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                                            k1.model, @{}, chain7, 21, &e);
                                        printf("    prepareChainingWithModel: %s\n", prep7 ? "YES" : "NO");
                                        if (!prep7 && e) printf("      error: %s\n", [[e description] UTF8String]);
                                        if (prep7) R.chaining_with_stats_works = true;
                                    } @catch (NSException *ex) {
                                        printf("    prepareChainingWithModel EXCEPTION: %s\n", [[ex reason] UTF8String]);
                                    }
                                } else {
                                    printf("    ChainingRequest creation: nil\n");
                                }
                            } @catch (NSException *ex) {
                                printf("    ChainingRequest EXCEPTION: %s\n", [[ex reason] UTF8String]);
                            }
                        }
                    } @catch (NSException *ex) {
                        printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    }

                    CFRelease(statsSurf);
                    if (R.outputsets_with_stats_works) break;
                }
            } else {
                printf("  Required classes not found\n");
            }
        }
        R.phase7_done = true;

        // =====================================================================
        // PHASE 8 (Exp B): Disk-based _ANEModel for symbol discovery
        // =====================================================================
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  PHASE 8: Disk-based _ANEModel path\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        {
            Class gANEModel = NSClassFromString(@"_ANEModel");
            if (gANEModel) {
                printf("\n  _ANEModel class found. Dumping API surface...\n");
                dump_class("_ANEModel");

                NSString *compiledPath = nil;
                @try {
                    compiledPath = [k1.model valueForKey:@"compiledModelPath"];
                    printf("  k1.compiledModelPath: %s\n",
                           compiledPath ? [compiledPath UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  compiledModelPath: <exception>\n");
                }
                if (!compiledPath) {
                    @try {
                        id url = [k1.model valueForKey:@"compiledModelURL"];
                        if (url) compiledPath = [url path];
                        printf("  k1.compiledModelURL.path: %s\n",
                               compiledPath ? [compiledPath UTF8String] : "nil");
                    } @catch (NSException *ex) {
                        printf("  compiledModelURL: <exception>\n");
                    }
                }

                NSString *modelDir = k1.tmpDir;
                printf("  k1.tmpDir: %s\n", [modelDir UTF8String]);

                NSFileManager *fm8 = [NSFileManager defaultManager];
                NSArray *contents = [fm8 contentsOfDirectoryAtPath:modelDir error:nil];
                printf("  tmpDir contents: %s\n", contents ? [[contents description] UTF8String] : "empty");

                SEL factorySelectors[] = {
                    @selector(modelAtURL:),
                    @selector(modelWithPath:),
                    NSSelectorFromString(@"modelAtPath:"),
                    NSSelectorFromString(@"modelWithURL:"),
                };
                const char *factoryNames[] = {"modelAtURL:", "modelWithPath:", "modelAtPath:", "modelWithURL:"};

                for (int fi = 0; fi < 4; fi++) {
                    if ([gANEModel respondsToSelector:factorySelectors[fi]]) {
                        printf("  +%s: available\n", factoryNames[fi]);
                    } else {
                        printf("  +%s: NOT available\n", factoryNames[fi]);
                    }
                }

                id diskModel = nil;
                @try {
                    if ([gANEModel respondsToSelector:@selector(modelAtURL:)]) {
                        NSURL *dirURL = [NSURL fileURLWithPath:modelDir];
                        diskModel = ((id(*)(Class,SEL,id))objc_msgSend)(gANEModel,
                            @selector(modelAtURL:), dirURL);
                        printf("  modelAtURL: %s\n", diskModel ? [[diskModel description] UTF8String] : "nil");
                    }
                } @catch (NSException *ex) {
                    printf("  modelAtURL EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }

                if (!diskModel) {
                    @try {
                        SEL s = NSSelectorFromString(@"modelAtPath:");
                        if ([gANEModel respondsToSelector:s]) {
                            diskModel = ((id(*)(Class,SEL,id))objc_msgSend)(gANEModel, s, modelDir);
                            printf("  modelAtPath: %s\n", diskModel ? [[diskModel description] UTF8String] : "nil");
                        }
                    } @catch (NSException *ex) {
                        printf("  modelAtPath EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    }
                }

                if (diskModel) {
                    R.disk_model_loads = true;
                    printf("\n  _ANEModel loaded! Querying symbol names...\n");
                    dump_all_properties(diskModel, gANEModel);

                    const char *symbol_keys[] = {
                        "inputSymbolNames", "outputSymbolNames",
                        "inputSymbolIndicesForProcedureIndex:",
                        "outputSymbolIndicesForProcedureIndex:",
                        NULL
                    };
                    for (int ki = 0; symbol_keys[ki]; ki++) {
                        @try {
                            if (strchr(symbol_keys[ki], ':')) {
                                SEL s = NSSelectorFromString(
                                    [NSString stringWithUTF8String:symbol_keys[ki]]);
                                if ([diskModel respondsToSelector:s]) {
                                    id result = ((id(*)(id,SEL,NSUInteger))objc_msgSend)(
                                        diskModel, s, (NSUInteger)0);
                                    printf("    %s(0) = %s\n", symbol_keys[ki],
                                           result ? [[result description] UTF8String] : "nil");
                                    R.disk_model_has_symbols = (result != nil);
                                } else {
                                    printf("    %s: NOT available\n", symbol_keys[ki]);
                                }
                            } else {
                                id val = [diskModel valueForKey:
                                    [NSString stringWithUTF8String:symbol_keys[ki]]];
                                printf("    %s = %s\n", symbol_keys[ki],
                                       val ? [[val description] UTF8String] : "nil");
                                if (val) R.disk_model_has_symbols = true;
                            }
                        } @catch (NSException *ex) {
                            printf("    %s: <exception: %s>\n", symbol_keys[ki], [[ex reason] UTF8String]);
                        }
                    }
                } else {
                    printf("  _ANEModel could not be loaded from tmpDir\n");
                }
            } else {
                printf("  _ANEModel: NOT FOUND\n");
            }
        }
        R.phase8_done = true;

        // =====================================================================
        // PHASE 9 (Exp C): processRequest via _ANEProgramForEvaluation
        // =====================================================================
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  PHASE 9: processRequest via ProgramForEvaluation\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        {
            @try {
                id prog = [k1.model valueForKey:@"program"];
                if (prog) {
                    printf("  k1.model.program: %s\n", [[prog description] UTF8String]);

                    id hexId = [k1.model valueForKey:@"hexStringIdentifier"];
                    printf("  hexStringIdentifier: %s\n", hexId ? [[hexId description] UTF8String] : "nil");

                    SEL procSel = @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:);
                    if ([prog respondsToSelector:procSel]) {
                        printf("  processRequest selector: available\n");

                        for (int warmup = 0; warmup < WARMUP; warmup++) {
                            @try {
                                BOOL rv = NO;
                                ((BOOL(*)(id,SEL,id,id,unsigned int,int,id,id,BOOL*,NSError**))objc_msgSend)(
                                    prog, procSel, req1, k1.model, 21, 0, hexId, @{}, &rv, &e);
                            } @catch (NSException *ex) { (void)ex; }
                        }

                        BOOL firstOk = NO;
                        @try {
                            BOOL rv = NO;
                            firstOk = ((BOOL(*)(id,SEL,id,id,unsigned int,int,id,id,BOOL*,NSError**))objc_msgSend)(
                                prog, procSel, req1, k1.model, 21, 0, hexId, @{}, &rv, &e);
                            printf("  processRequest single call: %s (rv=%s)\n",
                                   firstOk ? "YES" : "NO", rv ? "YES" : "NO");
                            if (!firstOk && e) printf("    error: %s\n", [[e description] UTF8String]);
                        } @catch (NSException *ex) {
                            printf("  processRequest EXCEPTION: %s\n", [[ex reason] UTF8String]);
                        }

                        if (firstOk) {
                            R.process_request_works = true;

                            uint64_t t9 = mach_absolute_time();
                            for (int i = 0; i < ITERS; i++) {
                                BOOL rv = NO;
                                ((BOOL(*)(id,SEL,id,id,unsigned int,int,id,id,BOOL*,NSError**))objc_msgSend)(
                                    prog, procSel, req1, k1.model, 21, 0, hexId, @{}, &rv, &e);
                            }
                            R.process_request_ms = tb_ms(mach_absolute_time() - t9) / ITERS;
                            printf("  processRequest: %.3f ms/eval (%d iters)\n",
                                   R.process_request_ms, ITERS);
                            printf("  vs RT eval: %.2fx\n",
                                   R.process_request_ms / (R.rt_eval_ms > 0 ? R.rt_eval_ms : 0.090));
                        }
                    } else {
                        printf("  processRequest selector: NOT available\n");
                    }
                } else {
                    printf("  k1.model.program: nil\n");
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }
        R.phase9_done = true;

        // =====================================================================
        // PHASE 10 (Exp D): Shared Events for hardware synchronization
        // =====================================================================
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  PHASE 10: Shared Events (hardware sync)\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        {
            const char *event_classes[] = {
                "_ANESharedEvents",
                "_ANESharedSignalEvent",
                "_ANESharedWaitEvent",
                NULL
            };

            for (int ei = 0; event_classes[ei]; ei++) {
                Class cls = NSClassFromString([NSString stringWithUTF8String:event_classes[ei]]);
                if (cls) {
                    R.shared_events_exist = true;
                    dump_class(event_classes[ei]);

                    @try {
                        id obj = [[cls alloc] init];
                        printf("  %s alloc/init: %s\n", event_classes[ei],
                               obj ? [[obj description] UTF8String] : "nil");
                        if (obj) dump_all_properties(obj, cls);
                    } @catch (NSException *ex) {
                        printf("  %s alloc/init EXCEPTION: %s\n", event_classes[ei],
                               [[ex reason] UTF8String]);
                    }
                } else {
                    printf("  %s: NOT FOUND\n", event_classes[ei]);
                }
            }

            if (R.shared_events_exist && gChain && client) {
                printf("\n  Attempting ChainingRequest with shared events...\n");
                Class sigCls = NSClassFromString(@"_ANESharedSignalEvent");
                Class waitCls = NSClassFromString(@"_ANESharedWaitEvent");

                if (sigCls && waitCls) {
                    @try {
                        id sigEvent = [[sigCls alloc] init];
                        id waitEvent = [[waitCls alloc] init];
                        printf("  signalEvent: %s\n", sigEvent ? [[sigEvent description] UTF8String] : "nil");
                        printf("  waitEvent: %s\n", waitEvent ? [[waitEvent description] UTF8String] : "nil");

                        if (sigEvent) {
                            Class gBuf = NSClassFromString(@"_ANEBuffer");
                            id bufIn10 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(gBuf,
                                @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                                wI1, @0, (long long)0);

                            id chain10 = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(gChain,
                                @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                                @[bufIn10], @[], @[@0], @[@0], @0, @[sigEvent], @0, @0, @0);

                            if (chain10) {
                                printf("  ChainingRequest with signalEvent: created\n");
                                BOOL valid10 = ((BOOL(*)(id,SEL))objc_msgSend)(chain10, @selector(validate));
                                printf("  validate: %s\n", valid10 ? "YES" : "NO");
                            } else {
                                printf("  ChainingRequest with signalEvent: nil\n");
                            }
                        }
                    } @catch (NSException *ex) {
                        printf("  Shared events EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    }
                }
            }
        }
        R.phase10_done = true;

        // Benchmark all eval paths side-by-side for final comparison
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  EVAL PATH COMPARISON (side-by-side)\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        {
            int BENCH_ITERS = 200;

            for (int w = 0; w < 10; w++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    k1.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req1, &e);
            }

            uint64_t ts = mach_absolute_time();
            for (int i = 0; i < BENCH_ITERS; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    k1.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req1, &e);
            }
            R.std_eval_ms = tb_ms(mach_absolute_time() - ts) / BENCH_ITERS;

            if (client) {
                for (int w = 0; w < 10; w++) {
                    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        client, @selector(evaluateRealTimeWithModel:options:request:error:),
                        k1.model, @{}, req1, &e);
                }
                ts = mach_absolute_time();
                for (int i = 0; i < BENCH_ITERS; i++) {
                    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        client, @selector(evaluateRealTimeWithModel:options:request:error:),
                        k1.model, @{}, req1, &e);
                }
                R.rt_eval_ms = tb_ms(mach_absolute_time() - ts) / BENCH_ITERS;

                for (int w = 0; w < 10; w++) {
                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                        k1.model, @{}, req1, 21, &e);
                }
                ts = mach_absolute_time();
                for (int i = 0; i < BENCH_ITERS; i++) {
                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                        k1.model, @{}, req1, 21, &e);
                }
                R.direct_eval_ms = tb_ms(mach_absolute_time() - ts) / BENCH_ITERS;
            }

            printf("  evaluateWithQoS (standard): %.3f ms/eval\n", R.std_eval_ms);
            printf("  evaluateRealTimeWithModel:  %.3f ms/eval (%.2fx)\n",
                   R.rt_eval_ms, R.std_eval_ms / R.rt_eval_ms);
            printf("  doEvaluateDirectWithModel:  %.3f ms/eval (%.2fx)\n",
                   R.direct_eval_ms, R.std_eval_ms / R.direct_eval_ms);
            if (R.process_request_works) {
                printf("  processRequest:             %.3f ms/eval (%.2fx)\n",
                       R.process_request_ms, R.std_eval_ms / R.process_request_ms);
            }
        }

        // =====================================================================
        // PHASE 6: Summary
        // =====================================================================
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        printf("║  PHASE 6: Results Summary                              ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        printf("┌──────────────────────────────────────────────────────────┐\n");
        printf("│ Phase 1: Class Introspection                           │\n");
        printf("│   Classes found:   %d                                   │\n", R.classes_found);
        printf("│   Classes missing: %d                                   │\n", R.classes_missing);
        printf("├──────────────────────────────────────────────────────────┤\n");
        printf("│ Phase 2: Symbol Discovery                              │\n");
        printf("│   inputSymbolNames:  %s                              │\n", R.has_input_symbols ? "YES" : "NO ");
        printf("│   outputSymbolNames: %s                              │\n", R.has_output_symbols ? "YES" : "NO ");
        printf("│   programHandle:     %s                              │\n", R.has_program_handle ? "YES" : "NO ");
        printf("├──────────────────────────────────────────────────────────┤\n");
        printf("│ Phase 3: IOSurface Mapping                             │\n");
        printf("│   Mapper works:      %s                              │\n", R.mapper_works ? "YES" : "NO ");
        printf("│   Buffer works:      %s                              │\n", R.buffer_works ? "YES" : "NO ");
        printf("│   Got symbolIndex:   %s                              │\n", R.got_symbol_index ? "YES" : "NO ");
        printf("├──────────────────────────────────────────────────────────┤\n");
        printf("│ Phase 4: ChainingRequest                               │\n");
        printf("│   validate passed:   %s                              │\n", R.validate_passed ? "YES" : "NO ");
        printf("│   Chaining executed: %s                              │\n", R.chaining_executed ? "YES" : "NO ");
        printf("│   Sequential:        %.3f ms/pair                    │\n", R.sequential_ms);
        if (R.chaining_executed) {
        printf("│   Chained:           %.3f ms/pair                    │\n", R.chained_ms);
        printf("│   Speedup:           %.2fx                            │\n", R.sequential_ms / R.chained_ms);
        }
        printf("├──────────────────────────────────────────────────────────┤\n");
        printf("│ Phase 5: Alternative Paths                             │\n");
        printf("│   RT eval works:     %s                              │\n", R.realtime_eval_works ? "YES" : "NO ");
        printf("│   PerfStats works:   %s                              │\n", R.perfstats_works ? "YES" : "NO ");
        if (R.perfstats_works) {
        printf("│   hwExecutionTime:   %llu ns                        │\n", R.hw_exec_time_ns);
        }
        printf("├──────────────────────────────────────────────────────────┤\n");
        printf("│ Phase 7: OutputSets with statsSurRef                   │\n");
        printf("│   OutputSets works:  %s                              │\n", R.outputsets_with_stats_works ? "YES" : "NO ");
        printf("│   Chaining works:    %s                              │\n", R.chaining_with_stats_works ? "YES" : "NO ");
        printf("├──────────────────────────────────────────────────────────┤\n");
        printf("│ Phase 8: Disk-based _ANEModel                          │\n");
        printf("│   Model loads:       %s                              │\n", R.disk_model_loads ? "YES" : "NO ");
        printf("│   Has symbols:       %s                              │\n", R.disk_model_has_symbols ? "YES" : "NO ");
        printf("├──────────────────────────────────────────────────────────┤\n");
        printf("│ Phase 9: processRequest                                │\n");
        printf("│   Works:             %s                              │\n", R.process_request_works ? "YES" : "NO ");
        if (R.process_request_works) {
        printf("│   Latency:           %.3f ms/eval                    │\n", R.process_request_ms);
        }
        printf("├──────────────────────────────────────────────────────────┤\n");
        printf("│ Phase 10: Shared Events                                │\n");
        printf("│   Classes exist:     %s                              │\n", R.shared_events_exist ? "YES" : "NO ");
        printf("├──────────────────────────────────────────────────────────┤\n");
        printf("│ Eval Path Comparison (200 iters)                       │\n");
        printf("│   Standard:  %.3f ms/eval                            │\n", R.std_eval_ms);
        printf("│   RT:        %.3f ms/eval (%.2fx)                    │\n", R.rt_eval_ms, R.std_eval_ms / (R.rt_eval_ms > 0 ? R.rt_eval_ms : 1));
        printf("│   Direct:    %.3f ms/eval (%.2fx)                    │\n", R.direct_eval_ms, R.std_eval_ms / (R.direct_eval_ms > 0 ? R.direct_eval_ms : 1));
        if (R.process_request_works) {
        printf("│   ProcReq:   %.3f ms/eval (%.2fx)                    │\n", R.process_request_ms, R.std_eval_ms / (R.process_request_ms > 0 ? R.process_request_ms : 1));
        }
        printf("└──────────────────────────────────────────────────────────┘\n");

        // Cleanup
        NSFileManager *fm = [NSFileManager defaultManager];
        NSError *cleanupErr = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k1.model, @selector(unloadWithQoS:error:), 21, &cleanupErr);
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k2.model, @selector(unloadWithQoS:error:), 21, &cleanupErr);
        [fm removeItemAtPath:k1.tmpDir error:nil];
        [fm removeItemAtPath:k2.tmpDir error:nil];
        if (k1.ioIn) CFRelease(k1.ioIn);
        if (k1.ioOut) CFRelease(k1.ioOut);
        if (k2.ioIn) CFRelease(k2.ioIn);
        if (k2.ioOut) CFRelease(k2.ioOut);

        printf("\n=== ChainingRequest Deep Exploration v2 complete ===\n");
    }
    return 0;
}
