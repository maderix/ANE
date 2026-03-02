// test_perf_stats.m — What does _ANEPerformanceStats expose?
// Probe class methods, properties, instantiate, pass to request, read back.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static void dump_class(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("  %s: NOT FOUND\n", name); return; }
    printf("\n=== %s ===\n", name);

    unsigned int count;
    Method *methods = class_copyMethodList(object_getClass(cls), &count);
    if (count) printf("  Class methods:\n");
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    + %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);

    methods = class_copyMethodList(cls, &count);
    if (count) printf("  Instance methods:\n");
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    - %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);

    unsigned int pcount;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    if (pcount) printf("  Properties:\n");
    for (unsigned int i = 0; i < pcount; i++) {
        const char *pname = property_getName(props[i]);
        const char *pattr = property_getAttributes(props[i]);
        printf("    @property %s  [%s]\n", pname, pattr ? pattr : "?");
    }
    free(props);
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("=== ANE Performance Stats Probe ===\n");

        dump_class("_ANEPerformanceStats");
        dump_class("_ANEPerfRequest");
        dump_class("ANEPerfRequest");
        dump_class("_ANEPerformanceCounters");
        dump_class("_ANEDeviceInfo");
        dump_class("_ANEModel");
        dump_class("_ANEInMemoryModel");
        dump_class("_ANERequest");
        dump_class("_ANEIOSurfaceObject");
        dump_class("_ANEInMemoryModelDescriptor");
        dump_class("_ANEClient");
        dump_class("_ANEVirtualClient");

        // Try to instantiate _ANEPerformanceStats
        printf("\n=== Instantiation Tests ===\n");
        Class perfClass = NSClassFromString(@"_ANEPerformanceStats");
        if (perfClass) {
            @try {
                id perfStats = [[perfClass alloc] init];
                printf("_ANEPerformanceStats alloc/init: %s\n",
                       perfStats ? [[perfStats description] UTF8String] : "nil");
                if (perfStats) {
                    unsigned int pcount;
                    objc_property_t *props = class_copyPropertyList(perfClass, &pcount);
                    for (unsigned int i = 0; i < pcount; i++) {
                        const char *pname = property_getName(props[i]);
                        @try {
                            id val = [perfStats valueForKey:[NSString stringWithUTF8String:pname]];
                            printf("  %s = %s\n", pname, val ? [[val description] UTF8String] : "nil");
                        } @catch (NSException *ex) {
                            printf("  %s = <exception: %s>\n", pname, [[ex reason] UTF8String]);
                        }
                    }
                    free(props);
                }
            } @catch (NSException *ex) {
                printf("Exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // Compile a working kernel and test perfStats in request
        printf("\n=== Compile kernel and test perfStats in request ===\n");
        Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        int CH = 64, SP = 32;
        _Float16 *w = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) w[i*CH+i] = (_Float16)1.0f;
        int ws = CH*CH*2, tot = 128+ws;
        uint8_t *blob = (uint8_t*)calloc(tot,1);
        blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
        *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
        memcpy(blob+128, w, ws);
        NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];
        free(w);

        NSString *mil = [NSString stringWithFormat:
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
            "}\n", CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP];

        NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:),
            md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);

        int ioBytes = CH * SP * 4; // fp32
        IOSurfaceRef ioIn = make_surface(ioBytes);
        IOSurfaceRef ioOut = make_surface(ioBytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);

        // Try creating request WITH perfStats
        if (perfClass) {
            id perfStats = [[perfClass alloc] init];
            printf("  Creating request with perfStats=%s\n", perfStats ? "non-nil" : "nil");

            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, perfStats, @0);
            printf("  Request: %s\n", req ? "created" : "nil");

            if (req) {
                IOSurfaceLock(ioIn, 0, NULL);
                float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
                for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f;
                IOSurfaceUnlock(ioIn, 0, NULL);

                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                printf("  Eval: %s\n", ok ? "OK" : [[e description] UTF8String]);

                if (ok && perfStats) {
                    printf("\n  PerfStats after 1 eval:\n");
                    unsigned int pcount;
                    objc_property_t *props = class_copyPropertyList(perfClass, &pcount);
                    for (unsigned int i = 0; i < pcount; i++) {
                        const char *pname = property_getName(props[i]);
                        @try {
                            id val = [perfStats valueForKey:[NSString stringWithUTF8String:pname]];
                            printf("    %s = %s\n", pname, val ? [[val description] UTF8String] : "nil");
                        } @catch (NSException *ex) {
                            printf("    %s = <exception>\n", pname);
                        }
                    }
                    free(props);

                    printf("\n  Running 100 evals...\n");
                    uint64_t t0 = mach_absolute_time();
                    for (int i = 0; i < 100; i++) {
                        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                    }
                    printf("  100 evals in %.1fms (%.2fms/eval)\n",
                           tb_ms(mach_absolute_time()-t0), tb_ms(mach_absolute_time()-t0)/100.0);

                    printf("\n  PerfStats after 101 evals:\n");
                    props = class_copyPropertyList(perfClass, &pcount);
                    for (unsigned int i = 0; i < pcount; i++) {
                        const char *pname = property_getName(props[i]);
                        @try {
                            id val = [perfStats valueForKey:[NSString stringWithUTF8String:pname]];
                            printf("    %s = %s\n", pname, val ? [[val description] UTF8String] : "nil");
                        } @catch (NSException *ex) {
                            printf("    %s = <exception>\n", pname);
                        }
                    }
                    free(props);
                }
            }
        } else {
            printf("  _ANEPerformanceStats class NOT FOUND\n");
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        CFRelease(ioIn); CFRelease(ioOut);
    }
    return 0;
}
