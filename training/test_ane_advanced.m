// test_ane_advanced.m — Probe advanced ANE interfaces
// SharedEvents, weightsBuffer, procedureIndex, VirtualClient, ChainingRequest
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

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

        printf("=== ANE Advanced Interface Probe ===\n");

        // === Part 1: Event/Sync classes ===
        printf("\n--- Part 1: Event/Sync Classes ---\n");
        dump_class("_ANESharedEvents");
        dump_class("_ANESharedSignalEvent");
        dump_class("_ANESharedWaitEvent");
        dump_class("_ANEEvent");
        dump_class("_ANEFenceEvent");

        const char *event_classes[] = {
            "_ANESharedEvents", "_ANESharedSignalEvent", "_ANESharedWaitEvent",
            "_ANEEvent", "_ANEFenceEvent", NULL
        };
        for (int i = 0; event_classes[i]; i++) {
            Class cls = NSClassFromString([NSString stringWithUTF8String:event_classes[i]]);
            if (!cls) continue;
            @try {
                id obj = [[cls alloc] init];
                printf("  %s alloc/init: %s\n", event_classes[i],
                       obj ? [[obj description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("  %s alloc/init: EXCEPTION: %s\n", event_classes[i], [[ex reason] UTF8String]);
            }
        }

        // === Part 2: VirtualClient and ChainingRequest ===
        printf("\n--- Part 2: VirtualClient / ChainingRequest ---\n");
        dump_class("_ANEVirtualClient");
        dump_class("_ANEChainingRequest");
        dump_class("_ANEMultiRequest");
        dump_class("_ANEBatchRequest");

        // === Part 3: Compile working kernel for weightsBuffer + procedureIndex tests ===
        printf("\n--- Part 3: weightsBuffer IOSurface test ---\n");
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
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);

        int ioBytes = CH * SP * 4;
        IOSurfaceRef ioIn = make_surface(ioBytes);
        IOSurfaceRef ioOut = make_surface(ioBytes);

        IOSurfaceLock(ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int c = 0; c < CH; c++) for (int s = 0; s < SP; s++) inp[c*SP+s] = (float)(s+1) * 0.1f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Baseline eval
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        id req0 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req0, &e);
        printf("  Baseline eval (weightsBuffer=nil, procIdx=0): %s\n", ok ? "OK" : "FAIL");

        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        float *out0 = (float*)IOSurfaceGetBaseAddress(ioOut);
        float baseline_0 = out0[0], baseline_1 = out0[1];
        printf("  Output[0..3]: [%.4f, %.4f, %.4f, %.4f]\n", out0[0], out0[1], out0[2], out0[3]);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        // Test weightsBuffer: IOSurface with 3x identity weights
        printf("\n  Testing weightsBuffer IOSurface...\n");
        _Float16 *w3 = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) w3[i*CH+i] = (_Float16)3.0f;
        IOSurfaceRef ioW = make_surface(ws);
        IOSurfaceLock(ioW, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(ioW), w3, ws);
        IOSurfaceUnlock(ioW, 0, NULL);
        free(w3);
        id wW = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioW);

        wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        id req_wb = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], wW, nil, @0);
        printf("  Request with weightsBuffer: %s\n", req_wb ? "created" : "nil");

        if (req_wb) {
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req_wb, &e);
            printf("  Eval with weightsBuffer: %s\n", ok ? "OK" : e ? [[e description] UTF8String] : "FAIL");
            if (ok) {
                IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
                float *outW = (float*)IOSurfaceGetBaseAddress(ioOut);
                printf("  Output[0..3]: [%.4f, %.4f, %.4f, %.4f]\n", outW[0], outW[1], outW[2], outW[3]);
                bool changed = fabsf(outW[0] - baseline_0) > 0.001f;
                bool is_3x = fabsf(outW[0] - baseline_0 * 3.0f) < 0.1f;
                printf("  weightsBuffer: output %s", changed ? "CHANGED" : "unchanged");
                if (changed) printf(" (%s)", is_3x ? "matches 3x — WORKS!" : "but not 3x as expected");
                printf("\n");
                IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
            }
        }
        CFRelease(ioW);

        // === Part 4: procedureIndex sweep ===
        printf("\n--- Part 4: procedureIndex sweep (0-15) ---\n");
        for (int pi = 0; pi < 16; pi++) {
            wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
            wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
            id req_p = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @(pi));
            if (!req_p) { printf("  procIdx %2d: request=nil\n", pi); continue; }
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req_p, &e);
            printf("  procIdx %2d: %s%s\n", pi, ok ? "OK" : "FAIL",
                   !ok && e ? [NSString stringWithFormat:@" (%@)", [e localizedDescription]].UTF8String : "");
        }

        // === Part 5: Scan all ANE classes ===
        printf("\n--- Part 5: All ANE-prefixed classes ---\n");
        unsigned int classCount;
        Class *allClasses = objc_copyClassList(&classCount);
        for (unsigned int i = 0; i < classCount; i++) {
            const char *name = class_getName(allClasses[i]);
            if (strstr(name, "ANE") || strstr(name, "ane")) {
                printf("  %s\n", name);
            }
        }
        free(allClasses);
        free(w);

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
        [fm removeItemAtPath:td error:nil];
        CFRelease(ioIn); CFRelease(ioOut);

        printf("\nDone.\n");
    }
    return 0;
}
