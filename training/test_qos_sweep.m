// test_qos_sweep.m — Does QoS affect frequency/latency?
// Sweep QoS 0-63 on compile, load, eval of a working kernel.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static int g_fp16_io = 0;  // M1/M2: cast op unsupported, use fp16 I/O directly

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

        Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        // 256x256 conv, spatial=64 for measurable latency
        int CH = 256, SP = 64;
        int ws = CH*CH*2, tot = 128+ws;
        uint8_t *blob = (uint8_t*)calloc(tot, 1);
        blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
        *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
        _Float16 *wp = (_Float16*)(blob+128);
        for (int i = 0; i < CH*CH; i++) wp[i] = (_Float16)(0.01f * (i % 100 - 50));
        NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];

        NSFileManager *fm = [NSFileManager defaultManager];

        printf("=== QoS Sweep: compile/load/eval with varying QoS ===\n");
        printf("Kernel: %dx%d conv, spatial=%d (%.1f MFLOPS)\n", CH, CH, SP, 2.0*CH*CH*SP/1e6);
        printf("%4s %10s %10s %10s %10s  %s\n", "QoS", "Compile", "Load", "Eval(1)", "Eval(avg10)", "Status");

        retry_mil:;
        NSString *mil;
        if (g_fp16_io) {
            mil = [NSString stringWithFormat:
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
                "    } -> (y);\n}\n", CH, SP, CH, CH, CH, CH, CH, SP];
        } else {
            mil = [NSString stringWithFormat:
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
                "    } -> (y);\n}\n", CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP];
        }
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

        unsigned int qos_values[] = {0, 1, 5, 10, 15, 17, 19, 21, 25, 31, 33, 40, 47, 50, 55, 60, 63};
        int n_qos = sizeof(qos_values)/sizeof(qos_values[0]);

        for (int qi = 0; qi < n_qos; qi++) {
            unsigned int qos = qos_values[qi];
            NSError *e = nil;

            // Make unique weights per iteration so hex differs
            _Float16 *wq = (_Float16*)(blob+128);
            wq[0] = (_Float16)(0.001f * qi);
            NSData *wdata_q = [NSData dataWithBytes:blob length:tot];
            NSDictionary *weights_q = @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata_q}};

            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:),
                milData, weights_q, nil);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [wdata_q writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

            uint64_t t0 = mach_absolute_time();
            BOOL cok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), qos, @{}, &e);
            double cms = tb_ms(mach_absolute_time() - t0);

            if (!cok) {
                if (!g_fp16_io) {
                    printf("[ANE] fp32 compile failed, retrying with fp16 I/O (M1/M2 fallback)\n");
                    g_fp16_io = 1;
                    [fm removeItemAtPath:td error:nil];
                    goto retry_mil;
                }
                printf("%4u %10s %10s %10s %10s  COMPILE_FAIL\n", qos, "-", "-", "-", "-");
                [fm removeItemAtPath:td error:nil];
                continue;
            }

            t0 = mach_absolute_time();
            BOOL lok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), qos, @{}, &e);
            double lms = tb_ms(mach_absolute_time() - t0);

            if (!lok) {
                printf("%4u %8.1fms %10s %10s %10s  LOAD_FAIL\n", qos, cms, "-", "-", "-");
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
                [fm removeItemAtPath:td error:nil];
                continue;
            }

            int ioBytes = CH * SP * (g_fp16_io ? 2 : 4);
            IOSurfaceRef ioIn = make_surface(ioBytes);
            IOSurfaceRef ioOut = make_surface(ioBytes);
            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            IOSurfaceLock(ioIn, 0, NULL);
            if (g_fp16_io) {
                _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
                for (int i = 0; i < CH*SP; i++) inp[i] = (_Float16)0.5f;
            } else {
                float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
                for (int i = 0; i < CH*SP; i++) inp[i] = 0.5f;
            }
            IOSurfaceUnlock(ioIn, 0, NULL);

            t0 = mach_absolute_time();
            BOOL eok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), qos, @{}, req, &e);
            double ems1 = tb_ms(mach_absolute_time() - t0);

            if (!eok) {
                printf("%4u %8.1fms %8.1fms %10s %10s  EVAL_FAIL\n", qos, cms, lms, "-", "-");
            } else {
                t0 = mach_absolute_time();
                for (int i = 0; i < 10; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdl, @selector(evaluateWithQoS:options:request:error:), qos, @{}, req, &e);
                }
                double ems_avg = tb_ms(mach_absolute_time() - t0) / 10.0;
                printf("%4u %8.1fms %8.1fms %8.2fms %8.2fms  OK\n", qos, cms, lms, ems1, ems_avg);
            }

            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            CFRelease(ioIn); CFRelease(ioOut);
            [fm removeItemAtPath:td error:nil];
        }

        printf("\nDone.\n");
    }
    return 0;
}
