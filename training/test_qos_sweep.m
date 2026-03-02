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

        NSDictionary *weights = @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}};
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        NSFileManager *fm = [NSFileManager defaultManager];

        printf("=== QoS Sweep: compile/load/eval with varying QoS ===\n");
        printf("Kernel: %dx%d conv, spatial=%d (%.1f MFLOPS)\n", CH, CH, SP, 2.0*CH*CH*SP/1e6);
        printf("%4s %10s %10s %10s %10s  %s\n", "QoS", "Compile", "Load", "Eval(1)", "Eval(avg10)", "Status");

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

            int ioBytes = CH * SP * 4;
            IOSurfaceRef ioIn = make_surface(ioBytes);
            IOSurfaceRef ioOut = make_surface(ioBytes);
            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            IOSurfaceLock(ioIn, 0, NULL);
            float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
            for (int i = 0; i < CH*SP; i++) inp[i] = 0.5f;
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
