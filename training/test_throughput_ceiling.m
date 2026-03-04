// test_throughput_ceiling.m — Experiment I: Multi-kernel throughput ceiling
// Measures CPU round-trip overhead for sequential ANE kernel execution
// Build: make test_throughput_ceiling && ./test_throughput_ceiling
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <dispatch/dispatch.h>
#include "ane_runtime.h"

static int g_fp16_io = 1;

static NSString *gen_conv_mil_fp16(int ch, int sp) {
    return [NSString stringWithFormat:
        @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>"
        "({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<string, []> pt = const()[name=tensor<string, []>(\"pt\"),"
        " val=tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=tensor<string, []>(\"st\"),"
        " val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=tensor<string, []>(\"pd\"),"
        " val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=tensor<string, []>(\"dl\"),"
        " val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, []> gr = const()[name=tensor<string, []>(\"gr\"),"
        " val=tensor<int32, []>(1)];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=tensor<string, []>(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>"
        "(\"@model_path/weights/weight.bin\"), offset=tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,"
        "pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
        "[name=tensor<string, []>(\"conv\")];\n"
        "    } -> (y);\n}\n", ch, sp, ch, ch, ch, ch, ch, sp];
}

static ANEKernel *compile_fp16_kernel(int ch, int sp) {
    int ws = ch * ch * 2;
    int tot = 128 + ws;
    uint8_t *blob = (uint8_t *)calloc((size_t)tot, 1);
    blob[0] = 1; blob[4] = 2;
    blob[64] = 0xEF; blob[65] = 0xBE; blob[66] = 0xAD; blob[67] = 0xDE;
    blob[68] = 1;
    *(uint32_t *)(blob + 72) = (uint32_t)ws;
    *(uint32_t *)(blob + 80) = 128;
    _Float16 *wp = (_Float16 *)(blob + 128);
    for (int i = 0; i < ch; i++) wp[i * ch + i] = (_Float16)1.0f;
    NSData *wdata = [NSData dataWithBytesNoCopy:blob length:(NSUInteger)tot
                                   freeWhenDone:YES];

    NSString *mil = gen_conv_mil_fp16(ch, sp);
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    size_t ioBytes = (size_t)ch * sp * 2;
    return ane_compile(md, wdata, 1, &ioBytes, 1, &ioBytes);
}

int main(int argc, const char *argv[]) {
    (void)argc; (void)argv;
    @autoreleasepool {
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);

        printf("============================================================\n");
        printf("  Experiment I: Multi-Kernel Throughput Ceiling\n");
        printf("  Measuring CPU round-trip overhead for sequential ANE ops\n");
        printf("============================================================\n\n");

        ane_init();
        if (!g_ane_ok) { printf("ANE not available\n"); return 1; }

        typedef struct { int ch; int sp; const char *name; } Config;
        Config configs[] = {
            {64,  32,  "64x32 (test)"},
            {256, 64,  "256x64 (small)"},
            {768, 256, "768x256 (prod)"},
        };
        int nconfigs = sizeof(configs) / sizeof(configs[0]);

        for (int ci = 0; ci < nconfigs; ci++) {
            Config cfg = configs[ci];
            printf("=== Config: %s ===\n", cfg.name);

            int nlayers = 12;
            ANEKernel *kernels[12];
            int compiled = 0;
            for (int i = 0; i < nlayers; i++) {
                @try {
                    kernels[i] = compile_fp16_kernel(cfg.ch, cfg.sp);
                    if (!kernels[i]) {
                        printf("  Kernel %d compile failed\n", i);
                        break;
                    }
                    compiled++;
                } @catch (NSException *ex) {
                    printf("  Kernel %d exception: %s\n", i,
                           [[ex reason] UTF8String]);
                    break;
                }
            }
            printf("  Compiled %d/%d kernels\n", compiled, nlayers);
            if (compiled < 2) {
                printf("  Need at least 2 kernels, skipping\n\n");
                for (int i = 0; i < compiled; i++) ane_free(kernels[i]);
                continue;
            }

            size_t ioBytes = (size_t)cfg.ch * cfg.sp * 2;
            int warmup = 5;
            int iters = 50;

            // --- Test 1: Sequential (run + memcpy chain) ---
            printf("\n  --- Test 1: Sequential (run + memcpy) ---\n");
            {
                for (int w = 0; w < warmup; w++) {
                    @try {
                        for (int i = 0; i < compiled; i++)
                            ane_eval(kernels[i]);
                    } @catch (NSException *ex) { (void)ex; }
                }

                uint64_t t0 = mach_absolute_time();
                for (int it = 0; it < iters; it++) {
                    for (int i = 0; i < compiled - 1; i++) {
                        @try {
                            ane_eval(kernels[i]);
                            IOSurfaceLock(kernels[i]->ioOutputs[0],
                                kIOSurfaceLockReadOnly, NULL);
                            IOSurfaceLock(kernels[i+1]->ioInputs[0], 0, NULL);
                            memcpy(
                                IOSurfaceGetBaseAddress(kernels[i+1]->ioInputs[0]),
                                IOSurfaceGetBaseAddress(kernels[i]->ioOutputs[0]),
                                ioBytes);
                            IOSurfaceUnlock(kernels[i+1]->ioInputs[0], 0, NULL);
                            IOSurfaceUnlock(kernels[i]->ioOutputs[0],
                                kIOSurfaceLockReadOnly, NULL);
                        } @catch (NSException *ex) { (void)ex; }
                    }
                    @try {
                        ane_eval(kernels[compiled - 1]);
                    } @catch (NSException *ex) { (void)ex; }
                }
                double totalMs = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
                double perIter = totalMs / iters;
                double perKernel = perIter / compiled;
                printf("  Total: %.2f ms/pass (%d kernels)\n", perIter, compiled);
                printf("  Per kernel: %.3f ms\n", perKernel);
                printf("  Throughput: %.0f kernels/s\n", compiled * 1000.0 / perIter);
            }

            // --- Test 2: Run-only (no memcpy, pure ANE overhead) ---
            printf("\n  --- Test 2: Run-only (no memcpy between) ---\n");
            {
                uint64_t t0 = mach_absolute_time();
                for (int it = 0; it < iters; it++) {
                    for (int i = 0; i < compiled; i++) {
                        @try {
                            ane_eval(kernels[i]);
                        } @catch (NSException *ex) { (void)ex; }
                    }
                }
                double totalMs = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
                double perIter = totalMs / iters;
                double perKernel = perIter / compiled;
                printf("  Total: %.2f ms/pass (%d kernels)\n", perIter, compiled);
                printf("  Per kernel: %.3f ms\n", perKernel);
                printf("  Throughput: %.0f kernels/s\n", compiled * 1000.0 / perIter);
            }

            // --- Test 3: Memcpy-only overhead ---
            printf("\n  --- Test 3: Memcpy-only overhead ---\n");
            {
                uint64_t t0 = mach_absolute_time();
                for (int it = 0; it < iters * 10; it++) {
                    for (int i = 0; i < compiled - 1; i++) {
                        IOSurfaceLock(kernels[i]->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
                        IOSurfaceLock(kernels[i+1]->ioInputs[0], 0, NULL);
                        memcpy(
                            IOSurfaceGetBaseAddress(kernels[i+1]->ioInputs[0]),
                            IOSurfaceGetBaseAddress(kernels[i]->ioOutputs[0]),
                            ioBytes);
                        IOSurfaceUnlock(kernels[i+1]->ioInputs[0], 0, NULL);
                        IOSurfaceUnlock(kernels[i]->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
                    }
                }
                double totalMs = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
                double perIter = totalMs / (iters * 10);
                double perCopy = perIter / (compiled - 1);
                printf("  Total: %.3f ms/pass (%d copies)\n", perIter, compiled - 1);
                printf("  Per memcpy: %.4f ms (%lu bytes)\n", perCopy, (unsigned long)ioBytes);
            }

            // --- Test 4: GCD serial queue ---
            printf("\n  --- Test 4: GCD serial queue ---\n");
            {
                ANEKernel **kptrs = (ANEKernel **)malloc(
                    (size_t)compiled * sizeof(ANEKernel *));
                for (int i = 0; i < compiled; i++) kptrs[i] = kernels[i];

                dispatch_queue_t q = dispatch_queue_create(
                    "ane.throughput", DISPATCH_QUEUE_SERIAL);
                dispatch_semaphore_t sem = dispatch_semaphore_create(0);
                const int ncomp = compiled;

                uint64_t t0 = mach_absolute_time();
                for (int it = 0; it < iters; it++) {
                    __block int done = 0;
                    for (int i = 0; i < ncomp; i++) {
                        ANEKernel *kp = kptrs[i];
                        dispatch_async(q, ^{
                            @try {
                                ane_eval(kp);
                            } @catch (NSException *ex) { (void)ex; }
                            done++;
                            if (done == ncomp)
                                dispatch_semaphore_signal(sem);
                        });
                    }
                    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
                }
                double totalMs = (double)(mach_absolute_time() - t0)
                    * tb.numer / tb.denom / 1e6;
                double perIter = totalMs / iters;
                printf("  Total: %.2f ms/pass (%d kernels, serial queue)\n",
                       perIter, ncomp);
                printf("  Per kernel: %.3f ms\n", perIter / ncomp);
                free(kptrs);
            }

            printf("\n  --- CPU Round-trip Overhead ---\n");
            printf("  Overhead = (Sequential - RunOnly) / %d copies\n", compiled - 1);
            printf("  This is what chaining would eliminate per layer.\n");

            for (int i = 0; i < compiled; i++) ane_free(kernels[i]);
            printf("\n");
        }

        printf("Done.\n");
    }
    return 0;
}
