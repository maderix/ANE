// test_bench_paths.m — Benchmark ANE evaluation paths at production dimensions
// Compares: standard, RT, processRequest, and ane_eval_rt wrapper
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static int g_fp16_io = 0;

#include "ane_runtime.h"

static NSString *gen_bench_conv(int ch, int sp) {
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

int main(int argc, char **argv) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        printf("=== ANE Eval Path Benchmark (production dimensions) ===\n\n");

        ane_init();
        if (!g_ane_ok) { printf("FATAL: ANE not available\n"); return 1; }

        typedef struct { int ch; int sp; const char *label; } TestConfig;
        TestConfig configs[] = {
            {64,  32,  "64x32  (test)"},
            {128, 64,  "128x64 (small)"},
            {256, 64,  "256x64 (med)"},
            {768, 256, "768x256 (prod)"},
            {512, 64,  "512x64 (large)"},
        };
        int nconfigs = sizeof(configs) / sizeof(configs[0]);
        int WARMUP = 20, ITERS = 200;

        id client = g_ane_client;
        printf("  Client: %s | Warmup: %d | Iters: %d\n\n", client ? "OK" : "NO", WARMUP, ITERS);
        printf("%-18s %10s %14s %14s %14s\n", "Config", "Standard", "RT", "ProcReq", "ane_eval_rt");
        printf("%-18s %10s %14s %14s %14s\n", "------", "--------", "--", "-------", "-----------");

        for (int ci = 0; ci < nconfigs; ci++) {
            int CH = configs[ci].ch, SP = configs[ci].sp;

            _Float16 *w = (_Float16*)calloc(CH*CH, sizeof(_Float16));
            for (int i = 0; i < CH; i++) w[i*CH+i] = (_Float16)0.5f;
            int ws = CH*CH*2, tot = 128+ws;
            uint8_t *blob = (uint8_t*)calloc(tot, 1);
            blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
            *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
            memcpy(blob+128, w, ws);
            NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];
            free(w);

            g_fp16_io = 1;
            NSString *mil = gen_bench_conv(CH, SP);
            NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
            size_t ioBytes = CH * SP * 2;
            ANEKernel *k = ane_compile(milData, wdata, 1, &ioBytes, 1, &ioBytes);
            if (!k) { printf("%-18s  (compile failed)\n", configs[ci].label); continue; }

            IOSurfaceLock(k->ioInputs[0], 0, NULL);
            _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(k->ioInputs[0]);
            for (int i = 0; i < CH*SP; i++) inp[i] = (_Float16)1.0f;
            IOSurfaceUnlock(k->ioInputs[0], 0, NULL);

            NSError *e = nil;

            for (int i = 0; i < WARMUP; i++) ane_eval(k);
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < ITERS; i++) ane_eval(k);
            double std_ms = tb_ms(mach_absolute_time() - t0) / ITERS;

            double rt_ms = -1;
            if (client) {
                @try {
                    for (int i = 0; i < WARMUP; i++)
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, @selector(evaluateRealTimeWithModel:options:request:error:),
                            k->model, @{}, k->request, &e);
                    t0 = mach_absolute_time();
                    for (int i = 0; i < ITERS; i++)
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, @selector(evaluateRealTimeWithModel:options:request:error:),
                            k->model, @{}, k->request, &e);
                    rt_ms = tb_ms(mach_absolute_time() - t0) / ITERS;
                } @catch (NSException *ex) { rt_ms = -1; }
            }

            double proc_ms = -1;
            @try {
                id prog = [k->model valueForKey:@"program"];
                id hexId = [k->model valueForKey:@"hexStringIdentifier"];
                SEL procSel = @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:);
                if (prog && [prog respondsToSelector:procSel]) {
                    for (int i = 0; i < WARMUP; i++) {
                        BOOL rv = NO;
                        ((BOOL(*)(id,SEL,id,id,unsigned int,int,id,id,BOOL*,NSError**))objc_msgSend)(
                            prog, procSel, k->request, k->model, 21, 0, hexId, @{}, &rv, &e);
                    }
                    t0 = mach_absolute_time();
                    for (int i = 0; i < ITERS; i++) {
                        BOOL rv = NO;
                        ((BOOL(*)(id,SEL,id,id,unsigned int,int,id,id,BOOL*,NSError**))objc_msgSend)(
                            prog, procSel, k->request, k->model, 21, 0, hexId, @{}, &rv, &e);
                    }
                    proc_ms = tb_ms(mach_absolute_time() - t0) / ITERS;
                }
            } @catch (NSException *ex) { (void)ex; }

            double wrap_ms = -1;
            @try {
                for (int i = 0; i < WARMUP; i++) ane_eval_rt(k);
                t0 = mach_absolute_time();
                for (int i = 0; i < ITERS; i++) ane_eval_rt(k);
                wrap_ms = tb_ms(mach_absolute_time() - t0) / ITERS;
            } @catch (NSException *ex) { wrap_ms = -1; }

            char s[32], r[32], p[32], w2[32];
            snprintf(s, 32, "%.3f ms", std_ms);
            snprintf(r, 32, rt_ms >= 0 ? "%.3f (%.1fx)" : "N/A", rt_ms, std_ms/rt_ms);
            snprintf(p, 32, proc_ms >= 0 ? "%.3f (%.1fx)" : "N/A", proc_ms, std_ms/proc_ms);
            snprintf(w2, 32, wrap_ms >= 0 ? "%.3f (%.1fx)" : "N/A", wrap_ms, std_ms/wrap_ms);
            printf("%-18s %10s %14s %14s %14s\n", configs[ci].label, s, r, p, w2);

            ane_free(k);
        }

        printf("\n=== Benchmark complete ===\n");
    }
    return 0;
}
