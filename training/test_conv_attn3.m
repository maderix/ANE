// Grouped conv causal attention with CORRECT layout A: blob[oc*ICg + ic]
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

#define HEADS 12
#define HD 64
#define DIM (HEADS*HD)
#define SEQ 64

static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}
static NSData *build_blob_raw(_Float16 *data, int count) {
    int wsize = count * 2, total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0]=1; buf[4]=2; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
    *(uint32_t*)(buf+72)=wsize; *(uint32_t*)(buf+80)=128;
    memcpy(buf+128, data, wsize);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}
typedef struct { id model; NSString *td; } Kern;
static Kern compile_mil(NSString *mil, NSDictionary *wd) {
    Kern k = {nil, nil};
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, wd ?: @{}, nil);
    if (!desc) { printf("desc=NULL\n"); return k; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in wd) {
        [wd[path][@"data"] writeToFile:[td stringByAppendingPathComponent:
            [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""]] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("compile FAIL: %s\n", e?[[e localizedDescription] UTF8String]:""); return k;
    }
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    k.model = mdl; k.td = td;
    return k;
}
static BOOL ane_eval(Kern *k, IOSurfaceRef *ins, int nin, IOSurfaceRef out) {
    NSMutableArray *inArr = [NSMutableArray array], *inIdx = [NSMutableArray array];
    for (int i = 0; i < nin; i++) {
        [inArr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ins[i])];
        [inIdx addObject:@(i)];
    }
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), out);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        inArr, inIdx, @[wO], @[@0], nil, nil, @0);
    NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}
static void cleanup_kern(Kern *k) {
    if (!k->model) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k->model, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:k->td error:nil];
}

static NSString *gen_conv_mil(int ic, int oc, int icg, int groups, int sp) {
    return [NSString stringWithFormat:
        @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(%d)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = W, x = x)[name = tensor<string, []>(\"cv\")];\n"
        "    } -> (y);\n}\n", ic, sp, oc, icg, oc, icg, groups, oc, sp];
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        printf("=== Grouped Conv Causal Attention (layout A) ===\n");
        printf("HEADS=%d HD=%d SEQ=%d\n\n", HEADS, HD, SEQ);

        srand48(42);
        float *Q = (float*)malloc(SEQ*DIM*sizeof(float));
        float *K = (float*)malloc(SEQ*DIM*sizeof(float));
        float *V = (float*)malloc(SEQ*DIM*sizeof(float));
        for (int i = 0; i < SEQ*DIM; i++) {
            Q[i] = 0.5f*(2*drand48()-1);
            K[i] = 0.5f*(2*drand48()-1);
            V[i] = 0.5f*(2*drand48()-1);
        }

        // Q@K^T grouped conv weight: [HEADS*SEQ, HD, 1, 1] with groups=HEADS
        // Layout A: blob[oc * ICg + ic] where ICg = HD
        // For head h: oc = h*SEQ+t2, ic = d (within group)
        // We want: output[h*SEQ+t2, t] = sum_d Q[h*HD+d, t] * K_weight[h*SEQ+t2, d]
        // So K_weight[oc, ic] = K[t2, h*HD+d] where oc=h*SEQ+t2, ic=d
        int kw_count = HEADS * SEQ * HD;
        _Float16 *kw = (_Float16*)malloc(kw_count * sizeof(_Float16));
        for (int h = 0; h < HEADS; h++)
            for (int t2 = 0; t2 < SEQ; t2++)
                for (int d = 0; d < HD; d++) {
                    int oc = h*SEQ + t2;
                    kw[oc*HD + d] = (_Float16)K[t2*DIM + h*HD + d];
                }
        NSDictionary *qkt_wd = @{@"@model_path/weights/w.bin": @{@"offset":@0, @"data":build_blob_raw(kw, kw_count)}};
        free(kw);

        // scores@V grouped conv weight: [HEADS*HD, SEQ, 1, 1] with groups=HEADS
        // oc = h*HD+d, ic = t2 (within group, ICg=SEQ)
        // V_weight[oc, ic] = V[t2, h*HD+d]
        int vw_count = HEADS * HD * SEQ;
        _Float16 *vw = (_Float16*)malloc(vw_count * sizeof(_Float16));
        for (int h = 0; h < HEADS; h++)
            for (int d = 0; d < HD; d++)
                for (int t2 = 0; t2 < SEQ; t2++) {
                    int oc = h*HD + d;
                    vw[oc*SEQ + t2] = (_Float16)V[t2*DIM + h*HD + d];
                }
        NSDictionary *sv_wd = @{@"@model_path/weights/w.bin": @{@"offset":@0, @"data":build_blob_raw(vw, vw_count)}};
        free(vw);

        // Compile
        printf("Compiling Q@K^T (grouped conv, groups=%d)...\n", HEADS);
        NSString *qkt_mil = gen_conv_mil(HEADS*HD, HEADS*SEQ, HD, HEADS, SEQ);
        Kern kQKT = compile_mil(qkt_mil, qkt_wd);
        printf("  %s\n", kQKT.model ? "OK" : "FAIL");

        printf("Compiling scores@V (grouped conv, groups=%d)...\n", HEADS);
        NSString *sv_mil = gen_conv_mil(HEADS*SEQ, HEADS*HD, SEQ, HEADS, SEQ);
        Kern kSV = compile_mil(sv_mil, sv_wd);
        printf("  %s\n", kSV.model ? "OK" : "FAIL");

        if (!kQKT.model || !kSV.model) { printf("FAIL\n"); goto done; }

        // Prepare Q IOSurface [1, DIM, 1, SEQ] fp16
        size_t q_bytes = DIM * SEQ * 2;
        IOSurfaceRef ioQ = make_surface(q_bytes);
        IOSurfaceLock(ioQ, 0, NULL);
        _Float16 *qp = (_Float16*)IOSurfaceGetBaseAddress(ioQ);
        for (int t = 0; t < SEQ; t++)
            for (int c = 0; c < DIM; c++)
                qp[c*SEQ + t] = (_Float16)Q[t*DIM + c];
        IOSurfaceUnlock(ioQ, 0, NULL);

        size_t sc_bytes = HEADS * SEQ * SEQ * 2;
        IOSurfaceRef ioScores = make_surface(sc_bytes);
        IOSurfaceRef ioOut = make_surface(q_bytes);

        // Step 1: Q@K^T
        IOSurfaceRef ins1[] = {ioQ};
        if (!ane_eval(&kQKT, ins1, 1, ioScores)) { printf("Q@K^T eval FAIL\n"); goto done; }

        // Step 2: Scale + causal mask + softmax (CPU)
        float scale = 1.0f / sqrtf((float)HD);
        IOSurfaceLock(ioScores, 0, NULL);
        _Float16 *sc = (_Float16*)IOSurfaceGetBaseAddress(ioScores);
        for (int h = 0; h < HEADS; h++)
            for (int t = 0; t < SEQ; t++) {
                float row[SEQ], maxs = -1e30f;
                for (int t2 = 0; t2 < SEQ; t2++) {
                    // scores channel = h*SEQ+t2, spatial = t
                    float s = (float)sc[(h*SEQ+t2)*SEQ + t] * scale;
                    if (t2 > t) s = -1e30f;
                    row[t2] = s;
                    if (s > maxs) maxs = s;
                }
                float sum = 0;
                for (int t2 = 0; t2 < SEQ; t2++) { row[t2] = expf(row[t2]-maxs); sum += row[t2]; }
                for (int t2 = 0; t2 < SEQ; t2++)
                    sc[(h*SEQ+t2)*SEQ + t] = (_Float16)(row[t2] / sum);
            }
        IOSurfaceUnlock(ioScores, 0, NULL);

        // Step 3: scores@V
        IOSurfaceRef ins2[] = {ioScores};
        if (!ane_eval(&kSV, ins2, 1, ioOut)) { printf("scores@V eval FAIL\n"); goto done; }

        // Verify
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(ioOut);
        float maxdiff = 0;
        for (int h = 0; h < HEADS; h++)
            for (int t = 0; t < SEQ; t++) {
                float sc2[SEQ], maxs = -1e30f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float s = 0;
                    for (int d = 0; d < HD; d++) s += Q[t*DIM+h*HD+d]*K[t2*DIM+h*HD+d];
                    s *= scale; sc2[t2] = s; if(s>maxs) maxs=s;
                }
                float sum = 0;
                for (int t2 = 0; t2 <= t; t2++) { sc2[t2]=expf(sc2[t2]-maxs); sum+=sc2[t2]; }
                for (int t2 = 0; t2 <= t; t2++) sc2[t2]/=sum;
                for (int d = 0; d < HD; d++) {
                    float ref = 0;
                    for (int t2 = 0; t2 <= t; t2++) ref += sc2[t2]*V[t2*DIM+h*HD+d];
                    float diff = fabsf((float)out[(h*HD+d)*SEQ+t] - ref);
                    if (diff > maxdiff) maxdiff = diff;
                }
            }
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
        printf("\nMax diff vs CPU causal ref: %.6f → %s\n", maxdiff, maxdiff < 0.05f ? "PASS" : "FAIL");

        // Benchmark
        printf("\n=== Benchmark ===\n");
        int N = 500;
        for (int i = 0; i < 20; i++) { ane_eval(&kQKT, ins1, 1, ioScores); ane_eval(&kSV, ins2, 1, ioOut); }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ane_eval(&kQKT, ins1, 1, ioScores);
            ane_eval(&kSV, ins2, 1, ioOut);
        }
        double ms_ane = tb_ms(mach_absolute_time() - t0);

        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ane_eval(&kQKT, ins1, 1, ioScores);
            IOSurfaceLock(ioScores, 0, NULL);
            _Float16 *s = (_Float16*)IOSurfaceGetBaseAddress(ioScores);
            for (int h = 0; h < HEADS; h++)
                for (int t = 0; t < SEQ; t++) {
                    float row[SEQ], maxs = -1e30f;
                    for (int t2 = 0; t2 < SEQ; t2++) {
                        float v = (float)s[(h*SEQ+t2)*SEQ+t]*scale;
                        if(t2>t) v=-1e30f; row[t2]=v; if(v>maxs) maxs=v;
                    }
                    float sum=0;
                    for (int t2=0;t2<SEQ;t2++){row[t2]=expf(row[t2]-maxs);sum+=row[t2];}
                    for (int t2=0;t2<SEQ;t2++) s[(h*SEQ+t2)*SEQ+t]=(_Float16)(row[t2]/sum);
                }
            IOSurfaceUnlock(ioScores, 0, NULL);
            ane_eval(&kSV, ins2, 1, ioOut);
        }
        double ms_full = tb_ms(mach_absolute_time() - t0);

        double flops = 2.0 * HEADS * SEQ * SEQ * HD * 2;
        printf("ANE-only (2 convs):  %.3f ms/iter  %.1f GFLOPS\n", ms_ane/N, N*flops/ms_ane/1e6);
        printf("Full pipeline:       %.3f ms/iter  %.1f GFLOPS\n", ms_full/N, N*flops/ms_full/1e6);
        printf("CPU softmax:         %.3f ms/iter\n", (ms_full-ms_ane)/N);

        CFRelease(ioQ); CFRelease(ioScores); CFRelease(ioOut);
        free(Q); free(K); free(V);
        done:
        cleanup_kern(&kQKT); cleanup_kern(&kSV);
        printf("\nDONE\n");
    }
    return 0;
}
