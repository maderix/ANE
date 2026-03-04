// Decomposed causal attention: Q@K^T on ANE, mask+softmax on CPU, scores@V on ANE
// This gives us causal masking with ANE acceleration for the matmuls
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

#define HEADS 12
#define HD 64
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

typedef struct { id model; NSString *td; } Kern;

static Kern compile_mil(NSString *mil) {
    Kern k = {nil, nil};
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, @{}, nil);
    if (!desc) { printf("desc=NULL\n"); return k; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("compile FAIL: %s\n", e?[[e localizedDescription] UTF8String]:"");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil]; return k;
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

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        // === Approach 1: Non-causal SDPA (baseline) ===
        printf("=== Non-causal SDPA (baseline) ===\n");
        NSString *sdpa_mil = [NSString stringWithFormat:
            @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
            "    func main<ios16>(tensor<fp16, [1, %d, %d, %d]> q, "
            "tensor<fp16, [1, %d, %d, %d]> k, tensor<fp16, [1, %d, %d, %d]> v) {\n"
            "        tensor<fp16, [1, %d, %d, %d]> att = scaled_dot_product_attention("
            "query = q, key = k, value = v)[name = tensor<string, []>(\"sdpa\")];\n"
            "    } -> (att);\n}\n",
            HEADS, SEQ, HD, HEADS, SEQ, HD, HEADS, SEQ, HD, HEADS, SEQ, HD];
        Kern kSDPA = compile_mil(sdpa_mil);
        printf("SDPA compile: %s\n", kSDPA.model ? "OK" : "FAIL");

        // === Approach 2: Decomposed causal via matmul ops ===
        // Step 1: Q @ K^T → scores [1, HEADS, SEQ, SEQ]
        // MIL matmul: matmul(x=Q, y=K, transpose_y=true)
        // Q shape: [1, HEADS, SEQ, HD], K shape: [1, HEADS, SEQ, HD]
        // scores = Q @ K^T → [1, HEADS, SEQ, SEQ]
        printf("\n=== Decomposed causal attention ===\n");
        NSString *qkt_mil = [NSString stringWithFormat:
            @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
            "    func main<ios16>(tensor<fp16, [1, %d, %d, %d]> q, "
            "tensor<fp16, [1, %d, %d, %d]> k) {\n"
            "        tensor<fp16, [1, %d, %d, %d]> scores = matmul("
            "x = q, y = k, transpose_y = true)[name = tensor<string, []>(\"qkt\")];\n"
            "    } -> (scores);\n}\n",
            HEADS, SEQ, HD, HEADS, SEQ, HD, HEADS, SEQ, SEQ];
        Kern kQKT = compile_mil(qkt_mil);
        printf("Q@K^T compile: %s\n", kQKT.model ? "OK" : "FAIL");

        // Step 3: scores_softmax @ V → output [1, HEADS, SEQ, HD]
        NSString *sv_mil = [NSString stringWithFormat:
            @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
            "    func main<ios16>(tensor<fp16, [1, %d, %d, %d]> s, "
            "tensor<fp16, [1, %d, %d, %d]> v) {\n"
            "        tensor<fp16, [1, %d, %d, %d]> out = matmul("
            "x = s, y = v)[name = tensor<string, []>(\"sv\")];\n"
            "    } -> (out);\n}\n",
            HEADS, SEQ, SEQ, HEADS, SEQ, HD, HEADS, SEQ, HD];
        Kern kSV = compile_mil(sv_mil);
        printf("scores@V compile: %s\n", kSV.model ? "OK" : "FAIL");

        if (!kSDPA.model || !kQKT.model || !kSV.model) {
            printf("Some kernels failed to compile, aborting\n");
            goto done;
        }

        // Generate test data
        srand48(42);
        int total_qkv = HEADS * SEQ * HD;
        _Float16 *Q = (_Float16*)malloc(total_qkv * 2);
        _Float16 *K = (_Float16*)malloc(total_qkv * 2);
        _Float16 *V = (_Float16*)malloc(total_qkv * 2);
        for (int i = 0; i < total_qkv; i++) {
            Q[i] = (_Float16)(0.5f * (2*drand48()-1));
            K[i] = (_Float16)(0.5f * (2*drand48()-1));
            V[i] = (_Float16)(0.5f * (2*drand48()-1));
        }

        // IOSurfaces for Q, K, V
        size_t qkv_bytes = total_qkv * 2;
        IOSurfaceRef ioQ = make_surface(qkv_bytes), ioK = make_surface(qkv_bytes), ioV = make_surface(qkv_bytes);
        IOSurfaceLock(ioQ, 0, NULL); memcpy(IOSurfaceGetBaseAddress(ioQ), Q, qkv_bytes); IOSurfaceUnlock(ioQ, 0, NULL);
        IOSurfaceLock(ioK, 0, NULL); memcpy(IOSurfaceGetBaseAddress(ioK), K, qkv_bytes); IOSurfaceUnlock(ioK, 0, NULL);
        IOSurfaceLock(ioV, 0, NULL); memcpy(IOSurfaceGetBaseAddress(ioV), V, qkv_bytes); IOSurfaceUnlock(ioV, 0, NULL);

        // Scores IOSurface: [1, HEADS, SEQ, SEQ]
        int total_scores = HEADS * SEQ * SEQ;
        size_t scores_bytes = total_scores * 2;
        IOSurfaceRef ioScores = make_surface(scores_bytes);
        IOSurfaceRef ioOut_sdpa = make_surface(qkv_bytes);
        IOSurfaceRef ioOut_decomp = make_surface(qkv_bytes);

        // === Run non-causal SDPA ===
        {
            IOSurfaceRef ins[] = {ioQ, ioK, ioV};
            if (!ane_eval(&kSDPA, ins, 3, ioOut_sdpa)) { printf("SDPA eval FAIL\n"); goto done; }
        }

        // === Run decomposed causal ===
        // Step 1: Q@K^T on ANE
        {
            IOSurfaceRef ins[] = {ioQ, ioK};
            if (!ane_eval(&kQKT, ins, 2, ioScores)) { printf("Q@K^T eval FAIL\n"); goto done; }
        }

        // Step 2: Scale + causal mask + softmax on CPU
        {
            IOSurfaceLock(ioScores, 0, NULL);
            _Float16 *scores = (_Float16*)IOSurfaceGetBaseAddress(ioScores);
            float scale = 1.0f / sqrtf((float)HD);
            for (int h = 0; h < HEADS; h++) {
                for (int t = 0; t < SEQ; t++) {
                    // Apply scale, causal mask, and softmax
                    float row[SEQ], maxs = -1e30f;
                    for (int t2 = 0; t2 < SEQ; t2++) {
                        float s = (float)scores[h*SEQ*SEQ + t*SEQ + t2] * scale;
                        if (t2 > t) s = -1e30f;  // causal mask
                        row[t2] = s;
                        if (s > maxs) maxs = s;
                    }
                    float sum = 0;
                    for (int t2 = 0; t2 < SEQ; t2++) { row[t2] = expf(row[t2] - maxs); sum += row[t2]; }
                    for (int t2 = 0; t2 < SEQ; t2++)
                        scores[h*SEQ*SEQ + t*SEQ + t2] = (_Float16)(row[t2] / sum);
                }
            }
            IOSurfaceUnlock(ioScores, 0, NULL);
        }

        // Step 3: softmax_scores @ V on ANE
        {
            IOSurfaceRef ins[] = {ioScores, ioV};
            if (!ane_eval(&kSV, ins, 2, ioOut_decomp)) { printf("scores@V eval FAIL\n"); goto done; }
        }

        // === Verify decomposed causal ===
        {
            float scale = 1.0f / sqrtf((float)HD);
            IOSurfaceLock(ioOut_decomp, kIOSurfaceLockReadOnly, NULL);
            _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(ioOut_decomp);
            float maxdiff = 0;
            for (int h = 0; h < HEADS; h++)
                for (int t = 0; t < SEQ; t++) {
                    float scores[SEQ], maxs = -1e30f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float s = 0;
                        for (int d = 0; d < HD; d++) s += (float)Q[h*SEQ*HD+t*HD+d]*(float)K[h*SEQ*HD+t2*HD+d];
                        s *= scale; scores[t2] = s; if(s>maxs) maxs=s;
                    }
                    float sum = 0;
                    for (int t2 = 0; t2 <= t; t2++) { scores[t2]=expf(scores[t2]-maxs); sum+=scores[t2]; }
                    for (int t2 = 0; t2 <= t; t2++) scores[t2]/=sum;
                    for (int d = 0; d < HD; d++) {
                        float ref = 0;
                        for (int t2 = 0; t2 <= t; t2++) ref += scores[t2]*(float)V[h*SEQ*HD+t2*HD+d];
                        float diff = fabsf((float)out[h*SEQ*HD+t*HD+d] - ref);
                        if(diff>maxdiff) maxdiff=diff;
                    }
                }
            IOSurfaceUnlock(ioOut_decomp, kIOSurfaceLockReadOnly, NULL);
            printf("\nDecomposed causal max diff vs CPU ref: %.6f\n", maxdiff);
        }

        // === Benchmark: SDPA vs decomposed ===
        printf("\n=== Benchmarks ===\n");
        int N = 500;
        {
            IOSurfaceRef ins[] = {ioQ, ioK, ioV};
            // Warmup
            for (int i = 0; i < 10; i++) ane_eval(&kSDPA, ins, 3, ioOut_sdpa);
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) ane_eval(&kSDPA, ins, 3, ioOut_sdpa);
            double ms = tb_ms(mach_absolute_time() - t0);
            double flops = 4.0 * HEADS * SEQ * SEQ * HD;
            printf("SDPA (non-causal): %.3f ms/eval, %.1f GFLOPS\n", ms/N, N*flops/ms/1e6);
        }
        {
            // Decomposed: QKT + CPU softmax + SV
            // Warmup
            for (int i = 0; i < 10; i++) {
                IOSurfaceRef ins1[] = {ioQ, ioK};
                ane_eval(&kQKT, ins1, 2, ioScores);
                // Skip CPU softmax in benchmark for ANE-only timing
                IOSurfaceRef ins2[] = {ioScores, ioV};
                ane_eval(&kSV, ins2, 2, ioOut_decomp);
            }
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                IOSurfaceRef ins1[] = {ioQ, ioK};
                ane_eval(&kQKT, ins1, 2, ioScores);
                // CPU softmax + causal mask
                IOSurfaceLock(ioScores, 0, NULL);
                _Float16 *sc = (_Float16*)IOSurfaceGetBaseAddress(ioScores);
                float scale = 1.0f / sqrtf((float)HD);
                for (int h = 0; h < HEADS; h++)
                    for (int t = 0; t < SEQ; t++) {
                        float row[SEQ], maxs = -1e30f;
                        for (int t2 = 0; t2 < SEQ; t2++) {
                            float s = (float)sc[h*SEQ*SEQ+t*SEQ+t2] * scale;
                            if (t2 > t) s = -1e30f;
                            row[t2] = s; if(s>maxs) maxs=s;
                        }
                        float sum = 0;
                        for (int t2 = 0; t2 < SEQ; t2++) { row[t2]=expf(row[t2]-maxs); sum+=row[t2]; }
                        for (int t2 = 0; t2 < SEQ; t2++)
                            sc[h*SEQ*SEQ+t*SEQ+t2] = (_Float16)(row[t2]/sum);
                    }
                IOSurfaceUnlock(ioScores, 0, NULL);
                IOSurfaceRef ins2[] = {ioScores, ioV};
                ane_eval(&kSV, ins2, 2, ioOut_decomp);
            }
            double ms = tb_ms(mach_absolute_time() - t0);
            double flops = 4.0 * HEADS * SEQ * SEQ * HD;
            printf("Decomposed causal: %.3f ms/eval, %.1f GFLOPS\n", ms/N, N*flops/ms/1e6);
        }

        CFRelease(ioQ); CFRelease(ioK); CFRelease(ioV);
        CFRelease(ioScores); CFRelease(ioOut_sdpa); CFRelease(ioOut_decomp);
        free(Q); free(K); free(V);

        done:
        cleanup_kern(&kSDPA);
        cleanup_kern(&kQKT);
        cleanup_kern(&kSV);
        printf("\nDONE\n");
    }
    return 0;
}
