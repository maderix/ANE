// Full fused forward: QKV convs → reshape → matmul(Q,K^T) → scale+mask → softmax → matmul(scores,V) → Wo conv
// If ANE compiler rejects matmul, we'll know and fall back
// Also test: fused scores@V + Wo (2 convs in 1 dispatch)
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

#define DIM 768
#define HEADS 12
#define HD (DIM/HEADS)
#define HIDDEN 2048
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
static NSData *build_blob(const float *w, int oc, int ic) {
    int wsize = oc*ic*2, total = 128+wsize;
    uint8_t *buf = (uint8_t*)calloc(total,1);
    buf[0]=1; buf[4]=2; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
    *(uint32_t*)(buf+72)=wsize; *(uint32_t*)(buf+80)=128;
    _Float16 *fp16 = (_Float16*)(buf+128);
    for (int i = 0; i < oc*ic; i++) fp16[i] = (_Float16)w[i];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}
static NSData *build_blob_fp16(_Float16 *data, int count) {
    int wsize = count*2, total = 128+wsize;
    uint8_t *buf = (uint8_t*)calloc(total,1);
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
    if (!desc) { printf("  desc=NULL\n"); return k; }
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
        printf("  compile FAIL: %s\n", e?[[[e localizedDescription] substringToIndex:MIN(300,(int)[[e localizedDescription] length])] UTF8String]:"");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil]; return k;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  load FAIL\n"); [[NSFileManager defaultManager] removeItemAtPath:td error:nil]; return k;
    }
    k.model = mdl; k.td = td;
    return k;
}
static BOOL ane_eval_io(Kern *k, IOSurfaceRef *ins, int nin, IOSurfaceRef *outs, int nout) {
    NSMutableArray *inArr = [NSMutableArray array], *inIdx = [NSMutableArray array];
    NSMutableArray *outArr = [NSMutableArray array], *outIdx = [NSMutableArray array];
    for (int i = 0; i < nin; i++) {
        [inArr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ins[i])];
        [inIdx addObject:@(i)];
    }
    for (int i = 0; i < nout; i++) {
        [outArr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), outs[i])];
        [outIdx addObject:@(i)];
    }
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        inArr, inIdx, outArr, outIdx, nil, nil, @0);
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

        srand48(42);
        float sc_d = 1.0f/sqrtf(DIM), sc_h = 1.0f/sqrtf(HIDDEN);
        float *Wq = (float*)malloc(DIM*DIM*4); for(int i=0;i<DIM*DIM;i++) Wq[i]=sc_d*(2*drand48()-1);
        float *Wk = (float*)malloc(DIM*DIM*4); for(int i=0;i<DIM*DIM;i++) Wk[i]=sc_d*(2*drand48()-1);
        float *Wv = (float*)malloc(DIM*DIM*4); for(int i=0;i<DIM*DIM;i++) Wv[i]=sc_d*(2*drand48()-1);
        float *Wo = (float*)malloc(DIM*DIM*4); for(int i=0;i<DIM*DIM;i++) Wo[i]=sc_d*(2*drand48()-1);
        float *W1 = (float*)malloc(HIDDEN*DIM*4); for(int i=0;i<HIDDEN*DIM;i++) W1[i]=sc_h*(2*drand48()-1);
        float *W2 = (float*)malloc(DIM*HIDDEN*4); for(int i=0;i<DIM*HIDDEN;i++) W2[i]=sc_d*(2*drand48()-1);
        float *W3 = (float*)malloc(HIDDEN*DIM*4); for(int i=0;i<HIDDEN*DIM;i++) W3[i]=sc_h*(2*drand48()-1);

        // === Test 1: Full attention in one MIL graph ===
        // QKV convs → reshape → matmul(Q,K^T) → scale → add causal mask → softmax → matmul(scores,V) → reshape → Wo conv
        printf("=== Test 1: Full fused attention (QKV + matmul + softmax + Wo) ===\n");
        {
            // Build causal mask blob [1, 1, SEQ, SEQ]
            _Float16 *mask = (_Float16*)calloc(SEQ*SEQ, sizeof(_Float16));
            for (int t = 0; t < SEQ; t++)
                for (int t2 = 0; t2 < SEQ; t2++)
                    mask[t*SEQ+t2] = (t2 <= t) ? (_Float16)0.0f : (_Float16)(-65504.0f);

            // scale constant
            float scale_val = 1.0f / sqrtf((float)HD);

            NSString *mil = [NSString stringWithFormat:
                @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
                "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                // Conv boilerplate
                "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
                "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
                "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
                "        tensor<int32, []> gr1 = const()[name = tensor<string, []>(\"g1\"), val = tensor<int32, []>(1)];\n"
                // QKV weights
                "        tensor<fp16, [%d, %d, 1, 1]> Wq = const()[name = tensor<string, []>(\"Wq\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wq.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [%d, %d, 1, 1]> Wk = const()[name = tensor<string, []>(\"Wk\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wk.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [%d, %d, 1, 1]> Wv = const()[name = tensor<string, []>(\"Wv\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wv.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [%d, %d, 1, 1]> Wout = const()[name = tensor<string, []>(\"Wo\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wo.bin\"), offset = tensor<uint64, []>(64)))];\n"
                // QKV projections
                "        tensor<fp16, [1, %d, 1, %d]> q_flat = conv(dilations = dl, groups = gr1, pad = pd, "
                "pad_type = pt, strides = st, weight = Wq, x = x)[name = tensor<string, []>(\"cq\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> k_flat = conv(dilations = dl, groups = gr1, pad = pd, "
                "pad_type = pt, strides = st, weight = Wk, x = x)[name = tensor<string, []>(\"ck\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> v_flat = conv(dilations = dl, groups = gr1, pad = pd, "
                "pad_type = pt, strides = st, weight = Wv, x = x)[name = tensor<string, []>(\"cv\")];\n"
                // Reshape: [1, DIM, 1, SEQ] → [1, HEADS, HD, SEQ] → transpose → [1, HEADS, SEQ, HD]
                "        tensor<int32, [4]> qsh = const()[name = tensor<string, []>(\"qsh\"), val = tensor<int32, [4]>([1, %d, %d, %d])];\n"
                "        tensor<fp16, [1, %d, %d, %d]> q_4d = reshape(shape = qsh, x = q_flat)[name = tensor<string, []>(\"rq\")];\n"
                "        tensor<int32, [4]> perm = const()[name = tensor<string, []>(\"pm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n"
                "        tensor<fp16, [1, %d, %d, %d]> q = transpose(perm = perm, x = q_4d)[name = tensor<string, []>(\"tq\")];\n"
                "        tensor<fp16, [1, %d, %d, %d]> k_4d = reshape(shape = qsh, x = k_flat)[name = tensor<string, []>(\"rk\")];\n"
                "        tensor<fp16, [1, %d, %d, %d]> k = transpose(perm = perm, x = k_4d)[name = tensor<string, []>(\"tk\")];\n"
                "        tensor<fp16, [1, %d, %d, %d]> v_4d = reshape(shape = qsh, x = v_flat)[name = tensor<string, []>(\"rv\")];\n"
                "        tensor<fp16, [1, %d, %d, %d]> v = transpose(perm = perm, x = v_4d)[name = tensor<string, []>(\"tv\")];\n"
                // Q @ K^T
                "        tensor<bool, []> ty = const()[name = tensor<string, []>(\"ty\"), val = tensor<bool, []>(true)];\n"
                "        tensor<bool, []> tx = const()[name = tensor<string, []>(\"tx\"), val = tensor<bool, []>(false)];\n"
                "        tensor<fp16, [1, %d, %d, %d]> scores = matmul(transpose_x = tx, transpose_y = ty, x = q, y = k)[name = tensor<string, []>(\"mm1\")];\n"
                // Scale
                "        tensor<fp16, []> sc = const()[name = tensor<string, []>(\"sc\"), val = fp16(%f)];\n"
                "        tensor<fp16, [1, %d, %d, %d]> scaled = mul(x = scores, y = sc)[name = tensor<string, []>(\"scl\")];\n"
                // Causal mask
                "        tensor<fp16, [1, 1, %d, %d]> cmask = const()[name = tensor<string, []>(\"cm\"), "
                "val = tensor<fp16, [1, 1, %d, %d]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/mask.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [1, %d, %d, %d]> masked = add(x = scaled, y = cmask)[name = tensor<string, []>(\"msk\")];\n"
                // Softmax
                "        tensor<int32, []> sax = const()[name = tensor<string, []>(\"sax\"), val = tensor<int32, []>(-1)];\n"
                "        tensor<fp16, [1, %d, %d, %d]> attn_w = softmax(axis = sax, x = masked)[name = tensor<string, []>(\"sm\")];\n"
                // scores @ V
                "        tensor<fp16, [1, %d, %d, %d]> attn_4d = matmul(transpose_x = tx, transpose_y = tx, x = attn_w, y = v)[name = tensor<string, []>(\"mm2\")];\n"
                // Reshape back: [1, HEADS, SEQ, HD] → transpose → [1, HEADS, HD, SEQ] → reshape → [1, DIM, 1, SEQ]
                "        tensor<fp16, [1, %d, %d, %d]> attn_t = transpose(perm = perm, x = attn_4d)[name = tensor<string, []>(\"ta\")];\n"
                "        tensor<int32, [4]> osh = const()[name = tensor<string, []>(\"osh\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
                "        tensor<fp16, [1, %d, 1, %d]> attn_flat = reshape(shape = osh, x = attn_t)[name = tensor<string, []>(\"ra\")];\n"
                // Wo projection
                "        tensor<fp16, [1, %d, 1, %d]> out = conv(dilations = dl, groups = gr1, pad = pd, "
                "pad_type = pt, strides = st, weight = Wout, x = attn_flat)[name = tensor<string, []>(\"co\")];\n"
                "    } -> (out);\n}\n",
                DIM, SEQ,                              // input
                DIM,DIM,DIM,DIM, DIM,DIM,DIM,DIM,      // Wq, Wk
                DIM,DIM,DIM,DIM, DIM,DIM,DIM,DIM,      // Wv, Wo
                DIM, SEQ, DIM, SEQ, DIM, SEQ,           // q_flat, k_flat, v_flat
                HEADS, HD, SEQ,                         // reshape shape
                HEADS, HD, SEQ,                         // q_4d
                HEADS, SEQ, HD,                         // q (after transpose)
                HEADS, HD, SEQ,                         // k_4d
                HEADS, SEQ, HD,                         // k
                HEADS, HD, SEQ,                         // v_4d
                HEADS, SEQ, HD,                         // v
                HEADS, SEQ, SEQ,                        // scores
                scale_val,
                HEADS, SEQ, SEQ,                        // scaled
                SEQ, SEQ, SEQ, SEQ,                     // mask
                HEADS, SEQ, SEQ,                        // masked
                HEADS, SEQ, SEQ,                        // attn_w (softmax)
                HEADS, SEQ, HD,                         // attn_4d
                HEADS, HD, SEQ,                         // attn_t
                DIM, SEQ,                               // reshape back
                DIM, SEQ,                               // attn_flat
                DIM, SEQ];                              // out

            NSDictionary *wd = @{
                @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":build_blob(Wq,DIM,DIM)},
                @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":build_blob(Wk,DIM,DIM)},
                @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":build_blob(Wv,DIM,DIM)},
                @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":build_blob(Wo,DIM,DIM)},
                @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":build_blob_fp16(mask,SEQ*SEQ)},
            };
            free(mask);
            Kern k = compile_mil(mil, wd);
            if (k.model) {
                printf("  COMPILED! Full fused attention works on ANE!\n");

                // Verify vs CPU
                float *x = (float*)malloc(SEQ*DIM*4);
                for (int i = 0; i < SEQ*DIM; i++) x[i] = 0.1f*(2*drand48()-1);

                IOSurfaceRef ioIn = make_surface(DIM*SEQ*2);
                IOSurfaceRef ioOut = make_surface(DIM*SEQ*2);
                IOSurfaceLock(ioIn, 0, NULL);
                _Float16 *p = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
                for (int t = 0; t < SEQ; t++)
                    for (int c = 0; c < DIM; c++)
                        p[c*SEQ+t] = (_Float16)x[t*DIM+c];
                IOSurfaceUnlock(ioIn, 0, NULL);

                IOSurfaceRef ins[] = {ioIn}, outs[] = {ioOut};
                BOOL ok = ane_eval_io(&k, ins, 1, outs, 1);
                printf("  Eval: %s\n", ok?"OK":"FAIL");

                if (ok) {
                    // CPU reference
                    float *q_cpu = (float*)calloc(SEQ*DIM, 4);
                    float *k_cpu = (float*)calloc(SEQ*DIM, 4);
                    float *v_cpu = (float*)calloc(SEQ*DIM, 4);
                    for (int t=0;t<SEQ;t++) for (int oc=0;oc<DIM;oc++) {
                        float sq=0,sk=0,sv=0;
                        for (int ic=0;ic<DIM;ic++) {
                            sq += Wq[oc*DIM+ic]*x[t*DIM+ic];
                            sk += Wk[oc*DIM+ic]*x[t*DIM+ic];
                            sv += Wv[oc*DIM+ic]*x[t*DIM+ic];
                        }
                        q_cpu[t*DIM+oc]=sq; k_cpu[t*DIM+oc]=sk; v_cpu[t*DIM+oc]=sv;
                    }
                    // Attention
                    float *attn = (float*)calloc(SEQ*DIM, 4);
                    float asc = 1.0f/sqrtf((float)HD);
                    float *sc2 = (float*)malloc(SEQ*4);
                    for (int h=0;h<HEADS;h++) for (int t=0;t<SEQ;t++) {
                        float maxs=-1e30f;
                        for (int t2=0;t2<=t;t2++) {
                            float s=0;
                            for (int d=0;d<HD;d++) s+=q_cpu[t*DIM+h*HD+d]*k_cpu[t2*DIM+h*HD+d];
                            s*=asc; sc2[t2]=s; if(s>maxs) maxs=s;
                        }
                        float sum=0;
                        for (int t2=0;t2<=t;t2++){sc2[t2]=expf(sc2[t2]-maxs);sum+=sc2[t2];}
                        for (int t2=0;t2<=t;t2++) sc2[t2]/=sum;
                        for (int d=0;d<HD;d++){
                            float r=0;
                            for (int t2=0;t2<=t;t2++) r+=sc2[t2]*v_cpu[t2*DIM+h*HD+d];
                            attn[t*DIM+h*HD+d]=r;
                        }
                    }
                    free(sc2);
                    // Wo
                    float *ref = (float*)calloc(SEQ*DIM, 4);
                    for (int t=0;t<SEQ;t++) for (int oc=0;oc<DIM;oc++){
                        float s=0;
                        for (int ic=0;ic<DIM;ic++) s+=Wo[oc*DIM+ic]*attn[t*DIM+ic];
                        ref[t*DIM+oc]=s;
                    }
                    IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
                    _Float16 *o = (_Float16*)IOSurfaceGetBaseAddress(ioOut);
                    float maxdiff=0;
                    for (int t=0;t<SEQ;t++) for (int c=0;c<DIM;c++){
                        float diff=fabsf((float)o[c*SEQ+t]-ref[t*DIM+c]);
                        if(diff>maxdiff) maxdiff=diff;
                    }
                    IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
                    printf("  Max diff vs CPU: %.6f → %s\n", maxdiff, maxdiff<0.1f?"PASS":"FAIL");

                    // Benchmark
                    for (int i=0;i<20;i++) ane_eval_io(&k, ins, 1, outs, 1);
                    int N=500;
                    uint64_t t0 = mach_absolute_time();
                    for (int i=0;i<N;i++) ane_eval_io(&k, ins, 1, outs, 1);
                    double ms = tb_ms(mach_absolute_time()-t0);
                    // FLOPs: QKV=3*2*D*D*S + QKT=2*H*S*S*HD + SV=2*H*S*S*HD + Wo=2*D*D*S
                    double flops = 4.0*2*DIM*DIM*SEQ + 4.0*HEADS*SEQ*SEQ*HD;
                    printf("  %.3f ms/iter  %.1f GFLOPS (%.1f TFLOPS)\n", ms/N, N*flops/ms/1e6, N*flops/ms/1e9);

                    free(q_cpu); free(k_cpu); free(v_cpu); free(attn); free(ref);
                }
                CFRelease(ioIn); CFRelease(ioOut);
                free(x);
                cleanup_kern(&k);
            } else {
                printf("  Full fused attention FAILED to compile on ANE\n");
            }
        }

        // === Test 2: Fused FFN (already proven, just benchmark for comparison) ===
        printf("\n=== Test 2: Fused FFN benchmark ===\n");
        {
            NSString *mil = [NSString stringWithFormat:
                @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
                "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
                "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
                "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
                "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
                "        tensor<fp16, [%d, %d, 1, 1]> W1 = const()[name = tensor<string, []>(\"W1\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w1.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [%d, %d, 1, 1]> W3 = const()[name = tensor<string, []>(\"W3\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w3.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [%d, %d, 1, 1]> W2 = const()[name = tensor<string, []>(\"W2\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w2.bin\"), offset = tensor<uint64, []>(64)))];\n"
                "        tensor<fp16, [1, %d, 1, %d]> h1 = conv(dilations = dl, groups = gr, pad = pd, "
                "pad_type = pt, strides = st, weight = W1, x = x)[name = tensor<string, []>(\"c1\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> h3 = conv(dilations = dl, groups = gr, pad = pd, "
                "pad_type = pt, strides = st, weight = W3, x = x)[name = tensor<string, []>(\"c3\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = h1)[name = tensor<string, []>(\"sg\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = h1, y = sig)[name = tensor<string, []>(\"si\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> gate = mul(x = silu, y = h3)[name = tensor<string, []>(\"gt\")];\n"
                "        tensor<fp16, [1, %d, 1, %d]> out = conv(dilations = dl, groups = gr, pad = pd, "
                "pad_type = pt, strides = st, weight = W2, x = gate)[name = tensor<string, []>(\"c2\")];\n"
                "    } -> (out);\n}\n",
                DIM, SEQ,
                HIDDEN,DIM,HIDDEN,DIM, HIDDEN,DIM,HIDDEN,DIM, DIM,HIDDEN,DIM,HIDDEN,
                HIDDEN,SEQ, HIDDEN,SEQ, HIDDEN,SEQ, HIDDEN,SEQ, HIDDEN,SEQ, DIM,SEQ];

            NSDictionary *wd = @{
                @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":build_blob(W1,HIDDEN,DIM)},
                @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":build_blob(W3,HIDDEN,DIM)},
                @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":build_blob(W2,DIM,HIDDEN)},
            };
            Kern k = compile_mil(mil, wd);
            printf("  FFN: %s\n", k.model?"OK":"FAIL");
            if (k.model) {
                IOSurfaceRef ioIn = make_surface(DIM*SEQ*2), ioOut = make_surface(DIM*SEQ*2);
                IOSurfaceRef ins[]={ioIn}, outs[]={ioOut};
                for (int i=0;i<20;i++) ane_eval_io(&k,ins,1,outs,1);
                int N=500;
                uint64_t t0 = mach_absolute_time();
                for (int i=0;i<N;i++) ane_eval_io(&k,ins,1,outs,1);
                double ms = tb_ms(mach_absolute_time()-t0);
                double flops = 2.0*(2*HIDDEN*DIM + DIM*HIDDEN)*SEQ;
                printf("  %.3f ms/iter  %.1f GFLOPS (%.1f TFLOPS)\n", ms/N, N*flops/ms/1e6, N*flops/ms/1e9);
                CFRelease(ioIn); CFRelease(ioOut);
                cleanup_kern(&k);
            }
        }

        printf("\n=== Summary ===\n");
        printf("Full transformer layer = Attention + FFN\n");
        printf("2 ANE dispatches (+ CPU RMSNorm/residual) for entire forward pass\n");

        free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
        printf("\nDONE\n");
    }
    return 0;
}
