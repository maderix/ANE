// Test: Fused QKV projections in single MIL graph (3 convs → concat output)
// Input: x [1, DIM, 1, SEQ]
// Output: concat(Q, K, V) [1, DIM*3, 1, SEQ]
// 3 convs with separate weights, 1 ANE dispatch
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

#define DIM 768
#define SEQ 64

static int g_fp16_io = 0;  // M1/M2: cast op unsupported, use fp16 I/O directly

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
    int wsize = oc * ic * 2, total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0]=1; buf[4]=2; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
    *(uint32_t*)(buf+72)=wsize; *(uint32_t*)(buf+80)=128;
    _Float16 *fp16 = (_Float16*)(buf+128);
    for (int i = 0; i < oc*ic; i++) fp16[i] = (_Float16)w[i]; // layout A: row-major [oc, ic]
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
        printf("compile %s: %s\n", g_fp16_io ? "FAIL" : "failed (will retry)",
               e ? [[e localizedDescription] UTF8String] : "");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        return k;
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

// Fused QKV: 3 convs + concat in one MIL
static NSString *gen_fused_qkv_mil(void) {
    if (g_fp16_io) {
        return [NSString stringWithFormat:
            @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
            "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
            "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
            "        tensor<fp16, [%d, %d, 1, 1]> Wq = const()[name = tensor<string, []>(\"Wq\"), "
            "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wq.bin\"), offset = tensor<uint64, []>(64)))];\n"
            "        tensor<fp16, [%d, %d, 1, 1]> Wk = const()[name = tensor<string, []>(\"Wk\"), "
            "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wk.bin\"), offset = tensor<uint64, []>(64)))];\n"
            "        tensor<fp16, [%d, %d, 1, 1]> Wv = const()[name = tensor<string, []>(\"Wv\"), "
            "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wv.bin\"), offset = tensor<uint64, []>(64)))];\n"
            "        tensor<fp16, [1, %d, 1, %d]> q = conv(dilations = dl, groups = gr, pad = pd, "
            "pad_type = pt, strides = st, weight = Wq, x = x)[name = tensor<string, []>(\"cq\")];\n"
            "        tensor<fp16, [1, %d, 1, %d]> k = conv(dilations = dl, groups = gr, pad = pd, "
            "pad_type = pt, strides = st, weight = Wk, x = x)[name = tensor<string, []>(\"ck\")];\n"
            "        tensor<fp16, [1, %d, 1, %d]> v = conv(dilations = dl, groups = gr, pad = pd, "
            "pad_type = pt, strides = st, weight = Wv, x = x)[name = tensor<string, []>(\"cv\")];\n"
            "        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n"
            "        tensor<bool, []> inter = const()[name = tensor<string, []>(\"il\"), val = tensor<bool, []>(false)];\n"
            "        tensor<fp16, [1, %d, 1, %d]> y = concat(axis = ax, interleave = inter, values = (q, k, v))[name = tensor<string, []>(\"cat\")];\n"
            "    } -> (y);\n}\n",
            DIM, SEQ,
            DIM, DIM, DIM, DIM,
            DIM, DIM, DIM, DIM,
            DIM, DIM, DIM, DIM,
            DIM, SEQ, DIM, SEQ, DIM, SEQ,
            DIM*3, SEQ];
    }
    return [NSString stringWithFormat:
        @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
        "    func main<ios16>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        tensor<string, []> d1 = const()[name = tensor<string, []>(\"d1\"), val = tensor<string, []>(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = d1, x = x)[name = tensor<string, []>(\"cx\")];\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wq = const()[name = tensor<string, []>(\"Wq\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wq.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wk = const()[name = tensor<string, []>(\"Wk\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wk.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wv = const()[name = tensor<string, []>(\"Wv\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wv.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> q = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wq, x = x16)[name = tensor<string, []>(\"cq\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> k = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wk, x = x16)[name = tensor<string, []>(\"ck\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> v = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wv, x = x16)[name = tensor<string, []>(\"cv\")];\n"
        "        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n"
        "        tensor<bool, []> inter = const()[name = tensor<string, []>(\"il\"), val = tensor<bool, []>(false)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> qkv = concat(axis = ax, interleave = inter, values = (q, k, v))[name = tensor<string, []>(\"cat\")];\n"
        "        tensor<string, []> d2 = const()[name = tensor<string, []>(\"d2\"), val = tensor<string, []>(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = d2, x = qkv)[name = tensor<string, []>(\"co\")];\n"
        "    } -> (y);\n}\n",
        DIM, SEQ, DIM, SEQ,
        DIM, DIM, DIM, DIM,
        DIM, DIM, DIM, DIM,
        DIM, DIM, DIM, DIM,
        DIM, SEQ, DIM, SEQ, DIM, SEQ,
        DIM*3, SEQ, DIM*3, SEQ];
}

// Single conv MIL for comparison
static NSString *gen_single_mil(void) {
    if (g_fp16_io) {
        return [NSString stringWithFormat:
            @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
            "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
            "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), "
            "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w.bin\"), offset = tensor<uint64, []>(64)))];\n"
            "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
            "        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
            "pad_type = pt, strides = st, weight = W, x = x)[name = tensor<string, []>(\"cv\")];\n"
            "    } -> (y);\n}\n",
            DIM, SEQ, DIM, DIM, DIM, DIM, DIM, SEQ];
    }
    return [NSString stringWithFormat:
        @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
        "    func main<ios16>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        tensor<string, []> d1 = const()[name = tensor<string, []>(\"d1\"), val = tensor<string, []>(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = d1, x = x)[name = tensor<string, []>(\"cx\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/w.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = W, x = x16)[name = tensor<string, []>(\"cv\")];\n"
        "        tensor<string, []> d2 = const()[name = tensor<string, []>(\"d2\"), val = tensor<string, []>(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = d2, x = y16)[name = tensor<string, []>(\"co\")];\n"
        "    } -> (y);\n}\n",
        DIM, SEQ, DIM, SEQ, DIM, DIM, DIM, DIM, DIM, SEQ, DIM, SEQ];
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        printf("=== Fused QKV vs 3x Separate Convs ===\n");
        printf("DIM=%d SEQ=%d\n\n", DIM, SEQ);

        srand48(42);
        float *Wq = (float*)malloc(DIM*DIM*sizeof(float));
        float *Wk = (float*)malloc(DIM*DIM*sizeof(float));
        float *Wv = (float*)malloc(DIM*DIM*sizeof(float));
        float sc = 1.0f/sqrtf(DIM);
        for (int i = 0; i < DIM*DIM; i++) { Wq[i]=sc*(2*drand48()-1); Wk[i]=sc*(2*drand48()-1); Wv[i]=sc*(2*drand48()-1); }

        float *x = (float*)malloc(SEQ*DIM*sizeof(float));
        for (int i = 0; i < SEQ*DIM; i++) x[i] = 0.1f*(2*drand48()-1);

        // === Compile fused QKV ===
        retry_compile:;
        NSDictionary *fused_wd = @{
            @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":build_blob(Wq, DIM, DIM)},
            @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":build_blob(Wk, DIM, DIM)},
            @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":build_blob(Wv, DIM, DIM)},
        };
        Kern kFused = compile_mil(gen_fused_qkv_mil(), fused_wd);
        if (!kFused.model && !g_fp16_io) {
            printf("[ANE] fp32 compile failed, retrying with fp16 I/O (M1/M2 fallback)\n");
            g_fp16_io = 1;
            goto retry_compile;
        }
        printf("Fused QKV: %s\n", kFused.model ? "OK" : "FAIL");

        // === Compile 3 separate ===
        Kern kQ = compile_mil(gen_single_mil(), @{@"@model_path/weights/w.bin": @{@"offset":@0, @"data":build_blob(Wq, DIM, DIM)}});
        Kern kK = compile_mil(gen_single_mil(), @{@"@model_path/weights/w.bin": @{@"offset":@0, @"data":build_blob(Wk, DIM, DIM)}});
        Kern kV = compile_mil(gen_single_mil(), @{@"@model_path/weights/w.bin": @{@"offset":@0, @"data":build_blob(Wv, DIM, DIM)}});
        printf("Separate Q,K,V: %s %s %s\n", kQ.model?"OK":"FAIL", kK.model?"OK":"FAIL", kV.model?"OK":"FAIL");

        if (!kFused.model || !kQ.model) goto done;

        // IOSurfaces
        size_t bpe = g_fp16_io ? 2 : 4;
        size_t in_bytes = DIM*SEQ*bpe, out1_bytes = DIM*SEQ*bpe, out3_bytes = DIM*3*SEQ*bpe;
        IOSurfaceRef ioIn = make_surface(in_bytes);
        IOSurfaceRef ioFused = make_surface(out3_bytes);
        IOSurfaceRef ioQ = make_surface(out1_bytes), ioK = make_surface(out1_bytes), ioV = make_surface(out1_bytes);

        IOSurfaceLock(ioIn, 0, NULL);
        if (g_fp16_io) {
            _Float16 *dst = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
            for (int t = 0; t < SEQ; t++)
                for (int c = 0; c < DIM; c++)
                    dst[c*SEQ+t] = (_Float16)x[t*DIM+c];
        } else {
            float *dst = (float*)IOSurfaceGetBaseAddress(ioIn);
            for (int t = 0; t < SEQ; t++)
                for (int c = 0; c < DIM; c++)
                    dst[c*SEQ+t] = x[t*DIM+c];
        }
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Eval fused
        IOSurfaceRef ins[] = {ioIn};
        ane_eval(&kFused, ins, 1, ioFused);
        // Eval separate
        ane_eval(&kQ, ins, 1, ioQ);
        ane_eval(&kK, ins, 1, ioK);
        ane_eval(&kV, ins, 1, ioV);

        // Compare fused output (concat Q,K,V) vs separate
        IOSurfaceLock(ioFused, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(ioQ, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(ioK, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(ioV, kIOSurfaceLockReadOnly, NULL);
        float dq=0, dk=0, dv=0;
        if (g_fp16_io) {
            _Float16 *fo = (_Float16*)IOSurfaceGetBaseAddress(ioFused);
            _Float16 *qo = (_Float16*)IOSurfaceGetBaseAddress(ioQ);
            _Float16 *ko = (_Float16*)IOSurfaceGetBaseAddress(ioK);
            _Float16 *vo = (_Float16*)IOSurfaceGetBaseAddress(ioV);
            for (int c = 0; c < DIM; c++)
                for (int t = 0; t < SEQ; t++) {
                    float d1 = fabsf((float)fo[c*SEQ+t] - (float)qo[c*SEQ+t]); if(d1>dq) dq=d1;
                    float d2 = fabsf((float)fo[(DIM+c)*SEQ+t] - (float)ko[c*SEQ+t]); if(d2>dk) dk=d2;
                    float d3 = fabsf((float)fo[(DIM*2+c)*SEQ+t] - (float)vo[c*SEQ+t]); if(d3>dv) dv=d3;
                }
        } else {
            float *fo = (float*)IOSurfaceGetBaseAddress(ioFused);
            float *qo = (float*)IOSurfaceGetBaseAddress(ioQ);
            float *ko = (float*)IOSurfaceGetBaseAddress(ioK);
            float *vo = (float*)IOSurfaceGetBaseAddress(ioV);
            for (int c = 0; c < DIM; c++)
                for (int t = 0; t < SEQ; t++) {
                    float d1 = fabsf(fo[c*SEQ+t] - qo[c*SEQ+t]); if(d1>dq) dq=d1;
                    float d2 = fabsf(fo[(DIM+c)*SEQ+t] - ko[c*SEQ+t]); if(d2>dk) dk=d2;
                    float d3 = fabsf(fo[(DIM*2+c)*SEQ+t] - vo[c*SEQ+t]); if(d3>dv) dv=d3;
                }
        }
        IOSurfaceUnlock(ioFused, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceUnlock(ioQ, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceUnlock(ioK, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceUnlock(ioV, kIOSurfaceLockReadOnly, NULL);
        printf("\nFused vs Separate: dQ=%.6f dK=%.6f dV=%.6f → %s\n",
               dq, dk, dv, (dq<0.001f && dk<0.001f && dv<0.001f) ? "PASS" : "FAIL");

        // === Benchmark ===
        printf("\n=== Benchmark ===\n");
        int N = 500;
        // Warmup
        for (int i = 0; i < 20; i++) { ane_eval(&kFused, ins, 1, ioFused); ane_eval(&kQ, ins, 1, ioQ); }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) ane_eval(&kFused, ins, 1, ioFused);
        double ms_fused = tb_ms(mach_absolute_time() - t0);

        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ane_eval(&kQ, ins, 1, ioQ);
            ane_eval(&kK, ins, 1, ioK);
            ane_eval(&kV, ins, 1, ioV);
        }
        double ms_sep = tb_ms(mach_absolute_time() - t0);

        double flops_one = 2.0 * DIM * DIM * SEQ;
        printf("Fused QKV (1 dispatch, 3 convs):  %.3f ms/iter  %.1f GFLOPS\n",
               ms_fused/N, N*3*flops_one/ms_fused/1e6);
        printf("Separate Q+K+V (3 dispatches):    %.3f ms/iter  %.1f GFLOPS\n",
               ms_sep/N, N*3*flops_one/ms_sep/1e6);
        printf("Speedup: %.2fx\n", ms_sep/ms_fused);

        CFRelease(ioIn); CFRelease(ioFused); CFRelease(ioQ); CFRelease(ioK); CFRelease(ioV);
        free(Wq); free(Wk); free(Wv); free(x);
        done:
        cleanup_kern(&kFused); cleanup_kern(&kQ); cleanup_kern(&kK); cleanup_kern(&kV);
        printf("\nDONE\n");
    }
    return 0;
}
