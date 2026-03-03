// tiny_train.m — Train a 2-layer linear model on ANE (forward AND backward)
// y = W2 @ relu(W1 @ x), MSE loss, SGD update
// Forward: ANE conv with baked weights
// Backward dx: ANE conv with transposed baked weights
// Backward dW: CPU (outer product, memory-bound)
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

static Class g_D, g_I, g_AR, g_AIO;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

static NSData *build_blob(const float *w, int rows, int cols) {
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf+72) = wsize;
    *(uint32_t*)(buf+80) = 128;
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (int i = 0; i < rows * cols; i++) fp16[i] = (_Float16)w[i];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Build blob with TRANSPOSED weights: W[rows,cols] → W^T[cols,rows]
static NSData *build_blob_transposed(const float *w, int rows, int cols) {
    int wsize = cols * rows * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf+72) = wsize;
    *(uint32_t*)(buf+80) = 128;
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fp16[j * rows + i] = (_Float16)w[i * cols + j]; // transpose
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static int g_fp16_io = 0;  // M1/M2: cast op unsupported, use fp16 I/O directly

static NSString *gen_conv_mil(int in_ch, int out_ch, int sp) {
    if (g_fp16_io) {
        return [NSString stringWithFormat:
            @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
            "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
            "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), "
            "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>(64)))];\n"
            "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
            "        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
            "pad_type = pt, strides = st, weight = W, x = x)[name = tensor<string, []>(\"cv\")];\n"
            "    } -> (y);\n}\n",
            in_ch, sp, out_ch, in_ch, out_ch, in_ch, out_ch, sp];
    }
    return [NSString stringWithFormat:
        @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
        "    func main<ios16>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        tensor<string, []> d1 = const()[name = tensor<string, []>(\"d1\"), val = tensor<string, []>(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = d1, x = x)[name = tensor<string, []>(\"cx\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>(64)))];\n"
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
        in_ch, sp, in_ch, sp, out_ch, in_ch, out_ch, in_ch, out_ch, sp, out_ch, sp];
}

typedef struct {
    void *model;    // CFBridgingRetain'd _ANEInMemoryModel
    IOSurfaceRef ioIn, ioOut;
    void *request;  // CFBridgingRetain'd _ANERequest
    void *tmpDir;   // CFBridgingRetain'd NSString
} Kern;

static Kern *compile_kern_with_blob(NSData *blob, int in_ch, int out_ch, int sp) {
    NSString *mil = gen_conv_mil(in_ch, out_ch, sp);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSDictionary *wd = @{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":blob}};
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), milData, wd, nil);
    if (!desc) return NULL;
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [blob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        if (!g_fp16_io) {
            printf("[ANE] fp32 compile failed, retrying with fp16 I/O (M1/M2 fallback)\n");
            g_fp16_io = 1;
            return compile_kern_with_blob(blob, in_ch, out_ch, sp);
        }
        return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) return NULL;
    size_t bpe = g_fp16_io ? 2 : 4;
    size_t inB = in_ch * sp * bpe, outB = out_ch * sp * bpe;
    IOSurfaceRef ioI = make_surface(inB), ioO = make_surface(outB);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioI);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioO);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    Kern *k = calloc(1, sizeof(Kern));
    k->model = (void*)CFBridgingRetain(mdl);
    k->ioIn = ioI; k->ioOut = ioO;
    k->request = (void*)CFBridgingRetain(req);
    k->tmpDir = (void*)CFBridgingRetain(td);
    return k;
}

static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    NSString *td = (__bridge id)k->tmpDir;
    [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
    CFRelease(k->model); CFRelease(k->request); CFRelease(k->tmpDir);
    free(k);
}

// ANE eval: input [S, in_ch] row-major ↔ [in_ch, S] channels-first
static void ane_eval(Kern *k, const float *in, float *out, int in_ch, int out_ch, int sp) {
    IOSurfaceLock(k->ioIn, 0, NULL);
    void *base_in = IOSurfaceGetBaseAddress(k->ioIn);
    if (g_fp16_io) {
        _Float16 *dst = (_Float16*)base_in;
        for (int t = 0; t < sp; t++)
            for (int c = 0; c < in_ch; c++)
                dst[c*sp + t] = (_Float16)in[t*in_ch + c];
    } else {
        float *dst = (float*)base_in;
        for (int t = 0; t < sp; t++)
            for (int c = 0; c < in_ch; c++)
                dst[c*sp + t] = in[t*in_ch + c];
    }
    IOSurfaceUnlock(k->ioIn, 0, NULL);
    NSError *e = nil;
    id mdl = (__bridge id)k->model;
    id req = (__bridge id)k->request;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    void *base_out = IOSurfaceGetBaseAddress(k->ioOut);
    if (g_fp16_io) {
        _Float16 *src = (_Float16*)base_out;
        for (int t = 0; t < sp; t++)
            for (int c = 0; c < out_ch; c++)
                out[t*out_ch + c] = (float)src[c*sp + t];
    } else {
        float *src = (float*)base_out;
        for (int t = 0; t < sp; t++)
            for (int c = 0; c < out_ch; c++)
                out[t*out_ch + c] = src[c*sp + t];
    }
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        ane_init();
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);

        int D = 64, H = 128, S = 16;
        int steps = 25; // 4 kernels × 25 = 100 compiles, under 119 limit
        float lr = 0.5f;
        int recompile_every = 1; // recompile every step for correct gradients

        float *W1 = (float*)malloc(H * D * sizeof(float));
        float *W2 = (float*)malloc(D * H * sizeof(float));
        for (int i = 0; i < H*D; i++) W1[i] = 0.01f * sinf(i * 1.3f + 0.7f);
        for (int i = 0; i < D*H; i++) W2[i] = 0.01f * cosf(i * 0.9f + 1.1f);

        float *x = (float*)calloc(S * D, sizeof(float));
        float *y_target = (float*)calloc(S * D, sizeof(float));
        for (int t = 0; t < S; t++)
            for (int i = 0; i < D; i++) {
                float v = sinf((t * D + i) * 0.1f);
                x[t*D + i] = v;
                y_target[t*D + i] = v;
            }

        printf("=== Tiny 2-Layer ANE Training (Forward + Backward on ANE) ===\n");
        printf("x:[%d,%d] → W1:[%d,%d] → ReLU → W2:[%d,%d] → y:[%d,%d]\n", S,D, H,D, D,H, S,D);
        printf("Forward: ANE conv | Backward dx: ANE conv(W^T) | Backward dW: CPU\n");
        printf("Steps: %d, LR: %.4f, Recompile every %d steps\n\n", steps, lr, recompile_every);

        float *h = (float*)malloc(S * H * sizeof(float));
        float *h_relu = (float*)malloc(S * H * sizeof(float));
        float *y = (float*)malloc(S * D * sizeof(float));
        float *dy = (float*)malloc(S * D * sizeof(float));
        float *dh_relu = (float*)malloc(S * H * sizeof(float));
        float *dh = (float*)malloc(S * H * sizeof(float));
        float *dx_layer = (float*)malloc(S * D * sizeof(float)); // not used for update but proves backward works
        float *dW1 = (float*)calloc(H * D, sizeof(float));
        float *dW2 = (float*)calloc(D * H, sizeof(float));

        // 4 ANE kernels: 2 forward + 2 backward (transposed weights)
        Kern *k1_fwd = NULL, *k2_fwd = NULL;  // W1: [H,D]→conv(D→H), W2: [D,H]→conv(H→D)
        Kern *k1_bwd = NULL, *k2_bwd = NULL;  // W1^T: [D,H]→conv(H→D), W2^T: [H,D]→conv(D→H)
        bool on_ane = true;

        printf("%-6s %-12s %-10s %-6s\n", "Step", "MSE Loss", "ms/step", "Backend");
        printf("--------------------------------------\n");

        for (int step = 0; step < steps; step++) {
            uint64_t t0 = mach_absolute_time();

            if (on_ane && step % recompile_every == 0) {
                free_kern(k1_fwd); free_kern(k2_fwd);
                free_kern(k1_bwd); free_kern(k2_bwd);
                k1_fwd = k2_fwd = k1_bwd = k2_bwd = NULL;
                @autoreleasepool {
                    k1_fwd = compile_kern_with_blob(build_blob(W1, H, D), D, H, S);
                    k2_fwd = compile_kern_with_blob(build_blob(W2, D, H), H, D, S);
                    // Backward: dx = W^T @ dy → conv with transposed weight
                    // W2^T: [H,D] as conv weight, input dy [1,D,1,S] → output dh [1,H,1,S]
                    k2_bwd = compile_kern_with_blob(build_blob_transposed(W2, D, H), D, H, S);
                    // W1^T: [D,H] as conv weight, input dh [1,H,1,S] → output dx [1,D,1,S]
                    k1_bwd = compile_kern_with_blob(build_blob_transposed(W1, H, D), H, D, S);
                }
                if (!k1_fwd || !k2_fwd || !k1_bwd || !k2_bwd) {
                    printf("ANE limit at step %d, continuing on CPU\n", step);
                    free_kern(k1_fwd); free_kern(k2_fwd);
                    free_kern(k1_bwd); free_kern(k2_bwd);
                    k1_fwd = k2_fwd = k1_bwd = k2_bwd = NULL;
                    on_ane = false;
                }
            }

            if (on_ane) {
                // === Forward on ANE ===
                ane_eval(k1_fwd, x, h, D, H, S);
                for (int i = 0; i < S*H; i++) h_relu[i] = h[i] > 0 ? h[i] : 0;
                ane_eval(k2_fwd, h_relu, y, H, D, S);
            } else {
                for (int t = 0; t < S; t++)
                    for (int i = 0; i < H; i++) {
                        float s = 0; for (int j = 0; j < D; j++) s += W1[i*D+j] * x[t*D+j];
                        h[t*H+i] = s;
                    }
                for (int i = 0; i < S*H; i++) h_relu[i] = h[i] > 0 ? h[i] : 0;
                for (int t = 0; t < S; t++)
                    for (int i = 0; i < D; i++) {
                        float s = 0; for (int j = 0; j < H; j++) s += W2[i*H+j] * h_relu[t*H+j];
                        y[t*D+i] = s;
                    }
            }

            // MSE loss + dL/dy
            float loss = 0;
            for (int i = 0; i < S*D; i++) {
                float diff = y[i] - y_target[i];
                loss += diff * diff;
                dy[i] = 2.0f * diff / (S * D);
            }
            loss /= (S * D);

            if (on_ane) {
                // === Backward dx on ANE ===
                // dh_relu = W2^T @ dy (ANE conv with transposed W2)
                ane_eval(k2_bwd, dy, dh_relu, D, H, S);
                // ReLU backward (CPU, element-wise)
                for (int i = 0; i < S*H; i++) dh[i] = h[i] > 0 ? dh_relu[i] : 0;
                // dx = W1^T @ dh (ANE conv with transposed W1)
                ane_eval(k1_bwd, dh, dx_layer, H, D, S);
            } else {
                memset(dh_relu, 0, S * H * sizeof(float));
                for (int t = 0; t < S; t++)
                    for (int j = 0; j < H; j++)
                        for (int i = 0; i < D; i++)
                            dh_relu[t*H + j] += W2[i*H + j] * dy[t*D + i];
                for (int i = 0; i < S*H; i++) dh[i] = h[i] > 0 ? dh_relu[i] : 0;
            }

            // dW on CPU (outer products — memory-bound, not worth ANE)
            memset(dW2, 0, D * H * sizeof(float));
            for (int t = 0; t < S; t++)
                for (int i = 0; i < D; i++)
                    for (int j = 0; j < H; j++)
                        dW2[i*H + j] += dy[t*D + i] * h_relu[t*H + j];
            memset(dW1, 0, H * D * sizeof(float));
            for (int t = 0; t < S; t++)
                for (int i = 0; i < H; i++)
                    for (int j = 0; j < D; j++)
                        dW1[i*D + j] += dh[t*H + i] * x[t*D + j];

            // SGD
            for (int i = 0; i < H*D; i++) W1[i] -= lr * dW1[i];
            for (int i = 0; i < D*H; i++) W2[i] -= lr * dW2[i];

            double ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;

            if (step % 1 == 0 || step == steps - 1)
                printf("%-6d %-12.6f %-10.1f %-6s\n", step, loss, ms, on_ane ? "ANE" : "CPU");

            if (loss < 1e-6f) { printf("\nConverged at step %d!\n", step); break; }
        }

        printf("\nFinal output vs target (first 8):\n");
        if (on_ane && k1_fwd && k2_fwd) {
            ane_eval(k1_fwd, x, h, D, H, S);
            for (int i = 0; i < S*H; i++) h_relu[i] = h[i] > 0 ? h[i] : 0;
            ane_eval(k2_fwd, h_relu, y, H, D, S);
        }
        printf("  y:      "); for (int i = 0; i < 8; i++) printf("%.4f ", y[i]); printf("\n");
        printf("  target: "); for (int i = 0; i < 8; i++) printf("%.4f ", y_target[i]); printf("\n");

        free_kern(k1_fwd); free_kern(k2_fwd); free_kern(k1_bwd); free_kern(k2_bwd);
        free(W1); free(W2); free(x); free(y_target);
        free(h); free(h_relu); free(y); free(dy); free(dh_relu); free(dh); free(dx_layer); free(dW1); free(dW2);
        printf("\nDone.\n");
    }
    return 0;
}
