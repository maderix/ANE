// tiny_train.m — Train a 2-layer linear model on ANE (forward AND backward)
// y = W2 @ relu(W1 @ x), MSE loss, SGD update
// Pipeline: compile next kernels on background thread while ANE runs current batch
// Bypasses ANE 119-compile limit via exec() self-restart
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>
#include <unistd.h>
#include <dispatch/dispatch.h>

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
            fp16[j * rows + i] = (_Float16)w[i * cols + j];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSString *gen_conv_mil(int in_ch, int out_ch, int sp) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string d1 = const()[name = string(\"d1\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = d1, x = x)[name = string(\"cx\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = W, x = x16)[name = string(\"cv\")];\n"
        "        string d2 = const()[name = string(\"d2\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = d2, x = y16)[name = string(\"co\")];\n"
        "    } -> (y);\n}\n",
        in_ch, sp, in_ch, sp, out_ch, in_ch, out_ch, in_ch, out_ch, sp, out_ch, sp];
}

typedef struct {
    void *model;    // CFBridgingRetain'd _ANEInMemoryModel
    IOSurfaceRef ioIn, ioOut;
    void *request;  // CFBridgingRetain'd _ANERequest
    void *tmpDir;   // CFBridgingRetain'd NSString
} Kern;

static int g_compile_count = 0;

static Kern *compile_kern_with_blob(NSData *blob, int in_ch, int out_ch, int sp) {
    @autoreleasepool {
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
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) return NULL;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) return NULL;
    __sync_fetch_and_add(&g_compile_count, 1);
    size_t inB = in_ch * sp * 4, outB = out_ch * sp * 4;
    IOSurfaceRef ioI = make_surface(inB), ioO = make_surface(outB);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioI);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioO);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    Kern *k = calloc(1, sizeof(Kern));
    k->model = CFBridgingRetain(mdl);
    k->ioIn = ioI; k->ioOut = ioO;
    k->request = CFBridgingRetain(req);
    k->tmpDir = CFBridgingRetain(td);
    return k;
    }
}

static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    NSString *td = (__bridge id)k->tmpDir;
    [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
    CFRelease(k->model);
    CFRelease(k->request);
    CFRelease(k->tmpDir);
    free(k);
}

static bool ane_eval_k(Kern *k, const float *in, float *out, int in_ch, int out_ch, int sp) {
    float *tmp = (float*)malloc(in_ch * sp * sizeof(float));
    for (int t = 0; t < sp; t++)
        for (int c = 0; c < in_ch; c++)
            tmp[c*sp + t] = in[t*in_ch + c];
    IOSurfaceLock(k->ioIn, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioIn), tmp, in_ch * sp * sizeof(float));
    IOSurfaceUnlock(k->ioIn, 0, NULL);
    free(tmp);
    NSError *e = nil;
    id mdl = (__bridge id)k->model;
    id req = (__bridge id)k->request;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok) {
        fprintf(stderr, "ANE eval failed: %s\n",
                e ? [[e description] UTF8String] : "unknown error");
        return false;
    }
    float *tmp2 = (float*)malloc(out_ch * sp * sizeof(float));
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    memcpy(tmp2, IOSurfaceGetBaseAddress(k->ioOut), out_ch * sp * sizeof(float));
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    for (int t = 0; t < sp; t++)
        for (int c = 0; c < out_ch; c++)
            out[t*out_ch + c] = tmp2[c*sp + t];
    free(tmp2);
    return true;
}

// === Checkpoint: save/restore training state for exec() restart ===
#define CKPT_PATH "/tmp/ane_train_ckpt.bin"

typedef struct {
    int step;
    float loss;
    int D, H, S, total_steps;
    float lr;
    double cum_compile_ms, cum_train_ms, cum_wall_ms;
    int cum_steps, cum_batches;
} CkptHeader;

static void save_checkpoint(const char *path, int step, float loss,
                            int D, int H, int S, int total_steps, float lr,
                            const float *W1, const float *W2,
                            double cc, double ct, double cw, int cs, int cb) {
    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    FILE *f = fopen(tmp_path, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s for checkpoint\n", tmp_path); return; }
    CkptHeader hdr = {step, loss, D, H, S, total_steps, lr, cc, ct, cw, cs, cb};
    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(W1, sizeof(float), H * D, f);
    fwrite(W2, sizeof(float), D * H, f);
    fclose(f);
    rename(tmp_path, path);  // atomic on POSIX
}

static bool load_checkpoint(const char *path, CkptHeader *hdr,
                            float *W1, float *W2, int H, int D) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    if (fread(hdr, sizeof(CkptHeader), 1, f) != 1) { fclose(f); return false; }
    if (fread(W1, sizeof(float), H * D, f) != (size_t)(H * D)) { fclose(f); return false; }
    if (fread(W2, sizeof(float), D * H, f) != (size_t)(D * H)) { fclose(f); return false; }
    fclose(f);
    return true;
}

#define MAX_COMPILES 100
#define KERNELS_PER_STEP 4
#define ACCUM_STEPS 10

// === Pipeline: background compile via GCD ===
typedef struct {
    Kern *k1_fwd, *k2_fwd, *k1_bwd, *k2_bwd;
    float *W1, *W2;
    int D, H, S;
    bool ok;
    double compile_ms;
} PipelineCompile;

static double tb_to_ms(uint64_t elapsed, mach_timebase_info_data_t tb) {
    return (double)elapsed * tb.numer / tb.denom / 1e6;
}

static mach_timebase_info_data_t g_tb;
// Serial queue ensures ANE compiles don't overlap with each other
static dispatch_queue_t g_compile_queue;

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);
        g_compile_queue = dispatch_queue_create("ane.compile", DISPATCH_QUEUE_SERIAL);

        int D = 64, H = 128, S = 16;
        int total_steps = 2000;
        float lr = 1.0f;
        int start_step = 0;
        bool resuming = false;

        float *W1 = (float*)malloc(H * D * sizeof(float));
        float *W2 = (float*)malloc(D * H * sizeof(float));

        if (argc > 1 && strcmp(argv[1], "--resume") == 0) {
            CkptHeader hdr;
            if (load_checkpoint(CKPT_PATH, &hdr, W1, W2, H, D)) {
                start_step = hdr.step;
                total_steps = hdr.total_steps;
                lr = hdr.lr;
                resuming = true;
                printf("[RESUMED at step %d, loss=%.6f, compiles reset]\n", start_step, hdr.loss);
            }
        }

        // Cumulative stats (restored from checkpoint if resuming)
        double cum_compile_ms = 0, cum_train_ms = 0, cum_wall_ms = 0;
        int cum_steps = 0, cum_batches = 0;
        if (resuming) {
            CkptHeader hdr2;
            FILE *f = fopen(CKPT_PATH, "rb");
            if (f) { fread(&hdr2, sizeof(hdr2), 1, f); fclose(f);
                cum_compile_ms = hdr2.cum_compile_ms;
                cum_train_ms = hdr2.cum_train_ms;
                cum_wall_ms = hdr2.cum_wall_ms;
                cum_steps = hdr2.cum_steps;
                cum_batches = hdr2.cum_batches;
            }
        }

        // FLOPs calculation
        // Forward: W1[H,D] @ x[D,S] = 2*H*D*S, W2[D,H] @ h[H,S] = 2*D*H*S → total fwd = 4*D*H*S
        // Backward dx: W2^T[H,D] @ dy[D,S] = 2*H*D*S, W1^T[D,H] @ dh[H,S] = 2*D*H*S → total bwd = 4*D*H*S
        // dW (CPU): dW2[D,H] = dy[D,S] @ h^T[S,H] = 2*D*S*H, dW1 same → total dW = 4*D*H*S
        // ANE FLOPs per step = 8*D*H*S (fwd + bwd on ANE)
        // CPU FLOPs per step = 4*D*H*S (dW accumulation)
        // Total FLOPs per step = 12*D*H*S
        double ane_flops_per_step = 8.0 * D * H * S;
        double cpu_flops_per_step = 4.0 * D * H * S;
        double total_flops_per_step = ane_flops_per_step + cpu_flops_per_step;
        double weight_bytes = (H*D + D*H) * 2.0; // FP16 weights on ANE

        if (!resuming) {
            for (int i = 0; i < H*D; i++) W1[i] = 0.01f * sinf(i * 1.3f + 0.7f);
            for (int i = 0; i < D*H; i++) W2[i] = 0.01f * cosf(i * 0.9f + 1.1f);
            printf("=== ANE Training: Pipeline Parallel + Grad Accumulation ===\n");
            printf("x:[%d,%d] -> W1:[%d,%d] -> ReLU -> W2:[%d,%d] -> y:[%d,%d]\n", S,D, H,D, D,H, S,D);
            printf("Accum %d steps per recompile | Pipeline: compile overlaps ANE eval\n", ACCUM_STEPS);
            printf("ANE FP16 peak: 15.8 TFLOPS (M4) | Weights: %.1f KB\n\n", weight_bytes/1024.0);
            printf("FLOPs/step: ANE=%.0f (fwd+bwd)  CPU=%.0f (dW)  Total=%.0f\n",
                   ane_flops_per_step, cpu_flops_per_step, total_flops_per_step);
            printf("Steps: %d, LR: %.4f, exec() budget: %d compiles\n\n",
                   total_steps, lr, MAX_COMPILES);
        }

        float *x = (float*)calloc(S * D, sizeof(float));
        float *y_target = (float*)calloc(S * D, sizeof(float));
        for (int t = 0; t < S; t++)
            for (int i = 0; i < D; i++) {
                float v = sinf((t * D + i) * 0.1f);
                x[t*D + i] = v;
                y_target[t*D + i] = v;
            }

        float *h = (float*)malloc(S * H * sizeof(float));
        float *h_relu = (float*)malloc(S * H * sizeof(float));
        float *y = (float*)malloc(S * D * sizeof(float));
        float *dy = (float*)malloc(S * D * sizeof(float));
        float *dh_relu = (float*)malloc(S * H * sizeof(float));
        float *dh = (float*)malloc(S * H * sizeof(float));
        float *dx_layer = (float*)malloc(S * D * sizeof(float));

        Kern *k1_fwd = NULL, *k2_fwd = NULL;
        Kern *k1_bwd = NULL, *k2_bwd = NULL;
        float last_loss = 999.0f;

        // Stats
        double total_compile_ms = 0, total_train_ms = 0, total_wall_ms = 0;
        double total_hidden_compile_ms = 0; // compile time hidden by pipeline
        int total_batches = 0;
        int total_steps_done = 0;
        uint64_t t_wall_start = mach_absolute_time();

        // First compile is synchronous (no pipeline yet)
        {
            uint64_t t0 = mach_absolute_time();
            k1_fwd = compile_kern_with_blob(build_blob(W1, H, D), D, H, S);
            k2_fwd = compile_kern_with_blob(build_blob(W2, D, H), H, D, S);
            k2_bwd = compile_kern_with_blob(build_blob_transposed(W2, D, H), D, H, S);
            k1_bwd = compile_kern_with_blob(build_blob_transposed(W1, H, D), H, D, S);
            double cms = tb_to_ms(mach_absolute_time() - t0, g_tb);
            total_compile_ms += cms;
            if (!k1_fwd || !k2_fwd || !k1_bwd || !k2_bwd) {
                printf("Initial compile failed!\n"); return 1;
            }
            printf("Initial compile: %.0fms\n", cms);
        }

        int step = start_step;
        while (step < total_steps) {
            // Check compile budget
            if (g_compile_count + KERNELS_PER_STEP > MAX_COMPILES) {
                free_kern(k1_fwd); free_kern(k2_fwd);
                free_kern(k1_bwd); free_kern(k2_bwd);
                save_checkpoint(CKPT_PATH, step, last_loss, D, H, S, total_steps, lr, W1, W2,
                                    cum_compile_ms + total_compile_ms, cum_train_ms + total_train_ms,
                                    cum_wall_ms + tb_to_ms(mach_absolute_time() - t_wall_start, g_tb),
                                    cum_steps + total_steps_done, cum_batches + total_batches);
                double wall = tb_to_ms(mach_absolute_time() - t_wall_start, g_tb);
                printf("[exec() restart at step %d, %d compiles, loss=%.6f, wall=%.0fms]\n",
                       step, g_compile_count, last_loss, wall);
                fflush(stdout);
                execl(argv[0], argv[0], "--resume", NULL);
                perror("execl failed"); return 1;
            }

            // === Run ACCUM_STEPS with current kernels ===
            float *aW1 = (float*)calloc(H * D, sizeof(float));
            float *aW2 = (float*)calloc(D * H, sizeof(float));
            int steps_this_batch = 0;

            // Pipeline: start compiling NEXT batch's kernels in background
            // We'll apply gradients first, then launch compile with updated W
            // But for pipeline, we compile AHEAD: while running batch N, compile for N+1
            // So we need to update weights BEFORE launching background compile

            uint64_t t_batch = mach_absolute_time();
            for (int a = 0; a < ACCUM_STEPS && step < total_steps; a++, step++) {
                ane_eval_k(k1_fwd, x, h, D, H, S);
                for (int i = 0; i < S*H; i++) h_relu[i] = h[i] > 0 ? h[i] : 0;
                ane_eval_k(k2_fwd, h_relu, y, H, D, S);

                float loss = 0;
                for (int i = 0; i < S*D; i++) {
                    float diff = y[i] - y_target[i];
                    loss += diff * diff;
                    dy[i] = 2.0f * diff / (S * D);
                }
                loss /= (S * D);
                last_loss = loss;

                ane_eval_k(k2_bwd, dy, dh_relu, D, H, S);
                for (int i = 0; i < S*H; i++) dh[i] = h[i] > 0 ? dh_relu[i] : 0;
                ane_eval_k(k1_bwd, dh, dx_layer, H, D, S);

                for (int t = 0; t < S; t++)
                    for (int i = 0; i < D; i++)
                        for (int j = 0; j < H; j++)
                            aW2[i*H + j] += dy[t*D + i] * h_relu[t*H + j];
                for (int t = 0; t < S; t++)
                    for (int i = 0; i < H; i++)
                        for (int j = 0; j < D; j++)
                            aW1[i*D + j] += dh[t*H + i] * x[t*D + j];

                steps_this_batch++;
            }
            double batch_ms = tb_to_ms(mach_absolute_time() - t_batch, g_tb);
            total_train_ms += batch_ms;

            // Apply accumulated gradients
            float scale = 1.0f / steps_this_batch;
            for (int i = 0; i < H*D; i++) W1[i] -= lr * aW1[i] * scale;
            for (int i = 0; i < D*H; i++) W2[i] -= lr * aW2[i] * scale;
            free(aW1); free(aW2);

            total_steps_done += steps_this_batch;
            total_batches++;

            // Print progress
            double step_ms = batch_ms / steps_this_batch;
            double ane_gflops = (ane_flops_per_step * steps_this_batch) / (batch_ms * 1e6);
            double total_gflops = (total_flops_per_step * steps_this_batch) / (batch_ms * 1e6);

            if (total_batches % 5 == 1 || total_batches <= 2 || step >= total_steps) {
                printf("step %-5d loss=%-10.6f  %5.1fms/step  ANE=%.2f GFLOPS  total=%.2f GFLOPS  compiles=%d\n",
                       step - steps_this_batch, last_loss, step_ms, ane_gflops, total_gflops, g_compile_count);
            }

            // Pipeline: launch background compile with updated weights,
            // then immediately start NEXT batch's ANE evals with OLD kernels
            // while compile runs concurrently on GCD queue
            bool can_pipeline = (step < total_steps) && (g_compile_count + KERNELS_PER_STEP <= MAX_COMPILES);

            if (can_pipeline) {
                // Snapshot weights for background compile
                PipelineCompile *pc = calloc(1, sizeof(PipelineCompile));
                pc->W1 = (float*)malloc(H * D * sizeof(float));
                pc->W2 = (float*)malloc(D * H * sizeof(float));
                memcpy(pc->W1, W1, H * D * sizeof(float));
                memcpy(pc->W2, W2, D * H * sizeof(float));
                pc->D = D; pc->H = H; pc->S = S;

                dispatch_semaphore_t sem = dispatch_semaphore_create(0);

                dispatch_async(g_compile_queue, ^{
                    @autoreleasepool {
                        uint64_t t0 = mach_absolute_time();
                        pc->k1_fwd = compile_kern_with_blob(build_blob(pc->W1, pc->H, pc->D), pc->D, pc->H, pc->S);
                        pc->k2_fwd = compile_kern_with_blob(build_blob(pc->W2, pc->D, pc->H), pc->H, pc->D, pc->S);
                        pc->k2_bwd = compile_kern_with_blob(build_blob_transposed(pc->W2, pc->D, pc->H), pc->D, pc->H, pc->S);
                        pc->k1_bwd = compile_kern_with_blob(build_blob_transposed(pc->W1, pc->H, pc->D), pc->H, pc->D, pc->S);
                        pc->compile_ms = tb_to_ms(mach_absolute_time() - t0, g_tb);
                        pc->ok = pc->k1_fwd && pc->k2_fwd && pc->k1_bwd && pc->k2_bwd;
                        dispatch_semaphore_signal(sem);
                    }
                });

                // === While compile runs in background, do ANOTHER batch with OLD kernels ===
                if (step < total_steps && k1_fwd && k2_fwd && k1_bwd && k2_bwd) {
                    float *aW1b = (float*)calloc(H * D, sizeof(float));
                    float *aW2b = (float*)calloc(D * H, sizeof(float));
                    int steps_overlap = 0;
                    uint64_t t_overlap = mach_absolute_time();

                    for (int a = 0; a < ACCUM_STEPS && step < total_steps; a++, step++) {
                        ane_eval_k(k1_fwd, x, h, D, H, S);
                        for (int i = 0; i < S*H; i++) h_relu[i] = h[i] > 0 ? h[i] : 0;
                        ane_eval_k(k2_fwd, h_relu, y, H, D, S);

                        float loss = 0;
                        for (int i = 0; i < S*D; i++) {
                            float diff = y[i] - y_target[i];
                            loss += diff * diff;
                            dy[i] = 2.0f * diff / (S * D);
                        }
                        loss /= (S * D);
                        last_loss = loss;

                        ane_eval_k(k2_bwd, dy, dh_relu, D, H, S);
                        for (int i = 0; i < S*H; i++) dh[i] = h[i] > 0 ? dh_relu[i] : 0;
                        ane_eval_k(k1_bwd, dh, dx_layer, H, D, S);

                        for (int t = 0; t < S; t++)
                            for (int i = 0; i < D; i++)
                                for (int j = 0; j < H; j++)
                                    aW2b[i*H + j] += dy[t*D + i] * h_relu[t*H + j];
                        for (int t = 0; t < S; t++)
                            for (int i = 0; i < H; i++)
                                for (int j = 0; j < D; j++)
                                    aW1b[i*D + j] += dh[t*H + i] * x[t*D + j];
                        steps_overlap++;
                    }
                    double overlap_ms = tb_to_ms(mach_absolute_time() - t_overlap, g_tb);
                    total_train_ms += overlap_ms;
                    total_steps_done += steps_overlap;
                    total_batches++;

                    // Apply these gradients with reduced LR (stale weights — 1 batch behind)
                    float sc = 0.5f / steps_overlap; // half LR for stale batch
                    for (int i = 0; i < H*D; i++) W1[i] -= lr * aW1b[i] * sc;
                    for (int i = 0; i < D*H; i++) W2[i] -= lr * aW2b[i] * sc;
                    free(aW1b); free(aW2b);

                    if (total_batches % 5 == 1) {
                        double sm = overlap_ms / steps_overlap;
                        printf("step %-5d loss=%-10.6f  %5.1fms/step  (overlapped with compile)  compiles=%d\n",
                               step - steps_overlap, last_loss, sm, g_compile_count);
                    }
                }

                // Wait for compile to finish
                dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
                total_compile_ms += pc->compile_ms;
                total_hidden_compile_ms += pc->compile_ms; // all hidden behind train

                free_kern(k1_fwd); free_kern(k2_fwd);
                free_kern(k1_bwd); free_kern(k2_bwd);

                if (pc->ok) {
                    k1_fwd = pc->k1_fwd; k2_fwd = pc->k2_fwd;
                    k1_bwd = pc->k1_bwd; k2_bwd = pc->k2_bwd;
                } else {
                    k1_fwd = k2_fwd = k1_bwd = k2_bwd = NULL;
                }
                free(pc->W1); free(pc->W2); free(pc);
            } else if (step < total_steps) {
                // Synchronous compile (no budget for pipeline)
                uint64_t t0 = mach_absolute_time();
                free_kern(k1_fwd); free_kern(k2_fwd);
                free_kern(k1_bwd); free_kern(k2_bwd);
                k1_fwd = compile_kern_with_blob(build_blob(W1, H, D), D, H, S);
                k2_fwd = compile_kern_with_blob(build_blob(W2, D, H), H, D, S);
                k2_bwd = compile_kern_with_blob(build_blob_transposed(W2, D, H), D, H, S);
                k1_bwd = compile_kern_with_blob(build_blob_transposed(W1, H, D), H, D, S);
                double cms = tb_to_ms(mach_absolute_time() - t0, g_tb);
                total_compile_ms += cms;
                if (!k1_fwd || !k2_fwd || !k1_bwd || !k2_bwd) {
                    save_checkpoint(CKPT_PATH, step, last_loss, D, H, S, total_steps, lr, W1, W2,
                                    cum_compile_ms + total_compile_ms, cum_train_ms + total_train_ms,
                                    cum_wall_ms + tb_to_ms(mach_absolute_time() - t_wall_start, g_tb),
                                    cum_steps + total_steps_done, cum_batches + total_batches);
                    fflush(stdout);
                    execl(argv[0], argv[0], "--resume", NULL);
                    perror("execl failed"); return 1;
                }
            }

            if (last_loss < 1e-6f) { printf("\nConverged at step %d!\n", step); break; }
        }

        total_wall_ms = tb_to_ms(mach_absolute_time() - t_wall_start, g_tb);
        // Add cumulative from previous exec() runs
        total_compile_ms += cum_compile_ms;
        total_train_ms += cum_train_ms;
        total_wall_ms += cum_wall_ms;
        total_steps_done += cum_steps;
        total_batches += cum_batches;

        // === Final output ===
        printf("\nFinal output vs target (first 8):\n");
        if (k1_fwd && k2_fwd) {
            ane_eval_k(k1_fwd, x, h, D, H, S);
            for (int i = 0; i < S*H; i++) h_relu[i] = h[i] > 0 ? h[i] : 0;
            ane_eval_k(k2_fwd, h_relu, y, H, D, S);
        }
        printf("  y:      "); for (int i = 0; i < 8; i++) printf("%.4f ", y[i]); printf("\n");
        printf("  target: "); for (int i = 0; i < 8; i++) printf("%.4f ", y_target[i]); printf("\n");

        // === Efficiency Report ===
        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:     %d\n", total_steps_done);
        printf("Total batches:   %d (accum %d steps each)\n", total_batches, ACCUM_STEPS);
        printf("Wall time:       %.0f ms\n", total_wall_ms);
        printf("Compile time:    %.0f ms (%.1f%%)\n", total_compile_ms, 100.0*total_compile_ms/total_wall_ms);
        printf("Train time:      %.0f ms (%.1f%%)\n", total_train_ms, 100.0*total_train_ms/total_wall_ms);
        printf("Overhead:        %.0f ms (%.1f%%)\n",
               total_wall_ms - total_compile_ms - total_train_ms,
               100.0*(total_wall_ms - total_compile_ms - total_train_ms)/total_wall_ms);
        printf("\n");
        printf("Avg compile:     %.1f ms per batch (4 kernels)\n", total_compile_ms / total_batches);
        printf("Avg train:       %.2f ms per step (ANE fwd+bwd + CPU dW)\n", total_train_ms / total_steps_done);
        printf("Avg wall/step:   %.2f ms\n", total_wall_ms / total_steps_done);
        printf("\n");
        double ane_total_flops = ane_flops_per_step * total_steps_done;
        double cpu_total_flops = cpu_flops_per_step * total_steps_done;
        printf("ANE FLOPs total: %.3f MFLOP  (%.2f GFLOPS sustained)\n",
               ane_total_flops / 1e6, ane_total_flops / (total_train_ms * 1e6));
        printf("CPU FLOPs total: %.3f MFLOP  (%.2f GFLOPS sustained)\n",
               cpu_total_flops / 1e6, cpu_total_flops / (total_train_ms * 1e6));
        printf("Total FLOPs:     %.3f MFLOP  (%.2f GFLOPS sustained)\n",
               (ane_total_flops + cpu_total_flops) / 1e6,
               (ane_total_flops + cpu_total_flops) / (total_train_ms * 1e6));
        printf("\n");
        printf("ANE utilization: %.4f%% of 15.8 TFLOPS peak\n",
               100.0 * ane_total_flops / (total_train_ms * 1e6) / 15800.0);
        printf("Weight params:   %d (%.1f KB FP16)\n",
               H*D + D*H, weight_bytes / 1024.0);
        printf("Compile amortization: %.1f ms compile / %d steps = %.2f ms/step overhead\n",
               total_compile_ms / total_batches, ACCUM_STEPS,
               total_compile_ms / total_batches / ACCUM_STEPS);
        printf("Compile fraction: %.1f%% of wall time\n", 100.0 * total_compile_ms / total_wall_ms);
        printf("Train fraction:   %.1f%% of wall time (useful work)\n", 100.0 * total_train_ms / total_wall_ms);

        free_kern(k1_fwd); free_kern(k2_fwd); free_kern(k1_bwd); free_kern(k2_bwd);
        free(W1); free(W2); free(x); free(y_target);
        free(h); free(h_relu); free(y); free(dy); free(dh_relu); free(dh); free(dx_layer);
        unlink(CKPT_PATH);
    }
    return 0;
}
