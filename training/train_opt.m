// train_opt.m — Optimized train_large with:
//   Phase 1: NEON Adam, vectorized embed ops, pre-allocated capture buffers
//   Phase 2: Concurrent dW dispatch, fp16 activation cache
//   Phase 3: Metal GPU for weight gradient computation (dW)
//
// Key perf wins:
//   - Pre-allocated LayerCaptures: eliminates ~132 malloc/free per step
//   - Concurrent dW queue: individual sgemms run in parallel (was serial)
//   - fp16 activation cache: skip fp16→fp32 on main thread for dW-only buffers
//   - Metal GPU dW: ~12ms for all weight gradients vs ~435ms serial CPU
//   - NEON Adam: ~3x faster optimizer step
//   - Vectorized embed: vDSP_mtrans instead of scalar scatter/gather

#include "stories_io.h"
#include "stories_mil.h"
#include "stories_cpu_ops_opt.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#define CKPT_PATH "ane_stories110M_ckpt.bin"
#define MODEL_PATH_DEFAULT "stories110M.bin"
#define DATA_PATH "tinystories_data00.bin"

// ===== Pre-allocated capture buffers per layer (Phase 1) =====
// Eliminates malloc/free in dispatch blocks
typedef struct {
    // FFN dW captures
    float *dffn;        // [DIM, SEQ]
    float *silu_out;    // [HIDDEN, SEQ]
    float *dh1;         // [HIDDEN, SEQ]
    float *dh3;         // [HIDDEN, SEQ]
    float *x2norm;      // [DIM, SEQ]
    // Attn dW captures
    float *do_buf;      // [DIM, SEQ]  (for dWo)
    float *attn_out;    // [DIM, SEQ]
    // QKV dW captures
    float *dq;          // [DIM, SEQ]
    float *dk;          // [DIM, SEQ]
    float *dv;          // [DIM, SEQ]
    float *xnorm;       // [DIM, SEQ]
    // fp16 backward gradient cache (read raw from IOSurface, convert in dispatch block)
    _Float16 *dh1_fp16;  // [HIDDEN, SEQ]
    _Float16 *dh3_fp16;  // [HIDDEN, SEQ]
    _Float16 *dq_fp16;   // [DIM, SEQ]
    _Float16 *dk_fp16;   // [DIM, SEQ]
    _Float16 *dv_fp16;   // [DIM, SEQ]
} LayerCaptures;

static LayerCaptures layer_captures_alloc(void) {
    LayerCaptures c;
    c.dffn     = (float*)malloc(SEQ * DIM * 4);
    c.silu_out = (float*)malloc(SEQ * HIDDEN * 4);
    c.dh1      = (float*)malloc(SEQ * HIDDEN * 4);
    c.dh3      = (float*)malloc(SEQ * HIDDEN * 4);
    c.x2norm   = (float*)malloc(SEQ * DIM * 4);
    c.do_buf   = (float*)malloc(SEQ * DIM * 4);
    c.attn_out = (float*)malloc(SEQ * DIM * 4);
    c.dq       = (float*)malloc(SEQ * DIM * 4);
    c.dk       = (float*)malloc(SEQ * DIM * 4);
    c.dv       = (float*)malloc(SEQ * DIM * 4);
    c.xnorm    = (float*)malloc(SEQ * DIM * 4);
    c.dh1_fp16 = (_Float16*)malloc(SEQ * HIDDEN * 2);
    c.dh3_fp16 = (_Float16*)malloc(SEQ * HIDDEN * 2);
    c.dq_fp16  = (_Float16*)malloc(SEQ * DIM * 2);
    c.dk_fp16  = (_Float16*)malloc(SEQ * DIM * 2);
    c.dv_fp16  = (_Float16*)malloc(SEQ * DIM * 2);
    return c;
}
static void layer_captures_free(LayerCaptures *c) {
    free(c->dffn); free(c->silu_out); free(c->dh1); free(c->dh3);
    free(c->x2norm); free(c->do_buf); free(c->attn_out);
    free(c->dq); free(c->dk); free(c->dv); free(c->xnorm);
    free(c->dh1_fp16); free(c->dh3_fp16);
    free(c->dq_fp16); free(c->dk_fp16); free(c->dv_fp16);
}

// ===== fp16 activation cache (Phase 2) =====
// Store activations that are only used for dW as fp16 (skip main-thread conversion)
typedef struct {
    _Float16 *xnorm_fp16;     // [DIM, SEQ]
    _Float16 *attn_out_fp16;  // [DIM, SEQ]
    _Float16 *x2norm_fp16;    // [DIM, SEQ]
    _Float16 *silu_out_fp16;  // [HIDDEN, SEQ]
} LayerFP16Cache;

static LayerFP16Cache layer_fp16_cache_alloc(void) {
    LayerFP16Cache c;
    c.xnorm_fp16    = (_Float16*)malloc(SEQ * DIM * 2);
    c.attn_out_fp16 = (_Float16*)malloc(SEQ * DIM * 2);
    c.x2norm_fp16   = (_Float16*)malloc(SEQ * DIM * 2);
    c.silu_out_fp16 = (_Float16*)malloc(SEQ * HIDDEN * 2);
    return c;
}
static void layer_fp16_cache_free(LayerFP16Cache *c) {
    free(c->xnorm_fp16); free(c->attn_out_fp16);
    free(c->x2norm_fp16); free(c->silu_out_fp16);
}

// ===== Metal GPU dW context (Phase 3) =====
typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    // Shared gradient accumulator buffers (one per weight matrix per layer)
    id<MTLBuffer> dW_bufs[NLAYERS][9]; // Wq,Wk,Wv,Wo,W1,W2,W3,rms_att,rms_ffn
    id<MTLCommandBuffer> lastCmdBuf;   // Track last submitted buffer for sync
} MetalDWContext;

// Weight matrix indices for Metal buffers
enum { MW_Q=0, MW_K, MW_V, MW_O, MW_1, MW_2, MW_3, MW_RMSA, MW_RMSF };

static bool metal_dw_init(MetalDWContext *ctx) {
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) { printf("[Metal] No GPU device\n"); return false; }
    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) { printf("[Metal] No command queue\n"); return false; }

    // Allocate shared-mode gradient accumulator buffers
    size_t sizes[9] = {WQ_SZ*4, WQ_SZ*4, WQ_SZ*4, WO_SZ*4,
                       W1_SZ*4, W2_SZ*4, W3_SZ*4, DIM*4, DIM*4};
    for (int L = 0; L < NLAYERS; L++) {
        for (int w = 0; w < 9; w++) {
            ctx->dW_bufs[L][w] = [ctx->device newBufferWithLength:sizes[w]
                                               options:MTLResourceStorageModeShared];
            if (!ctx->dW_bufs[L][w]) { printf("[Metal] Buffer alloc failed L=%d w=%d\n", L, w); return false; }
        }
    }
    printf("[Metal] GPU: %s\n", [[ctx->device name] UTF8String]);
    return true;
}

static void metal_dw_zero(MetalDWContext *ctx) {
    size_t sizes[9] = {WQ_SZ*4, WQ_SZ*4, WQ_SZ*4, WO_SZ*4,
                       W1_SZ*4, W2_SZ*4, W3_SZ*4, DIM*4, DIM*4};
    for (int L = 0; L < NLAYERS; L++) {
        for (int w = 0; w < 9; w++) {
            memset([ctx->dW_bufs[L][w] contents], 0, sizes[w]);
        }
    }
}

// Encode a single dW sgemm to Metal command buffer using MPS
// C[M,N] += A[M,K] @ B^T[N,K]  (i.e., C += A @ B^T, accumulating into C)
static void metal_encode_dw_sgemm(id<MTLCommandBuffer> cmdBuf,
                                   id<MTLDevice> device,
                                   const float *a_data, int M, int K,
                                   const float *b_data, int N,
                                   id<MTLBuffer> c_buf) {
    // Create temporary input buffers (shared mode = zero-copy on Apple Silicon)
    id<MTLBuffer> aBuf = [device newBufferWithBytesNoCopy:(void*)a_data
                                   length:M * K * sizeof(float)
                                   options:MTLResourceStorageModeShared
                                   deallocator:nil];
    id<MTLBuffer> bBuf = [device newBufferWithBytesNoCopy:(void*)b_data
                                   length:N * K * sizeof(float)
                                   options:MTLResourceStorageModeShared
                                   deallocator:nil];

    // A is [M, K] row-major, B is [N, K] row-major
    // We want C += A @ B^T, i.e., C[M, N] = A[M, K] * B[K, N]^T
    // MPS uses row-major by default
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
        columns:K rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:N
        columns:K rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
        columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:c_buf descriptor:descC];

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:device transposeLeft:NO transposeRight:YES
        resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:1.0];

    [mm encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
}

// ===== Weight loading from llama2.c format =====
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return false; }
    Llama2Config cfg;
    fread(&cfg, sizeof(cfg), 1, f);
    printf("  Model config: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, abs(cfg.vocab_size), cfg.seq_len);
    if (cfg.dim != DIM || cfg.hidden_dim != HIDDEN || cfg.n_layers != NLAYERS) {
        printf("  ERROR: Config mismatch! Expected dim=%d hidden=%d layers=%d\n", DIM, HIDDEN, NLAYERS);
        fclose(f); return false;
    }
    int V = abs(cfg.vocab_size);
    bool shared = cfg.vocab_size > 0;
    (void)V; (void)shared;

    fread(embed, 4, VOCAB * DIM, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_att, 4, DIM, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wq, 4, WQ_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wk, 4, WQ_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wv, 4, WQ_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wo, 4, WO_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_ffn, 4, DIM, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W1, 4, W1_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W2, 4, W2_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W3, 4, W3_SZ, f);
    fread(rms_final, 4, DIM, f);
    fclose(f);
    printf("  Loaded pretrained weights (%s)\n", shared ? "shared embed/cls" : "separate cls");
    return true;
}

// ===== Compile one layer's kernels =====
static bool compile_layer_kernels(LayerKernels *lk, LayerWeights *w) {
    lk->fwdAttn = compile_kern_mil_w(gen_sdpa_fwd_taps(), (@{
        @"@model_path/weights/rms1.bin": @{@"offset":@0, @"data":build_blob(w->rms_att,1,DIM)},
        @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":build_blob(w->Wq,DIM,DIM)},
        @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":build_blob(w->Wk,DIM,DIM)},
        @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":build_blob(w->Wv,DIM,DIM)},
        @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":build_blob(w->Wo,DIM,DIM)},
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
    }), DIM*SEQ*2, 6*DIM*SEQ*2);

    lk->fwdFFN = compile_kern_mil_w(gen_ffn_fwd_taps(), (@{
        @"@model_path/weights/rms2.bin": @{@"offset":@0, @"data":build_blob(w->rms_ffn,1,DIM)},
        @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":build_blob(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":build_blob(w->W3,HIDDEN,DIM)},
        @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":build_blob(w->W2,DIM,HIDDEN)},
    }), DIM*SEQ*2, (2*DIM+3*HIDDEN)*SEQ*2);

    lk->ffnBwd = compile_kern_mil_w(gen_ffn_bwd(), (@{
        @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(w->W2,DIM,HIDDEN)},
        @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(w->W1,HIDDEN,DIM)},
        @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(w->W3,HIDDEN,DIM)},
    }), (DIM+2*HIDDEN)*SEQ*2, (DIM+2*HIDDEN)*SEQ*2);

    lk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1(), (@{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/wot.bin": @{@"offset":@0, @"data":build_blob_t(w->Wo,DIM,DIM)},
    }), 4*DIM*SEQ*2, (DIM+2*SCORE_CH)*SEQ*2);

    lk->qkvBwd = compile_kern_mil_w(gen_qkvb(), (@{
        @"@model_path/weights/wqt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wq,DIM,DIM)},
        @"@model_path/weights/wkt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wk,DIM,DIM)},
        @"@model_path/weights/wvt.bin": @{@"offset":@0, @"data":build_blob_t(w->Wv,DIM,DIM)},
    }), 3*DIM*SEQ*2, DIM*SEQ*2);

    return lk->fwdAttn && lk->fwdFFN && lk->ffnBwd && lk->sdpaBwd1 && lk->qkvBwd;
}

static Kern *compile_sdpa_bwd2(void) {
    return compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*DIM)*SEQ*2, 2*DIM*SEQ*2);
}

static void free_layer_kernels(LayerKernels *lk) {
    free_kern(lk->fwdAttn); free_kern(lk->fwdFFN); free_kern(lk->ffnBwd);
    free_kern(lk->sdpaBwd1); free_kern(lk->qkvBwd);
    lk->fwdAttn = lk->fwdFFN = lk->ffnBwd = lk->sdpaBwd1 = lk->qkvBwd = NULL;
}

// ===== Checkpoint save/load =====
static void save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double cc, double ct, double cw, int cs, int cb, int adam_t,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 2;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_compile = cc; h.cum_train = ct; h.cum_wall = cw;
    h.cum_steps = cs; h.cum_batches = cb; h.adam_t = adam_t;
    fwrite(&h, sizeof(h), 1, f);
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WQ_SZ,f);
        fwrite(lw[L].Wv,4,WQ_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WQ_SZ,f); fwrite(la[L].Wk.v,4,WQ_SZ,f);
        fwrite(la[L].Wv.m,4,WQ_SZ,f); fwrite(la[L].Wv.v,4,WQ_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,VOCAB*DIM,f);
    fwrite(aembed->m,4,VOCAB*DIM,f); fwrite(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps, float *lr, float *loss,
                             double *cc, double *ct, double *cw, int *cs, int *cb, int *adam_t,
                             LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 2) { fclose(f); return false; }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *cc = h.cum_compile; *ct = h.cum_train; *cw = h.cum_wall;
    *cs = h.cum_steps; *cb = h.cum_batches; *adam_t = h.adam_t;
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WQ_SZ,f);
        fread(lw[L].Wv,4,WQ_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WQ_SZ,f); fread(la[L].Wk.v,4,WQ_SZ,f);
        fread(la[L].Wv.m,4,WQ_SZ,f); fread(la[L].Wv.v,4,WQ_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,VOCAB*DIM,f);
    fread(aembed->m,4,VOCAB*DIM,f); fread(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
    return true;
}

// ===== Main =====
int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        // Phase 2: Limit BLAS thread count to prevent oversubscription with concurrent dispatch
        setenv("VECLIB_MAXIMUM_THREADS", "2", 1);

        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float lr = 3e-4f;
        float adam_b1=0.9f, adam_b2=0.999f, adam_eps=1e-8f;
        int adam_t = 0, start_step = 0;

        // Parse args
        const char *model_path = MODEL_PATH_DEFAULT;
        bool do_resume = false;
        bool use_metal = false;  // default off: Metal dW contends with ANE for memory bandwidth
        int pos = 0;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--no-metal") == 0) use_metal = false;
            else if (strcmp(argv[i], "--metal") == 0) use_metal = true;
            else if (argv[i][0] != '-') {
                if (pos == 0) model_path = argv[i];
                pos++;
            }
        }

        // Allocate per-layer state
        LayerWeights lw[NLAYERS];
        LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS];
        LayerGrads grads[NLAYERS];
        LayerKernels kern[NLAYERS];
        LayerCaptures caps[NLAYERS];       // Phase 1: pre-allocated captures
        LayerFP16Cache fp16cache[NLAYERS]; // Phase 2: fp16 activation cache
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc();
            la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc();
            grads[L] = layer_grads_alloc();
            memset(&kern[L], 0, sizeof(LayerKernels));
            caps[L] = layer_captures_alloc();
            fp16cache[L] = layer_fp16_cache_alloc();
        }

        // Final RMSNorm + embedding + classifier
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed = (float*)calloc(VOCAB*DIM, 4);
        AdamState arms_final = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);

        // Phase 1: Pre-allocate dx_rms scratch (was calloc/free per step)
        float *dx_rms_scratch = (float*)malloc(SEQ*DIM*4);
        // Phase 1: Pre-allocate embed temp buffer for vectorized ops
        float *embed_tmp = (float*)malloc(SEQ*DIM*4);

        // Phase 3: Metal GPU for dW
        MetalDWContext metal_ctx;
        bool metal_ok = false;
        if (use_metal) {
            metal_ok = metal_dw_init(&metal_ctx);
            if (!metal_ok) printf("[Metal] GPU init failed, falling back to CPU cblas\n");
        }

        // Classifier dW capture buffers (pre-allocated, Phase 1)
        float *capt_dlogits = (float*)malloc(SEQ*VOCAB*4);
        float *capt_xfinal  = (float*)malloc(SEQ*DIM*4);

        double cum_compile=0, cum_train=0, cum_wall=0;
        int cum_steps=0, cum_batches=0;

        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            resuming = load_checkpoint(CKPT_PATH, &start_step, &total_steps, &lr, &resume_loss,
                &cum_compile, &cum_train, &cum_wall, &cum_steps, &cum_batches, &adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            if (resuming) printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
        }
        if (!resuming) {
            printf("=== ANE Training (OPTIMIZED): Stories110M (12 layers) ===\n");
            printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n", DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS);
            printf("Optimizations: NEON-Adam, vec-embed, pre-alloc, concurrent-dW, fp16-cache%s\n",
                   metal_ok ? ", Metal-GPU-dW" : "");
            if (!load_pretrained(lw, rms_final, embed, model_path)) {
                printf("Pretrained load failed, using random init\n");
                srand48(42);
                float scale_d=1.0f/sqrtf(DIM), scale_h=1.0f/sqrtf(HIDDEN);
                for (int L=0; L<NLAYERS; L++) {
                    for(size_t i=0;i<WQ_SZ;i++){lw[L].Wq[i]=scale_d*(2*drand48()-1);lw[L].Wk[i]=scale_d*(2*drand48()-1);}
                    for(size_t i=0;i<WQ_SZ;i++){lw[L].Wv[i]=scale_d*(2*drand48()-1);lw[L].Wo[i]=scale_d*(2*drand48()-1);}
                    for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                    for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                    for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f; lw[L].rms_ffn[i]=1.0f;}
                }
                for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
                float escale = 0.02f;
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
            }
            size_t tp = (size_t)NLAYERS*LAYER_PARAMS + DIM + (size_t)VOCAB*DIM;
            double xfmr_params = (double)NLAYERS*LAYER_PARAMS;
            double embed_params = (double)VOCAB*DIM;
            printf("Params: %.2fM (transformer %.2fM + embed %.2fM)\n", tp/1e6, xfmr_params/1e6, embed_params/1e6);
            printf("Kernels: %d (%d weight-bearing + %d static sdpaBwd2)\n",
                   TOTAL_WEIGHT_KERNELS+NLAYERS, TOTAL_WEIGHT_KERNELS, NLAYERS);
            printf("Accum %d steps per recompile | Adam LR=%.1e b1=%.1f b2=%.3f\n", ACCUM_STEPS, lr, adam_b1, adam_b2);
            double fwd_f = NLAYERS*(4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
            double bwd_dx_f = fwd_f, bwd_dw_f = fwd_f;
            double sdpa_f = NLAYERS*2.0*HEADS*5*SEQ*SEQ*HD;
            double cls_f = 2.0*VOCAB*DIM*SEQ;
            double total_f = fwd_f + bwd_dx_f + bwd_dw_f + sdpa_f + cls_f*3;
            double ane_f = fwd_f + bwd_dx_f + sdpa_f;
            printf("FLOPs/step: fwd=%.0fM bwd_dx=%.0fM bwd_dW=%.0fM sdpa_bwd=%.0fM total=%.0fM\n",
                   fwd_f/1e6, bwd_dx_f/1e6, bwd_dw_f/1e6, sdpa_f/1e6, total_f/1e6);
            printf("ANE FLOPs/step: %.0fM (fwd+bwd_dx+sdpa_bwd) | %s: dW+cls\n\n",
                   ane_f/1e6, metal_ok ? "GPU" : "CPU cblas");
        }

        // mmap token data
        int data_fd = open(DATA_PATH, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", DATA_PATH); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // Gradient buffers shared across layers (reused each step)
        float *dy = (float*)malloc(SEQ*DIM*4);
        float *dffn = (float*)malloc(SEQ*DIM*4);
        float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *do_out_buf = (float*)malloc(SEQ*DIM*4);
        float *dq = (float*)malloc(SEQ*DIM*4);
        float *dk = (float*)malloc(SEQ*DIM*4);
        float *dv = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);

        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(SEQ*VOCAB*4);
        float *dlogits = (float*)malloc(SEQ*VOCAB*4);

        // Compile static sdpaBwd2 kernels
        Kern *sdpaBwd2[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            sdpaBwd2[L] = compile_sdpa_bwd2();
            if (!sdpaBwd2[L]) { printf("sdpaBwd2 compile failed\n"); return 1; }
        }

        // Phase 2: Concurrent dW dispatch queue (was DISPATCH_QUEUE_SERIAL)
        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_CONCURRENT);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        double total_compile_ms=0, total_train_ms=0;
        int total_steps_done=0, total_batches=0;
        uint64_t t_wall_start = mach_absolute_time();

        srand48(42 + start_step);

        int step = start_step;
        while (step < total_steps) {
            // Check compile budget
            if (g_compile_count + TOTAL_WEIGHT_KERNELS > MAX_COMPILES) {
                for (int L=0; L<NLAYERS; L++) { free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]); }
                double wall = tb_ms(mach_absolute_time() - t_wall_start);
                save_checkpoint(CKPT_PATH, step, total_steps, lr, last_loss,
                    total_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
                    total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
                    lw, la, rms_final, &arms_final, embed, &aembed);
                printf("[exec() restart step %d, %d compiles, loss=%.4f]\n", step, g_compile_count, last_loss);
                fflush(stdout);
                // Preserve --metal flag across restarts (default is off)
                if (use_metal) execl(argv[0], argv[0], "--resume", "--metal", NULL);
                else execl(argv[0], argv[0], "--resume", NULL);
                perror("execl"); return 1;
            }

            // Compile all layers' weight-bearing kernels
            uint64_t tc = mach_absolute_time();
            for (int L=0; L<NLAYERS; L++) free_layer_kernels(&kern[L]);

            bool compile_ok = true;
            for (int L=0; L<NLAYERS; L++) {
                printf("  Compiling layer %d/%d... (%d compiles)\r", L+1, NLAYERS, g_compile_count);
                fflush(stdout);
                if (!compile_layer_kernels(&kern[L], &lw[L])) {
                    printf("\nCompile failed at layer %d, restart\n", L);
                    compile_ok = false; break;
                }
            }
            if (!compile_ok) { g_compile_count = MAX_COMPILES; continue; }

            for (int L=0; L<NLAYERS; L++) {
                if (!sdpaBwd2[L]) {
                    sdpaBwd2[L] = compile_sdpa_bwd2();
                    if (!sdpaBwd2[L]) { printf("sdpaBwd2 recompile failed\n"); return 1; }
                }
            }

            double cms = tb_ms(mach_absolute_time() - tc);
            total_compile_ms += cms;
            printf("  Compiled %d kernels in %.0fms                    \n", TOTAL_WEIGHT_KERNELS, cms);

            // Zero gradient accumulators
            for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
            memset(grms_final, 0, DIM*4);
            memset(gembed, 0, (size_t)VOCAB*DIM*4);
            if (metal_ok) metal_dw_zero(&metal_ctx);

            int steps_batch = 0;
            uint64_t tt = mach_absolute_time();
            double t_ane=0,t_io=0,t_elem=0,t_rms=0,t_cblas_wait=0,t_cls=0,t_metal=0,t_bwd=0;

            for (int a=0; a<ACCUM_STEPS && step<total_steps; a++, step++) {
                uint64_t t0,t1;
                size_t max_pos = n_tokens - SEQ - 1;
                size_t pos = (size_t)(drand48() * max_pos);
                uint16_t *input_tokens = token_data + pos;
                uint16_t *target_tokens = token_data + pos + 1;

                // Phase 1: Vectorized embedding lookup
                t0=mach_absolute_time();
                embed_lookup_opt(x_cur, embed, input_tokens, DIM, SEQ, embed_tmp);
                t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0);

                // ===== FORWARD (12 layers) =====
                for (int L=0; L<NLAYERS; L++) {
                    LayerActs *ac = &acts[L];
                    LayerFP16Cache *fc = &fp16cache[L];

                    memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                    // Attention forward
                    t0=mach_absolute_time();
                    dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                    t1=mach_absolute_time(); t_cblas_wait+=tb_ms(t1-t0); t0=t1;
                    io_write_fp16(kern[L].fwdAttn->ioIn, x_cur, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].fwdAttn);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;

                    // Read o_out (needed on main thread for residual)
                    io_read_fp16(kern[L].fwdAttn->ioOut, ac->o_out, 0, DIM, SEQ);
                    // Phase 2: Read dW-only activations as raw fp16 (skip conversion on main thread)
                    io_read_raw_fp16(kern[L].fwdAttn->ioOut, fc->attn_out_fp16, 4*DIM, DIM, SEQ);
                    io_read_raw_fp16(kern[L].fwdAttn->ioOut, fc->xnorm_fp16, 5*DIM, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                    vDSP_vadd(x_cur, 1, ac->o_out, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
                    t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0); t0=t1;

                    // FFN forward
                    io_write_fp16(kern[L].fwdFFN->ioIn, ac->x2, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].fwdFFN);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;

                    // Read ffn_out (needed on main thread for residual)
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->ffn_out, 0, DIM, SEQ);
                    // h1, h3 NOT read here — backward uses io_copy from fwdFFN->ioOut directly
                    // silu_out and x2norm are dW-only → read as fp16
                    io_read_raw_fp16(kern[L].fwdFFN->ioOut, fc->silu_out_fp16, DIM+2*HIDDEN, HIDDEN, SEQ);
                    io_read_raw_fp16(kern[L].fwdFFN->ioOut, fc->x2norm_fp16, DIM+3*HIDDEN, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0);

                    vDSP_vadd(ac->x2, 1, ac->ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                    t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0);
                }

                // Final RMSNorm (CPU)
                t0=mach_absolute_time();
                rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
                t1=mach_absolute_time(); t_rms+=tb_ms(t1-t0); t0=t1;

                // Classifier
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            VOCAB, SEQ, DIM, 1.0f,
                            embed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
                t1=mach_absolute_time(); t_cls+=tb_ms(t1-t0); t0=t1;

                float loss = cross_entropy_loss(dlogits, logits, target_tokens, VOCAB, SEQ);
                last_loss = loss;
                t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0); t0=t1;

                // ===== BACKWARD =====
                uint64_t t_bwd_start = mach_absolute_time();
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            DIM, SEQ, VOCAB, 1.0f,
                            embed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);

                // dW embed (classifier) — async on dW queue
                memcpy(capt_dlogits, dlogits, SEQ*VOCAB*4);
                memcpy(capt_xfinal, x_final, SEQ*DIM*4);

                // Classifier dW on CPU (gembed is CPU-side accumulator, not Metal buffer)
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                VOCAB, DIM, SEQ, 1.0f,
                                capt_dlogits, SEQ, capt_xfinal, SEQ, 1.0f, gembed, DIM);
                });

                // Final RMSNorm backward (using pre-allocated scratch)
                memset(dx_rms_scratch, 0, SEQ*DIM*4);
                rmsnorm_bwd(dx_rms_scratch, grms_final, dy, x_cur, rms_final, DIM, SEQ);
                memcpy(dy, dx_rms_scratch, SEQ*DIM*4);

                // ===== BACKWARD (12 layers, reverse) =====
                for (int L=NLAYERS-1; L>=0; L--) {
                    LayerActs *ac = &acts[L];
                    LayerGrads *gr = &grads[L];
                    LayerCaptures *cp = &caps[L];
                    LayerFP16Cache *fc = &fp16cache[L];

                    memcpy(dffn, dy, SEQ*DIM*4);

                    // FFN backward (ANE)
                    io_write_fp16_at(kern[L].ffnBwd->ioIn, 0, dffn, DIM, SEQ);
                    io_copy(kern[L].ffnBwd->ioIn, DIM, kern[L].fwdFFN->ioOut, DIM, 2*HIDDEN, SEQ);
                    ane_eval(kern[L].ffnBwd);
                    io_read_fp16(kern[L].ffnBwd->ioOut, dx_ffn, 0, DIM, SEQ);
                    // dh1, dh3: only used for dW captures → read as raw fp16
                    io_read_raw_fp16(kern[L].ffnBwd->ioOut, cp->dh1_fp16, DIM, HIDDEN, SEQ);
                    io_read_raw_fp16(kern[L].ffnBwd->ioOut, cp->dh3_fp16, DIM+HIDDEN, HIDDEN, SEQ);

                    memcpy(cp->dffn, dffn, SEQ*DIM*4);

                    if (metal_ok) {
                        // Metal path: convert all on main thread for GPU buffers
                        cvt_f16_f32(cp->dh1, cp->dh1_fp16, SEQ*HIDDEN);
                        cvt_f16_f32(cp->dh3, cp->dh3_fp16, SEQ*HIDDEN);
                        cvt_f16_f32(cp->silu_out, fc->silu_out_fp16, SEQ*HIDDEN);
                        cvt_f16_f32(cp->x2norm, fc->x2norm_fp16, SEQ*DIM);

                        @autoreleasepool {
                            id<MTLCommandBuffer> cmdBuf = [metal_ctx.queue commandBuffer];
                            metal_encode_dw_sgemm(cmdBuf, metal_ctx.device,
                                cp->dffn, DIM, SEQ, cp->silu_out, HIDDEN,
                                metal_ctx.dW_bufs[L][MW_2]);
                            metal_encode_dw_sgemm(cmdBuf, metal_ctx.device,
                                cp->dh1, HIDDEN, SEQ, cp->x2norm, DIM,
                                metal_ctx.dW_bufs[L][MW_1]);
                            metal_encode_dw_sgemm(cmdBuf, metal_ctx.device,
                                cp->dh3, HIDDEN, SEQ, cp->x2norm, DIM,
                                metal_ctx.dW_bufs[L][MW_3]);
                            [cmdBuf commit];
                            metal_ctx.lastCmdBuf = cmdBuf;
                        }
                    } else {
                        // CPU: concurrent dispatch, convert fp16→fp32 in each block
                        _Float16 *fc_silu = fc->silu_out_fp16;
                        _Float16 *cp_dh1_f16 = cp->dh1_fp16, *cp_dh3_f16 = cp->dh3_fp16;
                        float *cp_dffn = cp->dffn, *cp_silu = cp->silu_out;
                        float *cp_dh1 = cp->dh1, *cp_dh3 = cp->dh3, *cp_x2n = cp->x2norm;
                        float *gr_W2 = gr->W2, *gr_W1 = gr->W1, *gr_W3 = gr->W3;

                        // Convert shared x2norm on main thread (W1+W3 blocks read concurrently)
                        cvt_f16_f32(cp_x2n, fc->x2norm_fp16, SEQ*DIM);

                        dispatch_group_async(dw_grp, dw_q, ^{
                            cvt_f16_f32(cp_silu, fc_silu, SEQ*HIDDEN);
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                        1.0f, cp_dffn, SEQ, cp_silu, SEQ, 1.0f, gr_W2, HIDDEN);
                        });
                        dispatch_group_async(dw_grp, dw_q, ^{
                            cvt_f16_f32(cp_dh1, cp_dh1_f16, SEQ*HIDDEN);
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                        1.0f, cp_dh1, SEQ, cp_x2n, SEQ, 1.0f, gr_W1, DIM);
                        });
                        dispatch_group_async(dw_grp, dw_q, ^{
                            cvt_f16_f32(cp_dh3, cp_dh3_f16, SEQ*HIDDEN);
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                        1.0f, cp_dh3, SEQ, cp_x2n, SEQ, 1.0f, gr_W3, DIM);
                        });
                    }

                    // RMSNorm2 backward
                    memset(dx2, 0, SEQ*DIM*4);
                    rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                    for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];

                    // dWo async
                    memcpy(cp->do_buf, dx2, SEQ*DIM*4);

                    if (metal_ok) {
                        cvt_f16_f32(cp->attn_out, fc->attn_out_fp16, SEQ*DIM);
                        @autoreleasepool {
                            id<MTLCommandBuffer> cmdBuf = [metal_ctx.queue commandBuffer];
                            metal_encode_dw_sgemm(cmdBuf, metal_ctx.device,
                                cp->do_buf, DIM, SEQ, cp->attn_out, DIM,
                                metal_ctx.dW_bufs[L][MW_O]);
                            [cmdBuf commit];
                            metal_ctx.lastCmdBuf = cmdBuf;
                        }
                    } else {
                        _Float16 *fc_attn = fc->attn_out_fp16;
                        float *cp_do = cp->do_buf, *cp_attn = cp->attn_out;
                        float *gr_Wo = gr->Wo;
                        dispatch_group_async(dw_grp, dw_q, ^{
                            cvt_f16_f32(cp_attn, fc_attn, SEQ*DIM);
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                        1.0f, cp_do, SEQ, cp_attn, SEQ, 1.0f, gr_Wo, DIM);
                        });
                    }

                    // SDPA backward (ANE)
                    io_copy(kern[L].sdpaBwd1->ioIn, 0, kern[L].fwdAttn->ioOut, DIM, 3*DIM, SEQ);
                    io_write_fp16_at(kern[L].sdpaBwd1->ioIn, 3*DIM, dx2, DIM, SEQ);
                    ane_eval(kern[L].sdpaBwd1);
                    io_copy(sdpaBwd2[L]->ioIn, 0, kern[L].sdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ);
                    io_copy(sdpaBwd2[L]->ioIn, 2*SCORE_CH, kern[L].fwdAttn->ioOut, DIM, 2*DIM, SEQ);
                    ane_eval(sdpaBwd2[L]);

                    // dq, dk, dv: only used for dW captures → read as raw fp16
                    io_read_raw_fp16(sdpaBwd2[L]->ioOut, cp->dq_fp16, 0, DIM, SEQ);
                    io_read_raw_fp16(sdpaBwd2[L]->ioOut, cp->dk_fp16, DIM, DIM, SEQ);
                    io_read_raw_fp16(kern[L].sdpaBwd1->ioOut, cp->dv_fp16, 0, DIM, SEQ);

                    if (metal_ok) {
                        // Metal path: convert all on main thread for GPU buffers
                        cvt_f16_f32(cp->dq, cp->dq_fp16, SEQ*DIM);
                        cvt_f16_f32(cp->dk, cp->dk_fp16, SEQ*DIM);
                        cvt_f16_f32(cp->dv, cp->dv_fp16, SEQ*DIM);
                        cvt_f16_f32(cp->xnorm, fc->xnorm_fp16, SEQ*DIM);
                        @autoreleasepool {
                            id<MTLCommandBuffer> cmdBuf = [metal_ctx.queue commandBuffer];
                            metal_encode_dw_sgemm(cmdBuf, metal_ctx.device,
                                cp->dq, DIM, SEQ, cp->xnorm, DIM,
                                metal_ctx.dW_bufs[L][MW_Q]);
                            metal_encode_dw_sgemm(cmdBuf, metal_ctx.device,
                                cp->dk, DIM, SEQ, cp->xnorm, DIM,
                                metal_ctx.dW_bufs[L][MW_K]);
                            metal_encode_dw_sgemm(cmdBuf, metal_ctx.device,
                                cp->dv, DIM, SEQ, cp->xnorm, DIM,
                                metal_ctx.dW_bufs[L][MW_V]);
                            [cmdBuf commit];
                            metal_ctx.lastCmdBuf = cmdBuf;
                        }
                    } else {
                        _Float16 *cp_dq_f16 = cp->dq_fp16, *cp_dk_f16 = cp->dk_fp16, *cp_dv_f16 = cp->dv_fp16;
                        float *cp_dq = cp->dq, *cp_dk = cp->dk, *cp_dv = cp->dv, *cp_xn = cp->xnorm;
                        float *gr_Wq = gr->Wq, *gr_Wk = gr->Wk, *gr_Wv = gr->Wv;
                        // Convert shared xnorm on main thread (all 3 blocks read concurrently)
                        cvt_f16_f32(cp_xn, fc->xnorm_fp16, SEQ*DIM);
                        dispatch_group_async(dw_grp, dw_q, ^{
                            cvt_f16_f32(cp_dq, cp_dq_f16, SEQ*DIM);
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                        1.0f, cp_dq, SEQ, cp_xn, SEQ, 1.0f, gr_Wq, DIM);
                        });
                        dispatch_group_async(dw_grp, dw_q, ^{
                            cvt_f16_f32(cp_dk, cp_dk_f16, SEQ*DIM);
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                        1.0f, cp_dk, SEQ, cp_xn, SEQ, 1.0f, gr_Wk, DIM);
                        });
                        dispatch_group_async(dw_grp, dw_q, ^{
                            cvt_f16_f32(cp_dv, cp_dv_f16, SEQ*DIM);
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                        1.0f, cp_dv, SEQ, cp_xn, SEQ, 1.0f, gr_Wv, DIM);
                        });
                    }

                    // QKV backward (ANE)
                    io_copy(kern[L].qkvBwd->ioIn, 0, sdpaBwd2[L]->ioOut, 0, 2*DIM, SEQ);
                    io_copy(kern[L].qkvBwd->ioIn, 2*DIM, kern[L].sdpaBwd1->ioOut, 0, DIM, SEQ);
                    ane_eval(kern[L].qkvBwd);
                    io_read_fp16(kern[L].qkvBwd->ioOut, dx_attn, 0, DIM, SEQ);

                    // RMSNorm1 backward
                    memset(dx_rms_scratch, 0, SEQ*DIM*4);
                    rmsnorm_bwd(dx_rms_scratch, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                    for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_rms_scratch[i] + dx2[i];
                }

                // Embedding backward
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                // Phase 1: Vectorized embed backward
                embed_backward_opt(gembed, dy, input_tokens, DIM, SEQ, embed_tmp);
                t_bwd += tb_ms(mach_absolute_time() - t_bwd_start);

                steps_batch++;
                if (step % 10 == 0 || step == start_step)
                    printf("step %-4d loss=%.4f\n", step, loss);

                // JSON telemetry to stderr
                double step_ane = t_ane/steps_batch, step_io = t_io/steps_batch;
                double step_cls = t_cls/steps_batch, step_elem = t_elem/steps_batch;
                double step_rms = t_rms/steps_batch, step_cbw = t_cblas_wait/steps_batch;
                fprintf(stderr, "{\"type\":\"step\",\"step\":%d,\"loss\":%.6f,"
                    "\"t_ane\":%.3f,\"t_io\":%.3f,\"t_cls\":%.3f,"
                    "\"t_elem\":%.3f,\"t_rms\":%.3f,\"t_cblas_wait\":%.3f,"
                    "\"t_bwd\":%.3f,\"t_metal\":%.3f,\"compiles\":%d}\n",
                    step, loss, step_ane, step_io, step_cls, step_elem, step_rms, step_cbw,
                    t_bwd/steps_batch, t_metal/steps_batch, g_compile_count);
            }
            double tms = tb_ms(mach_absolute_time() - tt);
            total_train_ms += tms;
            total_steps_done += steps_batch;
            total_batches++;

            // Ensure all async dW finished (CPU cblas or Metal)
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

            // Phase 3: If Metal, wait for GPU then copy gradient accumulators to CPU grads
            if (metal_ok) {
                // Must wait for all GPU command buffers to complete before reading
                if (metal_ctx.lastCmdBuf) {
                    [metal_ctx.lastCmdBuf waitUntilCompleted];
                    metal_ctx.lastCmdBuf = nil;
                }
                for (int L = 0; L < NLAYERS; L++) {
                    float *gpu_ptrs[7];
                    float *cpu_ptrs[7];
                    size_t sizes[7] = {WQ_SZ*4, WQ_SZ*4, WQ_SZ*4, WO_SZ*4,
                                       W1_SZ*4, W2_SZ*4, W3_SZ*4};
                    int indices[7] = {MW_Q, MW_K, MW_V, MW_O, MW_1, MW_2, MW_3};
                    cpu_ptrs[0]=grads[L].Wq; cpu_ptrs[1]=grads[L].Wk; cpu_ptrs[2]=grads[L].Wv;
                    cpu_ptrs[3]=grads[L].Wo; cpu_ptrs[4]=grads[L].W1; cpu_ptrs[5]=grads[L].W2;
                    cpu_ptrs[6]=grads[L].W3;

                    for (int w = 0; w < 7; w++) {
                        gpu_ptrs[w] = (float*)[metal_ctx.dW_bufs[L][indices[w]] contents];
                        // Accumulate GPU gradients into CPU accumulators
                        vDSP_vadd(gpu_ptrs[w], 1, cpu_ptrs[w], 1, cpu_ptrs[w], 1,
                                  (vDSP_Length)(sizes[w]/4));
                    }
                }
            }

            // Adam update (scale gradients by 1/steps_batch)
            float gsc = 1.0f / steps_batch;
            adam_t++;
            for (int L=0; L<NLAYERS; L++) {
                LayerGrads *g = &grads[L];
                for(size_t i=0;i<WQ_SZ;i++){g->Wq[i]*=gsc;g->Wk[i]*=gsc;g->Wv[i]*=gsc;g->Wo[i]*=gsc;}
                for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
                for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
                for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
                for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}

                // Phase 1: NEON Adam
                adam_update_opt(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update_opt(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update_opt(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update_opt(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update_opt(lw[L].W1, g->W1, &la[L].W1, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update_opt(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update_opt(lw[L].W3, g->W3, &la[L].W3, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update_opt(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update_opt(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps);
            }
            for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
            adam_update_opt(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps);
            for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;
            adam_update_opt(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps);

            printf("  [batch %d: compile=%.0fms train=%.1fms (%.1fms/step) compiles=%d]\n",
                   steps_batch, cms, tms, tms/steps_batch, g_compile_count);
            printf("    fwd: ane=%.1f io=%.1f cls=%.1f elem=%.1f rms=%.1f | bwd=%.1f | cblas_wait=%.1f ms/step\n",
                   t_ane/steps_batch, t_io/steps_batch, t_cls/steps_batch, t_elem/steps_batch,
                   t_rms/steps_batch, t_bwd/steps_batch, t_cblas_wait/steps_batch);

            // JSON batch telemetry to stderr
            {
                double bf = NLAYERS * (4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
                double bs = NLAYERS * 2.0*HEADS*5*SEQ*SEQ*HD;
                double ane_f_batch = (bf*2 + bs) * steps_batch;
                double ane_tflops = ane_f_batch / (tms * 1e9);
                fprintf(stderr, "{\"type\":\"batch\",\"batch\":%d,\"compile_ms\":%.1f,"
                    "\"train_ms\":%.1f,\"ms_per_step\":%.1f}\n",
                    steps_batch, cms, tms, tms/steps_batch);
                fprintf(stderr, "{\"type\":\"perf\",\"ane_tflops\":%.3f,\"ane_util_pct\":%.2f,"
                    "\"metal_dw\":%s}\n",
                    ane_tflops, 100.0*ane_tflops/15.8, metal_ok ? "true" : "false");
            }
        }

        // Efficiency report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        total_compile_ms += cum_compile; total_train_ms += cum_train;
        wall += cum_wall; total_steps_done += cum_steps; total_batches += cum_batches;
        double fwd_flops = NLAYERS * (4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
        double sdpa_flops = NLAYERS * 2.0*HEADS*5*SEQ*SEQ*HD;
        double cls_flops = 2.0*VOCAB*DIM*SEQ;
        double total_flops = (fwd_flops*3 + sdpa_flops + cls_flops*3) * total_steps_done;
        double ane_flops = (fwd_flops*2 + sdpa_flops) * total_steps_done;
        printf("\n=== Efficiency Report (OPTIMIZED) ===\n");
        printf("Total steps:     %d\n", total_steps_done);
        printf("Wall time:       %.0f ms (%.1f s)\n", wall, wall/1000);
        printf("Compile time:    %.0f ms (%.1f%%)\n", total_compile_ms, 100*total_compile_ms/wall);
        printf("Train time:      %.0f ms (%.1f%%)\n", total_train_ms, 100*total_train_ms/wall);
        printf("Avg train:       %.1f ms/step\n", total_train_ms/total_steps_done);
        printf("ANE TFLOPS:      %.2f sustained\n", ane_flops / (total_train_ms * 1e9));
        printf("Total TFLOPS:    %.2f (ANE+%s)\n", total_flops / (total_train_ms * 1e9),
               metal_ok ? "GPU" : "CPU");
        printf("ANE utilization: %.1f%% of 15.8 TFLOPS\n", 100*ane_flops/(total_train_ms*1e9)/15.8);
        printf("Metal GPU dW:    %s\n", metal_ok ? "ENABLED" : "disabled");

        // Cleanup
        for (int L=0; L<NLAYERS; L++) {
            free_layer_kernels(&kern[L]);
            free_kern(sdpaBwd2[L]);
            layer_weights_free(&lw[L]);
            layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]);
            layer_grads_free(&grads[L]);
            layer_captures_free(&caps[L]);
            layer_fp16_cache_free(&fp16cache[L]);
        }
        munmap(token_data, data_len);
        close(data_fd);
        free(rms_final); free(embed); free(grms_final); free(gembed);
        adam_free(&arms_final); adam_free(&aembed);
        free(dy); free(dffn); free(dh1); free(dh3); free(dx_ffn); free(dx2);
        free(do_out_buf); free(dq); free(dk); free(dv); free(dx_attn);
        free(x_cur); free(x_final); free(logits); free(dlogits);
        free(dx_rms_scratch); free(embed_tmp);
        free(capt_dlogits); free(capt_xfinal);
    }
    return 0;
}
