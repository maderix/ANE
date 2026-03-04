// train.m — Dynamic weight ANE training for Stories110M
// Compile kernels ONCE at startup, update weights via IOSurface every step.
// No exec() restart needed — eliminates 76% compile overhead.
#include "mil_dynamic.h"
#include "cpu_ops.h"
#include "../data_validation.h"

#define CKPT_PATH "ane_stories110M_dyn_ckpt.bin"
#define MODEL_PATH "../../../assets/models/stories110M.bin"
#define DATA_PATH "../tinystories_data00.bin"

// Dynamic kernel set per layer
typedef struct {
    Kern *sdpaFwd;     // QKV matmul + SDPA + Wo matmul (dynamic weights via IOSurface)
    Kern *ffnW13;      // W1,W3 matmul (dynamic)
    Kern *ffnW2;       // W2 matmul (dynamic)
    Kern *ffnBwdW2t;   // dffn @ W2^T (dynamic)
    Kern *ffnBwdW13t;  // dh1@W1^T + dh3@W3^T (dynamic)
    Kern *wotBwd;      // dx2 @ Wo^T (dynamic)
    Kern *sdpaBwd1;    // Q,K,V,da → dV,probs,dp (weight-free, has mask const)
    Kern *sdpaBwd2;    // probs,dp,Q,K → dQ,dK (weight-free)
    Kern *qkvBwd;      // dq@Wq^T + dk@Wk^T + dv@Wv^T (dynamic)
} DynLayerKernels;

// ===== Weight loading from llama2.c format =====
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return false; }
    Llama2Config cfg;
    fread(&cfg, sizeof(cfg), 1, f);
    printf("  Model: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, abs(cfg.vocab_size), cfg.seq_len);
    if (cfg.dim != DIM || cfg.hidden_dim != HIDDEN || cfg.n_layers != NLAYERS) {
        printf("  ERROR: Config mismatch!\n"); fclose(f); return false;
    }
    int V = abs(cfg.vocab_size);
    fread(embed, 4, V * DIM, f);
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
    printf("  Loaded pretrained weights\n");
    return true;
}

// Transpose W[rows,cols] → W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// ===== Compile all dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk) {
    NSDictionary *mask_w = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};

    // SDPA forward: [1, DIM, 1, SEQ+4*DIM] fp32 → [1, 6*DIM, 1, SEQ] fp32
    printf("  Compiling sdpaFwd...\n");
    dk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), mask_w,
        DIM*(SEQ+4*DIM)*4, 6*DIM*SEQ*4);
    if (!dk->sdpaFwd) return false;

    // FFN W1+W3: [1, DIM, 1, SEQ+2*HIDDEN] fp32 → [1, 3*HIDDEN, 1, SEQ] fp32
    printf("  Compiling ffnW13...\n");
    dk->ffnW13 = compile_kern_mil_w(gen_ffn_w13_dynamic(), @{},
        DIM*(SEQ+2*HIDDEN)*4, 3*HIDDEN*SEQ*4);
    if (!dk->ffnW13) return false;

    // FFN W2: [1, HIDDEN, 1, SEQ+DIM] fp32 → [1, DIM, 1, SEQ] fp32
    printf("  Compiling ffnW2...\n");
    dk->ffnW2 = compile_kern_mil_w(gen_ffn_w2_dynamic(), @{},
        HIDDEN*(SEQ+DIM)*4, DIM*SEQ*4);
    if (!dk->ffnW2) return false;

    // FFN backward W2^T: [1, DIM, 1, SEQ+HIDDEN] fp32 → [1, HIDDEN, 1, SEQ] fp32
    printf("  Compiling ffnBwdW2t...\n");
    dk->ffnBwdW2t = compile_kern_mil_w(gen_ffn_bwd_w2t_dynamic(), @{},
        DIM*(SEQ+HIDDEN)*4, HIDDEN*SEQ*4);
    if (!dk->ffnBwdW2t) return false;

    // FFN backward W1^T+W3^T: [1, HIDDEN, 1, 2*SEQ+2*DIM] fp32 → [1, DIM, 1, SEQ] fp32
    printf("  Compiling ffnBwdW13t...\n");
    dk->ffnBwdW13t = compile_kern_mil_w(gen_ffn_bwd_w13t_dynamic(), @{},
        HIDDEN*(2*SEQ+2*DIM)*4, DIM*SEQ*4);
    if (!dk->ffnBwdW13t) return false;

    // Wo^T backward: [1, DIM, 1, SEQ+DIM] fp32 → [1, DIM, 1, SEQ] fp32
    printf("  Compiling wotBwd...\n");
    dk->wotBwd = compile_kern_mil_w(gen_wot_dynamic(), @{},
        DIM*(SEQ+DIM)*4, DIM*SEQ*4);
    if (!dk->wotBwd) return false;

    // SDPA bwd1 (no dynamic weights, has mask): [1, 4*DIM, 1, SEQ] fp16 → [1, DIM+2*SCORE_CH, 1, SEQ] fp16
    printf("  Compiling sdpaBwd1...\n");
    dk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1_noweight(), mask_w,
        4*DIM*SEQ*2, (DIM+2*SCORE_CH)*SEQ*2);
    if (!dk->sdpaBwd1) return false;

    // SDPA bwd2 (no weights): [1, 2*SCORE_CH+2*DIM, 1, SEQ] fp16 → [1, 2*DIM, 1, SEQ] fp16
    printf("  Compiling sdpaBwd2...\n");
    dk->sdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*DIM)*SEQ*2, 2*DIM*SEQ*2);
    if (!dk->sdpaBwd2) return false;

    // QKV backward: [1, DIM, 1, 3*SEQ+3*DIM] fp32 → [1, DIM, 1, SEQ] fp32
    printf("  Compiling qkvBwd...\n");
    dk->qkvBwd = compile_kern_mil_w(gen_qkvb_dynamic(), @{},
        DIM*(3*SEQ+3*DIM)*4, DIM*SEQ*4);
    if (!dk->qkvBwd) return false;

    return true;
}

// ===== Write dynamic weights into IOSurface =====
// sdpaFwd: [1, DIM, 1, SEQ+4*DIM] — xnorm at sp[0:S], Wq/Wk/Wv/Wo at sp[S:]
static void write_sdpa_fwd_input(DynLayerKernels *dk, const float *xnorm,
                                  const float *Wq, const float *Wk, const float *Wv, const float *Wo) {
    IOSurfaceLock(dk->sdpaFwd->ioIn, 0, NULL);
    float *buf = (float*)IOSurfaceGetBaseAddress(dk->sdpaFwd->ioIn);
    int sp = SEQ + 4*DIM;
    for (int d = 0; d < DIM; d++) {
        memcpy(buf + d*sp, xnorm + d*SEQ, SEQ*4);
        memcpy(buf + d*sp + SEQ,       Wq + d*DIM, DIM*4);
        memcpy(buf + d*sp + SEQ+DIM,   Wk + d*DIM, DIM*4);
        memcpy(buf + d*sp + SEQ+2*DIM, Wv + d*DIM, DIM*4);
        memcpy(buf + d*sp + SEQ+3*DIM, Wo + d*DIM, DIM*4);
    }
    IOSurfaceUnlock(dk->sdpaFwd->ioIn, 0, NULL);
}

// ffnW13: [1, DIM, 1, SEQ+2*HIDDEN] — xnorm at sp[0:S], W1,W3 at sp[S:]
static void write_ffn_w13_input(DynLayerKernels *dk, const float *xnorm,
                                const float *W1, const float *W3) {
    IOSurfaceLock(dk->ffnW13->ioIn, 0, NULL);
    float *buf = (float*)IOSurfaceGetBaseAddress(dk->ffnW13->ioIn);
    int sp = SEQ + 2*HIDDEN;
    for (int d = 0; d < DIM; d++) {
        memcpy(buf + d*sp, xnorm + d*SEQ, SEQ*4);
        memcpy(buf + d*sp + SEQ,        W1 + d*HIDDEN, HIDDEN*4);
        memcpy(buf + d*sp + SEQ+HIDDEN,  W3 + d*HIDDEN, HIDDEN*4);
    }
    IOSurfaceUnlock(dk->ffnW13->ioIn, 0, NULL);
}

// ffnW2: [1, HIDDEN, 1, SEQ+DIM] — gate at sp[0:S], W2 at sp[S:]
static void write_ffn_w2_input(DynLayerKernels *dk, const float *gate, const float *W2) {
    IOSurfaceLock(dk->ffnW2->ioIn, 0, NULL);
    float *buf = (float*)IOSurfaceGetBaseAddress(dk->ffnW2->ioIn);
    int sp = SEQ + DIM;
    for (int d = 0; d < HIDDEN; d++) {
        memcpy(buf + d*sp, gate + d*SEQ, SEQ*4);
        memcpy(buf + d*sp + SEQ, W2 + d*DIM, DIM*4);
    }
    IOSurfaceUnlock(dk->ffnW2->ioIn, 0, NULL);
}

// ===== Checkpoint =====
static void save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double ct, double cw, int cs, int adam_t,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 3;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_train = ct; h.cum_wall = cw; h.cum_steps = cs; h.adam_t = adam_t;
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
                             double *ct, double *cw, int *cs, int *adam_t,
                             LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 3) { fclose(f); return false; }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *ct = h.cum_train; *cw = h.cum_wall; *cs = h.cum_steps; *adam_t = h.adam_t;
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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float max_lr = 3e-4f;
        float adam_b1=0.9f, adam_b2=0.999f, adam_eps=1e-8f;
        int adam_t = 0, start_step = 0;
        int accum_steps = 10;
        int warmup_steps = 100;
        float grad_clip = 1.0f;
        float min_lr_frac = 0.1f;  // min_lr = max_lr * 0.1

        bool do_resume = false, from_scratch = false;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--scratch") == 0) from_scratch = true;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) max_lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--accum") == 0 && i+1<argc) accum_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--warmup") == 0 && i+1<argc) warmup_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--clip") == 0 && i+1<argc) grad_clip = atof(argv[++i]);
        }
        float lr = max_lr;

        // Allocate per-layer state
        LayerWeights lw[NLAYERS]; LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS]; LayerGrads grads[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc(); la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc(); grads[L] = layer_grads_alloc();
        }
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed = (float*)calloc(VOCAB*DIM, 4);
        AdamState arms_final = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);

        double cum_train=0, cum_wall=0; int cum_steps=0;
        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            resuming = load_checkpoint(CKPT_PATH, &start_step, &total_steps, &lr, &resume_loss,
                &cum_train, &cum_wall, &cum_steps, &adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            if (resuming) printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
        }
        if (!resuming) {
            printf("=== ANE Dynamic Training: Stories110M (12 layers) ===\n");
            printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n", DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS);
            // Param counts for dashboard
            double xformer_m = (double)NLAYERS*(4.0*WQ_SZ + 2.0*W1_SZ + W2_SZ + W3_SZ + 2.0*DIM) / 1e6;
            double embed_m = (double)VOCAB*DIM / 1e6;
            printf("Params: %.1fM (transformer %.1fM + embed %.1fM)\n", xformer_m+embed_m, xformer_m, embed_m);
            printf("Kernels: 9 compiled, 9 weight-bearing\n");
            printf("Accum %d steps, LR=%g\n", accum_steps, max_lr);
            // FLOPs estimate: 6*N*B*T for transformer (forward+backward ≈ 3x forward)
            double fwd_flops = 2.0*NLAYERS*(4.0*WQ_SZ + 2.0*W1_SZ + W2_SZ + W3_SZ) * SEQ;
            double total_flops = 3.0 * fwd_flops;  // fwd + bwd ≈ 3x fwd
            printf("FLOPs/step: fwd=%.1fM bwd_dx=%.1fM bwd_dW=%.1fM sdpa_bwd=0.0M total=%.1fM\n",
                   fwd_flops/1e6, fwd_flops/1e6, fwd_flops/1e6, total_flops/1e6);
            printf("ANE FLOPs/step: %.1fM\n", total_flops/1e6);
            if (from_scratch || !load_pretrained(lw, rms_final, embed, MODEL_PATH)) {
                if (from_scratch) printf("  Training from scratch (random init)\n");
                else printf("  Pretrained load failed, using random init\n");
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
        }

        // Precompute transposed weights (for backward pass kernels)
        // These get updated after each Adam step
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W2t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            Wqt_buf[L]=(float*)malloc(WQ_SZ*4); Wkt_buf[L]=(float*)malloc(WQ_SZ*4);
            Wvt_buf[L]=(float*)malloc(WQ_SZ*4); Wot_buf[L]=(float*)malloc(WO_SZ*4);
            W1t_buf[L]=(float*)malloc(W1_SZ*4); W2t_buf[L]=(float*)malloc(W2_SZ*4);
            W3t_buf[L]=(float*)malloc(W3_SZ*4);
            transpose_weight(Wqt_buf[L], lw[L].Wq, DIM, DIM);
            transpose_weight(Wkt_buf[L], lw[L].Wk, DIM, DIM);
            transpose_weight(Wvt_buf[L], lw[L].Wv, DIM, DIM);
            transpose_weight(Wot_buf[L], lw[L].Wo, DIM, DIM);
            transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
            transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
            transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        }

        // mmap token data
        int data_fd = open(DATA_PATH, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", DATA_PATH); return 1; }
        struct stat st;
        if (fstat(data_fd, &st) != 0) { perror("fstat"); close(data_fd); return 1; }
        size_t data_len = (size_t)st.st_size;
        size_t n_tokens = 0, extra_bytes = 0;
        if (!token_data_bytes_to_token_count(data_len, &n_tokens, &extra_bytes)) {
            fprintf(stderr,
                    "Token data validation failed: file size %zu bytes has %zu extra byte(s); expected 16-bit tokens\n",
                    data_len, extra_bytes);
            close(data_fd);
            return 1;
        }
        if (n_tokens == 0) {
            fprintf(stderr, "Token data validation failed: token file is empty\n");
            close(data_fd);
            return 1;
        }
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { perror("mmap"); close(data_fd); return 1; }
        close(data_fd); // mapping remains valid; avoid fd leaks across exec() restarts
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        TokenDataValidationError data_err = {0};
        TokenDataValidationCode data_code = token_data_validate(token_data, n_tokens, SEQ, VOCAB, &data_err);
        if (data_code == TOKEN_DATA_ERR_TOO_SHORT) {
            fprintf(stderr, "Token data validation failed: need at least %zu tokens (SEQ+1), got %zu\n",
                    data_err.required_tokens, n_tokens);
            munmap(token_data, data_len);
            return 1;
        }
        if (data_code == TOKEN_DATA_ERR_OOB_TOKEN) {
            fprintf(stderr, "Token data validation failed: token %u at index %zu is outside vocab [0, %d)\n",
                    data_err.bad_token, data_err.bad_index, VOCAB);
            munmap(token_data, data_len);
            return 1;
        }

        // Vocab compaction: map 32K sparse vocab → ~9K compact
        VocabMap vm = vocab_map_build(token_data, n_tokens, VOCAB);
        int CV = vm.compact_vocab;
        printf("Vocab compaction: %d → %d active tokens (%.1fx reduction)\n", VOCAB, CV, (float)VOCAB/CV);

        // Create compact embedding + adam state
        float *cembed = vocab_compact_embed(embed, &vm, DIM);
        float *gcembed = (float*)calloc((size_t)CV*DIM, 4);
        AdamState acembed = adam_alloc((size_t)CV*DIM);

        // ===== Compile all kernels ONCE =====
        printf("Compiling %d dynamic kernels (one-time)...\n", 9);
        uint64_t tc = mach_absolute_time();
        DynLayerKernels dk;
        if (!compile_dynamic_kernels(&dk)) {
            printf("Compilation failed!\n"); return 1;
        }
        double compile_ms = tb_ms(mach_absolute_time() - tc);
        printf("Compiled 9 kernels in %.0fms (shared across all %d layers)\n\n", compile_ms, NLAYERS);

        // Gradient + work buffers
        float *dy = (float*)malloc(SEQ*DIM*4);
        float *dffn = (float*)malloc(SEQ*DIM*4);
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);
        float *dq = (float*)malloc(SEQ*DIM*4);
        float *dk_buf = (float*)malloc(SEQ*DIM*4);
        float *dv = (float*)malloc(SEQ*DIM*4);
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *xnorm_buf = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(SEQ*CV*4);
        float *dlogits = (float*)malloc(SEQ*CV*4);
        float *gate_buf = (float*)malloc(SEQ*HIDDEN*4);
        float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
        float *dsilu = (float*)malloc(SEQ*HIDDEN*4);
        float *silu_tmp = (float*)malloc(SEQ*HIDDEN*4);
        float *silu_tmp2 = (float*)malloc(SEQ*HIDDEN*4);

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        double total_train_ms = 0;
        int total_steps_done = 0;
        uint64_t t_wall_start = mach_absolute_time();
        srand48(42 + start_step);

        for (int step = start_step; step < total_steps; step++) {
            uint64_t t0, t1, t_step = mach_absolute_time();

            // Sample data
            size_t max_pos = n_tokens - SEQ - 1;
            size_t pos = (size_t)(drand48() * max_pos);
            uint16_t *input_tokens = token_data + pos;
            uint16_t *target_tokens_raw = token_data + pos + 1;

            // Map targets to compact vocab IDs
            uint16_t ctargets[SEQ];
            for (int t = 0; t < SEQ; t++) ctargets[t] = (uint16_t)vm.full_to_compact[target_tokens_raw[t]];

            // Embedding lookup (uses full embed for now — input tokens are full IDs)
            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

            // Timing accumulators (reset each step)
            double t_rms=0, t_ane_fwd=0, t_io_fwd=0, t_cblas_wait=0;
            double t_ane_bwd=0, t_io_bwd=0, t_silu=0, t_rms_bwd=0, t_cls=0, t_dw_copy=0;

            // ===== FORWARD (12 layers) =====
            for (int L=0; L<NLAYERS; L++) {
                LayerActs *ac = &acts[L];
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                // RMSNorm1 (CPU)
                t0 = mach_absolute_time();
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);
                memcpy(ac->xnorm, xnorm_buf, SEQ*DIM*4);
                t_rms += tb_ms(mach_absolute_time() - t0);

                // Wait for any pending dW cblas
                t0 = mach_absolute_time();
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                t_cblas_wait += tb_ms(mach_absolute_time() - t0);

                // SDPA forward (ANE): xnorm + Wq,Wk,Wv,Wo → o_out,Q,K,V,attn_out,xnorm
                t0 = mach_absolute_time();
                write_sdpa_fwd_input(&dk, xnorm_buf, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L], Wot_buf[L]);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaFwd);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read output: [1, 6*DIM, 1, SEQ] fp32
                t0 = mach_absolute_time();
                IOSurfaceLock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                float *fwd_out = (float*)IOSurfaceGetBaseAddress(dk.sdpaFwd->ioOut);
                memcpy(ac->o_out,    fwd_out + 0*DIM*SEQ, DIM*SEQ*4);
                memcpy(ac->Q,       fwd_out + 1*DIM*SEQ, DIM*SEQ*4);
                memcpy(ac->K,       fwd_out + 2*DIM*SEQ, DIM*SEQ*4);
                memcpy(ac->V,       fwd_out + 3*DIM*SEQ, DIM*SEQ*4);
                memcpy(ac->attn_out, fwd_out + 4*DIM*SEQ, DIM*SEQ*4);
                IOSurfaceUnlock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // Residual: x2 = x_cur + o_out
                vDSP_vadd(x_cur, 1, ac->o_out, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));

                // RMSNorm2 (CPU)
                t0 = mach_absolute_time();
                rmsnorm(xnorm_buf, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                memcpy(ac->x2norm, xnorm_buf, SEQ*DIM*4);
                t_rms += tb_ms(mach_absolute_time() - t0);

                // FFN W1+W3 (ANE): xnorm → h1, h3, gate
                t0 = mach_absolute_time();
                write_ffn_w13_input(&dk, xnorm_buf, W1t_buf[L], W3t_buf[L]);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.ffnW13);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read h1, h3, gate from output [1, 3*HIDDEN, 1, SEQ]
                t0 = mach_absolute_time();
                IOSurfaceLock(dk.ffnW13->ioOut, kIOSurfaceLockReadOnly, NULL);
                float *ffn13_out = (float*)IOSurfaceGetBaseAddress(dk.ffnW13->ioOut);
                memcpy(ac->h1,       ffn13_out,                   HIDDEN*SEQ*4);
                memcpy(ac->h3,       ffn13_out + HIDDEN*SEQ,      HIDDEN*SEQ*4);
                memcpy(gate_buf,     ffn13_out + 2*HIDDEN*SEQ,    HIDDEN*SEQ*4);
                memcpy(ac->silu_out, gate_buf,                    HIDDEN*SEQ*4);
                IOSurfaceUnlock(dk.ffnW13->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // FFN W2 (ANE): gate @ W2 → ffn_out
                t0 = mach_absolute_time();
                write_ffn_w2_input(&dk, gate_buf, W2t_buf[L]);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.ffnW2);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                t0 = mach_absolute_time();
                IOSurfaceLock(dk.ffnW2->ioOut, kIOSurfaceLockReadOnly, NULL);
                memcpy(ac->ffn_out, (float*)IOSurfaceGetBaseAddress(dk.ffnW2->ioOut), DIM*SEQ*4);
                IOSurfaceUnlock(dk.ffnW2->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // Residual: x_cur = x2 + ffn_out
                vDSP_vadd(ac->x2, 1, ac->ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
            }

            // Final RMSNorm + classifier + loss (CPU)
            t0 = mach_absolute_time();
            rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
            t_rms += tb_ms(mach_absolute_time() - t0);
            t0 = mach_absolute_time();
            // Classifier: logits[CV, SEQ] = cembed[CV, DIM] @ x_final[DIM, SEQ]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        CV, SEQ, DIM, 1.0f, cembed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
            float loss = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);
            last_loss = loss;

            // ===== BACKWARD =====
            // Classifier backward: dy[DIM, SEQ] = cembed^T[DIM, CV] @ dlogits[CV, SEQ]
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        DIM, SEQ, CV, 1.0f, cembed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);

            // dEmbed async: gcembed[CV, DIM] += dlogits[CV, SEQ] @ x_final^T[SEQ, DIM]
            dispatch_group_async(dw_grp, dw_q, ^{
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            CV, DIM, SEQ, 1.0f, dlogits, SEQ, x_final, SEQ, 1.0f, gcembed, DIM);
            });

            // Final RMSNorm backward
            float *dx_rms_final = (float*)calloc(SEQ*DIM, 4);
            rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ);
            memcpy(dy, dx_rms_final, SEQ*DIM*4);
            free(dx_rms_final);

            // ===== BACKWARD (12 layers, reverse) =====
            for (int L=NLAYERS-1; L>=0; L--) {
                LayerActs *ac = &acts[L];
                LayerGrads *gr = &grads[L];
                memcpy(dffn, dy, SEQ*DIM*4);

                // FFN backward: dffn @ W2^T → dsilu_raw
                t0 = mach_absolute_time();
                io_write_dyn(dk.ffnBwdW2t->ioIn, dffn, DIM, SEQ, lw[L].W2, HIDDEN);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.ffnBwdW2t);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.ffnBwdW2t->ioOut, dsilu, HIDDEN, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // SiLU derivative (vectorized): dsilu → dh1, dh3
                // silu(h1) = h1*sig(h1), dsilu_dh1 = sig*(1+h1*(1-sig))
                // dh1 = dsilu * h3 * dsilu_dh1, dh3 = dsilu * silu(h1)
                t0 = mach_absolute_time();
                {
                    int n = HIDDEN*SEQ;
                    // sig = 1/(1+exp(-h1))
                    float minus1 = -1.0f, one = 1.0f;
                    vDSP_vsmul(ac->h1, 1, &minus1, silu_tmp, 1, (vDSP_Length)n);
                    vvexpf(silu_tmp, silu_tmp, &n);
                    vDSP_vsadd(silu_tmp, 1, &one, silu_tmp, 1, (vDSP_Length)n);
                    vvrecf(silu_tmp, silu_tmp, &n);  // silu_tmp = sig
                    // dh3 = dsilu * h1 * sig  (= dsilu * silu(h1))
                    vDSP_vmul(ac->h1, 1, silu_tmp, 1, dh3, 1, (vDSP_Length)n);
                    vDSP_vmul(dsilu, 1, dh3, 1, dh3, 1, (vDSP_Length)n);
                    // dsilu_dh1 = sig*(1+h1*(1-sig)), store in silu_tmp2
                    vDSP_vsadd(silu_tmp, 1, &minus1, silu_tmp2, 1, (vDSP_Length)n); // sig-1
                    vDSP_vneg(silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);          // 1-sig
                    vDSP_vmul(ac->h1, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n); // h1*(1-sig)
                    vDSP_vsadd(silu_tmp2, 1, &one, silu_tmp2, 1, (vDSP_Length)n);  // 1+h1*(1-sig)
                    vDSP_vmul(silu_tmp, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n); // full dsilu_dh1
                    // dh1 = dsilu * h3 * dsilu_dh1
                    vDSP_vmul(dsilu, 1, ac->h3, 1, dh1, 1, (vDSP_Length)n);
                    vDSP_vmul(dh1, 1, silu_tmp2, 1, dh1, 1, (vDSP_Length)n);
                }
                t_silu += tb_ms(mach_absolute_time() - t0);

                // dh1@W1^T + dh3@W3^T → dx_ffn (ANE)
                t0 = mach_absolute_time();
                {
                    IOSurfaceLock(dk.ffnBwdW13t->ioIn, 0, NULL);
                    float *buf = (float*)IOSurfaceGetBaseAddress(dk.ffnBwdW13t->ioIn);
                    int sp = 2*SEQ + 2*DIM;
                    for (int d = 0; d < HIDDEN; d++) {
                        memcpy(buf + d*sp,            dh1 + d*SEQ, SEQ*4);
                        memcpy(buf + d*sp + SEQ,      dh3 + d*SEQ, SEQ*4);
                        memcpy(buf + d*sp + 2*SEQ,        lw[L].W1 + d*DIM, DIM*4);
                        memcpy(buf + d*sp + 2*SEQ + DIM,  lw[L].W3 + d*DIM, DIM*4);
                    }
                    IOSurfaceUnlock(dk.ffnBwdW13t->ioIn, 0, NULL);
                }
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.ffnBwdW13t);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.ffnBwdW13t->ioOut, dx_ffn, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dW FFN async (cblas)
                t0 = mach_absolute_time();
                float *capt_dffn = (float*)malloc(SEQ*DIM*4); memcpy(capt_dffn, dffn, SEQ*DIM*4);
                float *capt_silu = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_silu, ac->silu_out, SEQ*HIDDEN*4);
                float *capt_dh1 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh1, dh1, SEQ*HIDDEN*4);
                float *capt_dh3 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh3, dh3, SEQ*HIDDEN*4);
                float *capt_x2n = (float*)malloc(SEQ*DIM*4); memcpy(capt_x2n, ac->x2norm, SEQ*DIM*4);
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                1.0f, capt_dffn, SEQ, capt_silu, SEQ, 1.0f, gr->W2, HIDDEN);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh1, SEQ, capt_x2n, SEQ, 1.0f, gr->W1, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh3, SEQ, capt_x2n, SEQ, 1.0f, gr->W3, DIM);
                    free(capt_dffn); free(capt_silu); free(capt_dh1); free(capt_dh3); free(capt_x2n);
                });

                // RMSNorm2 backward
                t0 = mach_absolute_time();
                memset(dx2, 0, SEQ*DIM*4);
                rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);

                // Wo^T backward (ANE): dx2 @ Wo^T → da
                t0 = mach_absolute_time();
                io_write_dyn(dk.wotBwd->ioIn, dx2, DIM, SEQ, lw[L].Wo, DIM);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.wotBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                float *da_buf = (float*)malloc(SEQ*DIM*4);
                io_read_dyn(dk.wotBwd->ioOut, da_buf, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dWo async
                t0 = mach_absolute_time();
                float *capt_do = (float*)malloc(SEQ*DIM*4); memcpy(capt_do, dx2, SEQ*DIM*4);
                float *capt_attn = (float*)malloc(SEQ*DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*DIM*4);
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, DIM);
                    free(capt_do); free(capt_attn);
                });

                // SDPA backward part 1 (ANE, fp16): Q,K,V,da → dV,probs,dp
                t0 = mach_absolute_time();
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 0,     ac->Q,  DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, DIM,   ac->K,  DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 2*DIM, ac->V,  DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 3*DIM, da_buf, DIM, SEQ);
                free(da_buf);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd1);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward part 2: probs,dp,Q,K → dQ,dK
                t0 = mach_absolute_time();
                io_copy(dk.sdpaBwd2->ioIn, 0, dk.sdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH,     ac->Q, DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH+DIM, ac->K, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd2);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                t0 = mach_absolute_time();
                io_read_fp16(dk.sdpaBwd2->ioOut, dq, 0,   DIM, SEQ);
                io_read_fp16(dk.sdpaBwd2->ioOut, dk_buf, DIM, DIM, SEQ);
                io_read_fp16(dk.sdpaBwd1->ioOut, dv, 0, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dWq/dWk/dWv async
                t0 = mach_absolute_time();
                float *capt_dq = (float*)malloc(SEQ*DIM*4); memcpy(capt_dq, dq, SEQ*DIM*4);
                float *capt_dk = (float*)malloc(SEQ*DIM*4); memcpy(capt_dk, dk_buf, SEQ*DIM*4);
                float *capt_dv = (float*)malloc(SEQ*DIM*4); memcpy(capt_dv, dv, SEQ*DIM*4);
                float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dk, SEQ, capt_xn, SEQ, 1.0f, gr->Wk, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
                    free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
                });

                // QKV backward (ANE): dq,dk,dv @ Wq^T,Wk^T,Wv^T → dx_attn
                t0 = mach_absolute_time();
                {
                    IOSurfaceLock(dk.qkvBwd->ioIn, 0, NULL);
                    float *buf = (float*)IOSurfaceGetBaseAddress(dk.qkvBwd->ioIn);
                    int sp = 3*SEQ + 3*DIM;
                    for (int d = 0; d < DIM; d++) {
                        memcpy(buf + d*sp,             dq     + d*SEQ, SEQ*4);
                        memcpy(buf + d*sp + SEQ,       dk_buf + d*SEQ, SEQ*4);
                        memcpy(buf + d*sp + 2*SEQ,     dv     + d*SEQ, SEQ*4);
                        memcpy(buf + d*sp + 3*SEQ,         lw[L].Wq + d*DIM, DIM*4);
                        memcpy(buf + d*sp + 3*SEQ+DIM,     lw[L].Wk + d*DIM, DIM*4);
                        memcpy(buf + d*sp + 3*SEQ+2*DIM,   lw[L].Wv + d*DIM, DIM*4);
                    }
                    IOSurfaceUnlock(dk.qkvBwd->ioIn, 0, NULL);
                }
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.qkvBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.qkvBwd->ioOut, dx_attn, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // RMSNorm1 backward
                t0 = mach_absolute_time();
                float *dx_rms1 = (float*)calloc(SEQ*DIM, 4);
                rmsnorm_bwd(dx_rms1, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_rms1[i] + dx2[i];
                free(dx_rms1);
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);
            }

            // Embedding backward
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            embed_backward(gembed, dy, input_tokens, DIM, SEQ);

            double step_ms = tb_ms(mach_absolute_time() - t_step);
            total_train_ms += step_ms;
            total_steps_done++;

            if (step % 10 == 0 || step == start_step) {
                printf("  timing: ane_fwd=%.1f io_fwd=%.1f rms=%.1f ane_bwd=%.1f io_bwd=%.1f silu=%.1f rms_bwd=%.1f cls=%.1f cblas_wait=%.1f dw_copy=%.1f\n",
                       t_ane_fwd, t_io_fwd, t_rms, t_ane_bwd, t_io_bwd, t_silu, t_rms_bwd, t_cls, t_cblas_wait, t_dw_copy);
                float xmx, xmn;
                vDSP_maxv(x_cur,1,&xmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(x_cur,1,&xmn,(vDSP_Length)(SEQ*DIM));
                float dmx, dmn;
                vDSP_maxv(dy,1,&dmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(dy,1,&dmn,(vDSP_Length)(SEQ*DIM));
                printf("step %-4d loss=%.4f  lr=%.2e  %.1fms/step  x[%.2f,%.2f] dy[%.3e,%.3e]\n",
                       step, loss, lr, step_ms, xmn, xmx, dmn, dmx);
            }

            // Adam update every accum_steps
            if ((step+1) % accum_steps == 0 || step == total_steps-1) {
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                float gsc = 1.0f / accum_steps;
                adam_t++;

                // Scale gradients by 1/accum_steps
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    for(size_t i=0;i<WQ_SZ;i++){g->Wq[i]*=gsc;g->Wk[i]*=gsc;g->Wv[i]*=gsc;g->Wo[i]*=gsc;}
                    for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
                    for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
                    for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
                    for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
                }
                for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
                // Merge compact classifier grads into full embed grads
                vocab_scatter_grads(gembed, gcembed, &vm, DIM);
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;

                // Global gradient norm
                float grad_norm_sq = 0;
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    float s;
                    vDSP_dotpr(g->Wq,1,g->Wq,1,&s,(vDSP_Length)WQ_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wk,1,g->Wk,1,&s,(vDSP_Length)WQ_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wv,1,g->Wv,1,&s,(vDSP_Length)WQ_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wo,1,g->Wo,1,&s,(vDSP_Length)WO_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W1,1,g->W1,1,&s,(vDSP_Length)W1_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W2,1,g->W2,1,&s,(vDSP_Length)W2_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W3,1,g->W3,1,&s,(vDSP_Length)W3_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_att,1,g->rms_att,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_ffn,1,g->rms_ffn,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                }
                { float s;
                  vDSP_dotpr(grms_final,1,grms_final,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                  vDSP_dotpr(gembed,1,gembed,1,&s,(vDSP_Length)(VOCAB*DIM)); grad_norm_sq+=s;
                }
                float grad_norm = sqrtf(grad_norm_sq);
                if ((step+1) % 10 == 0) printf("  grad_norm=%.4f\n", grad_norm);

                // Gradient clipping
                if (grad_clip > 0 && grad_norm > grad_clip) {
                    float clip_scale = grad_clip / grad_norm;
                    for (int L=0; L<NLAYERS; L++) {
                        LayerGrads *g = &grads[L];
                        vDSP_vsmul(g->Wq,1,&clip_scale,g->Wq,1,(vDSP_Length)WQ_SZ);
                        vDSP_vsmul(g->Wk,1,&clip_scale,g->Wk,1,(vDSP_Length)WQ_SZ);
                        vDSP_vsmul(g->Wv,1,&clip_scale,g->Wv,1,(vDSP_Length)WQ_SZ);
                        vDSP_vsmul(g->Wo,1,&clip_scale,g->Wo,1,(vDSP_Length)WO_SZ);
                        vDSP_vsmul(g->W1,1,&clip_scale,g->W1,1,(vDSP_Length)W1_SZ);
                        vDSP_vsmul(g->W2,1,&clip_scale,g->W2,1,(vDSP_Length)W2_SZ);
                        vDSP_vsmul(g->W3,1,&clip_scale,g->W3,1,(vDSP_Length)W3_SZ);
                        vDSP_vsmul(g->rms_att,1,&clip_scale,g->rms_att,1,(vDSP_Length)DIM);
                        vDSP_vsmul(g->rms_ffn,1,&clip_scale,g->rms_ffn,1,(vDSP_Length)DIM);
                    }
                    vDSP_vsmul(grms_final,1,&clip_scale,grms_final,1,(vDSP_Length)DIM);
                    vDSP_vsmul(gembed,1,&clip_scale,gembed,1,(vDSP_Length)(VOCAB*DIM));
                }

                // Cosine LR schedule with warmup
                if (step < warmup_steps) {
                    lr = max_lr * ((float)(step + 1)) / warmup_steps;
                } else {
                    float decay_ratio = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
                    float min_lr = max_lr * min_lr_frac;
                    lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay_ratio)) * (max_lr - min_lr);
                }

                // Adam update
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps);
                    adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps);

                    // Update transposed weight buffers
                    transpose_weight(Wqt_buf[L], lw[L].Wq, DIM, DIM);
                    transpose_weight(Wkt_buf[L], lw[L].Wk, DIM, DIM);
                    transpose_weight(Wvt_buf[L], lw[L].Wv, DIM, DIM);
                    transpose_weight(Wot_buf[L], lw[L].Wo, DIM, DIM);
                    transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
                    transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
                    transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
                }
                adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps);
                // Re-extract compact embed from updated full embed
                free(cembed);
                cembed = vocab_compact_embed(embed, &vm, DIM);

                // Zero grads
                for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
                memset(grms_final, 0, DIM*4);
                memset(gembed, 0, (size_t)VOCAB*DIM*4);
                memset(gcembed, 0, (size_t)CV*DIM*4);

                // Checkpoint
                if ((step+1) % 100 == 0) {
                    double wall = tb_ms(mach_absolute_time() - t_wall_start);
                    save_checkpoint(CKPT_PATH, step+1, total_steps, lr, last_loss,
                        total_train_ms+cum_train, wall+cum_wall, total_steps_done+cum_steps, adam_t,
                        lw, la, rms_final, &arms_final, embed, &aembed);
                }
            }
        }

        // Report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:  %d\n", total_steps_done);
        printf("Compile:      %.0fms (one-time, %.1f%%)\n", compile_ms, 100*compile_ms/(wall+cum_wall));
        printf("Train time:   %.0fms (%.1fms/step)\n", total_train_ms, total_train_ms/total_steps_done);
        printf("Wall time:    %.1fs\n", (wall+cum_wall)/1000);

        // Cleanup
        for (int L=0; L<NLAYERS; L++) {
            layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W2t_buf[L]); free(W3t_buf[L]);
        }
        free_kern(dk.sdpaFwd); free_kern(dk.ffnW13); free_kern(dk.ffnW2);
        free_kern(dk.ffnBwdW2t); free_kern(dk.ffnBwdW13t); free_kern(dk.wotBwd);
        free_kern(dk.sdpaBwd1); free_kern(dk.sdpaBwd2); free_kern(dk.qkvBwd);
        munmap(token_data, data_len);
    }
    return 0;
}
