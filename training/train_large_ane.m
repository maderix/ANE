// train_large_ane.m — Stories110M training with CPU ops offloaded to ANE
// Based on train_large.m but moves these operations from CPU to ANE:
//   1. Final RMSNorm (was CPU vDSP) → ANE kernel
//   2. Classifier forward embed@x (was CPU cblas) → ANE 32000-ch conv
//   3. Cross-entropy softmax (was CPU vDSP) → ANE softmax kernel
//   4. RMSNorm backward (was CPU vDSP) → ANE kernel
// Still on CPU: dW gradients (parallel via GCD), Adam optimizer (needs weight mutation),
//               classifier backward (ANE matmul slower than cblas for this shape),
//               NLL loss + gradient (needs target indexing)
//
// Build: make train_large_ane
// Run:   ./train_large_ane [--resume] [--steps N] [--lr F] [--data PATH]
#include "stories_io.h"
#include "stories_mil.h"
#include "stories_cpu_ops.h"
#include "ane_rmsnorm_bwd.h"
#include "ane_classifier.h"

#define CKPT_PATH_DEFAULT "ane_stories110M_ckpt.bin"
#define MODEL_PATH_DEFAULT "stories110M.bin"
#define DATA_PATH_DEFAULT "tinystories_data00.bin"

// ===== Weight loading from llama2.c format =====
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return false; }
    Llama2Config cfg;
    fread(&cfg, sizeof(cfg), 1, f);
    printf("  Model config: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, abs(cfg.vocab_size), cfg.seq_len);
    if (cfg.dim != DIM || cfg.hidden_dim != HIDDEN || cfg.n_layers != NLAYERS) {
        printf("  ERROR: Config mismatch!\n"); fclose(f); return false;
    }
    int V = abs(cfg.vocab_size);
    bool shared = cfg.vocab_size > 0;
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

// NEW: Compile RMSNorm backward kernels (one per layer pair: attn + ffn)
static Kern *compile_rmsnorm_bwd_kern(const float *rms_w) {
    return compile_kern_mil_w(gen_rmsnorm_bwd(), (@{
        @"@model_path/weights/rms_w.bin": @{@"offset":@0, @"data":build_blob(rms_w, 1, DIM)},
    }), 2*DIM*SEQ*2, DIM*SEQ*2);
}

// NEW: Compile classifier forward kernel
static Kern *compile_classifier_fwd(const float *embed) {
    return compile_kern_mil_w(gen_classifier_fwd(), (@{
        @"@model_path/weights/embed.bin": @{@"offset":@0, @"data":build_blob(embed, VOCAB, DIM)},
    }), DIM*SEQ*2, VOCAB*SEQ*2);
}

// NEW: Compile final RMSNorm kernel
static Kern *compile_final_rmsnorm_kern(const float *rms_w) {
    return compile_kern_mil_w(gen_final_rmsnorm(), (@{
        @"@model_path/weights/rms_w.bin": @{@"offset":@0, @"data":build_blob(rms_w, 1, DIM)},
    }), DIM*SEQ*2, DIM*SEQ*2);
}

// NEW: Compile softmax kernel (no weights)
static Kern *compile_softmax_kern(void) {
    return compile_kern_mil_w(gen_softmax_vocab(), @{}, VOCAB*SEQ*2, VOCAB*SEQ*2);
}

static void free_layer_kernels(LayerKernels *lk) {
    free_kern(lk->fwdAttn); free_kern(lk->fwdFFN); free_kern(lk->ffnBwd);
    free_kern(lk->sdpaBwd1); free_kern(lk->qkvBwd);
    lk->fwdAttn = lk->fwdFFN = lk->ffnBwd = lk->sdpaBwd1 = lk->qkvBwd = NULL;
}

// ===== Checkpoint save/load (same as train_large.m) =====
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
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float lr = 3e-4f;
        float adam_b1=0.9f, adam_b2=0.999f, adam_eps=1e-8f;
        int adam_t = 0, start_step = 0;
        const char *ckpt_path = CKPT_PATH_DEFAULT;
        const char *model_path = MODEL_PATH_DEFAULT;
        const char *data_path = DATA_PATH_DEFAULT;
        bool do_resume = false;
        bool ane_extras = true;  // classifier, softmax, rmsnorm_bwd on ANE
        int pos = 0;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--no-ane-extras") == 0) ane_extras = false;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--ckpt") == 0 && i+1<argc) ckpt_path = argv[++i];
            else if (strcmp(argv[i], "--model") == 0 && i+1<argc) model_path = argv[++i];
            else if (strcmp(argv[i], "--data") == 0 && i+1<argc) data_path = argv[++i];
            else if (argv[i][0] != '-') {
                if (pos == 0) model_path = argv[i];
                else if (pos == 1) { /* seq - compile-time constant */ }
                else if (pos == 2) total_steps = atoi(argv[i]);
                else if (pos == 3) lr = atof(argv[i]);
                pos++;
            }
        }

        LayerWeights lw[NLAYERS]; LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS]; LayerGrads grads[NLAYERS]; LayerKernels kern[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc(); la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc(); grads[L] = layer_grads_alloc();
            memset(&kern[L], 0, sizeof(LayerKernels));
        }
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed = (float*)calloc(VOCAB*DIM, 4);
        AdamState arms_final = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);
        double cum_compile=0, cum_train=0, cum_wall=0;
        int cum_steps=0, cum_batches=0;

        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            resuming = load_checkpoint(ckpt_path, &start_step, &total_steps, &lr, &resume_loss,
                &cum_compile, &cum_train, &cum_wall, &cum_steps, &cum_batches, &adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            if (resuming) printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
        }
        if (!resuming) {
            printf("=== ANE Training: Stories110M (ANE-offloaded) ===\n");
            printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n", DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS);
            if (ane_extras) printf("NEW: final_rmsnorm, classifier_fwd, softmax, rmsnorm_bwd on ANE\n");
            else printf("ANE extras DISABLED (classifier/softmax/rmsnorm_bwd on CPU)\n");
            {
                printf("  Training from scratch (random init)\n");
                srand48(42);
                float scale_d=1.0f/sqrtf(DIM), scale_h=1.0f/sqrtf(HIDDEN);
                float res_scale = 1.0f/sqrtf(2.0f*NLAYERS); // LLaMA-style output proj scaling
                for (int L=0; L<NLAYERS; L++) {
                    for(size_t i=0;i<WQ_SZ;i++){lw[L].Wq[i]=scale_d*(2*drand48()-1);lw[L].Wk[i]=scale_d*(2*drand48()-1);}
                    for(size_t i=0;i<WQ_SZ;i++){lw[L].Wv[i]=scale_d*(2*drand48()-1);lw[L].Wo[i]=scale_d*res_scale*(2*drand48()-1);}
                    for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                    for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                    for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f; lw[L].rms_ffn[i]=1.0f;}
                }
                for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
                float escale = 0.02f;
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
            }
        }

        // mmap token data
        int data_fd = open(data_path, O_RDONLY);
        if (data_fd < 0) {
            printf("Cannot open token data: %s\n", data_path);
            printf("Hint: run `bash download_data.sh` in training/ or pass --data /path/to/tinystories_data00.bin\n");
            return 1;
        }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // Gradient buffers
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
        float *probs = (float*)malloc(SEQ*VOCAB*4);   // NEW: for ANE softmax output

        // Compile static sdpaBwd2 kernels
        Kern *sdpaBwd2[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            sdpaBwd2[L] = compile_sdpa_bwd2();
            if (!sdpaBwd2[L]) { printf("sdpaBwd2 compile failed\n"); return 1; }
        }

        // NEW: Compile ANE-offloaded kernels (static — no per-batch recompile needed)
        // These have no weight-bearing or static weights that don't change per batch

        // RMSNorm backward kernels — one per layer for attn and ffn
        // These DO have baked weights (rms_att, rms_ffn) so they need recompile per batch
        // But they're small weights, and we compile them alongside the layer kernels
        Kern *rmsAttBwd[NLAYERS], *rmsFFNBwd[NLAYERS];
        memset(rmsAttBwd, 0, sizeof(rmsAttBwd));
        memset(rmsFFNBwd, 0, sizeof(rmsFFNBwd));

        // Softmax kernel (no weights — compile once)
        Kern *softmaxKern = NULL;
        if (ane_extras) {
            softmaxKern = compile_softmax_kern();
            if (!softmaxKern) { printf("softmax compile failed\n"); return 1; }
            printf("Softmax kernel compiled (no weights)\n");
        }

        // Final RMSNorm and classifier are recompiled per batch since they have baked weights
        Kern *finalRmsKern = NULL, *classifierKern = NULL;

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        double total_compile_ms=0, total_train_ms=0;
        int total_steps_done=0, total_batches=0;
        uint64_t t_wall_start = mach_absolute_time();
        srand48(42 + start_step);

        int step = start_step;
        while (step < total_steps) {
            // Check compile budget — account for new kernels
            // Per batch: 60 layer kernels [+ 24 rmsnorm_bwd + 1 classifier + 1 final_rms = 86 with extras]
            int kernels_needed = TOTAL_WEIGHT_KERNELS + (ane_extras ? 2*NLAYERS + 2 : 0);
            if (g_compile_count + kernels_needed > MAX_COMPILES) {
                for (int L=0; L<NLAYERS; L++) {
                    free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]);
                    free_kern(rmsAttBwd[L]); free_kern(rmsFFNBwd[L]);
                }
                free_kern(softmaxKern); free_kern(finalRmsKern); free_kern(classifierKern);
                double wall = tb_ms(mach_absolute_time() - t_wall_start);
                save_checkpoint(ckpt_path, step, total_steps, lr, last_loss,
                    total_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
                    total_steps_done+cum_steps, total_batches+cum_batches, adam_t,
                    lw, la, rms_final, &arms_final, embed, &aembed);
                printf("[exec() restart step %d, %d compiles, loss=%.4f]\n", step, g_compile_count, last_loss);
                fflush(stdout);
                if (ane_extras)
                    execl(argv[0], argv[0], "--resume", "--ckpt", ckpt_path, "--data", data_path, NULL);
                else
                    execl(argv[0], argv[0], "--resume", "--ckpt", ckpt_path, "--data", data_path, "--no-ane-extras", NULL);
                perror("execl"); return 1;
            }

            // Compile all layer kernels
            uint64_t tc = mach_absolute_time();
            for (int L=0; L<NLAYERS; L++) free_layer_kernels(&kern[L]);
            bool compile_ok = true;
            for (int L=0; L<NLAYERS; L++) {
                printf("  Compiling layer %d/%d... (%d compiles)\r", L+1, NLAYERS, g_compile_count);
                fflush(stdout);
                if (!compile_layer_kernels(&kern[L], &lw[L])) {
                    printf("\nCompile failed at layer %d\n", L);
                    compile_ok = false; break;
                }
                // Compile RMSNorm backward kernels for this layer (if ane_extras)
                if (ane_extras) {
                    free_kern(rmsAttBwd[L]); free_kern(rmsFFNBwd[L]);
                    rmsAttBwd[L] = compile_rmsnorm_bwd_kern(lw[L].rms_att);
                    rmsFFNBwd[L] = compile_rmsnorm_bwd_kern(lw[L].rms_ffn);
                    if (!rmsAttBwd[L] || !rmsFFNBwd[L]) {
                        printf("\nrmsnorm_bwd compile failed at layer %d\n", L);
                        compile_ok = false; break;
                    }
                }
            }
            if (!compile_ok) { g_compile_count = MAX_COMPILES; continue; }

            // Re-compile sdpaBwd2 if needed
            for (int L=0; L<NLAYERS; L++) {
                if (!sdpaBwd2[L]) {
                    sdpaBwd2[L] = compile_sdpa_bwd2();
                    if (!sdpaBwd2[L]) { printf("sdpaBwd2 recompile failed\n"); return 1; }
                }
            }

            // Compile final RMSNorm and classifier with current weights (if ane_extras)
            if (ane_extras) {
                free_kern(finalRmsKern); free_kern(classifierKern);
                finalRmsKern = compile_final_rmsnorm_kern(rms_final);
                classifierKern = compile_classifier_fwd(embed);
                if (!finalRmsKern || !classifierKern) {
                    printf("finalRms or classifier compile failed\n");
                    g_compile_count = MAX_COMPILES; continue;
                }
                if (!softmaxKern) {
                    softmaxKern = compile_softmax_kern();
                    if (!softmaxKern) { printf("softmax recompile failed\n"); return 1; }
                }
            }

            double cms = tb_ms(mach_absolute_time() - tc);
            total_compile_ms += cms;
            printf("  Compiled %d kernels in %.0fms                    \n", kernels_needed, cms);

            // Zero gradient accumulators
            for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
            memset(grms_final, 0, DIM*4);
            memset(gembed, 0, (size_t)VOCAB*DIM*4);

            int steps_batch = 0;
            uint64_t tt = mach_absolute_time();
            double t_ane=0,t_io=0,t_elem=0,t_rms=0,t_cblas_wait=0,t_cls=0;

            for (int a=0; a<ACCUM_STEPS && step<total_steps; a++, step++) {
                uint64_t t0,t1;
                size_t max_pos = n_tokens - SEQ - 1;
                size_t pos = (size_t)(drand48() * max_pos);
                uint16_t *input_tokens = token_data + pos;
                uint16_t *target_tokens = token_data + pos + 1;

                // Embedding lookup
                t0=mach_absolute_time();
                embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);
                t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0);

                // ===== FORWARD (12 layers) =====
                for (int L=0; L<NLAYERS; L++) {
                    LayerActs *ac = &acts[L];
                    memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                    t0=mach_absolute_time();
                    dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                    t1=mach_absolute_time(); t_cblas_wait+=tb_ms(t1-t0); t0=t1;

                    io_write_fp16(kern[L].fwdAttn->ioIn, x_cur, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].fwdAttn);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(kern[L].fwdAttn->ioOut, ac->o_out, 0, DIM, SEQ);
                    io_read_fp16(kern[L].fwdAttn->ioOut, ac->attn_out, 4*DIM, DIM, SEQ);
                    io_read_fp16(kern[L].fwdAttn->ioOut, ac->xnorm, 5*DIM, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                    vDSP_vadd(x_cur, 1, ac->o_out, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
                    t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0); t0=t1;

                    io_write_fp16(kern[L].fwdFFN->ioIn, ac->x2, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                    ane_eval(kern[L].fwdFFN);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->ffn_out, 0, DIM, SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->h1, DIM, HIDDEN, SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->h3, DIM+HIDDEN, HIDDEN, SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->silu_out, DIM+2*HIDDEN, HIDDEN, SEQ);
                    io_read_fp16(kern[L].fwdFFN->ioOut, ac->x2norm, DIM+3*HIDDEN, DIM, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                    vDSP_vadd(ac->x2, 1, ac->ffn_out, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                    t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0);
                }

                t0=mach_absolute_time();
                if (ane_extras) {
                    // Final RMSNorm on ANE
                    io_write_fp16(finalRmsKern->ioIn, x_cur, DIM, SEQ);
                    ane_eval(finalRmsKern);
                    io_read_fp16(finalRmsKern->ioOut, x_final, 0, DIM, SEQ);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;

                    // Classifier on ANE
                    io_write_fp16(classifierKern->ioIn, x_final, DIM, SEQ);
                    ane_eval(classifierKern);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;

                    // Softmax on ANE
                    io_copy(softmaxKern->ioIn, 0, classifierKern->ioOut, 0, VOCAB, SEQ);
                    ane_eval(softmaxKern);
                    t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;

                    io_read_fp16(softmaxKern->ioOut, probs, 0, VOCAB, SEQ);
                    t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                } else {
                    // CPU fallback: rmsnorm + classifier + softmax
                    rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
                    t1=mach_absolute_time(); t_rms+=tb_ms(t1-t0); t0=t1;

                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                VOCAB, SEQ, DIM, 1.0f,
                                embed, DIM, x_final, SEQ, 0.0f, probs, SEQ);
                    t1=mach_absolute_time(); t_cls+=tb_ms(t1-t0); t0=t1;

                    // CPU softmax
                    for (int t = 0; t < SEQ; t++) {
                        float maxv = -1e30f;
                        for (int v = 0; v < VOCAB; v++) { float val = probs[v*SEQ+t]; if (val > maxv) maxv = val; }
                        float sum = 0;
                        for (int v = 0; v < VOCAB; v++) { probs[v*SEQ+t] = expf(probs[v*SEQ+t] - maxv); sum += probs[v*SEQ+t]; }
                        for (int v = 0; v < VOCAB; v++) probs[v*SEQ+t] /= sum;
                    }
                    t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0); t0=t1;
                }

                // NLL loss + gradient on CPU: dlogits = probs - one_hot(targets)
                float total_loss = 0;
                float invS = 1.0f / SEQ;
                memcpy(dlogits, probs, (size_t)VOCAB*SEQ*4);
                for (int t = 0; t < SEQ; t++) {
                    int tgt = target_tokens[t];
                    total_loss -= logf(probs[tgt*SEQ+t] + 1e-10f);
                    dlogits[tgt*SEQ+t] -= 1.0f;  // subtract one_hot
                }
                // Scale by 1/S
                vDSP_vsmul(dlogits, 1, &invS, dlogits, 1, (vDSP_Length)((size_t)VOCAB*SEQ));
                float loss = total_loss / SEQ;
                last_loss = loss;
                t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0); t0=t1;

                // ===== BACKWARD =====
                // Classifier backward: dx_final = embed^T @ dlogits (CPU — ANE is slower)
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            DIM, SEQ, VOCAB, 1.0f,
                            embed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);
                // dembed async on CPU
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                VOCAB, DIM, SEQ, 1.0f,
                                dlogits, SEQ, x_final, SEQ, 1.0f, gembed, DIM);
                });

                // Final RMSNorm backward (CPU — just one call, not worth ANE overhead)
                {
                    float *dx_rms_final = (float*)calloc(SEQ*DIM, 4);
                    rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ);
                    memcpy(dy, dx_rms_final, SEQ*DIM*4);
                    free(dx_rms_final);
                }
                t1=mach_absolute_time(); t_rms+=tb_ms(t1-t0);

                // ===== BACKWARD (12 layers, reverse) =====
                for (int L=NLAYERS-1; L>=0; L--) {
                    LayerActs *ac = &acts[L];
                    LayerGrads *gr = &grads[L];
                    memcpy(dffn, dy, SEQ*DIM*4);

                    // FFN backward (ANE) — same as original
                    io_write_fp16_at(kern[L].ffnBwd->ioIn, 0, dffn, DIM, SEQ);
                    io_copy(kern[L].ffnBwd->ioIn, DIM, kern[L].fwdFFN->ioOut, DIM, 2*HIDDEN, SEQ);
                    ane_eval(kern[L].ffnBwd);
                    io_read_fp16(kern[L].ffnBwd->ioOut, dx_ffn, 0, DIM, SEQ);
                    io_read_fp16(kern[L].ffnBwd->ioOut, dh1, DIM, HIDDEN, SEQ);
                    io_read_fp16(kern[L].ffnBwd->ioOut, dh3, DIM+HIDDEN, HIDDEN, SEQ);

                    // dW FFN async (CPU — parallel with ANE)
                    float *capt_dffn = (float*)malloc(SEQ*DIM*4); memcpy(capt_dffn, dffn, SEQ*DIM*4);
                    float *capt_silu = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_silu, ac->silu_out, SEQ*HIDDEN*4);
                    float *capt_dh1 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh1, dh1, SEQ*HIDDEN*4);
                    float *capt_dh3 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh3, dh3, SEQ*HIDDEN*4);
                    float *capt_x2n = (float*)malloc(SEQ*DIM*4); memcpy(capt_x2n, ac->x2norm, SEQ*DIM*4);
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
                    if (ane_extras) {
                        io_write_fp16_at(rmsFFNBwd[L]->ioIn, 0, dx_ffn, DIM, SEQ);
                        io_write_fp16_at(rmsFFNBwd[L]->ioIn, DIM, ac->x2, DIM, SEQ);
                        ane_eval(rmsFFNBwd[L]);
                        io_read_fp16(rmsFFNBwd[L]->ioOut, dx2, 0, DIM, SEQ);
                    }
                    // dw for rmsnorm_ffn on CPU (accumulate per step)
                    {
                        float *dw_tmp = (float*)calloc(DIM, 4);
                        float *dx_scratch = (float*)malloc(SEQ*DIM*4);
                        rmsnorm_bwd(dx_scratch, dw_tmp, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                        if (!ane_extras) memcpy(dx2, dx_scratch, SEQ*DIM*4);
                        for(int i=0;i<DIM;i++) gr->rms_ffn[i] += dw_tmp[i];
                        free(dx_scratch); free(dw_tmp);
                    }
                    // Add residual: dx2 += dy
                    for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];

                    // dWo async (CPU)
                    memcpy(do_out_buf, dx2, SEQ*DIM*4);
                    float *capt_do = (float*)malloc(SEQ*DIM*4); memcpy(capt_do, do_out_buf, SEQ*DIM*4);
                    float *capt_attn = (float*)malloc(SEQ*DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*DIM*4);
                    dispatch_group_async(dw_grp, dw_q, ^{
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, DIM);
                        free(capt_do); free(capt_attn);
                    });

                    // SDPA backward (ANE) — same as original
                    io_copy(kern[L].sdpaBwd1->ioIn, 0, kern[L].fwdAttn->ioOut, DIM, 3*DIM, SEQ);
                    io_write_fp16_at(kern[L].sdpaBwd1->ioIn, 3*DIM, dx2, DIM, SEQ);
                    ane_eval(kern[L].sdpaBwd1);
                    io_copy(sdpaBwd2[L]->ioIn, 0, kern[L].sdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ);
                    io_copy(sdpaBwd2[L]->ioIn, 2*SCORE_CH, kern[L].fwdAttn->ioOut, DIM, 2*DIM, SEQ);
                    ane_eval(sdpaBwd2[L]);

                    io_read_fp16(sdpaBwd2[L]->ioOut, dq, 0, DIM, SEQ);
                    io_read_fp16(sdpaBwd2[L]->ioOut, dk, DIM, DIM, SEQ);
                    io_read_fp16(kern[L].sdpaBwd1->ioOut, dv, 0, DIM, SEQ);

                    // dWq/dWk/dWv async (CPU)
                    float *capt_dq = (float*)malloc(SEQ*DIM*4); memcpy(capt_dq, dq, SEQ*DIM*4);
                    float *capt_dk = (float*)malloc(SEQ*DIM*4); memcpy(capt_dk, dk, SEQ*DIM*4);
                    float *capt_dv = (float*)malloc(SEQ*DIM*4); memcpy(capt_dv, dv, SEQ*DIM*4);
                    float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
                    dispatch_group_async(dw_grp, dw_q, ^{
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, capt_dk, SEQ, capt_xn, SEQ, 1.0f, gr->Wk, DIM);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                    1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
                        free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
                    });

                    // QKV backward (ANE) — same as original
                    io_copy(kern[L].qkvBwd->ioIn, 0, sdpaBwd2[L]->ioOut, 0, 2*DIM, SEQ);
                    io_copy(kern[L].qkvBwd->ioIn, 2*DIM, kern[L].sdpaBwd1->ioOut, 0, DIM, SEQ);
                    ane_eval(kern[L].qkvBwd);
                    io_read_fp16(kern[L].qkvBwd->ioOut, dx_attn, 0, DIM, SEQ);

                    // RMSNorm1 backward
                    float *dx_rms1 = (float*)malloc(SEQ*DIM*4);
                    if (ane_extras) {
                        io_write_fp16_at(rmsAttBwd[L]->ioIn, 0, dx_attn, DIM, SEQ);
                        io_write_fp16_at(rmsAttBwd[L]->ioIn, DIM, ac->layer_in, DIM, SEQ);
                        ane_eval(rmsAttBwd[L]);
                        io_read_fp16(rmsAttBwd[L]->ioOut, dx_rms1, 0, DIM, SEQ);
                    }
                    // dw for rmsnorm_att on CPU
                    {
                        float *dw_tmp = (float*)calloc(DIM, 4);
                        float *dx_scratch = (float*)malloc(SEQ*DIM*4);
                        rmsnorm_bwd(dx_scratch, dw_tmp, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                        if (!ane_extras) memcpy(dx_rms1, dx_scratch, SEQ*DIM*4);
                        for(int i=0;i<DIM;i++) gr->rms_att[i] += dw_tmp[i];
                        free(dx_scratch); free(dw_tmp);
                    }

                    for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_rms1[i] + dx2[i];
                    free(dx_rms1);
                }

                // Embedding backward
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                embed_backward(gembed, dy, input_tokens, DIM, SEQ);

                steps_batch++;
                if (step % 10 == 0 || step == start_step)
                    printf("step %-4d loss=%.4f\n", step, loss);
            }
            double tms = tb_ms(mach_absolute_time() - tt);
            total_train_ms += tms;
            total_steps_done += steps_batch;
            total_batches++;

            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

            // Adam update
            float gsc = 1.0f / steps_batch;
            adam_t++;
            for (int L=0; L<NLAYERS; L++) {
                LayerGrads *g = &grads[L];
                for(size_t i=0;i<WQ_SZ;i++){g->Wq[i]*=gsc;g->Wk[i]*=gsc;g->Wv[i]*=gsc;g->Wo[i]*=gsc;}
                for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
                for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
                for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
                for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
                adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps);
                adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps);
            }
            for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
            adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps);
            for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;
            adam_update(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps);

            printf("  [batch %d: compile=%.0fms train=%.1fms (%.1fms/step) compiles=%d]\n",
                   steps_batch, cms, tms, tms/steps_batch, g_compile_count);
        }

        // Efficiency report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        total_compile_ms += cum_compile; total_train_ms += cum_train;
        wall += cum_wall; total_steps_done += cum_steps; total_batches += cum_batches;

        // FLOP accounting — same as train_large.m but classifier+softmax now on ANE
        double fwd_flops = NLAYERS * (4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
        double sdpa_flops = NLAYERS * 2.0*HEADS*5*SEQ*SEQ*HD;
        double cls_flops = 2.0*VOCAB*DIM*SEQ;
        double total_flops = (fwd_flops*3 + sdpa_flops + cls_flops*3) * total_steps_done;
        // In train_large_ane: classifier fwd + softmax run on ANE (not CPU)
        double ane_flops = (fwd_flops*2 + sdpa_flops + cls_flops) * total_steps_done;

        printf("\n=== NEW Efficiency Report ===\n");
        printf("Total steps:     %d\n", total_steps_done);
        printf("Wall time:       %.0f ms (%.1f s)\n", wall, wall/1000);
        printf("Compile time:    %.0f ms (%.1f%%)\n", total_compile_ms, 100*total_compile_ms/wall);
        printf("Train time:      %.0f ms (%.1f%%)\n", total_train_ms, 100*total_train_ms/wall);
        printf("Avg train:       %.1f ms/step\n", total_train_ms/total_steps_done);
        printf("ANE TFLOPS:      %.2f sustained\n", ane_flops / (total_train_ms * 1e9));
        printf("Total TFLOPS:    %.2f (ANE+CPU)\n", total_flops / (total_train_ms * 1e9));
        printf("ANE utilization: %.1f%% of 15.8 TFLOPS\n", 100*ane_flops/(total_train_ms*1e9)/15.8);
        // Cleanup
        for (int L=0; L<NLAYERS; L++) {
            free_layer_kernels(&kern[L]); free_kern(sdpaBwd2[L]);
            free_kern(rmsAttBwd[L]); free_kern(rmsFFNBwd[L]);
            layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
        }
        free_kern(softmaxKern); free_kern(finalRmsKern); free_kern(classifierKern);
        munmap(token_data, data_len); close(data_fd);
        free(rms_final); free(embed); free(grms_final); free(gembed);
        adam_free(&arms_final); adam_free(&aembed);
        free(dy); free(dffn); free(dh1); free(dh3); free(dx_ffn); free(dx2);
        free(do_out_buf); free(dq); free(dk); free(dv); free(dx_attn);
        free(x_cur); free(x_final); free(logits); free(dlogits); free(probs);
    }
    return 0;
}
