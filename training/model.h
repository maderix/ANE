// model.h — Stories110M model struct + weight loading + ANE kernel compilation
// Training version: baked-weight conv kernels, recompile when weights update
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ane_runtime.h"
#include "ane_mil_gen.h"

#define N_LAYERS 12
#define DIM 768
#define HIDDEN_DIM 2048
#define N_HEADS 12
#define HEAD_DIM 64
#define VOCAB_SIZE 32000
#define MAX_SEQ 1024

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
    Config cfg;
    int seq_len; // training sequence length

    // Raw weights (f32)
    float *token_embedding;  // [vocab_size, dim]
    float *rms_att_w[N_LAYERS]; // [dim]
    float *wq[N_LAYERS];       // [dim, dim]
    float *wk[N_LAYERS];       // [dim, dim]
    float *wv[N_LAYERS];       // [dim, dim]
    float *wo[N_LAYERS];       // [dim, dim]
    float *rms_ffn_w[N_LAYERS]; // [dim]
    float *w1[N_LAYERS];       // [hidden_dim, dim]
    float *w2[N_LAYERS];       // [dim, hidden_dim]
    float *w3[N_LAYERS];       // [hidden_dim, dim]
    float *rms_final_w;        // [dim]
    float *wcls;               // [vocab_size, dim]

    // Per-layer ANE conv kernels (baked weights, recompiled on update)
    ANEKernel *kern_q[N_LAYERS];   // Q projection: dim→dim
    ANEKernel *kern_k[N_LAYERS];   // K projection: dim→dim
    ANEKernel *kern_v[N_LAYERS];   // V projection: dim→dim
    ANEKernel *kern_o[N_LAYERS];   // O projection: dim→dim
    ANEKernel *kern_w1[N_LAYERS];  // FFN w1: dim→hidden
    ANEKernel *kern_w2[N_LAYERS];  // FFN w2: hidden→dim
    ANEKernel *kern_w3[N_LAYERS];  // FFN w3: dim→hidden
    ANEKernel *kern_cls;           // Classifier: dim→vocab

    // Gradient accumulators (f32)
    float *grad_wq[N_LAYERS], *grad_wk[N_LAYERS], *grad_wv[N_LAYERS], *grad_wo[N_LAYERS];
    float *grad_w1[N_LAYERS], *grad_w2[N_LAYERS], *grad_w3[N_LAYERS];
    float *grad_wcls;
    float *grad_emb;

    // Adam optimizer state
    float *adam_m, *adam_v;
    int adam_step;
    size_t total_params;

    // Activation cache for backward
    float *act_x[N_LAYERS];
    float *act_xnorm[N_LAYERS];
    float *act_q[N_LAYERS];
    float *act_k[N_LAYERS];
    float *act_v[N_LAYERS];
    float *act_attn_out[N_LAYERS];
    float *act_ffn_in[N_LAYERS];
    float *act_h1[N_LAYERS];
    float *act_h3[N_LAYERS];
    float *act_silu[N_LAYERS];
    float *act_final;
    float *act_pre_final;
    float *logits;
} Model;

static int model_load_weights(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    if (fread(&m->cfg, sizeof(Config), 1, f) != 1) {
        fprintf(stderr, "ERROR: failed to read config from %s\n", path);
        fclose(f); return -1;
    }

    if (m->cfg.n_layers < 1 || m->cfg.n_layers > N_LAYERS) {
        fprintf(stderr, "ERROR: n_layers (%d) exceeds maximum allowed (%d)\n", m->cfg.n_layers, N_LAYERS);
        fclose(f); return -1;
    }

    if (m->cfg.dim < 1 || m->cfg.dim > 8192 ||
        m->cfg.hidden_dim < 1 || m->cfg.hidden_dim > 32768) {
        fprintf(stderr, "ERROR: model dimensions out of safe bounds\n");
        fclose(f); return -1;
    }

    bool shared = m->cfg.vocab_size > 0;
    if (m->cfg.vocab_size < 0) m->cfg.vocab_size = -m->cfg.vocab_size;

    if (m->cfg.vocab_size == 0 || m->cfg.vocab_size > 256000) {
        fprintf(stderr, "ERROR: vocab_size out of safe bounds\n");
        fclose(f); return -1;
    }

    printf("Model: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
           m->cfg.dim, m->cfg.hidden_dim, m->cfg.n_layers, m->cfg.n_heads,
           m->cfg.vocab_size, m->cfg.seq_len);

    size_t d = (size_t)m->cfg.dim, hd = (size_t)m->cfg.hidden_dim, nl = (size_t)m->cfg.n_layers, vs = (size_t)m->cfg.vocab_size;

    m->token_embedding = (float*)malloc(vs * d * sizeof(float));
    if (!m->token_embedding) {
        fprintf(stderr, "ERROR: OOM allocating token_embedding\n");
        fclose(f); return -1;
    }
    if (fread(m->token_embedding, sizeof(float), vs * d, f) != (vs * d)) {
        fprintf(stderr, "ERROR: short read on token_embedding (file truncated?)\n");
        fclose(f); return -1;
    }

    float *rms_att_all = (float*)malloc(nl * d * sizeof(float));
    float *wq_all      = (float*)malloc(nl * d * d * sizeof(float));
    float *wk_all      = (float*)malloc(nl * d * d * sizeof(float));
    float *wv_all      = (float*)malloc(nl * d * d * sizeof(float));
    float *wo_all      = (float*)malloc(nl * d * d * sizeof(float));
    float *rms_ffn_all = (float*)malloc(nl * d * sizeof(float));
    float *w1_all      = (float*)malloc(nl * hd * d * sizeof(float));
    float *w2_all      = (float*)malloc(nl * d * hd * sizeof(float));
    float *w3_all      = (float*)malloc(nl * hd * d * sizeof(float));

    if (!rms_att_all || !wq_all || !wk_all || !wv_all || !wo_all ||
        !rms_ffn_all || !w1_all || !w2_all || !w3_all) {
        fprintf(stderr, "ERROR: OOM allocating layer weights\n");
        fclose(f); return -1;
    }

    #define FREAD_CHECK(buf, count, file, label) do { \
        size_t _n = fread(buf, sizeof(float), count, file); \
        if (_n != (size_t)(count)) { \
            fprintf(stderr, "ERROR: short read on %s: got %zu, expected %zu (file truncated?)\n", \
                    label, _n, (size_t)(count)); \
            fclose(file); return -1; \
        } \
    } while(0)

    FREAD_CHECK(rms_att_all, nl * d, f, "rms_att");
    FREAD_CHECK(wq_all, nl * d * d, f, "wq");
    FREAD_CHECK(wk_all, nl * d * d, f, "wk");
    FREAD_CHECK(wv_all, nl * d * d, f, "wv");
    FREAD_CHECK(wo_all, nl * d * d, f, "wo");
    FREAD_CHECK(rms_ffn_all, nl * d, f, "rms_ffn");
    FREAD_CHECK(w1_all, nl * hd * d, f, "w1");
    FREAD_CHECK(w2_all, nl * d * hd, f, "w2");
    FREAD_CHECK(w3_all, nl * hd * d, f, "w3");

    #define SAFE_MALLOC_MEMCPY(dest, src, size) do { \
        dest = (float*)malloc(size); \
        if (!(dest)) { \
            fprintf(stderr, "ERROR: memory allocation failed for size %zu\n", (size_t)(size)); \
            fclose(f); return -1; \
        } \
        memcpy(dest, src, size); \
    } while(0)

    for (int l = 0; l < nl; l++) {
        SAFE_MALLOC_MEMCPY(m->rms_att_w[l], rms_att_all + l*d, d * sizeof(float));
        SAFE_MALLOC_MEMCPY(m->wq[l], wq_all + l*d*d, d*d*sizeof(float));
        SAFE_MALLOC_MEMCPY(m->wk[l], wk_all + l*d*d, d*d*sizeof(float));
        SAFE_MALLOC_MEMCPY(m->wv[l], wv_all + l*d*d, d*d*sizeof(float));
        SAFE_MALLOC_MEMCPY(m->wo[l], wo_all + l*d*d, d*d*sizeof(float));
        SAFE_MALLOC_MEMCPY(m->rms_ffn_w[l], rms_ffn_all + l*d, d * sizeof(float));
        SAFE_MALLOC_MEMCPY(m->w1[l], w1_all + l*hd*d, hd*d*sizeof(float));
        SAFE_MALLOC_MEMCPY(m->w2[l], w2_all + l*d*hd, d*hd*sizeof(float));
        SAFE_MALLOC_MEMCPY(m->w3[l], w3_all + l*hd*d, hd*d*sizeof(float));
    }

    #undef SAFE_MALLOC_MEMCPY
    free(rms_att_all); free(wq_all); free(wk_all); free(wv_all); free(wo_all);
    free(rms_ffn_all); free(w1_all); free(w2_all); free(w3_all);

    m->rms_final_w = (float*)malloc(d * sizeof(float));
    FREAD_CHECK(m->rms_final_w, d, f, "rms_final");

    if (shared) {
        m->wcls = m->token_embedding;
    } else {
        m->wcls = (float*)malloc(vs * d * sizeof(float));
        FREAD_CHECK(m->wcls, vs * d, f, "wcls");
    }
    #undef FREAD_CHECK
    fclose(f);
    return 0;
}

// Compile a single baked-weight conv kernel
static ANEKernel *compile_conv_kernel(const float *weights, int in_ch, int out_ch, int spatial) {
    NSData *wb = mil_build_weight_blob(weights, out_ch, in_ch);
    NSString *mil = mil_gen_conv(in_ch, out_ch, spatial);
    size_t inBytes = (size_t)in_ch * spatial * 4;
    size_t outBytes = (size_t)out_ch * spatial * 4;
    return ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], wb, 1, &inBytes, 1, &outBytes);
}

// Compile all per-layer ANE kernels with current weights
static int model_compile_kernels(Model *m, int seq_len) {
    m->seq_len = seq_len;
    int d = m->cfg.dim, hd = m->cfg.hidden_dim, vs = m->cfg.vocab_size;
    int S = seq_len;
    printf("Compiling %d ANE conv kernels (S=%d)...\n", N_LAYERS * 7 + 1, S);

    for (int l = 0; l < N_LAYERS; l++) {
        m->kern_q[l] = compile_conv_kernel(m->wq[l], d, d, S);
        m->kern_k[l] = compile_conv_kernel(m->wk[l], d, d, S);
        m->kern_v[l] = compile_conv_kernel(m->wv[l], d, d, S);
        m->kern_o[l] = compile_conv_kernel(m->wo[l], d, d, S);
        m->kern_w1[l] = compile_conv_kernel(m->w1[l], d, hd, S);
        m->kern_w2[l] = compile_conv_kernel(m->w2[l], hd, d, S);
        m->kern_w3[l] = compile_conv_kernel(m->w3[l], d, hd, S);
        if (!m->kern_q[l]) { fprintf(stderr, "L%d kern_q fail\n",l); return -1; }
        if (!m->kern_k[l]) { fprintf(stderr, "L%d kern_k fail\n",l); return -1; }
        if (!m->kern_v[l]) { fprintf(stderr, "L%d kern_v fail\n",l); return -1; }
        if (!m->kern_o[l]) { fprintf(stderr, "L%d kern_o fail\n",l); return -1; }
        if (!m->kern_w1[l]) { fprintf(stderr, "L%d kern_w1 fail\n",l); return -1; }
        if (!m->kern_w2[l]) { fprintf(stderr, "L%d kern_w2 fail\n",l); return -1; }
        if (!m->kern_w3[l]) { fprintf(stderr, "L%d kern_w3 fail\n",l); return -1; }
        printf("  Layer %d OK\n", l);
    }
    m->kern_cls = compile_conv_kernel(m->wcls, d, vs, S);
    if (!m->kern_cls) {
        fprintf(stderr, "Classifier kernel compile failed (dim=%d→vocab=%d too large?), using CPU for cls\n", d, vs);
    }
    printf("  All kernels compiled (%d conv + %s)\n", N_LAYERS * 7, m->kern_cls ? "cls" : "cls=CPU");
    return 0;
}

// Recompile all kernels after weight update — compile new first, then swap
static int model_recompile_kernels(Model *m) {
    int d = m->cfg.dim, hd = m->cfg.hidden_dim, vs = m->cfg.vocab_size;
    int S = m->seq_len;

    // Phase 1: compile new kernels into temporaries
    ANEKernel *new_q[N_LAYERS], *new_k[N_LAYERS], *new_v[N_LAYERS], *new_o[N_LAYERS];
    ANEKernel *new_w1[N_LAYERS], *new_w2[N_LAYERS], *new_w3[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        new_q[l] = compile_conv_kernel(m->wq[l], d, d, S);
        new_k[l] = compile_conv_kernel(m->wk[l], d, d, S);
        new_v[l] = compile_conv_kernel(m->wv[l], d, d, S);
        new_o[l] = compile_conv_kernel(m->wo[l], d, d, S);
        new_w1[l] = compile_conv_kernel(m->w1[l], d, hd, S);
        new_w2[l] = compile_conv_kernel(m->w2[l], hd, d, S);
        new_w3[l] = compile_conv_kernel(m->w3[l], d, hd, S);
        if (!new_q[l] || !new_k[l] || !new_v[l] || !new_o[l] ||
            !new_w1[l] || !new_w2[l] || !new_w3[l]) {
            // Cleanup partially compiled new kernels
            for (int i = 0; i <= l; i++) {
                ane_free(new_q[i]); ane_free(new_k[i]); ane_free(new_v[i]); ane_free(new_o[i]);
                ane_free(new_w1[i]); ane_free(new_w2[i]); ane_free(new_w3[i]);
            }
            fprintf(stderr, "Recompile failed at layer %d, keeping old kernels\n", l);
            return -1;
        }
    }
    ANEKernel *new_cls = compile_conv_kernel(m->wcls, d, vs, S);

    // Phase 2: all compiles succeeded — swap and free old
    for (int l = 0; l < N_LAYERS; l++) {
        ane_free(m->kern_q[l]); ane_free(m->kern_k[l]); ane_free(m->kern_v[l]); ane_free(m->kern_o[l]);
        ane_free(m->kern_w1[l]); ane_free(m->kern_w2[l]); ane_free(m->kern_w3[l]);
        m->kern_q[l] = new_q[l]; m->kern_k[l] = new_k[l];
        m->kern_v[l] = new_v[l]; m->kern_o[l] = new_o[l];
        m->kern_w1[l] = new_w1[l]; m->kern_w2[l] = new_w2[l]; m->kern_w3[l] = new_w3[l];
    }
    if (m->kern_cls) ane_free(m->kern_cls);
    m->kern_cls = new_cls;  // may be NULL for large vocab — forward uses CPU fallback
    return 0;
}

static int model_alloc_training(Model *m) {
    
    size_t d = (size_t)m->cfg.dim, hd = (size_t)m->cfg.hidden_dim;
    size_t vs = (size_t)m->cfg.vocab_size, S = (size_t)m->seq_len;

    #define SAFE_CALLOC(dest, count) do { \
        dest = (float*)calloc(count, sizeof(float)); \
        if (!(dest)) { \
            fprintf(stderr, "ERROR: OOM in model_alloc_training for size %zu\n", (size_t)(count)); \
            return -1; \
        } \
    } while(0)

    for (int l = 0; l < N_LAYERS; l++) {
        SAFE_CALLOC(m->act_x[l], S * d);
        SAFE_CALLOC(m->act_xnorm[l], S * d);
        SAFE_CALLOC(m->act_q[l], S * d);
        SAFE_CALLOC(m->act_k[l], S * d);
        SAFE_CALLOC(m->act_v[l], S * d);
        SAFE_CALLOC(m->act_attn_out[l], S * d);
        SAFE_CALLOC(m->act_ffn_in[l], S * d);
        SAFE_CALLOC(m->act_h1[l], S * hd);
        SAFE_CALLOC(m->act_h3[l], S * hd);
        SAFE_CALLOC(m->act_silu[l], S * hd);

        SAFE_CALLOC(m->grad_wq[l], d * d);
        SAFE_CALLOC(m->grad_wk[l], d * d);
        SAFE_CALLOC(m->grad_wv[l], d * d);
        SAFE_CALLOC(m->grad_wo[l], d * d);
        SAFE_CALLOC(m->grad_w1[l], hd * d);
        SAFE_CALLOC(m->grad_w2[l], d * hd);
        SAFE_CALLOC(m->grad_w3[l], hd * d);
    }
    SAFE_CALLOC(m->act_final, S * d);
    SAFE_CALLOC(m->act_pre_final, S * d);
    SAFE_CALLOC(m->logits, S * vs);
    SAFE_CALLOC(m->grad_wcls, vs * d);
    SAFE_CALLOC(m->grad_emb, vs * d);

    m->total_params = 0;
    for (int l = 0; l < N_LAYERS; l++)
        m->total_params += 4*d*d + 2*hd*d + d*hd;
    m->total_params += vs * d * 2;
    SAFE_CALLOC(m->adam_m, m->total_params);
    SAFE_CALLOC(m->adam_v, m->total_params);
    m->adam_step = 0;

    #undef SAFE_CALLOC

    printf("Total trainable params: %zu (%.1f M)\n", m->total_params, m->total_params/1e6);
    return 0;
}
