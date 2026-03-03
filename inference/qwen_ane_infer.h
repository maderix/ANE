// qwen_ane_infer.h — Qwen2.5-0.5B inference on Apple Neural Engine
// Linear projections on ANE (baked-weight conv kernels), CPU for element-wise ops.
// Based on maderix/ANE runtime + MIL generation.
#pragma once

#include "../training/ane_runtime.h"
#include "../training/ane_mil_gen.h"

// Compile a matmul kernel: W[out_ch, in_ch] @ x[in_ch] → y[out_ch]
// Uses the two-input matmul MIL variant (weights passed as input, not baked)
static ANEKernel *compile_matmul_kernel(int in_ch, int out_ch) {
    NSString *mil = mil_gen_matmul(in_ch, out_ch, 1);
    size_t inputSizes[2] = {(size_t)in_ch * 1 * 4, (size_t)out_ch * in_ch * 4};
    size_t outBytes = (size_t)out_ch * 1 * 4;
    return ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 2, inputSizes, 1, &outBytes);
}

// Compile a baked-weight conv kernel (from model.h)
static ANEKernel *compile_conv_kernel(const float *weights, int in_ch, int out_ch, int spatial) {
    NSData *wb = mil_build_weight_blob(weights, out_ch, in_ch);
    NSString *mil = mil_gen_conv(in_ch, out_ch, spatial);
    size_t inBytes = (size_t)in_ch * spatial * 4;
    size_t outBytes = (size_t)out_ch * spatial * 4;
    return ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], wb, 1, &inBytes, 1, &outBytes);
}
#include <math.h>
#include <string.h>
#include <time.h>

// Qwen2.5-0.5B-Instruct architecture
#define QWEN_DIM         896
#define QWEN_HIDDEN      4864
#define QWEN_LAYERS      24
#define QWEN_HEADS       14
#define QWEN_KV_HEADS    2
#define QWEN_HEAD_DIM    64
#define QWEN_VOCAB       151936
#define QWEN_RMS_EPS     1e-6f
#define QWEN_ROPE_THETA  1000000.0f
#define QWEN_MAX_SEQ     512

// GQA: each KV head serves (HEADS / KV_HEADS) query heads
#define QWEN_GQA_FACTOR  (QWEN_HEADS / QWEN_KV_HEADS)

// Sizes for GQA projections
#define QWEN_Q_DIM       (QWEN_HEADS * QWEN_HEAD_DIM)      // 896
#define QWEN_KV_DIM      (QWEN_KV_HEADS * QWEN_HEAD_DIM)   // 128

typedef struct {
    // Weights (f32)
    float *embed;                          // [vocab, dim]
    float *rms_att[QWEN_LAYERS];          // [dim]
    float *wq[QWEN_LAYERS];              // [q_dim, dim]
    float *wk[QWEN_LAYERS];              // [kv_dim, dim]
    float *wv[QWEN_LAYERS];              // [kv_dim, dim]
    float *wo[QWEN_LAYERS];              // [dim, q_dim]
    float *rms_ffn[QWEN_LAYERS];         // [dim]
    float *w_gate[QWEN_LAYERS];          // [hidden, dim]
    float *w_up[QWEN_LAYERS];            // [hidden, dim]
    float *w_down[QWEN_LAYERS];          // [dim, hidden]
    float *rms_final;                      // [dim]
    // wcls = embed (tied)

    // ANE kernels (one per linear projection per layer)
    ANEKernel *k_q[QWEN_LAYERS];
    ANEKernel *k_k[QWEN_LAYERS];
    ANEKernel *k_v[QWEN_LAYERS];
    ANEKernel *k_o[QWEN_LAYERS];
    ANEKernel *k_gate[QWEN_LAYERS];
    ANEKernel *k_up[QWEN_LAYERS];
    ANEKernel *k_down[QWEN_LAYERS];
    // LM head chunked: vocab too large for single ANE kernel (max 65536)
    #define QWEN_LM_CHUNKS 16
    #define QWEN_LM_CHUNK_SIZE 9496  // 151936 / 16
    ANEKernel *k_lmhead[QWEN_LM_CHUNKS];

    // Q/K/V biases per layer
    float *q_bias[QWEN_LAYERS];   // [q_dim]
    float *k_bias[QWEN_LAYERS];   // [kv_dim]
    float *v_bias[QWEN_LAYERS];   // [kv_dim]

    // KV cache [layer][kv_heads * head_dim * max_seq]
    float *kv_cache_k[QWEN_LAYERS];
    float *kv_cache_v[QWEN_LAYERS];
    int pos;  // current position in sequence

    // Scratch buffers
    float *x;       // [dim]
    float *xb;      // [dim]
    float *q;       // [q_dim]
    float *k;       // [kv_dim]
    float *v;       // [kv_dim]
    float *att;     // [heads * max_seq]
    float *hb;      // [hidden]
    float *hb2;     // [hidden]
    float *logits;  // [vocab]
} QwenModel;

// ── CPU ops ──────────────────────────────────────────────────────────

static void qwen_rmsnorm(float *out, const float *x, const float *w, int D) {
    float ss = 0;
    for (int i = 0; i < D; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / D + QWEN_RMS_EPS);
    for (int i = 0; i < D; i++) out[i] = x[i] * ss * w[i];
}

static void qwen_rope(float *q, float *k, int pos, int n_q_heads, int n_kv_heads, int head_dim) {
    // Qwen uses rotate_half RoPE (NOT interleaved pairs):
    //   rotate_half(x) = [-x[dim/2:], x[:dim/2]]
    //   q_embed = q * cos + rotate_half(q) * sin
    // cos/sin have shape [head_dim/2] and are applied to both halves
    int half = head_dim / 2;

    // Precompute cos/sin for this position (head_dim/2 frequencies)
    float cos_v[half], sin_v[half];
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(QWEN_ROPE_THETA, (float)(2 * i) / head_dim);
        float angle = pos * freq;
        cos_v[i] = cosf(angle);
        sin_v[i] = sinf(angle);
    }

    // Apply to Q heads
    for (int h = 0; h < n_q_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float q_first = qh[i];
            float q_second = qh[i + half];
            // rotate_half: [-q_second, q_first]
            qh[i]        = q_first * cos_v[i] + (-q_second) * sin_v[i];
            qh[i + half]  = q_second * cos_v[i] + q_first * sin_v[i];
        }
    }

    // Apply to K heads
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float k_first = kh[i];
            float k_second = kh[i + half];
            kh[i]        = k_first * cos_v[i] + (-k_second) * sin_v[i];
            kh[i + half]  = k_second * cos_v[i] + k_first * sin_v[i];
        }
    }
}

static void qwen_silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

// ── ANE projection helper (single token: spatial=1) ─────────────────

static void ane_project(ANEKernel *kernel, const float *in, float *out,
                        int in_dim, int out_dim) {
    // For single-token inference: spatial=1
    ane_write_input(kernel, 0, in, in_dim * sizeof(float));
    ane_eval(kernel);
    ane_read_output(kernel, 0, out, out_dim * sizeof(float));
}

// CPU matmul via Accelerate BLAS: y = W @ x, W[out_dim, in_dim]
#include <Accelerate/Accelerate.h>

static void cpu_project(const float *W, const float *x, float *y, int in_dim, int out_dim) {
    // y = W @ x where W is [out_dim, in_dim] row-major
    // cblas_sgemv: y = alpha * A * x + beta * y
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                out_dim, in_dim,
                1.0f, W, in_dim,
                x, 1,
                0.0f, y, 1);
}

// Toggle: 1 = use ANE for projections, 0 = CPU fallback
#define USE_ANE_PROJECTIONS 0

// ── Forward one token ────────────────────────────────────────────────

static int qwen_forward(QwenModel *m, int token) {
    int D = QWEN_DIM, HD = QWEN_HIDDEN;
    int pos = m->pos;

    // Token embedding
    memcpy(m->x, m->embed + token * D, D * sizeof(float));

    for (int l = 0; l < QWEN_LAYERS; l++) {
        // Attention RMSNorm
        qwen_rmsnorm(m->xb, m->x, m->rms_att[l], D);

        // Debug: print first layer input/output norms
        if (l == 0 && pos == 0) {
            float xnorm = 0, qnorm = 0;
            for (int i = 0; i < D; i++) xnorm += m->xb[i] * m->xb[i];
            printf("  L0 RMSNorm out norm=%.4f (first 4: %.4f %.4f %.4f %.4f)\n",
                   sqrtf(xnorm), m->xb[0], m->xb[1], m->xb[2], m->xb[3]);
        }

        // QKV projections (ANE) + bias
        #if USE_ANE_PROJECTIONS
        ane_project(m->k_q[l], m->xb, m->q, D, QWEN_Q_DIM);
        ane_project(m->k_k[l], m->xb, m->k, D, QWEN_KV_DIM);
        ane_project(m->k_v[l], m->xb, m->v, D, QWEN_KV_DIM);
        #else
        cpu_project(m->wq[l], m->xb, m->q, D, QWEN_Q_DIM);
        cpu_project(m->wk[l], m->xb, m->k, D, QWEN_KV_DIM);
        cpu_project(m->wv[l], m->xb, m->v, D, QWEN_KV_DIM);
        #endif
        // Apply Q/K biases
        if (m->q_bias[l]) {
            for (int i = 0; i < QWEN_Q_DIM; i++) m->q[i] += m->q_bias[l][i];
        }
        if (m->k_bias[l]) {
            for (int i = 0; i < QWEN_KV_DIM; i++) m->k[i] += m->k_bias[l][i];
        }
        if (m->v_bias[l]) {
            for (int i = 0; i < QWEN_KV_DIM; i++) m->v[i] += m->v_bias[l][i];
        }

        if (l == 0 && pos == 0) {
            float qn = 0;
            for (int i = 0; i < QWEN_Q_DIM; i++) qn += m->q[i] * m->q[i];
            printf("  L0 ANE Q norm=%.4f (first 4: %.4f %.4f %.4f %.4f)\n",
                   sqrtf(qn), m->q[0], m->q[1], m->q[2], m->q[3]);
            // CPU reference
            float cpu_q[4] = {0};
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < D; j++)
                    cpu_q[i] += m->wq[0][i * D + j] * m->xb[j];
                cpu_q[i] += m->q_bias[0][i];
            }
            printf("  L0 CPU Q first 4: %.4f %.4f %.4f %.4f\n",
                   cpu_q[0], cpu_q[1], cpu_q[2], cpu_q[3]);
        }

        // RoPE
        qwen_rope(m->q, m->k, pos, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM);

        // Store K, V in cache
        memcpy(m->kv_cache_k[l] + pos * QWEN_KV_DIM,
               m->k, QWEN_KV_DIM * sizeof(float));
        memcpy(m->kv_cache_v[l] + pos * QWEN_KV_DIM,
               m->v, QWEN_KV_DIM * sizeof(float));

        // GQA attention (CPU — element-wise ops)
        float scale = 1.0f / sqrtf((float)QWEN_HEAD_DIM);
        float *attn_out = m->xb;  // reuse buffer
        memset(attn_out, 0, QWEN_Q_DIM * sizeof(float));

        for (int h = 0; h < QWEN_HEADS; h++) {
            int kv_h = h / QWEN_GQA_FACTOR;
            float *qh = m->q + h * QWEN_HEAD_DIM;

            // Attention scores: Q @ K^T for all positions up to pos
            float max_score = -1e9f;
            for (int t = 0; t <= pos; t++) {
                float *kt = m->kv_cache_k[l] + t * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                // Use BLAS dot product for precision
                float score = cblas_sdot(QWEN_HEAD_DIM, qh, 1, kt, 1);
                m->att[h * QWEN_MAX_SEQ + t] = score * scale;
                if (score * scale > max_score) max_score = score * scale;
            }
            // Softmax (double accumulation for precision)
            double sum = 0;
            for (int t = 0; t <= pos; t++) {
                m->att[h * QWEN_MAX_SEQ + t] = expf(m->att[h * QWEN_MAX_SEQ + t] - max_score);
                sum += (double)m->att[h * QWEN_MAX_SEQ + t];
            }
            float inv_sum = (float)(1.0 / sum);
            for (int t = 0; t <= pos; t++)
                m->att[h * QWEN_MAX_SEQ + t] *= inv_sum;

            // Weighted sum of V: attn_out[h] += att[t] * V[t] for each t
            for (int t = 0; t <= pos; t++) {
                float a = m->att[h * QWEN_MAX_SEQ + t];
                float *vt = m->kv_cache_v[l] + t * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                cblas_saxpy(QWEN_HEAD_DIM, a, vt, 1,
                           attn_out + h * QWEN_HEAD_DIM, 1);
            }
        }

        float o_out[QWEN_DIM];
        #if USE_ANE_PROJECTIONS
        ane_project(m->k_o[l], attn_out, o_out, QWEN_Q_DIM, D);
        #else
        cpu_project(m->wo[l], attn_out, o_out, QWEN_Q_DIM, D);
        #endif

        // Residual
        for (int i = 0; i < D; i++) m->x[i] += o_out[i];

        if (l == 0 && pos == 0) {
            float pan = 0;
            for (int i = 0; i < D; i++) pan += m->x[i] * m->x[i];
            printf("  L0 post-attn norm=%.4f first4=[%.6f, %.6f, %.6f, %.6f]\n",
                   sqrtf(pan), m->x[0], m->x[1], m->x[2], m->x[3]);
            float on = 0;
            for (int i = 0; i < D; i++) on += o_out[i] * o_out[i];
            printf("  L0 o_proj out norm=%.4f first4=[%.6f, %.6f, %.6f, %.6f]\n",
                   sqrtf(on), o_out[0], o_out[1], o_out[2], o_out[3]);
        }

        // FFN RMSNorm
        qwen_rmsnorm(m->xb, m->x, m->rms_ffn[l], D);

        // SwiGLU FFN
        #if USE_ANE_PROJECTIONS
        ane_project(m->k_gate[l], m->xb, m->hb, D, HD);
        ane_project(m->k_up[l], m->xb, m->hb2, D, HD);
        #else
        cpu_project(m->w_gate[l], m->xb, m->hb, D, HD);
        cpu_project(m->w_up[l], m->xb, m->hb2, D, HD);
        #endif

        if (l == 0 && pos == 0) {
            float gn = 0, un = 0;
            for (int i = 0; i < HD; i++) { gn += m->hb[i]*m->hb[i]; un += m->hb2[i]*m->hb2[i]; }
            printf("  L0 gate norm=%.4f up norm=%.4f\n", sqrtf(gn), sqrtf(un));
            printf("  L0 gate first4=[%.6f, %.6f, %.6f, %.6f]\n",
                   m->hb[0], m->hb[1], m->hb[2], m->hb[3]);
        }

        qwen_silu(m->hb, HD);
        for (int i = 0; i < HD; i++) m->hb[i] *= m->hb2[i];

        float ffn_out[QWEN_DIM];
        #if USE_ANE_PROJECTIONS
        ane_project(m->k_down[l], m->hb, ffn_out, HD, D);
        #else
        cpu_project(m->w_down[l], m->hb, ffn_out, HD, D);
        #endif

        // Residual
        for (int i = 0; i < D; i++) m->x[i] += ffn_out[i];

        // Debug: hidden state after each layer (first 3 layers, first token only)
        if (l < 3 && pos == 0) {
            float hn = 0;
            for (int i = 0; i < D; i++) hn += m->x[i] * m->x[i];
            printf("  C hidden[%d] norm=%.4f first4=[%.4f, %.4f, %.4f, %.4f]\n",
                   l+1, sqrtf(hn), m->x[0], m->x[1], m->x[2], m->x[3]);
        }
    }

    // Final RMSNorm
    qwen_rmsnorm(m->xb, m->x, m->rms_final, D);

    // Debug: check final hidden state before LM head
    if (m->pos < 2) {
        float fn = 0;
        for (int i = 0; i < D; i++) fn += m->xb[i] * m->xb[i];
        printf("  Final hidden norm=%.4f (first 4: %.6f %.6f %.6f %.6f)\n",
               sqrtf(fn), m->xb[0], m->xb[1], m->xb[2], m->xb[3]);
    }

    // LM head via Accelerate BLAS: logits = embed @ xb
    // embed is [vocab, dim] row-major
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                QWEN_VOCAB, D,
                1.0f, m->embed, D,
                m->xb, 1,
                0.0f, m->logits, 1);

    // Debug: check logits
    if (m->pos < 2) {
        float lmax = m->logits[0], lmin = m->logits[0];
        int nonzero = 0;
        for (int i = 0; i < QWEN_VOCAB; i++) {
            if (m->logits[i] > lmax) lmax = m->logits[i];
            if (m->logits[i] < lmin) lmin = m->logits[i];
            if (m->logits[i] != 0.0f) nonzero++;
        }
        printf("  Logits: min=%.4f max=%.4f nonzero=%d/%d\n", lmin, lmax, nonzero, QWEN_VOCAB);
    }

    m->pos++;

    // Argmax
    int max_idx = 0;
    float max_val = m->logits[0];
    for (int i = 1; i < QWEN_VOCAB; i++) {
        if (m->logits[i] > max_val) {
            max_val = m->logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ── Compile all ANE kernels ──────────────────────────────────────────

static void qwen_compile_kernels(QwenModel *m) {
    int D = QWEN_DIM, HD = QWEN_HIDDEN;
    printf("Compiling %d ANE kernels...\n", QWEN_LAYERS * 7 + 1);
    for (int l = 0; l < QWEN_LAYERS; l++) {
        m->k_q[l]    = compile_conv_kernel(m->wq[l],    D, QWEN_Q_DIM,  1);
        m->k_k[l]    = compile_conv_kernel(m->wk[l],    D, QWEN_KV_DIM, 1);
        m->k_v[l]    = compile_conv_kernel(m->wv[l],    D, QWEN_KV_DIM, 1);
        m->k_o[l]    = compile_conv_kernel(m->wo[l],    QWEN_Q_DIM, D,  1);
        m->k_gate[l] = compile_conv_kernel(m->w_gate[l], D, HD,          1);
        m->k_up[l]   = compile_conv_kernel(m->w_up[l],   D, HD,          1);
        m->k_down[l] = compile_conv_kernel(m->w_down[l], HD, D,          1);
        printf("  Layer %d/%d compiled\r", l+1, QWEN_LAYERS);
        fflush(stdout);
    }
    // LM head (tied = embedding, chunked into 16 pieces)
    for (int c = 0; c < QWEN_LM_CHUNKS; c++) {
        float *chunk_weights = m->embed + c * QWEN_LM_CHUNK_SIZE * D;
        m->k_lmhead[c] = compile_conv_kernel(chunk_weights, D, QWEN_LM_CHUNK_SIZE, 1);
        if (!m->k_lmhead[c]) {
            printf("  LM head chunk %d FAILED to compile\n", c);
        }
    }
    printf("\nAll kernels compiled.\n");
}

// ── Allocate buffers ─────────────────────────────────────────────────

static void qwen_alloc(QwenModel *m) {
    m->x      = (float*)calloc(QWEN_DIM, sizeof(float));
    m->xb     = (float*)calloc(QWEN_DIM, sizeof(float));
    m->q      = (float*)calloc(QWEN_Q_DIM, sizeof(float));
    m->k      = (float*)calloc(QWEN_KV_DIM, sizeof(float));
    m->v      = (float*)calloc(QWEN_KV_DIM, sizeof(float));
    m->att    = (float*)calloc(QWEN_HEADS * QWEN_MAX_SEQ, sizeof(float));
    m->hb     = (float*)calloc(QWEN_HIDDEN, sizeof(float));
    m->hb2    = (float*)calloc(QWEN_HIDDEN, sizeof(float));
    m->logits = (float*)calloc(QWEN_VOCAB, sizeof(float));
    for (int l = 0; l < QWEN_LAYERS; l++) {
        m->kv_cache_k[l] = (float*)calloc(QWEN_MAX_SEQ * QWEN_KV_DIM, sizeof(float));
        m->kv_cache_v[l] = (float*)calloc(QWEN_MAX_SEQ * QWEN_KV_DIM, sizeof(float));
    }
    m->pos = 0;
}
