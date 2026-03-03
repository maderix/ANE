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
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>

#ifndef QWEN_DEBUG
#define QWEN_DEBUG 0
#endif

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

// ── Precomputed RoPE table ───────────────────────────────────────────

static float g_rope_cos[QWEN_MAX_SEQ][QWEN_HEAD_DIM / 2];
static float g_rope_sin[QWEN_MAX_SEQ][QWEN_HEAD_DIM / 2];
static int g_rope_initialized = 0;

static void qwen_rope_init(void) {
    if (g_rope_initialized) return;
    int half = QWEN_HEAD_DIM / 2;
    for (int pos = 0; pos < QWEN_MAX_SEQ; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(QWEN_ROPE_THETA, (float)(2 * i) / QWEN_HEAD_DIM);
            float angle = pos * freq;
            g_rope_cos[pos][i] = cosf(angle);
            g_rope_sin[pos][i] = sinf(angle);
        }
    }
    g_rope_initialized = 1;
}

// ── CPU ops (vectorized with NEON + vDSP) ────────────────────────────

static void qwen_rmsnorm(float *out, const float *x, const float *w, int D) {
    float ss;
    vDSP_svesq(x, 1, &ss, (vDSP_Length)D);
    ss = 1.0f / sqrtf(ss / D + QWEN_RMS_EPS);
    vDSP_vsmul(x, 1, &ss, out, 1, (vDSP_Length)D);
    vDSP_vmul(out, 1, w, 1, out, 1, (vDSP_Length)D);
}

static void qwen_rope(float *q, float *k, int pos, int n_q_heads, int n_kv_heads, int head_dim) {
    int half = head_dim / 2;
    const float *cv = g_rope_cos[pos];
    const float *sv = g_rope_sin[pos];

    for (int h = 0; h < n_q_heads; h++) {
        float *qh = q + h * head_dim;
        int i = 0;
        for (; i + 3 < half; i += 4) {
            float32x4_t first  = vld1q_f32(qh + i);
            float32x4_t second = vld1q_f32(qh + i + half);
            float32x4_t c = vld1q_f32(cv + i);
            float32x4_t s = vld1q_f32(sv + i);
            vst1q_f32(qh + i,        vmlsq_f32(vmulq_f32(first, c), second, s));
            vst1q_f32(qh + i + half, vmlaq_f32(vmulq_f32(second, c), first, s));
        }
        for (; i < half; i++) {
            float f = qh[i], se = qh[i + half];
            qh[i]        = f * cv[i] - se * sv[i];
            qh[i + half] = se * cv[i] + f * sv[i];
        }
    }

    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        int i = 0;
        for (; i + 3 < half; i += 4) {
            float32x4_t first  = vld1q_f32(kh + i);
            float32x4_t second = vld1q_f32(kh + i + half);
            float32x4_t c = vld1q_f32(cv + i);
            float32x4_t s = vld1q_f32(sv + i);
            vst1q_f32(kh + i,        vmlsq_f32(vmulq_f32(first, c), second, s));
            vst1q_f32(kh + i + half, vmlaq_f32(vmulq_f32(second, c), first, s));
        }
        for (; i < half; i++) {
            float f = kh[i], se = kh[i + half];
            kh[i]        = f * cv[i] - se * sv[i];
            kh[i + half] = se * cv[i] + f * sv[i];
        }
    }
}

static void qwen_silu(float *x, int n) {
    int i = 0;
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float neg[4];
        vst1q_f32(neg, vnegq_f32(v));
        float exp_neg[4];
        for (int j = 0; j < 4; j++) exp_neg[j] = expf(neg[j]);
        float32x4_t denom = vaddq_f32(one, vld1q_f32(exp_neg));
        vst1q_f32(x + i, vdivq_f32(v, denom));
    }
    for (; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

// ── ANE projection helper (single token: spatial=1) ─────────────────

static inline bool ane_run(ANEKernel *k) { return ane_eval(k); }

static void ane_project(ANEKernel *kernel, const float *in, float *out,
                        int in_dim, int out_dim) {
    ane_write_input(kernel, 0, in, in_dim * sizeof(float));
    ane_run(kernel);
    ane_read_output(kernel, 0, out, out_dim * sizeof(float));
}

// CPU matmul via Accelerate BLAS: y = W @ x, W[out_dim, in_dim]
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

#if QWEN_DEBUG
        if (l == 0 && pos == 0) {
            float xnorm = 0;
            for (int i = 0; i < D; i++) xnorm += m->xb[i] * m->xb[i];
            printf("  L0 RMSNorm out norm=%.4f (first 4: %.4f %.4f %.4f %.4f)\n",
                   sqrtf(xnorm), m->xb[0], m->xb[1], m->xb[2], m->xb[3]);
        }
#endif

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
        // Apply Q/K/V biases (vectorized)
        if (m->q_bias[l])
            vDSP_vadd(m->q, 1, m->q_bias[l], 1, m->q, 1, (vDSP_Length)QWEN_Q_DIM);
        if (m->k_bias[l])
            vDSP_vadd(m->k, 1, m->k_bias[l], 1, m->k, 1, (vDSP_Length)QWEN_KV_DIM);
        if (m->v_bias[l])
            vDSP_vadd(m->v, 1, m->v_bias[l], 1, m->v, 1, (vDSP_Length)QWEN_KV_DIM);

#if QWEN_DEBUG
        if (l == 0 && pos == 0) {
            float qn = 0;
            for (int i = 0; i < QWEN_Q_DIM; i++) qn += m->q[i] * m->q[i];
            printf("  L0 ANE Q norm=%.4f (first 4: %.4f %.4f %.4f %.4f)\n",
                   sqrtf(qn), m->q[0], m->q[1], m->q[2], m->q[3]);
            float cpu_q[4] = {0};
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < D; j++)
                    cpu_q[i] += m->wq[0][i * D + j] * m->xb[j];
                cpu_q[i] += m->q_bias[0][i];
            }
            printf("  L0 CPU Q first 4: %.4f %.4f %.4f %.4f\n",
                   cpu_q[0], cpu_q[1], cpu_q[2], cpu_q[3]);
        }
#endif

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
            float *att_h = m->att + h * QWEN_MAX_SEQ;
            int seq_len = pos + 1;

            // Attention scores: Q @ K^T
            float max_score = -1e9f;
            for (int t = 0; t <= pos; t++) {
                float *kt = m->kv_cache_k[l] + t * QWEN_KV_DIM + kv_h * QWEN_HEAD_DIM;
                float score = cblas_sdot(QWEN_HEAD_DIM, qh, 1, kt, 1);
                att_h[t] = score * scale;
                if (att_h[t] > max_score) max_score = att_h[t];
            }
            // Softmax: subtract max, exp, normalize (vDSP)
            float neg_max = -max_score;
            vDSP_vsadd(att_h, 1, &neg_max, att_h, 1, (vDSP_Length)seq_len);
            int n_exp = seq_len;
            vvexpf(att_h, att_h, &n_exp);
            float sum;
            vDSP_sve(att_h, 1, &sum, (vDSP_Length)seq_len);
            float inv_sum = 1.0f / sum;
            vDSP_vsmul(att_h, 1, &inv_sum, att_h, 1, (vDSP_Length)seq_len);

            // Weighted sum of V
            for (int t = 0; t <= pos; t++) {
                float a = att_h[t];
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

        // Residual (vectorized)
        vDSP_vadd(m->x, 1, o_out, 1, m->x, 1, (vDSP_Length)D);

#if QWEN_DEBUG
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
#endif

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

#if QWEN_DEBUG
        if (l == 0 && pos == 0) {
            float gn = 0, un = 0;
            for (int i = 0; i < HD; i++) { gn += m->hb[i]*m->hb[i]; un += m->hb2[i]*m->hb2[i]; }
            printf("  L0 gate norm=%.4f up norm=%.4f\n", sqrtf(gn), sqrtf(un));
            printf("  L0 gate first4=[%.6f, %.6f, %.6f, %.6f]\n",
                   m->hb[0], m->hb[1], m->hb[2], m->hb[3]);
        }
#endif

        qwen_silu(m->hb, HD);
        // SiLU(gate) * up (vectorized element-wise multiply)
        vDSP_vmul(m->hb, 1, m->hb2, 1, m->hb, 1, (vDSP_Length)HD);

        float ffn_out[QWEN_DIM];
        #if USE_ANE_PROJECTIONS
        ane_project(m->k_down[l], m->hb, ffn_out, HD, D);
        #else
        cpu_project(m->w_down[l], m->hb, ffn_out, HD, D);
        #endif

        // Residual (vectorized)
        vDSP_vadd(m->x, 1, ffn_out, 1, m->x, 1, (vDSP_Length)D);

#if QWEN_DEBUG
        if (l < 3 && pos == 0) {
            float hn = 0;
            for (int i = 0; i < D; i++) hn += m->x[i] * m->x[i];
            printf("  C hidden[%d] norm=%.4f first4=[%.4f, %.4f, %.4f, %.4f]\n",
                   l+1, sqrtf(hn), m->x[0], m->x[1], m->x[2], m->x[3]);
        }
#endif
    }

    // Final RMSNorm
    qwen_rmsnorm(m->xb, m->x, m->rms_final, D);

#if QWEN_DEBUG
    if (m->pos < 2) {
        float fn = 0;
        for (int i = 0; i < D; i++) fn += m->xb[i] * m->xb[i];
        printf("  Final hidden norm=%.4f (first 4: %.6f %.6f %.6f %.6f)\n",
               sqrtf(fn), m->xb[0], m->xb[1], m->xb[2], m->xb[3]);
    }
#endif

    // LM head via Accelerate BLAS: logits = embed @ xb
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                QWEN_VOCAB, D,
                1.0f, m->embed, D,
                m->xb, 1,
                0.0f, m->logits, 1);

#if QWEN_DEBUG
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
#endif

    m->pos++;

    // Argmax (vDSP, single call over 151936 elements)
    float max_val;
    vDSP_Length max_idx_vdsp;
    vDSP_maxvi(m->logits, 1, &max_val, &max_idx_vdsp, (vDSP_Length)QWEN_VOCAB);
    return (int)max_idx_vdsp;
}

// ── Compile all ANE kernels ──────────────────────────────────────────

static void qwen_compile_kernels(QwenModel *m) {
#if USE_ANE_PROJECTIONS
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
    for (int c = 0; c < QWEN_LM_CHUNKS; c++) {
        float *chunk_weights = m->embed + c * QWEN_LM_CHUNK_SIZE * D;
        m->k_lmhead[c] = compile_conv_kernel(chunk_weights, D, QWEN_LM_CHUNK_SIZE, 1);
        if (!m->k_lmhead[c]) {
            printf("  LM head chunk %d FAILED to compile\n", c);
        }
    }
    printf("\nAll kernels compiled.\n");
#else
    printf("CPU-only mode (ANE kernel compilation skipped).\n");
    (void)m;
#endif
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

static void qwen_reset(QwenModel *m) {
    for (int l = 0; l < QWEN_LAYERS; l++) {
        memset(m->kv_cache_k[l], 0, QWEN_MAX_SEQ * QWEN_KV_DIM * sizeof(float));
        memset(m->kv_cache_v[l], 0, QWEN_MAX_SEQ * QWEN_KV_DIM * sizeof(float));
    }
    m->pos = 0;
}
