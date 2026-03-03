// stories_cpu_ops_opt.h — Optimized CPU operations: NEON Adam, vectorized embedding
#pragma once
#include "stories_cpu_ops.h"
#include <arm_neon.h>

// ===== NEON-vectorized Adam optimizer =====
// ~3-3.5x faster than scalar version for large param counts
// Uses vrsqrteq_f32 + one Newton-Raphson step for fast reciprocal sqrt
static void adam_update_opt(float *w, const float *g, AdamState *s, int t,
                            float lr, float b1, float b2, float eps) {
    float bc1 = 1.0f - powf(b1, t);
    float bc2 = 1.0f - powf(b2, t);
    float inv_bc1 = 1.0f / bc1;
    float inv_bc2 = 1.0f / bc2;
    float one_minus_b1 = 1.0f - b1;
    float one_minus_b2 = 1.0f - b2;

    float32x4_t vb1       = vdupq_n_f32(b1);
    float32x4_t vb2       = vdupq_n_f32(b2);
    float32x4_t v1mb1     = vdupq_n_f32(one_minus_b1);
    float32x4_t v1mb2     = vdupq_n_f32(one_minus_b2);
    float32x4_t vinv_bc1  = vdupq_n_f32(inv_bc1);
    float32x4_t vinv_bc2  = vdupq_n_f32(inv_bc2);
    float32x4_t vneg_lr   = vdupq_n_f32(-lr);
    float32x4_t veps      = vdupq_n_f32(eps);

    size_t n = s->n;
    size_t i = 0;

    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        // Load
        float32x4_t vm = vld1q_f32(s->m + i);
        float32x4_t vv = vld1q_f32(s->v + i);
        float32x4_t vg = vld1q_f32(g + i);
        float32x4_t vw = vld1q_f32(w + i);

        // m = b1*m + (1-b1)*g
        vm = vmlaq_f32(vmulq_f32(vb1, vm), v1mb1, vg);
        // v = b2*v + (1-b2)*g*g
        float32x4_t g2 = vmulq_f32(vg, vg);
        vv = vmlaq_f32(vmulq_f32(vb2, vv), v1mb2, g2);

        // Store updated m, v
        vst1q_f32(s->m + i, vm);
        vst1q_f32(s->v + i, vv);

        // mhat = m / bc1, vhat = v / bc2
        float32x4_t mhat = vmulq_f32(vm, vinv_bc1);
        float32x4_t vhat = vmulq_f32(vv, vinv_bc2);

        // Fast reciprocal sqrt: vrsqrteq + one Newton-Raphson iteration
        // rsqrt_est ≈ 1/sqrt(vhat)
        float32x4_t rsqrt_est = vrsqrteq_f32(vhat);
        // Newton-Raphson: rsqrt *= (3 - vhat * rsqrt^2) / 2
        float32x4_t rsqrt_sq = vmulq_f32(rsqrt_est, rsqrt_est);
        float32x4_t nr_step = vrsqrtsq_f32(vhat, rsqrt_sq);
        rsqrt_est = vmulq_f32(rsqrt_est, nr_step);

        // w -= lr * mhat / (sqrt(vhat) + eps)
        // = w + (-lr) * mhat * (1/(sqrt(vhat) + eps))
        // Compute sqrt(vhat) from rsqrt: sqrt = vhat * rsqrt(vhat) (avoids division)
        float32x4_t sqrt_vhat = vmulq_f32(vhat, rsqrt_est);
        float32x4_t denom = vaddq_f32(sqrt_vhat, veps);

        // Use vdivq_f32 for the final division (accurate, eps-adjusted)
        float32x4_t update = vmulq_f32(vneg_lr, vdivq_f32(mhat, denom));
        vw = vaddq_f32(vw, update);

        vst1q_f32(w + i, vw);
    }

    // Scalar tail
    for (; i < n; i++) {
        s->m[i] = b1 * s->m[i] + one_minus_b1 * g[i];
        s->v[i] = b2 * s->v[i] + one_minus_b2 * g[i] * g[i];
        float mh = s->m[i] * inv_bc1;
        float vh = s->v[i] * inv_bc2;
        w[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}

// ===== Vectorized embedding lookup =====
// Gather rows from [VOCAB, DIM] row-major embed table → x [DIM, SEQ] channel-first
// Strategy: gather token rows into temp buffer [SEQ, DIM], then transpose via vDSP_mtrans
static void embed_lookup_opt(float *x, const float *embed, const uint16_t *tokens,
                             int dim, int seq, float *tmp) {
    // Gather: tmp[t*dim + d] = embed[tokens[t]*dim + d]
    for (int t = 0; t < seq; t++) {
        memcpy(tmp + t * dim, embed + tokens[t] * dim, dim * sizeof(float));
    }
    // Transpose [SEQ, DIM] → [DIM, SEQ]: x[d*seq + t] = tmp[t*dim + d]
    vDSP_mtrans(tmp, 1, x, 1, (vDSP_Length)dim, (vDSP_Length)seq);
}

// ===== Vectorized embedding backward =====
// Accumulate dE[tok] += dx[:,t] for each position
// Strategy: transpose dx [DIM, SEQ] → tmp [SEQ, DIM], then accumulate rows
static void embed_backward_opt(float *d_embed, const float *dx, const uint16_t *tokens,
                               int dim, int seq, float *tmp) {
    // Transpose [DIM, SEQ] → [SEQ, DIM]: tmp[t*dim + d] = dx[d*seq + t]
    vDSP_mtrans(dx, 1, tmp, 1, (vDSP_Length)seq, (vDSP_Length)dim);
    // Scatter-add: d_embed[tok*dim .. (tok+1)*dim] += tmp[t*dim .. (t+1)*dim]
    for (int t = 0; t < seq; t++) {
        vDSP_vadd(tmp + t * dim, 1,
                  d_embed + tokens[t] * dim, 1,
                  d_embed + tokens[t] * dim, 1,
                  (vDSP_Length)dim);
    }
}
