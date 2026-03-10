// stories_cpu_ops_opt.h — Cache-optimized embedding ops using vDSP
// Replaces strided element-by-element access with contiguous memcpy + vDSP_mtrans
#pragma once
#include "stories_cpu_ops.h"

// Embedding lookup: gather rows then transpose via vDSP_mtrans
// The original embed_lookup uses x[d*seq + t] = embed[tok*dim + d] in a double loop,
// causing stride-seq cache misses on every write. This version gathers contiguous rows
// into tmp[t*dim + d] = embed[tok*dim + d] via memcpy, then transposes with vDSP_mtrans.
// Requires caller-provided scratch buffer tmp of size seq*dim floats.
static void embed_lookup_opt(float *x, const float *embed, const uint16_t *tokens,
                             int dim, int seq, float *tmp) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= VOCAB) { memset(tmp + t * dim, 0, dim * sizeof(float)); continue; }
        memcpy(tmp + t * dim, embed + tok * dim, dim * sizeof(float));
    }
    vDSP_mtrans(tmp, 1, x, 1, (vDSP_Length)dim, (vDSP_Length)seq);
}

// Embedding backward: transpose then scatter-add via vDSP_vadd
// The original embed_backward uses d_embed[tok*dim + d] += dx[d*seq + t] with strided
// reads on dx. This version transposes dx [DIM, SEQ] -> tmp [SEQ, DIM] first, then
// accumulates contiguous rows with vDSP_vadd.
// Requires caller-provided scratch buffer tmp of size seq*dim floats.
static void embed_backward_opt(float *d_embed, const float *dx, const uint16_t *tokens,
                               int dim, int seq, float *tmp) {
    vDSP_mtrans(dx, 1, tmp, 1, (vDSP_Length)seq, (vDSP_Length)dim);
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= VOCAB) { continue; }
        vDSP_vadd(tmp + t * dim, 1,
                  d_embed + tok * dim, 1,
                  d_embed + tok * dim, 1,
                  (vDSP_Length)dim);
    }
}
