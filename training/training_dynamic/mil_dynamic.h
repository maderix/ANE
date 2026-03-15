// mil_dynamic.h — MIL generators for Qwen3-0.6B with GQA
// Q_DIM=2048 != DIM=1024, KV_DIM=1024, GQA_RATIO=2
// SDPA split: sdpaFwd (QKV proj + attention, no Wo) + woFwd (Wo matmul)
// Backward: qBwd + kvBwd (split from qkvBwd)
#pragma once
#include "io.h"

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

// Helper: generate a dynamic matmul within a MIL function
static void gen_dyn_matmul(NSMutableString *m, const char *prefix,
                           int ic, int oc, int seq,
                           int act_sp_off, int w_sp_off,
                           const char *input_var) {
    [m appendFormat:@"        tensor<int32, [4]> %s_ba = const()[name=string(\"%s_ba\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", prefix, prefix, act_sp_off];
    [m appendFormat:@"        tensor<int32, [4]> %s_sa = const()[name=string(\"%s_sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", prefix, prefix, ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %s_act = slice_by_size(x=%s,begin=%s_ba,size=%s_sa)[name=string(\"%s_act\")];\n", ic, seq, prefix, input_var, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_bw = const()[name=string(\"%s_bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", prefix, prefix, w_sp_off];
    [m appendFormat:@"        tensor<int32, [4]> %s_sw = const()[name=string(\"%s_sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", prefix, prefix, ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %s_wt = slice_by_size(x=%s,begin=%s_bw,size=%s_sw)[name=string(\"%s_wt\")];\n", ic, oc, prefix, input_var, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_ra = const()[name=string(\"%s_ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", prefix, prefix, ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_a2 = reshape(shape=%s_ra,x=%s_act)[name=string(\"%s_a2\")];\n", ic, seq, prefix, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_pm = const()[name=string(\"%s_pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n", prefix, prefix];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_a3 = transpose(perm=%s_pm,x=%s_a2)[name=string(\"%s_a3\")];\n", seq, ic, prefix, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_rw = const()[name=string(\"%s_rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", prefix, prefix, ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_W = reshape(shape=%s_rw,x=%s_wt)[name=string(\"%s_W\")];\n", ic, oc, prefix, prefix, prefix, prefix];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_yh = matmul(transpose_x=bF,transpose_y=bF,x=%s_a3,y=%s_W)[name=string(\"%s_yh\")];\n", seq, oc, prefix, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_yt = transpose(perm=%s_pm,x=%s_yh)[name=string(\"%s_yt\")];\n", oc, seq, prefix, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_ro = const()[name=string(\"%s_ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", prefix, prefix, oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %s_y = reshape(shape=%s_ro,x=%s_yt)[name=string(\"%s_y\")];\n", oc, seq, prefix, prefix, prefix, prefix];
}

// Simple dynamic matmul kernel: y = x @ W, input [1,IC,1,SEQ+OC], output [1,OC,1,SEQ]
static NSString *gen_dyn_matmul_mil(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    int sp = seq + oc;
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sp];
    gen_dyn_matmul(m, "mm", ic, oc, seq, 0, seq, "x");
    [m appendString:@"    } -> (mm_y);\n}\n"];
    return m;
}

// ===== SDPA forward with GQA (no Wo) =====
// Input: [1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM] fp16
//   sp[0:SEQ]                     = xnorm [DIM, SEQ]
//   sp[SEQ:SEQ+Q_DIM]             = Wq [DIM, Q_DIM]
//   sp[SEQ+Q_DIM:SEQ+Q_DIM+KVD]  = Wk [DIM, KV_DIM]
//   sp[SEQ+Q_DIM+KVD:...]         = Wv [DIM, KV_DIM]
// Output: [1, Q_DIM+Q_DIM+KV_DIM+KV_DIM+DIM, 1, SEQ] fp16
//   = concat(attn_out, Q_rope, K_rope, V, xnorm_pass)
static NSString *gen_sdpa_fwd_dynamic(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int sp_in = SDPA_FWD_SP;
    int out_ch = Q_DIM + Q_DIM + KV_DIM + KV_DIM + DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, sp_in];

    // Slice xnorm [1,DIM,1,SEQ]
    [m appendString:@"        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = slice_by_size(x=x,begin=bx,size=sx)[name=string(\"xn\")];\n", DIM, SEQ];

    // Slice Wq [1,DIM,1,Q_DIM]
    [m appendFormat:@"        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<int32, [4]> swq = const()[name=string(\"swq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, Q_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wq = slice_by_size(x=x,begin=bq,size=swq)[name=string(\"Wq\")];\n", DIM, Q_DIM];

    // Slice Wk [1,DIM,1,KV_DIM]
    [m appendFormat:@"        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ+Q_DIM];
    [m appendFormat:@"        tensor<int32, [4]> swk = const()[name=string(\"swk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wk = slice_by_size(x=x,begin=bk,size=swk)[name=string(\"Wk\")];\n", DIM, KV_DIM];

    // Slice Wv [1,DIM,1,KV_DIM]
    [m appendFormat:@"        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ+Q_DIM+KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wv = slice_by_size(x=x,begin=bv,size=swk)[name=string(\"Wv\")];\n", DIM, KV_DIM];

    // Reshape xnorm for matmul: [1,DIM,1,SEQ] → [1,1,DIM,SEQ] → [1,1,SEQ,DIM]
    [m appendFormat:@"        tensor<int32, [4]> r2 = const()[name=string(\"r2\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xn2 = reshape(shape=r2,x=xn)[name=string(\"xn2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];\n", SEQ, DIM];

    // Reshape weights
    [m appendFormat:@"        tensor<int32, [4]> rwq = const()[name=string(\"rwq\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, Q_DIM];
    [m appendFormat:@"        tensor<int32, [4]> rwk = const()[name=string(\"rwk\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wq2 = reshape(shape=rwq,x=Wq)[name=string(\"Wq2\")];\n", DIM, Q_DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wk2 = reshape(shape=rwk,x=Wk)[name=string(\"Wk2\")];\n", DIM, KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wv2 = reshape(shape=rwk,x=Wv)[name=string(\"Wv2\")];\n", DIM, KV_DIM];

    // QKV matmul
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    // Q: [1,1,SEQ,DIM] @ [1,1,DIM,Q_DIM] → [1,1,SEQ,Q_DIM]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq2)[name=string(\"qm\")];\n", SEQ, Q_DIM];
    // K: [1,1,SEQ,DIM] @ [1,1,DIM,KV_DIM] → [1,1,SEQ,KV_DIM]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk2)[name=string(\"km\")];\n", SEQ, KV_DIM];
    // V: same as K
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv2)[name=string(\"vm\")];\n", SEQ, KV_DIM];

    // Transpose back: [1,1,SEQ,X] → [1,1,X,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];\n", KV_DIM, SEQ];

    // Reshape to [1,X,1,SEQ]
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> kvsh = const()[name=string(\"kvsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = reshape(shape=qsh,x=qt)[name=string(\"qf\")];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = reshape(shape=kvsh,x=kt)[name=string(\"kf\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = reshape(shape=kvsh,x=vt)[name=string(\"vf\")];\n", KV_DIM, SEQ];

    // Reshape to heads for attention
    // Q: [1,Q_DIM,1,SEQ] → [1,HEADS,HD,SEQ] → transpose → [1,HEADS,SEQ,HD]
    [m appendFormat:@"        tensor<int32, [4]> qhsh = const()[name=string(\"qhsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qhsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    // K: [1,KV_DIM,1,SEQ] → [1,KV_HEADS,HD,SEQ] → [1,KV_HEADS,SEQ,HD]
    [m appendFormat:@"        tensor<int32, [4]> khsh = const()[name=string(\"khsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=khsh,x=kf)[name=string(\"rk\")];\n", KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", KV_HEADS, SEQ, HD];
    // V: same reshape as K
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=khsh,x=vf)[name=string(\"rv\")];\n", KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", KV_HEADS, SEQ, HD];

    // RoPE on Q: [1,HEADS,SEQ,HD]
    int pairs_q = SEQ * HD / 2;
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> rope_cos = const()[name=string(\"rc\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];\n", SEQ, HD, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> rope_sin = const()[name=string(\"rs\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];\n", SEQ, HD, SEQ, HD];
    [m appendFormat:@"        tensor<int32, [4]> rp_sh = const()[name=string(\"rp_sh\"), val=tensor<int32, [4]>([1,%d,%d,2])];\n", HEADS, pairs_q];
    [m appendFormat:@"        tensor<int32, [4]> rp_s1 = const()[name=string(\"rp_s1\"), val=tensor<int32, [4]>([1,%d,%d,1])];\n", HEADS, pairs_q];
    [m appendString:@"        tensor<int32, [4]> rp_b0 = const()[name=string(\"rp_b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendString:@"        tensor<int32, [4]> rp_b1 = const()[name=string(\"rp_b1\"), val=tensor<int32, [4]>([0,0,0,1])];\n"];
    [m appendString:@"        fp16 neg1 = const()[name=string(\"neg1\"), val=fp16(-1)];\n"];
    [m appendString:@"        int32 rpax = const()[name=string(\"rpax\"), val=int32(3)];\n"];
    [m appendString:@"        bool rpil = const()[name=string(\"rpil\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<int32, [4]> rp_bk_q = const()[name=string(\"rp_bk_q\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, SEQ, HD];

    // rotate_half(q)
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,2]> q_p = reshape(shape=rp_sh,x=q)[name=string(\"q_p\")];\n", HEADS, pairs_q];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> q_e = slice_by_size(x=q_p,begin=rp_b0,size=rp_s1)[name=string(\"q_e\")];\n", HEADS, pairs_q];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> q_o = slice_by_size(x=q_p,begin=rp_b1,size=rp_s1)[name=string(\"q_o\")];\n", HEADS, pairs_q];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> nq = mul(x=q_o,y=neg1)[name=string(\"nq\")];\n", HEADS, pairs_q];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,2]> qrp = concat(axis=rpax,interleave=rpil,values=(nq,q_e))[name=string(\"qrp\")];\n", HEADS, pairs_q];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q_rot = reshape(shape=rp_bk_q,x=qrp)[name=string(\"q_rot\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qc = mul(x=q,y=rope_cos)[name=string(\"qc\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qrs = mul(x=q_rot,y=rope_sin)[name=string(\"qrs\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q_rope = add(x=qc,y=qrs)[name=string(\"q_rope\")];\n", HEADS, SEQ, HD];

    // RoPE on K: [1,KV_HEADS,SEQ,HD]
    int pairs_k = SEQ * HD / 2;
    [m appendFormat:@"        tensor<int32, [4]> rp_sh_k = const()[name=string(\"rp_sh_k\"), val=tensor<int32, [4]>([1,%d,%d,2])];\n", KV_HEADS, pairs_k];
    [m appendFormat:@"        tensor<int32, [4]> rp_s1_k = const()[name=string(\"rp_s1_k\"), val=tensor<int32, [4]>([1,%d,%d,1])];\n", KV_HEADS, pairs_k];
    [m appendFormat:@"        tensor<int32, [4]> rp_bk_k = const()[name=string(\"rp_bk_k\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", KV_HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,2]> k_p = reshape(shape=rp_sh_k,x=k)[name=string(\"k_p\")];\n", KV_HEADS, pairs_k];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> k_e = slice_by_size(x=k_p,begin=rp_b0,size=rp_s1_k)[name=string(\"k_e\")];\n", KV_HEADS, pairs_k];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> k_o = slice_by_size(x=k_p,begin=rp_b1,size=rp_s1_k)[name=string(\"k_o\")];\n", KV_HEADS, pairs_k];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> nk = mul(x=k_o,y=neg1)[name=string(\"nk\")];\n", KV_HEADS, pairs_k];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,2]> krp = concat(axis=rpax,interleave=rpil,values=(nk,k_e))[name=string(\"krp\")];\n", KV_HEADS, pairs_k];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_rot = reshape(shape=rp_bk_k,x=krp)[name=string(\"k_rot\")];\n", KV_HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kc = mul(x=k,y=rope_cos)[name=string(\"kc\")];\n", KV_HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> krs = mul(x=k_rot,y=rope_sin)[name=string(\"krs\")];\n", KV_HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_rope = add(x=kc,y=krs)[name=string(\"k_rope\")];\n", KV_HEADS, SEQ, HD];

    // GQA: tile K,V from KV_HEADS to HEADS
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    // For GQA_RATIO=2: concat(k_rope, k_rope) along head dim
    NSMutableString *k_vals = [NSMutableString string];
    NSMutableString *v_vals = [NSMutableString string];
    for (int r = 0; r < GQA_RATIO; r++) {
        if (r > 0) { [k_vals appendString:@","]; [v_vals appendString:@","]; }
        [k_vals appendString:@"k_rope"]; [v_vals appendString:@"v"];
    }
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_tiled = concat(axis=cax,interleave=cid,values=(%@))[name=string(\"ktile\")];\n", HEADS, SEQ, HD, k_vals];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v_tiled = concat(axis=cax,interleave=cid,values=(%@))[name=string(\"vtile\")];\n", HEADS, SEQ, HD, v_vals];

    // Q_rope @ K_tiled^T → [1,HEADS,SEQ,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q_rope,y=k_tiled)[name=string(\"mm1\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS, SEQ, SEQ];

    // Causal mask
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ, SEQ, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS, SEQ, SEQ];

    // Softmax
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS, SEQ, SEQ];

    // scores @ V_tiled → [1,HEADS,SEQ,HD]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=v_tiled)[name=string(\"mm2\")];\n", HEADS, SEQ, HD];

    // Reshape attn_out to [1,Q_DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=qsh,x=at)[name=string(\"ra\")];\n", Q_DIM, SEQ];

    // Convert RoPE'd Q,K back to flat layout for backward
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qrt = transpose(perm=pm,x=q_rope)[name=string(\"qrt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qrf = reshape(shape=qsh,x=qrt)[name=string(\"qrf\")];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> krt = transpose(perm=pm,x=k_rope)[name=string(\"krt\")];\n", KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> krf = reshape(shape=kvsh,x=krt)[name=string(\"krf\")];\n", KV_DIM, SEQ];

    // Output: concat(attn_out[Q_DIM], Q_rope[Q_DIM], K_rope[KV_DIM], V[KV_DIM], xnorm[DIM])
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(af,qrf,krf,vf,xn))[name=string(\"cat\")];\n", out_ch, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// woFwd: attn_out[Q_DIM,SEQ] @ Wo → o_out[DIM,SEQ]
// Simple dyn_matmul: IC=Q_DIM, OC=DIM
static NSString *gen_wo_fwd_dynamic(void) {
    return gen_dyn_matmul_mil(Q_DIM, DIM, SEQ);
}

// ===== Fused FFN forward: W1,W3 + SiLU + W2 + residual =====
// Same structure as before, just with Qwen3 DIM=1024, HIDDEN=3072
static NSString *gen_ffn_fused_dynamic(void) {
    int sp_in = FFN_FUSED_SP;
    int out_ch = DIM + 3*HIDDEN;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, sp_in];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];

    // Slice x2norm, x2, W1, W3, W2_orig
    [m appendString:@"        tensor<int32, [4]> b_xn = const()[name=string(\"b_xn\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> s_ds = const()[name=string(\"s_ds\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2norm = slice_by_size(x=x,begin=b_xn,size=s_ds)[name=string(\"x2norm\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b_x2 = const()[name=string(\"b_x2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = slice_by_size(x=x,begin=b_x2,size=s_ds)[name=string(\"x2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b_w1 = const()[name=string(\"b_w1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ];
    [m appendFormat:@"        tensor<int32, [4]> s_wh = const()[name=string(\"s_wh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W1 = slice_by_size(x=x,begin=b_w1,size=s_wh)[name=string(\"W1\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<int32, [4]> b_w3 = const()[name=string(\"b_w3\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ+HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W3 = slice_by_size(x=x,begin=b_w3,size=s_wh)[name=string(\"W3\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<int32, [4]> b_w2 = const()[name=string(\"b_w2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ+2*HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W2r = slice_by_size(x=x,begin=b_w2,size=s_wh)[name=string(\"W2r\")];\n", DIM, HIDDEN];

    // xnorm matmul
    [m appendFormat:@"        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xn2 = reshape(shape=rd,x=x2norm)[name=string(\"xn2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W12 = reshape(shape=rw,x=W1)[name=string(\"W12\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W32 = reshape(shape=rw,x=W3)[name=string(\"W32\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W12)[name=string(\"h1m\")];\n", SEQ, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W32)[name=string(\"h3m\")];\n", SEQ, HIDDEN];

    // Reshape back
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];\n", HIDDEN, SEQ];

    // SiLU + gate
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n", HIDDEN, SEQ];

    // gate @ W2: W2 is [DIM, HIDDEN] stored as-is, transpose inside kernel
    [m appendFormat:@"        tensor<int32, [4]> rg = const()[name=string(\"rg\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> g2 = reshape(shape=rg,x=gate)[name=string(\"g2\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> gt = transpose(perm=pm,x=g2)[name=string(\"gtt\")];\n", SEQ, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W22 = reshape(shape=rw,x=W2r)[name=string(\"W22\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W2t = transpose(perm=pm,x=W22)[name=string(\"W2t\")];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> fm = matmul(transpose_x=bF,transpose_y=bF,x=gt,y=W2t)[name=string(\"fm\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> ft = transpose(perm=pm,x=fm)[name=string(\"ft\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rd2 = const()[name=string(\"rd2\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> ffn_out = reshape(shape=rd2,x=ft)[name=string(\"ffn_out\")];\n", DIM, SEQ];

    // Residual: x_next = x2 + alpha * ffn_out
    float alpha = 1.0f / sqrtf(2.0f * NLAYERS);
    [m appendFormat:@"        fp16 res_alpha = const()[name=string(\"res_alpha\"), val=fp16(%g)];\n", alpha];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> ffn_scaled = mul(x=ffn_out,y=res_alpha)[name=string(\"ffn_sc\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x_next = add(x=x2,y=ffn_scaled)[name=string(\"x_next\")];\n", DIM, SEQ];

    // Output: concat(x_next, h1, h3, gate)
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(x_next,h1,h3,gate))[name=string(\"cat\")];\n", out_ch, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ===== Backward kernels =====

// ffnBwdW2t: dffn @ W2 → dsilu_raw (IC=DIM, OC=HIDDEN)
static NSString *gen_ffn_bwd_w2t_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, HIDDEN, SEQ);
}

// ffnBwdW13t: dh1 @ W1 + dh3 @ W3 → dx_ffn (IC=HIDDEN, two matmuls added)
static NSString *gen_ffn_bwd_w13t_dynamic(void) {
    int sp_in = FFN_BWD_W13T_SP;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", HIDDEN, sp_in];

    [m appendFormat:@"        tensor<int32, [4]> sh = const()[name=string(\"sh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh1 = slice_by_size(x=x,begin=b0,size=sh)[name=string(\"dh1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh3 = slice_by_size(x=x,begin=b1,size=sh)[name=string(\"dh3\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W1t = slice_by_size(x=x,begin=b2,size=sw)[name=string(\"W1t\")];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W3t = slice_by_size(x=x,begin=b3,size=sw)[name=string(\"W3t\")];\n", HIDDEN, DIM];

    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh12 = reshape(shape=ra,x=dh1)[name=string(\"dh12\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh1t = transpose(perm=pm,x=dh12)[name=string(\"dh1t\")];\n", SEQ, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh32 = reshape(shape=ra,x=dh3)[name=string(\"dh32\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh3t = transpose(perm=pm,x=dh32)[name=string(\"dh3t\")];\n", SEQ, HIDDEN];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W1t2 = reshape(shape=rw,x=W1t)[name=string(\"W1t2\")];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W3t2 = reshape(shape=rw,x=W3t)[name=string(\"W3t2\")];\n", HIDDEN, DIM];

    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dx1m = matmul(transpose_x=bF,transpose_y=bF,x=dh1t,y=W1t2)[name=string(\"dx1m\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dx3m = matmul(transpose_x=bF,transpose_y=bF,x=dh3t,y=W3t2)[name=string(\"dx3m\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxm = add(x=dx1m,y=dx3m)[name=string(\"dxm\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxt = transpose(perm=pm,x=dxm)[name=string(\"dxt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = reshape(shape=ro,x=dxt)[name=string(\"dx\")];\n", DIM, SEQ];
    [m appendString:@"    } -> (dx);\n}\n"];
    return m;
}

// wotBwd: dy @ Wo → da (IC=DIM, OC=Q_DIM)
static NSString *gen_wot_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, Q_DIM, SEQ);
}

// qBwd: dq @ Wq → dx_q (IC=Q_DIM, OC=DIM)
static NSString *gen_q_bwd_dynamic(void) {
    return gen_dyn_matmul_mil(Q_DIM, DIM, SEQ);
}

// kvBwd: dk @ Wk + dv @ Wv → dx_kv (IC=KV_DIM)
// Input: [1, KV_DIM, 1, 2*SEQ+2*DIM] fp16
// Same pattern as ffnBwdW13t but with KV_DIM channels
static NSString *gen_kv_bwd_dynamic(void) {
    int sp_in = KV_BWD_SP;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", KV_DIM, sp_in];

    [m appendFormat:@"        tensor<int32, [4]> sh = const()[name=string(\"sh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dk = slice_by_size(x=x,begin=b0,size=sh)[name=string(\"dk\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dv = slice_by_size(x=x,begin=b1,size=sh)[name=string(\"dv\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wkt = slice_by_size(x=x,begin=b2,size=sw)[name=string(\"Wkt\")];\n", KV_DIM, DIM];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wvt = slice_by_size(x=x,begin=b3,size=sw)[name=string(\"Wvt\")];\n", KV_DIM, DIM];

    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dk2 = reshape(shape=ra,x=dk)[name=string(\"dk2\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dkt = transpose(perm=pm,x=dk2)[name=string(\"dkt\")];\n", SEQ, KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dv2 = reshape(shape=ra,x=dv)[name=string(\"dv2\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dvt = transpose(perm=pm,x=dv2)[name=string(\"dvt\")];\n", SEQ, KV_DIM];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", KV_DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wkt2 = reshape(shape=rw,x=Wkt)[name=string(\"Wkt2\")];\n", KV_DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wvt2 = reshape(shape=rw,x=Wvt)[name=string(\"Wvt2\")];\n", KV_DIM, DIM];

    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxk = matmul(transpose_x=bF,transpose_y=bF,x=dkt,y=Wkt2)[name=string(\"dxk\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxv = matmul(transpose_x=bF,transpose_y=bF,x=dvt,y=Wvt2)[name=string(\"dxv\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxm = add(x=dxk,y=dxv)[name=string(\"dxm\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxt = transpose(perm=pm,x=dxm)[name=string(\"dxt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = reshape(shape=ro,x=dxt)[name=string(\"dx\")];\n", DIM, SEQ];
    [m appendString:@"    } -> (dx);\n}\n"];
    return m;
}

// SDPA backward part 1: recompute attention + dV, dp
// Uses tiled K,V at HEADS dimension (CPU pre-tiles)
// Input: [1, 2*Q_DIM+2*Q_DIM, 1, SEQ] fp16 = (Q, K_tiled, V_tiled, da)
// Output: [1, Q_DIM+2*SCORE_CH, 1, SEQ] fp16 = (dV_full, probs, dp)
static NSString *gen_sdpa_bwd1_noweight(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int in_ch = 4*Q_DIM;  // Q + K_tiled + V_tiled + da, all at Q_DIM (HEADS*HD)
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", in_ch, SEQ];

    // Slice Q,K_tiled,V_tiled,da — all [Q_DIM, SEQ]
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", Q_DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", Q_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*Q_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 3*Q_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> da = slice_by_size(x=x,begin=b3,size=sz)[name=string(\"s3\")];\n", Q_DIM, SEQ];

    // Reshape to heads [1,HEADS,HD,SEQ] → [1,HEADS,SEQ,HD]
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> vr = reshape(shape=rsh,x=vf)[name=string(\"rv\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=vr)[name=string(\"tv\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dr = reshape(shape=rsh,x=da)[name=string(\"rd\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dat = transpose(perm=pm,x=dr)[name=string(\"td\")];\n", HEADS, SEQ, HD];

    // Recompute attention scores
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ, SEQ, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS, SEQ, SEQ];

    // dV = probs^T @ da, dp = da @ V^T
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=dat)[name=string(\"dv\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=dat,y=v)[name=string(\"dp\")];\n", HEADS, SEQ, SEQ];

    // Reshape dV to [Q_DIM, SEQ] (will be reduced to KV_DIM on CPU)
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];\n", Q_DIM, SEQ];

    // Flatten probs and dp
    [m appendFormat:@"        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = reshape(shape=scs,x=probs)[name=string(\"pf\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = reshape(shape=scs,x=dp4)[name=string(\"dpf\")];\n", SCORE_CH, SEQ];

    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=string(\"cat\")];\n", Q_DIM+2*SCORE_CH, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 2: probs, dp, Q, K_tiled → dQ, dK_full
// Input: [1, 2*SCORE_CH + 2*Q_DIM, 1, SEQ]
// Output: [1, 2*Q_DIM, 1, SEQ] = (dQ, dK_full)
static NSString *gen_sdpa_bwd2(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int bwd2_in = 2*SCORE_CH + 2*Q_DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", bwd2_in, SEQ];

    [m appendFormat:@"        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=string(\"s0\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=string(\"s1\")];\n", SCORE_CH, SEQ];

    [m appendFormat:@"        tensor<int32, [4]> sz_q = const()[name=string(\"szq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b2,size=sz_q)[name=string(\"s2\")];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH+Q_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b3,size=sz_q)[name=string(\"s3\")];\n", Q_DIM, SEQ];

    [m appendFormat:@"        tensor<int32, [4]> ssh = const()[name=string(\"ssh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = reshape(shape=ssh,x=pf)[name=string(\"rp\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp = reshape(shape=ssh,x=dpf)[name=string(\"rdp\")];\n", HEADS, SEQ, SEQ];

    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", HEADS, SEQ, HD];

    // Softmax backward: ds = (dp - sum(dp*probs)) * probs * scale
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> pdp = mul(x=probs,y=dp)[name=string(\"pdp\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=string(\"rs\")];\n", HEADS, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dps = sub(x=dp,y=spdp)[name=string(\"dps\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds0 = mul(x=probs,y=dps)[name=string(\"ds0\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds = mul(x=ds0,y=scv)[name=string(\"ds\")];\n", HEADS, SEQ, SEQ];

    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)[name=string(\"dk\")];\n", HEADS, SEQ, HD];

    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dqt = transpose(perm=pm,x=dq4)[name=string(\"dqt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dkt = transpose(perm=pm,x=dk4)[name=string(\"dkt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> fs = const()[name=string(\"fs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dqf = reshape(shape=fs,x=dqt)[name=string(\"dqf\")];\n", Q_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dkf = reshape(shape=fs,x=dkt)[name=string(\"dkf\")];\n", Q_DIM, SEQ];

    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf))[name=string(\"cat\")];\n", 2*Q_DIM, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Causal mask blob
static NSData *g_mask_blob = nil;
static NSData *get_mask_blob(void) {
    if (!g_mask_blob) {
        _Float16 *mask = (_Float16*)calloc(SEQ*SEQ, sizeof(_Float16));
        for(int t=0;t<SEQ;t++) for(int t2=0;t2<SEQ;t2++)
            mask[t*SEQ+t2] = (t2<=t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
        g_mask_blob = build_blob_fp16(mask, SEQ*SEQ);
        free(mask);
    }
    return g_mask_blob;
}

// RoPE cos/sin blobs [1, 1, SEQ, HD]
static NSData *g_rope_cos_blob = nil;
static NSData *g_rope_sin_blob = nil;

static NSData *get_rope_cos_blob(void) {
    if (!g_rope_cos_blob) {
        _Float16 *buf = (_Float16*)calloc(SEQ * HD, sizeof(_Float16));
        for (int p = 0; p < SEQ; p++)
            for (int i = 0; i < HD/2; i++) {
                float theta = p / powf(10000.0f, 2.0f * i / (float)HD);
                _Float16 cv = (_Float16)cosf(theta);
                buf[p * HD + 2*i] = cv;
                buf[p * HD + 2*i + 1] = cv;
            }
        g_rope_cos_blob = build_blob_fp16(buf, SEQ * HD);
        free(buf);
    }
    return g_rope_cos_blob;
}

static NSData *get_rope_sin_blob(void) {
    if (!g_rope_sin_blob) {
        _Float16 *buf = (_Float16*)calloc(SEQ * HD, sizeof(_Float16));
        for (int p = 0; p < SEQ; p++)
            for (int i = 0; i < HD/2; i++) {
                float theta = p / powf(10000.0f, 2.0f * i / (float)HD);
                _Float16 sv = (_Float16)sinf(theta);
                buf[p * HD + 2*i] = sv;
                buf[p * HD + 2*i + 1] = sv;
            }
        g_rope_sin_blob = build_blob_fp16(buf, SEQ * HD);
        free(buf);
    }
    return g_rope_sin_blob;
}
