// test_bridge.m — functional tests for ane_bridge API
// Tests both BLOBFILE (upstream compat) and dynamic IOSurface (our approach)

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>
#include "ane_bridge.h"

static mach_timebase_info_data_t g_tb;
static double ms(void) {
    return (double)mach_absolute_time() * g_tb.numer / g_tb.denom / 1e6;
}

static int passed = 0, failed = 0;
#define PASS(msg) do { printf("  [PASS] %s\n", msg); passed++; } while(0)
#define FAIL(msg) do { printf("  [FAIL] %s\n", msg); failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(msg); else FAIL(msg); } while(0)

// Correct MIL header — must match exactly what ANE compiler expects
#define MIL_HDR \
    "program(1.3)\n" \
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

// Dynamic weight matmul: y[1,N,M] = W[1,N,K] @ x[1,K,M]
// x declared as [1,K,1,M] (4D), W as [1,N,K] (3D dynamic weight)
static const char *mil_dyn_matmul(int N, int K, int M) {
    static char buf[2048];
    snprintf(buf, sizeof(buf),
        MIL_HDR
        "    func main<ios18>(\n"
        "        tensor<fp16, [1, %d, 1, %d]> x,\n"
        "        tensor<fp16, [1, %d, %d]> W) {\n"
        "        tensor<int32, [3]> sh = const()[name=string(\"sh\"), val=tensor<int32, [3]>([1,%d,%d])];\n"
        "        tensor<fp16, [1,%d,%d]> x3 = reshape(shape=sh,x=x)[name=string(\"rx\")];\n"
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"
        "        tensor<fp16, [1,%d,%d]> out = matmul(transpose_x=bF,transpose_y=bF,x=W,y=x3)[name=string(\"mm\")];\n"
        "    } -> (out);\n"
        "}\n",
        K, M,   // x: [1,K,1,M]
        N, K,   // W: [1,N,K]
        K, M,   // reshape sh
        K, M,   // x3
        N, M    // out
    );
    return buf;
}

// No-weight MIL: elementwise add x+x = 2x
static const char *mil_scale2(int rows, int cols) {
    static char buf[1024];
    snprintf(buf, sizeof(buf),
        MIL_HDR
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [1,%d,1,%d]> out = add(x=x,y=x)[name=string(\"out\")];\n"
        "    } -> (out);\n"
        "}\n",
        rows, cols, rows, cols
    );
    return buf;
}

static void fill_fp16(void *buf, float val, int count) {
    uint16_t *p = (uint16_t *)buf;
    _Float16 fval = (_Float16)val;
    uint16_t bits;
    memcpy(&bits, &fval, 2);
    for (int i = 0; i < count; i++) p[i] = bits;
}

static float read_fp16_elem(const void *buf, int idx) {
    const uint16_t *p = (const uint16_t *)buf;
    _Float16 v;
    memcpy(&v, &p[idx], 2);
    return (float)v;
}

// --- Test 1: init ---
static void test_init(void) {
    printf("\n[1] ane_bridge_init\n");
    int r = ane_bridge_init();
    CHECK(r == 0, "init returns 0");
}

// --- Test 2: compile_dyn — compile once with dynamic weight ---
static ANEKernelHandle *g_dyn_kern = NULL;
static void test_compile_dyn(void) {
    printf("\n[2] ane_bridge_compile_dyn (dynamic weight IOSurface)\n");

    int N=64, K=64, M=64;
    const char *mil = mil_dyn_matmul(N, K, M);
    size_t mil_len = strlen(mil);

    // x: [1,K,1,M] fp16, W: [1,N,K] fp16, out: [1,N,M] fp16
    size_t in_sz  = (size_t)K * M * 2;
    size_t w_sz   = (size_t)N * K * 2;
    size_t out_sz = (size_t)N * M * 2;

    double t0 = ms();
    g_dyn_kern = ane_bridge_compile_dyn(mil, mil_len, 1, &in_sz, 1, &w_sz, out_sz);
    double elapsed = ms() - t0;

    CHECK(g_dyn_kern != NULL, "compile_dyn returns non-NULL handle");
    printf("  Compile time: %.1fms\n", elapsed);
}

// --- Test 3: write_weight + eval — dynamic weights actually update ---
static void test_dynamic_weight_update(void) {
    printf("\n[3] Dynamic weight update (write W, eval, check output changes)\n");
    if (!g_dyn_kern) { FAIL("no kernel from test 2"); return; }

    int N=64, K=64, M=64;
    size_t in_sz  = (size_t)K * M * 2;
    size_t w_sz   = (size_t)N * K * 2;
    size_t out_sz = (size_t)N * M * 2;

    uint16_t *xbuf   = (uint16_t *)malloc(in_sz);
    uint16_t *wbuf_A = (uint16_t *)calloc(N*K, 2);
    uint16_t *wbuf_B = (uint16_t *)calloc(N*K, 2);
    uint16_t *out    = (uint16_t *)malloc(out_sz);

    // x = all 1.0
    fill_fp16(xbuf, 1.0f, K*M);

    // W_A = identity (scale=1): diagonal 1.0
    _Float16 one = 1.0f; uint16_t one_bits; memcpy(&one_bits, &one, 2);
    _Float16 two = 2.0f; uint16_t two_bits; memcpy(&two_bits, &two, 2);
    for (int i = 0; i < N && i < K; i++) wbuf_A[i*K + i] = one_bits;
    for (int i = 0; i < N && i < K; i++) wbuf_B[i*K + i] = two_bits;

    // Eval A: W = identity
    ane_bridge_write_input(g_dyn_kern, 0, xbuf, in_sz);
    ane_bridge_write_weight(g_dyn_kern, 0, wbuf_A, w_sz);
    bool ok = ane_bridge_eval(g_dyn_kern);
    CHECK(ok, "eval A succeeds");
    ane_bridge_read_output(g_dyn_kern, 0, out, out_sz);
    float sum_A = 0;
    for (int i = 0; i < N*M; i++) sum_A += read_fp16_elem(out, i);

    // Eval B: W = 2x identity, NO recompile
    double t0 = ms();
    ane_bridge_write_weight(g_dyn_kern, 0, wbuf_B, w_sz);
    double write_ms = ms() - t0;

    t0 = ms();
    ok = ane_bridge_eval(g_dyn_kern);
    double eval_ms = ms() - t0;

    CHECK(ok, "eval B succeeds after weight update");
    ane_bridge_read_output(g_dyn_kern, 0, out, out_sz);
    float sum_B = 0;
    for (int i = 0; i < N*M; i++) sum_B += read_fp16_elem(out, i);

    printf("  sum(out) W=identity: %.1f  W=2x: %.1f  ratio: %.2f (expect ~2.0)\n",
           sum_A, sum_B, sum_B / (sum_A + 1e-9f));
    CHECK(fabsf(sum_B / (sum_A + 1e-9f) - 2.0f) < 0.1f, "output doubled after weight update");
    printf("  write_weight: %.3fms  eval: %.2fms\n", write_ms, eval_ms);

    free(xbuf); free(wbuf_A); free(wbuf_B); free(out);
}

// --- Test 4: write_weight_f32 ---
static void test_write_weight_f32(void) {
    printf("\n[4] ane_bridge_write_weight_f32 (fp32 -> fp16 conversion)\n");
    if (!g_dyn_kern) { FAIL("no kernel"); return; }

    int N=64, K=64, M=64;
    float *w_fp32  = (float *)calloc(N*K, 4);
    uint16_t *out1 = (uint16_t *)malloc((size_t)N*M*2);
    uint16_t *out2 = (uint16_t *)malloc((size_t)N*M*2);
    uint16_t *xbuf = (uint16_t *)malloc((size_t)K*M*2);
    fill_fp16(xbuf, 1.0f, K*M);

    // 3x identity in fp32
    for (int i = 0; i < N && i < K; i++) w_fp32[i*K + i] = 3.0f;
    ane_bridge_write_input(g_dyn_kern, 0, xbuf, (size_t)K*M*2);
    ane_bridge_write_weight_f32(g_dyn_kern, 0, w_fp32, (size_t)N*K);
    bool ok = ane_bridge_eval(g_dyn_kern);
    CHECK(ok, "eval with fp32-written weight succeeds");
    ane_bridge_read_output(g_dyn_kern, 0, out1, (size_t)N*M*2);
    float sum = 0;
    for (int i = 0; i < N*M; i++) sum += read_fp16_elem(out1, i);
    float expected = 3.0f * N; // 3x identity: each row sums x (all 1s) scaled by 3
    // Actually identity matmul with x=all-1s: row i of output = row i of W dotted with x
    // row i of identity (scaled 3) = 3 at position i, 0 elsewhere. So out[i][t] = 3*x[i][t] = 3
    printf("  sum(out) with 3x identity (fp32 write): %.1f\n", sum);
    CHECK(fabsf(sum - 3.0f * N * M) < N*M*0.1f, "fp32 weight write produces correct output");

    free(w_fp32); free(out1); free(out2); free(xbuf);
}

// --- Test 5: copy_io ---
static void test_copy_io(void) {
    printf("\n[5] ane_bridge_copy_io (direct IOSurface-to-IOSurface)\n");

    // Compile a simple no-weight kernel: scale x by 2
    int R=32, C=32;
    const char *mil = mil_scale2(R, C);
    size_t sz = (size_t)R * C * 2;

    ANEKernelHandle *k1 = ane_bridge_compile_dyn(mil, strlen(mil), 1, &sz, 0, NULL, sz);
    ANEKernelHandle *k2 = ane_bridge_compile_dyn(mil, strlen(mil), 1, &sz, 0, NULL, sz);
    CHECK(k1 != NULL && k2 != NULL, "two scale2 kernels compiled");
    if (!k1 || !k2) { ane_bridge_free(k1); ane_bridge_free(k2); return; }

    // k1 input = 1.0, k1 out = 2.0, copy to k2 in, k2 out = 4.0
    uint16_t *buf = (uint16_t *)malloc(sz);
    uint16_t *out = (uint16_t *)malloc(sz);
    fill_fp16(buf, 1.0f, R*C);

    ane_bridge_write_input(k1, 0, buf, sz);
    ane_bridge_eval(k1);
    ane_bridge_copy_io(k1, 0, k2, 0);  // k2 input = k1 output (should be 2.0)
    ane_bridge_eval(k2);
    ane_bridge_read_output(k2, 0, out, sz);

    float val = read_fp16_elem(out, 0);
    printf("  input=1.0 -> k1(x2) -> copy -> k2(x2) -> output=%.1f (expect 4.0)\n", val);
    CHECK(fabsf(val - 4.0f) < 0.2f, "copy_io chains two kernels without CPU round-trip");

    free(buf); free(out);
    ane_bridge_free(k1);
    ane_bridge_free(k2);
}

// --- Test 6: begin/end realtime ---
static void test_realtime(void) {
    printf("\n[6] ane_bridge_begin/end_realtime\n");
    if (!g_dyn_kern) { FAIL("no kernel"); return; }

    // Just verify they don't crash. Jitter improvement only visible over many samples.
    ane_bridge_begin_realtime();
    bool ok = ane_bridge_eval(g_dyn_kern);
    ane_bridge_end_realtime();
    CHECK(ok, "eval inside realtime task succeeds");
    printf("  (p99 jitter improvement requires statistical measurement — see test_realtime_task.m)\n");
}

// --- Test 7: compile cache ---
static void test_compile_cache(void) {
    printf("\n[7] Compile cache (cache files written to ~/.ane_cache/)\n");

    int N=32, M=32;
    const char *mil = mil_scale2(N, M);
    size_t sz = (size_t)N*M*2;

    ANEKernelHandle *k = ane_bridge_compile_dyn(mil, strlen(mil), 1, &sz, 0, NULL, sz);
    CHECK(k != NULL, "kernel compiled for cache test");
    if (!k) return;

    // Verify cache files exist on disk
    // We need the hex ID — compile a fresh model to get it, then check cache
    NSString *home = NSHomeDirectory();
    NSString *cacheRoot = [home stringByAppendingPathComponent:@".ane_cache"];
    NSFileManager *fm = [NSFileManager defaultManager];
    // Dynamic-weight models produce net.plist only (no data file)
    NSArray *entries = [fm contentsOfDirectoryAtPath:cacheRoot error:nil];
    BOOL found_plist = NO;
    for (NSString *entry in entries) {
        NSString *plistPath = [[cacheRoot stringByAppendingPathComponent:entry]
                                stringByAppendingPathComponent:@"net.plist"];
        if ([fm fileExistsAtPath:plistPath]) { found_plist = YES; break; }
    }
    CHECK(found_plist, "compiled kernel cached to ~/.ane_cache/");

    // Free k first, then recompile — should hit cache (load only, no compile)
    ane_bridge_free(k);
    int before = ane_bridge_get_compile_count();
    ANEKernelHandle *k2 = ane_bridge_compile_dyn(mil, strlen(mil), 1, &sz, 0, NULL, sz);
    int after = ane_bridge_get_compile_count();
    CHECK(k2 != NULL, "cache hit returns valid kernel");
    CHECK(after == before, "cache hit does not increment compile count");
    ane_bridge_free(k2);
}

// --- Test 8: free ---
static void test_free(void) {
    printf("\n[8] ane_bridge_free\n");
    ane_bridge_free(g_dyn_kern);
    g_dyn_kern = NULL;
    PASS("free does not crash");
}

int main(void) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        printf("=== ane_bridge test suite ===\n");

        test_init();
        test_compile_dyn();
        test_dynamic_weight_update();
        test_write_weight_f32();
        test_copy_io();
        test_realtime();
        test_compile_cache();
        test_free();

        printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
        return failed > 0 ? 1 : 0;
    }
}
