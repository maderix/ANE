// benchmark_ane.m — Measure ANE inference performance for Stories110M
#import "stories_io.h"
#import "stories_mil.h"

// Globals
float *embed, *rms_final;
LayerWeights lw[NLAYERS];
LayerKernels kern[NLAYERS];
IOSurfaceRef causal_mask_surf;

void load_checkpoint_inference(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Failed to open %s\n", path); exit(1); }
    CkptHdr hdr;
    fread(&hdr, sizeof(CkptHdr), 1, f);
    printf("Loading checkpoint: step=%d dim=%d layers=%d\n", hdr.step, hdr.dim, hdr.n_layers);

    for (int L=0; L<NLAYERS; L++) {
        lw[L] = layer_weights_alloc();
        fread(lw[L].Wq, WQ_SZ*4, 1, f);
        fread(lw[L].Wk, WQ_SZ*4, 1, f);
        fread(lw[L].Wv, WQ_SZ*4, 1, f);
        fread(lw[L].Wo, WO_SZ*4, 1, f);
        fread(lw[L].W1, W1_SZ*4, 1, f);
        fread(lw[L].W2, W2_SZ*4, 1, f);
        fread(lw[L].W3, W3_SZ*4, 1, f);
        fread(lw[L].rms_att, DIM*4, 1, f);
        fread(lw[L].rms_ffn, DIM*4, 1, f);
        // Skip Adam state: 2 * total params per layer
        size_t layer_state_size = (WQ_SZ*3 + WO_SZ + W1_SZ + W2_SZ + W3_SZ + DIM*2) * 2;
        fseek(f, layer_state_size * 4, SEEK_CUR);
    }
    rms_final = (float*)malloc(DIM*4);
    fread(rms_final, DIM*4, 1, f);
    fseek(f, DIM*2*4, SEEK_CUR); // skip rms_final adam
    embed = (float*)malloc(VOCAB*DIM*4);
    fread(embed, (size_t)VOCAB*DIM*4, 1, f);
    fclose(f);
}

// Compile one layer's kernels (subset of train_large.m)
static bool compile_fwd_kernels(LayerKernels *lk) {
    int fwdAttn_ins[] = { DIM*SEQ*2, DIM*2, WQ_SZ*2, WQ_SZ*2, WQ_SZ*2, WO_SZ*2, SEQ*SEQ*2 };
    lk->fwdAttn = compile_kern_mil_w(gen_sdpa_fwd_flex(), @{}, fwdAttn_ins, 7, 6*DIM*SEQ*2);

    int fwdFFN_ins[] = { DIM*SEQ*2, DIM*2, W1_SZ*2, W2_SZ*2, W3_SZ*2 };
    lk->fwdFFN = compile_kern_mil_w(gen_ffn_fwd_flex(), @{}, fwdFFN_ins, 5, (2*DIM+3*HIDDEN)*SEQ*2);

    return lk->fwdAttn && lk->fwdFFN;
}

static void update_fwd_ane_weights(LayerKernels *lk, LayerWeights *w, IOSurfaceRef cms) {
    // fwdAttn: x(0), rw(1), Wq(2), Wk(3), Wv(4), Wo(5), cm(6)
    io_write_fp16(lk->fwdAttn->inputs[1], w->rms_att, 1, DIM);
    io_write_fp16(lk->fwdAttn->inputs[2], w->Wq, DIM, DIM);
    io_write_fp16(lk->fwdAttn->inputs[3], w->Wk, DIM, DIM);
    io_write_fp16(lk->fwdAttn->inputs[4], w->Wv, DIM, DIM);
    io_write_fp16(lk->fwdAttn->inputs[5], w->Wo, DIM, DIM);
    
    // Swap causal mask surface
    CFRelease(lk->fwdAttn->inputs[6]);
    lk->fwdAttn->inputs[6] = (IOSurfaceRef)CFRetain(cms);
    
    // Update request with new input (this is tricky since request is opaque, 
    // but in stories_io.h it's created with these surfaces)
    // Actually, update_ane_weights in train_large just writes to existing.
    // Here we can just write once to CMS.
    static NSData *m_blob = nil; if(!m_blob) m_blob = get_mask_blob();
    IOSurfaceLock(cms, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(cms), (uint8_t*)[m_blob bytes]+128, SEQ*SEQ*2);
    IOSurfaceUnlock(cms, 0, NULL);

    // fwdFFN: x(0), rw(1), W1(2), W2(3), W3(4)
    io_write_fp16(lk->fwdFFN->inputs[1], w->rms_ffn, 1, DIM);
    io_write_fp16(lk->fwdFFN->inputs[2], w->W1, HIDDEN, DIM);
    io_write_fp16(lk->fwdFFN->inputs[3], w->W2, DIM, HIDDEN);
    io_write_fp16(lk->fwdFFN->inputs[4], w->W3, HIDDEN, DIM);
}

int main(int argc, char **argv) {
    @autoreleasepool {
    ane_init();
    mach_timebase_info(&g_tb);

    const char *ckpt = (argc > 1) ? argv[1] : "ane_stories110M_ckpt.bin";
    load_checkpoint_inference(ckpt);

    printf("Compiling ANE kernels...\n");
    uint64_t t_start = mach_absolute_time();
    
    causal_mask_surf = make_surface(SEQ*SEQ*2);

    for (int L=0; L<NLAYERS; L++) {
        if (!compile_fwd_kernels(&kern[L])) { printf("Compile failed layer %d\n", L); return 1; }
        update_fwd_ane_weights(&kern[L], &lw[L], causal_mask_surf);
    }
    uint64_t t_end = mach_absolute_time();
    printf("Kernels compiled in %.2f ms\n", tb_ms(t_end - t_start));

    // Warmup
    for(int i=0; i<3; i++) {
        for(int L=0; L<NLAYERS; L++) {
            ane_eval(kern[L].fwdAttn);
            ane_eval(kern[L].fwdFFN);
        }
    }

    printf("Benchmarking ANE Inference (SEQ=%d, LAYERS=%d)...\n", SEQ, NLAYERS);
    int iterations = 100;
    uint64_t t_bench_start = mach_absolute_time();
    
    for (int i=0; i<iterations; i++) {
        for (int L=0; L<NLAYERS; L++) {
            ane_eval(kern[L].fwdAttn);
            ane_eval(kern[L].fwdFFN);
        }
    }
    
    uint64_t t_bench_end = mach_absolute_time();
    double total_ms = tb_ms(t_bench_end - t_bench_start);
    double avg_ms = total_ms / iterations;
    
    // Calculate TFLOPS
    // Forward pass roughly: 2 * SEQ * DIM * (4*DIM + 3*HIDDEN) * NLAYERS FLOPs
    // 110M params, so roughly 2 * 110M * SEQ flops per pass
    double flops_per_pass = 2.0 * 110e6 * SEQ;
    double tflops = (flops_per_pass * 1e-12) / (avg_ms * 1e-3);
    
    printf("\nResults:\n");
    printf("  Average Forward Pass (SEQ=256): %.2f ms\n", avg_ms);
    printf("  Tokens / second:                %.2f\n", (double)SEQ * 1000.0 / avg_ms);
    printf("  Total parameters through ANE:   110M\n");
    printf("  ANE Forward Throughput:         %.2f TFLOPS\n", tflops);

    return 0;
    }
}
