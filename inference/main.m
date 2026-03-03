// main.m — Qwen2.5-0.5B inference on Apple Neural Engine
// Compiles ANE kernels for all linear projections, runs autoregressive decode.
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface \
//     -framework CoreML -framework Accelerate -ldl -lobjc \
//     -o qwen_ane main.m
//
// Run:
//   ./qwen_ane qwen05b.bin "Hello world"
//
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "qwen_ane_infer.h"

static QwenModel g_model;

static int load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    // Read config header
    int config[7];
    fread(config, sizeof(int), 7, f);
    int dim = config[0], hidden = config[1], n_layers = config[2];
    int n_heads = config[3], n_kv_heads = config[4], vocab = config[5];
    printf("Config: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d\n",
           dim, hidden, n_layers, n_heads, n_kv_heads, vocab);

    int q_dim = n_heads * QWEN_HEAD_DIM;
    int kv_dim = n_kv_heads * QWEN_HEAD_DIM;

    // Embedding
    g_model.embed = (float*)malloc((size_t)vocab * dim * sizeof(float));
    fread(g_model.embed, sizeof(float), (size_t)vocab * dim, f);

    // Per-layer
    for (int l = 0; l < n_layers; l++) {
        g_model.rms_att[l] = (float*)malloc(dim * sizeof(float));
        fread(g_model.rms_att[l], sizeof(float), dim, f);

        g_model.wq[l] = (float*)malloc((size_t)q_dim * dim * sizeof(float));
        fread(g_model.wq[l], sizeof(float), (size_t)q_dim * dim, f);
        g_model.wk[l] = (float*)malloc((size_t)kv_dim * dim * sizeof(float));
        fread(g_model.wk[l], sizeof(float), (size_t)kv_dim * dim, f);
        g_model.wv[l] = (float*)malloc((size_t)kv_dim * dim * sizeof(float));
        fread(g_model.wv[l], sizeof(float), (size_t)kv_dim * dim, f);
        g_model.wo[l] = (float*)malloc((size_t)q_dim * dim * sizeof(float)); // o_proj is [dim, q_dim]
        fread(g_model.wo[l], sizeof(float), (size_t)dim * q_dim, f);

        // Q/K/V biases
        g_model.q_bias[l] = (float*)malloc(q_dim * sizeof(float));
        g_model.k_bias[l] = (float*)malloc(kv_dim * sizeof(float));
        g_model.v_bias[l] = (float*)malloc(kv_dim * sizeof(float));
        fread(g_model.q_bias[l], sizeof(float), q_dim, f);
        fread(g_model.k_bias[l], sizeof(float), kv_dim, f);
        fread(g_model.v_bias[l], sizeof(float), kv_dim, f);

        g_model.rms_ffn[l] = (float*)malloc(dim * sizeof(float));
        fread(g_model.rms_ffn[l], sizeof(float), dim, f);

        g_model.w_gate[l] = (float*)malloc((size_t)hidden * dim * sizeof(float));
        fread(g_model.w_gate[l], sizeof(float), (size_t)hidden * dim, f);
        g_model.w_up[l] = (float*)malloc((size_t)hidden * dim * sizeof(float));
        fread(g_model.w_up[l], sizeof(float), (size_t)hidden * dim, f);
        g_model.w_down[l] = (float*)malloc((size_t)dim * hidden * sizeof(float));
        fread(g_model.w_down[l], sizeof(float), (size_t)dim * hidden, f);
    }

    g_model.rms_final = (float*)malloc(dim * sizeof(float));
    fread(g_model.rms_final, sizeof(float), dim, f);

    fclose(f);
    printf("Weights loaded (%.0f MB)\n",
           (float)ftell(f) / 1024 / 1024);
    return 0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s <weights.bin> <prompt>\n", argv[0]);
            return 1;
        }

        printf("=== Qwen2.5-0.5B ANE Inference ===\n\n");

        // Load weights
        printf("Loading weights...\n");
        if (load_weights(argv[1]) != 0) return 1;

        // Allocate buffers
        qwen_alloc(&g_model);

        // Compile ANE kernels
        printf("Compiling ANE kernels (169 total)...\n");
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        qwen_compile_kernels(&g_model);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double compile_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("Compile time: %.1fs\n\n", compile_sec);

        // Parse token IDs from argv[2] (space-separated)
        // argv[3] = max generation tokens
        int max_gen = 50;
        if (argc >= 4) max_gen = atoi(argv[3]);

        // Parse input token IDs
        int prompt_ids[2048];
        int n_prompt = 0;
        char *tok_str = strdup(argv[2]);
        char *saveptr;
        char *p = strtok_r(tok_str, " ", &saveptr);
        while (p && n_prompt < 2048) {
            prompt_ids[n_prompt++] = atoi(p);
            p = strtok_r(NULL, " ", &saveptr);
        }
        free(tok_str);
        printf("Prompt: %d tokens, generating up to %d\n", n_prompt, max_gen);

        clock_gettime(CLOCK_MONOTONIC, &t0);

        // Prefill: feed all prompt tokens
        int next = 0;
        for (int i = 0; i < n_prompt; i++) {
            next = qwen_forward(&g_model, prompt_ids[i]);
        }

        struct timespec t_prefill;
        clock_gettime(CLOCK_MONOTONIC, &t_prefill);
        double prefill_sec = (t_prefill.tv_sec - t0.tv_sec) + (t_prefill.tv_nsec - t0.tv_nsec) / 1e9;
        printf("Prefill: %d tokens in %.2fs (%.1f t/s)\n", n_prompt, prefill_sec, n_prompt / prefill_sec);

        // Generate
        int eos = 151645;  // <|im_end|>
        int eos2 = 151643; // <|endoftext|>
        printf("OUT:");
        for (int i = 0; i < max_gen; i++) {
            printf(" %d", next);
            fflush(stdout);
            if (next == eos || next == eos2) break;
            next = qwen_forward(&g_model, next);
        }
        printf("\n");

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double gen_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        int total_tokens = g_model.pos;
        int gen_tokens = total_tokens - n_prompt;
        double decode_sec = gen_sec - prefill_sec;
        printf("\nTotal: %d tokens in %.2fs\n", total_tokens, gen_sec);
        printf("Prefill: %.1f t/s (%d tokens)\n", n_prompt / prefill_sec, n_prompt);
        printf("Decode:  %.1f t/s (%d tokens)\n",
               decode_sec > 0 ? gen_tokens / decode_sec : 0, gen_tokens);

        return 0;
    }
}
