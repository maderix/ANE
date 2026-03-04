// train.m — Stories110M training loop on ANE
// Usage: ./train <model.bin> [seq_len] [steps] [lr] [--cpu]
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "backward.h"

int g_fp16_io = 0;  // M1/M2: use fp16 I/O when cast op unsupported

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

int main(int argc, char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        if (argc < 2) {
            fprintf(stderr, "Usage: %s <model.bin> [seq_len=16] [steps=100] [lr=1e-4] [--cpu]\n", argv[0]);
            return 1;
        }

        int seq_len = argc > 2 ? atoi(argv[2]) : 16;
        int steps = argc > 3 ? atoi(argv[3]) : 100;
        float lr = argc > 4 ? atof(argv[4]) : 1e-4f;
        bool use_ane = true;
        for (int i = 1; i < argc; i++)
            if (strcmp(argv[i], "--cpu") == 0) use_ane = false;

        printf("=== Stories110M ANE Training ===\n");
        printf("Seq len: %d, Steps: %d, LR: %.2e, Backend: %s\n\n",
               seq_len, steps, lr, use_ane ? "ANE" : "CPU");

        Model m = {0};
        printf("Loading weights...\n");
        if (model_load_weights(&m, argv[1]) != 0) return 1;

        if (use_ane) {
            if (model_compile_kernels(&m, seq_len) != 0) {
                fprintf(stderr, "ANE kernel compilation failed, falling back to CPU\n");
                use_ane = false;
            }
        }
        if (!use_ane) m.seq_len = seq_len;

        model_alloc_training(&m);

        // Training tokens: simple repeating pattern to overfit on
        int *train_tokens = (int*)malloc(seq_len * sizeof(int));
        for (int i = 0; i < seq_len; i++)
            train_tokens[i] = (i * 7 + 13) % 256 + 1;

        printf("\nTraining tokens (first 16): ");
        for (int i = 0; i < 16 && i < seq_len; i++) printf("%d ", train_tokens[i]);
        printf("...\n\n");

        printf("%-6s %-10s %-12s %-10s %-10s\n", "Step", "Loss", "GradNorm", "ms/step", "tok/s");
        printf("------------------------------------------------------\n");

        int recompile_interval = 1; // Recompile ANE kernels every N steps
        for (int step = 0; step < steps; step++) {
            uint64_t t0 = mach_absolute_time();

            float loss = model_forward(&m, train_tokens, use_ane);
            if (isnan(loss) || isinf(loss)) {
                printf("NaN/Inf loss at step %d, stopping.\n", step);
                break;
            }

            model_backward(&m, train_tokens);
            model_clip_gradients(&m, 1.0f);
            model_adam_step(&m, lr, 0.9f, 0.999f, 1e-8f);

            // Recompile ANE kernels with updated weights
            if (use_ane && (step + 1) % recompile_interval == 0) {
                if (model_recompile_kernels(&m) != 0) {
                    printf("Recompile failed at step %d, switching to CPU\n", step);
                    use_ane = false;
                }
            }

            double ms = ticksToMs(mach_absolute_time() - t0);
            double tps = (seq_len - 1) / (ms / 1000.0);

            if (step % 10 == 0 || step == steps - 1) {
                double gnorm = 0;
                int d2 = m.cfg.dim;
                for (int i = 0; i < d2*d2; i++) gnorm += (double)m.grad_wq[0][i]*m.grad_wq[0][i];
                gnorm = sqrt(gnorm);
                printf("%-6d %-10.4f %-12.4f %-10.1f %-10.1f\n", step, loss, gnorm, ms, tps);
            }

            if (loss < 0.01f) {
                printf("\nConverged at step %d! Loss: %.6f\n", step, loss);
                break;
            }
        }

        free(train_tokens);
        printf("\nDone.\n");
    }
    return 0;
}
