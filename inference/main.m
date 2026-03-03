// main.m -- Qwen2.5-0.5B inference on Apple Neural Engine
// Supports four modes:
//   1. Single-shot:  ./qwen_ane weights.bin "token_ids" [max_tokens]
//   2. Stdin server:  ./qwen_ane weights.bin --server
//   3. Socket server: ./qwen_ane weights.bin --server /tmp/qwen_ane.sock
//   4. HTTP API:      ./qwen_ane weights.bin --http 8000 --model-dir ~/models/Qwen2.5-0.5B-Instruct
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface \
//     -framework CoreML -framework Accelerate -ldl -lobjc -fobjc-arc \
//     -o qwen_ane main.m
//
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include "qwen_ane_infer.h"
#include "tokenizer.h"
#include "http_server.h"

int g_fp16_io = 0;
static QwenModel g_model;
static const char *g_sock_path = NULL;
static Tokenizer g_tokenizer;
static int g_tokenizer_loaded = 0;

static void cleanup_socket(void) {
    if (g_sock_path) unlink(g_sock_path);
}

static void handle_signal(int sig) {
    (void)sig;
    cleanup_socket();
    _exit(0);
}

static int load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    int config[7];
    fread(config, sizeof(int), 7, f);
    int dim = config[0], hidden = config[1], n_layers = config[2];
    int n_heads = config[3], n_kv_heads = config[4], vocab = config[5];
    printf("Config: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d\n",
           dim, hidden, n_layers, n_heads, n_kv_heads, vocab);

    int q_dim = n_heads * QWEN_HEAD_DIM;
    int kv_dim = n_kv_heads * QWEN_HEAD_DIM;

    g_model.embed = (float*)malloc((size_t)vocab * dim * sizeof(float));
    fread(g_model.embed, sizeof(float), (size_t)vocab * dim, f);

    for (int l = 0; l < n_layers; l++) {
        g_model.rms_att[l] = (float*)malloc(dim * sizeof(float));
        fread(g_model.rms_att[l], sizeof(float), dim, f);

        g_model.wq[l] = (float*)malloc((size_t)q_dim * dim * sizeof(float));
        fread(g_model.wq[l], sizeof(float), (size_t)q_dim * dim, f);
        g_model.wk[l] = (float*)malloc((size_t)kv_dim * dim * sizeof(float));
        fread(g_model.wk[l], sizeof(float), (size_t)kv_dim * dim, f);
        g_model.wv[l] = (float*)malloc((size_t)kv_dim * dim * sizeof(float));
        fread(g_model.wv[l], sizeof(float), (size_t)kv_dim * dim, f);
        g_model.wo[l] = (float*)malloc((size_t)q_dim * dim * sizeof(float));
        fread(g_model.wo[l], sizeof(float), (size_t)dim * q_dim, f);

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

    long file_size = ftell(f);
    fclose(f);
    printf("Weights loaded (%.0f MB)\n", (float)file_size / 1024 / 1024);
    return 0;
}

// Parse space-separated token IDs from a string. Returns count.
static int parse_tokens(const char *str, int *ids, int max_ids) {
    int n = 0;
    char *buf = strdup(str);
    char *saveptr;
    char *p = strtok_r(buf, " \t\n\r", &saveptr);
    while (p && n < max_ids) {
        ids[n++] = atoi(p);
        p = strtok_r(NULL, " \t\n\r", &saveptr);
    }
    free(buf);
    return n;
}

static double timespec_diff(struct timespec *a, struct timespec *b) {
    return (b->tv_sec - a->tv_sec) + (b->tv_nsec - a->tv_nsec) / 1e9;
}

// Run one generation pass. Writes output token IDs to out_ids, returns count.
// If out_fd >= 0, writes formatted results there; otherwise prints to stdout.
static int generate(int *prompt_ids, int n_prompt, int max_gen,
                    int *out_ids, int max_out,
                    double *prefill_tps, double *decode_tps) {
    struct timespec t0, t1, t_pre;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int next = 0;
    for (int i = 0; i < n_prompt; i++)
        next = qwen_forward(&g_model, prompt_ids[i]);

    clock_gettime(CLOCK_MONOTONIC, &t_pre);
    double ps = timespec_diff(&t0, &t_pre);
    *prefill_tps = ps > 0 ? n_prompt / ps : 0;

    int eos = 151645, eos2 = 151643;
    int n_out = 0;
    for (int i = 0; i < max_gen && n_out < max_out; i++) {
        if (n_out < max_out) out_ids[n_out++] = next;
        if (next == eos || next == eos2) break;
        next = qwen_forward(&g_model, next);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ds = timespec_diff(&t_pre, &t1);
    int gen_tokens = n_out > 1 ? n_out - 1 : 0;
    *decode_tps = ds > 0 ? gen_tokens / ds : 0;

    return n_out;
}

// --- Stdin server mode ---
static void run_stdin_server(void) {
    printf("READY\n");
    fflush(stdout);

    char line[65536];
    while (fgets(line, sizeof(line), stdin)) {
        // Format: "token_id token_id ... [|max_tokens]"
        int max_gen = 50;
        char *pipe = strchr(line, '|');
        if (pipe) {
            max_gen = atoi(pipe + 1);
            *pipe = '\0';
        }

        int prompt_ids[2048];
        int n_prompt = parse_tokens(line, prompt_ids, 2048);
        if (n_prompt == 0) {
            printf("ERR: empty prompt\n");
            fflush(stdout);
            continue;
        }

        int out_ids[4096];
        double p_tps, d_tps;
        int n_out = generate(prompt_ids, n_prompt, max_gen, out_ids, 4096, &p_tps, &d_tps);

        printf("OUT:");
        for (int i = 0; i < n_out; i++) printf(" %d", out_ids[i]);
        printf("\n");
        printf("PERF: prefill=%.1f decode=%.1f prompt=%d gen=%d\n",
               p_tps, d_tps, n_prompt, n_out);
        fflush(stdout);

        qwen_reset(&g_model);
    }
}

// --- Socket server mode ---
static void run_socket_server(const char *sock_path) {
    g_sock_path = sock_path;
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    atexit(cleanup_socket);

    unlink(sock_path);

    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return; }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(srv); return;
    }
    if (listen(srv, 4) < 0) {
        perror("listen"); close(srv); return;
    }

    printf("Listening on %s\n", sock_path);
    printf("READY\n");
    fflush(stdout);

    while (1) {
        int client = accept(srv, NULL, NULL);
        if (client < 0) { perror("accept"); continue; }

        // Read request: {"tokens": [1,2,3], "max_tokens": 50}
        char buf[131072];
        ssize_t total = 0;
        while (total < (ssize_t)sizeof(buf) - 1) {
            ssize_t n = read(client, buf + total, sizeof(buf) - 1 - total);
            if (n <= 0) break;
            total += n;
            if (memchr(buf, '\n', total) || memchr(buf, '}', total)) break;
        }
        buf[total] = '\0';

        // Minimal JSON parsing for {"tokens": [...], "max_tokens": N}
        int prompt_ids[2048];
        int n_prompt = 0;
        int max_gen = 50;

        char *tok_start = strstr(buf, "\"tokens\"");
        if (tok_start) {
            char *bracket = strchr(tok_start, '[');
            if (bracket) {
                char *p = bracket + 1;
                while (*p && *p != ']' && n_prompt < 2048) {
                    while (*p && (*p == ' ' || *p == ',')) p++;
                    if (*p == ']') break;
                    prompt_ids[n_prompt++] = (int)strtol(p, &p, 10);
                }
            }
        }

        char *mt = strstr(buf, "\"max_tokens\"");
        if (mt) {
            char *colon = strchr(mt, ':');
            if (colon) max_gen = (int)strtol(colon + 1, NULL, 10);
        }

        if (n_prompt == 0) {
            const char *err = "{\"error\": \"no tokens\"}\n";
            write(client, err, strlen(err));
            close(client);
            continue;
        }

        int out_ids[4096];
        double p_tps, d_tps;
        int n_out = generate(prompt_ids, n_prompt, max_gen, out_ids, 4096, &p_tps, &d_tps);

        // Build JSON response
        char resp[131072];
        int off = snprintf(resp, sizeof(resp),
            "{\"output\": [");
        for (int i = 0; i < n_out; i++)
            off += snprintf(resp + off, sizeof(resp) - off,
                "%s%d", i ? ", " : "", out_ids[i]);
        off += snprintf(resp + off, sizeof(resp) - off,
            "], \"prefill_tps\": %.1f, \"decode_tps\": %.1f, "
            "\"prompt_tokens\": %d, \"gen_tokens\": %d}\n",
            p_tps, d_tps, n_prompt, n_out);

        write(client, resp, off);
        close(client);

        printf("[socket] prompt=%d gen=%d prefill=%.1f decode=%.1f t/s\n",
               n_prompt, n_out, p_tps, d_tps);
        fflush(stdout);

        qwen_reset(&g_model);
    }
}

// --- HTTP API handler ---
static void http_api_handler(int client_fd, HttpRequest *req, void *ctx) {
    (void)ctx;

    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/health") == 0) {
        http_send_json(client_fd, 200, "{\"status\":\"ok\",\"mode\":\"http\"}");
        return;
    }

    if (strcmp(req->method, "POST") != 0 || strcmp(req->path, "/v1/completions") != 0) {
        http_send_json(client_fd, 404, "{\"error\":\"not found, use POST /v1/completions\"}");
        return;
    }

    if (req->body_len == 0) {
        http_send_json(client_fd, 400, "{\"error\":\"empty body\"}");
        return;
    }

    char prompt[32768];
    if (http_json_get_string(req->body, "prompt", prompt, sizeof(prompt)) < 0) {
        http_send_json(client_fd, 400, "{\"error\":\"missing 'prompt' field\"}");
        return;
    }

    int max_tokens = http_json_get_int(req->body, "max_tokens", 50);
    if (max_tokens > 512) max_tokens = 512;
    if (max_tokens < 1) max_tokens = 1;

    char system_prompt[4096];
    if (http_json_get_string(req->body, "system", system_prompt, sizeof(system_prompt)) < 0)
        strcpy(system_prompt, "You are a helpful assistant. Be concise.");

    // Time tokenization separately
    struct timespec t_tok0, t_tok1, t_gen0, t_gen1, t_det0, t_det1;

    clock_gettime(CLOCK_MONOTONIC, &t_tok0);
    int input_ids[4096];
    int n_input = tok_encode_chat(&g_tokenizer, system_prompt, prompt, input_ids, 4096);
    clock_gettime(CLOCK_MONOTONIC, &t_tok1);
    double tokenize_ms = timespec_diff(&t_tok0, &t_tok1) * 1000.0;

    if (n_input == 0) {
        http_send_json(client_fd, 400, "{\"error\":\"tokenization produced no tokens\"}");
        return;
    }

    // Pure inference timing
    clock_gettime(CLOCK_MONOTONIC, &t_gen0);
    int out_ids[4096];
    double p_tps, d_tps;
    int n_out = generate(input_ids, n_input, max_tokens, out_ids, 4096, &p_tps, &d_tps);
    clock_gettime(CLOCK_MONOTONIC, &t_gen1);
    double inference_ms = timespec_diff(&t_gen0, &t_gen1) * 1000.0;

    // Prefill time = inference of prompt tokens only (from generate's internal timing)
    double prefill_s = p_tps > 0 ? n_input / p_tps : 0;
    double ttft_ms = prefill_s * 1000.0;

    // Time detokenization separately
    clock_gettime(CLOCK_MONOTONIC, &t_det0);
    char decoded[65536];
    tok_decode(&g_tokenizer, out_ids, n_out, decoded, sizeof(decoded));
    clock_gettime(CLOCK_MONOTONIC, &t_det1);
    double detokenize_ms = timespec_diff(&t_det0, &t_det1) * 1000.0;

    double total_ms = tokenize_ms + inference_ms + detokenize_ms;

    // Escape the decoded text for JSON
    char escaped[131072];
    int ei = 0;
    for (int i = 0; decoded[i] && ei < (int)sizeof(escaped) - 6; i++) {
        switch (decoded[i]) {
            case '"':  escaped[ei++] = '\\'; escaped[ei++] = '"'; break;
            case '\\': escaped[ei++] = '\\'; escaped[ei++] = '\\'; break;
            case '\n': escaped[ei++] = '\\'; escaped[ei++] = 'n'; break;
            case '\r': escaped[ei++] = '\\'; escaped[ei++] = 'r'; break;
            case '\t': escaped[ei++] = '\\'; escaped[ei++] = 't'; break;
            default:
                if ((unsigned char)decoded[i] < 0x20) {
                    ei += snprintf(escaped + ei, 7, "\\u%04x", (unsigned char)decoded[i]);
                } else {
                    escaped[ei++] = decoded[i];
                }
        }
    }
    escaped[ei] = '\0';

    // Build JSON response with detailed timing breakdown
    char resp[HTTP_MAX_RESPONSE];
    snprintf(resp, sizeof(resp),
        "{\"text\":\"%s\",\"prompt_tokens\":%d,\"gen_tokens\":%d,"
        "\"prefill_tps\":%.1f,\"decode_tps\":%.1f,"
        "\"tokenize_ms\":%.1f,\"inference_ms\":%.1f,\"detokenize_ms\":%.1f,"
        "\"ttft_ms\":%.1f,\"total_ms\":%.1f}",
        escaped, n_input, n_out, p_tps, d_tps,
        tokenize_ms, inference_ms, detokenize_ms, ttft_ms, total_ms);

    http_send_json(client_fd, 200, resp);

    printf("[http] prompt=%d gen=%d prefill=%.1f decode=%.1f t/s | tok=%.1f inf=%.1f detok=%.1f ms\n",
           n_input, n_out, p_tps, d_tps, tokenize_ms, inference_ms, detokenize_ms);
    fflush(stdout);

    qwen_reset(&g_model);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr,
                "Usage:\n"
                "  %s <weights.bin> \"token_ids\" [max_tokens]                  (single-shot)\n"
                "  %s <weights.bin> --server                                    (stdin loop)\n"
                "  %s <weights.bin> --server /tmp/qwen_ane.sock                 (socket server)\n"
                "  %s <weights.bin> --http 8000 --model-dir ~/models/Qwen2.5   (HTTP API)\n",
                argv[0], argv[0], argv[0], argv[0]);
            return 1;
        }

        printf("=== Qwen2.5-0.5B ANE Inference ===\n\n");

        setbuf(stdout, NULL);

        printf("Loading weights...\n");
        if (load_weights(argv[1]) != 0) return 1;

        qwen_alloc(&g_model);
        qwen_rope_init();

        printf("Compiling ANE kernels (169 total)...\n");
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        qwen_compile_kernels(&g_model);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double compile_sec = timespec_diff(&t0, &t1);
        printf("Compile time: %.1fs\n\n", compile_sec);

        // Parse flags
        int server_mode = 0;
        int http_port = 0;
        int test_ane = 0;
        const char *sock_path = NULL;
        const char *model_dir = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--server") == 0) {
                server_mode = 1;
                if (i + 1 < argc && argv[i+1][0] != '-')
                    sock_path = argv[++i];
            } else if (strcmp(argv[i], "--http") == 0) {
                if (i + 1 < argc) http_port = atoi(argv[++i]);
                else { fprintf(stderr, "--http requires a port number\n"); return 1; }
            } else if (strcmp(argv[i], "--model-dir") == 0) {
                if (i + 1 < argc) model_dir = argv[++i];
                else { fprintf(stderr, "--model-dir requires a path\n"); return 1; }
            } else if (strcmp(argv[i], "--test-ane") == 0) {
                test_ane = 1;
            }
        }

        // ANE vs CPU correctness test
        if (test_ane) {
            printf("=== ANE vs CPU Projection Test ===\n\n");

            // Use a realistic input: embed token 2610 ("What"), RMSNorm it
            int test_token = 2610;
            memcpy(g_model.x, g_model.embed + test_token * QWEN_DIM, QWEN_DIM * sizeof(float));
            qwen_rmsnorm(g_model.xb, g_model.x, g_model.rms_att[0], QWEN_DIM);

            // Also prepare a realistic Q output for the O projection test
            cpu_project(g_model.wq[0], g_model.xb, g_model.q, QWEN_DIM, QWEN_Q_DIM);

            float *cpu_out = (float*)calloc(QWEN_HIDDEN, sizeof(float));
            float *ane_out = (float*)calloc(QWEN_HIDDEN, sizeof(float));

            struct {
                const char *name;
                ANEKernel *kernel;
                const float *weights;
                int in_dim, out_dim;
            } tests[] = {
                {"L0 Q proj",   g_model.k_q[0],    g_model.wq[0],     QWEN_DIM, QWEN_Q_DIM},
                {"L0 K proj",   g_model.k_k[0],    g_model.wk[0],     QWEN_DIM, QWEN_KV_DIM},
                {"L0 V proj",   g_model.k_v[0],    g_model.wv[0],     QWEN_DIM, QWEN_KV_DIM},
                {"L0 O proj",   g_model.k_o[0],    g_model.wo[0],     QWEN_Q_DIM, QWEN_DIM},
                {"L0 Gate",     g_model.k_gate[0],  g_model.w_gate[0], QWEN_DIM, QWEN_HIDDEN},
                {"L0 Up",       g_model.k_up[0],    g_model.w_up[0],   QWEN_DIM, QWEN_HIDDEN},
                {"L0 Down",     g_model.k_down[0],  g_model.w_down[0], QWEN_HIDDEN, QWEN_DIM},
                {"LM Head c0",  g_model.k_lmhead[0], g_model.embed,   QWEN_DIM, QWEN_LM_CHUNK_SIZE},
            };
            int n_tests = sizeof(tests) / sizeof(tests[0]);
            int all_pass = 1;

            for (int t = 0; t < n_tests; t++) {
                if (!tests[t].kernel) {
                    printf("  %-14s SKIP (kernel not compiled)\n", tests[t].name);
                    continue;
                }
                const float *input;
                if (tests[t].in_dim == QWEN_Q_DIM) {
                    input = g_model.q;
                } else if (tests[t].in_dim == QWEN_HIDDEN) {
                    cpu_project(g_model.w_gate[0], g_model.xb, g_model.hb, QWEN_DIM, QWEN_HIDDEN);
                    input = g_model.hb;
                } else {
                    input = g_model.xb;
                }

                cpu_project(tests[t].weights, input, cpu_out, tests[t].in_dim, tests[t].out_dim);

                // ANE projection with return-value check
                ane_write_input(tests[t].kernel, 0, input, tests[t].in_dim * sizeof(float));
                bool ane_ok = ane_run(tests[t].kernel);
                ane_read_output(tests[t].kernel, 0, ane_out, tests[t].out_dim * sizeof(float));
                if (!ane_ok) printf("    !! ANE execution returned false\n");

                float max_diff = 0, sum_diff = 0;
                float cpu_norm = 0, ane_norm = 0;
                for (int i = 0; i < tests[t].out_dim; i++) {
                    float d = fabsf(cpu_out[i] - ane_out[i]);
                    if (d > max_diff) max_diff = d;
                    sum_diff += d;
                    cpu_norm += cpu_out[i] * cpu_out[i];
                    ane_norm += ane_out[i] * ane_out[i];
                }
                float avg_diff = sum_diff / tests[t].out_dim;
                float rel_err = (sqrtf(cpu_norm) > 0) ?
                    sqrtf(sum_diff * sum_diff / tests[t].out_dim) / sqrtf(cpu_norm / tests[t].out_dim) : 0;

                int pass = (max_diff < 0.5f && rel_err < 0.05f);
                if (!pass) all_pass = 0;

                printf("  %-14s [%d→%d]  max_diff=%.6f  avg_diff=%.6f  rel_err=%.4f  %s\n",
                       tests[t].name, tests[t].in_dim, tests[t].out_dim,
                       max_diff, avg_diff, rel_err,
                       pass ? "PASS" : "FAIL");
                printf("    CPU first4: %.6f %.6f %.6f %.6f  norm=%.4f\n",
                       cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3], sqrtf(cpu_norm));
                printf("    ANE first4: %.6f %.6f %.6f %.6f  norm=%.4f\n",
                       ane_out[0], ane_out[1], ane_out[2], ane_out[3], sqrtf(ane_norm));
            }

            printf("\n%s\n", all_pass ?
                "ALL TESTS PASSED -- ANE projections match CPU (within FP16 tolerance)" :
                "SOME TESTS FAILED -- ANE projections have accuracy issues");

            // If all pass, benchmark one layer ANE vs CPU speed
            if (all_pass) {
                printf("\n=== Speed comparison (1000 iterations, L0 Q proj %d→%d) ===\n",
                       QWEN_DIM, QWEN_Q_DIM);
                struct timespec ts0, ts1;

                clock_gettime(CLOCK_MONOTONIC, &ts0);
                for (int i = 0; i < 1000; i++)
                    cpu_project(g_model.wq[0], g_model.xb, cpu_out, QWEN_DIM, QWEN_Q_DIM);
                clock_gettime(CLOCK_MONOTONIC, &ts1);
                double cpu_us = timespec_diff(&ts0, &ts1) * 1e6 / 1000;

                clock_gettime(CLOCK_MONOTONIC, &ts0);
                for (int i = 0; i < 1000; i++)
                    ane_project(g_model.k_q[0], g_model.xb, ane_out, QWEN_DIM, QWEN_Q_DIM);
                clock_gettime(CLOCK_MONOTONIC, &ts1);
                double ane_us = timespec_diff(&ts0, &ts1) * 1e6 / 1000;

                printf("  CPU: %.1f us/call\n", cpu_us);
                printf("  ANE: %.1f us/call\n", ane_us);
                printf("  Ratio: %.2fx %s\n", cpu_us / ane_us,
                       ane_us < cpu_us ? "(ANE faster)" : "(CPU faster)");
            }

            free(cpu_out);
            free(ane_out);
            return all_pass ? 0 : 1;
        }

        if (server_mode) {
            if (sock_path)
                run_socket_server(sock_path);
            else
                run_stdin_server();
            return 0;
        }

        // HTTP API mode
        if (http_port > 0) {
            if (!model_dir) {
                // Default to ~/models/Qwen2.5-0.5B-Instruct
                static char default_dir[4096];
                const char *home = getenv("HOME");
                snprintf(default_dir, sizeof(default_dir), "%s/models/Qwen2.5-0.5B-Instruct", home ? home : ".");
                model_dir = default_dir;
            }
            printf("Loading tokenizer from %s...\n", model_dir);
            if (tok_init(&g_tokenizer, model_dir) != 0) {
                fprintf(stderr, "Failed to load tokenizer from %s\n", model_dir);
                return 1;
            }
            g_tokenizer_loaded = 1;
            printf("Tokenizer ready.\n\n");

            signal(SIGINT, handle_signal);
            signal(SIGTERM, handle_signal);

            http_serve(http_port, http_api_handler, NULL);
            tok_free(&g_tokenizer);
            return 0;
        }

        // Single-shot mode (original behavior)
        if (argc < 3) {
            fprintf(stderr, "Error: provide token IDs or --server\n");
            return 1;
        }

        int max_gen = 50;
        if (argc >= 4 && strcmp(argv[3], "--server") != 0)
            max_gen = atoi(argv[3]);

        int prompt_ids[2048];
        int n_prompt = parse_tokens(argv[2], prompt_ids, 2048);
        printf("Prompt: %d tokens, generating up to %d\n", n_prompt, max_gen);

        int out_ids[4096];
        double p_tps, d_tps;
        int n_out = generate(prompt_ids, n_prompt, max_gen, out_ids, 4096, &p_tps, &d_tps);

        printf("OUT:");
        for (int i = 0; i < n_out; i++) printf(" %d", out_ids[i]);
        printf("\n");

        printf("\nPrefill: %.1f t/s (%d tokens)\n", p_tps, n_prompt);
        printf("Decode:  %.1f t/s (%d tokens)\n", d_tps, n_out > 1 ? n_out - 1 : 0);

        return 0;
    }
}
