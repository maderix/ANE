CC = xcrun clang
CC_C = xcrun clang

ANE_COMPAT = -Wno-deprecated-declarations
SEC_FLAGS = -fstack-protector-strong -Wformat-security

CFLAGS = -O2 -Wall $(ANE_COMPAT) -fobjc-arc $(SEC_FLAGS)
CFLAGS_C = -O2 -Wall -Wextra -Werror -std=c11
CFLAGS_DEBUG = -O0 -g -Wall $(ANE_COMPAT) -fobjc-arc -fsanitize=address,undefined
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl

HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h data_validation.h

HEADERS_ANE = $(HEADERS_LARGE) ane_rmsnorm_bwd.h ane_classifier.h

train: train.m ane_runtime.h ane_mil_gen.h model.h forward.h backward.h
	$(CC) $(CFLAGS) -o $@ train.m $(LDFLAGS) -framework Accelerate

train_large: train_large.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -o $@ train_large.m $(LDFLAGS) -framework Accelerate

train_large_ane: train_large_ane.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) -o $@ train_large_ane.m $(LDFLAGS) -framework Accelerate

HEADERS_OPT = $(HEADERS_LARGE) stories_cpu_ops_opt.h

train_opt: train_opt.m $(HEADERS_OPT)
	$(CC) $(CFLAGS) -o $@ train_opt.m $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders

train_double_buffer: train_double_buffer.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -o $@ train_double_buffer.m $(LDFLAGS) -framework Accelerate

PROBES = test_weight_reload test_perf_stats test_qos_sweep test_ane_advanced

test_rmsnorm_bwd: test_rmsnorm_bwd.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

test_classifier: test_classifier.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

test_weight_reload: test_weight_reload.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_perf_stats: test_perf_stats.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_qos_sweep: test_qos_sweep.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_ane_advanced: test_ane_advanced.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_chaining: test_chaining.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_chaining_v2: test_chaining_v2.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_bench_paths: test_bench_paths.m ane_runtime.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_ane_model: test_ane_model.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Metal

test_throughput_ceiling: test_throughput_ceiling.m ane_runtime.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_coreml_chaining: test_coreml_chaining.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Metal

test_e5_validate: test_e5_validate.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Metal

test_mil_custom: test_mil_custom.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

test_data_validation: test_data_validation.c data_validation.h
	$(CC_C) $(CFLAGS_C) -o $@ $<

probes: $(PROBES)

security-tests: test_data_validation

data: tokenize
	@bash download_data.sh

tokenize:
	python3 tokenize.py

setup: data
	@echo "=== Setup complete ==="
	@echo "Data:  tinystories_data00.bin"
	@echo "To train: make train_large && ./train_large"
	@echo "Override paths: ANE_MODEL_PATH=... ANE_DATA_PATH=... ./train_large"

verify-flags:
	@echo "=== Active CFLAGS ==="
	@echo "$(CFLAGS)"
	@echo "=== Compiler version ==="
	@xcrun clang --version

clean:
	rm -f train train_large train_large_ane train_opt train_double_buffer $(PROBES) test_rmsnorm_bwd test_classifier test_data_validation test_chaining test_chaining_v2 test_bench_paths test_ane_model test_throughput_ceiling test_coreml_chaining test_e5_validate test_mil_custom

.PHONY: clean tokenize probes security-tests verify-flags data setup
