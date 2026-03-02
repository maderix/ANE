CC = xcrun clang

# Required: Private ANE APIs (_ANEInMemoryModelDescriptor etc.) via objc_msgSend
# trigger deprecation warnings on newer macOS SDKs. This flag is intentional.
# Run 'make check-deprecated' to inspect which declarations are currently suppressed.
ANE_COMPAT = -Wno-deprecated-declarations

# Security hardening flags (ref: docs/reports/security-audit-2026-03-02.md LOW-01)
# Note: -D_FORTIFY_SOURCE=2 is implicitly active at -O2 on macOS (Apple LLVM toolchain).
SEC_FLAGS = -fstack-protector-strong -Wformat-security

CFLAGS = -O2 -Wall $(ANE_COMPAT) -fobjc-arc $(SEC_FLAGS)
CFLAGS_DEBUG = -O0 -g -Wall $(ANE_COMPAT) -fobjc-arc -fsanitize=address,undefined
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl

HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h

train: train.m ane_runtime.h ane_mil_gen.h model.h forward.h backward.h
	$(CC) $(CFLAGS) -o $@ train.m $(LDFLAGS)

train_large: train_large.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -o $@ train_large.m $(LDFLAGS) -framework Accelerate

PROBES = test_weight_reload test_perf_stats test_qos_sweep test_ane_advanced

test_weight_reload: test_weight_reload.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_perf_stats: test_perf_stats.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_qos_sweep: test_qos_sweep.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_ane_advanced: test_ane_advanced.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

probes: $(PROBES)

tokenize:
	python3 tokenize.py

# Show active build flags — verify security hardening is active (LOW-01)
verify-flags:
	@echo "=== Active CFLAGS ==="
	@echo "$(CFLAGS)"
	@echo "=== Compiler version ==="
	@xcrun clang --version

# Build train_large without deprecation suppression to inspect hidden warnings (LOW-02)
# Informational only — deprecated warnings from private ANE API usage are expected.
check-deprecated:
	@echo "=== Building without deprecation suppression (informational) ==="
	@$(CC) $(CFLAGS:-Wno-deprecated-declarations=-Wdeprecated-declarations) \
	    -o /dev/null train_large.m $(LDFLAGS) -framework Accelerate 2>&1 | \
	    grep -i "deprecated" | sort -u || echo "(keine zusaetzlichen deprecated-Warnungen)"

clean:
	rm -f train train_large $(PROBES)

.PHONY: clean tokenize probes verify-flags check-deprecated
