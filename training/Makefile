CC = xcrun clang
CFLAGS = -O2 -Wall -DACCELERATE_NEW_LAPACK -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl

HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h

HEADERS_ANE = $(HEADERS_LARGE) ane_rmsnorm_bwd.h ane_classifier.h

train: train.m ane_runtime.h ane_mil_gen.h model.h forward.h backward.h
	$(CC) $(CFLAGS) -o $@ train.m $(LDFLAGS)

train_large: train_large.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -o $@ train_large.m $(LDFLAGS) -framework Accelerate

train_large_ane: train_large_ane.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) -o $@ train_large_ane.m $(LDFLAGS) -framework Accelerate

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

probes: $(PROBES)

tokenize:
	python3 tokenize.py

clean:
	rm -f train train_large train_large_ane $(PROBES) test_rmsnorm_bwd test_classifier

.PHONY: clean tokenize probes

