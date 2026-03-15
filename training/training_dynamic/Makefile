CC = xcrun clang
CFLAGS = -O2 -DACCELERATE_NEW_LAPACK -framework Foundation -framework IOSurface -framework Accelerate \
         -isysroot $(shell xcrun --show-sdk-path) -fobjc-arc

# Model selection: make MODEL=qwen3_06b  (default)
#   Available: stories110m, qwen3_06b
MODEL ?= qwen3_06b
MODEL_HDR = models/$(MODEL).h

train: train.m config.h io.h cpu_ops.h mil_dynamic.h $(MODEL_HDR)
	@echo "Building for model: $(MODEL)"
	$(CC) $(CFLAGS) -include $(MODEL_HDR) -o train train.m

clean:
	rm -f train

.PHONY: clean
