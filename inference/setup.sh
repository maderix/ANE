#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_DIR="$HOME/models/Qwen2.5-0.5B-Instruct"
WEIGHTS_BIN="$SCRIPT_DIR/qwen05b.bin"
BINARY="$SCRIPT_DIR/qwen_ane"
VENV_DIR="$SCRIPT_DIR/.venv"
EXPECTED_WEIGHT_SIZE=1976131100

info()  { printf "\033[1;34m==> %s\033[0m\n" "$1"; }
ok()    { printf "\033[1;32m  ✓ %s\033[0m\n" "$1"; }
warn()  { printf "\033[1;33m  ! %s\033[0m\n" "$1"; }
fail()  { printf "\033[1;31m  ✗ %s\033[0m\n" "$1"; exit 1; }

info "ANE Inference Setup"
echo "Model: $MODEL_ID"
echo "Target: $SCRIPT_DIR"
echo ""

# --- Step 1: Prerequisites ---
info "Checking prerequisites..."

if ! command -v xcrun &>/dev/null; then
    fail "Xcode Command Line Tools not found. Install with: xcode-select --install"
fi
ok "xcrun clang available"

if ! command -v python3 &>/dev/null; then
    fail "Python 3 not found"
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]); then
    fail "Python 3.11+ required (found $PY_VER). coremltools needs 3.11-3.13."
fi
ok "Python $PY_VER"

# --- Step 2: Virtual environment ---
info "Setting up Python environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    ok "Created venv at $VENV_DIR"
else
    ok "Venv already exists"
fi

source "$VENV_DIR/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet safetensors torch transformers huggingface-hub
ok "Python dependencies installed"

# --- Step 3: Download model ---
info "Downloading model from HuggingFace..."

if [ -f "$MODEL_DIR/model.safetensors" ] && [ -f "$MODEL_DIR/tokenizer.json" ]; then
    ok "Model already downloaded at $MODEL_DIR"
else
    mkdir -p "$MODEL_DIR"
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
    else
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR')
"
    fi
    ok "Model downloaded to $MODEL_DIR"
fi

# Verify key files exist
for f in model.safetensors tokenizer.json vocab.json merges.txt config.json; do
    if [ ! -f "$MODEL_DIR/$f" ]; then
        fail "Missing $f in $MODEL_DIR"
    fi
done
ok "All model files present"

# --- Step 4: Convert weights ---
info "Converting weights to binary format..."

if [ -f "$WEIGHTS_BIN" ]; then
    ACTUAL_SIZE=$(stat -f%z "$WEIGHTS_BIN" 2>/dev/null || stat -c%s "$WEIGHTS_BIN" 2>/dev/null)
    if [ "$ACTUAL_SIZE" -eq "$EXPECTED_WEIGHT_SIZE" ]; then
        ok "Weights already converted ($((ACTUAL_SIZE / 1024 / 1024)) MB)"
    else
        warn "Weight file exists but wrong size ($ACTUAL_SIZE vs $EXPECTED_WEIGHT_SIZE), reconverting"
        python3 "$SCRIPT_DIR/convert_weights.py" "$MODEL_DIR" "$WEIGHTS_BIN"
        ok "Weights converted"
    fi
else
    python3 "$SCRIPT_DIR/convert_weights.py" "$MODEL_DIR" "$WEIGHTS_BIN"
    ok "Weights converted"
fi

# --- Step 5: Build binary ---
info "Building qwen_ane binary..."

NEEDS_BUILD=0
if [ ! -f "$BINARY" ]; then
    NEEDS_BUILD=1
elif [ "$SCRIPT_DIR/main.m" -nt "$BINARY" ] || \
     [ "$SCRIPT_DIR/qwen_ane_infer.h" -nt "$BINARY" ] || \
     [ "$SCRIPT_DIR/tokenizer.h" -nt "$BINARY" ] 2>/dev/null || \
     [ "$SCRIPT_DIR/http_server.h" -nt "$BINARY" ] 2>/dev/null; then
    NEEDS_BUILD=1
    warn "Source files newer than binary, rebuilding"
fi

if [ "$NEEDS_BUILD" -eq 1 ]; then
    xcrun clang -O2 -framework Foundation -framework IOSurface \
        -framework CoreML -framework Accelerate -ldl -lobjc -fobjc-arc \
        -o "$BINARY" "$SCRIPT_DIR/main.m"
    ok "Binary built: $BINARY"
else
    ok "Binary up to date"
fi

# --- Step 6: Smoke test ---
info "Running smoke test..."

# Quick single-shot test with known token IDs for "system\nYou are a helpful assistant."
TEST_OUTPUT=$("$BINARY" "$WEIGHTS_BIN" "151644 8948 198" 3 2>&1 || true)

if echo "$TEST_OUTPUT" | grep -q "OUT:"; then
    ok "Smoke test passed (model generates output)"
else
    warn "Smoke test: no output tokens detected (this may be OK on first run)"
    echo "  Output was: $(echo "$TEST_OUTPUT" | tail -3)"
fi

# --- Done ---
echo ""
info "Setup complete!"
echo ""
echo "  Binary:  $BINARY"
echo "  Weights: $WEIGHTS_BIN ($(du -h "$WEIGHTS_BIN" | cut -f1) )"
echo "  Model:   $MODEL_DIR"
echo ""
echo "Quick start:"
echo "  # Single prompt (slow, compiles every time)"
echo "  python3 $SCRIPT_DIR/run.py \"What is 2+2?\""
echo ""
echo "  # Server mode (fast, compile once)"
echo "  $BINARY $WEIGHTS_BIN --server /tmp/qwen_ane.sock &"
echo "  python3 $SCRIPT_DIR/run.py \"What is 2+2?\""
echo ""
echo "  # HTTP API (fast, no Python needed for queries)"
echo "  $BINARY $WEIGHTS_BIN --http 8000 --model-dir $MODEL_DIR"
echo "  curl http://localhost:8000/v1/completions -d '{\"prompt\":\"Hi\",\"max_tokens\":20}'"
echo ""
echo "  # Run throughput benchmark"
echo "  $SCRIPT_DIR/benchmark.sh"
