#!/bin/bash
# run_benchmarks.sh -- ANE Training Benchmark Runner
# Builds and runs benchmarks, collects results into a timestamped report.
#
# Usage:
#   bash scripts/run_benchmarks.sh [OPTIONS]
#
# Options:
#   --all             Run everything (default)
#   --training-only   Run only training benchmarks
#   --probes-only     Run only probe/test suite
#   --benchmarks-only Run only root-level benchmarks (inmem_peak)
#   --steps N         Training steps (default: 100)
#   --help            Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="$ROOT_DIR/benchmark_results_${TIMESTAMP}.txt"

# Defaults
RUN_TRAINING=true
RUN_PROBES=true
RUN_BENCHMARKS=true
STEPS=100

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $*"; }
log_fail()    { echo -e "${RED}[FAIL]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_header()  { echo -e "\n${CYAN}========================================${NC}"; echo -e "${CYAN} $*${NC}"; echo -e "${CYAN}========================================${NC}"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            RUN_TRAINING=true; RUN_PROBES=true; RUN_BENCHMARKS=true; shift ;;
        --training-only)
            RUN_TRAINING=true; RUN_PROBES=false; RUN_BENCHMARKS=false; shift ;;
        --probes-only)
            RUN_TRAINING=false; RUN_PROBES=true; RUN_BENCHMARKS=false; shift ;;
        --benchmarks-only)
            RUN_TRAINING=false; RUN_PROBES=false; RUN_BENCHMARKS=true; shift ;;
        --steps)
            STEPS="$2"; shift 2 ;;
        --help|-h)
            head -14 "$0" | tail -13
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Initialize results file
{
    echo "ANE Training Benchmark Results"
    echo "=============================="
    echo "Date:     $(date)"
    echo "Machine:  $(sysctl -n hw.model 2>/dev/null || echo 'unknown')"
    echo "macOS:    $(sw_vers -productVersion 2>/dev/null || echo 'unknown')"
    echo "Chip:     $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
    echo "Steps:    $STEPS"
    echo ""
} > "$RESULTS_FILE"

log_info "Results will be saved to: $RESULTS_FILE"

# ──────────────────────────────────────────────
# Prerequisite checks
# ──────────────────────────────────────────────

log_header "Prerequisite Checks"

if [[ "$(uname)" != "Darwin" ]]; then
    log_fail "This benchmark requires macOS. Detected: $(uname)"
    exit 1
fi
log_success "macOS detected"

if ! sysctl -n hw.optional.arm64 2>/dev/null | grep -q 1; then
    log_fail "Apple Silicon required. This appears to be an Intel Mac."
    exit 1
fi
log_success "Apple Silicon detected"

if ! xcrun --find clang >/dev/null 2>&1; then
    log_fail "Xcode command line tools required. Run: xcode-select --install"
    exit 1
fi
log_success "Xcode CLI tools available"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

run_build_and_test() {
    local name="$1"
    local build_cmd="$2"
    local run_cmd="$3"
    local workdir="${4:-$ROOT_DIR}"

    log_info "Building $name..."
    local build_output
    if ! build_output=$(cd "$workdir" && bash -c "$build_cmd" 2>&1); then
        log_fail "$name -- build failed"
        echo "[$name] BUILD FAILED" >> "$RESULTS_FILE"
        echo "$build_output" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    log_info "Running $name..."
    echo "--- $name ---" >> "$RESULTS_FILE"

    local output
    if output=$(cd "$workdir" && bash -c "$run_cmd" 2>&1); then
        echo "$output" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        log_success "$name completed"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "$output" >> "$RESULTS_FILE"
        echo "EXIT CODE: $?" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        log_fail "$name -- run failed (output captured in results file)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

# ──────────────────────────────────────────────
# Training Benchmarks
# ──────────────────────────────────────────────

if $RUN_TRAINING; then
    log_header "Training Benchmarks ($STEPS steps)"

    echo "" >> "$RESULTS_FILE"
    echo "=== TRAINING BENCHMARKS ===" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    run_build_and_test \
        "train_large (CPU classifier)" \
        "make train_large" \
        "./train_large --steps $STEPS" \
        "$TRAINING_DIR" || true

    run_build_and_test \
        "train_large_ane (ANE classifier)" \
        "make train_large_ane" \
        "./train_large_ane --steps $STEPS" \
        "$TRAINING_DIR" || true
fi

# ──────────────────────────────────────────────
# Probe Tests
# ──────────────────────────────────────────────

if $RUN_PROBES; then
    log_header "Probe Tests"

    echo "" >> "$RESULTS_FILE"
    echo "=== PROBE TESTS ===" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    PROBE_TESTS=("test_rmsnorm_bwd" "test_classifier" "test_weight_reload" "test_perf_stats" "test_qos_sweep" "test_ane_advanced")

    for test_name in "${PROBE_TESTS[@]}"; do
        run_build_and_test \
            "$test_name" \
            "make $test_name" \
            "./$test_name" \
            "$TRAINING_DIR" || true
    done
fi

# ──────────────────────────────────────────────
# Root-Level Benchmarks
# ──────────────────────────────────────────────

if $RUN_BENCHMARKS; then
    log_header "Root-Level Benchmarks"

    echo "" >> "$RESULTS_FILE"
    echo "=== ROOT-LEVEL BENCHMARKS ===" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    CC="xcrun clang"
    CFLAGS="-O2 -fobjc-arc -framework Foundation -framework CoreML -framework IOSurface -ldl"

    # Generate mlpackage models needed by sram/inmem benchmarks
    if ! ls /tmp/ane_sram_*ch_*sp.mlpackage >/dev/null 2>&1; then
        log_info "Generating mlpackage models for benchmarks..."
        VENV_PYTHON=""
        if [[ -x /tmp/ane_venv/bin/python3 ]]; then
            VENV_PYTHON="/tmp/ane_venv/bin/python3"
        else
            for pyver in 3.12 3.13 3.11; do
                PY="/opt/homebrew/opt/python@${pyver}/bin/python${pyver}"
                if [[ -x "$PY" ]]; then
                    log_info "Creating venv with Python $pyver for coremltools..."
                    "$PY" -m venv /tmp/ane_venv && /tmp/ane_venv/bin/pip install -q coremltools numpy 2>/dev/null
                    VENV_PYTHON="/tmp/ane_venv/bin/python3"
                    break
                fi
            done
        fi
        if [[ -n "$VENV_PYTHON" ]] && "$VENV_PYTHON" "$SCRIPT_DIR/gen_mlpackages.py" 2>/dev/null; then
            log_success "mlpackage models generated"
        else
            log_warn "Failed to generate mlpackage models (need Python 3.11-3.13 + coremltools)"
        fi
    else
        log_info "mlpackage models already exist in /tmp/"
    fi

    run_build_and_test \
        "inmem_peak (Peak TFLOPS)" \
        "$CC $CFLAGS -o inmem_peak inmem_peak.m" \
        "./inmem_peak" \
        "$ROOT_DIR" || true

    for bench in inmem_basic inmem_bench sram_bench sram_probe; do
        if ls /tmp/ane_sram_*ch_*sp.mlpackage >/dev/null 2>&1; then
            run_build_and_test \
                "$bench" \
                "$CC $CFLAGS -o $bench ${bench}.m" \
                "./$bench" \
                "$ROOT_DIR" || true
        else
            log_warn "$bench -- SKIPPED (mlpackage generation failed)"
            echo "[$bench] SKIPPED -- mlpackage generation failed" >> "$RESULTS_FILE"
            echo "" >> "$RESULTS_FILE"
            SKIP_COUNT=$((SKIP_COUNT + 1))
        fi
    done
fi

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────

log_header "Summary"

TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))

{
    echo ""
    echo "=== SUMMARY ==="
    echo "Total:   $TOTAL"
    echo "Passed:  $PASS_COUNT"
    echo "Failed:  $FAIL_COUNT"
    echo "Skipped: $SKIP_COUNT"
} >> "$RESULTS_FILE"

echo ""
log_info "Total:   $TOTAL"
log_success "Passed:  $PASS_COUNT"
if [[ $FAIL_COUNT -gt 0 ]]; then
    log_fail "Failed:  $FAIL_COUNT"
else
    log_info "Failed:  0"
fi
if [[ $SKIP_COUNT -gt 0 ]]; then
    log_warn "Skipped: $SKIP_COUNT"
fi
echo ""
log_info "Full results saved to: $RESULTS_FILE"
