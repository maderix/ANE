#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present (LMS_API_KEY, LMS_PORT, LMS_MODEL)
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

BINARY="$SCRIPT_DIR/qwen_ane"
WEIGHTS="$SCRIPT_DIR/qwen05b.bin"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen2.5-0.5B-Instruct}"
SOCK="/tmp/qwen_ane_bench.sock"
HTTP_PORT=8877
RESULTS_JSON="$SCRIPT_DIR/benchmark_results.json"

# --- Prompt suite ---
PROMPT_NAMES=(   "tiny"   "short"          "medium"                                                 "long"                                                                      "stress")
PROMPTS=(        "Hi"     "What is 2+2?"   "Explain how neural networks work in 3 sentences."       "Write a short story about a robot learning to paint. Include dialogue."     "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.")
MAX_TOKENS=(     10       20               100                                                      200                                                                         50)

info()  { printf "\033[1;34m%s\033[0m\n" "$1"; }
dim()   { printf "\033[2m%s\033[0m\n" "$1"; }

# Extract a numeric or string value from flat JSON. No python needed.
# Usage: json_val '{"key":123}' "key"  →  123
json_val() {
    local json="$1" key="$2"
    echo "$json" | sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^,\"}\]*\)\"\{0,1\}.*/\1/p" | head -1
}

# Extract the "text" field which may contain escaped chars and commas.
# Grabs everything between "text":" and the next unescaped quote.
json_text() {
    local json="$1"
    echo "$json" | sed -n 's/.*"text":"\(.*\)","prompt_tokens".*/\1/p' | sed 's/\\n/ /g; s/\\"//g'
}

# Truncate a float string to integer: "317.2" → "317"
trunc() { echo "${1%%.*}"; }

# Average an array of numbers using awk. Handles both ints and floats.
# Usage: shell_avg "1.5" "2.3" "3.1"  →  2.3
shell_avg() { printf '%s\n' "$@" | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "0"}'; }
shell_avg_int() { printf '%s\n' "$@" | awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "0"}'; }

# --- Preflight ---
if [ ! -f "$BINARY" ]; then
    echo "Binary not found: $BINARY"
    echo "Run setup.sh first: $SCRIPT_DIR/setup.sh"
    exit 1
fi
if [ ! -f "$WEIGHTS" ]; then
    echo "Weights not found: $WEIGHTS"
    echo "Run setup.sh first: $SCRIPT_DIR/setup.sh"
    exit 1
fi

# Detect hardware
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
MACOS=$(sw_vers -productVersion 2>/dev/null || echo "Unknown")
MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
MEM_GB=$((MEM_BYTES / 1073741824))

echo ""
info "=== ANE Inference Benchmark (qwen_ane) ==="
echo "Hardware: $CHIP"
echo "macOS:    $MACOS"
echo "Memory:   ${MEM_GB} GB"
echo "Model:    Qwen2.5-0.5B-Instruct (BF16, 494M params)"
echo ""

# --- Phase 1: Server mode benchmark (HTTP API) ---
info "Phase 1: Server mode (persistent ANE kernels via HTTP API)"
dim "Starting server on port $HTTP_PORT..."

# Start HTTP server in background
"$BINARY" "$WEIGHTS" --http "$HTTP_PORT" --model-dir "$MODEL_DIR" > /tmp/qwen_bench_server.log 2>&1 &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    rm -f "$SOCK" /tmp/qwen_bench_server.log
}
trap cleanup EXIT

# Wait for READY
for i in $(seq 1 30); do
    if grep -q "READY" /tmp/qwen_bench_server.log 2>/dev/null; then
        break
    fi
    sleep 1
done

if ! grep -q "READY" /tmp/qwen_bench_server.log 2>/dev/null; then
    echo "Server failed to start. Log:"
    cat /tmp/qwen_bench_server.log
    exit 1
fi
dim "Server ready (PID $SERVER_PID)"
echo ""

# Warmup: first request primes any remaining caches
dim "Warmup run (discarded)..."
curl -s "http://127.0.0.1:$HTTP_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","max_tokens":5}' > /dev/null 2>&1
echo ""

# Print table header
printf "%-10s %5s %5s %10s %10s %10s %10s %10s %10s\n" \
    "Prompt" "In" "Out" "Prefill" "Decode" "TTFT" "Infer" "Rndtrip" "Overhead"
printf "%-10s %5s %5s %10s %10s %10s %10s %10s %10s\n" \
    "" "tok" "tok" "(t/s)" "(t/s)" "(ms)" "(ms)" "(ms)" "(ms)"
printf '%.0s─' {1..85}; echo ""

# Arrays for averages
declare -a P_TPS_ARR D_TPS_ARR INF_MS_ARR TTFT_MS_ARR RT_MS_ARR

JSON_ENTRIES=""
NUM_PROMPTS=${#PROMPTS[@]}

for i in $(seq 0 $((NUM_PROMPTS - 1))); do
    NAME="${PROMPT_NAMES[$i]}"
    PROMPT="${PROMPTS[$i]}"
    MAXTOK="${MAX_TOKENS[$i]}"

    RT_T0=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
    RESP=$(curl -s "http://127.0.0.1:$HTTP_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $MAXTOK}" 2>&1)
    RT_T1=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
    RT_MS=$(echo "$RT_T0 $RT_T1" | awk '{printf "%.0f", ($2 - $1) * 1000}')

    # Parse server JSON with pure shell -- no python
    P_TOKENS=$(json_val "$RESP" "prompt_tokens")
    G_TOKENS=$(json_val "$RESP" "gen_tokens")
    P_TPS=$(json_val "$RESP" "prefill_tps")
    D_TPS=$(json_val "$RESP" "decode_tps")
    TTFT_MS=$(trunc "$(json_val "$RESP" "ttft_ms")")
    INF_MS=$(trunc "$(json_val "$RESP" "inference_ms")")
    TOTAL_MS=$(trunc "$(json_val "$RESP" "total_ms")")
    TEXT=$(json_text "$RESP")
    OVERHEAD=$((RT_MS - TOTAL_MS))

    printf "%-10s %5s %5s %10s %10s %10s %10s %10s %10s\n" \
        "$NAME" "$P_TOKENS" "$G_TOKENS" "$P_TPS" "$D_TPS" "$TTFT_MS" "$INF_MS" "$RT_MS" "$OVERHEAD"

    P_TPS_ARR+=("$P_TPS")
    D_TPS_ARR+=("$D_TPS")
    INF_MS_ARR+=("$INF_MS")
    TTFT_MS_ARR+=("$TTFT_MS")
    RT_MS_ARR+=("$RT_MS")

    # Build JSON entry
    JSON_ENTRIES="$JSON_ENTRIES{\"name\":\"$NAME\",\"prompt_tokens\":$P_TOKENS,\"gen_tokens\":$G_TOKENS,\"prefill_tps\":$P_TPS,\"decode_tps\":$D_TPS,\"ttft_ms\":$TTFT_MS,\"inference_ms\":$INF_MS,\"roundtrip_ms\":$RT_MS},"

    # Print response text indented below
    echo "    → $TEXT"
    echo ""
done

printf '%.0s─' {1..85}; echo ""

# Averages (pure shell, no python)
AVG_P=$(shell_avg "${P_TPS_ARR[@]}")
AVG_D=$(shell_avg "${D_TPS_ARR[@]}")
AVG_INF=$(shell_avg_int "${INF_MS_ARR[@]}")
AVG_TTFT=$(shell_avg_int "${TTFT_MS_ARR[@]}")
AVG_RT=$(shell_avg_int "${RT_MS_ARR[@]}")
AVG_OVERHEAD=$((AVG_RT - AVG_INF))
printf "%-10s %5s %5s %10s %10s %10s %10s %10s %10s\n" "Average" "" "" "$AVG_P" "$AVG_D" "$AVG_TTFT" "$AVG_INF" "$AVG_RT" "$AVG_OVERHEAD"
echo ""
info "Infer = server-reported (pure processing). Rndtrip = wall-clock (what clients see)."
echo ""

# --- Phase 2: Cold start measurement ---
info "Phase 2: Cold start (single-shot, recompiles ANE kernels)"

# Kill server, run single-shot
kill "$SERVER_PID" 2>/dev/null || true
sleep 1

# Use perl for sub-second timing (available on all macOS, no python)
COLD_T0=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
COLD_OUT=$("$BINARY" "$WEIGHTS" "151644 8948 198 2610 525 264 10950 17847 13 151645 198 151644 872 198 13048 151645 198 151644 77091 198" 10 2>&1 || true)
COLD_T1=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
COLD_MS=$(echo "$COLD_T0 $COLD_T1" | awk '{printf "%.0f", ($2 - $1) * 1000}')

echo "Cold start latency: ${COLD_MS}ms (includes ANE kernel compilation)"
echo ""

# Re-start server for any additional tests
"$BINARY" "$WEIGHTS" --http "$HTTP_PORT" --model-dir "$MODEL_DIR" > /tmp/qwen_bench_server.log 2>&1 &
SERVER_PID=$!

# --- Phase 3: Repeated prompt (consistency check) ---
info "Phase 3: Decode speed consistency (5x same prompt)"

for retry in $(seq 1 15); do
    if grep -q "READY" /tmp/qwen_bench_server.log 2>/dev/null; then break; fi
    sleep 1
done

printf "%-6s %10s %10s %10s\n" "Run" "Prefill" "Decode" "Infer(ms)"
printf '%.0s─' {1..40}; echo ""

for run in $(seq 1 5); do
    RESP=$(curl -s "http://127.0.0.1:$HTTP_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Count from 1 to 10", "max_tokens": 50}' 2>&1)
    P=$(json_val "$RESP" "prefill_tps")
    D=$(json_val "$RESP" "decode_tps")
    IM=$(trunc "$(json_val "$RESP" "inference_ms")")
    printf "%-6s %10s %10s %10s\n" "#$run" "$P" "$D" "$IM"
done
echo ""

# --- Save JSON results ---
JSON="{
  \"hardware\": \"$CHIP\",
  \"macos\": \"$MACOS\",
  \"memory_gb\": $MEM_GB,
  \"model\": \"Qwen2.5-0.5B-Instruct\",
  \"mode\": \"http_server\",
  \"cold_start_ms\": $COLD_MS,
  \"avg_prefill_tps\": $AVG_P,
  \"avg_decode_tps\": $AVG_D,
  \"avg_inference_ms\": $AVG_INF,
  \"avg_roundtrip_ms\": $AVG_RT,
  \"avg_ttft_ms\": $AVG_TTFT,
  \"results\": [${JSON_ENTRIES%,}]
}"
echo "$JSON" > "$RESULTS_JSON"
dim "Results saved to $RESULTS_JSON"
echo ""

# --- Phase 4: LM Studio comparison (if running) ---
LMS_PORT="${LMS_PORT:-1234}"
LMS_MODEL="${LMS_MODEL:-qwen2.5-0.5b-instruct}"
LMS_API_KEY="${LMS_API_KEY:-}"

# Check if LM Studio is running
LMS_REACHABLE=0
if curl -s --max-time 2 "http://localhost:$LMS_PORT/api/v1/chat" -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1; then
    LMS_REACHABLE=1
fi

if [ "$LMS_REACHABLE" -eq 1 ]; then
    info "Phase 4: LM Studio comparison (localhost:$LMS_PORT)"

    # If no API key, prompt for it
    if [ -z "$LMS_API_KEY" ]; then
        echo ""
        echo "  LM Studio requires an API key."
        echo "  Find it in LM Studio > Developer tab > API key"
        echo "  Or set LMS_API_KEY env var before running."
        echo ""
        printf "  Enter LM Studio API key (or press Enter to skip): "
        read -r LMS_API_KEY
        if [ -z "$LMS_API_KEY" ]; then
            dim "Skipping LM Studio benchmark."
            LMS_REACHABLE=0
        fi
    fi
fi

if [ "$LMS_REACHABLE" -eq 1 ] && [ -n "$LMS_API_KEY" ]; then
    echo ""
    printf "%-10s %5s %5s %10s %10s %10s\n" \
        "Prompt" "In" "Out" "Decode" "TTFT" "Rndtrip"
    printf "%-10s %5s %5s %10s %10s %10s\n" \
        "" "tok" "tok" "(t/s)" "(ms)" "(ms)"
    printf '%.0s─' {1..55}; echo ""

    declare -a LMS_LATENCIES LMS_TPS_ARR LMS_TTFT_ARR
    LMS_JSON_ENTRIES=""

    for i in $(seq 0 $((NUM_PROMPTS - 1))); do
        NAME="${PROMPT_NAMES[$i]}"
        PROMPT="${PROMPTS[$i]}"

        T0=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
        LMS_RESP=$(curl -s --max-time 120 "http://localhost:$LMS_PORT/api/v1/chat" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $LMS_API_KEY" \
            -d "{\"model\":\"$LMS_MODEL\",\"system_prompt\":\"You are a helpful assistant. Be concise.\",\"input\":\"$PROMPT\"}" 2>&1)
        T1=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
        LMS_MS=$(echo "$T0 $T1" | awk '{printf "%.0f", ($2 - $1) * 1000}')

        eval "$(echo "$LMS_RESP" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    text = r.get('output', [{}])[0].get('content', '').replace(chr(10),' ').replace('\"', '')
    s = r.get('stats', {})
    tps = s.get('tokens_per_second', 0)
    ttft = int(s.get('time_to_first_token_seconds', 0) * 1000)
    in_tok = s.get('input_tokens', 0)
    out_tok = s.get('total_output_tokens', 0)
    print(f'LMS_TEXT=\"{text}\"')
    print(f'LMS_TPS={tps:.1f}')
    print(f'LMS_TTFT={ttft}')
    print(f'LMS_IN={in_tok}')
    print(f'LMS_OUT={out_tok}')
except Exception as e:
    print(f'LMS_TEXT=\"(parse error)\"')
    print('LMS_TPS=0')
    print('LMS_TTFT=0')
    print('LMS_IN=0')
    print('LMS_OUT=0')
" 2>/dev/null)"

        printf "%-10s %5s %5s %10s %10s %10s\n" "$NAME" "$LMS_IN" "$LMS_OUT" "$LMS_TPS" "$LMS_TTFT" "$LMS_MS"
        echo "    → $LMS_TEXT"
        echo ""
        LMS_LATENCIES+=("$LMS_MS")
        LMS_TPS_ARR+=("$LMS_TPS")
        LMS_TTFT_ARR+=("$LMS_TTFT")
        LMS_JSON_ENTRIES="$LMS_JSON_ENTRIES{\"name\":\"$NAME\",\"latency_ms\":$LMS_MS,\"tps\":$LMS_TPS,\"ttft_ms\":$LMS_TTFT,\"input_tokens\":$LMS_IN,\"output_tokens\":$LMS_OUT},"
    done

    printf '%.0s─' {1..55}; echo ""

    # Averages (awk, no python)
    LMS_AVG_LAT=$(shell_avg_int "${LMS_LATENCIES[@]}")
    LMS_AVG_TPS=$(shell_avg "${LMS_TPS_ARR[@]}")
    LMS_AVG_TTFT=$(shell_avg_int "${LMS_TTFT_ARR[@]}")
    printf "%-10s %5s %5s %10s %10s %10s\n" "Average" "" "" "$LMS_AVG_TPS" "$LMS_AVG_TTFT" "$LMS_AVG_LAT"
    echo ""

    # Side-by-side comparison
    info "=== Side-by-Side Comparison ==="
    dim "(Round-trip = wall-clock from client, apples-to-apples)"
    echo ""
    printf "%-24s %15s %15s\n" "" "ANE (qwen_ane)" "LM Studio"
    printf '%.0s─' {1..56}; echo ""
    printf "%-24s %12s t/s %12s t/s\n" "Decode speed" "$AVG_D" "$LMS_AVG_TPS"
    printf "%-24s %12s t/s %12s\n"     "Prefill speed" "$AVG_P" "N/A"
    printf "%-24s %12s ms  %12s ms\n"  "TTFT" "$AVG_TTFT" "$LMS_AVG_TTFT"
    printf "%-24s %12s ms  %12s ms\n"  "Avg round-trip" "$AVG_RT" "$LMS_AVG_LAT"
    printf "%-24s %12s ms  %12s ms\n"  "  (server-only)" "$AVG_INF" "N/A"
    printf "%-24s %12s ms  %12s\n"     "Cold start" "$COLD_MS" "N/A"
    printf "%-24s %15s %15s\n"         "Precision" "F32 (from BF16)" "GGUF quantized"
    printf "%-24s %15s %15s\n"         "Accelerator" "Neural Engine" "CPU/GPU"
    printf "%-24s %15s %15s\n"         "Timing method" "Wall-clock" "Wall-clock"
    echo ""

    # Append LM Studio block to JSON results (pure shell, no python)
    # Remove trailing "}" and newline, append lm_studio object
    LMS_JSON_BLOCK=",
  \"lm_studio\": {
    \"port\": $LMS_PORT,
    \"model\": \"$LMS_MODEL\",
    \"avg_latency_ms\": $LMS_AVG_LAT,
    \"avg_tps\": $LMS_AVG_TPS,
    \"avg_ttft_ms\": $LMS_AVG_TTFT,
    \"results\": [${LMS_JSON_ENTRIES%,}]
  }
}"
    # Replace the final "}" with the LM Studio block
    sed -i '' '$ s/}$//' "$RESULTS_JSON"
    printf '%s\n' "$LMS_JSON_BLOCK" >> "$RESULTS_JSON"
    dim "LM Studio results added to $RESULTS_JSON"
else
    info "=== LM Studio Comparison ==="
    echo ""
    if [ "$LMS_REACHABLE" -eq 0 ]; then
        echo "  LM Studio server not detected on localhost:$LMS_PORT"
        echo ""
        echo "  To enable automatic comparison:"
        echo "  1. Open LM Studio, download Qwen2.5-0.5B-Instruct (GGUF)"
        echo "  2. Load the model, go to Developer tab > Start Server"
        echo "  3. Re-run this benchmark"
        echo ""
        echo "  Or set env vars: LMS_PORT=1234 LMS_API_KEY=your-key ./benchmark.sh"
    fi
    echo ""
    echo "  Manual test:"
    echo "  curl http://localhost:1234/api/v1/chat \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -H 'Authorization: Bearer YOUR_API_KEY' \\"
    echo "    -d '{\"model\":\"qwen2.5-0.5b-instruct\",\"system_prompt\":\"You are a helpful assistant.\",\"input\":\"What is 2+2?\"}'"
    echo ""
    echo "  ANE (this benchmark):  prefill=${AVG_P} t/s, decode=${AVG_D} t/s, inference=${AVG_INF}ms"
    echo ""
    echo "  Note: LM Studio uses quantized GGUF (CPU/GPU) while we use"
    echo "  BF16 weights (full precision) running on the Neural Engine."
fi
echo ""
