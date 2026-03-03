#!/usr/bin/env bash
set -euo pipefail

# Repro commands used for this submission.
# Machine: Mac Studio (Apple M3 Ultra)
# Commit: 443194bca4491fae4400bae9dad2a0470692bdbf

REPO="${REPO:-$HOME/Dev/ANE-upstream}"
ART="${ART:-$REPO/bench_artifacts/m3-ultra-2026-03-03/raw}"

mkdir -p "$ART"
cd "$REPO"

# System capture
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sw_vers
  uname -a
  echo
  echo "=== sysctl ==="
  sysctl hw.model hw.memsize hw.ncpu hw.physicalcpu hw.logicalcpu \
    hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu \
    machdep.cpu.brand_string 2>/dev/null || true
  echo
  echo "=== system_profiler SPHardwareDataType ==="
  system_profiler SPHardwareDataType
  echo
  echo "=== toolchain ==="
  xcode-select -p
  xcrun clang --version
} > "$ART/system_info.txt"

# Root benchmark
xcrun clang -O2 -framework Foundation -framework IOSurface -framework CoreML \
  -ldl -lobjc -o inmem_peak inmem_peak.m
./inmem_peak > "$ART/inmem_peak.log" 2>&1

# Optional root benchmarks (may fail on clean setups)
xcrun clang -O2 -framework Foundation -framework IOSurface -framework CoreML \
  -ldl -lobjc -o inmem_bench inmem_bench.m
./inmem_bench > "$ART/inmem_bench.log" 2>&1 || true

xcrun clang -O2 -framework Foundation -framework IOSurface -framework CoreML \
  -ldl -lobjc -o sram_bench sram_bench.m
./sram_bench > "$ART/sram_bench.log" 2>&1 || true

# Training benchmarks
cd "$REPO/training"
bash download_data.sh > "$ART/download_data.log" 2>&1
make train_large train_large_ane > "$ART/training_make.log" 2>&1
./train_large --steps 20 --lr 1e-4 --ckpt "$ART/train_large.ckpt" > "$ART/train_large.log" 2>&1
./train_large_ane --steps 20 --lr 1e-4 --ckpt "$ART/train_large_ane.ckpt" > "$ART/train_large_ane.log" 2>&1
./train_large_ane --no-ane-extras --steps 20 --lr 1e-4 --ckpt "$ART/train_large_ane_no_extras.ckpt" > "$ART/train_large_ane_no_extras.log" 2>&1

cd "$REPO/training/training_dynamic"
make train > "$ART/training_dynamic_make.log" 2>&1
./train --scratch --steps 20 --lr 1e-4 > "$ART/train_dynamic.log" 2>&1

cd "$REPO"
git rev-parse HEAD > "$ART/upstream_commit.txt"

echo "Done. Raw logs are in: $ART"
