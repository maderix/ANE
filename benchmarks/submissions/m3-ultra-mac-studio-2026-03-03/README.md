# Mac Studio M3 Ultra Benchmark Submission (2026-03-03)

This submission targets upstream issue: `#3` (collecting results across Apple Silicon variants).

## Environment

- Upstream commit: `443194bca4491fae4400bae9dad2a0470692bdbf`
- Machine: Mac Studio (`Mac15,14`)
- Chip: Apple M3 Ultra
- CPU cores: 28 total (20P + 8E)
- Memory: 256 GB (`274877906944` bytes)
- OS: macOS 26.3 (`25D125`)
- Toolchain: Apple clang 17.0.0 (`/Library/Developer/CommandLineTools`)

Raw system capture: [`raw/system_info.txt`](raw/system_info.txt)

## Commands Run

Exact commands used are included in [`commands.sh`](commands.sh).

Highlights:

```bash
# Root benchmark
xcrun clang -O2 -framework Foundation -framework IOSurface -framework CoreML \
  -ldl -lobjc -o inmem_peak inmem_peak.m
./inmem_peak

# Training benchmarks
cd training
bash download_data.sh
make train_large train_large_ane
./train_large --steps 20 --lr 1e-4 --ckpt /tmp/train_large.ckpt
./train_large_ane --steps 20 --lr 1e-4 --ckpt /tmp/train_large_ane.ckpt
./train_large_ane --no-ane-extras --steps 20 --lr 1e-4 --ckpt /tmp/train_large_ane_no_extras.ckpt
cd training_dynamic
make train
./train --scratch --steps 20 --lr 1e-4
```

## Training Results (20 steps)

| Pipeline | Wall time | Compile time | Train time | Avg train | ANE TFLOPS | Total TFLOPS |
|---|---:|---:|---:|---:|---:|---:|
| `train_large` | 9471 ms | 7545 ms (79.7%) | 1623 ms (17.1%) | 81.2 ms/step | 1.15 | 2.15 |
| `train_large_ane` | 10898 ms | 9090 ms (83.4%) | 1428 ms (13.1%) | 71.4 ms/step | 1.48 | 2.44 |
| `train_large_ane --no-ane-extras` | 10248 ms | 7455 ms (72.7%) | 2476 ms (24.2%) | 123.8 ms/step | 0.85 | 1.41 |
| `training_dynamic/train --scratch` | 2.9 s | 353 ms (one-time, 12.0%) | 2309 ms | 115.4 ms/step | n/a | n/a |

Raw logs:

- [`raw/train_large.log`](raw/train_large.log)
- [`raw/train_large_ane.log`](raw/train_large_ane.log)
- [`raw/train_large_ane_no_extras.log`](raw/train_large_ane_no_extras.log)
- [`raw/train_dynamic.log`](raw/train_dynamic.log)

## In-Memory Peak Results

Best observed from `inmem_peak`:

- 8.08 TFLOPS at `128x conv 512ch sp64` (`4.29 GFLOP`, `0.531 ms/eval`)

Raw log:

- [`raw/inmem_peak.log`](raw/inmem_peak.log)

## Additional Root Benchmarks

- `inmem_bench`: all configs returned `FAIL(-1)` on this clean setup
- `sram_bench`: all configs returned `FAIL(-1)` on this clean setup

Raw logs:

- [`raw/inmem_bench.log`](raw/inmem_bench.log)
- [`raw/sram_bench.log`](raw/sram_bench.log)

## Notes

- `train_large_ane` had the best per-step throughput in this run.
- Dynamic had the best short-run wall-clock due to one-time compile cost.
- Static pipelines remained compile-dominated over 20 steps.
