# Contributing to ANE Training

Thanks for your interest in contributing! This community fork welcomes benchmark submissions, bug fixes, and research contributions.

## Benchmark Submissions (Easiest Way to Contribute)

The single most valuable thing you can do is run the benchmark on your hardware and submit results.

### Quick Version

```bash
bash scripts/run_community_benchmark.sh
```

The script will guide you through everything, including optional auto-submission to the dashboard.

### What Gets Collected

- Your chip model (e.g., Apple M4 Max)
- macOS version, memory, core counts
- SRAM probe results (TFLOPS vs weight size)
- In-memory peak TFLOPS
- Training performance (optional, requires training data)
- Your GitHub username (optional)

No personal data, no IP addresses stored (only hashed for rate limiting).

## Bug Reports

Open an issue with:
- Your hardware (chip, macOS version, memory)
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output

## Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes
4. Test on your hardware
5. Submit a Pull Request

### Code Style

- Objective-C: follow the existing style in `training/` (no ARC annotations in headers, `_Float16` for fp16)
- Shell scripts: use `set -euo pipefail`, quote variables
- Python: minimal dependencies, Python 3.11+ compatible

### Areas Where Help is Needed

- **Benchmarks on hardware we don't have**: M1, M2, M3, M3 Pro/Max/Ultra, M4 Pro, M5
- **Reducing compilation overhead**: currently 80-85% of wall time
- **`_ANEChainingRequest` research**: pipelining multiple ANE operations without recompile
- **`_ANEPerformanceStats` investigation**: getting real hardware timing data
- **Larger model support**: scaling beyond Stories110M

## Questions?

Open a GitHub issue or discussion. We're happy to help.
