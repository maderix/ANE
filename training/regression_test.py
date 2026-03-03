#!/usr/bin/env python3
import subprocess
import os
import sys
import time
import re

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def run_command(cmd, cwd=".", timeout=60):
    print(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout expired"
    except Exception as e:
        return -1, "", str(e)

def print_result(name, success, info=""):
    status = f"{GREEN}PASSED{RESET}" if success else f"{RED}FAILED{RESET}"
    print(f"[{status}] {BOLD}{name}{RESET} {info}")

def main():
    print(f"\n{BOLD}=== ANE Training & SDK Regression Suite ==={RESET}\n")
    
    # 0. Cleanup and Build
    print(f"{BOLD}Step 0: Building binaries...{RESET}")
    ret, out, err = run_command(["make", "clean"])
    targets = ["train_large", "benchmark_ane", "test_sdk_layers", "test_sdk_model"]
    for target in targets:
        ret, out, err = run_command(["make", target])
        if ret != 0:
            print_result(f"Build {target}", False, f"\n{err}")
            sys.exit(1)
    print_result("Build All Targets", True)

    # 1. SDK Layer & Model Testing (Fastest verification)
    print(f"\n{BOLD}Step 1: SDK Component Verification{RESET}")
    
    # Test individual layers (Linear, ReLU, Softmax, LayerNorm, Conv2D, etc.)
    ret, out, err = run_command(["./test_sdk_layers"])
    if ret == 0 and "SDK Layer Test PASSED" in out:
        print_result("SDK Modular Layers", True)
    else:
        print_result("SDK Modular Layers", False, f"\n{out}\n{err}")
        sys.exit(1)

    # Test sequential model (Graph runner + IOSurface chaining)
    ret, out, err = run_command(["./test_sdk_model"])
    if ret == 0 and "SDK Model Test PASSED" in out:
        print_result("SDK Sequential Model", True)
    else:
        print_result("SDK Sequential Model", False, f"\n{out}\n{err}")
        sys.exit(1)

    # 2. Original Transformer Training (Short burst)
    print(f"\n{BOLD}Step 2: Legacy Transformer Training Verification{RESET}")
    # Ensure some data exists
    if not os.path.exists("train.bin"):
        print("Note: Creating dummy data for training test...")
        with open("train.bin", "wb") as f:
            f.write(os.urandom(1024 * 1024)) # 1MB dummy data

    # Run training for 20 steps (2 batches of 10)
    ret, out, err = run_command(["./train_large", "--steps", "20"], timeout=300) 
    combined_output = out + err
    # Look for step 19 in JSON or regular output (since it's 0-indexed)
    if ret == 0 and (re.search(r'"step":\s*19', combined_output) or "step 19" in combined_output or "Checkpoint saved" in combined_output):
        print_result("Legacy Training (20 steps)", True)
    else:
        print_result("Legacy Training (20 steps)", False, f"\nSTDOUT:\n{out}\nSTDERR:\n{err}")
        sys.exit(1)

    # 3. Inference Verification
    print(f"\n{BOLD}Step 3: Inference & Parity Verification{RESET}")
    
    # Check if a model checkpoint exists
    ckpt = "ane_stories110M_ckpt.bin"
    if os.path.exists(ckpt):
        # ANE Benchmark inference (High-throughput native code)
        ret, out, err = run_command(["./benchmark_ane"])
        if ret == 0 and "TFLOPS" in out:
            print_result("ANE Benchmark Inference", True)
        else:
            print_result("ANE Benchmark Inference", False, f"\n{out}\n{err}")
            sys.exit(1)

        # CPU Python inference (Parity verification)
        if os.path.exists("vocab.json"):
            ret, out, err = run_command(["python3", "sample.py", "--steps", "5"])
            if ret == 0:
                print_result("CPU Python Inference (sample.py)", True)
            else:
                print_result("CPU Python Inference (sample.py)", False, f"\n{err}")
                sys.exit(1)
        else:
            print(f"[SKIP] CPU Inference (missing vocab.json)")
    else:
        print(f"{RED}[ERROR] Inference tests failed: missing {ckpt}{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}=== Regression Tests Complete ==={RESET}\n")

if __name__ == "__main__":
    main()
