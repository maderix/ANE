# Contribution submission guide

This file summarizes what was done on branch `contribution/benchmark-m5-and-fixes` and how to submit it.

---

## 1. Benchmark (submit to Issue #3)

**Link:** https://github.com/maderix/ANE/issues/3

**Post this as a new comment:**

```
## M5 MacBook Pro benchmark (static pipeline, 20 steps)

- **Chip:** Apple M5, 10-core (4P+6E)
- **RAM:** 24 GB
- **macOS:** 26.3 (Build 25D125)
- **Run:** `./train_large --data ./tinystories_data00.bin --steps 20 --lr 1e-4`

### Efficiency report
- Total steps: 20
- Wall time: 10423 ms (10.4 s)
- Compile time: 7187 ms (69.0%)
- Train time: 2542 ms (24.4%)
- **Avg train: 127.1 ms/step**
- ANE TFLOPS: 0.73 sustained
- ANE utilization: 4.6% of 15.8 TFLOPS

Full output with JSON lines is in `benchmarks/my_m5_benchmark_output.txt` (or paste the contents below).
```

Then paste the contents of `benchmarks/my_m5_benchmark_output.txt` in the same comment, or attach it.

---

## 2. Bug fix (PR)

**Fix:** Guard short token datasets in `train_large_ane.m` and `training/training_dynamic/train.m`.

**Why:** When `n_tokens <= SEQ + 1`, the expression `max_pos = n_tokens - SEQ - 1` underflows (unsigned), leading to a huge random range and possible out-of-bounds reads. `train_large.m` already had this guard; the other two pipelines did not.

**Changes:**
- `training/train_large_ane.m`: After `n_tokens = data_len / 2`, add a check that fails early with a clear error, munmap and close the fd, and return 1.
- `training/training_dynamic/train.m`: Same guard added.

**Suggested PR title:** `fix: guard short token datasets in train_large_ane and dynamic pipeline`

**Suggested PR description:**

```markdown
## Summary
- Add a token dataset length guard in `training/train_large_ane.m`
- Add the same guard in `training/training_dynamic/train.m`
- Fail early with a clear error when the dataset is too short for one (input, target) window

## Why
Both paths use `max_pos = n_tokens - SEQ - 1`. When `n_tokens <= SEQ + 1`, this unsigned subtraction underflows, producing a huge range and potentially out-of-bounds reads. `train_large.m` already had this guard (lines 299–304); this PR aligns the other two pipelines.

## Validation
- `make -C training train_large_ane` — builds
- `make -C training/training_dynamic train` — builds
- With a too-short data file, both binaries exit with the new error message.
```

---

## 3. Optional: benchmark data in repo

Branch also adds:
- `benchmarks/my_m5_benchmark_output.txt` — full benchmark log
- One new entry in `benchmarks/community_results.json` for this M5 run (contributor: `log-wade`)

You can either:
- Include the `community_results.json` update in the same PR as the bug fix, or
- Omit it and only post the benchmark to Issue #3 (maintainer may update the report from the issue).

---

## 4. Before opening the PR

1. **Fork the repo** on GitHub (if you haven’t): https://github.com/maderix/ANE → Fork.
2. **Add your fork as a remote and push:**
   ```bash
   git remote add myfork git@github.com:YOUR_USERNAME/ANE.git
   git push myfork contribution/benchmark-m5-and-fixes
   ```
3. Open a PR from `myfork/contribution/benchmark-m5-and-fixes` to `maderix/ANE` main.
4. Post the benchmark comment to Issue #3 (link above).

---

## 5. Replace contributor name

In `benchmarks/community_results.json`, the new entry uses `"contributor": "log-wade"`. Change that to your GitHub username if different.
