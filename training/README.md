# ANE Training — Stories110M on Apple Neural Engine

Training a 109M-parameter Llama2-architecture transformer (Stories110M) directly on Apple's Neural Engine using private ANE APIs. This implementation uses a "Weights-as-Tensors" optimization to bypass compilation limits and achieve high throughput.

![Dashboard](dashboard.gif)

## Architecture

- **Model**: Stories110M — dim=768, hidden=2048, heads=12, layers=12, vocab=5000, seq=256
- **Optimization**: **Weights-as-Tensors**. All model weights are passed as dynamic input tensors via IOSurfaces. Kernels are compiled exactly once at startup.
- **72 ANE kernels** total (60 weight-bearing, 12 weight-free `sdpaBwd2`).
- **6 kernel types per layer**: `fwdAttn`, `fwdFFN`, `ffnBwd`, `sdpaBwd1`, `sdpaBwd2`, `qkvBwd`.

## Performance (Optimized)

| Metric | Value |
|-----------|---------------|
| **Training Latency** | **~79.6 ms/step** |
| **Inference Latency (SEQ=256)** | **0.60 ms** |
| **Sustained ANE Throughput** | **~94.4 TFLOPS** |
| **Theoretical Inference TPS** | **~429,000 Tokens/sec** |
| **Weight Sync** | ~3.4 ms per layer (NEON-accelerated) |
| **Compile Budget** | **0 restarts** (Dynamic weight updates) |

## Configuration Variables

Most configuration is handled in [stories_config.h](stories_config.h) and [train_large.m](train_large.m).

### Model Hyperparameters (`stories_config.h`)
- `DIM`: Model dimension (default: 768)
- `HIDDEN`: FFN hidden dimension (default: 2048)
- `NLAYERS`: Number of transformer layers (default: 12)
- `VOCAB`: Vocabulary size (default: 5000)
- `SEQ`: Sequence length / context window (default: 256)

### Training Paths (`train_large.m`)
- `DATA_PATH`: Path to the tokenized binary dataset (default: `tinystories_data00.bin`)
- `MODEL_PATH`: Path to the initial pretrained weights in llama2.c format.
- `CKPT_PATH`: Output path for training checkpoints.

## Compiling & Running

### 1. Prerequisites
Ensure you have a modern Mac with Apple Silicon (M1/M2/M3/M4). 
You will need `xcrun` (Xcode Command Line Tools) and various Python dependencies for data prep and monitoring.

### 2. Prepare Data
The trainer expects a flat binary file of `uint16_t` token IDs.
```bash
# Tokenize raw text into the expected format
python3 tokenize.py
```

### 3. Build and Train
```bash
# Compile the training binary
make train_large

# Start training (fresh start or default steps)
./train_large

# Resume with custom steps and learning rate
./train_large --resume --steps 1000 --lr 1e-4
```

## Dataset Adaptation

To adapt this trainer to any custom text dataset:
1. **Tokenize**: Use a tokenizer to convert your text corpus into a sequence of IDs.
2. **Export**: Save the IDs as a raw binary file of `uint16_t` values.
3. **Configure**: Update `VOCAB`, `SEQ`, and `DATA_PATH` in the config files to match your dataset.
4. **Compile**: Re-run `make train_large`. The ANE kernels will automatically adjust to your new shapes.

## Monitoring with Dashboard

The TUI dashboard provides real-time telemetry on loss, power usage, and model generation.
```bash
pip install blessed psutil numpy
# Dashboard may require sudo for powermetrics access
python3 dashboard.py --resume
```

## Testing the Model

You can test the trained model using the standalone inference script. It uses standard vanilla NumPy to perform the forward pass on the CPU, making it easy to inspect.

### Generate Text
```bash
# Test with a custom prompt and checkpoint
python3 sample.py --prompt "Once upon a time" --ckpt ane_stories110M_ckpt.bin --steps 100
```

### Parameters
- `--prompt`: The starting text for generation.
- `--ckpt`: Path to the training checkpoint (`.bin`).
- `--vocab`: Path to the BPE vocabulary (`vocab.json`).
- `--steps`: Maximum number of tokens to generate.
- `--temp`: Sampling temperature (default 0.8).

### ANE Hardware Benchmark
To measure raw hardware throughput and verify the **Weights-as-Tensors** optimization on the actual ANE silicon, use the C-based benchmark utility:

```bash
# Build the benchmark
make benchmark_ane

# Run 100 iterations of full-model forward pass
./benchmark_ane
```
This utility measure tokens per second and TFLOPS directly on the ANE by running 24 kernels (Attn+FFN) in a continuous loop.

---

## Key Optimization: Weights as Tensors

Previously, ANE training required recompiling kernels every time weights changed, hitting an OS-enforced 119-compile limit. 

The current implementation defines weights as formal function parameters (`tensor<fp16, [dim, dim]>`) in the MIL program. This allows us to:
1. Compile the kernel logic **once**.
2. Update weights between batches by writing directly to **IOSurfaces** via NEON-accelerated loops (`io_write_fp16_t`).
3. Maintain resident memory for the model, eliminating the need for `exec()` restarts.
