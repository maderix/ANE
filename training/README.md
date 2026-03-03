# ANE Training & SDK — General-Purpose Neural Engine Platform
Training a 109M-parameter Llama2-architecture transformer (Stories110M) directly on Apple's Neural Engine. This repository has evolved into a fully-featured **ANE SDK** for developing and training arbitrary neural network architectures on Apple Silicon.

![Dashboard](dashboard.gif)

## 🚀 The ANE SDK
The ANE SDK provides a high-level API for defining, training, and benchmarking models on the Neural Engine without manual MIL (Model Intermediate Language) string concatenation.

### Key Features
- **Modular Layer Library**: High-level builders for NLP and Vision (`Linear`, `Conv2D`, `LayerNorm`, `Softmax`, etc.).
- **Graph Orchestration**: Automatic activation chaining and IOSurface management via a `Sequential` model container.
- **Weights-as-Tensors**: Every layer utilizes a zero-recompile optimization pattern, allowing dynamic weight updates for training.
- **Native Performance**: Sustained throughput of **>90 TFLOPS** across modular components.

### Architecture Comparison

| Specialized (Legacy) | ANE SDK (General-Purpose) |
|----------------------|---------------------------|
| **Fixed Topology**: Transformer only | **Dynamic Topology**: Arbitrary layers |
| **Manual I/O**: Manual surface pointers | **Automated Chaining**: Sequential runner |
| **Hardcoded MIL**: `stories_mil.h` | **Modular MIL**: `layers/core.h`, `layers/cnn.h` |
| **Optimized Path**: Hand-tuned SDPA | **Ease of Use**: PyTorch-like API |

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

## ANE SDK Usage

You can build arbitrary models using the modular layer library in `layers/`.

### 1. Define Model Architecture
```objectivec
#import "layers/anesdk.h"

// Define layers
ANESDKLayer l1 = anesdk_linear_create("fc1", 768, 2048, 256);
ANESDKLayer l2 = anesdk_relu_create("relu1", 2048, 1, 256);
ANESDKLayer l3 = anesdk_layernorm_create("ln1", 2048, 256);

// Assemble into Sequential model
ANESDKLayer layers[] = { l1, l2, l3 };
ANESDKModel model = anesdk_model_sequential_create(layers, 3);
```

### 2. Run Forward Pass
The SDK automatically manages IOSurface chaining between layers.
```objectivec
// Write input to the first layer
io_write_fp16(model.layers[0].kern->inputs[0], input_data, 768, 256);

// Run the whole graph on ANE
anesdk_model_forward(&model);

// Read result from the last layer
io_read_fp16(model.layers[2].kern->ioOut, output_data, 0, 2048, 256);
```

### 3. Automated Verification
The repository includes a regression suite that verifies both the legacy Transformer and your new SDK layers.
```bash
# Build and run all tests (Fast SDK tests -> Training -> Inference)
make regression
```

---

## Performance Utilities

### ANE Hardware Benchmark
To measure raw hardware throughput and verify the **Weights-as-Tensors** optimization, use the native C-based benchmark:
```bash
make benchmark_ane
./benchmark_ane
```
Average Forward Pass (SEQ=256): **0.60 ms** | Throughput: **~94.4 TFLOPS**.

### Model Inference Utility (`sample.py`)
Verify trained checkpoints on the CPU using vanilla NumPy.
```bash
python3 sample.py --prompt "Once upon a time" --ckpt ane_stories110M_ckpt.bin
```

---

## Key Optimization: Weights as Tensors

Previously, ANE training required recompiling kernels every time weights changed, hitting an OS-enforced 119-compile limit. 

The current implementation defines weights as formal function parameters (`tensor<fp16, [dim, dim]>`) in the MIL program. This allows us to:
1. Compile the kernel logic **once**.
2. Update weights between batches by writing directly to **IOSurfaces** via NEON-accelerated loops (`io_write_fp16_t`).
3. Maintain resident memory for the model, eliminating the need for `exec()` restarts.
