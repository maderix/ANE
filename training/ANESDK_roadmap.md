# ANE SDK Roadmap: General-Purpose Neural Engine Development Kit

This roadmap outlines the evolution of the current Apple Neural Engine (ANE) training infrastructure into a modular, high-level SDK for developing and training arbitrary neural network architectures on Apple Silicon.

## 🌟 Strategic Vision: "PyTorch for ANE"
Transform low-level, transformer-specific MIL (Model Intermediate Language) generation into a modular, layer-based system that allows developers to define, train, and benchmark any architecture (CNNs, MLPs, RNNs) with minimal boilerplate.

---

## 🛠 Phase 1: Modular Layer Abstractions (Short Term)
**Goal:** Decouple MIL generation from the Transformer-specific logic.
- [x] **ANE-MIL Layer Library**: Created a repository of optimized MIL builders for core primitives:
  - `Linear(in, out)`, `Conv2D(kernel, stride, padding)`
  - `ReLU`, `GELU`, `Sigmoid`, `Softmax` activations
  - `LayerNorm` and `RMSNorm`
- [x] **Unified Tensor API**: High-level wrapper around `IOSurface` and `NEON` via `anesdk.h`.
- [x] **Weights-as-Tensors by Default**: Every layer automatically utilizes the dynamic weight update optimization (zero-recompile).

## 🚀 Phase 2: Automated Graph Engine (Medium Term)
**Goal:** Automate the orchestration of multiple kernels into a cohesive model.
- [x] **ANEGraph Orchestrator**: Implemented **Sequential** model container that automates execution order.
- [ ] **Automatic Backward Pass**: Orchestration of backward kernels in reverse order.
- [ ] **Automatic Gradient Management**: Logic to handle gradient accumulation and weight updates across multi-layer graphs.
- [ ] **Optimizer Library**: Implement standard optimizers (SGD, Adam, AdamW) as native C++ components using the Accelerate framework.

## 📈 Phase 3: Developer Ecosystem & Tooling (Long Term)
**Goal:** Improve developer velocity and integration.
- [ ] **Python Bridge (PyANE)**: A lightweight Python library for defining models that compiles directly to ANE-executable graph binaries.
- [ ] **Model Profiler**: Native tools to measure TFLOPS, memory bandwidth, and ANE utilization per-layer.
- [ ] **Deployment Export**: One-click export to CoreML `.mlpackage` for final production deployment.

---

## 🏁 Success Metrics
- **Agnosticism**: Ability to run a CIFAR-10 CNN and a Stories110M Transformer using the same core runtime.
- **Performance**: Maintain >90 TFLOPS sustained throughput across various architectures.
- **Simplicity**: Reduce the lines of code required to define a new model by >70%.

> [!NOTE]
> This SDK leverages private ANE infrastructure to bypass the limitations of public CoreML training, specifically focusing on high-throughput, on-device weight updates.
