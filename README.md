# LLM Inference Optimization — Huawei Internship Track 3

## Task 1: SGLang Paper Analysis & Demo

**Goal:** Study the [SGLang paper](https://arxiv.org/abs/2312.07104) and demonstrate key concepts.

**Contents:**
- `task1/summary.md` — Paper summary: RadixAttention, Compressed FSM, DSL primitives
- `task1/sglang_demo.py` — Local demo using SGLang DSL via Ollama (Apple Silicon / CPU)

**Demo covers:** `gen()`, `select()`, `fork()`, multi-step chaining, shared prefix (RadixAttention concept), structured output (Compressed FSM concept).

### Running locally

```bash
# Install Ollama and pull the model
brew install ollama
ollama serve &
ollama pull qwen2.5:0.5b

# Set up Python environment
cd task1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python sglang_demo.py
```

## Task 2: LLM Inference Server Simulator

**Goal:** Build a discrete-event simulator that processes [Azure LLM Inference Trace](https://github.com/Azure/AzurePublicDataset) (1M multimodal requests) and finds the minimum number of accelerators satisfying SLA constraints.

**Contents:**
- `task2/simulator.py` — Event-driven simulation engine (~860 lines, fully documented)
- `task2/main.py` — CLI entry point with binary search for optimal N

### Key features

- **3-stage pipeline:** image preprocessing (stage 1) → context prefill (stage 2) → token decode (stage 3)
- **Memory-aware scheduling:** tracks per-accelerator VRAM, prevents OOM
- **Batching with sublinear cost:** `Cost(B) = A * sqrt(B)` models GPU parallelism
- **Shrinking batch decode:** requests finishing early free memory for others
- **EDF scheduling:** earliest-deadline-first for SLA compliance
- **Binary search:** exponential + binary search for minimum N (14 runs instead of 75)

### Running

```bash
cd task2

# Download the trace dataset into task2/ directory:
# AzureLMMInferenceTrace_multimodal.csv (~33MB)
# https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLLMInferenceTrace/

# Quick test (1000 requests)
python main.py --max-requests 1000

# Full run (1M requests, finds minimum N)
python main.py
```

### Result

Minimum **N = 75** accelerators for 1M requests with default SLA (TTFT ≤ 5000ms, per-token ≤ 100ms).
