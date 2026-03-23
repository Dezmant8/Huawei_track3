# SGLang: Efficient Execution of Structured Language Model Programs

**Paper:** https://arxiv.org/html/2312.07104v2

## 1. Motivation and Problem Setting

Modern LLM applications such as agents, RAG systems, multi-turn chat, and structured output generation require multiple generation calls with intermediate logic. Existing systems such as vLLM, Guidance, and LMQL process each request in isolation, which leads to:

- Redundant computation due to repeated processing of shared prefixes
- Inefficient GPU memory usage because the KV cache is discarded between requests
- Slow constrained decoding due to character-level filtering

SGLang addresses these problems through a **co-design of the language and the runtime**.

## 2. Frontend: A DSL for LLM Programs

SGLang is a Python-embedded DSL with the following primitives:

| Primitive | Description |
|----------|-------------|
| `gen(name, regex=...)` | Generates text with optional regex constraints |
| `select(name, options)` | Chooses the most likely option from a list |
| `extend / +=` | Appends text to the prompt state |
| `fork(n)` | Creates `n` parallel prompt branches |
| `join` | Merges parallel branches |
| `image / video` | Multimodal inputs |

The prompt state is represented as an **asynchronous stream**: primitives are dispatched in a non-blocking manner, and synchronization happens only when the result is read. This enables automatic parallelism within the program.

**Execution modes:**
- **Interpreter mode** (default): asynchronous streams with background execution
- **Compiler mode**: traces the program into a computation graph for static optimizations

## 3. Runtime: RadixAttention

The key innovation is **RadixAttention** — a system for automatic KV-cache reuse based on a radix tree (compressed trie).

### How it works:
1. The KV cache is **not discarded** after a request finishes; instead, it is stored in a radix tree
2. Keys are token sequences, and values are KV-cache tensors stored in GPU memory
3. New requests search for **prefix matches**, and the matched portion does not need to be recomputed

### Memory management mechanisms:
- **LRU eviction**: leaves are evicted first, while internal nodes are preserved for reuse
- **Reference counting**: protects nodes currently used by active requests
- **Dynamic memory allocation** between cache and active requests, without fixed pools

### Cache-aware scheduling:
- **Theorem 3.1**: a DFS processing order achieves the optimal cache hit rate when the cache is large enough
- In practice, SGLang uses **longest-shared-prefix-first** (equivalent to DFS), a greedy policy based on shared prefix length
- This achieves about **96% of the optimal cache hit rate**

### Distributed mode:
- Each worker stores a subtree, while the router maintains a meta-tree over all subtrees
- Request dispatch is optimized for data locality

## 4. Compressed Finite State Machine (FSM) for Constrained Decoding

The standard approach — character-level masking of invalid tokens using an FSM — is inefficient.

**SGLang’s optimization:**
1. An FSM is built at the character/string level
2. Chains of **singular transitions** are identified, where there is only one valid next token
3. These singular transitions are **compressed** into a single step, so the entire deterministic substring is decoded in one forward pass

**Example:** the string `{"summary": "` (which may span multiple tokens) can be generated in one step instead of character-by-character decoding.

**Result:** a **1.6x speedup** for JSON decoding.

**Limitation:** probability distributions may become distorted when the constraints span multiple tokens.

## 5. Benchmarks and Results

**Models:** Llama-7B/70B, Mixtral-8x7B, LLaVA (multimodal)  
**GPUs:** NVIDIA A10G (24GB), A100 (80GB)

### Key results:

| Task | Throughput speedup | Latency reduction |
|------|--------------------|------------------|
| Few-shot (MMLU, HellaSwag) | up to 6.4x | up to 3.7x |
| Agent tasks (ReAct) | ~3–5x | ~2–3x |
| Multimodal (LLaVA) | up to 6x | — |
| JSON decoding | 1.6x (FSM only) | — |
| Multi-turn chat | ~2–3x | — |

### Cache hit rates:
- 50–99%, depending on the task
- In production (Chatbot Arena): 52.4% for LLaVA-Next-34B, 74.1% for Vicuna-33B
- 1.7x reduction in first-token latency in production

**RadixAttention overhead is below 0.3%** on tasks without cache reuse.

## 6. Comparison with Competing Systems

| Aspect | LMQL | Guidance | vLLM | SGLang |
|--------|------|----------|------|--------|
| Language | Custom syntax | Python | API | Python DSL |
| Batching | No | No | Yes | Yes |
| KV-cache reuse | No | No | Only system prompt | Radix tree (multi-level) |
| Constrained decoding | Yes | Yes | No | Compressed FSM |
| Fork/join parallelism | No | No | No | Yes |

SGLang is unique because it **combines** a specialized language with a co-designed runtime.

## 7. Architecture

```text
┌─────────────────────────┐
│   Frontend (Python DSL) │
│  gen, select, fork, join│
│  async stream execution │
│  prefix hints → runtime │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   SGLang Runtime (SRT)  │
│  ┌────────────────────┐ │
│  │  RadixAttention    │ │
│  │  (KV cache in      │ │
│  │   radix tree)      │ │
│  ├────────────────────┤ │
│  │  Cache-aware       │ │
│  │  Scheduler         │ │
│  ├────────────────────┤ │
│  │  Compressed FSM    │ │
│  │  (constrained gen) │ │
│  ├────────────────────┤ │
│  │  FlashInfer kernels│ │
│  │  Tensor parallelism│ │
│  └────────────────────┘ │
└─────────────────────────┘
```

## 8. Key Takeaways

1. **RadixAttention** is an elegant solution to KV-cache reuse through a radix tree with LRU eviction
2. **Compressed FSM** is a simple but effective optimization for constrained decoding
3. The **co-design of frontend and runtime** is crucial: the frontend provides hints such as prefix boundaries at `fork`, and the runtime uses them for scheduling
4. **Practical impact**: up to **6.4x speedup** on real workloads with minimal overhead
5. SGLang shows that LLM inference can be treated as a **systems problem**, where knowledge of program structure critically improves performance
