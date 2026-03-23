#!/usr/bin/env python3
"""
SGLang DSL Demo via Ollama backend.

Prerequisites:
    ollama serve && ollama pull qwen2.5:0.5b
    pip install sglang openai tiktoken

Usage:
    python sglang_demo.py
"""

import time
import json
import sglang as sgl
import openai

MODEL = "qwen2.5:0.5b"
BASE_URL = "http://localhost:11434/v1"
API_KEY = "none"

backend = sgl.OpenAI(MODEL, base_url=BASE_URL, api_key=API_KEY)
sgl.set_default_backend(backend)

# Direct client for operations unsupported by SGLang's OpenAI chat backend
client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# --- 1. gen() -----------------------------------------------------------------

@sgl.function
def simple_gen(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=128, temperature=0.7))


def demo_gen():
    separator("1. gen() — basic generation")

    questions = [
        "What is KV-cache in LLM inference? Explain in 2 sentences.",
        "What is the difference between prefill and decode phases?",
    ]

    for q in questions:
        state = simple_gen.run(question=q)
        print(f"Q: {q}")
        print(f"A: {state['answer'].strip()}\n")


# --- 2. gen() + extend — multi-step chaining ---------------------------------

@sgl.function
def qa_chain(s):
    """Each gen() sees the full conversation history via prompt state"""
    s += sgl.user("What is the capital of France? Answer in one word.")
    s += sgl.assistant(sgl.gen("capital", max_tokens=10, temperature=0.0))

    s += sgl.user("What is a famous landmark there? Answer in one word.")
    s += sgl.assistant(sgl.gen("landmark", max_tokens=15, temperature=0.0))

    s += sgl.user("Describe it in one sentence.")
    s += sgl.assistant(sgl.gen("description", max_tokens=80, temperature=0.3))


def demo_chain():
    separator("2. gen() + extend — multi-step chaining")

    state = qa_chain.run()
    print(f"  capital:     {state['capital'].strip()}")
    print(f"  landmark:    {state['landmark'].strip()}")
    print(f"  description: {state['description'].strip()}")
    print(f"\n  Note: SGLang Runtime would reuse KV-cache between steps (RadixAttention).")
    print(f"  Via OpenAI API the full context is recomputed each call.")


# --- 3. select() — log-probability choice ------------------------------------

def manual_select(prompt: str, options: list[str]) -> tuple[str, dict]:
    """Approximation of sgl.select() — native select needs logprobs (unavailable in chat API)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt + " Choose exactly one: " + "/".join(options)},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    answer = response.choices[0].message.content.strip().lower()

    for opt in options:
        if opt.lower() in answer:
            return opt, {"raw": answer}
    return options[0], {"raw": answer, "note": "fallback"}


def demo_select():
    separator("3. select() — choice by probability")

    reviews = [
        ("This GPU is amazing, best performance ever!", "positive"),
        ("Terrible, overheats and crashes constantly.", "negative"),
        ("Works fine for the price. Nothing special.", "neutral"),
    ]
    options = ["positive", "negative", "neutral"]

    print("  Sentiment classification:\n")
    for review, expected in reviews:
        result, _ = manual_select(f"Review: \"{review}\"\nSentiment:", options)
        match = "ok" if result == expected else f"expected {expected}"
        print(f"  \"{review[:50]}\"")
        print(f"    -> {result} ({match})\n")

    print("  Note: SGLang Runtime computes exact log-probs per option in one forward pass.")


# --- 4. fork() — parallel generation -----------------------------------------

def demo_fork():
    separator("4. fork() — parallel generation")

    question = "Name one interesting fact about GPUs."
    print(f"  Q: {question}")
    print(f"  Generating 3 independent answers...\n")

    # SGLang fork() shares KV-cache of the prefix across branches.
    # Here we emulate with sequential API calls.
    t0 = time.time()
    answers = []
    for _ in range(3):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
            temperature=0.9,
            max_tokens=80,
        )
        answers.append(resp.choices[0].message.content.strip())
    elapsed = time.time() - t0

    for i, ans in enumerate(answers):
        print(f"  Fork {i}: {ans[:120]}\n")

    print(f"  Time: {elapsed:.2f}s (sequential)")
    print(f"  SGLang Runtime: fork(3) runs branches in parallel with shared prefix KV-cache.")


# --- 5. Shared prefix — RadixAttention concept --------------------------------

@sgl.function
def expert_qa(s, system_prompt, question):
    s += sgl.system(system_prompt)
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=80, temperature=0.3))


def demo_shared_prefix():
    separator("5. Shared prefix — RadixAttention concept")

    system = (
        "You are an expert in LLM inference optimization. "
        "Answer concisely in 1-2 sentences."
    )
    questions = [
        "What is the decode phase bottleneck?",
        "How does FlashAttention help?",
        "What does PagedAttention solve?",
        "Why is continuous batching better than static batching?",
    ]

    t0 = time.time()
    results = []
    for q in questions:
        state = expert_qa.run(system_prompt=system, question=q)
        results.append((q, state['answer'].strip()))
    elapsed = time.time() - t0

    for q, a in results:
        print(f"  Q: {q}")
        print(f"  A: {a[:150]}\n")

    n = len(questions)
    print(f"  Time: {elapsed:.2f}s for {n} queries")
    print(f"\n  RadixAttention savings:")
    print(f"    Without: system prompt prefilled {n} times")
    print(f"    With:    prefilled once, KV-cache reused {n-1} times")


# --- 6. Structured output — JSON + Compressed FSM concept ---------------------

@sgl.function
def json_gen(s):
    s += sgl.user(
        "Output ONLY a valid JSON object with exactly these fields:\n"
        '{"name": "<GPU name>", "vram_gb": <integer>, "tflops_fp16": <integer>}\n'
        "JSON:"
    )
    s += sgl.assistant(sgl.gen("json_output", max_tokens=60, temperature=0.0))


def demo_structured():
    separator("6. Structured output — JSON (Compressed FSM concept)")

    state = json_gen.run()
    raw = state['json_output'].strip()

    start = raw.find("{")
    end = raw.rfind("}") + 1
    json_str = raw[start:end] if start >= 0 and end > start else raw

    print(f"  Raw:    {raw}")
    print(f"  Parsed: ", end="")
    try:
        parsed = json.loads(json_str)
        print(json.dumps(parsed, indent=4))
    except json.JSONDecodeError as e:
        print(f"error: {e} (small model may produce imperfect JSON)")

    print(f"\n  Compressed FSM (SGLang optimization):")
    print(f"    - Builds FSM from regex/JSON schema")
    print(f"    - Masks invalid tokens at each decode step")
    print(f'    - Compresses deterministic chains (e.g. {{"name": ") into one forward pass')
    print(f"    - Result: guaranteed valid JSON + 1.6x speedup")


# --- Main ---------------------------------------------------------------------

def main():
    print("SGLang DSL Demo")
    print(f"Backend: Ollama ({MODEL}) at {BASE_URL}")
    print(f"SGLang version: {sgl.__version__}")

    try:
        client.models.list()
    except Exception:
        print(f"\nERROR: Cannot connect to Ollama at {BASE_URL}")
        print(f"  ollama serve && ollama pull {MODEL}")
        raise SystemExit(1)

    t_total = time.time()

    demo_gen()
    demo_chain()
    demo_select()
    demo_fork()
    demo_shared_prefix()
    demo_structured()

    separator("Summary")
    elapsed = time.time() - t_total
    print(f"  Total time: {elapsed:.1f}s\n")
    print("  | SGLang DSL     | Demonstrated              | OpenAI backend limitation     |")
    print("  |----------------|---------------------------|-------------------------------|")
    print("  | gen()          | Basic generation           | None                          |")
    print("  | gen() + extend | Multi-step chaining        | No KV-cache reuse             |")
    print("  | select()       | Probability-based choice   | No logprobs, approximation    |")
    print("  | fork()         | Parallel generation        | Sequential API calls          |")
    print("  | Shared prefix  | RadixAttention concept     | Prefix recomputed each time   |")
    print("  | Structured out | JSON + Compressed FSM      | No regex constraints          |")
    print()
    print("  SGLang Runtime (sm80+ GPU) removes all right-column limitations:")
    print("  RadixAttention, native select(), parallel fork(), Compressed FSM.")


if __name__ == "__main__":
    main()
