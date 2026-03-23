#!/usr/bin/env python3
"""
LLM Inference Server Simulator — entry point.

Loads Azure LLM Inference Trace, finds minimum N accelerators satisfying SLA
via exponential + binary search (O(log N) runs instead of O(N)).

Usage:
    python main.py                          # full dataset, search for min N
    python main.py --max-requests 1000      # quick test
    python main.py --N 50                   # fixed N, no search
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

from simulator import SimConfig, RequestInput, Simulator


def load_trace(filepath: str, max_requests: int = 0) -> list[RequestInput]:
    """Load CSV trace, normalize timestamps to ms from 0."""
    requests = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_requests > 0 and i >= max_requests:
                break
            ts_str = row['TIMESTAMP']
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            arrival_ms = ts.timestamp() * 1000.0
            requests.append(RequestInput(
                id=i,
                arrival_time=arrival_ms,
                num_images=int(row['NumImages']),
                context_tokens=int(row['ContextTokens']),
                generated_tokens=int(row['GeneratedTokens']),
            ))

    if not requests:
        print("ERROR: No requests loaded from trace file.")
        sys.exit(1)

    base = requests[0].arrival_time
    for r in requests:
        r.arrival_time -= base

    span_sec = requests[-1].arrival_time / 1000.0
    print(f"Loaded {len(requests)} requests")
    print(f"Time span: {span_sec:.1f}s ({span_sec/3600:.1f}h)")

    imgs = [r.num_images for r in requests]
    ctxs = [r.context_tokens for r in requests]
    gens = [r.generated_tokens for r in requests]
    print(f"\nDataset summary:")
    print(f"  Images:    min={min(imgs)}, max={max(imgs)}, mean={sum(imgs)/len(imgs):.1f}")
    print(f"  Context:   min={min(ctxs)}, max={max(ctxs)}, mean={sum(ctxs)/len(ctxs):.1f}")
    print(f"  Generated: min={min(gens)}, max={max(gens)}, mean={sum(gens)/len(gens):.1f}")
    print(f"  Requests with images: {sum(1 for x in imgs if x > 0)} ({100*sum(1 for x in imgs if x > 0)/len(imgs):.1f}%)")

    return requests


def run_simulation(cfg: SimConfig, inputs: list[RequestInput]) -> tuple[bool, dict]:
    sim = Simulator(cfg, inputs)
    ok = sim.run()
    stats = sim.get_stats() if ok else {}
    return ok, stats


def find_min_n(cfg: SimConfig, inputs: list[RequestInput]) -> tuple[int, dict]:
    """Two-phase search: exponential upper bound, then binary search.
    SLA compliance is monotonic in N, so binary search is correct.
    """
    print("\n--- Phase 1: Exponential search for upper bound ---")
    hi = 1
    while hi <= 200_000:
        cfg.N = hi
        t0 = time.time()
        ok, stats = run_simulation(cfg, inputs)
        elapsed = time.time() - t0
        status = "OK" if ok else "SLA VIOLATED"
        print(f"  N={hi:>6d}: {status} ({elapsed:.1f}s)")
        if ok:
            break
        hi *= 2

    if hi > 200_000:
        print("ERROR: Cannot satisfy SLA even with 200K accelerators.")
        sys.exit(1)

    if hi == 1:
        cfg.N = 1
        _, stats = run_simulation(cfg, inputs)
        return 1, stats

    lo = max(1, hi // 2)
    print(f"\n--- Phase 2: Binary search [{lo}, {hi}] ---")
    best_stats = stats

    while lo < hi:
        mid = (lo + hi) // 2
        cfg.N = mid
        t0 = time.time()
        ok, stats = run_simulation(cfg, inputs)
        elapsed = time.time() - t0
        status = "OK" if ok else "FAIL"
        print(f"  N={mid:>6d}: {status} ({elapsed:.1f}s)")
        if ok:
            hi = mid
            best_stats = stats
        else:
            lo = mid + 1

    cfg.N = lo
    ok, stats = run_simulation(cfg, inputs)
    if ok:
        best_stats = stats

    return lo, best_stats


def print_stats(n: int, stats: dict, cfg: SimConfig):
    print("\n" + "=" * 70)
    print("  SIMULATION RESULTS")
    print("=" * 70)

    print(f"\n  Minimal N (accelerators): {n}")

    print(f"\n  Parameters:")
    print(f"    Compute costs:  A={cfg.A}ms/img  B={cfg.B}ms/ctx_tok  C={cfg.C}ms/gen_tok")
    print(f"    Memory costs:   X={cfg.X}MB/img  Y={cfg.Y}MB/ctx_tok  Z={cfg.Z}MB/gen_tok")
    print(f"    Accelerator:    M={cfg.M}MB  max_batch={cfg.max_batch_size}")
    print(f"    SLA:            TTFT<={cfg.P}ms  per_token<={cfg.D}ms")
    print(f"    Batch function: {cfg.batch_func}")

    if not stats:
        print("\n  No statistics available (SLA violated).")
        return

    print(f"\n  Total requests processed: {stats['total_requests']}")

    ttft = stats['ttft']
    print(f"\n  TTFT (Time To First Token) [ms]:")
    print(f"    Min:     {ttft['min']:>10.2f}")
    print(f"    Median:  {ttft['median']:>10.2f}")
    print(f"    Mean:    {ttft['mean']:>10.2f}")
    print(f"    p95:     {ttft['p95']:>10.2f}")
    print(f"    p99:     {ttft['p99']:>10.2f}")
    print(f"    Max:     {ttft['max']:>10.2f}")

    tpt = stats['time_per_token']
    print(f"\n  Time per generated token [ms]:")
    print(f"    Min:     {tpt['min']:>10.2f}")
    print(f"    Median:  {tpt['median']:>10.2f}")
    print(f"    Mean:    {tpt['mean']:>10.2f}")
    print(f"    p95:     {tpt['p95']:>10.2f}")
    print(f"    p99:     {tpt['p99']:>10.2f}")
    print(f"    Max:     {tpt['max']:>10.2f}")

    util = stats['utilization']
    print(f"\n  Accelerator utilization:")
    print(f"    Mean:    {util['mean']*100:>9.1f}%")
    print(f"    Min:     {util['min']*100:>9.1f}%")
    print(f"    Max:     {util['max']*100:>9.1f}%")

    if tpt['max'] > cfg.D * 0.9:
        print(f"\n  Warning: per-token time near SLA limit ({tpt['max']:.1f}/{cfg.D}ms)")
    if ttft['max'] > cfg.P * 0.9:
        print(f"\n  Warning: TTFT near SLA limit ({ttft['max']:.1f}/{cfg.P}ms)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Inference Server Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--trace', default='AzureLMMInferenceTrace_multimodal.csv',
                        help='Path to trace CSV file')
    parser.add_argument('--max-requests', type=int, default=0,
                        help='Limit number of requests (0 = all)')
    parser.add_argument('--N', type=int, default=0,
                        help='Fixed N (0 = search for minimum)')

    parser.add_argument('--M', type=float, default=80_000.0, help='Memory per accelerator (MB)')
    parser.add_argument('--K', type=float, default=1000.0, help='Compute capability (tokens/sec)')

    parser.add_argument('--A', type=float, default=10.0, help='Image preprocessing cost (ms/image)')
    parser.add_argument('--B', type=float, default=0.05, help='Context token cost (ms/token)')
    parser.add_argument('--C', type=float, default=20.0, help='Generated token cost (ms/token)')

    parser.add_argument('--X', type=float, default=50.0, help='Memory per image (MB)')
    parser.add_argument('--Y', type=float, default=0.002, help='Memory per context token (MB)')
    parser.add_argument('--Z', type=float, default=0.002, help='Memory per generated token (MB)')

    parser.add_argument('--P', type=float, default=5000.0, help='Max TTFT (ms)')
    parser.add_argument('--D', type=float, default=100.0, help='Max time per generated token (ms)')

    parser.add_argument('--max-batch-size', type=int, default=32, help='Max batch size')
    parser.add_argument('--batch-func', choices=['sqrt', 'cbrt'], default='sqrt',
                        help='Batching speedup function')

    args = parser.parse_args()

    trace_path = args.trace
    if not os.path.isabs(trace_path):
        trace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), trace_path)

    if not os.path.exists(trace_path):
        print(f"ERROR: Trace file not found: {trace_path}")
        sys.exit(1)

    cfg = SimConfig(
        N=args.N if args.N > 0 else 1,
        M=args.M, K=args.K,
        A=args.A, B=args.B, C=args.C,
        X=args.X, Y=args.Y, Z=args.Z,
        P=args.P, D=args.D,
        max_batch_size=args.max_batch_size,
        batch_func=args.batch_func,
    )

    print(f"Loading trace from: {trace_path}")
    inputs = load_trace(trace_path, args.max_requests)

    if args.N > 0:
        print(f"\nRunning simulation with N={args.N}...")
        t0 = time.time()
        ok, stats = run_simulation(cfg, inputs)
        elapsed = time.time() - t0
        print(f"Simulation completed in {elapsed:.1f}s")
        if ok:
            print_stats(args.N, stats, cfg)
        else:
            print(f"\nSLA VIOLATED with N={args.N}. Try increasing N.")
    else:
        print("\nSearching for minimal N...")
        t0 = time.time()
        min_n, stats = find_min_n(cfg, inputs)
        elapsed = time.time() - t0
        print(f"\nTotal search time: {elapsed:.1f}s")
        print_stats(min_n, stats, cfg)


if __name__ == '__main__':
    main()
