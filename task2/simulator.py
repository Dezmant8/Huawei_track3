"""
LLM Inference Server Simulator — discrete event simulation engine.

Processes multimodal LLM requests through a 3-stage pipeline on N accelerators
with memory constraints and SLA limits. Uses heapq-based event queue for
O(log n) push/pop on ~4M events.
"""

import heapq
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
from collections import deque


@dataclass
class SimConfig:
    """Simulation parameters from the task specification."""
    N: int = 1
    M: float = 80_000.0       # accelerator memory, MB

    # Compute costs (ms per unit, batch_size=1)
    A: float = 10.0            # per image
    B: float = 0.05            # per context token
    C: float = 20.0            # per generated token

    # Memory costs (MB per unit)
    X: float = 50.0            # per image (freed after stage 1)
    Y: float = 0.002           # per context token (KV-cache, persists through stage 3)
    Z: float = 0.002           # per generated token

    # SLA
    P: float = 5000.0          # max TTFT, ms
    D: float = 100.0           # max time per generated token, ms

    K: float = 1000.0
    max_batch_size: int = 32
    batch_func: str = 'sqrt'

    def batch_cost(self, base_cost_per_unit: float, total_units: int) -> float:
        """Sublinear batch cost: Cost(B) = A * sqrt(B). Models GPU parallelism."""
        if total_units <= 0:
            return 0.0
        if self.batch_func == 'cbrt':
            return base_cost_per_unit * (total_units ** (1.0 / 3.0))
        return base_cost_per_unit * math.sqrt(total_units)


# --- Request model ------------------------------------------------------------

class RequestInput:
    """Immutable request data from trace. Reused across binary search runs."""
    __slots__ = ('id', 'arrival_time', 'num_images', 'context_tokens', 'generated_tokens')

    def __init__(self, id: int, arrival_time: float, num_images: int,
                 context_tokens: int, generated_tokens: int):
        self.id = id
        self.arrival_time = arrival_time
        self.num_images = num_images
        self.context_tokens = context_tokens
        self.generated_tokens = generated_tokens


class RequestState:
    """Mutable per-run state. Separated from RequestInput to avoid deepcopy on each run."""
    __slots__ = ('inp', 'stage1_end', 'stage2_end', 'stage3_start', 'stage3_end',
                 'accel_id', 'images_remaining')

    def __init__(self, inp: RequestInput):
        self.inp = inp
        self.stage1_end: float = 0.0
        self.stage2_end: float = 0.0
        self.stage3_start: float = 0.0
        self.stage3_end: float = 0.0
        self.accel_id: int = -1
        self.images_remaining: int = inp.num_images

    @property
    def ttft(self) -> float:
        return self.stage2_end - self.inp.arrival_time

    @property
    def time_per_token(self) -> float:
        if self.inp.generated_tokens == 0:
            return 0.0
        return (self.stage3_end - self.stage3_start) / self.inp.generated_tokens

    def mem_stage1(self, cfg: SimConfig) -> float:
        return self.inp.num_images * cfg.X

    def mem_stage2(self, cfg: SimConfig) -> float:
        return self.inp.context_tokens * cfg.Y

    def mem_stage3_extra(self, cfg: SimConfig) -> float:
        return self.inp.generated_tokens * cfg.Z

    def mem_total_during_generation(self, cfg: SimConfig) -> float:
        return self.inp.context_tokens * cfg.Y + self.inp.generated_tokens * cfg.Z


# --- Event system -------------------------------------------------------------

class EventType(IntEnum):
    REQUEST_ARRIVAL = 0
    STAGE_COMPLETE = 1


@dataclass(order=True)
class Event:
    """Heap-ordered by time only; payload fields excluded from comparison."""
    time: float
    etype: EventType = field(compare=False)
    accel_id: int = field(compare=False, default=-1)
    stage: int = field(compare=False, default=0)
    batch: list = field(compare=False, default_factory=list)
    mem_allocated: float = field(compare=False, default=0.0)
    request: Optional[RequestState] = field(compare=False, default=None)


# --- Accelerator model --------------------------------------------------------

class Accelerator:
    __slots__ = ('id', 'memory_used', 'memory_capacity', 'busy_until',
                 'stage1_queue', 'stage2_queue', 'stage3_queue',
                 'total_busy_time', '_last_busy_start')

    def __init__(self, id: int, memory_capacity: float):
        self.id = id
        self.memory_used: float = 0.0
        self.memory_capacity = memory_capacity
        self.busy_until: float = 0.0
        self.stage1_queue: deque[RequestState] = deque()
        self.stage2_queue: deque[RequestState] = deque()
        self.stage3_queue: deque[RequestState] = deque()
        self.total_busy_time: float = 0.0
        self._last_busy_start: float = 0.0

    @property
    def free_memory(self) -> float:
        return self.memory_capacity - self.memory_used

    @property
    def queue_len(self) -> int:
        return len(self.stage1_queue) + len(self.stage2_queue) + len(self.stage3_queue)

    def estimated_memory_need(self, cfg: SimConfig) -> float:
        total = self.memory_used
        for rs in self.stage1_queue:
            total += rs.images_remaining * cfg.X
        for rs in self.stage2_queue:
            total += rs.inp.context_tokens * cfg.Y
        for rs in self.stage3_queue:
            total += rs.mem_stage3_extra(cfg)
        return total


# --- Simulator core -----------------------------------------------------------

class Simulator:
    """Event-driven LLM inference simulator.

    Scheduling:
        - Request assignment: memory-aware least-loaded (by busy_until, free memory, queue length)
        - Stage priority: stage3 > stage2 > stage1
          (stage3 frees KV-cache memory; stage2 determines TTFT; stage1 can wait)
        - Within stage: FIFO order equals EDF when all requests share the same SLA deadline P
    """

    def __init__(self, cfg: SimConfig, inputs: list[RequestInput]):
        self.cfg = cfg
        self.inputs = inputs
        self.events: list[Event] = []
        self.accels: list[Accelerator] = []
        self.completed: list[RequestState] = []
        self.sla_violated = False
        self._current_time = 0.0

    def run(self) -> bool:
        """Run simulation. Returns True if all SLA constraints are met.
        Early exit on first SLA violation (optimization for binary search).
        """
        cfg = self.cfg

        self.accels = [Accelerator(i, cfg.M) for i in range(cfg.N)]
        self.events = []
        self.completed = []
        self.sla_violated = False
        self._current_time = 0.0

        for inp in self.inputs:
            rs = RequestState(inp)
            heapq.heappush(self.events, Event(
                time=inp.arrival_time,
                etype=EventType.REQUEST_ARRIVAL,
                request=rs,
            ))

        while self.events:
            event = heapq.heappop(self.events)
            self._current_time = event.time

            if event.etype == EventType.REQUEST_ARRIVAL:
                self._handle_arrival(event.request)
            else:
                self._handle_stage_complete(event)

            if self.sla_violated:
                return False

        # Check for stuck requests (e.g. memory too small to fit any request)
        for accel in self.accels:
            if accel.stage1_queue or accel.stage2_queue or accel.stage3_queue:
                self.sla_violated = True
                return False

        return True

    def _handle_arrival(self, rs: RequestState):
        """Assign request to least-loaded accelerator with most free memory."""
        best = min(self.accels, key=lambda a: (a.busy_until, -a.free_memory, a.queue_len))
        rs.accel_id = best.id

        if rs.inp.num_images == 0:
            best.stage2_queue.append(rs)
        else:
            best.stage1_queue.append(rs)

        self._try_schedule(best)

    def _try_schedule(self, accel: Accelerator):
        """Schedule next batch if accelerator is idle. Priority: stage3 > stage2 > stage1."""
        if accel.busy_until > self._current_time:
            return

        if accel.stage3_queue:
            self._start_stage3(accel)
        elif accel.stage2_queue:
            self._start_stage2(accel)
        elif accel.stage1_queue:
            self._start_stage1(accel)

    def _form_batch(self, queue: deque, mem_fn, accel: Accelerator) -> list[RequestState]:
        """Greedily form a batch from queue respecting memory and max_batch_size."""
        batch = []
        skipped = []
        mem_total = 0.0

        while queue and len(batch) < self.cfg.max_batch_size:
            rs = queue.popleft()
            mem = mem_fn(rs)
            if mem_total + mem <= accel.free_memory:
                batch.append(rs)
                mem_total += mem
            else:
                skipped.append(rs)

        for rs in reversed(skipped):
            queue.appendleft(rs)

        return batch

    # --- Stage 1: image preprocessing -----------------------------------------

    def _start_stage1(self, accel: Accelerator):
        """Process images. Sub-batches large requests that exceed accelerator memory."""
        cfg = self.cfg

        max_images_fit = max(1, int(accel.free_memory / cfg.X)) if cfg.X > 0 else 10**9

        batch = []
        skipped = []
        total_images = 0
        mem_total = 0.0

        while accel.stage1_queue and len(batch) < cfg.max_batch_size:
            rs = accel.stage1_queue.popleft()
            imgs = min(rs.images_remaining, max_images_fit - total_images)
            if imgs <= 0:
                skipped.append(rs)
                continue
            img_mem = imgs * cfg.X
            if mem_total + img_mem > accel.free_memory:
                skipped.append(rs)
                continue
            batch.append((rs, imgs))
            total_images += imgs
            mem_total += img_mem

        for rs in reversed(skipped):
            accel.stage1_queue.appendleft(rs)

        if not batch:
            return

        batch_time = cfg.batch_cost(cfg.A, total_images)
        accel.memory_used += mem_total
        end_time = self._current_time + batch_time
        accel.busy_until = end_time
        accel.total_busy_time += batch_time

        done_batch = []
        for rs, imgs_processed in batch:
            rs.images_remaining -= imgs_processed
            done_batch.append(rs)
            rs.stage1_end = end_time

        heapq.heappush(self.events, Event(
            time=end_time,
            etype=EventType.STAGE_COMPLETE,
            accel_id=accel.id,
            stage=1,
            batch=done_batch,
            mem_allocated=mem_total,
        ))

    # --- Stage 2: context prefill (compute-bound) -----------------------------

    def _start_stage2(self, accel: Accelerator):
        """Prefill context tokens. KV-cache memory persists into stage 3."""
        cfg = self.cfg
        batch = self._form_batch(
            accel.stage2_queue,
            lambda rs: rs.inp.context_tokens * cfg.Y,
            accel
        )
        if not batch:
            return

        total_ctx = sum(rs.inp.context_tokens for rs in batch)
        batch_time = cfg.batch_cost(cfg.B, total_ctx)

        mem = sum(rs.inp.context_tokens * cfg.Y for rs in batch)
        accel.memory_used += mem
        end_time = self._current_time + batch_time
        accel.busy_until = end_time
        accel.total_busy_time += batch_time

        for rs in batch:
            rs.stage2_end = end_time

        heapq.heappush(self.events, Event(
            time=end_time,
            etype=EventType.STAGE_COMPLETE,
            accel_id=accel.id,
            stage=2,
            batch=batch,
            mem_allocated=mem,
        ))

    # --- Stage 3: token generation (memory-bound, shrinking batch) ------------

    def _start_stage3(self, accel: Accelerator):
        """Generate tokens using shrinking batch: as shorter requests finish,
        batch size decreases, adjusting cost per step accordingly.

        Example: requests with 100, 300, 500 tokens ->
          chunk 1: 100 steps at batch_size=3
          chunk 2: 200 steps at batch_size=2
          chunk 3: 200 steps at batch_size=1
        Each request gets its individual stage3_end timestamp.
        """
        cfg = self.cfg

        batch = self._form_batch(
            accel.stage3_queue,
            lambda rs: rs.mem_stage3_extra(cfg),
            accel
        )
        if not batch:
            return

        sorted_batch = sorted(batch, key=lambda rs: rs.inp.generated_tokens)

        total_time = 0.0
        prev_tokens = 0
        start_time = self._current_time

        for i, rs in enumerate(sorted_batch):
            remaining = len(sorted_batch) - i
            tokens_chunk = rs.inp.generated_tokens - prev_tokens
            if tokens_chunk > 0:
                chunk_cost = tokens_chunk * cfg.batch_cost(cfg.C, remaining)
                total_time += chunk_cost
            prev_tokens = rs.inp.generated_tokens

            rs.stage3_start = start_time
            rs.stage3_end = start_time + total_time

        mem = sum(rs.mem_stage3_extra(cfg) for rs in batch)
        accel.memory_used += mem

        end_time = start_time + total_time
        accel.busy_until = end_time
        accel.total_busy_time += total_time

        heapq.heappush(self.events, Event(
            time=end_time,
            etype=EventType.STAGE_COMPLETE,
            accel_id=accel.id,
            stage=3,
            batch=batch,
            mem_allocated=mem,
        ))

    # --- Event handling -------------------------------------------------------

    def _handle_stage_complete(self, event: Event):
        accel = self.accels[event.accel_id]
        cfg = self.cfg
        stage = event.stage
        batch = event.batch

        if stage == 1:
            # Free image memory; re-queue requests with remaining images
            accel.memory_used -= event.mem_allocated
            for rs in batch:
                if rs.images_remaining > 0:
                    accel.stage1_queue.appendleft(rs)
                else:
                    accel.stage2_queue.append(rs)

        elif stage == 2:
            for rs in batch:
                if rs.ttft > cfg.P:
                    self.sla_violated = True
                    return

            for rs in batch:
                if rs.inp.generated_tokens == 0:
                    rs.stage3_start = self._current_time
                    rs.stage3_end = self._current_time
                    accel.memory_used -= rs.inp.context_tokens * cfg.Y
                    self.completed.append(rs)
                else:
                    accel.stage3_queue.append(rs)

        elif stage == 3:
            # Free generated token memory + KV-cache from stage 2
            accel.memory_used -= event.mem_allocated
            for rs in batch:
                accel.memory_used -= rs.inp.context_tokens * cfg.Y
                if rs.time_per_token > cfg.D:
                    self.sla_violated = True
                    return
                self.completed.append(rs)

        self._try_schedule(accel)

    # --- Stats ----------------------------------------------------------------

    def get_stats(self) -> dict:
        if not self.completed:
            return {}

        ttfts = [rs.ttft for rs in self.completed]
        tpts = [rs.time_per_token for rs in self.completed
                if rs.inp.generated_tokens > 0]

        def _stats(values):
            if not values:
                return {'median': 0, 'mean': 0, 'min': 0, 'max': 0,
                        'p95': 0, 'p99': 0}
            sv = sorted(values)
            n = len(sv)
            return {
                'median': sv[n // 2],
                'mean': sum(sv) / n,
                'min': sv[0],
                'max': sv[-1],
                'p95': sv[int(n * 0.95)],
                'p99': sv[int(n * 0.99)],
            }

        if self._current_time > 0:
            utilizations = [a.total_busy_time / self._current_time for a in self.accels]
        else:
            utilizations = [0.0] * len(self.accels)

        return {
            'ttft': _stats(ttfts),
            'time_per_token': _stats(tpts),
            'total_requests': len(self.completed),
            'utilization': {
                'mean': sum(utilizations) / len(utilizations) if utilizations else 0,
                'min': min(utilizations) if utilizations else 0,
                'max': max(utilizations) if utilizations else 0,
            },
        }
