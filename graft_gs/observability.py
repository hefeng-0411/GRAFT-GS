"""Low-overhead, rank-aware progress and profiling instrumentation.

The records emitted here are a control-plane contract.  They never call
``torch.cuda.synchronize`` and a heartbeat never claims semantic advancement.
An external supervisor can therefore distinguish a long legitimate CUDA stage
from a process that has stopped advancing without perturbing the training
stream it observes.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Iterator, Mapping, Optional

import torch
from torch.profiler import record_function


PROGRESS_PREFIX = "GRAFT_GS_PROGRESS "


@dataclass(frozen=True)
class ProgressConfig:
    enabled: bool = True
    heartbeat_interval_seconds: float = 30.0
    include_cuda_memory: bool = True
    nvtx: bool = False
    profiler_ranges: bool = True
    cuda_event_timing: bool = False

    def __post_init__(self) -> None:
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("progress heartbeat interval must be positive")
        for name in (
            "enabled",
            "include_cuda_memory",
            "nvtx",
            "profiler_ranges",
            "cuda_event_timing",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"ProgressConfig.{name} must be Boolean")


@dataclass(frozen=True)
class TrainingProfilerConfig:
    """Bounded first-step PyTorch profiler policy."""

    enabled: bool = False
    nvtx: bool = False
    torch_profiler: bool = True
    first_n_steps: int = 5
    wait_steps: int = 1
    warmup_steps: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "nvtx",
            "torch_profiler",
            "record_shapes",
            "profile_memory",
            "with_stack",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"TrainingProfilerConfig.{name} must be Boolean")
        if self.first_n_steps < 1:
            raise ValueError("profiler first_n_steps must be positive")
        if self.wait_steps < 0 or self.warmup_steps < 0:
            raise ValueError("profiler wait/warmup steps must be non-negative")
        if self.first_n_steps <= self.wait_steps + self.warmup_steps:
            raise ValueError(
                "profiler first_n_steps must include at least one active step"
            )

    @property
    def active_steps(self) -> int:
        return self.first_n_steps - self.wait_steps - self.warmup_steps


def _resident_set_bytes() -> int:
    """Return current Linux RSS without importing a monitoring dependency."""

    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return 0


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu" and value.numel() == 1:
            return value.detach().item()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


class ProgressReporter:
    """Emit atomic JSONL progress records and optional nonblocking ranges."""

    def __init__(
        self,
        config: ProgressConfig = ProgressConfig(),
        *,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.config = config
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
        self.local_rank = (
            int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank is None
            else int(local_rank)
        )
        self.world_size = (
            int(os.environ.get("WORLD_SIZE", "1"))
            if world_size is None
            else int(world_size)
        )
        self.device = torch.device(device) if device is not None else None
        self._hostname = socket.gethostname()
        self._pid = os.getpid()
        self._started_ns = time.monotonic_ns()
        self._lock = threading.RLock()
        self._sequence = 0
        self._semantic_sequence = 0
        self._context: dict[str, object] = {}
        self._active_stages: list[tuple[int, str, int, dict[str, object]]] = []
        self._next_stage_token = 0
        self._closed = False
        self._stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._pending_cuda_events: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event, dict[str, object]]
        ] = []
        if self.config.enabled:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"graft-progress-rank-{self.rank}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    @property
    def enabled(self) -> bool:
        return self.config.enabled and not self._closed

    def set_device(self, device: torch.device | str) -> None:
        self.device = torch.device(device)

    def set_context(self, **values: object) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._context = {
                str(name): _json_value(value) for name, value in values.items()
            }

    def update_context(self, **values: object) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._context.update(
                {str(name): _json_value(value) for name, value in values.items()}
            )

    def event(
        self,
        stage: str,
        event: str,
        *,
        semantic: bool = True,
        **values: object,
    ) -> None:
        if not self.enabled:
            return
        self._emit(stage, event, semantic=semantic, values=values)
        if threading.current_thread() is threading.main_thread():
            self._harvest_cuda_timings()

    @contextmanager
    def stage(self, stage: str, **values: object) -> Iterator[None]:
        """Describe one semantic stage without synchronizing its CUDA work."""

        if not self.enabled:
            yield
            return
        started_ns = time.monotonic_ns()
        with self._lock:
            token = self._next_stage_token
            self._next_stage_token += 1
            saved_values = {
                str(name): _json_value(value) for name, value in values.items()
            }
            self._active_stages.append((token, stage, started_ns, saved_values))
        self.event(stage, "begin", **values)
        nvtx_active = bool(
            self.config.nvtx
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        if nvtx_active:
            torch.cuda.nvtx.range_push(stage)
        profile_context = (
            record_function(f"graft_gs_progress/{stage}")
            if self.config.profiler_ranges
            else nullcontext()
        )
        start_event: Optional[torch.cuda.Event] = None
        end_event: Optional[torch.cuda.Event] = None
        if (
            self.config.cuda_event_timing
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        try:
            with profile_context:
                yield
        except StopIteration:
            self.event(
                stage,
                "end",
                iterator_exhausted=True,
                host_elapsed_seconds=(time.monotonic_ns() - started_ns) / 1.0e9,
                **values,
            )
            raise
        except BaseException as error:
            self.event(
                stage,
                "failed",
                exception_type=type(error).__name__,
                exception=str(error)[:2048],
                host_elapsed_seconds=(time.monotonic_ns() - started_ns) / 1.0e9,
                **values,
            )
            raise
        else:
            if end_event is not None and start_event is not None:
                end_event.record()
                with self._lock:
                    self._pending_cuda_events.append(
                        (stage, start_event, end_event, saved_values)
                    )
            self.event(
                stage,
                "end",
                host_elapsed_seconds=(time.monotonic_ns() - started_ns) / 1.0e9,
                **values,
            )
        finally:
            if nvtx_active:
                torch.cuda.nvtx.range_pop()
            with self._lock:
                for index in range(len(self._active_stages) - 1, -1, -1):
                    if self._active_stages[index][0] == token:
                        self._active_stages.pop(index)
                        break

    def close(self) -> None:
        if self._closed:
            return
        if self.config.enabled:
            self.event("process", "reporter_close")
        self._closed = True
        self._stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(self.config.heartbeat_interval_seconds, 1.0))

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.config.heartbeat_interval_seconds):
            if self._closed:
                return
            with self._lock:
                if self._active_stages:
                    _, stage, started_ns, values = self._active_stages[-1]
                    heartbeat_values = {
                        **values,
                        "stage_elapsed_seconds": (
                            time.monotonic_ns() - started_ns
                        )
                        / 1.0e9,
                    }
                else:
                    stage = "process.idle"
                    heartbeat_values = {}
            # A heartbeat is deliberately non-semantic. It proves the Python
            # control thread is alive but cannot keep a hung stage alive.
            self._emit(
                stage,
                "heartbeat",
                semantic=False,
                values=heartbeat_values,
            )

    def _cuda_memory(self) -> dict[str, object]:
        device = self.device
        if (
            not self.config.include_cuda_memory
            or device is None
            or device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            return {}
        # Allocator counters do not wait for queued kernels. In particular, do
        # not replace these with mem_get_info + synchronize in the heartbeat.
        try:
            return {
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "cuda_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
        except (RuntimeError, AssertionError):
            return {"cuda_memory_unavailable": True}

    def _emit(
        self,
        stage: str,
        event: str,
        *,
        semantic: bool,
        values: Mapping[str, object],
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._sequence += 1
            if semantic:
                self._semantic_sequence += 1
            payload: dict[str, object] = {
                "schema": "graft-gs-progress-v1",
                "rank": self.rank,
                "local_rank": self.local_rank,
                "world_size": self.world_size,
                "hostname": self._hostname,
                "pid": self._pid,
                "sequence": self._sequence,
                "semantic_sequence": self._semantic_sequence,
                "semantic_progress": semantic,
                "stage": str(stage),
                "event": str(event),
                "monotonic_ns": time.monotonic_ns(),
                "process_elapsed_seconds": (
                    time.monotonic_ns() - self._started_ns
                )
                / 1.0e9,
                "cpu_rss_bytes": _resident_set_bytes(),
                **self._context,
                **self._cuda_memory(),
                **{
                    str(name): _json_value(value)
                    for name, value in values.items()
                },
            }
            print(
                PROGRESS_PREFIX
                + json.dumps(payload, sort_keys=True, allow_nan=False),
                flush=True,
            )

    def _harvest_cuda_timings(self) -> None:
        if not self.config.cuda_event_timing:
            return
        ready: list[tuple[str, float, dict[str, object]]] = []
        with self._lock:
            pending = self._pending_cuda_events
            self._pending_cuda_events = []
        remaining: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event, dict[str, object]]
        ] = []
        for stage, start_event, end_event, values in pending:
            if end_event.query():
                ready.append(
                    (stage, float(start_event.elapsed_time(end_event)), values)
                )
            else:
                remaining.append((stage, start_event, end_event, values))
        with self._lock:
            self._pending_cuda_events.extend(remaining)
        for stage, elapsed_ms, values in ready:
            self._emit(
                stage,
                "cuda_timing",
                semantic=False,
                values={**values, "cuda_elapsed_ms": elapsed_ms},
            )

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "PROGRESS_PREFIX",
    "ProgressConfig",
    "ProgressReporter",
    "TrainingProfilerConfig",
]
