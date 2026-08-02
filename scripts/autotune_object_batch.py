"""Select a safe, high-throughput object batch on an explicit GPU set.

Every candidate runs in a fresh torchrun process group. An allocation failure,
mutated optimizer, allocator cache, or stochastic state from a probe therefore
cannot leak into the launched training job.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROGRESS_PREFIX = "GRAFT_GS_PROGRESS "
OOM_MARKERS = (
    "torch.outofmemoryerror",
    "cuda out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "failed to allocate cuda",
    "cuda malloc failed",
)
_ACCUMULATION_OPTIONS = (
    "--gradient-accumulation-steps",
    "--global-object-batch",
    "--minimum-global-object-batch",
)

NONFINITE_MARKERS = (
    "graft_gs_nonfinite",
    "non-finite training tensors",
    "floatingpointerror",
)
COLLECTIVE_FAILURE_MARKERS = (
    "processgroupnccl",
    "nccl error",
    "collective operation timeout",
    "distbackenderror",
    "ddp atlas key alignment failed",
)
DATA_FAILURE_MARKERS = (
    "filenotfounderror",
    "dataset contract",
    "manifest",
    "no vggt evidence lies inside",
    "dataloader worker",
)
MODEL_FAILURE_MARKERS = (
    "checkpoint",
    "state_dict",
    "upstream contract",
    "requires declared",
    "modulenotfounderror",
    "importerror",
)

DEFAULT_STAGE_TIMEOUT_SECONDS = {
    "model_load": 1800.0,
    "data.fetch_batch": 900.0,
    "forward.mesh_supervision_targets": 1200.0,
    "forward.trellis.device_load": 1200.0,
    "forward.trellis.conditioning": 900.0,
    "forward.trellis.posterior_draw": 600.0,
    "forward.vggt_geometry": 1200.0,
    "forward.evidence_lift": 900.0,
    "forward.atlas": 1200.0,
    "forward.sparse_uot": 1800.0,
    "forward.gauge_sparse_attention": 1200.0,
    "forward.topology_stratum": 1800.0,
    "forward.barrier_riemannian_flow": 1800.0,
    "forward.analytical_readout": 900.0,
    "forward.gaussian_render": 1800.0,
    "train.loss": 1200.0,
    "train.backward": 3600.0,
    "train.optimizer_step": 1200.0,
    "collective": 1800.0,
    "checkpoint": 1800.0,
}


@dataclass
class _RankProgress:
    rank: int
    pid: int | None = None
    stage: str = "unreported"
    event: str = "unreported"
    sequence: int = 0
    semantic_sequence: int = 0
    last_record_monotonic: float = 0.0
    last_semantic_monotonic: float = 0.0
    context: dict[str, object] = field(default_factory=dict)


class SemanticProgressMonitor:
    """Track meaningful child advancement independently of total wall time."""

    def __init__(
        self,
        *,
        started: float,
        bootstrap_timeout_seconds: float,
        no_progress_timeout_seconds: float,
        world_size: int | None = None,
        stage_timeout_seconds: dict[str, float] | None = None,
        stage_latency_mad_multiplier: float = 6.0,
    ) -> None:
        if bootstrap_timeout_seconds <= 0 or no_progress_timeout_seconds <= 0:
            raise ValueError("semantic watchdog timeouts must be positive")
        self.started = started
        self.bootstrap_timeout_seconds = bootstrap_timeout_seconds
        self.no_progress_timeout_seconds = no_progress_timeout_seconds
        self.world_size = world_size
        self.stage_timeout_seconds = {
            **DEFAULT_STAGE_TIMEOUT_SECONDS,
            **(stage_timeout_seconds or {}),
        }
        if stage_latency_mad_multiplier < 0:
            raise ValueError("stage latency MAD multiplier must be non-negative")
        self.stage_latency_mad_multiplier = stage_latency_mad_multiplier
        self.last_semantic_monotonic: float | None = None
        self.ranks: dict[int, _RankProgress] = {}
        self.semantic_events = 0
        self._active_stage_starts: dict[tuple[int, str], list[float]] = {}
        self.stage_duration_seconds: dict[str, list[float]] = {}

    def observe(self, line: str, now: float) -> bool:
        marker = line.find(PROGRESS_PREFIX)
        if marker < 0:
            return False
        raw = line[marker + len(PROGRESS_PREFIX) :].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict) or payload.get("schema") != "graft-gs-progress-v1":
            return False
        try:
            rank = int(payload["rank"])
            sequence = int(payload["sequence"])
            semantic_sequence = int(payload.get("semantic_sequence", 0))
        except (KeyError, TypeError, ValueError):
            return False
        state = self.ranks.setdefault(rank, _RankProgress(rank=rank))
        if sequence <= state.sequence:
            return False
        state.sequence = sequence
        try:
            state.pid = int(payload["pid"])
        except (KeyError, TypeError, ValueError):
            pass
        state.semantic_sequence = semantic_sequence
        state.stage = str(payload.get("stage", "unknown"))
        state.event = str(payload.get("event", "unknown"))
        state.last_record_monotonic = now
        state.context = {
            name: payload[name]
            for name in (
                "phase",
                "epoch",
                "global_step",
                "microstep",
                "object_ids",
                "draw_index",
                "draw_count",
                "layer_index",
                "layer_count",
                "active_charts",
                "sparse_tokens",
                "transport_edges",
                "cuda_allocated_bytes",
                "cuda_reserved_bytes",
            )
            if name in payload
        }
        stage = state.stage
        event = state.event
        stage_key = (rank, stage)
        if event == "begin":
            self._active_stage_starts.setdefault(stage_key, []).append(now)
        elif event in {"end", "failed"}:
            starts = self._active_stage_starts.get(stage_key)
            if starts:
                duration = max(0.0, now - starts.pop())
                history = self.stage_duration_seconds.setdefault(stage, [])
                history.append(duration)
                if len(history) > 128:
                    del history[:-128]
                if not starts:
                    self._active_stage_starts.pop(stage_key, None)
        semantic = payload.get("semantic_progress") is True
        if semantic:
            state.last_semantic_monotonic = now
            self.last_semantic_monotonic = now
            self.semantic_events += 1
        return semantic

    def _stage_budget(self, stage: str) -> float:
        matches = [
            (prefix, seconds)
            for prefix, seconds in self.stage_timeout_seconds.items()
            if stage.startswith(prefix)
        ]
        minimum = self.no_progress_timeout_seconds
        if matches:
            _, minimum = max(matches, key=lambda item: len(item[0]))
        history = self.stage_duration_seconds.get(stage, [])
        if len(history) < 3:
            return float(minimum)
        ordered = sorted(history)
        quantile_index = min(
            len(ordered) - 1,
            max(0, int(0.99 * len(ordered))),
        )
        q99 = ordered[quantile_index]
        median = statistics.median(ordered)
        mad = statistics.median(abs(value - median) for value in ordered)
        robust_budget = q99 + self.stage_latency_mad_multiplier * mad
        return max(float(minimum), robust_budget)

    def deadline_status(self, now: float) -> tuple[bool, str, float, float]:
        if self.last_semantic_monotonic is None:
            elapsed = now - self.started
            return (
                elapsed >= self.bootstrap_timeout_seconds,
                "bootstrap_no_semantic_progress",
                elapsed,
                self.bootstrap_timeout_seconds,
            )
        elapsed = now - self.last_semantic_monotonic
        active_budget = max(
            [self._stage_budget(state.stage) for state in self.ranks.values()]
            or [self.no_progress_timeout_seconds]
        )
        return (
            elapsed >= active_budget,
            "stage_no_semantic_progress",
            elapsed,
            active_budget,
        )

    def snapshot(self, now: float) -> dict[str, object]:
        expired, reason, elapsed, budget = self.deadline_status(now)
        expected = (
            list(range(self.world_size)) if self.world_size is not None else []
        )
        missing = [rank for rank in expected if rank not in self.ranks]
        return {
            "schema": "graft-gs-semantic-watchdog-v1",
            "expired": expired,
            "reason": reason,
            "seconds_since_semantic_progress": elapsed,
            "active_budget_seconds": budget,
            "stage_timeout_policy": "max_configured_minimum_q99_plus_mad",
            "stage_latency_mad_multiplier": self.stage_latency_mad_multiplier,
            "semantic_events": self.semantic_events,
            "missing_ranks": missing,
            "ranks": [
                {
                    "rank": state.rank,
                    "pid": state.pid,
                    "stage": state.stage,
                    "event": state.event,
                    "sequence": state.sequence,
                    "semantic_sequence": state.semantic_sequence,
                    "seconds_since_record": (
                        now - state.last_record_monotonic
                        if state.last_record_monotonic
                        else None
                    ),
                    "seconds_since_rank_semantic_progress": (
                        now - state.last_semantic_monotonic
                        if state.last_semantic_monotonic
                        else None
                    ),
                    **state.context,
                }
                for state in sorted(self.ranks.values(), key=lambda item: item.rank)
            ],
        }


def _parse_stage_timeouts(values: Sequence[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("stage timeout must have the form PREFIX=SECONDS")
        prefix, raw_seconds = value.rsplit("=", 1)
        prefix = prefix.strip().rstrip(".")
        if not prefix:
            raise ValueError("stage timeout prefix must be non-empty")
        try:
            seconds = float(raw_seconds)
        except ValueError as error:
            raise ValueError(f"invalid stage timeout {value!r}") from error
        if seconds <= 0:
            raise ValueError("stage timeouts must be positive")
        parsed[prefix] = seconds
    return parsed


def _classify_probe_failure(
    log_path: Path,
    *,
    return_code: int,
    oom: bool,
    timed_out: bool,
) -> str | None:
    if oom:
        return "capacity.cuda_oom"
    if timed_out:
        return "liveness.no_semantic_progress"
    if return_code == 0:
        return None
    normalized = log_path.read_text(encoding="utf8", errors="replace").lower()
    for name, markers in (
        ("numerics.nonfinite", NONFINITE_MARKERS),
        ("distributed.collective_failure", COLLECTIVE_FAILURE_MARKERS),
        ("data.contract_failure", DATA_FAILURE_MARKERS),
        ("model_or_dependency.failure", MODEL_FAILURE_MARKERS),
    ):
        if any(marker in normalized for marker in markers):
            return name
    if return_code < 0:
        return "process.signal_failure"
    return "process.nonzero_exit"


def _probe_outcome(
    *,
    return_code: int,
    oom: bool,
    timed_out: bool,
    failure_class: str | None,
) -> str:
    """Map detailed diagnostics to the stable external candidate contract."""

    if return_code == 0 and not oom and not timed_out:
        return "SUCCESS"
    if oom:
        return "OOM"
    if timed_out:
        return "NO_PROGRESS"
    return {
        "numerics.nonfinite": "NONFINITE",
        "distributed.collective_failure": "COLLECTIVE_FAILURE",
        "data.contract_failure": "DATA_FAILURE",
        "model_or_dependency.failure": "MODEL_FAILURE",
        "process.signal_failure": "EXTERNAL_TERMINATION",
        "process.nonzero_exit": "MODEL_FAILURE",
    }.get(failure_class, "MODEL_FAILURE")


def _runtime_policy_from_training_arguments(
    arguments: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    configured = _option_value(arguments, "--config")
    path = (
        Path(configured)
        if configured is not None
        else ROOT / "configs" / "graft_gs_a800_native.yaml"
    )
    if not path.is_absolute():
        path = ROOT / path
    data = yaml.safe_load(path.read_text(encoding="utf8"))
    if not isinstance(data, dict):
        raise ValueError("training configuration root must be a mapping")
    watchdog = data.get("watchdog", {})
    autotune = data.get("autotune", {})
    instrumentation = data.get("instrumentation", {})
    if (
        not isinstance(watchdog, dict)
        or not isinstance(autotune, dict)
        or not isinstance(instrumentation, dict)
    ):
        raise ValueError(
            "watchdog/autotune/instrumentation sections must be mappings"
        )
    if watchdog.get("policy", "semantic_progress") != "semantic_progress":
        raise ValueError("autotuning requires watchdog.policy=semantic_progress")
    if (
        watchdog.get("stage_timeout_policy", "robust_q99_mad")
        != "robust_q99_mad"
    ):
        raise ValueError(
            "autotuning requires watchdog.stage_timeout_policy=robust_q99_mad"
        )
    if instrumentation.get("enabled", True) is not True:
        raise ValueError("autotuning requires instrumentation.enabled=true")
    return watchdog, autotune


def _validate_candidates(values: Sequence[int]) -> tuple[int, ...]:
    candidates = tuple(int(value) for value in values)
    if not candidates or any(value < 1 for value in candidates):
        raise ValueError("batch candidates must be positive")
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError("batch candidates must be strictly increasing")
    return candidates


def _gpu_environment(gpus: str | None) -> tuple[dict[str, str], str]:
    selected = gpus if gpus is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if selected is None or not selected.strip():
        raise ValueError(
            "--gpus or an existing CUDA_VISIBLE_DEVICES value must explicitly "
            "name the available devices"
        )
    selected = selected.strip()
    if re.fullmatch(r"[A-Za-z0-9:._,/\\-]+", selected) is None:
        raise ValueError("GPU identifiers contain unsupported characters")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = selected
    environment["PYTHONUNBUFFERED"] = "1"
    # Match torchrun's safe per-rank default explicitly and avoid repeating its
    # warning at the beginning of every expensive candidate process group.
    environment.setdefault("OMP_NUM_THREADS", "1")
    # The autotuner invokes torchrun directly, so it cannot rely on the shell
    # launcher to translate PyTorch's deprecated NCCL variable.  Keeping both
    # names makes every rank print the warning twice and leaves behavior
    # version-dependent.
    legacy_async_handling = environment.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    environment.setdefault(
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        legacy_async_handling if legacy_async_handling is not None else "1",
    )
    environment.setdefault("TORCH_NCCL_DESYNC_DEBUG", "1")
    environment.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    environment.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000")
    return environment, selected


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    """Return one value from either ``--x value`` or ``--x=value``."""

    matches: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            matches.append(arguments[index + 1])
        elif argument.startswith(option + "="):
            matches.append(argument.split("=", 1)[1])
    if len(matches) > 1:
        raise ValueError(f"{option} may be specified only once")
    return matches[0] if matches else None


def _training_batch_policy(
    arguments: Sequence[str],
) -> tuple[str | None, int | None]:
    present = [
        (option, _option_value(arguments, option))
        for option in _ACCUMULATION_OPTIONS
    ]
    present = [(option, value) for option, value in present if value is not None]
    if len(present) > 1:
        raise ValueError(
            "gradient accumulation, global object batch, and minimum global "
            "object batch are mutually exclusive"
        )
    if not present:
        return None, None
    option, raw_value = present[0]
    assert raw_value is not None
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{option} must be an integer") from error
    if value < 1:
        raise ValueError(f"{option} must be positive")
    return option, value


def _single_microbatch_probe_arguments(
    arguments: Sequence[str],
) -> list[str]:
    """Replace only the probe's optimizer-batch policy with one microbatch.

    Activation peak and optimizer/DDP state are materialized by this step, but
    a candidate no longer re-samples enough objects to satisfy the production
    global batch.  The unmodified arguments are used for the actual launch.
    """

    stripped: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        exact = argument in _ACCUMULATION_OPTIONS
        assigned = any(
            argument.startswith(option + "=")
            for option in _ACCUMULATION_OPTIONS
        )
        if exact:
            if index + 1 >= len(arguments):
                raise ValueError(f"{argument} requires a value")
            index += 2
            continue
        if assigned:
            index += 1
            continue
        stripped.append(argument)
        index += 1
    return [*stripped, "--gradient-accumulation-steps", "1"]


def _candidate_policy_failure(
    policy: tuple[str | None, int | None],
    world_size: int,
    object_batch_size: int,
) -> str | None:
    option, value = policy
    if option != "--global-object-batch":
        return None
    assert value is not None
    physical_global_batch = world_size * object_batch_size
    if value < physical_global_batch or value % physical_global_batch:
        return (
            f"production --global-object-batch {value} is not a positive "
            f"multiple of WORLD_SIZE * object batch ({physical_global_batch})"
        )
    return None


def _cuda_inventory(
    python: str,
    environment: dict[str, str],
) -> list[dict[str, object]]:
    source = (
        "import json,torch\n"
        "rows=[]\n"
        "for i in range(torch.cuda.device_count()):\n"
        " free,total=torch.cuda.mem_get_info(i)\n"
        " rows.append({'logical_device':i,'name':torch.cuda.get_device_name(i),"
        "'free_bytes':int(free),'total_bytes':int(total),"
        "'free_fraction':float(free/total)})\n"
        "print(json.dumps(rows,sort_keys=True))\n"
    )
    completed = subprocess.run(
        [python, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=ROOT,
    )
    inventory = json.loads(completed.stdout)
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("the explicit GPU set exposes no usable CUDA device")
    return inventory


def _run_and_tee(
    command: Sequence[str],
    log_path: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    termination_grace_seconds: float = 15.0,
    no_progress_timeout_seconds: float | None = None,
    world_size: int | None = None,
    stage_timeout_seconds: dict[str, float] | None = None,
    stage_latency_mad_multiplier: float = 6.0,
    control_scope: str = "probe",
) -> tuple[int, bool, bool]:
    """Run one isolated group; bound bootstrap/stalls, never total wall time."""

    if timeout_seconds <= 0 or termination_grace_seconds <= 0:
        raise ValueError("probe timeout and termination grace must be positive")
    if control_scope not in {"probe", "training"}:
        raise ValueError("control_scope must be 'probe' or 'training'")
    control_prefix = (
        "GRAFT_GS_AUTOTUNE_PROBE_CONTROL"
        if control_scope == "probe"
        else "GRAFT_GS_TRAINING_SUPERVISOR_CONTROL"
    )
    timeout_prefix = (
        "GRAFT_GS_AUTOTUNE_PROGRESS_TIMEOUT"
        if control_scope == "probe"
        else "GRAFT_GS_TRAINING_PROGRESS_TIMEOUT"
    )
    with log_path.open("w", encoding="utf8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        output: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    output.put(line)
            finally:
                output.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        oom = False
        timed_out = False
        output_closed = False
        started = time.monotonic()
        monitor = SemanticProgressMonitor(
            started=started,
            bootstrap_timeout_seconds=timeout_seconds,
            no_progress_timeout_seconds=(
                timeout_seconds
                if no_progress_timeout_seconds is None
                else no_progress_timeout_seconds
            ),
            world_size=world_size,
            stage_timeout_seconds=stage_timeout_seconds,
            stage_latency_mad_multiplier=stage_latency_mad_multiplier,
        )
        termination_started: float | None = None
        forced_kill = False

        def emit_control(
            reason: str,
            action: str,
            details: dict[str, object] | None = None,
        ) -> None:
            line = (
                control_prefix
                + " "
                + json.dumps(
                    {
                        "action": action,
                        "reason": reason,
                        **({"details": details} if details is not None else {}),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()

        def terminate_group(
            reason: str,
            details: dict[str, object] | None = None,
        ) -> None:
            nonlocal termination_started
            if termination_started is not None:
                return
            termination_started = time.monotonic()
            emit_control(reason, "SIGTERM_PROCESS_GROUP", details)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        try:
            while not output_closed or process.poll() is None:
                try:
                    line = output.get(timeout=0.25)
                except queue.Empty:
                    line = ""
                if line is None:
                    output_closed = True
                elif line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                    monitor.observe(line, time.monotonic())
                    normalized = line.lower()
                    if any(marker in normalized for marker in OOM_MARKERS):
                        oom = True
                        terminate_group("cuda_allocation_failure")

                now = time.monotonic()
                expired, timeout_reason, _, _ = monitor.deadline_status(now)
                if not timed_out and termination_started is None and expired:
                    timed_out = True
                    snapshot = monitor.snapshot(now)
                    diagnostic = (
                        timeout_prefix
                        + " "
                        + json.dumps(snapshot, sort_keys=True)
                        + "\n"
                    )
                    sys.stdout.write(diagnostic)
                    sys.stdout.flush()
                    log.write(diagnostic)
                    log.flush()
                    worker_pids = sorted(
                        {
                            state.pid
                            for state in monitor.ranks.values()
                            if state.pid is not None
                        }
                    )
                    if worker_pids:
                        emit_control(
                            timeout_reason,
                            "SIGUSR2_WORKER_STACK_DUMP",
                            {"worker_pids": worker_pids},
                        )
                        for worker_pid in worker_pids:
                            try:
                                os.kill(worker_pid, signal.SIGUSR2)
                            except ProcessLookupError:
                                pass
                        # Let faulthandler write through the already-drained
                        # output pipe before terminating the complete group.
                        time.sleep(0.25)
                    terminate_group(
                        f"probe_timeout_{timeout_reason}",
                        snapshot,
                    )
                if (
                    termination_started is not None
                    and not forced_kill
                    and process.poll() is None
                    and now - termination_started >= termination_grace_seconds
                ):
                    forced_kill = True
                    emit_control(
                        "termination_grace_expired",
                        "SIGKILL_PROCESS_GROUP",
                    )
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except BaseException:
            terminate_group("autotuner_interrupted")
            try:
                process.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            reader.join(timeout=1.0)
            process.stdout.close()
            raise

        return_code = process.wait()
        reader.join(timeout=1.0)
        process.stdout.close()
        return return_code, oom, timed_out


def _torchrun_command(
    python: str,
    world_size: int,
    train_arguments: Sequence[str],
) -> list[str]:
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(ROOT / "scripts" / "train_a800.py"),
        *train_arguments,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus")
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=os.environ.get("GRAFT_GS_PYTHON", sys.executable),
    )
    parser.add_argument("--maximum-allocated-fraction", type=float, default=0.85)
    parser.add_argument("--maximum-reserved-fraction", type=float, default=0.88)
    parser.add_argument("--minimum-driver-free-fraction", type=float, default=0.08)
    parser.add_argument(
        "--minimum-initial-driver-free-fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument("--throughput-fraction", type=float, default=0.97)
    parser.add_argument(
        "--maximum-throughput-cv",
        type=float,
        help="maximum steady-state step-throughput coefficient of variation",
    )
    parser.add_argument(
        "--maximum-rank-step-time-ratio",
        type=float,
        help="maximum slowest/fastest rank step-time ratio",
    )
    parser.add_argument(
        "--probe-warmup-steps",
        type=int,
        help="optimizer steps excluded from throughput (defaults to config)",
    )
    parser.add_argument(
        "--probe-measurement-steps",
        type=int,
        help="optimizer steps included in throughput (defaults to config)",
    )
    parser.add_argument(
        "--probe-trellis-cache-scope",
        choices=("candidate", "shared"),
        help=(
            "candidate preserves fair cold/mixed-cache timing; shared reuses "
            "exact priors across candidates (defaults to config)"
        ),
    )
    parser.add_argument(
        "--probe-timeout-seconds",
        type=float,
        help=(
            "backward-compatible bootstrap bound before the first semantic "
            "progress record; defaults to config and is not a total wall limit"
        ),
    )
    parser.add_argument(
        "--probe-no-progress-timeout-seconds",
        type=float,
        help="fallback no-progress bound (defaults to config)",
    )
    parser.add_argument(
        "--stage-timeout",
        action="append",
        default=[],
        metavar="PREFIX=SECONDS",
        help=(
            "override a semantic stage no-progress budget; may be repeated, "
            "and the longest matching stage prefix wins"
        ),
    )
    parser.add_argument(
        "--termination-grace-seconds",
        type=float,
        help="SIGTERM-to-SIGKILL grace period (defaults to config)",
    )
    parser.add_argument(
        "--continue-after-oom",
        action="store_true",
        help=(
            "probe larger candidates after an OOM; by default monotonic larger "
            "physical batches are skipped"
        ),
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="launch the full training command with the selected batch",
    )
    parser.add_argument(
        "train_arguments",
        nargs=argparse.REMAINDER,
        help="arguments for scripts/train_a800.py, following --",
    )
    args = parser.parse_args()
    candidates = _validate_candidates(args.candidates)
    for name in (
        "maximum_allocated_fraction",
        "maximum_reserved_fraction",
        "minimum_driver_free_fraction",
        "minimum_initial_driver_free_fraction",
        "throughput_fraction",
    ):
        value = float(getattr(args, name))
        if not 0 < value < 1:
            raise ValueError(f"{name.replace('_', '-')} must lie in (0,1)")
    train_arguments = list(args.train_arguments)
    if train_arguments and train_arguments[0] == "--":
        train_arguments.pop(0)
    if not train_arguments:
        raise ValueError("train_a800.py arguments must follow --")
    watchdog_config, autotune_config = _runtime_policy_from_training_arguments(
        train_arguments
    )
    if args.probe_timeout_seconds is None:
        args.probe_timeout_seconds = float(
            watchdog_config.get("bootstrap_timeout_seconds", 1800.0)
        )
    if args.probe_no_progress_timeout_seconds is None:
        args.probe_no_progress_timeout_seconds = float(
            watchdog_config.get("no_progress_timeout_seconds", 900.0)
        )
    if args.probe_warmup_steps is None:
        args.probe_warmup_steps = int(
            autotune_config.get("warmup_optimizer_steps", 1)
        )
    if args.probe_measurement_steps is None:
        args.probe_measurement_steps = int(
            autotune_config.get("measurement_optimizer_steps", 2)
        )
    if args.probe_trellis_cache_scope is None:
        args.probe_trellis_cache_scope = str(
            autotune_config.get("probe_trellis_cache_scope", "candidate")
        )
    if args.probe_trellis_cache_scope not in {"candidate", "shared"}:
        raise ValueError("autotune probe TRELLIS cache scope is invalid")
    if args.maximum_throughput_cv is None:
        args.maximum_throughput_cv = float(
            autotune_config.get("maximum_throughput_cv", 0.25)
        )
    if args.maximum_rank_step_time_ratio is None:
        args.maximum_rank_step_time_ratio = float(
            autotune_config.get("maximum_rank_step_time_ratio", 3.0)
        )
    if args.termination_grace_seconds is None:
        args.termination_grace_seconds = float(
            watchdog_config.get("termination_grace_seconds", 15.0)
        )
    stage_latency_mad_multiplier = float(
        watchdog_config.get("stage_latency_mad_multiplier", 6.0)
    )
    forbidden = {
        "--object-batch-size",
        "--batch-probe",
        "--batch-probe-warmup-steps",
        "--batch-probe-measurement-steps",
    }
    if any(
        argument in forbidden
        or any(argument.startswith(option + "=") for option in forbidden)
        for argument in train_arguments
    ):
        raise ValueError(
            "the autotuner owns --object-batch-size and --batch-probe"
        )
    if (
        args.probe_timeout_seconds <= 0
        or args.probe_no_progress_timeout_seconds <= 0
        or args.termination_grace_seconds <= 0
        or stage_latency_mad_multiplier < 0
    ):
        raise ValueError("probe watchdog timeouts must be positive")
    if args.probe_warmup_steps < 0 or args.probe_measurement_steps < 1:
        raise ValueError("probe warmup/measurement step counts are invalid")
    if (
        args.maximum_throughput_cv < 0
        or args.maximum_rank_step_time_ratio < 1
    ):
        raise ValueError("probe stability thresholds are outside their domains")
    configured_stage_timeouts = watchdog_config.get(
        "stage_timeout_seconds",
        {},
    )
    if not isinstance(configured_stage_timeouts, dict):
        raise ValueError("watchdog.stage_timeout_seconds must be a mapping")
    stage_timeouts = {
        str(prefix): float(seconds)
        for prefix, seconds in configured_stage_timeouts.items()
    }
    if any(seconds <= 0 for seconds in stage_timeouts.values()):
        raise ValueError("configured stage timeouts must be positive")
    stage_timeouts.update(_parse_stage_timeouts(args.stage_timeout))
    production_batch_policy = _training_batch_policy(train_arguments)
    probe_train_arguments = _single_microbatch_probe_arguments(train_arguments)
    if not Path(args.python).is_file():
        raise FileNotFoundError(args.python)
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError("autotune output directory must be fresh")
    args.output.mkdir(parents=True, exist_ok=True)
    configured_trellis_cache = _option_value(
        train_arguments,
        "--trellis-cache-directory",
    )
    probe_trellis_cache_root = (
        Path(configured_trellis_cache).expanduser().resolve()
        if configured_trellis_cache is not None
        else (args.output / "trellis_exact_cache").resolve()
    )
    production_trellis_cache = (
        probe_trellis_cache_root
        if configured_trellis_cache is not None
        else (args.output / "production_trellis_exact_cache").resolve()
    )
    effective_probe_cache_scope = (
        "explicit"
        if configured_trellis_cache is not None
        else args.probe_trellis_cache_scope
    )

    environment, selected_gpus = _gpu_environment(args.gpus)
    subprocess.run(
        [
            args.python,
            str(ROOT / "scripts" / "validate_environment.py"),
            "--requirements",
            str(ROOT / "requirements.txt"),
            "--output",
            str(args.output / "environment.json"),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )
    inventory = _cuda_inventory(args.python, environment)
    occupied = [
        row
        for row in inventory
        if float(row["free_fraction"])
        < args.minimum_initial_driver_free_fraction
    ]
    inventory_path = args.output / "initial_cuda_memory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "cuda_visible_devices": selected_gpus,
                "devices": inventory,
                "minimum_initial_driver_free_fraction": (
                    args.minimum_initial_driver_free_fraction
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    if occupied:
        raise RuntimeError(
            f"an explicitly selected GPU is already occupied; inspect {inventory_path}"
        )

    world_size = len(inventory)
    runs: list[dict[str, object]] = []
    safe: list[dict[str, object]] = []
    for candidate_index, batch_size in enumerate(candidates):
        run_directory = args.output / f"batch-{batch_size:03d}"
        run_directory.mkdir()
        candidate_trellis_cache = (
            probe_trellis_cache_root
            if effective_probe_cache_scope in {"shared", "explicit"}
            else probe_trellis_cache_root / f"batch-{batch_size:03d}"
        )
        candidate_cache_arguments = (
            []
            if configured_trellis_cache is not None
            else ["--trellis-cache-directory", str(candidate_trellis_cache)]
        )
        probe_path = run_directory / "probe.json"
        command = _torchrun_command(
            args.python,
            world_size,
            [
                *probe_train_arguments,
                *candidate_cache_arguments,
                "--object-batch-size",
                str(batch_size),
                "--batch-probe",
                str(probe_path),
                "--batch-probe-warmup-steps",
                str(args.probe_warmup_steps),
                "--batch-probe-measurement-steps",
                str(args.probe_measurement_steps),
                "--output",
                str(run_directory / "training"),
            ],
        )
        policy_failure = _candidate_policy_failure(
            production_batch_policy,
            world_size,
            batch_size,
        )
        if policy_failure is not None:
            runs.append(
                {
                    "object_batch_size": batch_size,
                    "return_code": None,
                    "oom": False,
                    "timed_out": False,
                    "seconds": 0.0,
                    "probe": None,
                    "admissible": False,
                    "skipped": True,
                    "reasons": [policy_failure],
                    "command": command,
                }
            )
            continue
        print(
            "GRAFT_GS_AUTOTUNE_CANDIDATE_START "
            + json.dumps(
                {
                    "object_batch_size": batch_size,
                    "probe_gradient_accumulation_steps": 1,
                    "probe_warmup_optimizer_steps": args.probe_warmup_steps,
                    "probe_measurement_optimizer_steps": (
                        args.probe_measurement_steps
                    ),
                    "maximum_throughput_cv": args.maximum_throughput_cv,
                    "maximum_rank_step_time_ratio": (
                        args.maximum_rank_step_time_ratio
                    ),
                    "trellis_cache_scope": effective_probe_cache_scope,
                    "trellis_cache_directory": str(candidate_trellis_cache),
                    "bootstrap_timeout_seconds": args.probe_timeout_seconds,
                    "no_progress_timeout_seconds": (
                        args.probe_no_progress_timeout_seconds
                    ),
                    "stage_timeout_seconds": {
                        **DEFAULT_STAGE_TIMEOUT_SECONDS,
                        **stage_timeouts,
                    },
                    "stage_latency_mad_multiplier": stage_latency_mad_multiplier,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        started = time.perf_counter()
        return_code, oom, timed_out = _run_and_tee(
            command,
            run_directory / "run.log",
            environment,
            args.probe_timeout_seconds,
            termination_grace_seconds=args.termination_grace_seconds,
            no_progress_timeout_seconds=(
                args.probe_no_progress_timeout_seconds
            ),
            world_size=world_size,
            stage_timeout_seconds=stage_timeouts,
            stage_latency_mad_multiplier=stage_latency_mad_multiplier,
        )
        probe = (
            json.loads(probe_path.read_text(encoding="utf8"))
            if return_code == 0 and probe_path.is_file()
            else None
        )
        reasons: list[str] = []
        failure_class = _classify_probe_failure(
            run_directory / "run.log",
            return_code=return_code,
            oom=oom,
            timed_out=timed_out,
        )
        outcome = _probe_outcome(
            return_code=return_code,
            oom=oom,
            timed_out=timed_out,
            failure_class=failure_class,
        )
        if return_code != 0:
            reasons.append(f"probe exited with status {return_code}")
        if oom:
            reasons.append("CUDA allocation failure")
        if timed_out:
            reasons.append(
                "probe stopped making semantic stage progress; inspect the "
                "rank-state timeout record in run.log"
            )
        if isinstance(probe, dict):
            if (
                int(probe["minimum_realized_object_batch_size"])
                != batch_size
                or int(probe["maximum_realized_object_batch_size"])
                != batch_size
            ):
                reasons.append("candidate object batch was not fully realized")
            if (
                float(probe["maximum_peak_allocated_fraction"])
                > args.maximum_allocated_fraction
            ):
                reasons.append("allocated-memory headroom failed")
            if (
                float(probe["maximum_peak_reserved_fraction"])
                > args.maximum_reserved_fraction
            ):
                reasons.append("reserved-memory headroom failed")
            if (
                float(probe["minimum_ending_driver_free_fraction"])
                < args.minimum_driver_free_fraction
            ):
                reasons.append("driver-free-memory headroom failed")
            if (
                float(
                    probe["measurement_throughput_coefficient_of_variation"]
                )
                > args.maximum_throughput_cv
            ):
                reasons.append("steady-state throughput stability failed")
            if (
                float(probe["maximum_rank_step_time_ratio"])
                > args.maximum_rank_step_time_ratio
            ):
                reasons.append("cross-rank step-time balance failed")
        else:
            reasons.append("probe report is unavailable")
        record = {
            "object_batch_size": batch_size,
            "return_code": return_code,
            "oom": oom,
            "timed_out": timed_out,
            "failure_class": failure_class,
            "outcome": outcome,
            "seconds": time.perf_counter() - started,
            "probe": probe,
            "admissible": not reasons,
            "reasons": reasons,
            "command": command,
        }
        runs.append(record)
        if not reasons:
            safe.append(record)
        print(
            "GRAFT_GS_AUTOTUNE_CANDIDATE_END "
            + json.dumps(
                {
                    "admissible": not reasons,
                    "object_batch_size": batch_size,
                    "oom": oom,
                    "reasons": reasons,
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "failure_class": failure_class,
                    "outcome": outcome,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        stop_reason = None
        if oom and not args.continue_after_oom:
            stop_reason = "skipped after smaller physical batch exhausted CUDA memory"
        elif timed_out:
            stop_reason = (
                "skipped after smaller physical batch stopped making semantic progress"
            )
        if stop_reason is not None:
            for skipped_batch in candidates[candidate_index + 1 :]:
                runs.append(
                    {
                        "object_batch_size": skipped_batch,
                        "return_code": None,
                        "oom": False,
                        "timed_out": False,
                        "seconds": 0.0,
                        "probe": None,
                        "admissible": False,
                        "skipped": True,
                        "reasons": [stop_reason],
                        "command": None,
                    }
                )
            break

    if not safe:
        summary_path = args.output / "selection.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema": "graft-gs-object-batch-selection-v3",
                    "selected_object_batch_size": None,
                    "probe_policy": "isolated_warmup_and_steady_state_measurement",
                    "probe_warmup_optimizer_steps": args.probe_warmup_steps,
                    "probe_measurement_optimizer_steps": (
                        args.probe_measurement_steps
                    ),
                    "production_batch_policy": production_batch_policy,
                    "runs": runs,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf8",
        )
        raise RuntimeError(f"no batch candidate passed; inspect {summary_path}")

    fastest = max(
        float(record["probe"]["aggregate_objects_per_second"])
        for record in safe
    )
    near_fastest = [
        record
        for record in safe
        if float(record["probe"]["aggregate_objects_per_second"])
        >= args.throughput_fraction * fastest
    ]
    selected = max(
        near_fastest, key=lambda record: int(record["object_batch_size"])
    )
    selected_batch = int(selected["object_batch_size"])
    launch_command = _torchrun_command(
        args.python,
        world_size,
        [
            *train_arguments,
            *(
                []
                if configured_trellis_cache is not None
                else [
                    "--trellis-cache-directory",
                    str(production_trellis_cache),
                ]
            ),
            "--object-batch-size",
            str(selected_batch),
        ],
    )
    selection_path = args.output / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "graft-gs-object-batch-selection-v3",
                "cuda_visible_devices": selected_gpus,
                "world_size": world_size,
                "selected_object_batch_size": selected_batch,
                "throughput_fraction": args.throughput_fraction,
                "maximum_throughput_cv": args.maximum_throughput_cv,
                "maximum_rank_step_time_ratio": (
                    args.maximum_rank_step_time_ratio
                ),
                "probe_policy": "isolated_warmup_and_steady_state_measurement",
                "probe_warmup_optimizer_steps": args.probe_warmup_steps,
                "probe_measurement_optimizer_steps": (
                    args.probe_measurement_steps
                ),
                "production_batch_policy": production_batch_policy,
                "probe_trellis_cache_scope": effective_probe_cache_scope,
                "probe_trellis_cache_root": str(probe_trellis_cache_root),
                "production_trellis_cache_directory": str(
                    production_trellis_cache
                ),
                "watchdog": {
                    "policy": "semantic_progress",
                    "bootstrap_timeout_seconds": args.probe_timeout_seconds,
                    "no_progress_timeout_seconds": (
                        args.probe_no_progress_timeout_seconds
                    ),
                    "stage_timeout_seconds": {
                        **DEFAULT_STAGE_TIMEOUT_SECONDS,
                        **stage_timeouts,
                    },
                    "stage_latency_mad_multiplier": stage_latency_mad_multiplier,
                },
                "launch_command": launch_command,
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    print(
        json.dumps(
            {
                "selection": str(selection_path),
                "selected_object_batch_size": selected_batch,
                "world_size": world_size,
                "aggregate_objects_per_second": selected["probe"][
                    "aggregate_objects_per_second"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.launch:
        launch_started = time.perf_counter()
        launch_return_code, launch_oom, launch_timed_out = _run_and_tee(
            launch_command,
            args.output / "production.log",
            environment,
            args.probe_timeout_seconds,
            termination_grace_seconds=args.termination_grace_seconds,
            no_progress_timeout_seconds=(
                args.probe_no_progress_timeout_seconds
            ),
            world_size=world_size,
            stage_timeout_seconds=stage_timeouts,
            stage_latency_mad_multiplier=stage_latency_mad_multiplier,
            control_scope="training",
        )
        launch_result = {
            "schema": "graft-gs-supervised-training-launch-v1",
            "return_code": launch_return_code,
            "oom": launch_oom,
            "timed_out": launch_timed_out,
            "failure_class": _classify_probe_failure(
                args.output / "production.log",
                return_code=launch_return_code,
                oom=launch_oom,
                timed_out=launch_timed_out,
            ),
            "seconds": time.perf_counter() - launch_started,
            "command": launch_command,
        }
        launch_result["outcome"] = _probe_outcome(
            return_code=launch_return_code,
            oom=launch_oom,
            timed_out=launch_timed_out,
            failure_class=launch_result["failure_class"],
        )
        (args.output / "launch_result.json").write_text(
            json.dumps(launch_result, indent=2, sort_keys=True) + "\n",
            encoding="utf8",
        )
        raise SystemExit(launch_return_code)


if __name__ == "__main__":
    main()
