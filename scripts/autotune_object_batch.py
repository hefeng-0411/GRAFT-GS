"""Select a safe, high-throughput object batch on an explicit GPU set.

Every candidate runs in a fresh torchrun process group. An allocation failure,
mutated optimizer, allocator cache, or stochastic state from a probe therefore
cannot leak into the launched training job.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
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
) -> tuple[int, bool, bool]:
    """Run one isolated process group and terminate all ranks on OOM/timeout."""

    if timeout_seconds <= 0 or termination_grace_seconds <= 0:
        raise ValueError("probe timeout and termination grace must be positive")
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
        termination_started: float | None = None
        forced_kill = False

        def emit_control(reason: str, action: str) -> None:
            line = (
                "GRAFT_GS_AUTOTUNE_PROBE_CONTROL "
                + json.dumps(
                    {"action": action, "reason": reason},
                    sort_keys=True,
                )
                + "\n"
            )
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()

        def terminate_group(reason: str) -> None:
            nonlocal termination_started
            if termination_started is not None:
                return
            termination_started = time.monotonic()
            emit_control(reason, "SIGTERM_PROCESS_GROUP")
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
                    normalized = line.lower()
                    if any(marker in normalized for marker in OOM_MARKERS):
                        oom = True
                        terminate_group("cuda_allocation_failure")

                now = time.monotonic()
                if (
                    not timed_out
                    and termination_started is None
                    and now - started >= timeout_seconds
                ):
                    timed_out = True
                    terminate_group("probe_timeout")
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
        "--probe-timeout-seconds",
        type=float,
        default=1800.0,
        help="terminate the complete candidate process group after this wall time",
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
    forbidden = {"--object-batch-size", "--batch-probe"}
    if any(
        argument in forbidden
        or any(argument.startswith(option + "=") for option in forbidden)
        for argument in train_arguments
    ):
        raise ValueError(
            "the autotuner owns --object-batch-size and --batch-probe"
        )
    if args.probe_timeout_seconds <= 0:
        raise ValueError("--probe-timeout-seconds must be positive")
    production_batch_policy = _training_batch_policy(train_arguments)
    probe_train_arguments = _single_microbatch_probe_arguments(train_arguments)
    if not Path(args.python).is_file():
        raise FileNotFoundError(args.python)
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError("autotune output directory must be fresh")
    args.output.mkdir(parents=True, exist_ok=True)

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
        probe_path = run_directory / "probe.json"
        command = _torchrun_command(
            args.python,
            world_size,
            [
                *probe_train_arguments,
                "--object-batch-size",
                str(batch_size),
                "--batch-probe",
                str(probe_path),
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
                    "timeout_seconds": args.probe_timeout_seconds,
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
        )
        probe = (
            json.loads(probe_path.read_text(encoding="utf8"))
            if return_code == 0 and probe_path.is_file()
            else None
        )
        reasons: list[str] = []
        if return_code != 0:
            reasons.append(f"probe exited with status {return_code}")
        if oom:
            reasons.append("CUDA allocation failure")
        if timed_out:
            reasons.append(
                f"probe exceeded {args.probe_timeout_seconds:g} seconds"
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
        else:
            reasons.append("probe report is unavailable")
        record = {
            "object_batch_size": batch_size,
            "return_code": return_code,
            "oom": oom,
            "timed_out": timed_out,
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
                },
                sort_keys=True,
            ),
            flush=True,
        )
        stop_reason = None
        if oom and not args.continue_after_oom:
            stop_reason = "skipped after smaller physical batch exhausted CUDA memory"
        elif timed_out:
            stop_reason = "skipped after smaller physical batch exceeded probe timeout"
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
                    "schema": "graft-gs-object-batch-selection-v2",
                    "selected_object_batch_size": None,
                    "probe_policy": "single_microbatch_optimizer_step",
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
            "--object-batch-size",
            str(selected_batch),
        ],
    )
    selection_path = args.output / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "graft-gs-object-batch-selection-v2",
                "cuda_visible_devices": selected_gpus,
                "world_size": world_size,
                "selected_object_batch_size": selected_batch,
                "throughput_fraction": args.throughput_fraction,
                "probe_policy": "single_microbatch_optimizer_step",
                "production_batch_policy": production_batch_policy,
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
        raise SystemExit(
            subprocess.run(
                launch_command,
                cwd=ROOT,
                env=environment,
                check=False,
            ).returncode
        )


if __name__ == "__main__":
    main()
