# DDP training incident analysis

## Evidence and root cause

The supplied Phase B tuning log contains two distinct candidate records.

- Batch 1 initializes all four NCCL ranks. Each rank reports a long
  `forward_and_loss` interval for a different object, then all four ranks report
  `GRAFT_GS_DDP_LOCAL_READY` with finite loss tensors and sub-millisecond local
  CUDA waits before backward. This is synchronized, advancing execution—not a
  rank divergence.
- Batch 2 initializes all four ranks and continues emitting completed TRELLIS
  sampling displays. At the fixed parent deadline the log records
  `GRAFT_GS_AUTOTUNE_PROBE_CONTROL` with `reason=probe_timeout`, followed by
  SIGTERM/elastic `SignalException`. There is no preceding CUDA OOM, non-finite
  marker, NCCL transport error, or worker traceback establishing a child-side
  failure in that candidate.

The immediate cause is therefore a supervisory policy error: the former tuner
treated total candidate wall time as lack of progress and killed a healthy
process group. Repeated `Sampling: 12/12` displays are the configured eight
frozen TRELLIS posterior draws per uncached object, not an infinite retry loop.

This does not erase the separate batch-8 capacity event in the earlier log. In
that event approximately 77.5 GiB was live on an 80-GiB device with negligible
driver-free memory and a real allocation failed. It is classified as a CUDA
capacity failure and remains inadmissible. Similarly, the earlier scalar NCCL
all-reduce timeout was a downstream symptom: at least one rank did not reach
the same health collective before NCCL's deadline. A larger NCCL timeout alone
cannot diagnose or repair the rank-local stage that delayed it.

## Corrective architecture

The training workers now emit `graft-gs-progress-v1` JSON records containing
rank/device, epoch/step/microstep, object IDs, semantic stage/event, TRELLIS draw
or GSTA layer position, workload cardinalities, CPU RSS, and non-synchronizing
CUDA allocator counters. Long backward passes expose evenly spaced
gradient-ready sentinels whose hooks return gradients unchanged. Heartbeats are
marked `semantic_progress=false`.

The tuner now has separate bootstrap and stage no-progress deadlines. Total
wall time is unbounded while semantic advancement continues. Before terminating
a frozen candidate it writes `GRAFT_GS_AUTOTUNE_PROGRESS_TIMEOUT` with every
rank's last stage, event, workload, memory, and time since semantic progress,
then terminates the complete process group. Failures are classified separately
as CUDA capacity, non-finite numerics, collective failure, data contract,
model/dependency, signal, or no semantic progress.

Candidate timing now excludes a configurable warmup and uses multiple
steady-state optimizer steps. Memory admission includes both periods. The
production `--global-object-batch` is still preserved exactly; probes use one
physical microbatch only inside fresh disposable processes.

## Sampling and memory policy

TRELLIS remains scientifically unchanged: the same selected views, seeds,
posterior count, sampler steps, decoded support, and precision are used. Exact
sample results may be reused within a run or an explicitly shared cache only when a SHA-256 key
over the full conditioning tensor bytes and sampling seed matches inside a
namespace derived from the checkpoint, upstream source digest, and sampling
policy. The disk cache is atomic, process-locked, provenance-checked, and
bounded. Candidate-local cache scope is the tuning default so cache warmth does
not bias batch comparisons. Frozen TRELLIS weights are offloaded and inactive allocator blocks are
released before the differentiable VGGT/GRAFT-GS lifetime begins.

DDP remains one process per visible GPU, with explicit local device binding,
`find_unused_parameters=false`, gradient bucket views, cost-balanced exact-view
batches, `no_sync()` only for non-terminal accumulation microsteps, collective
finite-state failure decisions, and rank-local RNG checkpoint state. No dummy
allocation or memory-filling tensor is used.

## Required production validation

Run the same phase/checkpoint/manifest independently on the exact A6000 pool
and the exact A100/A800 pool. A result is valid only when all requested ranks
are initially unoccupied, every candidate produces a probe-v3 report, the
selected batch satisfies all allocator/driver headroom bounds, and the launched
job advances through at least 200 optimizer steps with finite losses and no
watchdog, NCCL, or CUDA-capacity termination. Hardware-specific batch results
must not be copied between 48-GiB A6000 and 80-GiB production devices.
