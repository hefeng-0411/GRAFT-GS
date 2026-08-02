# Multi-GPU execution topology

This document is the pre-change execution map used to audit Phase B and the
other staged training paths.  It distinguishes rank-local work from
collectives and records which repeated operations are scientifically required.

## Process topology

`scripts/launch_a800_6gpu.sh` and `scripts/autotune_object_batch.py` launch one
`torch.distributed.run` worker per entry in `CUDA_VISIBLE_DEVICES`.  Each worker
binds `LOCAL_RANK` to one logical CUDA device before constructing CUDA state.
`DistributedDataParallel` owns one complete GRAFT-GS replica per rank.  Normal
object-level DDP gives different, cost-balanced objects to ranks; only model
gradients synchronize.  The optional same-object mode instead shards one
object's views and explicitly gathers evidence before the nonlinear atlas
solve.

## Phase B critical path

| Vertex | Rank ownership | Precision | Gradient | Synchronization | Repetition/cache policy |
| --- | --- | --- | --- | --- | --- |
| Manifest lookup, image/mesh decode | CPU worker, rank-local | source/FP32 | none | equal batch count is guaranteed by the distributed batch sampler | one load per selected object and epoch |
| Host-to-device transfer | rank-local CUDA stream | FP32 | none | none | one transfer per microbatch |
| TRELLIS conditioning and sparse-structure prior | rank-local in ordinary DDP; rank 0 in same-object mode | upstream native inference | frozen/no-grad | same-object mode broadcasts the resulting exact sparse measure | eight posterior draws, twelve diffusion steps per draw; exact-value cache only |
| VGGT (`vggt.models.vggt.VGGT`) geometry | rank-local | BF16 backbone, FP32 outputs | frozen in Phase B | none | once per object batch; this is VGGT, not “VGDT” |
| Camera alignment and evidence lift | rank-local | FP32, FP64 diagnostics only | evidence calibrator is trainable | optional differentiable all-gather in same-object mode | once per object |
| Persistent atlas construction/refinement | rank-local in ordinary DDP | FP32 state, FP64 conditional diagnostics | discrete structure; continuous mapping remains differentiable | exact atlas/split agreement only in same-object mode | initial solve plus configured refinement rounds |
| Sparse UOT fixed point | rank-local | FP64/log-space solve, FP32 returned plan | implicit differentiable solve | no cross-rank reduction in ordinary DDP | configured transport/refinement iterations |
| Gauge sparse transport attention | rank-local | FP32 | trainable | DDP synchronizes resulting parameter gradients | four encoder layers in the server profile |
| Topology proposal and feasible-stratum selection | rank-local | FP32 values, FP64 certification where declared | trainable energies on the selected stratum; discrete identity detached | none in ordinary DDP | once per scene |
| Analytical readout and input-view rendering | rank-local | FP32 | differentiable; deterministic render is activation-checkpointed | none | one scene readout; all admitted supervision views render |
| Phase B objective | rank-local | FP32 with declared FP64 diagnostics | trainable | a scalar finite-state health all-reduce precedes backward | one mean-reduced objective per microbatch |
| Backward | rank-local autograd plus NCCL | native parameter/gradient dtypes | yes | DDP all-reduces the same ordered gradient buckets on every rank | `no_sync()` only on non-terminal accumulation microsteps |
| AdamW update | rank-local identical replicas | parameter-native state | yes | gradients are already reduced; finite-state checks are collective | once per configured global object batch |
| Checkpoint | rank-local RNG gather, rank-0 serialization | exact stored tensors | none | all ranks enter RNG gather and failure fence | configured optimizer-step cadence |

Phase A stops after evidence calibration.  Phase C stops after constrained flow
training and does not render assets.  Phases D--F traverse the full graph;
phases E/F add the declared teacher, quantization, or gradient-purification
work.  Phase-specific trainability is established before DDP wrapping, so the
server profile uses `find_unused_parameters: false`.

## Collective ordering invariant

Every rank receives the same number of physical batches and accumulation
microsteps.  Data-dependent object complexity may change arrival time but may
not change collective order.  The ordered collective sequence is: optional
same-object evidence/atlas collectives, pre-backward finite-state all-reduce,
DDP gradient bucket all-reduces on terminal microsteps, gradient/optimizer
finite-state all-reduces, and checkpoint/validation collectives.  No rank-local
OOM, invalid sample, or empty topology branch may silently continue while peers
enter a later collective; failures must terminate the complete candidate group
or be converted into an identical collective decision.

## Liveness interpretation

TRELLIS prints `Sampling: 12/12` once per posterior draw.  With eight configured
draws this is expected repeated work, not an optimizer loop.  A tuner must not
bound the entire candidate by one wall timer: model loading, sampling,
transport, rendering, and backward have different legitimate durations.
Supervision therefore consumes structured rank/stage progress and terminates
only after a stage-specific interval with no semantic advancement.  Heartbeats
describe liveness but do not reset that semantic deadline.
