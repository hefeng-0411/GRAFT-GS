# Unresolved blockers and incomplete research work

Updated 2026-07-17. An item is called externally blocked only when code or a
server validation definition exists and progress requires unavailable hardware,
checkpoints, data, or a compiled server dependency.

## External execution blockers

- Full VGGT/TRELLIS adapter forwards require execution of the new real-view
  preflight in the enterprise A800 environment. The user reports that both
  released checkpoints exist in the server's default caches; their GRAFT-GS
  adapter results and provenance JSON have not yet been returned to this local
  workspace. The validator now also requires and fingerprints the declared
  `/mnt/sda2/hef/Base/vggt` and `/mnt/sda2/hef/Base/TRELLIS` checkouts, but
  their actual server hashes have not been observed locally. A trained GRAFT
  checkpoint still requires staged training.
- CUDA/reference raster equivalence has now executed once and failed under the
  pre-repair renderer (RGB max absolute error `0.6104467`, 14.2% elements out
  of tolerance). The TRELLIS mip semantics, pixel centers, covariance path,
  and auxiliary backgrounds were repaired; the exact off-axis/nonblack test
  must be rerun on the A800 before equivalence is claimed.
- Scheduler-visible multi-rank DDP equivalence, rank-local RNG resume, peak memory, and throughput
  measurements require a multi-A800 `torchrun` execution.
- One-object overfitting and real multiview inference require the remote full
  train/test corpus and trained GRAFT-GS phase checkpoints. The local single
  sample is schema/audit provenance only and is never a production root.
- TRELLIS hidden-support and structured-latent behavior remains numerically
  pending until the server executes its cached released checkpoint through
  `scripts/validate_external_models.py trellis` and the production suite.
- Learned VGG perceptual behavior cannot be validated without an explicitly
  pinned local checkpoint; the implementation deliberately has no downloader.
- The one-step quantization inequality requires a conservative A800-measured
  downstream vector-field Lipschitz upper bound and observed normalized
  query/key quantization error. The geometric topology margin and inference
  report are implemented; absent measurements disable certification.

## Implemented research paths still needing server validation

- Conditional post-split atlas gradients and overlap/multilevel objectives.
- OT-cost/uncertainty-biased GSTA in the production encoder.
- Adaptive occupancy and camera-exact sparse reprojection octree refinement,
  including camera-gradient and multi-resolution threshold calibration.
- Orientability-filtered topology proposals with persistence-critical,
  adaptive-quantile, and fixed coverage thresholds.
- Predictor-and-corrector feasibility backtracking and QP primal-margin checks.
- Curvature-quadrature Gaussian allocation and PBR GLB reload.
- Phase B/C execution-stage isolation.
- Format-6 exact DDP checkpoint, native-precision provenance, and Phase-F
  Fisher-state continuation.
- Hilbert-space multiview gradient purification and manual post-purification
  DDP averaging.
- Bounded quantization-scale inner maximization and dimensionless positive
  topology/feasibility-margin hardening.
- Gauge-covariant internal activation and product-manifold vector-field
  Jacobian distillation.
- Effective flow multiplicity spectral policy and normalized residual path;
  composed field-bound measurement remains in the external section above.
- Differentiable camera/depth track-cycle supervision and VGGT-depth-derived
  world normal pseudo-targets.
- Topology-fixed robust teacher bundle adjustment, analytical pseudo-assets,
  confidence/provenance serialization, and Phase-C dataset activation.
- Hash-pinned frozen VGG16 perceptual supervision and exact-resume provenance.
- Explicit Jeffreys-smoothed TRELLIS candidate shape likelihood separated from
  observed UOT evidence and persistence.
- Metric topology-boundary distance and complete conditional quantization
  certificate inference report.

## Optimization work gated by external reference validation

- A true matched-precision INT4/FP4 backward kernel. Current QAT is an explicit
  straight-through approximation. Per the reference-first policy, kernel work
  is gated on the unexecuted A800 native/fake-quant numerical and topology-
  margin equivalence suite; no unvalidated low-bit kernel is substituted.
- Numerically equivalent fused sparse UOT/GSTA Triton or CUDA kernels. Their
  high-precision reference contracts and prerequisite invariant tests exist;
  backend-specific equivalence tests must be introduced together with a real
  kernel interface. Implementation/build remains gated on passing the A800
  sparse UOT, implicit-gradient, equivariance, and production-path baselines
  and on the server compiler/backend versions.

## 2026-07-17 pinned A800 rerun boundary

- The supplied 76-test A800 report predates the tensor-broadcast, finite-cycle-
  gradient, certificate-dtype, prior-reliability, remote-manifest, and strict
  environment repairs. The corrected suite must be rerun with
  `/mnt/sda1/miniforge3/envs/CRAFT/bin/python scripts/validate_server.py` and
  the declared `/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97`
  root. No corrected numerical result is claimed yet.
- The manifest handoff now locally passes stale-schema, wrong-root,
  record-count, modality-intersection, discovered-ID-digest, missing-summary,
  compatible-reuse, and many-object ordering tests. What remains external is executing the full
  rebuild against the large mounted remote corpus and retaining its digest and
  measured build duration.
- The reference command intentionally does not execute visible-rank DDP or a real
  image/checkpoint inference corpus. Those retain their dedicated commands in
  `docs/A800_VALIDATION_PROTOCOL.md`; all dataset, CUDA renderer, and
  nvdiffrast skips in the reference suite are now hard orchestration failures.
- The accelerator probe contract is locally tested with synthetic metadata,
  but its CUDA 11.8, BF16, A800 identity, compute capability, and memory record
  remains pending until `validate_server.py` is rerun on the enterprise host.
- The visible-rank validator now rejects world-size/device aliasing and aggregates
  pass/fail across ranks, but NCCL initialization, distinct A800 assignment,
  global-evidence gradients, prior broadcast, and rank-local RNG replay remain
  genuinely external until its structured JSON/log is produced on the server.
- Phase launches are now pinned to the exact CRAFT interpreter and audit its
  requirements first. Executing the Bash launcher, NCCL initialization, staged
  backward passes, checkpoint boundaries, runtime, and peak memory remains an
  external scheduler-visible A800 task.

## 2026-07-22 TRELLIS decoded-grid rerun boundary

- The supplied real-checkpoint run established a pre-repair failure:
  flow-latent resolution 16 was incorrectly applied to decoded coordinates in
  `[0,63]`. Runtime decoder-shape capture and CPU regression definitions are
  implemented. The focused CRAFT adapter tests, checkpoint-backed TRELLIS
  preflight, and one-object DDP overfit must be rerun on the server before the
  repair is called A800 validated.
- The first post-grid overfit advanced through source-only TRELLIS sampling but
  failed because Torch 2.4 NCCL rejects `int16` atlas levels. The next run
  passed that boundary and exposed raw PCA-frame gauge disagreement. Exact
  int64 discrete transport, source-owned autograd floating synchronization,
  and eigengap-stratified PCA derivatives are now implemented with mock,
  numerical and torchrun regression paths. The corrected
  `validate_ddp_server.py` suite and overfit must execute before same-object
  atlas synchronization or its backward path is called A800 validated.
- The subsequent smoke passed both prior failures and completed one optimizer
  update, then the second forward rejected non-finite evidence mass. The
  Phase-B repeated-spectrum readout and optimizer-containment repairs are
  implemented. What remains external is running the focused float64 gradient
  tests under the pinned Torch 2.4 environment, then rerunning two steps. A
  successful gate must show two finite steps; a failure must now name the
  first non-finite loss/gradient/parameter rather than reaching a corrupted
  second-forward atlas. Do not resume a checkpoint produced by the failed run.

## 2026-07-22 A800 concurrency measurement boundary

- The supplied process table proves the old run had cross-device allocator
  ownership. Early binding and a hard foreign-device allocator check are now
  implemented, but one-PID-per-GPU must be confirmed on the server after all
  old processes have exited. Any ownership exception is a correctness failure,
  not a reason to disable the guard.
- The locally selected default is 24 views per ordinary object-level rank. For
  same-object overfit, 16, 24, 32, 48, and 64 views per rank must be profiled on
  the actual visible subset. Choose the highest measured global useful views/s
  that preserves finite steps and leaves at least phase/object-dependent
  headroom; 100% reserved memory is not a completion criterion. Phase D/F and
  highly refined objects can peak above the two-step Phase-B fixture.

## 2026-07-22 strict topology/UOT rerun boundary

- The supplied smoke is pre-repair evidence: it used an older entry point
  (`find_unused_parameters=True`, all-rank final evaluation) and failed on a
  real `8.60e-5 < 1.0e-4` embedding separation. The current source adds strict
  restoration, rank-zero evaluation, FP64 gradient norm accumulation, and a
  collective checkpoint commit fence. These changes must be deployed together;
  no checkpoint from the failed run is a validated result.
- The synthetic restoration/gradient test and sparse UOT forward/adjoint
  failure tests require PyTorch and are not executable in the local drafting
  runtime. Run the focused command in `A800_VALIDATION_PROTOCOL.md`, then rerun
  the two-step checkpoint-backed smoke. A pass requires positive final area,
  orientation, separation, and covariance margins on the recertifying
  projector, two finite optimizer steps, a committed checkpoint, and clean
  rank-zero-only asset evaluation.
- If restoration reports constraint-qualification failure or exhausts the
  configured displacement budget, this is an invalid proposed embedding, not
  permission to reduce `minimum_separation`. Retain the complete candidate
  failure string for topology-proposal diagnosis.

## 2026-07-22 useful-concurrency measurement boundary

- The supplied 16/24/32/48/64 sweep is pre-repair failure evidence, not a
  utilization result: 16 failed on FP32 UOT storage underflow and every larger
  candidate failed topology proposal. Deploy the FP64/log-domain UOT and
  maximal-support filtration changes, pass the five focused numerical tests,
  rerun the sweep with the new telemetry schema, and retain `selection.json`.
  The selected value is not A800 validated until every candidate process has
  exited cleanly and the selector accepts at least one report.
- Larger 4096/16384 radius-graph chunks require the existing sparse-support and
  dense/sparse UOT numerical tests under Torch 2.4. If a threshold-boundary
  fixture differs, retain both supports and revert the chunk sizes rather than
  redefining equivalence.
- Phase-B selection does not prove Phase-D/F memory safety. Repeat a shortened
  sweep after flow/rendering activation; retain at least 15% peak reserved
  headroom for corpus variation unless measured tail-object profiling supports
  a different limit.

## 2026-07-23 persistence-memory rerun boundary

- The post-log-UOT smoke passed the previously failing transport and topology
  proposal boundaries, then failed before backward because dense persistence
  matching requested 11.42 GiB with approximately 8.5 GiB free. This run is
  not an OOM-recovery candidate and provides no throughput report.
- Deploy the zero-pair simplification and hybrid exact/sliced matcher, then run
  the protocol's five focused numerical tests. The large matcher test must
  pass with `torch.cdist` deliberately disabled. Rerun the two-step smoke with
  the mandatory `--output "$SMOKE_DIR"` argument.
- A valid `overfit_metrics.json` must record three persistence cardinality
  pairs and three modes (`exact` or `sliced`) for the selected topology. Peak
  allocated/reserved memory must then be measured anew; the 69.67 GiB
  pre-repair allocation is not assumed to be the post-repair peak.

## 2026-07-23 mesh-target raster-memory rerun boundary

- The next run passed topology and failed later in nvdiffrast target
  rasterization with CUDA error 2. It used the old post-forward full-view
  target schedule and a malformed command ending in
  `tee "$SMOKE_DIR/run.log"-1`; it produced no valid metrics/checkpoint/assets
  certificate.
- Deploy the bounded rasterizer and pre-forward target schedule. Run the
  protocol's six focused tests; the MeshFleet test must establish continuous
  chunk/full-batch agreement at `1e-6` and exact Boolean masks on the A800
  implementation.
- Rerun a fresh two-step smoke with explicit `--evaluation-views 24`,
  `--steps 2`, `--minimum-relative-improvement -1`, and
  `--output "$SMOKE_DIR"`. Do not reuse any output directory from a malformed
  or failed command.
- A pass requires two finite optimizer steps, converged FP64/log UOT,
  cardinality/mode persistence telemetry, positive final feasibility margins,
  a committed final checkpoint, rank-zero evaluation, and independently
  reloadable PLY/GLB. New rank-local peak allocated/reserved memory is required
  before increasing views per rank.

## 2026-07-24 high-view CUDA renderer rerun boundary

- The 32-view/rank run predates per-view checkpoint recomputation and is not a
  valid concurrency candidate. The final `normalize` frame is not interpreted
  as an algorithmic normal defect because CUDA reported asynchronous OOM.
- Deploy the renderer/trainer/config/sweep changes together. Run the protocol's
  seven focused tests; checkpointed and uncheckpointed CUDA rendering must
  agree for color, alpha, depth, normal, and finite Gaussian gradients.
- Use a new timestamped sweep root. Format-7 reports must record
  `rendering.backend == "cuda"` and `rendering.checkpoint_views == true`.
  Pre-repair reports are deliberately ineligible.
- If no report is selected, inspect the always-written `selection.json`.
  Do not increase the 0.85 memory gate until the recorded reasons establish
  that an otherwise finite, converged, feasible candidate is rejected only for
  reserved fraction and tail-object profiling justifies a smaller headroom.
- The selected Phase-B view count remains invalid for Phase D/F until those
  phases repeat a shortened sweep; activation recomputation removes native
  renderer tapes but not `O(KHW)` outputs, VGGT tokens, evidence, or
  object-dependent atlas/UOT growth.
## 2026-07-24 allocator/error-certificate server validation

- This historical boundary is superseded by the schema-v4 section below.
  At this point a fresh schema-v3 A800 sweep was required. Existing schema-v2 reports lack
  ending allocator/driver telemetry and FP64-referenced transport storage-error
  mass, so they are intentionally inadmissible.
- The vpr-48 run returned code 1 after 83.29 seconds but the supplied attachment
  contains only the sweep summary, not `vpr-48/run.log`. Its cause is genuinely
  external until that log is supplied or the fresh sweep reproduces it.
- The vpr-64 OOM remains a measured upper-bound failure, not a reason to weaken
  precision or supervision. Re-evaluate it only after vpr-48 and the new
  allocator lifetime telemetry are understood; the then-current sweep stopped
  after the first recognized OOM by default.
- Required server evidence: all nine focused numerical/allocator tests, per-rank
  cache-release bytes, peak allocated/active/reserved memory, ending
  allocated/reserved/driver-free memory, UOT storage relative-L1 and discarded
  mass, full child logs, and selector schema v3. No corrected memory budget or
  throughput is claimed before those artifacts exist.

## 2026-07-25 source-offload and schema-v4 rerun boundary

- The latest supplied schema-v3 reports are pre-repair evidence. They prove
  healthy FP64/log-domain transport storage but fail memory admission, and
  vpr-32 fails backward at cuBLAS handle creation with virtually no driver
  headroom. They cannot select a production view budget.
- Deploy source-rank-only TRELLIS ownership, exact host offload, the
  16-conditioning-view hidden-prior cap, stage telemetry, and exhaustive sweep
  together. Existing v3 reports intentionally lack the v4 offload,
  conditioning, lifetime-peak, and stage-memory certificates.
- Run the focused Torch/CUDA tests in `A800_VALIDATION_PROTOCOL.md`, then use a
  fresh timestamped root. `initial_cuda_memory.json` must show every logical
  device at or above 95% free before checkpoint loading. If it fails, select a
  different idle `CUDA_VISIBLE_DEVICES`; do not lower training precision or
  the admission thresholds.
- A successful report must show source-rank pipeline offload, a positive frozen
  prior peak, selected conditioning views no greater than 16, all required
  stage records, converged UOT, positive feasibility margins, and at least 5%
  ending driver-visible headroom. On failure, retain every
  `GRAFT_GS_CUDA_FAILURE_DIAGNOSTICS=` line.
- The conditioning cap requires a server ablation against uncapped TRELLIS on
  a representative validation subset. Reconstruction/topology quality, not
  VRAM occupancy alone, determines whether 16 remains the production value.
- No post-repair A800 optimizer step, throughput, memory reduction, selected
  view budget, or quality improvement has yet been demonstrated.
# 2026-07-25 server-only validation after offload-v4 repair

- External: the local drafting interpreter has no PyTorch/CUDA/TRELLIS/VGGT,
  so the new sparse-vs-dense barrier equivalence, zero-reliability backward,
  implicit-UOT cotangent gate, and offline DINOv2 cache tests have not executed
  here. Run the focused command in `A800_VALIDATION_PROTOCOL.md`.
- External: rerun a fresh view sweep; schema-v5 intentionally rejects all old
  reports that lack `evaluation_execution_stage=atlas_autoencoding`. Verify
  vpr-8 backward, vpr-12 offline construction, and vpr-24 terminal export.
- External: Phase-C/D full-flow profiling must demonstrate linear barrier
  memory and positive recertified margins on a real refined atlas. A Phase-B
  sweep no longer executes flow and therefore cannot validate this requirement.
- No claim of zero loss or universally error-free training is supportable.
  Admission remains fail-closed on finite gradients, converged UOT, positive
  feasibility margins, deterministic assets, and explicit A800 memory limits.

## 2026-07-25 posterior-measure A800 validation boundary

- Supplied pre-repair schema-v5 evidence: vpr-8/12/16 completed and vpr-16 was
  selected at `0.027420849` aggregate views/s with `14.8232%` peak allocated,
  `19.0835%` peak reserved, and `80.1792%` ending driver-free memory. Vpr
  24/32/48/64 failed with respectively 845, 7639, 19219, and 17933 non-finite
  upstream plan-cotangent edges. None of these four failures was OOM.
- Local drafting cannot execute the new PyTorch numerical tests. First run the
  focused posterior/zero-distance suite, then one fresh 24-view, one-step
  Phase-B smoke with two evaluation views. This directly crosses the first
  prior failure boundary without paying for another complete sweep.
- If 24 passes, run the same one-step smoke at 32. Do not launch 48/64 or
  change production view count until 24 and 32 both produce finite optimizer
  steps and named gradient boundaries remain silent.
- Existing vpr-16 is a valid measurement of the pre-repair code, not proof of
  post-repair correctness. After the focused gate passes, 16 views/rank is the
  conservative training start; a new schema-v5 sweep is required before
  increasing it. Full Phase-D/F memory and numerical validation remain
  separate.

## 2026-07-25 SPD metric A800 validation boundary

- Resolved evidence: the ten posterior numerical tests and the real 24-view
  Phase-B optimizer/asset gate pass on the A800.
- External remaining gate: run the three SPD/readout regressions and only the
  32-view one-step command in `A800_VALIDATION_PROTOCOL.md`. A pass must emit a
  final checkpoint, PLY, GLB, finite metric gradients, and no
  `state_initialization.*` or `analytical_readout.*` cotangent error.
- Full Phase C/F execution is still required to validate the now-shared
  SPD-native product-metric and barrier paths. No claim of universally zero
  loss/error is supportable; exact finite/convergence/feasibility admission
  remains mandatory.
- The first Cholesky deployment did not pass 32 views: it localized the
  remaining failure to Torch's factorization backward. The analytical
  pullback is implemented and locally syntax/static-validated, but its five
  Torch numerical tests and second 32-view production attempt are external
  pending work.
## Resolved A800 Phase-B numerical closure gate (2026-07-25)

- The v2 preflight and 32-view one-step production gate passed on the remote
  A800. It completed backward, optimizer update, checkpoint, PLY, GLB,
  converged FP64 UOT, and positive feasibility certificates.
- This resolves the former immediate chart-metric blocker. Steady-state
  training quality, exact resume, multi-rank equivalence, later Phase C-F
  execution, and full-corpus evaluation remain separate unresolved validation
  requirements.

## 2026-07-30 A100 exact-collision rerun boundary

- The supplied Phase-B failure is locally resolved at its two causal
  boundaries: exact zero-distance constraints now have a conditional
  frame-equivariant restoration direction, and rejected candidate trials no
  longer retain autograd graphs. The selected restoration's unrolled dual
  computation is block-checkpointed without changing its numerical result.
- A local synthetic CUDA probe demonstrates a 15.8x reduction in retained
  forward allocation for the sparse dual solve. This is not a substitute for
  the real VGGT/TRELLIS/atlas memory composition on the remote A100 ranks.
- Remaining external gate: rerun the Phase-B command at 16 maximum global views
  on an idle 2–4×A100 80-GB allocation. Capture the first feasibility report,
  per-stage/peak/ending memory metrics, optimizer completion, and checkpoint.

## 2026-07-30 Phase-B DDP watchdog correction

- The reported rank-zero `ALLREDUCE NumelIn=1` timeout is the trainer's scalar
  finite-state gate after Phase-B forward work, not evidence of a large
  gradient collective or an NCCL bandwidth limit. The process-group default
  expired at exactly 600 seconds while rank zero's sequence 38 waited for at
  least one peer.
- Ordinary batch-one DDP previously used random `DistributedSampler` sharding,
  so ranks could receive meshes with very different face/support cardinality.
  CUDA work was asynchronous: the scalar NCCL work timer could start while a
  peer's raster/atlas/topology stream was still processing its tail object.
- The implementation now cost-cohorts all ordinary distributed batches,
  completes local CUDA dependencies before scalar health collectives, uses a
  configurable 1800-second process-group timeout, records rank/GPU ownership,
  and emits rank/object local-stall diagnostics. Gradient-bucket overlap during
  backward is unchanged.
- External closure still requires one idle four-GPU A6000 smoke and the
  production four-A100 rerun. A pass consists of all four
  `GRAFT_GS_DDP_INITIALIZED` records, no ownership duplicates, finite optimizer
  completion, and no NCCL watchdog/desynchronization dump. Batch tuning must be
  performed independently on 48-GB A6000 and 80-GB A100 allocations.

## 2026-07-31 Phase-B object-batch probe failure localization

- The supplied batch-8 log is from four A800-SXM4-80GB devices. Rank 3 fails in
  the trainable GSTA forward at `gsta.py`'s scalar-to-tensor path: 77.49 GiB is
  live PyTorch allocation, only 55 MiB is driver-free, and only 237.65 MiB is
  reserved-but-unallocated when a further 68 MiB is requested. This is a real
  capacity failure, not allocator fragmentation and not an NCCL fault.
- The batch-4 record has no collective timeout or OOM. All four ranks emit a
  local `backward` stage warning after 120 seconds. The variable-atlas scene
  graphs are processed locally and remain live for backward; a stage warning
  is diagnostic and does not prove deadlock, but an unbounded probe is not an
  admissible production selector.
- This historical revision used one optimizer-bearing microbatch and a total
  wall timeout. The 2026-08-02 incident below supersedes that policy: the real
  launch still retains `--global-object-batch` exactly and OOM remains
  process-group-fatal, but timing now uses warmup/measurement and liveness is
  bounded by semantic stage progress rather than total wall time.
- GSTA now shares the exact target scalar gather across its three coupling
  paths and broadcasts head attention over multiplicities without materialized
  `repeat_interleave` tensors. Forward values, reduction order, precision, and
  the training objective are unchanged. Protocol 9 additionally checkpoints
  the deterministic prepared-edge kernel using per-forward effective spectral
  weights and evaluates normalized inner products without normalized edge
  copies. The 8,192-vertex/four-layer audit reduces autograd-retained storage
  from 580,188,592 to 20,434,064 bytes with zero output/gradient relative-L2
  error. Fresh A6000/A100 allocator probes and the batch-8 soak remain the
  external validation boundary because this container currently exposes no
  NVIDIA device.

## 2026-08-02 legacy probe-timeout resolution

- The latest supplied log has four finite `backward` local-ready records after
  long Phase-B forwards. Its next candidate is terminated by the parent record
  `reason=probe_timeout` while TRELLIS sampling still advances; no preceding
  CUDA OOM, non-finite marker, NCCL failure, or worker traceback establishes a
  child failure. `scripts/analyze_training_log.py` classifies it as
  `supervisor.legacy_fixed_wall_timeout`.
- Workers now emit semantic rank/stage/workload progress through model load,
  sampling, VGGT, atlas/UOT/GSTA/topology/render, loss, backward sentinels,
  optimizer, collectives, and checkpointing. Heartbeats are non-semantic.
  Tuning and `--launch` supervise lack of progress by stage, dump rank state
  plus Python stacks, and then terminate the complete group. There is no total
  candidate or production wall deadline.
- Probes use one warmup plus two steady-state optimizer steps by default;
  memory gates include all steps, throughput excludes warmup, and variance plus
  cross-rank skew are admission criteria. Exact TRELLIS caches are atomic,
  provenance-namespaced, bounded, and candidate-local by default so timings are
  comparable.
- Local CRAFT validation passes the focused 87-test DDP/sampler/telemetry suite,
  a one-GPU NCCL/progress smoke, a two-GPU NCCL/progress smoke, and all five
  two-rank differentiable evidence/RNG/prior collectives. Four-GPU validation
  was not run because GPUs 0 and 1 are occupied by unrelated processes. The
  local environment also differs from the pinned 444-package server lock.
- External closure remains a fresh four-A6000 candidate sweep plus 200-step
  soak and an independent four-A100/A800 sweep plus soak. No batch result may
  be transferred between those pools.
