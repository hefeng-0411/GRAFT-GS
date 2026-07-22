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
  same-object overfit, 8, 12, and 16 views per rank must be profiled on the
  actual visible subset. Choose the highest measured global useful views/s
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
