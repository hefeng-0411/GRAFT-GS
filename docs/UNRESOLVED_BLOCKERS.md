# Unresolved blockers and incomplete research work

Updated 2026-07-16. An item is called externally blocked only when code or a
server validation definition exists and progress requires unavailable hardware,
checkpoints, data, or a compiled server dependency.

## External execution blockers

- Full VGGT/TRELLIS/GRAFT checkpoint forward and backward require the enterprise
  A800 environment and checkpoint access.
- CUDA/reference raster equivalence requires the server-built
  `diff_gaussian_rasterization` extension.
- Six-rank DDP equivalence, rank-local RNG resume, peak memory, and throughput
  measurements require a multi-A800 `torchrun` execution.
- One-object overfitting and real multiview inference require an actual training
  subset and trained GRAFT-GS phase checkpoints. The audited local MeshFleet
  tree has one canonical test object and no usable train population.
- TRELLIS hidden-support and structured-latent behavior cannot be numerically
  validated without the pinned external TRELLIS checkpoint.
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
- Format-5 exact DDP checkpoint and Phase-F Fisher-state continuation.
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
  record-count, canonical-identity, missing-summary, compatible-reuse, and
  many-object ordering tests. What remains external is executing the full
  rebuild against the large mounted remote corpus and retaining its digest and
  measured build duration.
- The reference command intentionally does not execute six-rank DDP or a real
  image/checkpoint inference corpus. Those retain their dedicated commands in
  `docs/A800_VALIDATION_PROTOCOL.md`; all dataset, CUDA renderer, and
  nvdiffrast skips in the reference suite are now hard orchestration failures.
- The accelerator probe contract is locally tested with synthetic metadata,
  but its CUDA 11.8, BF16, A800 identity, compute capability, and memory record
  remains pending until `validate_server.py` is rerun on the enterprise host.
- The six-rank validator now rejects world-size/device aliasing and aggregates
  pass/fail across ranks, but NCCL initialization, distinct A800 assignment,
  global-evidence gradients, prior broadcast, and rank-local RNG replay remain
  genuinely external until its structured JSON/log is produced on the server.
- Phase launches are now pinned to the exact CRAFT interpreter and audit its
  requirements first. Executing the Bash launcher, NCCL initialization, staged
  backward passes, checkpoint boundaries, runtime, and peak memory remains an
  external six-A800 task.
