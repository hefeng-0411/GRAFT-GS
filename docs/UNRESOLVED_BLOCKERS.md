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
