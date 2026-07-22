# Mathematical assumptions and validity domain

## Released-model boundary

- VGGT input is floating RGB in `[0,1]`, with shape `[B,K,3,H,W]`. The audited
  released checkpoint has four non-null concatenated frame/global taps of width
  2048 and camera/depth/point heads with the inspected signatures. An upstream
  version changing these fields is rejected, not silently adapted.
- VGGT camera conversion is OpenCV world-to-camera with pixel intrinsics. The
  single supervised Sim(3) alignment is valid only for nondegenerate multi-view
  camera centers and removes no per-camera residual.
- Released VGGT includes a query-conditioned track head but not the
  specification's dense patch-descriptor tensor. Projective track-cycle
  supervision assumes depth/camera predictions describe the same static scene.
- TRELLIS tensor conditioning is floating RGB `[K,3,H,W]` in `[0,1]`. Its
  sparse-structure flow resolution is the latent grid (16 for the released
  checkpoint), not the coordinate domain returned after decoding. The adapter
  observes the actual cubic decoder output `[1,1,R,R,R]` on every draw and
  validates returned `[batch,x,y,z]` coordinates against that decoded `R` (64
  for the released checkpoint). The empirical Beta-Bernoulli support assumes
  posterior draws are deterministic under the recorded object seed and
  checkpoint. It is a prior, not calibrated ground-truth occupancy.
- In same-object DDP the TRELLIS pipeline is frozen and evaluated under
  `no_grad`; source-only sampling followed by exact tensor broadcast is
  mathematically equivalent to redundant identical-rank sampling conditional
  on equal gathered images, checkpoint, sampler parameters, and RNG seed.
- Same-object atlas synchronization treats discrete topology as an exact
  source-authoritative integer state. Because Torch 2.4 NCCL lacks `int16`, its
  portable wire representation is int64 with a checked round trip. Continuous
  atlas values use the source rank in the forward pass but the identity
  derivative of each rank's local global-evidence construction; this is valid
  only after metadata equality, finite-state checks, and the configured
  cross-rank numerical tolerance pass.

## Cameras and evidence

- MeshFleet camera manifests are Blender/OpenGL camera-to-world transforms.
  Right multiplication by `diag(1,-1,-1,1)` gives an OpenCV camera-to-world
  transform before inversion to VGGT's camera-from-world convention.
- VGGT extrinsics are OpenCV camera-from-world maps with x right, y down, and z
  forward. Intrinsics are in processed-image pixels.
- Supervised canonicalization removes one global Sim(3) from VGGT predictions;
  it assumes at least two non-coincident camera centers and proper rotations.
- Depth is positive camera-z depth, not Euclidean ray distance.
- Confidence calibration is only statistically meaningful after Phase A; SPD
  covariance before calibration is a conservative parameterized model.
- Evidence mass approximates visible world area and is not total object mass.
- MeshFleet voxel PLY vertices are quantized surface samples. They do not
  supervise inside/outside occupancy or signed distance without an additional
  watertight-mesh derivation.
- Surface likelihood adds uniform cell quantization covariance `h^2/12 I` to
  learned evidence covariance.
- DINOv2 patch tokens and TRELLIS structured latents are pseudo labels in
  unrelated learned bases. Only their confidence-weighted overlap-edge cosine
  geometry supervises gauge-invariant chart scalars; no channel equality,
  geometric truth, or topology truth is assumed.

## Sparse transport

- Every retained source and target has at least one support edge. Fallback
  nearest edges guarantee algebraic support but do not imply geometric validity.
- KL marginal relaxations are finite, hence both generalized Sinkhorn exponents
  are strictly below one and the implicit fixed-point solve is contractive.
- The sparse objective is the restriction of the reference measure to retained
  edges. Radius truncation error is measured against dense small-case solves.
- The default visibility term treats each particle as an observed first
  surface. Charts more than two calibrated depth standard deviations behind it
  are penalized; unobserved hidden charts are expected to retain source mass
  rather than consume that target.

## Atlas and charts

- The root bounds contain the intended reconstruction and use a fixed world
  gauge during a discrete stage.
- With an audited dataset AABB, at least one evidence particle must lie inside
  the cube. Out-of-root particles do not initialize cells and are handled only
  as rejectable UOT target mass.
- Active leaves are 2:1 balanced. Chart overlap need only be searched between
  equal or adjacent levels under this assumption.
- Quadratic Monge charts are used only inside a sampled positive immersion and
  overlap margin. Isotropic PCA neighborhoods require frame synchronization.
- Splitting, point assignment, and topology selection are discrete operations;
  gradients do not pass through the selected index itself. Conditional on a
  chosen split/assignment, functional chart refitting retains gradients through
  centers, frames, covariance, curvature, and mass.
- The compact partition-of-unity metric is SPD only when every supplied node
  metric is SPD and weights are non-negative. Queries outside all supports use
  the nearest SPD chart metric and are continuous only away from fallback
  Voronoi boundaries.
- A TRELLIS structure coordinate denotes an occupied canonical cell in the
  same root cube as the atlas. Its empirical posterior is a pseudo-prior, not
  an observed surface sample. Prior-only support carries separately serialized
  mass/count fields and never enters the target marginal of image-to-atlas OT.
- TRELLIS sample votes use a Jeffreys Beta(1/2,1/2) posterior. Fine-cell mass
  variances are summed as an independence approximation inside an atlas cell;
  the configured lower-confidence hazard explicitly discounts this estimate.
  Correlated generative samples can make that variance optimistic and must be
  checked by the server ablation/calibration path.
- For a chart with transported/source mass ratio `r`, conditional image
  moments are shrunk by `r/(r+kappa)`. This assumes low retained mass is weak
  observation support; it prevents entropy-floor transport from relocating a
  hidden prior chart to an unrelated visible particle.

## Topology

- Guarantees apply to the selected finite simplicial complex, not unknown true
  hidden geometry.
- Raw source-mesh connectivity is not topology ground truth until checks for
  valid indices, nondegenerate triangles, edge incidence, purity, closure,
  orientability, and orientation consistency pass. Non-manifold connectivity
  may still supervise rasterized depth/normals and surface geometry.
- Dataset Betti loss requires a true activation mask, non-null target,
  validated/repaired provenance, and confidence in `(0,1]`. Persistent-diagram
  supervision additionally requires an explicit filtration; atlas-stratum
  supervision additionally requires a source-to-atlas complex correspondence.
- The selector's candidate-to-reference persistence energy is an internal soft
  structural prior, not a target extracted from the raw MeshFleet mesh.
- A correct topology is recoverable only if it occurs in the proposed candidate
  family and the evidence/prior energy selects it.
- Betti numbers and persistence are exact over Z2 for the finite complex.
- Every admitted stratum has edge incidence one or two and a consistent face
  orientation. Boundary edges are permitted and penalized; a closed-surface
  guarantee additionally requires a zero boundary-edge count.

## Flow and feasibility

- The initial atlas embedding lies strictly inside every certified constraint.
- Every stacked position tangent is capped by one global positive scale at
  `v_max` and integration runs over unit
  time. The fixed KD-tree broad phase includes every nonadjacent pair initially
  within `d_min + 2 v_max`; triangle inequality therefore excludes collision
  for every omitted pair over the complete flow, not merely one step.
- The safe vector field is locally Lipschitz and the barrier projection is
  solved to its configured tolerance. The implementation rejects a projected
  tangent whose measured active linearized primal margin is below ten dual
  tolerances.
- Nonlinear backtracking accepts only steps with positive measured margins.
- Both the Heun predictor used for field evaluation and the accepted corrector
  are nonlinearly backtracked; the accepted predictor step may be shorter than
  the nominal stage time, so the reference solver is a safeguarded numerical
  integrator rather than an exact fixed-step Heun discretization.
- Fixed connectivity plus injective orientation-preserving embedding implies
  topology preservation; it does not establish semantic correctness.

## Analytical assets

- Every Gaussian mean is an exact chart evaluation and normal offsets default
  to zero.
- Positive chart immersion, positive sampling spacing, and positive normal
  scale imply SPD Gaussian covariance.
- PLY and GLB validity is a serialization claim; visual fidelity remains
  conditional on atlas and appearance quality.
- The per-tile opacity quantity is a conservative bound because it sums peak
  optical depths for every 3-sigma projected bbox overlapping a tile. It may be
  loose; it is not an equality with the rasterized maximum alpha.

## Robust gradients

- Phase-F purification assumes at least two valid rendered views and that a
  reliability-weighted majority has a gradient component aligned with the
  desired static geometry objective. It cannot recover a direction rejected
  by every view.
- Consensus and artifact subspaces are exact for the retained view gradients;
  limiting the set to the eight most reliable views is a bounded-memory
  sampling policy, not an exact statistic over an arbitrarily large view set.
- The empirical Fisher is diagonal and lagged by one optimizer boundary. It is
  a stable normalization metric, not the exact model Fisher information.
- Appearance/segmentation artifact directions are estimated by deterministic
  luminance and one-pixel soft-boundary perturbations with geometry fixed.
  Their removal is a robustness heuristic, while cone membership and Gram
  projection are exact numerical operations on the sampled gradients.
- Phase-F scale hardening solves a one-step signed inner problem in bounded log
  scale. Replaying the saved Torch RNG makes stochastic rounding and sampled
  geometric perturbations identical between the clean and adversarial
  forwards; this is an FGSM approximation, not the exact maximum of a
  nonconvex quantized objective.
- Feasibility hardening margins are dimensionless threshold ratios. Zero is
  the original hard certificate boundary, while a configured value `m>0`
  requests relative slack; it does not change the barrier projector's actual
  admissible set.
- Phase-E activation loss requires identical persistent active atlas rows.
  Local vector and rank-2 activations are lifted through their chart frames,
  so their discrepancy is gauge independent and SE(3) invariant.
- The affine-invariant SPD parallel transport assumes strictly positive
  covariance eigenvalues and the unique principal geodesic. The deterministic
  JVP is an exact directional derivative for its normalized probe but only a
  one-direction estimator of the full vector-field Jacobian discrepancy.
- VGGT track cycles assume aligned OpenCV world-to-camera matrices, positive
  projective depth, and locally correct visibility. Bilinear target-depth
  sampling is differentiable; image-boundary and validity decisions are
  discrete. Depth-derived normals are pseudo-labels detached from VGGT when
  supervising the renderer, and are invalid at finite-difference boundaries
  or degenerate/depth-discontinuous neighborhoods.
- Offline teacher refinement assumes the initialized selected stratum is
  admissible and semantically plausible. Camera corrections are bounded around
  audited initialization; state changes use one tangent at the initialized
  product-manifold point and are shortened until the exact nonlinear barrier
  report is feasible. Confidence is a pseudo-label weight, not a guarantee of
  hidden-surface correctness.
- Learned perceptual supervision assumes a complete torchvision VGG16 feature
  state whose SHA-256 matches configuration. Its parameters are frozen and its
  target branch is detached. No LPIPS calibration is inferred from VGG weights;
  absent external weights select the explicitly weaker fixed feature metric.
- TRELLIS candidate likelihood treats active-cell occupancies as conditionally
  independent Bernoulli variables after Jeffreys smoothing. Combined hazard
  may create/propose hidden support, but UOT-only occupancy is retained for the
  evidence and persistence energies. Correlated structure samples reduce the
  effective prior sample size and require server calibration.
- Sparse image-plane refinement assumes each particle's `view_index` refers to
  the retained aligned OpenCV camera table and that UOT support is a meaningful
  visibility-conditioned correspondence neighborhood. Pixel residuals are
  normalized by the projected cell half diagonal with a one-pixel floor. The
  statistic is a population variance across supported views, so it intentionally
  vanishes for a chart with fewer than two views; geometry, curvature, and
  occupancy criteria still govern such charts.
- Persistence-critical candidate thresholds assume the topology family is the
  lower-star filtration of the current atlas support probability. Birth/death
  endpoints are exhaustive event values for that scalar filtration, not for
  alternate clique connectivities or discrete-Morse matchings. Endpoint
  ranking and candidate truncation are detached discrete choices.
- The quantization topology distance uses the retained barrier constraint
  family and inverse evidence metric at the evaluated state. Its gradients are
  exact only within the current closest-feature and broad-phase stratum. The
  boxed one-step result is conditional on a conservative measured downstream
  vector-field Lipschitz upper bound; local spectral normalization alone is not
  asserted to provide that bound.
- Reprojection cycle norms use a debiased Charbonnier radius with
  `epsilon=1e-8` in the depth tensor dtype. It preserves a zero loss at exact
  closure and a finite zero gradient; residuals far above epsilon retain the
  Euclidean-distance interpretation.
- Observation reliability is evidence provenance, not atlas existence:
  TRELLIS-only hidden-support rows may have exactly zero observation
  reliability. Any row with positive transported evidence mass must have
  strictly positive reliability, and all rows remain below one under the
  finite Beta posterior contract.
- Environment validation assumes every active `requirements.txt` entry is one
  unconditional exact distribution pin. CUDA driver compatibility and external
  VGGT/TRELLIS checkpoint identity are separate recorded preconditions.
- Dynamic corpus discovery assumes object identity is the lowercase 64-hex
  Objaverse-XL SHA-256 serialized by the inspected TRELLIS tools. The default
  primary-union/required-intersection policy defines manifest membership;
  optional modality absence is missing supervision, never a zero target.
  Multiple physical artifacts for the same `(split, modality, object, kind)`
  are ambiguous and are not silently resolved.
- A runtime object subset is selection over an immutable audited manifest, not
  a new corpus definition. Every selected ID must be a valid 64-hex identity,
  must belong to the manifest catalog when a catalog was used, and must pass
  the same split- and phase-relative admission predicate as an unfiltered
  record before any tensor is loaded.
- Released VGGT and TRELLIS tensor conditioning assumes finite floating RGB in
  `[0,1]` with at least one physical view. Alpha compositing and resizing occur
  in the audited MeshFleet loader; neither adapter silently rescales byte
  images, clips outliers, fills invalid views, or imputes a missing channel.
- Distributed launches assume `CUDA_VISIBLE_DEVICES` is an explicit
  scheduler-owned assignment of idle A800s. The logical CUDA indices are the
  post-mask indices reported by PyTorch; the launcher never interprets them as
  physical ordinals. Exact in-phase checkpoint continuation remains
  conditional on unchanged world size, while model-only phase initialization
  may use a different visible rank count.
- Renderer equivalence is conditional on the audited TRELLIS mip-splatting ABI
  and constants: a 16x16 tile, integer pixel samples, `kernel_size=0.1`, alpha
  ceiling 0.99, alpha pruning below 1/255, and early termination below 1e-4
  transmittance. Sorting, tile inclusion, pruning, and termination are discrete
  visibility decisions. Retained contributions remain differentiable, but no
  gradient is claimed across those event boundaries. The CUDA kernel performs
  native FP32 accumulation; the PyTorch path is the transparent mathematical
  reference, not evidence of bitwise CUDA determinism.
- `highest` float32 matmul precision plus disabled CUDA/cuDNN TF32 is part of
  the numerical model, not a performance hint. Checkpoint/inference
  comparability assumes this policy is applied before model execution. BF16 is
  permitted only within the configured VGGT aggregator; its camera, depth, and
  point heads and all GRAFT-GS geometric state execute with autocast disabled
  or explicit FP32 tensors.
- A covariance PCA frame is a gauge-valued state. Its ordinary eigenvector
  derivative is assumed only on the simple-spectrum stratum where both
  adjacent gaps exceed `frame_epsilon + frame_relative_eigengap * spectral_scale`.
  At a repeated or near-repeated spectrum the forward SO(3) frame remains a
  valid source-selected gauge, while its unidentifiable gauge derivative is
  exactly zero; center, covariance, mass and fixed-frame curvature derivatives
  remain active.
- Same-object DDP assumes every rank participates in both the autograd-aware
  global evidence all-gather and source atlas broadcast in identical order.
  Under that collective graph, broadcast backward sums downstream atlas
  derivatives at the source and all-gather backward reduce-scatters evidence
  derivatives to their owning ranks. Raw frame/curvature coordinates need not
  agree before source gauge selection.
- A chart principal frame is differentiable only when the 2D first-form
  eigengap exceeds `metric_epsilon + metric_relative_eigengap * scale`. At an
  unresolved spectrum, tangent directions are gauge-valued: their derivative
  is defined to be zero while the repeated eigenvalue retains the common trace
  derivative. Gaussian covariance is evaluated in the basis-free form
  `a^2 J J^T + sigma_n^2 n n^T`, so this gauge convention cannot change the
  rendered Gaussian measure.
- The SPD shift-and-contract box is a conditional differentiable projection.
  It is the identity strictly inside the eigenvalue interval, continuously
  shifts an under-floor spectrum, and contracts an over-ceiling spectrum.
  At exact min/max multiplicity its directional derivative is a selected
  subgradient; finiteness is guaranteed, but classical differentiability at
  the spectral active-set boundary is not claimed.
- High-precision training assumes every optimized tensor remains finite before
  and after Adam. Gradient clipping is not a recovery operator for NaN/Inf.
  The collective finite gate deliberately aborts all ranks before mutation;
  it never clamps, replaces, or silently skips a non-finite scientific state.
- Single-node DDP assumes `torchrun` assigns one process to each logical device
  exposed by `CUDA_VISIBLE_DEVICES`, so local rank is a valid CUDA index. A rank
  must own no allocator state on its siblings. More views per rank increase
  distinct geometric evidence but do not monotonically improve examples per
  second; the operational choice is therefore empirical and phase-specific.
  Peak reservation should retain headroom for object-dependent octree and UOT
  growth. Occupying all 80 GiB is neither a loss term nor a quality guarantee.
- Pre-flow embedding restoration assumes the active hard constraints satisfy a
  local constraint qualification: their normalized Jacobian must contain a
  separating direction inside the configured cumulative displacement budget.
  The active collision/closest-feature set is discrete, so differentiability
  is conditional on that set remaining fixed. Exact degenerate intersections
  may have a zero squared-distance gradient and are rejected if the sequential
  QP cannot reduce violation; no feasibility guarantee is claimed in that
  case. The fixed broad phase remains complete because every restored vertex
  stays within `maximum_position_speed` of its transported input.
- Sparse implicit UOT assumes positive finite relaxation scales and a support
  covering every source and target node. Zero input mass is represented only
  through the declared `mass_floor`; if exponentiation removes all transported
  mass from a supported row or column, the solve aborts. Implicit gradients are
  valid only after both the primal fixed-point and transposed fixed-point
  equations meet their recorded scale-relative tolerances.
- Transport chunk-size changes preserve the mathematical radius graph provided
  no computed distance lies exactly on the strict support threshold and no
  nearest-neighbor tie changes under floating reduction ordering. The A800
  sweep must rerun sparse-support/plan equivalence before the larger chunks are
  treated as performance-validated.
- Sparse support membership is a discrete active-set operation and has no
  derivative across radius or nearest-neighbor switches. Conditional on a
  fixed support, cost, plan, chart moments, and evidence coordinates retain
  their ordinary gradient; suppressing the support-search `cdist` tape removes
  no mathematically defined gradient used by this architecture.
- A minimum global object batch is realized as
  `WORLD_SIZE * ceil(target/WORLD_SIZE)` independent objects. It may exceed the
  requested minimum when the target is not divisible by the visible rank
  count. Learning-rate invariance is not inferred from batch scaling; the
  configured learning rate remains fixed unless an explicit ablation changes
  it. Same-object multiview sharding does not satisfy this independence
  assumption and therefore cannot activate the policy.
