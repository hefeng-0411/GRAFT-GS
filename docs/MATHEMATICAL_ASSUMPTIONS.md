# Mathematical assumptions and validity domain

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
