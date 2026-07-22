# Deviations and strengthening relative to GRAFT-GS.md

1. **VGGT feature width.** The released aggregator caches concatenated frame and
   global features of width 2048, not the document's 1024 tensor. Integration
   uses a traceable orthogonal multiplicity projection rather than silently
   truncating or concatenating downstream features.
2. **Octree plus simplicial complex.** Octree adjacency alone does not define a
   2-manifold or Betti-2. The implementation separates persistent spatial
   indexing from an explicitly selected simplicial surface complex.
3. **Implicit Sinkhorn derivative.** The document states a damped inverse of a
   fixed-point Jacobian but does not define its blocks. The implementation uses
   the exact sparse row/column conditional plan operators in the transposed
   implicit system.
4. **Gauge representation.** For l=2 the implementation transports Cartesian
   symmetric-traceless tensors exactly rather than constructing Wigner matrices.
   This is representation-equivalent and materially easier to verify.
5. **Topology scoring.** A persistence distance alone cannot propose topology.
   The implementation first constructs a finite candidate distribution, verifies
   boundary ranks/manifold incidence, then applies evidence, persistence, and
   shape-prior energies.
6. **Safety.** Log penalties are insufficient for the stated theorem. The flow
   uses control-barrier projection plus nonlinear feasibility backtracking.
7. **Precision.** The native A800 baseline does not use FP4. Geometry and
   analytical solves remain FP32/optional FP64; quantization is a later,
   equivalence-tested deployment phase.
8. **TRELLIS role.** TRELLIS sparse transformer concepts and pretrained priors
   may initialize multiplicity maps/shape energies, but its independent neural
   Gaussian and marching-cubes-style mesh decoders are not on the critical asset
   path.
9. **Collision broad phase.** Literal all-pairs self-collision is quadratic and
   unusable at the target atlas size. A fixed KD broad phase is complete under
   the enforced unit-time speed bound; the retained pairs still receive exact
   differentiable margins, CBF projection, and nonlinear backtracking.
10. **Analytical thickness and opacity.** Relative transported SPD uncertainty
    modulates only the chart-normal thickness; tangent covariance remains fixed
    by the first fundamental form. Flow opacity is interpreted as bounded chart
    optical depth and divided by the chart area Jacobian at readout, preserving
    surface-measure consistency without an unconstrained decoder.
11. **Dataset-grounded gauge handling.** The audited data provides exact
    normalized Blender cameras while VGGT predicts geometry up to a similarity
    gauge. The implementation removes one differentiable scene Sim(3) and
    transforms depths/points/cameras jointly; it does not mix ground-truth
    extrinsics with unaligned predicted depth.
12. **Surface rather than volume supervision.** The TRELLIS voxel PLY is a
    surface raster at 64 cubed cell centers. Losses use quantization-aware
    surface likelihoods and screened topology-fixed projection, never a
    fabricated filled-occupancy or SDF target.
13. **Uncertain source topology.** The canonical source mesh is non-manifold:
    eight components and 313 incidence-four edges. Raw connectivity is kept as
    a diagnostic measure, while hard Betti/persistence/stratum labels remain
    null. Validated repair or a confidence-weighted teacher is a separate
    provenance class; no zero Betti vector is substituted.
14. **Hidden-support prior timing.** A late TRELLIS score over VGGT-created
    leaves cannot propose an unseen surface. TRELLIS structure samples are
    therefore converted to a typed sparse surface measure before atlas
    initialization. This measure may create cells and affect topology hazard,
    but it never becomes image evidence, direct supervision, or an independent
    Gaussian/mesh decoder.
15. **Conditional differentiability of refinement.** Morton creation, split
    masks, point assignments, and topology indices are discrete. Unlike the
    earlier implementation, continuous child chart refitting is differentiable
    conditional on those choices; no straight-through estimator is claimed for
    the discrete index.
16. **Sparse reprojection normalization.** The earlier 3D/depth disagreement
    proxy has been retired. Evidence now retains one calibrated camera table per
    scene, and sparse UOT edges retain camera identity through `view_index`.
    The implementation evaluates the exact view-conditional image residual and
    reports its population variance after division by the projected octree-cell
    half diagonal. This dimensionless normalization strengthens the unspecified
    threshold units in the Markdown. A chart observed by only one view has zero
    cross-view variance by definition and remains governed by the other three
    split criteria.
17. **Continuous metric realization.** The smooth metric is a compact bump
    partition-of-unity weighted sum of SPD chart metrics. This preserves SPD
    and SE(3) covariance and is exercised by analytical readout; node-based flow
    still evaluates the metric at its selected chart vertices.
18. **Perceptual image term.** No pinned/audited LPIPS checkpoint is supplied.
    The reference objective uses deterministic multiscale color and gradient
    features and is reported as a fixed perceptual surrogate, never as LPIPS.
19. **Topology proposal family.** Exact lower-star persistence birth/death
    endpoints, ranked by lifetime, now supply the primary occupancy cuts;
    object-adaptive quantiles and fixed cuts are deterministic coverage
    fallbacks. This exhausts topology changes along the sampled scalar
    filtration only within the retained candidate budget. It remains a finite
    structured distribution, not a claim of enumerating all discrete-Morse
    matchings or alternate simplicial connectivities.
20. **Low-bit training.** Phase-E fake quantization uses a straight-through
    gradient and must not be described as a physically matched FP4 backward
    kernel. Native BF16/FP32 remains the uncompromised baseline.
21. **Matrix-free gradient purification.** The specification writes an SVD of
    a full parameter-by-view gradient matrix. The implementation obtains the
    same retained left-singular subspace from its small weighted view Gram
    matrix and applies it as gradient linear combinations. This avoids a dense
    billion-parameter matrix without weakening the sampled-view projection.
22. **Dimensionless topology-margin hardening.** A single additive `m_h` is
    dimensionally invalid across area, squared separation, cosine, and
    covariance constraints. Phase F instead requires the same positive
    dimensionless relative slack for each independently normalized hard
    certificate margin.
23. **Jacobian distillation by manifold JVP.** Materializing the complete
    vector-field Jacobian is quadratic in atlas state. The implementation
    matches an exact directional derivative along one deterministic normalized
    product-manifold probe, with all teacher factors parallel-transported into
    the student tangent space. This is scalable but does not identify every
    singular direction of the Jacobian.
24. **Derived VGGT tracks and normals.** Released VGGT has a query-conditioned
    track head, but it does not expose the Markdown's asserted
    `[B,K,1369,256]` per-patch track-descriptor tensor and has no normal head.
    Densely querying every patch would add a separate iterative correlation
    solve and would still not produce the specified descriptor contract.
    Rather than inventing that tensor, the reference path constructs dense
    track cycles by depth/camera unprojection and derives world normals from
    adjacent depth rays, with confidence, visibility, and explicit
    pseudo-label provenance. The untouched baseline script remains available
    for query-conditioned upstream tracking experiments.
25. **Deployment-safe upstream discovery.** Production code imports installed
    `vggt`/`trellis` packages or explicit server checkout roots; it contains no
    developer-workstation path or sibling-repository assumption. Checkpoints
    resolve in the order CLI, `GRAFT_GS_*_CHECKPOINT`, legacy upstream
    environment variable, then the released model-hub identifier. This is an
    execution-boundary clarification rather than a mathematical change.
26. **Source-only TRELLIS sampling in same-object DDP.** The structured prior
    is frozen and sampled under `no_grad`; sampling it independently on every visible
    ranks cannot add a gradient. Same-object mode therefore gathers every view,
    samples only on the designated source rank, and broadcasts the typed
    probability/mass/variance measure before atlas construction. Object-level
    DDP remains rank-local because its ranks contain different objects.
27. **Exact frozen-prior reuse.** TRELLIS structure sampling is deterministic
    for fixed checkpoint, tensor conditioning, sampler policy, and seed. A
    bounded LRU hashes the exact float tensor bytes and all sampler identity
    fields, then stores only integer coordinates. This is exact memoization,
    not latent quantization or an approximate learned substitute.
25. **Rendering-based offline bundle adjustment.** The specification writes
    sparse reprojection correspondences `u_kj`, but the released local contract
    provides no persistent cross-view track IDs. The teacher therefore refines
    bounded cameras and certified atlas state through robust differentiable
    rendering plus derived depth cycles, preserving the same geometric closure
    objective without fabricating correspondences.
26. **Perceptual checkpoint boundary.** The repository cannot validate or
    redistribute a learned perceptual checkpoint. It implements a strict local
    hash-pinned frozen VGG16 path and otherwise retains the deterministic
    multiscale surrogate; neither path is mislabeled as trained LPIPS.
27. **Factorized TRELLIS shape likelihood.** The specification leaves
    `p_shape(tau)` abstract. The implementation uses the sampled TRELLIS
    posterior as a Jeffreys-smoothed independent-cell Bernoulli likelihood for
    each explicit candidate, while keeping observed UOT evidence in a separate
    energy. This is calibrated and traceable but omits posterior cell
    correlations.
28. **Explicit Lipschitz provenance in the quantization certificate.** Local
    spectral parametrization does not by itself prove the downstream
    vector-field constant `L_v` once attention, tensor products, and manifold
    heads are composed. The implementation computes the metric topology margin
    exactly for the retained constraint stratum, but requires `L_v` to be
    supplied as an explicit server measurement. If it is absent, inference
    emits no certificate rather than substituting a configuration heuristic.
29. **Strict pinned-environment gate.** The mathematical specification does
    not define Python package identity, but the production reference harness
    now rejects missing or mismatched versions before importing the model and
    records the `requirements.txt` SHA-256. This is an execution-contract
    strengthening, not evidence that dependency identity alone validates the
    numerical architecture.
30. **Physical modality intersection replaces enumerated dataset membership.**
    The specification assumes an object dataset but does not define how a
    partially completed, modality-centric preprocessing corpus determines
    membership. The implementation uses a primary-modality union followed by
    a configurable required-modality intersection, records optional absence,
    and retains rejected candidates separately. An optional ID file can narrow
    the result but is never the default source of truth.
31. **TRELLIS mip-EWA is the canonical rendering measure.** The specification
    requires a differentiable Gaussian renderer but does not fix pixel-center,
    antialiasing, or tail-pruning conventions. The implementation adopts the
    actual TRELLIS mip-splatting equations for both reference and optimized
    paths, passes analytical SPD covariance directly, and rejects a generic
    graphdeco ABI rather than silently using different low-pass semantics.
    The determinant peak factor preserves integrated screen-space Gaussian
    measure under the mip filter; discrete tile/pruning behavior remains an
    explicit approximation boundary.
32. **TF32 is rejected for the native reference.** Although A800 TF32 would
    improve throughput while leaving tensor dtypes labeled FP32, its shortened
    product mantissa is not accepted for sparse transport, SPD, topology, or
    barrier decisions. The executable YAML policy disables TF32 and is stored
    in format-6 checkpoints. This is stricter than the Markdown's broad mixed-
    precision allowance and intentionally prioritizes reconstruction fidelity.
33. **Decoded TRELLIS occupancy resolution is runtime-observed.** The released
    flow model's `resolution=16` is a latent-domain size, whereas its decoder
    produces the 64-cubed sparse-coordinate domain. GRAFT-GS observes the
    decoder tensor extent on each posterior draw and rejects non-cubic or
    inconsistent outputs. It does not hard-code the released checkpoint's 64
    and does not infer a grid from the largest occupied coordinate, which would
    contract shapes whose sampled support does not touch the boundary.
34. **Discrete DDP state has an explicit NCCL wire dtype.** Persistent atlas
    storage retains compact int16 levels, int8 child slots, Boolean activity,
    and int64 identities, but Torch 2.4 NCCL does not accept `Short`. Collective
    transport therefore uses an independent contiguous int64 representation
    with exact restoration.
35. **Same-object DDP selects one nonlinear chart gauge.** PCA frames are not
    unique tensors: eigenvector signs and repeated-eigenspace bases are gauge
    choices. Continuous atlas state therefore uses an autograd-aware source
    broadcast instead of a local straight-through surrogate. All rank losses
    reduce through the source atlas and the differentiable global-evidence
    gather returns their derivative to each evidence shard. Raw equality is
    required only for gauge-independent atlas fields.
36. **PCA derivatives are eigengap-stratified.** The Markdown treats chart
    frames as smooth SO(3) state, but an eigenframe is not differentiable at a
    repeated covariance spectrum. The implementation differentiates it only
    when both adjacent relative eigengaps are resolved; the unidentifiable
    gauge has a finite zero derivative otherwise. This is a conditional
    derivative guarantee, not a claim of smoothness across spectral strata.
37. **Analytical tangent covariance is basis-free.** The specification writes
    tangent scales in principal first-form coordinates. The implementation
    still exports those exact scales and a valid representative rotation, but
    computes the covariance as `a^2 J J^T + sigma_n^2 n n^T`. This is
    algebraically equivalent and removes an unnecessary, undefined
    eigenvector derivative at flat/isotropic charts.
38. **Non-finite optimization state is a hard distributed error.** The
    specification requests high precision but does not define recovery from a
    NaN update. GRAFT-GS does not sanitize such values: it collectively checks
    losses, gradients, parameters, and Adam state, then aborts before mutation
    with rank/name provenance. This favors scientific diagnosability over a
    silent skip or zero replacement.
