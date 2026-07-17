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
24. **Derived VGGT tracks and normals.** The released adapter does not expose a
    distinct correspondence or normal head. Rather than inventing tensors, the
    implementation constructs track cycles by depth/camera unprojection and
    derives world normals from adjacent depth rays, with confidence,
    visibility, and explicit pseudo-label provenance.
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
