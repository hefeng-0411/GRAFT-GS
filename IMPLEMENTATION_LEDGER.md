# GRAFT-GS implementation ledger

Target runtime: enterprise Linux server with up to 6x NVIDIA A800 80 GB;
the scheduler-visible subset is authoritative. Native FP32
geometry and BF16/FP16 backbone execution are the baseline. The local RTX 2060
is an editing environment only; no numerical result is inferred from it.

## Implemented data flow

1. Released VGGT aggregation, camera/depth/point heads, four 2048-wide cached
   taps, deterministic 2048-to-1024 projection, and optional late-block LoRA.
2. OpenCV camera-from-world unprojection into area-weighted evidence particles;
   learned log-confidence calibration and ray-aligned SPD covariance.
3. Persistent adaptive octree with stable `(level, Morton)` identity, retained
   ancestors, measured residual covariance, weighted PCA SO(3) frames,
   quadratic Monge charts, actual variance/curvature refinement, 2:1 closure,
   sparse overlap connection, and variable-size checkpoint restoration.
4. Chunked sparse support with source/target coverage; fused Mahalanobis,
   ray/depth, visibility, and fixed-point feature costs.
5. Log-domain KL-unbalanced Sinkhorn with a custom implicit backward solve,
   exact sparse primal objective, barycentric centers/mass/color, and
   transport-induced SPD metrics.
6. Gauge-covariant chart writing in the exact
   `48(0e)+12(0e)+16(1o)+4(2e)=128` layout.
7. Connection-aware sparse transport attention with exact Cartesian `l=2`
   symmetric-traceless transport, parity-valid contractions, sparse segmented
   softmax, uncertainty/OT biases, and spectral-normalized multiplicity maps.
8. Finite topology candidates from evidence-filtered surface complexes,
   manifold edge-incidence checks, exact Betti numbers and Z2 persistence,
   piecewise-differentiable persistence coordinates, energy distribution, and
   lowest-energy hard-feasible stratum selection.
9. Product-manifold state and operations on
   `R3 x SO(3) x SPD(3) x R x appearance x latent`; stable SO(3) log/exp,
   affine-invariant SPD geodesics/retractions, conditional Riemannian flow
   matching, and manifold Heun integration.
10. Hard feasibility with speed-bounded KD broad phases for nonadjacent vertices
    and faces, exact piecewise triangle distance, area/orientation/SPD margins,
    active CBF metric projection, and nonlinear backtracking.
11. Deterministic area-adaptive chart sampling; exact surface-attached means;
    first-form/curvature/uncertainty covariance; measure-consistent bounded
    opacity; weighted ridge degree-3 SH; shared atlas mesh; binary 3DGS PLY and
    glTF 2.0 GLB serialization.
12. A transparent differentiable PyTorch renderer and a server CUDA-rasterizer
    adapter guarded by numerical equivalence tests.
13. Optional TRELLIS multi-image sparse-structure sampling as a hidden-surface
    occupancy prior; TRELLIS asset decoders do not enter the final path.
14. Six-stage A-F trainer, native precision, object-level DDP, explicit
    same-object view sharding, autograd-aware global atlas measure and
    continuous-state equality checks, rank-zero discrete decisions, complete
    autograd evidence gather and replicated global UOT, LoRA, Phase-E INT8
    QAT/distillation, Phase-F robustness/topology hardening, atomic checkpoints,
    exact epoch/microstep/in-epoch resume,
    and cross-phase parametrization-key translation.
15. Server entry points for untouched baseline reproduction, inference,
    deterministic assets/renders/metrics, one-object overfit with a required
    loss reduction, visible-GPU training, validation, profiling, and structural
    ablations.
16. Audited MeshFleet/TRELLIS object contract with a deterministic JSONL
    manifest, physical/declaration view reconciliation, relational sparse-grid
    checks, direct/derived/pseudo-label provenance, black RGBA composition,
    separate continuous alpha and boolean evidence masks, exact
    Blender/OpenGL-to-OpenCV conversion, and configurable object/view sampling.
17. Differentiable scene-level Sim(3) canonicalization transforms VGGT cameras,
    depth, and world points jointly. Stable camera residuals, direct surface
    Chamfer, quantization-aware evidence covariance NLL, confidence Brier loss,
    and screened/barrier-certified Phase-C surface targets are wired into the
    staged objective.
18. MeshFleet DDP view sharding keeps images, alpha, masks, intrinsics, and
    extrinsics index-aligned; validation metrics are globally reduced; epoch
    view sampling and checkpoint/validation cadence are deterministic.
19. Phase-aware checkpoint loading reconstructs LoRA and QAT parametrizations,
    validates model configuration, and records manifest digest/split/view set.
    Checkpoint-required MeshFleet inference and one-object overfit entry points
    use the audited cameras and surface targets.
20. Raw render-mesh topology receives a deterministic finite-complex audit:
    component/V-E-F/Euler counts, complete edge-incidence histogram,
    boundary/non-manifold/isolated/degenerate checks, watertightness, raw
    orientation consistency, orientability where defined, and hard-label
    admissibility. Typed raw/validated/repaired/derived/teacher/unavailable
    states propagate to the sample contract. Provenance/confidence masks gate
    hard expected-Betti loss; internal persistence prior and teacher
    pseudo-topology are separately weighted.
21. Non-manifold source connectivity remains usable as triangle-soup geometry:
    an optional A800 nvdiffrast target path derives exact-camera visibility,
    camera-z depth, and camera-frame source normals. Gaussian rendering now
    emits analytical camera-frame atlas normals; log-depth and unoriented
    normal losses preserve geometry supervision without asserting manifold
    topology.
22. Same-object DDP now autograd-all-gathers positions, rays, features, SPD
    covariance, confidence, mass, colors, and view identity, then replicates one
    mathematically global sparse UOT solve. The previous sum of rank-local nonlinear
    local UOT barycenters remains only as a rejected legacy approximation.
23. Phase A is a genuine evidence-only stage: it stops before atlas/topology/
    flow construction and trains only confidence plus ray-aligned covariance
    scales using quantization-aware surface likelihood and Brier calibration.
    Sparse transport cost learning begins in Phase B. Checkpoints restore
    sampler epoch, accumulation microstep, in-epoch position, and verify the
    dataset manifest digest.
24. The audited canonical AABB is the persistent octree root. Evidence outside
    the cube remains in UOT for unbalanced rejection but cannot be clamped into
    atlas initializer cells; rejected particle count and mass are reported.
25. Crash-integrity recovery recompiled every package/script/test, regenerated
    the audited MeshFleet manifest byte-for-byte, and traced topology policy
    through the manifest, object sample, collation, provenance-weighted Betti
    objective, and A--F loss schedule. No partial write or contract divergence
    was found: the canonical object keeps all geometry supervision while every
    hard source-topology mask and nullable target remains inactive.
26. TRELLIS structure samples now form a Jeffreys-smoothed Beta-Bernoulli sparse canonical
    surface measure before atlas construction. Their union can create genuinely
    hidden persistent leaves, while `evidence_mass`/`point_count` and
    `prior_mass`/`prior_point_count`/posterior mass variance remain disjoint through refinement,
    checkpoints, and DDP synchronization. Same-object DDP gathers every view
    shard for multi-image TRELLIS conditioning before broadcasting the sampled
    sparse prior. Image particles remain the only UOT
    target marginal. Low-retention conditional centers, irreps, metrics, and
    colors shrink toward the chart/uninformative state; topology combines
    observed and lower-confidence prior surface hazards without allowing absent prior support to
    erase observed geometry. The same combined occupancy initializes optical
    depth on the selected manifold state, so hidden charts are not discarded at
    analytical readout. A800 scripts record the external TRELLIS checkpoint
    and sampling policy and include an explicit no-hidden-prior ablation.
27. Audited DINOv2 surface tokens and TRELLIS structured latents now enter
    Phases B/D/E/F only through confidence- and provenance-gated relational
    distillation. Exact verified sparse coordinates assign pseudo features to
    persistent charts; the loss matches overlap-edge cosine kernels against
    gauge-invariant `0e` fields. It never equates unrelated learned channel
    bases, concatenates tokens, or promotes either pseudo-label family to
    geometry/topology truth. Enablement and confidence are checkpointed as part
    of the dataset/training contract.
28. The repository-wide specification audit is recorded in
    `docs/SPECIFICATION_TRACEABILITY.md`. Production GSTA now receives
    conditional sparse-OT cost and reliability uncertainty on the exact active
    adjacency; the default transport cost includes an uncertainty-normalized
    one-sided visibility barrier instead of a zero fallback. Occupancy entropy
    and conditional residual/depth variance now drive the two previously dead
    octree split criteria.
29. Octree split indices remain discrete, but post-split continuous chart
    fitting is no longer detached. Functional indexed writes preserve evidence
    gradients through centers, frames, covariance, curvature, and masses.
    Smooth compact partition-of-unity metric evaluation, overlap `C0/C1`,
    world-curvature, and persistent parent/child objectives are integrated into
    Phase B and later losses.
30. Topology candidates are consistently oriented by an explicit Z2 face
    constraint solve, carry incidence/orientation validity, and are rejected
    before manifold construction if either check fails. Object-adaptive
    filtration quantiles augment fixed proposal thresholds. Flow interpolation
    additionally requires identical persistent node, edge, and face strata.
31. Barrier projection now uses one global positive speed rescaling, verifies
    the solved linearized primal margin, and rejects an unconverged QP. Safe
    Heun evaluates its second field sample only at a nonlinearly feasible,
    backtracked predictor. These changes repair two violations of the stated
    conditional safety assumptions.
32. Phase execution is isolated: Phase B skips continuous flow, Phase C runs
    constrained flow but stops before analytical asset construction, and D--F
    retain the complete path. Phase-C targets receive exact minibatch Hungarian
    OT coupling within compatible topology strata.
33. Analytical readout allocates Gaussians from deterministic curvature-aware
    surface-area quadrature and uses the continuous partition-of-unity evidence
    metric in uncertainty thickness. A conservative projected-tile optical-depth
    upper bound is trained explicitly. GLB now contains a deterministic PBR
    material consuming atlas-derived vertex colors.
34. The image objective now contains foreground-aware robust RGB, SSIM, a
    documented fixed multiscale color/gradient perceptual surrogate, mask BCE,
    VGGT depth reprojection, direct surface likelihood, and audited mesh
    depth/normal targets. Loss weights are explicit in server configuration and
    checkpoint format 5 refuses objective drift.
35. Phase-E distillation now includes SO(3) body-log state error, generalized
    KL for unequal-mass transport measures, rendered color/alpha/depth/normal,
    and product-metric vector-field matching with angular-velocity frame
    transport. TRELLIS sampling uses stable object-derived seeds shared by DDP
    ranks and teacher/student.
36. Trainer checkpoint format 5 stores and restores each rank's independent
    Torch/CUDA/NumPy/Python RNG stream, records world size and loss weights, and
    refuses an exact-resume claim across a different world size. Legacy
    checkpoints remain loadable but do not establish exact multi-rank replay.
37. Phase-F view-conditioned gradients now execute the specification's robust
    path in the production trainer: a smoothed weighted geometric median,
    exact circular-cone projection, Gram-space consensus SVD, deterministic
    appearance/segmentation artifact rejection, and diagonal empirical-Fisher
    clipping. Global transport/atlas/topology/barrier gradients bypass the
    purifier. Purified gradients are reduced after projection under DDP, and
    synchronized Fisher state is part of exact format-5 resume.
38. Image supervision now distinguishes object foreground alpha `[B,K,1,H,W]`
    from view availability `[B,K]`; missing alpha no longer misinterprets a
    view-validity vector as a spatial mask.
39. Phase F now performs a one-step inner maximization over every active QAT
    block's bounded log scale. It differentiates the clean hardening objective
    with respect to scale, applies the worst-case signed radius, restores the
    exact pre-forward Torch RNG state, and recomputes the production path with
    identical stochastic rounding and geometry perturbations. Scale buffers
    are reset before optimizer state or checkpoints are serialized.
40. Feasibility training now uses dimensionless hard-certificate slack ratios
    for face area, orientation, squared separation, and covariance eigenvalue
    bounds. Phase F requires a configurable positive relative slack instead of
    combining physically incompatible margins under one scalar temperature.
41. Phase-E distillation now captures the input and every output of the
    production gauge-covariant encoder. Scalars are compared directly while
    vectors and rank-2 irreps are compared in world tensors. It additionally
    matches a deterministic product-manifold vector-field JVP; teacher probes
    and outputs are parallel-transported through SO(3) body gauges, the
    affine-invariant SPD connection, and packed latent irreps before the
    student product metric is evaluated.
42. VGGT-derived multiview supervision now includes deterministic sparse
    depth/camera reprojection cycles and confidence-weighted world normal
    fields from neighboring unprojected pixels. The cycle remains
    differentiable to VGGT depth/cameras; normal targets are detached when
    supervising rendered normals. These are typed derived pseudo-targets and
    do not fabricate absent track or normal heads in the adapter.
43. Offline teacher construction is now an executable topology-fixed bundle
    adjustment: bounded camera extrinsics/intrinsics and one product-manifold
    atlas state are robustly optimized, every state update is nonlinearly
    retracted through the hard BarrierProjector, and the production analytical
    readout generates both PLY and GLB. Bundle confidence combines final
    reprojection RMSE, topology entropy, and track-cycle residual.
44. Refined teacher bundles use schema/identity/manifest/checkpoint provenance.
    MeshFleet loading exposes explicit availability/confidence/provenance masks;
    only admitted bundles populate Phase-C target states, and flow losses are
    confidence weighted. Serialized direct manifold targets retain a distinct
    direct-target provenance.
45. Foreground/image supervision accepts both object-level view availability
    `[B,K]` and the audited MeshFleet spatial evidence mask `[B,K,1,H,W]`.
    Track cycles now retain source and bilinearly sampled target pixel validity
    rather than collapsing a spatial mask prematurely.
46. Learned perceptual supervision now has a strict optional production path:
    a local torchvision VGG16 state is SHA-256 verified, feature-layer
    completeness checked, frozen, ImageNet normalized, and applied with
    downsampled foreground masks. No download API is invoked. The checkpoint
    path/digest is part of exact resume; without it the documented fixed
    multiscale surrogate remains active and is not called LPIPS.
47. Topology selection now keeps TRELLIS shape probability mathematically
    separate from observed UOT evidence. The combined observed/prior hazard is
    used only to propose support; evidence likelihood and reference persistence
    use observed occupancy, while each candidate receives an explicit
    Jeffreys-smoothed Bernoulli `-log p_shape(tau)` over active cells with its
    own learned positive weight.
48. Evidence particles now retain one OpenCV camera table per scene, with
    exact `view_index` provenance and autograd-aware same-object DDP gathering.
    Adaptive octree refinement groups sparse UOT edges by chart/view, compares
    plan-conditional observed pixel barycenters with calibrated projections of
    transported chart centers, and uses their cross-view population variance
    in projected-cell units. This removes the previous conditional 3D/depth
    disagreement proxy without duplicating camera matrices per edge.
49. Topology support proposals now prioritize occupancy thresholds induced by
    birth/death endpoints of the exact lower-star persistence diagrams, ranked
    by feature lifetime. Quantile and fixed thresholds remain deterministic
    coverage fallbacks. This focuses the bounded candidate budget on filtration
    events capable of changing homology while retaining hard manifold and
    orientability rejection before flow.
50. The bounded-score quantization certificate now computes its topology
    boundary distance from actual barrier constraints and the inverse evidence
    metric, evaluating scalar piecewise gradients without materializing the
    full constraint Jacobian. Inference records score, field, step-displacement,
    and geometric-margin terms and accepts only an explicitly supplied measured
    vector-field Lipschitz bound. MeshFleet inference now uses `no_grad` instead
    of `inference_mode`, preserving the barrier's internal certified JVP/Jacobian
    path without retaining the ordinary model graph.
51. `FlowConfig.spectral_bound` is no longer dead metadata: every spectrally
    parametrized irrep multiplicity map in the Riemannian vector field applies
    the configured positive operator scale without mixing magnetic components.
    Attention/flow configuration domains are validated, and the fixed-topology
    ablation now disables both persistence-critical and adaptive-quantile cuts
    rather than accidentally retaining the newly strengthened proposal path.
52. The first pinned A800 reference run exposed four numerical contracts and
    one orchestration gap. Persistent root cell sizes now broadcast a tensor
    scale instead of passing it as `full_like`'s Python-number argument, so the
    continuous root-bound gradient is retained under PyTorch 2.4. Exact-zero
    reprojection cycles use a debiased Charbonnier norm with finite derivative;
    quantization certificates inherit the float64 geometric-margin dtype; and
    hidden-prior-only atlas rows explicitly permit zero observation reliability
    while observed rows remain positive. Server validation now audits all 444
    exact requirement pins before importing the model, binds the verified
    multi-object remote root, regenerates missing/stale manifests, dynamically
    selects an admitted validation object, and treats unexpected dataset/backend
    skips as failures.
53. Remote manifest handoff is now a typed pre-model contract rather than a
    path-existence check. `validate_server.py` parses the v2 summary and every
    JSONL identity record once, checks resolved dataset root, schema, declared
    versus physical record count, and unique valid 64-hex object identities.
    A missing, malformed, stale, wrong-root, count-drifted, or identity-invalid
    manifest forces deterministic full regeneration; a compatible multi-object
    manifest is reused. Validation applies the production admission predicate
    and a documented deterministic selection policy rather than assuming the
    first JSONL record or embedding a schema-object ID.
54. Reference validation now probes the accelerator in a subprocess after the
    exact package and `pip check` gates but before dataset/model import. The
    structured record includes PyTorch/CUDA versions, visible-device count,
    names, compute capabilities, and physical memory. The native baseline
    rejects non-CUDA, non-CUDA-11.8, non-BF16, empty-device, and non-A800 paths;
    this prevents a locally convenient GPU or mismatched CUDA build from being
    reported as the A800 reference environment.
55. The distributed validator now establishes the scheduler-visible execution contract
    before testing global evidence: it audits exact dependencies on every
    process, binds each NCCL rank to its local CUDA device, gathers hostname/
    local-rank/device identity, rejects duplicate device assignments or a world
    size different from `torch.cuda.device_count()` after masking, and applies
    the CUDA-11.8/BF16/A800 gate to all gathered ranks. Rank zero serializes preflight and per-rank test completion;
    a global MIN reduction prevents one passing rank from masking another
    rank's failed unittest suite.
56. The visible-GPU phase launcher no longer resolves an arbitrary `torchrun` from
    `PATH`. It requires the configured CRAFT interpreter (defaulting to the
    verified remote conda path), audits all exact pins before every training
    launch, records that audit, and enters `torch.distributed.run` through the
    same interpreter. Phase A-F commands therefore share the validated Python
    environment by construction.

## Generated verification paths

- camera convention and unprojection/reprojection;
- Morton identity, persistence, 2:1 balance, chart Jacobian/immersion, measured
  refinement state, and atlas checkpoint round-trip;
- sparse/dense UOT agreement, fixed-point residual, independent KKT
  stationarity, implicit finite gradients, and `gradcheck`;
- global SE(3) invariance/equivariance and local SO(2)/SO(3) gauge covariance;
- exact finite Betti/persistence and persistence-value gradients;
- SO(3), SPD, geodesic, retraction, area, orientation, vertex collision,
  triangle crossing, broad-phase completeness assumption, and speed cap;
- analytical Gaussian SPD/rotation/opacity, exact non-floating centers, state
  uncertainty/opacity gradient paths, renderer backward to evidence;
- deterministic PLY/GLB bytes and independent reload;
- reference/CUDA renderer agreement;
- model/trainer checkpoint round-trip;
- quantization error/topology-step certificate;
- one-object overfit and checkpoint-backed real multiview inference.
- deterministic MeshFleet manifest rebuild; physical/missing view accounting;
  sparse coordinate equality; OpenGL/OpenCV axes; Sim(3) gauge removal;
  quantization-aware likelihood gradients; screened target normal equations;
  audited-object loading and camera-aligned collation.
- canonical raw topology audit (eight components, 313 incidence-four edges),
  hard-label rejection, null-target enforcement, policy masks, and distinct
  internal-prior versus teacher-pseudo topology paths.
- evidence-only Phase-A backward; fixed canonical atlas root and outlier
  rejection; triangle-soup depth/normal targets; global-evidence multi-rank
  autograd; exact resume state and manifest guard.
- sparse TRELLIS posterior voting, prior-only atlas leaves, strict separation
  from the observed UOT marginal, retention shrinkage, prior checkpoint
  round-trip, and observed-occupancy monotonicity.
- coordinate-verified DINO/TRELLIS relational pseudo-label loss, explicit
  provenance/confidence rejection, and gradient to gauge-invariant chart
  scalars.
- static guards for production OT/uncertainty attention wiring, differentiable
  refined charts, stage isolation, hard topology/barrier checks, format-5
  per-rank checkpoint state, and the Phase-F purifier;
- post-refinement atlas gradients; partition-of-unity SPD/SE(3) covariance;
  overlap and multilevel gauge invariance; orientability rejection; CBF primal
  margin; curvature-adaptive Gaussian counts; tile-opacity upper bound;
  SSIM/fixed-perceptual failure cases; unbalanced distillation KL; compatible-
  stratum minibatch coupling; PBR GLB reload.

## Validation status

- Python parsing/bytecode compilation: performed locally after each major
  source increment; final result is recorded in `VALIDATION_LEDGER.md`.
- MeshFleet static manifest/topology tests: 6/6 passed locally; a rebuilt manifest was
  byte-identical to the checked manifest and all recorded relational checks
  passed.
- Numerical unit/invariant/gradient suite: generated, A800 execution pending.
- Untouched VGGT/TRELLIS baseline artifacts: server execution pending.
- Real renders, PLY, GLB, logs, checkpoints, wall time, and CUDA memory:
  server execution pending; no values have been fabricated.

## Current external blockers and limitations

- This task has no connection to the enterprise A800 server or its checkpoint
  paths, so checkpoint-backed execution and profiling cannot be performed here.
- The supplied `MeshFleet_TRELLIS/train` tree contains no objects and `test`
  contains exactly one. Cross-object schema variation and training statistics
  are externally blocked until the complete dataset is mounted.
- No Python 3.10+ local runtime with PyTorch is available. CPU mathematical
  tests are generated and syntax checked; only PyTorch-independent dataset
  tests can execute locally without violating the no-install directive.
- The local TRELLIS environment lacks optional compiled `flexicubes`; per the
  execution directive it was neither downloaded nor built.
- The CUDA Gaussian adapter remains implemented but numerically unvalidated
  until the server extension is available.
- Topological correctness is conditional on the correct surface complex being
  present in the finite proposal family. Exact homology of a candidate does not
  prove semantic correctness relative to an unseen true object.
- The collision certificate is exact for retained piecewise triangle distances
  under the configured unit-time speed bound; closest-feature switches are
  nonsmooth strata and dense contact neighborhoods can make the reference dual
  solve expensive.
- Octree split indices and point assignments are discrete. Conditional on those
  choices, post-split chart centers, frames, covariance, curvature, and masses
  retain continuous gradients to evidence. No gradient is claimed through the
  Boolean split or topology-candidate index.

## Numerical assumptions

The full conditional validity domain is maintained in
`docs/MATHEMATICAL_ASSUMPTIONS.md`; architectural choices and deviations are in
`docs/RESEARCH_DECISIONS.md` and `docs/DEVIATIONS_FROM_SPEC.md`.

## 2026-07-17 dynamic modality-intersection discovery

- Requirement: discover every complete object in the remote `train` and
  `test` modality trees without using an example ID or a fixed ID catalog.
- Production path: `scripts/build_meshfleet_manifest.py` ->
  `build_meshfleet_manifest` -> `MeshFleetObjectDataset` -> staged
  training/inference/evaluation entry points.
- Replaced per-object recursive path searches with one deterministic artifact
  index per split and modality. The index follows the inspected TRELLIS
  serializers, including model-nested
  `features|latents|ss_latents/<model>/<id>.npz` leaves.
- Candidate IDs are the union of the configured primary modalities
  (`latents`, `mesh_normalized`). The default required intersection is
  `renders`, `latents`, and complete `mesh_normalized` artifacts. Conditional
  and evaluation renders, features, structure latents, and surface voxels are
  optional at discovery time and remain available to stronger phase-specific
  admission policies.
- Every admitted record now carries available/missing-optional modalities and
  an exact relative structural map. Rejected candidates are retained in a
  deterministic `.rejected.jsonl` inventory with explicit missing required
  modalities; summary counts distinguish candidates, admitted records, and
  rejections for each split.
- Multiple optional model variants are never chosen by lexical accident: all
  paths remain in the structural map, the ambiguous decoded modality is
  omitted with a warning, and required ambiguity rejects the candidate.
- Dataset paths remain root-relative in the manifest and are resolved through
  a containment-checked configurable root at load time. The loader also
  rejects a manifest summary belonging to a different resolved root.
- The optional `id.txt` catalog remains an explicit allowlist feature only; it
  is not enabled by the training config, manifest builder, evaluator, or
  server validator by default.
- Server manifest reuse now additionally requires the modality-centric policy,
  required-modality intersection, valid 64-hex identities, valid split names,
  consistent split counts, and discovered-ID digest. No canonical object or
  record ordering is assumed.

## 2026-07-17 released-model integration and remote-only deployment

- Requirement: use the remote full corpus and default cached upstream models
  without embedding local dataset/checkpoint/repository paths.
- Production path: all training, inference, evaluation, overfit, ablation,
  teacher-refinement, and baseline entry points now resolve checkpoints as
  explicit CLI value, GRAFT-GS environment value, compatible legacy upstream
  value, then official model-hub ID. Package import prefers explicit roots,
  then the physically present declared server checkouts, with an installed
  package as the portable fallback and module-origin provenance throughout.
- Verified against upstream source: VGGT aggregator cached taps
  `{4,11,17,23}`, concatenated width 2048, camera/depth/point head signatures,
  OpenCV pose conversion, and `[0,1]` input domain; TRELLIS tensor conditioning,
  `cond/neg_cond`, multi-image sampler injection, `[batch,x,y,z]` structure
  output, structure-flow resolution, and `Pipeline.to` contract.
- Repaired TRELLIS posterior sampling so each sequential draw owns fresh
  upstream injection state. Added finite/floating/[0,1] tensor validation and
  explicit `cond`/`neg_cond` contract checks.
- Added a bounded LRU for the frozen deterministic prior keyed by SHA-256 of
  exact conditioning tensor bytes, dtype/shape, seed, sample count, and sampler
  steps. It reuses only identical integer support and therefore removes
  duplicate Phase-E teacher/student and Phase-F replay work without changing a
  numerical result; changed views or perturbations cannot alias by policy.
- Same-object DDP no longer executes the frozen TRELLIS sampler redundantly on
  every visible rank. All views are gathered, source rank samples, and a dtype-preserving
  probability/mass/variance/vote measure is broadcast before every rank builds
  the same persistent atlas. Ordinary object-level DDP is unchanged.
- Added `scripts/validate_external_models.py`; `validate_server.py` invokes its
  VGGT and TRELLIS real-view passes in separate processes and records upstream
  provenance, tensor/support contracts, SO(3) error, runtime, and peak CUDA
  allocation. The smoke object is manifest-selected, not hardcoded.
- Removed executable-source workstation dataset defaults. Real-data tests use
  the server environment or the generated manifest-summary provenance; only
  `docs/DATASET_AUDIT.md` retains the local audit path as historical evidence.

## 2026-07-17 bounded remote smoke-record selection

- Requirement `DATA-REMOTE-SELECT-01`: checkpoint-backed VGGT/TRELLIS
  validation must use the full remote manifest without constructing every
  object in a split merely to load one smoke fixture.
- Production path: `validate_server.py` -> `validate_external_models.py` ->
  `MeshFleetDatasetConfig.include_object_ids` -> `MeshFleetObjectDataset`.
- `include_object_ids` is an explicit runtime subset, separate from the
  optional `object_id_file` manifest identity catalog. It is validated as a
  unique set of 64-hex IDs; when both mechanisms are present, the runtime set
  must be a subset of the catalog.
- Selection is applied before phase-relative admission checks and tensor I/O.
  Coverage reports selected and split-absent IDs, and an empty selection now
  names every absent ID rather than emitting an uninformative empty detail.
- The external-model preflight first applies the same
  `meshfleet_record_admission_reasons` predicate as the loader, then chooses
  one deterministic record and instantiates only that object. A record with
  two render files but invalid task-specific normalization can no longer hide
  a later valid record.

## 2026-07-17 released input-domain parity

- Requirement `UPSTREAM-INPUT-01`: both external adapters must enforce their
  audited tensor contract before checkpoint code executes.
- Production path: MeshFleet `[0,1]` composites -> `VGGTAdapter.forward` and
  `TrellisPriorAdapter.sample`.
- VGGT now matches the existing TRELLIS boundary by rejecting empty scene/view
  batches, non-RGB shape, non-floating tensors, non-finite values, and values
  outside `[0,1]`. This prevents invalid padding or normalization from entering
  the released aggregator and producing opaque downstream geometry failures.
- `test_external_adapters.py` contains CPU-safe mock-boundary tests whose mock
  aggregator fails if invalid input reaches upstream inference.

## 2026-07-20 scheduler-visible A800 and declared upstream roots

- Requirement `DEPLOY-DYNAMIC-01`: use exactly the idle GPU subset assigned by
  `CUDA_VISIBLE_DEVICES`, never a hardcoded six processes or an unmasked
  physical-device assumption.
- Production path: `launch_a800_6gpu.sh` -> pinned CRAFT interpreter -> exact
  environment audit -> `torch.cuda.device_count()` -> `torch.distributed.run`;
  `validate_ddp_server.py` independently checks world-size/device-count
  equality and unique rank bindings. The legacy launcher filename is retained
  for compatibility, but its behavior is rank-count agnostic.
- Requirement `DEPLOY-ROOTS-01`: the declared remote roots
  `/mnt/sda2/hef/Base/vggt`, `/mnt/sda2/hef/Base/TRELLIS`, and
  `/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97`
  are now the physically checked server defaults. Environment/CLI overrides
  remain available; no Windows reference path enters production.
- `validate_server.py` fails before accelerator/model work unless each declared
  checkout contains its package and known runnable entrypoint
  (`demo_gradio.py` or `app.py`). It records SHA-256 identities for both the
  package initializer and entrypoint, then propagates the resolved roots into
  every checkpoint preflight and test subprocess.
- `integration/external.py` prefers an explicit root, then the declared root
  when it contains the expected package, then a portable installed package.
  Existing module-origin verification proves the imported source is inside the
  chosen checkout. Checkpoints still resolve from explicit CLI/environment to
  the official cached hub identifier.

## 2026-07-20 A800 renderer-equivalence and executable-precision repair

- Requirement `RENDER-MIP-EQUIV-01`: the returned A800 comparison exposed a
  real image-model mismatch (`109/768` RGB elements, max absolute error
  `0.6104467`), rather than insufficient tolerance. The production reference
  and CUDA adapter now share `RasterizationContract`, matching the inspected
  TRELLIS mip kernel's EWA covariance filter, determinant opacity
  compensation, integer pixel centers, 16x16 tile extent, near cull, alpha
  cap/pruning, and transmittance termination.
- Production path: analytical atlas covariance -> packed
  `[xx,xy,xz,yy,yz,zz]` -> TRELLIS `cov3D_precomp`. Quaternion/scale
  reconstruction was removed from CUDA rendering. OpenCV intrinsics now map
  exactly through TRELLIS `ndc2Pix`, including off-axis principal points.
  Color retains the requested background; alpha/depth/normal use a separate
  zero-background rasterizer. Alpha and depth share one auxiliary pass, reducing
  total passes from four to three without changing the compositing measure.
- The renderer boundary now rejects nonfinite or non-OpenCV intrinsics,
  nonpositive focal lengths, non-SO(3) world-to-camera rotations, mixed camera
  devices/dtypes, malformed backgrounds, and attempts to override constants
  compiled into the TRELLIS CUDA kernel.
- Requirement `PRECISION-A800-01`: the formerly descriptive YAML precision
  section is executable. `NativePrecisionPolicy` confines BF16/FP16 to the VGGT
  aggregator, requires FP32 geometry/OT/manifold/analytical solve/render state,
  reserves FP64 for diagnostics, selects `highest` FP32 matmul, and disables
  CUDA/cuDNN TF32. Training, overfit, inference, evaluation, ablation, teacher
  refinement, and external-model preflight apply the policy. The VGGT adapter
  receives its configured backbone dtype explicitly.
- Detached nonlinear feasibility acceptance and metric topology-boundary
  certification now recompute areas, orientation, separation, covariance
  eigenvalues, constraint gradients, and evidence-metric dual norms in FP64.
  The trainable CBF/JVP/QP path remains FP32, preserving its gradient and A800
  throughput while making the final marginal accept/reject decision stricter.
- Checkpoint format 6 records the precision boundary. Exact resume, phase
  initialization, distillation teachers, inference, evaluation, ablations, and
  teacher refinement reject incompatible precision provenance; documented
  format-5 defaults remain loadable only under the same native default.
- Requirement `ENV-PIN-CONSISTENCY-01`: `requirements.txt` already pins
  `ipykernel==7.3.0` and compatible `jupyter_client==8.9.1`. The server's
  installed 7.4.9 client is an exact-environment violation. The validator now
  records and prints exact synchronization/verification commands for either a
  pin mismatch or `pip check` failure; no dependency constraint was loosened.

## 2026-07-22 TRELLIS decoded-grid contract repair

- Requirement `TRELLIS-DECODED-GRID-01`: the returned A800 trace proved that
  `sparse_structure_flow_model.resolution=16` describes the sampled latent,
  while `sparse_structure_decoder` emits occupancy on a 64-cubed lattice and
  `sample_sparse_structure` returns coordinates in `[0,63]`.
- Production path changed in `integration/trellis_prior.py`: a temporary
  decoder forward hook records the authoritative cubic output extent for every
  posterior draw, requires `[1,1,R,R,R]`, requires one consistent `R`, removes
  itself under `finally`, and validates coordinates/support mass using decoded
  `R`. No value is hard-coded and no resolution is inferred from occupied
  coordinate maxima. Exact cache hits retain the observed decoded resolution.
- `test_external_adapters.py` now covers 16-to-64 decoding, boundary cells,
  cache identity, exact canonical cell centers and area mass, and rejection of
  malformed/non-cubic/inconsistent decoder domains. The existing flow-latent
  metadata remains validated separately.
- `overfit_meshfleet_object.py` now explicitly activates the existing
  same-object DDP contract. Ranks receive deterministic view shards, build one
  global autograd-gathered evidence measure, synchronize the atlas/stratum, and
  run the frozen TRELLIS sampler only on the source rank. Ordinary corpus
  object-level DDP is unchanged.
- Requirement `DDP-ATLAS-TRANSPORT-01`: the subsequent A800 run passed TRELLIS
  sampling and exposed Torch-2.4 NCCL rejection of atlas `levels:int16`.
  `AtlasDDPSynchronizer` now preflights complete atlas config/shape/dtype/device
  metadata, transports every discrete field through an independent contiguous
  int64 buffer with an exact checked round trip, and applies the same codec to
  Boolean split masks. This also protects int8 child slots, noncontiguous int64
  connectivity, and Morton identities beyond floating-point exactness.
- Requirement `DDP-ATLAS-GAUGE-02`: the next A800 smoke run passed the int64
  transport boundary and exposed a raw `chart_frames` mismatch of exactly one.
  This is a PCA chart-gauge ambiguity: eigenvector signs and bases inside a
  nearly repeated eigenspace are not unique. Same-object DDP now defines one
  source-owned nonlinear atlas and uses autograd-aware floating broadcasts.
  Backward reduces every rank's downstream atlas derivative to the source; the
  preceding differentiable evidence all-gather then routes exact derivatives
  to each rank's local VGGT evidence. The invalid local straight-through frame
  derivative was removed.
- Gauge-coordinate fields (`chart_frames`, local curvature, overlap rotations
  and translations) are source-authoritative and are not compared in raw
  coordinates. Gauge-independent state retains collective finite and mixed
  absolute/relative replica checks, all discrete state retains exact int64
  transport, and the synchronized source atlas is structurally revalidated.
- Requirement `ATLAS-PCA-EIGENGAP-01`: PCA frame differentiation is restricted
  to the simple-spectrum stratum measured by an explicit relative eigengap.
  Repeated or near-repeated eigenvectors retain their valid SO(3) forward gauge
  but receive a finite zero derivative for that mathematically unidentifiable
  gauge. The threshold is typed in `AtlasConfig`, loaded from the server YAML,
  and does not reduce FP32 geometric-state precision.

## 2026-07-22 Phase-B finite-gradient containment and readout repair

- Requirement `READOUT-SPECTRAL-STRATA-01`: the returned two-rank smoke passed
  decoded-grid, int64 NCCL transport, source-gauge synchronization, the first
  complete forward/backward/update, and its format-6 checkpoint. Its second
  forward then found non-finite particle mass. Since calibrated mass is a
  product of non-negative finite-domain factors, this is evidence that the
  first update corrupted a calibrator parameter; the old compound atlas error
  hid whether the cause was shape, sign, or NaN.
- Phase B executes analytical readout. Flat initialization gives repeated
  eigenvalues in the 2D first fundamental form, where an ordinary eigenvector
  derivative contains an undefined inverse eigengap. Readout now uses an
  eigengap-stratified chart-metric decomposition: exact differentiable
  eigenpairs on the simple-spectrum stratum, exact forward gauge with zero
  gauge derivative at an unresolved spectrum, and the common trace derivative
  for isotropic scale. The configured relative threshold is explicit.
- Gaussian covariance no longer reconstructs a tangent covariance through
  principal eigenvectors. The algebraically identical expression
  `a^2 J J^T + sigma_n^2 n n^T` is basis-free, SPD, and has no eigenvector
  derivative. Zero-curvature thickness uses a smoothed Frobenius bound instead
  of the nondifferentiable spectral norm. Initial state covariance boxing uses
  an eigenvector-free shift-and-contract spectral map.
- Requirement `TRAIN-FINITE-01`: loss terms, gradients, trainable parameters,
  post-update parameters, and every tensor-valued optimizer state are checked
  collectively. Any rank with NaN/Inf causes all ranks to raise before the
  next collective. Gradient clipping accumulates the norm in FP64 after the
  collective finite-state gate, so finite FP32 gradients cannot overflow while
  being squared and a NaN norm cannot poison every Adam state. Evidence,
  calibrator, and atlas constructors now name non-finite fields explicitly and
  never reinterpret them as absent or zero mass.
- Production files: `readout/assets.py`, `manifold/geometry.py`,
  `integration/pipeline.py`, `mapping/manifold_mapping.py`,
  `geometry/atlas.py`, `engine/trainer.py`, `engine/configuration.py`, and the
  A800 YAML. Numerical regressions are wired into `validate_ddp_server.py`.

## 2026-07-22 A800 rank ownership and useful concurrency

- Requirement `DDP-DEVICE-OWNERSHIP-02`: `LOCAL_RANK` is bound before either
  released checkpoint is instantiated and before NCCL process-group creation.
  Trainer construction rejects any process-local PyTorch allocation or cache
  reservation on a non-local visible CUDA device. This directly addresses the
  supplied run in which one PID appeared on two A800s.
- Requirement `TRAIN-VIEW-CONCURRENCY-01`: same-object overfit accepts an exact
  `--views-per-rank` budget and derives the global loader budget from the
  dynamic `WORLD_SIZE`. CPU tensors are deterministically sharded before
  non-blocking device transfer, so additional memory is spent on useful local
  VGGT evidence rather than replicated global images and cameras.
- Production training exposes `--maximum-views`; the selected budget is stored
  in exact-resume provenance. Object-level DDP remains one variable-topology
  object per rank, with configurable pinned-memory workers/prefetch. The A800
  YAML uses native BF16 only in the VGGT backbone and FP32 for geometric,
  transport, manifold, barrier, readout, renderer, and optimizer state.
- One-object output records per-rank peak allocated/reserved fractions, local
  views and views/s. Final asset evaluation is rank-zero-only and has a separate
  deterministic view cap, preventing a global same-object training budget from
  being redundantly evaluated and serialized on every rank.
- Changed production files: `graft_gs/engine/trainer.py`,
  `graft_gs/engine/__init__.py`, `scripts/train_a800.py`,
  `scripts/overfit_meshfleet_object.py`, and
  `configs/graft_gs_a800_native.yaml`.

## 2026-07-22 strict stratum restoration and implicit-solver certification

- Requirement `TOPOLOGY-EMBED-RESTORE-01`: the supplied post-update A800 run
  reached final evaluation but rejected its only topology candidate at
  separation margin `-2.605944407442015e-09`. This corresponds to an actual
  closest distance of about `8.60e-5`, below the configured `1.0e-4`; it is not
  accepted as roundoff and the hard separation threshold is unchanged.
- Topology selection now performs a pre-flow, FP64, evidence-metric
  minimum-displacement QP over active area, orientation, vertex-separation,
  and triangle-separation inequalities. Constraint families are normalized in
  their native units, projected-dual steps use deterministic merit
  backtracking, total displacement is bounded by the collision broad-phase
  certificate, and a newly constructed projector independently recertifies
  strict feasibility. Failure rejects that discrete candidate; it never turns
  into a soft penalty or fabricated feasible report.
- Requirement `UOT-CONVERGENCE-02`: sparse KL-UOT now validates finite
  non-negative costs/measures, exact support index domains, common
  FP32/FP64 dtype/device, and the actual coupled fixed-point equations. Both
  forward and implicit-adjoint solves fail closed on non-convergence or
  non-finite state. The later `UOT-LOG-DOMAIN-03` cycle supersedes the original
  absolute row/column positivity gate with FP64 log-conditional validation,
  because KL-unbalanced marginals may be exponentially small. Relative
  stopping scales are explicit, while device/host convergence synchronization
  is reduced from every iteration to every eighth iteration.
- Requirement `DDP-CHECKPOINT-COMMIT-01`: distributed checkpoint serialization
  is a collective commit transaction. Rank zero atomically replaces the file,
  then broadcasts success/failure before any peer can enter the next NCCL
  forward. Save failures are reported on all ranks. This removes a potential
  collective-order race around a slow rank-zero filesystem write.
- Production files: `graft_gs/manifold/barrier.py`,
  `graft_gs/integration/pipeline.py`,
  `graft_gs/mapping/manifold_mapping.py`,
  `graft_gs/engine/configuration.py`, `graft_gs/engine/trainer.py`, and
  `configs/graft_gs_a800_native.yaml`. Numerical regressions are in
  `tests/test_geometry_invariants.py`, `tests/test_atlas_mapping.py`, and
  `tests/test_distributed_evidence.py`.
- The one-object artifact now records the selected topology, UOT iterations,
  residual/effective tolerance, minimum transported row/column mass, and both
  initial and final feasibility reports. This makes restoration and subsequent
  barrier preservation directly auditable from `overfit_metrics.json`.
- Barrier dual residuals are checked in scale-relative batches of eight
  iterations, and the five FP64 nonlinear certificate minima are transferred
  to the host in one operation. Acceptance remains a strict positive-margin
  predicate; the change removes synchronization stalls, not checks.

## 2026-07-22 measured A800 view and object concurrency

- Requirement `A800-USEFUL-CONCURRENCY-02`: a new report-driven selector audits
  the complete overfit artifact for every candidate view count. It rejects
  missing/duplicated ranks, non-finite loss, unconverged UOT, non-positive
  transported row/column mass, non-positive final hard margins, and peak
  reserved memory above the configured limit. Among runs within 3% of fastest
  aggregate useful views/s it selects the largest measured per-rank view count,
  favoring coverage without accepting a large throughput regression.
- The server protocol now sweeps `16,24,32,48,64` views per rank. This expands
  the earlier conservative sweep in response to the supplied 15--16 GiB A800
  occupancy while preserving an 85% reserved-memory ceiling for irregular
  object/topology peaks. The chosen value is passed explicitly to corpus
  training and remains checkpoint provenance.
- Requirement `DDP-DYNAMIC-BATCH-02`: production training accepts a minimum
  global independent-object batch and derives accumulation as
  `ceil(target/WORLD_SIZE)`. This preserves statistical batch scale across a
  scheduler-variable visible GPU subset while maintaining exactly one process
  per GPU. Same-object view sharding rejects this option because repeated views
  of one object are not independent object samples.
- A800 reference transport chunks are raised to 4096 atlas by 16384 evidence
  rows to reduce Python/kernel-launch overhead. The mathematical support is
  unchanged away from the discrete radius-threshold boundary; equivalence and
  throughput remain part of the server sweep.
- Radius/nearest-neighbor support discovery is explicitly `no_grad`: sparse
  indices are discrete and cannot carry a derivative, while the subsequent
  selected-edge cost is evaluated from the original tensors and preserves the
  conditional transport gradient. This removes a large unusable `cdist` tape
  before spending memory on additional physical views.
- Changed production files: `scripts/select_a800_view_budget.py`,
  `scripts/overfit_meshfleet_object.py`, `scripts/train_a800.py`,
  `graft_gs/engine/configuration.py`,
  `graft_gs/mapping/manifold_mapping.py`, and the A800 YAML/protocol.

## 2026-07-22 high-view sparse-UOT and topology-proposal repair

- Requirement `UOT-LOG-DOMAIN-03`: the supplied 16-view/rank run reached the
  real sparse solver and failed because an exponentially rejected supported
  component was required to remain positive after FP32 exponentiation. UOT
  potentials, absolute plan state used by the custom adjoint, and row/column
  conditional probabilities now use FP64 and log-segment normalization. The
  returned geometric plan retains its configured FP32 storage contract;
  representational underflow is explicit telemetry rather than a fabricated
  mass floor. A run still fails if every edge is below storage range.
- Requirement `TOPOLOGY-FILTRATION-ENDPOINT-02`: the 24/32/48/64-view runs
  reached topology selection but every quantile/fixed cut removed the overlap
  triangles. The valid maximal atlas support, which was already used to define
  reference persistence, is now an explicit terminal filtration candidate at
  a threshold strictly below `min(p)`. It competes under the same evidence,
  persistence, geometry, complexity, boundary, and TRELLIS-prior energies.
- Surface face construction now inserts cells through a parity union-find. An
  orientation-conflicting face is rejected locally; it can no longer erase a
  valid incidence-constrained surface component. The all-support reference is
  independently required to meet minimum vertex, nondegeneracy, incidence,
  and orientation conditions. Failure diagnostics include occupancy range,
  reference `V/E/F`, support cardinalities, and rejection classes.
- One-object reports and the view-budget selector now certify internal solve
  dtype, minimum log plan, exact storage-underflow counts and graph
  cardinalities. The selector treats small acknowledged FP32 underflow as an
  approximation with a configured fraction bound, not as either exact
  positivity or an unconditional pass.
- Production files: `graft_gs/mapping/manifold_mapping.py`,
  `graft_gs/topology/strata.py`, `graft_gs/engine/configuration.py`,
  `scripts/overfit_meshfleet_object.py`,
  `scripts/select_a800_view_budget.py`, and
  `configs/graft_gs_a800_native.yaml`.

## 2026-07-23 scalable persistence matching at refined-atlas scale

- Requirement `TOPOLOGY-PERSISTENCE-MEMORY-03`: the first post-log-UOT A800
  smoke passed transport and surface-complex proposal, then both ranks failed
  in `torch.cdist` while persistence matching requested 11.42 GiB on top of
  approximately 69.67 GiB of retained differentiable state. The former
  diagonal-augmented Hungarian construction required `O(nm+(n+m)^2)` memory
  and `O((n+m)^3)` assignment time; it is not a valid production operator for
  refined diagrams.
- Exact zero-lifetime lower-star pairs are now removed in one vectorized
  detached active-set operation. This is not pruning: diagonal points have
  exactly zero optimal diagonal cost. Diagrams with at most 512 combined
  off-diagonal points retain the exact Hungarian reference.
- Larger diagrams use deterministic midpoint-quadrature sliced persistence
  Wasserstein. Each direction solves the exact projected one-dimensional
  assignment by sorting diagonal-augmented projections, using `O(n+m)` peak
  working memory and `O(L(n+m) log(n+m))` time for `L=32`. Values and gradients
  remain on-device; only ordering/ties define the usual piecewise stratum.
  Identical diagrams select the finite zero subgradient rather than evaluating
  `sqrt(0)` with an infinite outer derivative.
- Selected topology artifacts now record per-homology-dimension diagram
  cardinalities and whether matching was `exact` or `sliced`. The view-budget
  selector rejects reports without this certificate, preventing pre-repair
  artifacts from entering concurrency selection.
- A verified accidental local paste of the supplied server log was removed
  from between `_sinkhorn_fixed_point` and `_ImplicitUnbalancedSinkhorn`;
  exactly 19,044 contaminated characters were deleted and the restored module
  compiles.
- Production files: `graft_gs/topology/strata.py`,
  `graft_gs/topology/__init__.py`, `graft_gs/engine/configuration.py`,
  `scripts/overfit_meshfleet_object.py`,
  `scripts/select_a800_view_budget.py`, and the A800 YAML.

## 2026-07-23 bounded exact-camera mesh supervision

- Requirement `MESHFLEET-RASTER-MEMORY-02`: the post-persistence two-rank
  16-view/rank run advanced through TRELLIS, UOT, atlas, and topology, then
  failed in nvdiffrast `cudaMalloc` while deriving immutable mesh depth/normal
  targets after the full trainable forward graph had already occupied device
  memory. This was a tensor-lifetime and batch-workspace defect, not permission
  to drop geometrically valid supervision.
- `MeshGroundTruthRasterizer` now renders deterministic contiguous view chunks
  of size two against the same triangle soup, cameras, projection, and
  nvdiffrast context. Transient transformed-vertex/binning storage scales with
  chunk size rather than every local view. Depth, normal, visibility, and
  normal-validity tensors are concatenated in original view order.
- Train and validation target derivation now occurs before the model forward.
  The target path is explicitly `no_grad`; no learned gradient is removed
  because source mesh geometry and audited cameras are immutable supervision.
  The total target storage remains present for the loss, while nvdiffrast's
  transient workspace no longer overlaps the forward activation peak.
- The chunk size is a positive, checkpoint-resume-sensitive
  `TrainerConfig` field read by both `train_a800.py` and
  `overfit_meshfleet_object.py` from the A800 YAML. The overfit entry point now
  also honors the existing derive/require mesh-supervision YAML controls
  instead of relying on dataclass defaults.
- The A800 MeshFleet test compares chunk size one with the former two-view
  batch at `1e-6` for continuous outputs and exactly for Boolean masks.
- A verified stray `z` after `GraftGSConfig()` in the current configuration
  loader was removed after direct `py_compile` reproduced the syntax failure.
- Production files: `graft_gs/data/mesh_supervision.py`,
  `graft_gs/engine/trainer.py`, `graft_gs/engine/configuration.py`,
  `scripts/train_a800.py`, `scripts/overfit_meshfleet_object.py`, and
  `configs/graft_gs_a800_native.yaml`.

## 2026-07-24 high-view renderer recomputation and fail-closed sweep

- Requirement `CUDA-RENDER-TAPE-MEMORY-03`: the supplied fresh sweep advanced
  through target rasterization and failed at 32 views/rank while the production
  Gaussian renderer was accumulating three native CUDA rasterizer autograd
  states per view. The reported `normalize` line is only the allocation
  synchronization point; the retained per-view splat tapes are the causal
  linear-in-view memory state.
- `CudaGaussianRenderer` now wraps each complete deterministic
  RGB/alpha-depth/normal view operator in PyTorch 2.4 non-reentrant activation
  checkpointing. All four output tensors and all losses are unchanged; native
  raster intermediates are recomputed one view at a time during backward.
  Shared zero-background/subpixel tensors and FP32 analytical inputs are
  constructed once per camera batch.
- Checkpointing is a Boolean `TrainerConfig` execution policy, not a
  `GraftGSConfig` model parameter. Both training entry points read it from the
  A800 YAML, the trainer applies it before DDP wrapping, metrics record it, and
  exact resume records it. Checkpoint format 7 adds this field plus the mesh
  target chunk policy while accepting format-6 checkpoints only at their
  declared legacy defaults.
- `select_a800_view_budget.py` schema v2 rejects stale reports without the
  active CUDA checkpoint certificate. When no candidate is admissible it now
  writes `selection.json` with every rejection reason before raising; “no
  candidate” is no longer an opaque terminal message.
- New `sweep_a800_view_budget.py` eliminates shell-redirection/quoting defects,
  binds the dynamic visible GPU count to `CUDA_VISIBLE_DEVICES`, requires a
  fresh root, retains exact child commands/logs, and invokes the scientific
  selector over completed reports only. Its original monotone-OOM stop policy
  is superseded by the 2026-07-25 exhaustive non-monotone repair below.
- The A800 regression compares checkpointed and uncheckpointed CUDA forward
  color/alpha/depth/normal at `1e-6` and Gaussian-state gradients at
  `2e-5 + 2e-4 relative`; no local CUDA pass is claimed.
- Production files: `graft_gs/readout/renderer.py`,
  `graft_gs/engine/trainer.py`, `graft_gs/engine/configuration.py`,
  `scripts/overfit_meshfleet_object.py`, `scripts/train_a800.py`,
  `scripts/select_a800_view_budget.py`,
  `scripts/sweep_a800_view_budget.py`, and the A800 YAML.

## 2026-07-24 allocator-lifetime and transport-error certification

- Requirement `A800-UPSTREAM-CACHE-LIFETIME-04`: the supplied completed
  16/24/32-view reports all showed rank-0 peak reserved fractions between
  `0.9854` and `0.9859`, nearly independent of view count. Same-object DDP
  samples the frozen TRELLIS posterior only on rank 0, and the exact
  conditioning cache prevents resampling after the first step. The causal
  defect was retention of inactive diffusion workspaces by PyTorch's caching
  allocator, not an equally large live GRAFT-GS state.
- `TrellisPriorAdapter` now releases inactive CUDA cache exactly after the
  posterior sparse coordinates and CPU cache entry have been materialized.
  It synchronizes the source device and asserts that `memory_allocated` is
  identical before and after release. Live TRELLIS weights, returned prior
  coordinates, numerical precision, and all trainable gradients are
  unchanged. The policy is Boolean, config-validated, enabled in the A800 YAML,
  and passed by all production training/inference/ablation entry points.
- When audited canonical atlas bounds are supplied, production now samples and
  synchronizes the frozen TRELLIS measure before VGGT constructs its
  differentiable multiscale geometry state. This prevents the first cache
  miss's diffusion workspace from overlapping the VGGT tape. Unbounded
  inference still derives its root from VGGT evidence and uses the original
  ordering.
- Training telemetry now separates peak allocated, peak active, peak reserved,
  ending allocated/reserved/inactive-reserved, and ending driver-free bytes and
  fractions. It also records whether the source-rank cache release occurred
  and how many reserved bytes it returned. Schema-v3 concurrency selection
  gates live peak, ending allocator state, and driver headroom; historical peak
  reservation remains diagnostic.
- Requirement `UOT-STORAGE-ERROR-CERTIFICATE-04`: the previous selector treated
  the fraction of sparse edges that became zero in FP32 storage as lost
  transport mass. The supplied values (`0.3781`--`0.4091`) therefore rejected
  every completed run even though those edges may carry exponentially
  negligible FP64 mass. The implicit solver now reports underflow mass,
  zero-source/target mass, and total relative L1 cast error against its FP64
  log-domain plan. Selector admission uses these measure-valued errors; exact
  zero counts remain diagnostics and no positive mass floor is fabricated.
- The old count-based selector flags remain accepted with an explicit warning
  and are recorded as ignored legacy gates, preventing existing launch scripts
  from failing argument parsing while avoiding the invalid scientific proxy.
- Production files: `graft_gs/integration/trellis_prior.py`,
  `graft_gs/mapping/manifold_mapping.py`, `graft_gs/engine/trainer.py`,
  `graft_gs/engine/configuration.py`, all TRELLIS-backed entry points,
  `scripts/select_a800_view_budget.py`,
  `scripts/sweep_a800_view_budget.py`, and the A800 YAML/protocol.

## 2026-07-25 source-prior lifetime and non-monotone A800 sweep repair

- Requirement `A800-FROZEN-PRIOR-RESIDENCY-05`: the supplied schema-v3 sweep
  completed 16 and 24 views/rank but left only `0.061%`--`0.18%`
  driver-visible memory free. The 32-view candidate then failed in backward
  while cuBLAS created its handle. Transport remained numerically healthy:
  relative-L1 storage error was approximately `2.2e-8`, zero marginal mass was
  exactly zero, and discarded underflow mass was approximately `2e-43`.
- Same-object DDP now constructs the full TRELLIS checkpoint only on the
  designated source rank. Other ranks retain the analytical probability/mass
  operations through a sampler-inert proxy and receive the typed structure
  measure by broadcast. After each source cache miss, exact frozen weights move
  to CPU and inactive CUDA allocator blocks are released before VGGT builds the
  differentiable graph.
- All observed views remain in VGGT evidence and all direct/derived
  supervision. Only the frozen hidden-surface prior is capped to 16
  deterministic endpoint-covering conditioning views in the A800 config.
  The distributed gather exactly inverts the rank-strided view partition before
  selection, so coverage follows original camera order rather than rank blocks.
  Available and selected counts are serialized.
- The view-budget path records per-stage allocated, reserved, driver-free, and
  non-allocator-visible CUDA memory, emits a rank-local JSON certificate on
  allocation failure without entering a collective, and separates the frozen
  prior peak from the differentiable graph peak.
- The sweep now rejects a preoccupied scheduler-visible device before loading
  checkpoints, evaluates every candidate by default because object-dependent
  atlas/render memory is not monotone in view count, and offers
  `--stop-after-oom` only as an explicit diagnostic shortcut. Selector schema
  v4 and sweep schema v3 reject stale reports lacking the new certificates.
- Production files: `graft_gs/integration/trellis_prior.py`,
  `graft_gs/integration/pipeline.py`, `graft_gs/engine/trainer.py`,
  `graft_gs/engine/configuration.py`, all TRELLIS-backed production entry
  points, `scripts/overfit_meshfleet_object.py`,
  `scripts/select_a800_view_budget.py`,
  `scripts/sweep_a800_view_budget.py`, A800 config, tests, and ledgers.

## 2026-07-25 sparse-barrier, offline-upstream, and zero-mass adjoint repair

- Requirement `A800-NUMERICAL-WORKFLOW-06`: the supplied fresh sweep exposed
  three independent failures. Vpr-8 reached backward but a zero-reliability
  attention edge differentiated the singular expression
  `sqrt(reliability_i * reliability_j)` and surfaced later as a non-finite
  Sinkhorn cotangent. Vpr-12 failed before model construction because Torch
  2.4 probed GitHub while a complete DINOv2 Hub checkout was already cached.
  Vpr-24 completed training, then incorrectly executed Phase-D flow during the
  Phase-B final measurement and attempted a 3.78-GiB dense barrier Jacobian.
- The attention bias now uses an endpoint-preserving Charbonnier continuation
  of the geometric mean. It maps exact zero and one to zero and one, retains
  monotonicity, and has a finite derivative at zero. The implicit UOT adjoint
  separately diagnoses non-finite upstream positive-mass cotangents, internal
  adjoint failures, and gradients outside the destination dtype range.
- TRELLIS construction scopes a Torch Hub redirect for
  `facebookresearch/dinov2` to the deterministic cached `main`/`master`
  checkout and restores the original loader after construction. A missing
  cache fails before any network loader is invoked.
- `BarrierProjector` now represents every position constraint by at most six
  vertex indices and local derivatives. Exact products with
  `A G^-1 A^T` are matrix-free, and a Cauchy--Schwarz incident-vertex bound
  supplies a valid projected-gradient step. Both flow projection and
  pre-flow embedding restoration use this path. Storage is `O(J+V)` instead
  of dense `O(JV+J^2)`; the same piecewise linearized CBF QP is solved.
- The overfit/concurrency executable evaluates at the Phase-B
  `atlas_autoencoding` boundary. Selection schema v5 rejects missing or
  full-flow evaluation metadata, so stale reports cannot be admitted.
- Production files: `graft_gs/integration/pipeline.py`,
  `graft_gs/integration/trellis_prior.py`,
  `graft_gs/mapping/manifold_mapping.py`,
  `graft_gs/manifold/barrier.py`, `scripts/overfit_meshfleet_object.py`,
  `scripts/select_a800_view_budget.py`, `scripts/validate_ddp_server.py`,
  numerical/static tests, protocol, and project ledgers.

## 2026-07-25 posterior-measure transport-gradient repair

- Requirement `A800-POSTERIOR-MOMENT-STABILITY-07`: the supplied schema-v5
  sweep completed 8, 12, and 16 views on one A800, but 24/32/48/64 all failed
  before the optimizer step because the implicit Sinkhorn plan received
  non-finite upstream cotangents. The failing edge counts grew with evidence
  size and included both positive stored mass and representationally
  negligible FP32-underflow edges. This is not an allocator failure.
- Root cause: chart writing formed conditional quantities
  `sum(pi f)/sum(pi)` and then multiplied them by
  `sum(pi)/(sum(pi)+lambda)`. Although the factors cancel analytically for
  positive mass, evaluating them separately creates a storage-dependent
  `1/epsilon_float32` derivative on an underflow row.
- Chart position, metric, every irrep moment, auxiliary statistics, and color
  now use the fused posterior denominator `m_i + lambda_i`, where
  `lambda_i = retention_shrinkage * source_area_i > 0`. This is exactly equal
  to the previous positive-mass Bayesian shrinkage and defines its continuous,
  finite zero-observation extension. Raw transported mass remains separate for
  occupancy and UOT diagnostics.
- Transport-conditioned attention uses the same posterior prior: its cost is
  the ordinary conditional cost multiplied by observation reliability.
  Named identity boundaries now reject and attribute non-finite cotangents at
  chart fields, attention evidence, and analytical Gaussian attributes without
  detaching, clipping, or replacing any gradient.
- Exact-zero geometric derivatives were repaired at chart radial directions,
  GSTA self/coincident edges, zero-weight curvature fits, surface-attached SH
  distances, represented-area square roots, uncertainty thickness, and
  squared Chamfer. These changes use either algebraically exact squared
  quantities or scale-aware zero-preserving smooth norms.
- Production files: `graft_gs/mapping/manifold_mapping.py`,
  `graft_gs/mapping/__init__.py`, `graft_gs/integration/pipeline.py`,
  `graft_gs/equivariant/gsta.py`, `graft_gs/geometry/atlas.py`,
  `graft_gs/readout/assets.py`, and `graft_gs/engine/losses.py`.
  Numerical regressions were added to `tests/test_atlas_mapping.py`,
  `tests/test_geometry_invariants.py`, and
  `tests/test_assets_and_vertical_slice.py`; the static production guard was
  extended in `tests/test_scientific_trace_static.py`.
- Whole-tree compilation and all locally executable environment/dataset/
  manifest/production/selection guards passed (`76/76`, one expected
  PyTorch-dependent loader skip). The new mathematical gradient tests and
  first-failing 24/32-view backward gates are server-pending.

## 2026-07-25 SPD-native metric-readout repair

- Requirement `A800-SPD-METRIC-BACKWARD-08`: the supplied post-posterior A800
  gate passed all ten focused numerical tests in 2.556 seconds. Its 24-view
  production Phase-B step completed and emitted `final.pt`,
  `step-00000001.pt`, PLY, GLB, and metrics. The 32-view step did not OOM; it
  failed because all 1,008 entries of a `[112,3,3]` chart-metric cotangent
  became non-finite downstream of the chart writer.
- Root cause boundary: production formed a generic LU inverse of the selected
  node metric for initial covariance and another generic inverse of the
  partition-of-unity metric for every analytical Gaussian. The prior flat
  readout regression did not retain/check the chart-metric gradient and did
  not exercise a high-condition 3D metric field.
- `spd_inverse_cholesky` now evaluates an SPD inverse as
  `s^-1 L^-T L^-1` with a detached, algebraically cancelled scale and
  triangular solves. `spd_inverse_quadratic_trace` computes only
  `n^T M^-1 n` and `tr(M^-1)` for analytical uncertainty, so readout never
  materializes its former per-Gaussian inverse. No jitter, NaN replacement,
  detach of geometric state, loss removal, or precision reduction is used.
- The same SPD primitive replaces generic evidence/covariance inverses in
  state initialization, the product-manifold metric, and barrier restoration,
  boundary certification, and velocity projection. Named cotangent boundaries
  distinguish state initialization, inverse boxing, readout node metrics, and
  continuous interpolated metrics.
- Numerical regressions now cover inverse/contraction agreement, repeated and
  condition-spread spectra, finite matrix/vector gradients, retained metric
  gradients in the flat full readout, and a full anisotropic metric readout.
  Production files: `graft_gs/manifold/geometry.py`,
  `graft_gs/manifold/barrier.py`, `graft_gs/manifold/__init__.py`,
  `graft_gs/integration/pipeline.py`, and `graft_gs/readout/assets.py`.
  Local whole-tree compilation and the 31-test production static suite pass.
  The three new Torch tests and unresolved 32-view A800 gate remain
  server-pending.

### Torch 2.4 analytical-pullback follow-up

- The first deployed Cholesky repair moved the named failure from the shared
  chart-writer boundary to `state_initialization.riemannian_metric`.
  `state_initialization.metric_inverse` did not fire, proving that the
  downstream spectral-box cotangent was finite and that the composed
  Cholesky/triangular-solve backward alone created all 1,008 non-finite values.
- `_SPDInverseCholesky` now uses the Cholesky/triangular solve only for the
  forward SPD value and implements the exact Fréchet pullback
  `-M^-1 sym(G) M^-1`. `_SPDInverseQuadraticTrace` applies the same pullback to
  the normal quadratic and trace cotangents. This is the exact derivative, not
  clipping, sanitization, a straight-through estimator, or a detached metric.
- New tests reproduce the real `[112,3,3]` FP32 shape and approximately
  `6e-8` output-cotangent scale, compare the pullback to its closed form, and
  gradcheck both inverse and contraction operators in FP64. Local compilation
  and all 76 executable static/environment/data/selection guards pass; Torch
  numerical and 32-view production reruns remain server-pending.

## 2026-07-25 bounded precision-to-covariance closure

- Requirement `A800-BOUNDED-PRECISION-CLOSURE-09`: the second supplied
  32-view A800 gate still reached
  `state_initialization.riemannian_metric` with all 1,008 cotangent entries
  non-finite (`maximum_finite_abs=5.650222e-08`). The run completed VGGT,
  TRELLIS, sparse transport, atlas construction, readout, and rendering and
  did not OOM. This disproved the assumption that changing only the
  factorization backward was sufficient.
- Root cause class: the production map first materialized an unbounded
  precision inverse and only then applied a covariance spectral box. Even an
  exact inverse pullback contains two factors of `M^-1`; feasibility imposed
  after that operation cannot bound the preceding intermediate derivative.
- State initialization and continuous analytical readout now use the fused
  orthogonally covariant rational map
  `C(M)=lI+(u-l)(I+(u-l)M)^-1`. For every SPD eigenvalue `lambda`, its
  covariance eigenvalue is
  `l+(u-l)/(1+(u-l)lambda)`, strictly in `(l,u)`. The map is inverse-like in
  the resolved interval, smoothly saturates at the feasibility limits, and
  has a bounded derivative. Its well-conditioned `I+(u-l)M` solve runs in
  FP64 and returns FP32 geometric storage.
- The same numerical closure was applied systematically to the Phase-B
  graph: evidence precision/log-determinant are factored once per evidence
  particle and shared by cost and chart writing; their forward factorization
  has an exact joint analytical pullback. Surface uncertainty calibration,
  atlas curvature normal equations, and SH ridge equations use the shared
  FP64 SPD primitives rather than backend LU/Cholesky backward paths.
- `scripts/overfit_meshfleet_object.py` now executes a 112-chart rotated,
  condition-spread CUDA forward/backward preflight before resolving or loading
  VGGT/TRELLIS checkpoints. A patched run must print
  `GRAFT_GS_NUMERICAL_PREFLIGHT=phase-b-rational-spd-zero-dual-v2:passed`; otherwise
  training never begins.
- Numerical tests cover the 112-chart FP32 cotangent, strict covariance
  interval up to storage rounding, SO(3) covariance, first- and second-order
  finite differences, joint inverse/logdet derivatives, and full analytical
  readout. Whole-tree compilation and all 76 locally executable static,
  environment, discovery, manifest, handoff, selection, and production-path
  guards pass (one expected no-PyTorch loader skip). The new Torch tests and
  one fresh 32-view A800 optimizer/asset gate remain server-pending.

## 2026-07-25 zero-padded feasibility dual-norm repair

- Requirement `A800-FEASIBILITY-METRIC-GRADIENT-10`: the supplied post-rational
  32-view gate printed the v1 CUDA numerical-preflight marker, completed the
  full forward path, and then again reported all 1,008
  `state_initialization.riemannian_metric` cotangent entries non-finite
  (`maximum_finite_abs=5.604653e-08`). This proves the bounded
  precision-to-covariance operator itself is healthy and localizes the
  surviving branch to the selected-stratum feasibility restoration, the only
  other Phase-B consumer of `ManifoldState.evidence_metric`.
- Root cause: sparse barrier rows share a fixed `[J,6,3]` representation.
  Triangle rows use all six slots, while face and vertex-pair rows deliberately
  pad unused slots with exact zero covectors. The Gram spectral bound evaluated
  `sqrt(g^T G^-1 g)` on every slot. Torch 2.4 consequently differentiated
  `sqrt(0)` at padding and formed the indeterminate product `0*infinity`,
  contaminating the entire chart-metric cotangent. Existing restoration tests
  asserted only the position gradient and could not detect it.
- The local dual norm is now the conservative bound
  `sqrt(q + delta^2)`, with detached, dtype-relative
  `delta^2=eps*max(stopgrad(max q), tiny/eps)`. It remains greater than or
  equal to the exact norm, so the spectral estimate and projected-gradient
  safety are preserved; only the QP step can become infinitesimally smaller.
  The derivative is finite at every padded zero without detaching or replacing
  a training gradient.
- The restoration regression now requires a finite evidence-metric gradient,
  and a dedicated mixed two/three/six-vertex sparse-row test exercises zero
  padding explicitly. The A800 preflight v2 evaluates the same metric
  derivative before checkpoint loading, and a dedicated
  `feasibility_restoration.position_metric` boundary attributes any remaining
  QP failure.
- Whole-tree compilation and all 76 locally executable static/environment/
  dataset/manifest/selection/production guards pass (`6.498 s`, one expected
  no-PyTorch loader skip). Torch restoration tests and one 32-view A800 gate
  remain server-pending.
