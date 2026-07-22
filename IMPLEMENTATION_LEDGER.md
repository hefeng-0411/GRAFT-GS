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
