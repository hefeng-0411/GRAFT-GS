# Validation ledger

Status vocabulary: **generated** means executable verification code exists;
**syntax checked** means Python bytecode compilation succeeded locally;
**server pending** means no numerical result is claimed before A800 execution.

| Requirement | Verification path | Target tolerance/status |
|---|---|---|
| Camera convention | `CameraConventionTest` | reprojection `atol=1e-10` in float64; generated, server pending |
| MeshFleet camera conversion | `CameraContractTest` | Blender/OpenGL to OpenCV axis/determinant and pixel-K scaling; generated, server pending |
| VGGT canonical gauge | `CameraContractTest` | exact synthetic Sim(3) recovery at `2e-10`; generated, server pending |
| Unprojection/reprojection | `CameraConventionTest` | exact plane depth and patch centers; generated, server pending |
| Sparse UOT convergence | `ImplicitSinkhornTest` | fixed-point residual below `1e-10`; generated, server pending |
| Sparse/dense UOT agreement | `ImplicitSinkhornTest` | `atol=2e-10`, `rtol=2e-9`; generated, server pending |
| Implicit UOT gradients | `gradcheck` and finite-gradient tests | `atol=3e-5`, `rtol=3e-4`; generated, server pending |
| Global SE(3) equivariance | mapping and GSTA tests | costs/plans/irreps/metrics, float64 tolerances; generated, server pending |
| Local gauge covariance | `GaugeEquivarianceTest` | 0e invariant, 1o/2e covariant at `3e-8`; generated, server pending |
| Valid SO(3) | atlas/state/Gaussian validation | determinant and orthogonality margins; generated, server pending |
| SPD covariance/metric | evidence, state, Gaussian tests | strictly positive eigenspectrum; generated, server pending |
| Manifold interpolation/retraction | `TopologyAndManifoldTest` | SO(3) log/exp and SPD endpoints; generated, server pending |
| Chart immersion | atlas validation/Jacobian test | sampled `lambda_min(J^T J)>1e-6`; generated, server pending |
| Area/collision/orientation | barrier tests | positive accepted nonlinear margins; generated, server pending |
| Exact finite topology | tetrahedral sphere test | Betti `(1,0,1)` over Z2; generated, server pending |
| Analytical Gaussian validity | `AnalyticalAssetTest` | reconstruction/covariance checks; generated, server pending |
| Non-floating means | `AnalyticalAssetTest` | means equal chart evaluation at `1e-10`; generated, server pending |
| Deterministic PLY/GLB | byte-equality tests | identical repeated bytes; generated, server pending |
| Independent reload | `plyfile`, `pygltflib` | element/accessor counts; generated, server pending |
| Renderer equivalence | CUDA/reference small scene | RGB/alpha tolerance `5e-2`; generated, server pending |
| End-to-end backward | asset vertical-slice test | finite nonzero evidence position/feature gradients; generated, server pending |
| Checkpoint resume | real server training test | exact parameter/optimizer/RNG/global-step/epoch/microstep/in-epoch restoration plus manifest digest guard; generated, server pending |
| One-object overfit | `scripts/overfit_one_object.py` | executable, metrics/checkpoints emitted; server pending |
| Real multiview inference | `test_real_multiview.py` | render + PLY/GLB reload; server pending |
| Quantization certificate | `test_quantization.py`, `TopologyAndManifoldTest.test_so3_spd_geodesics_and_barrier`, inference production trace | score bound and topology step predicate plus computed `min h/(||grad_g h||+eps)`; metric scaling by `4I` scales the distance by two; generated numerical test/server pending, complete inference wiring locally static-validated |
| Manifest/topology determinism | `test_meshfleet_manifest_static.py` | 6/6 tests passed locally; rebuilt JSONL byte-identical; oriented tetra accepted |
| Raw mesh topology audit | static manifest tests | `V=78448,E=236075,F=157592`, chi `-35`, 8 components, 313 incidence-four edges; local pass |
| Topology label rejection | static + PyTorch contract tests | hard topology/Betti/persistence/stratum/certification false and targets null; static local pass, loss test server pending |
| Non-manifold mesh depth/normal | `MeshFleetAuditTest` | triangle-soup nvdiffrast target has finite depth/normals and visible pixels; generated, A800 pending |
| Same-object global UOT | distributed server suite | autograd complete-evidence gather, unique view identity, identical atlas/support/plan; generated, A800 multi-rank pending |
| Canonical atlas root | MeshFleet loader/pipeline test | exact `[-0.5,0.5]^3`, initializer excludes/reports out-of-root evidence; generated, server pending |
| Physical view availability | static manifest and loader tests | 150/150 main, 1/24 conditioning; local static pass, loader server pending |
| Sparse modality alignment | manifest relational checks | voxel=DINO=latent coordinates for 7,996 rows, zero grid residual; local pass |
| Evidence uncertainty calibration | `DerivedSurfaceTargetTest` | finite position/SPD/confidence gradients with `h^2/12 I`; generated, server pending |
| Evidence-only Phase A | `DerivedSurfaceTargetTest` | no scene/topology construction and finite calibrator gradients; generated, server pending |
| Screened Phase-C target | `DerivedSurfaceTargetTest` | matrix-free normal equation and hard feasibility; generated, server pending |
| MeshFleet real inference | `scripts/infer_meshfleet.py` | checkpoint required, exact cameras, render + PLY/GLB + metrics; server pending |
| Runtime/memory | inference, trainer, ablation scripts | measured wall time and CUDA peak bytes; server pending |
| Hidden-surface prior separation | atlas/mapping and analytical-asset tests | Beta posterior mean/variance, prior-only leaves, unchanged evidence target count, monotone lower-confidence hazard fusion, retention shrinkage, selected-state opacity continuity, and checkpoint fields; generated, syntax checked, A800 pending |
| Relational pseudo-label supervision | analytical vertical-slice and MeshFleet contract tests | exact sparse-coordinate alignment, explicit DINO/TRELLIS provenance/confidence, finite nonzero scalar-field gradient; generated, syntax checked, A800 pending |
| Production scientific trace | `test_scientific_trace_static.py` | 16/16 locally passed: prior production guards plus TRELLIS separation, camera-exact refinement, persistence-critical proposals, computed quantization topology margin, and effective spectral policy |
| Refined-atlas gradient | `PersistentAtlasTest.test_refined_continuous_charts_retain_evidence_gradients` | finite nonzero evidence-position gradient after split/refit; generated, A800/CPU-PyTorch pending |
| Continuous metric field | partition-of-unity atlas test | strictly SPD and SE(3)-covariant at `2e-10` in float64; generated, pending |
| Atlas overlap/multilevel invariance | analytical vertical-slice test | `C0/C1`, world curvature, and hierarchy loss invariant to global SE(3) and local SO(2) at `3e-8`; generated, pending |
| Attention transport biases | atlas/mapping and gauge tests | finite edge bias, nonzero cost/visibility gradients, SE(3)/gauge covariance with active biases; generated, pending |
| Topology orientability | topology invariant test | valid tetra accepted, one-face reversal rejected; generated, pending |
| Barrier QP primal feasibility | collision/speed test | global speed factor and minimum active linearized margin `>= -10*tolerance`; generated, pending |
| Curvature-adaptive readout | analytical asset test | curved chart allocates more samples than flat chart under identical area budget; generated, pending |
| Tile opacity bound | renderer vertical slice | every rendered pixel alpha is below conservative overlapping-tile bound; generated, pending |
| Structural image losses | analytical asset test | identical images produce zero SSIM/fixed-feature loss; shifted image produces nonzero failure; generated, pending |
| Unbalanced distillation | analytical asset test | generalized KL is zero only at equal measure and positive with finite nonzero gradient under mass mismatch; generated, pending |
| Phase-C minibatch OT | analytical asset test | exact Hungarian coupling swaps two compatible targets to the lower product-manifold cost; generated, pending |
| GLB PBR material | independent reload test | one deterministic material and primitive material index zero; generated, pending |
| Rank-local RNG resume | format-6 checkpoint and distributed server suite | single-rank next-sample replay generated; scheduler-visible multi-rank independent-stream replay still A800 pending |
| Robust multiview gradients | `test_gradient_purification.py` plus Phase-F server path | cone-boundary optimality, weighted-median outlier resistance, artifact-direction removal, Fisher state round trip, manual post-purification DDP synchronization; numerical suite generated/A800 pending, production wiring static local pass |
| Quantization-scale adversary | `test_quantization.py` and Phase-F production trace | bounded log scale changes quantized forward, receives finite inner gradient, resets to zero, and reruns with restored RNG; numerical test generated/A800 pending, static wiring locally passed |
| Dimensionless safety hardening | `differentiable_feasibility_loss` and production trace | zero is every hard boundary; configurable positive relative margin has consistent units across area/orientation/separation/covariance; static wiring locally passed, numerical margin test A800 pending |
| SPD parallel transport | `TopologyAndManifoldTest.test_affine_spd_parallel_transport_is_an_isometry` | affine-invariant tangent norm preserved at `atol=2e-10, rtol=2e-10`; generated, PyTorch/A800 pending |
| Gauge-irrep transport | `TopologyAndManifoldTest.test_packed_irrep_transport_round_trip` | packed `0e+1o+2e` round trip at `3e-10`; generated, PyTorch/A800 pending |
| Phase-E activation/Jacobian distillation | production static trace plus server Phase-E backward | captures every GSTA stage, world-tensor activation loss, deterministic manifold JVP, nonzero finite student gradient and frozen teacher; static wiring locally passed, numerical/A800 pending |
| VGGT derived track cycle | `CameraConventionTest.test_vggt_derived_track_cycle_and_plane_normals` | identical calibrated views give zero at `2e-12`; perturbed depth gives positive loss and finite depth gradient; generated, PyTorch/A800 pending |
| VGGT depth normal | same camera test and production render loss | constant plane gives world `+z` normal at `2e-12`, validity excludes unsupported boundary; generated, PyTorch/A800 pending |
| Offline teacher refinement | `scripts/refine_teacher_bundle.py`, `TopologyFixedTeacherBundleRefiner`, production static trace | fixed complex, barrier-feasible state, bounded cameras, decreasing robust loss, analytical PLY/GLB, confidence in `[0,1]`; static wiring locally passed, checkpoint/A800 run pending |
| Teacher bundle contract | `load_teacher_bundle` and MeshFleet Phase-C path | schema/object/manifest/provenance/confidence validation; unavailable/low-confidence labels never create target states; generated, PyTorch/A800 pending |
| Learned perceptual provenance | `LearnedPerceptualPyramid`, checkpoint resume policy, production static trace | SHA-256 match, complete VGG16 feature keys, frozen parameters, masked feature loss; static no-download wiring locally passed, numerical/checkpoint quality A800 pending |
| TRELLIS candidate shape prior | hidden-support atlas test and production trace | zero-vote Jeffreys mean `0.5/(S+1)`, candidate Bernoulli NLL, observed evidence/persistence unchanged by prior channel; numerical generated/A800 pending, static separation locally passed |
| Sparse image-plane refinement | `PersistentAtlasTest.test_sparse_reprojection_variance_retains_camera_gradient` and production trace | two-view camera displacement produces positive dimensionless variance and a nonzero camera gradient; zero/consistent cases and full production split run generated for PyTorch/A800; camera-retention/DDP wiring passed locally as a static contract |
| Persistence-critical topology proposals | `TopologyAndManifoldTest.test_persistence_critical_proposal_thresholds` and production trace | longest-lived lower-star endpoints map deterministically to occupancy cuts before quantile/fixed fallbacks; numerical test generated for PyTorch/A800, production ordering passed locally as a static contract |
| Flow spectral policy | `GaugeEquivarianceTest.test_multiplicity_spectral_policy_scale_is_effective` and production trace | changing the configured multiplicity bound scales the irrep-safe operator, invalid domains reject; numerical PyTorch test generated/A800 pending, dead-option and ablation wiring locally static-validated |

Local recovery verification performed after the server-only directive includes
Python 3.10 bytecode compilation, deterministic manifest reconstruction, and
six PyTorch-independent manifest/topology tests. No local PyTorch 3.10+ runtime is
installed, so no forward, backward, performance, or quality result is reported
from the local machine.

The 2026-07-16 crash-integrity rerun completed with `compileall` success and
6/6 static MeshFleet tests passing under the bundled Python
3.12 runtime. Source tracing confirmed that an inadmissible raw mesh produces
false topology/Betti/persistence/stratum/manifold-certification masks, a null
Betti target, and exactly zero hard topology-supervision contribution in every
training phase; the separately named internal topology prior is unaffected.

The hidden-support increment also passes full project bytecode compilation;
the six PyTorch-independent manifest tests remain green. Its float64 atlas/UOT
invariant test and multi-rank prior broadcast test are generated for A800
execution because the local bundled runtime has no PyTorch installation.

The specification-audit repair cycle again passed full `compileall`. The six
MeshFleet static tests and six scientific production-trace static tests passed
locally (12/12 total locally executable tests). Parsing the YAML through PyYAML
could not run because the bundled local runtime lacks `yaml`; no package was
installed under the A800-first/no-local-dependency rule. All new numerical,
gradient, invariant, renderer, and DDP tests remain explicitly server-pending.

The Phase-E/F, VGGT-derived-supervision, offline-teacher, perceptual,
topology-prior, exact refinement-statistic, persistence-critical proposal, and
computed quantization-certificate and spectral-policy cycles passed whole-tree
`compileall` and all 22 locally executable static/manifest tests. The PyTorch numerical purifier,
quantization-adversary, manifold-transport, Jacobian, and margin suites were
not executed locally because the approved bundled runtime has no PyTorch.

Final local completion pass (2026-07-16): whole-tree `compileall` passed in
2.5 seconds; the 6 MeshFleet manifest tests and 16 production scientific-trace
tests all passed (22/22 in 3.078 seconds). No CUDA/PyTorch numerical result,
GPU memory figure, render, asset, or checkpoint result is inferred from these
static executions.

Pinned A800 reference repair cycle (2026-07-17): the supplied server report
executed 76 tests under PyTorch 2.4 and reported 9 errors, 4 failures, and 9
skips. All nine errors shared one invalid tensor-valued `torch.full_like`
argument in persistent atlas construction. The four independent failures were
a finite-but-overstrict fixed-feature threshold, prior-only reliability being
correctly zero, a zero-norm reprojection derivative producing NaN, and a
float64 topology margin rounded through a float32 caller scalar. Each root
cause has a production or semantically targeted test repair. The server harness
now checks the exact 444-pin `requirements.txt`, `pip check`, the declared
remote train/test root, manifest ownership/digest, and unexpected skip reasons.
Locally, the final whole-tree `compileall` rerun passed and 28/28 executable
tests passed in 3.317 s: 4 environment-contract, 6 canonical MeshFleet
manifest/topology, and 18 production trace tests. The corrected PyTorch numerical suite has not been
rerun locally or on A800; its status is server-ready and unexecuted, not passed.

Network-recovery manifest handoff cycle (2026-07-17): inspected the interrupted
validator/static-guard writes against `MANIFEST_SCHEMA =
meshfleet-trellis-object-v2`, the builder's summary contract, and the checked
local summary. No truncation or duplicate block was present. Factored the
handoff into `_inspect_manifest_contract` and `_manifest_requires_rebuild`, then
executed six pure-Python adversarial cases: stale schema, record-count drift,
wrong root, missing/duplicate canonical identity, missing/malformed summary,
and a compatible three-object manifest whose canonical object is second. All
made the expected rebuild/reuse decision. Whole-tree `compileall` and the exact
four-suite recovery command passed 34/34 tests in 3.807 s (4 environment, 6
handoff, 6 MeshFleet manifest/topology, 18 scientific trace). No remote manifest
was modified and no A800 numerical result is inferred.

Accelerator-provenance increment (2026-07-17): added pure contract checks for
CUDA availability, the pinned CUDA 11.8 PyTorch build, native BF16 support, and
visible NVIDIA A800 identity, with subprocess-only runtime probing so the
validator remains import-light until environment identity passes. Whole-tree
`compileall` and the four local recovery suites passed 35/35 in 3.465 s. The
synthetic contract test rejects CUDA 12.1, RTX 2060, absent CUDA, and absent
BF16; it does not claim that the inaccessible remote hardware has passed.

Distributed validation-entry increment (2026-07-17, revised 2026-07-20):
`validate_ddp_server.py` records exact-environment agreement, one unique
host/local CUDA assignment per scheduler-visible rank, A800/CUDA-11.8/BF16 properties, and all-rank unittest success before returning
zero. Whole-tree `compileall` and the four local static/manifest suites passed
36/36 in 3.890 s. The NCCL process-group path remains A800-unexecuted; only its
source contract and import syntax were locally validated.

Pinned training-launch increment (2026-07-17): the Phase A-F shell launcher now
uses `/mnt/sda1/miniforge3/envs/CRAFT/bin/python` (or explicit
`GRAFT_GS_PYTHON`), executes the exact-pin audit, and invokes
`torch.distributed.run` through that same interpreter. Whole-tree `compileall`
and the four local suites passed 37/37 in 4.135 s. Bash/NCCL execution remains
server-pending; the local result is a source/contract validation only.

Exact final local command:
`C:\Users\10992\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
-m compileall -q graft_gs scripts tests`, followed by
`-m unittest tests.test_environment_contract_static
tests.test_server_manifest_handoff_static tests.test_meshfleet_manifest_static
tests.test_scientific_trace_static -v`.

Exact next server command is the reference invocation in
`docs/A800_VALIDATION_PROTOCOL.md`, using
`/mnt/sda1/miniforge3/envs/CRAFT/bin/python scripts/validate_server.py` with the
declared requirements, dataset root, manifest, and JSON output arguments. Its
result remains pending and must replace—not be merged with—the failing supplied
76-test report.

Dynamic dataset discovery cycle (2026-07-17): whole-tree `compileall` passed
under the bundled Python 3.12 runtime. A real local manifest was regenerated
from `D:\VsCode\MVG\Base\MeshFleet_TRELLIS`; it found one primary candidate,
admitted that candidate through the required intersection, recorded all nine
available modalities, preserved the audited non-manifold topology contract,
and produced zero rejections. The focused command executed 43 tests: 42 passed
and one tensor-loader execution was skipped because the bundled desktop Python
has no PyTorch. The passing cases include synthetic multi-split discovery,
model-nested latent paths, incomplete-required rejection, optional-modality
retention, deterministic manifest/rejection bytes, real-manifest deterministic
regeneration, stale remote-manifest rejection, many-object order independence,
and all scientific production-path static guards. A follow-up ambiguity and
path-containment increment raised the final focused inventory to 49 tests: 48
passed and the same one PyTorch-dependent loader execution remained skipped.
It verifies that multiple optional model variants are recorded but not guessed
and that a manifest path cannot escape the configured root. The loader tensor
case is implemented and server-ready but is not reported as locally passed.

Exact local command:
`C:\Users\10992\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
-m compileall -q graft_gs scripts tests`, followed by
`-m unittest tests.test_meshfleet_dynamic_discovery
tests.test_meshfleet_manifest_static tests.test_server_manifest_handoff_static
tests.test_scientific_trace_static -v`.

Released-model integration cycle (2026-07-17): upstream VGGT, TRELLIS, and
MeshFleet producer source was re-read at the adapter boundaries. Whole-tree
`compileall` passed after checkpoint-resolution, sampler-lifetime,
source-rank-prior, and remote-path repairs. The final locally executable command
ran 53 tests: 52 passed and one PyTorch-dependent dynamic-loader test was
skipped because the bundled desktop runtime has no PyTorch. This inventory
includes five exact-environment/checkpoint-resolution tests, eight server
manifest/accelerator handoff tests, seven dynamic-discovery tests, ten checked
manifest/topology tests, and twenty-three scientific production-path static
guards. No A800/model forward result is inferred.

Server-ready additions not locally executed: `tests/test_external_adapters.py`
checks per-posterior TRELLIS injection ownership, tensor-domain rejection, and
RNG isolation under PyTorch; the updated visible-rank test supplies a prior only on
rank zero and verifies float64 typed broadcast; and
`scripts/validate_external_models.py` runs actual cached VGGT/TRELLIS
checkpoints on real dynamically selected MeshFleet views. `validate_server.py`
now fails before the main suite if either checkpoint-backed adapter preflight
fails. Expected output records are `outputs/validation/external_vggt.json`,
`external_trellis.json`, and the enclosing `reference.json`.

Bounded remote smoke-record cycle (2026-07-17): whole-tree `compileall` passed.
The final focused command executed 55 tests: 54 passed and the one
PyTorch-dependent tensor-loader test was skipped because the bundled desktop
runtime has no PyTorch. New meaningful failure cases verify that runtime
selection is not confused with a manifest catalog, an ID absent from the
selected split is named explicitly, and the checkpoint preflight applies the
production admission predicate before constructing exactly one object-level
dataset. Requirement `DATA-REMOTE-SELECT-01` is locally validated at the
manifest/orchestration boundary; actual image tensors and released checkpoint
forwards remain A800-pending.

Exact command:
`C:\Users\10992\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
-m compileall -q graft_gs scripts tests`, followed by
`-m unittest -v tests.test_meshfleet_dynamic_discovery
tests.test_scientific_trace_static tests.test_environment_contract_static
tests.test_server_manifest_handoff_static tests.test_meshfleet_manifest_static`.

Released input-domain cycle (2026-07-17): whole-tree `compileall` and the 33
locally executable scientific/dynamic-discovery tests passed with the one
expected PyTorch loader skip. Static production tracing confirms the VGGT
`[0,1]` guard. Two PyTorch mock-boundary tests were added to
`tests/test_external_adapters.py`; they are discovered by the server suite but
were not executed in the bundled no-PyTorch desktop runtime. Requirement
`UPSTREAM-INPUT-01` is therefore syntax/static validated locally and awaits
the exact pinned server Python for its numerical execution.

Dynamic deployment cycle (2026-07-20): whole-tree `compileall` passed. The
focused environment, manifest, dynamic-discovery, topology, and scientific
trace command ran 56 tests: 55 passed and one tensor-loader test was skipped
because the bundled desktop runtime has no PyTorch. Guards now fail on a fixed
`--nproc-per-node=6`, fixed configuration world size, stale parent dataset
root, missing scheduler GPU mask, or absent dynamic device-count derivation.
Requirements `DEPLOY-DYNAMIC-01` and `DEPLOY-ROOTS-01` are locally validated as
source/orchestration contracts; the real paths, visible A800 identities, NCCL
collectives, and checkpoint forwards remain server-pending.

A800 renderer/precision repair cycle (2026-07-20): the supplied remote run is
recorded as a genuine pre-repair failure, not overwritten: RGB had 14.2%
mismatched elements and max absolute error 0.6104467; `pip check` also reported
installed `jupyter-client==7.4.9` against ipykernel's `>=8.9.0` requirement.
Source inspection of the exact local TRELLIS CUDA extension established the
correct mip determinant factor, 0.1 kernel variance, NDC/pixel equation,
16-pixel tiles, alpha thresholds, and packed-covariance ABI. Whole-tree
`compileall` then passed under the bundled Python 3.12 runtime. The focused
pure/static command ran 40 tests and all 40 passed: 5 exact-environment tests,
26 scientific production-path guards, and 9 server manifest/accelerator
handoff tests. A final expanded dataset/environment/static command then ran 57
tests: 56 passed and the one PyTorch-dependent dynamic-loader case was skipped
by the no-PyTorch desktop runtime. This does not execute PyTorch numerical
rendering locally.

New server tests are implemented but unexecuted: CPU/CUDA-independent OpenCV
projection and mip-filter gradient tests, runtime precision flag/provenance
tests, the existing reference renderer backward, and an upgraded A800 CUDA
equivalence case using off-axis intrinsics, nonblack background, RGB, alpha,
visible depth, and normal cosine. The exact rerun commands and the required
`jupyter_client==8.9.1` synchronization are in
`docs/A800_VALIDATION_PROTOCOL.md`. No post-repair A800 pass, runtime, VRAM,
training convergence, or asset-quality result is claimed.

Exact locally executed commands:
`C:\Users\10992\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall -q graft_gs scripts tests`, then
`-m unittest -v tests.test_meshfleet_dynamic_discovery tests.test_meshfleet_manifest_static tests.test_server_manifest_handoff_static tests.test_scientific_trace_static tests.test_environment_contract_static`.

TRELLIS decoded-grid repair cycle (2026-07-22): the supplied checkpoint-backed
A800 preflight failed after successful sampling because 30,875 coordinate
components were outside the incorrectly declared latent resolution 16; emitted
coordinates were integer `[P,3]` with range `[0,63]`. Both overfit DDP ranks
failed at the same validation boundary, so the elastic `ChildFailedError` was
secondary rather than a distinct collective failure. Upstream source inspection
confirmed two decoder upsampling stages and a 64-cubed occupancy output.

After repair, whole-tree `compileall`, `git diff --check`, and all 26
PyTorch-independent scientific production-path guards passed locally. Seven
new PyTorch regression cases in `test_external_adapters.py` are syntax checked
but locally unexecuted because the drafting runtime has no PyTorch. The exact
CRAFT checkpoint-backed TRELLIS preflight and DDP overfit reruns remain A800
pending; no post-repair model-forward or convergence result is claimed.
The overfit entry point is now statically guarded to activate same-object view
sharding/source-only prior sampling; its NCCL collectives and global-evidence
gradients remain part of the same pending server rerun.

Same-object atlas transport cycle (2026-07-22): the post-decoded-grid A800 run
showed exactly eight 12-step sampling bars for the configured eight-draw prior,
consistent with source-rank-only sampling, then both ranks failed on the first
`levels:int16` atlas broadcast because Torch 2.4 ProcessGroupNCCL does not
support `Short`. This is recorded as a new transport failure, not a recurrence
of the TRELLIS grid defect.

The repair has locally passed whole-tree `compileall` and 26/26 scientific
static guards. New unskipped mock tests exercise int16/int8/
bool-to-int64 transport, noncontiguous and greater-than-2^53 int64 identity,
alias isolation, exact dtype restoration, split masks, and metadata mismatch
before typed collectives.

The next real two-rank A800 smoke run advanced beyond that transport and failed
at the former raw replica guard with `chart_frames: maximum error 1.000e+00`.
This is now classified as PCA gauge non-uniqueness rather than inconsistent
geometry. The DDP suite has been upgraded to inject an exact pi tangent-gauge
rotation, require bitwise source-forward equality, and backpropagate through
source broadcast plus global evidence all-gather to finite nonzero local point
gradients on every rank. CPU tests additionally cover finite zero gauge
gradients at a repeated covariance spectrum and finite nonzero derivatives at
a separated spectrum. Whole-tree compilation and 26/26 production-path static
guards pass after this repair. PyTorch numerical execution and the corrected
ProcessGroupNCCL/overfit rerun remain server-pending; no successful training
step is claimed yet.

Phase-B non-finite-state repair cycle (2026-07-22): the next supplied A800
smoke is genuine forward progress. Both ranks completed source-only TRELLIS
sampling, synchronized the persistent atlas, executed backward and an optimizer
step, and gathered a checkpoint. The second forward failed while rebuilding
the atlas because particle mass was non-finite; the former error text combined
shape, sign, and NaN and therefore did not identify the numerical domain.

The repaired path adds four numerical failure cases: an isotropic 2x2 chart
metric must reconstruct exactly and backpropagate a finite trace derivative; a
flat analytical chart must backpropagate finite evidence and curvature
gradients; a repeated-spectrum 3x3 covariance must remain inside the SPD box
with finite gradient; and NaN particle mass must be rejected by a specific
diagnostic. The distributed suite already executes the new collective
pre-optimizer finite guard and now loads all four tests on every rank.

Locally executed after repair: whole-tree `compileall` passed, followed by
31/31 PyTorch-independent exact-environment and scientific production-path
tests. The bundled local runtime has no PyTorch, so the attempted five-test
numerical command failed at import with `ModuleNotFoundError: torch`; it is
recorded as unavailable, not as a model failure or pass. The targeted
float64/gradient tests, corrected DDP gate, and a two-step checkpoint-backed
smoke remain A800-pending. No post-repair loss decrease is claimed.

The broader 57-test dataset/environment/static rerun was attempted after the
same change. It reached 39 passes and one expected no-PyTorch skip, while 17
fixture-writing cases failed uniformly with local `OSError: [Errno 28] No
space left on device` under the Windows temporary directory. These are not
recorded as code failures or passes. No cache or artifact cleanup was performed.
The non-writing 31-test gate was then rerun with
`PYTHONDONTWRITEBYTECODE=1` and passed 31/31.
