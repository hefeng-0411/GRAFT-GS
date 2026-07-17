# MeshFleet / TRELLIS dataset audit

## Audit boundary

The inspected source roots were `D:/VsCode/MVG/Base/MeshFleet` and the
unmodified TRELLIS `dataset_toolkits` tree. The inspected data root was
`D:/VsCode/MVG/Base/MeshFleet_TRELLIS`. The generated manifest is
`data_manifests/meshfleet_local_audit.jsonl`.

The supplied local data contains zero objects under `train` and one object
under `test`. Consequently, the requested cross-object train/test variation
audit cannot be performed from the supplied files. The loader supports flat
and category-nested modality trees, but schema variation beyond the single
physical object remains an external validation item. No additional objects or
modalities are inferred.

## Verified source lineage

MeshFleet supplies Objaverse-XL acquisition and upstream asset processing. The
files in the reconstructed dataset follow the TRELLIS toolkit, not the older
MeshFleet per-view `RT`/azimuth-elevation-distance output:

- `TRELLIS/dataset_toolkits/render.py` invokes Blender rendering and defaults
  to 150 views, 512 pixels, radius 2, and 40 degree field of view.
- `TRELLIS/dataset_toolkits/blender_script/render.py` applies one uniform
  scale equal to the inverse maximum bounding-box extent and translates the
  bounding-box center to the origin. The resulting world frame is the
  canonical unit cube `[-0.5,0.5]^3`.
- Blender cameras track local `-Z` with local `Y` up. The serialized 4x4 matrix
  is therefore an OpenGL/Blender camera-to-world transform.
- `TRELLIS/dataset_toolkits/voxelize.py` surface-voxelizes the normalized mesh
  on a 64 cubed grid and writes cell centers
  `(index + 0.5) / 64 - 0.5`. It does not create a filled volume or SDF.
- `TRELLIS/dataset_toolkits/extract_feature.py` composites RGBA on black,
  resizes to 518, converts camera axes with `c2w[:3,1:3] *= -1`, projects the
  surface voxels, samples DINOv2 ViT-L/14-reg patch tokens, and averages tokens
  across views at each sparse coordinate.
- `TRELLIS/dataset_toolkits/encode_latent.py` applies the pretrained structured
  latent encoder to sparse coordinates plus DINO tokens and stores deterministic
  encoder features.
- `TRELLIS/dataset_toolkits/encode_ss_latent.py` constructs the binary 64 cubed
  sparse surface grid and stores the pretrained structure encoder mean.
- `TRELLIS/dataset_toolkits/render_cond.py` declares 24 randomized conditioning
  cameras at 1024 pixels and chooses radius/FOV so the canonical cube fits.

No current producer for `renders_eval_70`, `renders_eval_90`, or the exact
`mesh_normalized` packaging was found in the inspected source. Their contents
are audited as evaluation RGBA/camera sets and normalized mesh data only; the
numeric suffixes are not assigned an undocumented meaning.

## Camera and normalization contract

For a serialized Blender/OpenGL camera-to-world matrix `C_gl`, the loader uses

```text
C_cv = C_gl diag(1,-1,-1,1)
E_cv = inverse(C_cv)[0:3,:]
```

Thus `E_cv` is OpenCV camera-from-world with x right, y down, and z forward,
matching VGGT. For native width `W`, height `H`, horizontal FOV `theta_x`, and
vertical FOV `theta_y`, pixel intrinsics are

```text
fx = W / (2 tan(theta_x/2)),  fy = H / (2 tan(theta_y/2))
cx = W/2,                     cy = H/2.
```

Rows zero and one are scaled by the exact output/native resize ratio. All
sample camera rotations have positive determinant and maximum orthogonality
error below `1.5e-6`. The main render manifest reports scale
`0.22145314688829665`, offset approximately
`[0, 0.0003345311, -0.1574604511]`, and the canonical cube AABB.

VGGT has a scene-level Sim(3) gauge. Training does not replace individual VGGT
cameras with ground truth. A differentiable Kabsch/scale solve removes one
global Sim(3), then transforms VGGT camera centers, rotations, depth, and world
points together. Relative pose and focal residuals remain supervised.

## Physical sample inventory

| Modality | Verified contents | Semantics / status |
|---|---|---|
| `renders` | 150/150 RGBA PNG, 512x512; transforms; mesh PLY | Direct RGB, alpha, cameras, normalized render mesh |
| `renders_cond` | 1/24 physical RGBA PNG, 1024x1024 | 23 declared images are missing; no duplication or fabrication |
| `renders_eval_70` | 12/12 RGBA PNG, 1024x1024 | Direct evaluation images/cameras; suffix meaning unknown |
| `renders_eval_90` | 12/12 RGBA PNG, 1024x1024 | Direct evaluation images/cameras; suffix meaning unknown |
| `voxels` | binary little-endian PLY, 7,996 XYZ vertices | Canonical surface cell centers, not solid occupancy |
| `features` | `indices [7996,3] uint8`, `patchtokens [7996,1024] float16` | DINOv2 aggregate pseudo-labels |
| `latents` | `coords [7996,3] uint8`, `feats [7996,8] float32` | Pretrained TRELLIS structured latent pseudo-labels |
| `ss_latents` | `mean [8,16,16,16] float32` | Pretrained TRELLIS structure posterior mean pseudo-label |
| render mesh | 78,448 vertices, 157,592 faces, normals | Direct normalized surface geometry; raw connectivity is diagnostic until topology validation |
| normalized GLB | valid glTF 2.0 binary, 688,892 bytes | Evaluation geometry after applying its scene graph |
| bounding box | min/max plus width/height/length JSON | Direct normalized bounds |

The voxel coordinates are exactly on 64 cubed cell centers with maximum
residual zero. Their recovered integer indices exactly equal both DINO feature
indices and TRELLIS latent coordinates for all 7,996 entries. This relation is
verified in the manifest rather than inferred from filenames.

The raw render-mesh complex has 236,075 unique edges, Euler characteristic
`-35`, and eight connected components. It has no boundary edges or isolated
vertices and no degenerate faces, but 313 edges have incidence four (the other
235,762 have incidence two). It is therefore neither watertight under the
strict exactly-two-faces-per-edge definition nor a closed simplicial
2-manifold. Whole-complex orientability and orientation consistency are
indeterminate/invalid on this non-manifold complex, although all incidence-two
edges have opposing raw orientation. No Betti tuple is emitted for it.

## Supervision provenance

| Loss or use | Provenance | Mathematical use |
|---|---|---|
| RGB and alpha rendering | direct | robust RGB and continuous alpha likelihood under audited cameras |
| Camera residual | direct | one-gauge Sim(3)-aligned center, SO(3), and log-focal residuals |
| Surface Chamfer | direct quantized surface | analytical Gaussian/atlas surface fidelity |
| Evidence covariance NLL | direct quantized surface | Gaussian residual likelihood with added `h^2/12 I` voxel quantization covariance |
| Confidence Brier score | derived from direct surface | smooth inlier probability at the cell half-diagonal scale |
| Phase-C target | derived from direct surface | nearest-surface attraction plus screened edge-vector preservation, followed by the hard barrier projector |
| Depth, normals, visibility | derived, not stored | the A800 nvdiffrast path treats connectivity as triangle soup, rasterizes camera-z depth and source vertex normals under exact cameras, and never presents these arrays as stored modalities |
| Raw mesh topology | direct connectivity, diagnostic only | component/incidence/Euler/orientation audit; never a hard label unless validation passes |
| Validated/repaired topology | unavailable for canonical sample | only a closed, orientable, consistently oriented validated complex may emit a confidence-weighted Betti label |
| Internal topology prior | model-derived | candidate-complex persistence/energy prior, explicitly distinct from dataset supervision |
| Teacher topology | pseudo-label | optional Phase-E distillation with configured confidence and checkpoint provenance |
| DINO/TRELLIS arrays | pseudo-label | confidence-weighted relational chart distillation; optional prior initialization only; never geometry ground truth |

RGBA alpha and evidence validity are distinct tensors. `alpha` remains a
continuous opacity target. `evidence_mask = alpha >= threshold` only controls
which image patches may emit VGGT evidence. Images are composited on black to
match the verified TRELLIS feature producer.

## Object-level tensor contract

For one selected object with `K` physically available views resized to `H,W`:

```text
images                         float32 [K,3,H,W], [0,1]
alpha                          float32 [K,1,H,W], [0,1]
evidence_mask                  bool    [K,1,H,W]
camera_to_world_opencv         float32 [K,4,4]
extrinsics_world_to_camera     float32 [K,3,4]
intrinsics                     float32 [K,3,3], processed pixels
frame_indices                  int64   [K]
surface_voxel_centers          float32 [N,3], optional/verified
surface_voxel_indices          int64   [N,3], optional/verified
trellis_patchtokens            float16 [N,1024], optional pseudo-label
trellis_latent_features        float32 [N,8], optional pseudo-label
trellis_structure_latent_mean  float32 [8,16,16,16], optional pseudo-label
dino_pseudo_supervision_mask   bool    [], explicit availability/policy gate
trellis_latent_pseudo_supervision_mask bool [], explicit availability/policy gate
dino_pseudo_confidence         float32 [], configurable pseudo-label weight
trellis_latent_pseudo_confidence float32 [], configurable pseudo-label weight
```

The manifest stores relative paths, physical/declaration counts, array
shape/dtype headers, PLY/GLB headers, relational checks, provenance, and
warnings. It never hard-codes an object identifier in loader logic.

Topology metadata is partitioned into `raw_source_mesh_topology`,
`validated_topology_ground_truth`, `repaired_topology`,
`derived_topology_statistics`, `teacher_pseudo_topology`, and `selected_label`.
The sample exposes an explicit boolean activation mask, provenance string,
confidence, and nullable target. A false mask with a non-null Betti target is a
runtime error rather than a silently ignored label.

The camera-manifest cube is also the persistent octree root. VGGT particles
outside that normalized cube remain target mass in unbalanced OT, where they
can be rejected, but are excluded from chart initialization rather than being
clamped into boundary cells. This prevents visible-evidence bounds from
removing space required by hidden-surface priors.
