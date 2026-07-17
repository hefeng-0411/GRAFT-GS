# Repository and runtime audit

The current files, not repository history, are the source of truth for this
integration.  Upstream directories remain unmodified.

## VGGT

- `VGGT` is a 1024-embedding model whose aggregator returns four cached
  multiscale token tensors. Frame and global streams are concatenated by the
  released aggregator, so head-facing patch tokens are 2048-wide.
- The public forward accepts `[S,3,H,W]` or `[B,S,3,H,W]` images and returns
  pose encoding `[B,S,9]`, depth `[B,S,H,W,1]`, depth confidence
  `[B,S,H,W]`, world points `[B,S,H,W,3]`, and point confidence.
- Depth uses `exp` activation and confidence uses `expp1`; confidence is not a
  calibrated probability. GRAFT-GS therefore learns a log-confidence
  calibrator before assigning evidence mass/covariance.
- `pose_encoding_to_extri_intri` explicitly defines OpenCV camera-from-world
  `[R|t]`, x-right/y-down/z-forward, and pixel intrinsics. This convention is
  used without an axis flip in evidence unprojection.
- Released loading is `VGGT.from_pretrained("facebook/VGGT-1B")`. The local
  implementation instantiated successfully during the pre-hardware-update
  audit and contains 1,190,596,120 parameters. No post-update forward pass was
  attempted locally.
- Reuse decision: aggregator, camera head, depth head, and point head are kept;
  four 2048-wide taps are fused with learned scalar tap weights and a
  deterministically orthogonal-initialized 2048-to-1024 projection. Late
  attention/FFN maps can receive LoRA only in phases D-F.

## TRELLIS

- The released image pipeline first samples a dense sparse-structure occupancy
  latent, decodes occupied coordinates, samples a structured sparse latent, and
  independently decodes mesh, Gaussian, and radiance-field assets.
- Sparse tensors default to `spconv`; the attention backend defaults to
  `flash_attn`, with `torchsparse` and `xformers` selectable by environment.
- The official setup targets Python 3.10/PyTorch 2.4 and separately installs
  compiled spconv, flash-attention, diffoctreerast, mip-splatting,
  nvdiffrast, kaolin, and other optional extensions.
- Local import reached the optional compiled dependency boundary and was
  blocked by absent `flexicubes`. Per the A800 execution directive, no local
  dependency download or build was performed.
- Reuse decision: TRELLIS contributes an optional sampled sparse-occupancy
  shape prior and architectural initialization source. Its independent asset
  decoders are excluded because they would break the one-atlas PLY/GLB
  invariant.

## Integration and execution boundary

- `scripts/reproduce_baseline.py` calls the untouched public VGGT and TRELLIS
  pipelines and emits control artifacts on the server.
- `scripts/infer_multiview.py` runs the GRAFT-GS vertical slice and exports both
  formats from the selected atlas.
- Both inference paths optionally emit a computed metric topology-margin
  quantization certificate when supplied measured query/key error and a
  provenance-explicit downstream field Lipschitz upper bound. MeshFleet uses
  `no_grad`, not `inference_mode`, because barrier checks require local
  JVP/Jacobian differentiation.
- `scripts/train_a800.py` is the six-phase `torchrun` entry point. Normal mode
  uses object-level DDP. `--same-object-view-shards` makes every rank iterate
  the same object order, rank-shards its views, broadcasts discrete atlas
  decisions, and differentiably sums continuous sufficient statistics.
- Local Windows/WSL is treated as an editor only. Since the hardware update,
  validation is restricted to parsing/bytecode compilation. Checkpoint
  downloads, forward/backward execution, profiling, and extension builds are
  intentionally deferred to the 6x A800 server.

## 2026-07-16 production-path audit

- Training and inference both enter `GraftGS.forward`; there is no alternate
  simplified asset decoder. Phase-specific execution stages now stop only at
  mathematically intentional boundaries.
- Sparse UOT output is consumed by chart writing, adaptive refinement, the
  OT/uncertainty attention bias, topology occupancy, continuous state, and
  analytical readout. The previously optional attention biases and two octree
  split statistics are no longer dead production options.
- Persistent charts retain conditional continuous gradients after a split;
  overlap and ancestor data now have active losses. The discrete Morton and
  topology choices remain nondifferentiable by design.
- Every manifold state is backed by an incidence-valid, consistently oriented
  explicit complex. Both flow predictor and corrector are checked against the
  same hard feasibility family.
- PLY and GLB remain one-state outputs. GLB adds only a deterministic PBR
  material; it does not call TRELLIS decoding, marching cubes, or a second mesh
  representation.
- Checkpoint format 5 makes exact resume objective-, world-size-, and Phase-F
  Fisher-state-aware and stores rank-local RNG streams rather than cloning
  rank zero onto all ranks.
- The exhaustive requirement mapping and honest status classification are in
  `SPECIFICATION_TRACEABILITY.md`; remaining work is separated in
  `UNRESOLVED_BLOCKERS.md`.
