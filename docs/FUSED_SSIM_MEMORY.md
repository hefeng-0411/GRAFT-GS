# Fused SSIM memory contract

## Failure verdict

The 2026-08-05 Phase-B failure is a live-capacity OOM, not an allocator
fragmentation event. Rank 2 had 75.39 GiB allocated by PyTorch and only 237 MiB
free when eager SSIM requested another 590 MiB tensor at
`1 - similarity.clamp(...)`. The 2.24 GiB reserved-but-unallocated region is
far smaller than the eager SSIM working set and does not explain the retained
75.39 GiB graph.

For the logged shape `[8,24,3,518,518]`, one FP32 RGB field contains
154,554,624 values, or 618,218,496 bytes (589.58 MiB). The former expression
retained or transiently created full fields for the two means, two variances,
covariance, numerator/denominator factors, similarity, clamp, and subtraction.
Changing only the final subtraction cannot fix that lifetime.

## Kernel and adjoint

`graft_gs/kernels/fused_ssim.py` evaluates each output channel's padded 3x3
moments in registers. It emits one numerator and one spatial-weight partial per
256 pixels. A fixed hierarchical reduction produces the scalar objective; it
uses no atomics, so scheduling order cannot change the result.

For a window sample `x_j`, with `m_x=mean(x)`, the required local derivatives
are

```text
d m_x / d x_j       = 1/9
d variance_x/d x_j = 2 (x_j - m_x)/9
d covariance/d x_j = (y_j - m_y)/9.
```

Applying the product and quotient rules to the two SSIM factors gives the
exact vector-Jacobian product used by eager autograd. Backward assigns one
program lane to each input channel/pixel. That lane revisits the at most nine
SSIM windows containing its pixel, recomputes their moments, and writes its
gradient exactly once. This gather construction requires no atomic gradient
adds and retains no image-sized statistic. Clamp derivatives match PyTorch:
one on the closed interval `[-1,1]`, zero outside it. The denominator clamp
also follows `clamp_min`'s active branch.

The kernel accumulates spatial statistics in FP32 because the renderer and
geometric loss boundary are FP32 in the audited precision policy. Moving those
statistics to BF16 would change the objective and is therefore not presented
as an accuracy-preserving optimization. Triton compiles the same source to
device-specific PTX; sm_86 is measured below and sm_80 must pass the same
deployment gate before production training.

## Measured A6000 result

On the local RTX A6000 (sm_86), PyTorch 2.4/CUDA 11.8/Triton 3.0, at the exact
logged shape:

| implementation | forward retained increment | peak increment | forward + backward |
|---|---:|---:|---:|
| eager oracle | 7,423,919,104 B | 10,517,217,280 B | 0.2230 s |
| fused/recomputed | 1,024 B | 618,661,376 B | 0.0262 s |

The 618 MiB fused peak is the final output gradient, not a saved SSIM field.
Loss absolute error was `5.960464477539063e-08`; gradient relative L2 was
`3.174521350158499e-07`. Three repeated executions were bitwise identical on
the tested device.

Reproduce on one explicitly selected GPU:

```bash
CUDA_VISIBLE_DEVICES=2 "$GRAFT_GS_PYTHON" \
  scripts/benchmark_fused_ssim.py \
  --batch 8 --views 24 --height 518 --width 518 \
  --output outputs/validation/fused_ssim.json
```

The command exits nonzero on numerical mismatch or if peak allocation is not
reduced. Phase B/D/E/F startup also fails early when Triton is unavailable,
rather than silently dispatching the eager CUDA path and failing hours later.

The adjacent robust-RGB and fixed/learned-perceptual objectives now use
non-reentrant deterministic recomputation boundaries. At the same production
shape, the combined robust + fused-SSIM + fixed-perceptual tape retained only
1,024 bytes above its already-live inputs instead of 3,473,201,664 bytes. This
does not change their operators; it regenerates their internal tensors during
the adjoint and leaves the rendered image itself live for all supervision.

## Explicit non-solutions

- `empty_cache()` cannot release these live autograd tensors.
- `expandable_segments` does not reduce allocated tensor capacity.
- Whole-step CUDA Graph capture is invalid for the current data-dependent
  atlas refinement, topology selection, variable sparse shapes, host-visible
  feasibility decisions, and TRELLIS lifetime transitions. Fixed-shape
  subgraphs require a separate trace-backed optimization.
- Newton--Krylov, metriplectic, or symplectic updates do not accelerate this
  failure: step zero never reached backward or the optimizer. Substituting them
  would change the optimization trajectory and checkpoint semantics.

The log's multi-hour latency is dominated by variable geometry before the
loss (`atlas_refine` reaches 836 seconds on rank 0). Use the existing
cost-balanced sampler and profiler/scene memory trace to optimize that path;
SSIM fusion resolves the capacity failure but is not misrepresented as a cure
for atlas/topology stragglers.
