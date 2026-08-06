# Geometry-kernel convergence and memory certificate

This note states exactly what the fused implementation proves, what is an
approximation, and which claims require target-GPU measurement. TRELLIS and
VGGT are unchanged and remain outside the trainable geometry adaptation.

## 1. Operator map

| Previous operator | Replacement | Peak distance/OT workspace |
|---|---|---|
| Nested chunked `cdist` radius support | FRNN grid-hash radius graph with fixed degree `K` | `O(NK+N+M)=O(N+M)` |
| Chunked nearest-surface `cdist` | FRNN `K=1` CUDA grid hashing | `O(N+M)` |
| Dense Chamfer blocks | KeOps symbolic `Min` reductions | `O(N+M)` |
| Dense chart partition weights | KeOps compact-bump weighted reduction | `O(N+M)` |
| Dense local Gaussian appearance weights | KeOps Gaussian weighted reductions | `O(N+M)` globally under fixed FRNN degree and fixed readout samples/chart |
| Large/small split for persistence matching | Sliced persistence Wasserstein at every size | `O(n+m)` |
| Surface Chamfer training objective | GeomLoss debiased Sinkhorn divergence, `online` KeOps backend | `O(N+M)` |
| Fused unbalanced GW structure term | KeOps edge-pair fourth-order contraction plus sparse UOT BCD | `O(E+N+M)=O(N+M)` |
| Per-view legacy CUDA rasterization calls | One batched `gsplat.rasterization` call with packed intersections, tile sort, expected depth, and fused normal features | Sparse packed raster state |
| Any future packed volumetric integration | `nerfacc` occupancy sampling, transmittance scan, and ray accumulation | Linear in occupied samples and rays |

The mapping operator needs an edgewise transport plan for chart moments.
FUGW therefore operates on the bounded FRNN support and retains the existing
implicit sparse-UOT block solver instead of replacing the plan with the scalar
result returned by `SamplesLoss`.

## 2. FRNN support certificate

Let `K` be `frnn_max_neighbors`, `N` active charts, and `M` evidence points.
FRNN returns at most `NK` radius candidates. Bidirectional coverage contributes
at most `N+M` nearest edges, hence

`E <= NK + N + M`.

For fixed `K`, edge and Sinkhorn storage are `O(N+M)`. If FRNN's last returned
neighbor is still inside a query's true radius, the row may be truncated; the
implementation raises instead of accepting a changed support. Conditional on
no saturation and no point exactly on the strict radius boundary, the FRNN
edge set equals the former radius set plus the same coverage fallbacks.

Uniform-grid hashing has expected `O(N+M+E)` construction time under bounded
cell occupancy. No fixed-radius grid method has unconditional linear time for
adversarially coincident points; the fixed-degree saturation certificate is
the production guard against that degeneration.

## 3. Symbolic distance equivalence

KeOps represents

`D_ij = sum_d (x_id - y_jd)^2`

symbolically and fuses `min_j D_ij`, `sum_j k(D_ij)v_j`, or both into map-reduce
kernels. Since no algebraic approximation is introduced, these outputs equal
the corresponding dense reductions up to floating-point reduction order.
Gradients are identical on strata with a unique nearest neighbor. At ties the
nearest map is set-valued for both implementations; the selected branch is a
valid subgradient. CUDA inputs remain on-device, and only discrete indices are
computed under `no_grad`; continuous selected-edge costs use the original
autograd tensors.

## 4. Exact KeOps FUGW contraction

Let `S={(i_e,j_e)}_{e=1}^E` be the FRNN support and let `p_e` be a coupling on
that support. For squared intra-domain costs

`C1(i,k)=||x_i-x_k||^2`, `C2(j,l)=||y_j-y_l||^2`,

the fourth-order GW operator restricted to `S` is

`T_e(p)=sum_f [C1(i_e,i_f)-C2(j_e,j_f)]^2 p_f`.

Expanding the square gives

`T_e=A_e+B_e-2Q_e`,

where `Q_e=sum_f C1(i_e,i_f)p_f C2(j_e,j_f)` is precisely
`(C1 p C2^T)_(i_e,j_e)` on the sparse coupling. The implementation represents
the two edge-indexed distances and `p_f` as KeOps `LazyTensor` variables and
computes `A`, `B`, and `Q` as three symbolic reductions. Thus the returned
operator is algebraically identical to the dense fourth-order contraction up
to floating-point reduction order, while the `E x E` formula is never stored.

The two-coupling FUGW relaxation is biconvex. Holding either coupling fixed
makes its block objective an entropic sparse UOT problem with linearized cost
`(1-alpha)M + alpha T(p)`. Each certified inner solve minimizes its block; exact
BCD is therefore monotone and every limit point is block-stationary. The code
requires the measured coupling residual to pass its configured tolerance.
The custom autograd function replays exactly the completed finite BCD map in
backward, while each UOT block uses its implicit adjoint. Since `E` is bounded
by the FRNN certificate and the BCD iteration cap is constant, peak asymptotic
storage remains `O(E+N+M)=O(N+M)`.

## 5. Sinkhorn convergence

For positive masses and entropic regularization `epsilon > 0`, the Gibbs kernel
is positive. A balanced Sinkhorn scaling is a contraction in Hilbert's
projective metric; the unbalanced updates use

`rho_s = tau_s / (tau_s + epsilon)` and
`rho_t = tau_t / (tau_t + epsilon)`,

both strictly below one, so the alternating log-scaling fixed point is
contractive. The sparse solver certifies its coupled residual before returning
and solves the transposed fixed-point system in backward, avoiding an iteration
tape. With the FRNN bound above, every forward and adjoint iterate stores
`O(E+N+M)=O(N+M)` values.

For the global surface objective, GeomLoss uses epsilon scaling and KeOps
online soft-min reductions. The debiased divergence is zero on identical
measures. As the blur tends to zero, entropic OT Gamma-converges to the
unregularized Wasserstein objective under the standard finite-moment
assumptions. A finite blur is therefore a controlled regularized objective,
not an exact unregularized Wasserstein distance.

## 6. Persistence approximation

For each projection direction, sorting gives the exact one-dimensional
Wasserstein coupling of the diagonal-augmented persistence diagrams with
linear storage. Midpoint angular quadrature converges to sliced Wasserstein as
the direction count tends to infinity. It does not converge to full
two-dimensional persistence Wasserstein; the former dense Hungarian branch
was removed to satisfy the strict linear-memory invariant.

## 7. Rasterization and volumetric composition

`gsplat` performs fused projection, tile intersection generation, radix sorting,
and front-to-back alpha composition on contiguous FP32 arenas. Packed mode
stores visible intersections rather than a camera-by-Gaussian dense state.
RGB, expected depth, alpha, and oriented normal features are produced from one
batched call. Raster visibility decisions remain piecewise differentiable,
as in every splatting rasterizer; retained contributions use native autograd.

`nerfacc` evaluates

`alpha_i = 1-exp(-sigma_i delta_i)`,
`T_i = exp(-sum_{j<i} sigma_j delta_j)`, and `w_i=T_i alpha_i`

with packed scans, followed by compiled ray accumulation. The current main
pipeline has no volumetric ray marcher, so the adapter is available without
inserting an unused occupancy grid into the Gaussian path.

## 8. Manifold stability

The refactor does not alter VGGT inference, TRELLIS prior decoding, the
SO(3)/SPD product manifold, barrier projection, or safe Heun retraction. On a
fixed feasible topology stratum, the existing bounded vector field and
backtracked barrier steps remain the diffeomorphic trajectory constraint.
Discrete FRNN support changes are topology-routing decisions outside autograd,
not modifications of the frozen baseline manifolds.

## 9. Explicit non-claims and ABI gate

- Contiguity is guaranteed at compiled boundaries. `.contiguous()` is
  zero-copy only when the caller already satisfies the layout. FRNN and gsplat
  inputs are required to be contiguous before entry, then passed through an
  explicit `.contiguous()` ABI call.
- SASS/PTX bandwidth saturation and peak VRAM require measurement on the target
  A800. Source inspection alone cannot prove a hardware percentage.
- The sibling `gsplat` source declares `torch>=2.7`, while the immutable baseline
  snapshot pins Torch 2.4. The code fails fast on ABI mismatch; do not upgrade
  the frozen TRELLIS/VGGT environment without a separate compatibility audit.

Run `python scripts/validate_geometry_extensions.py` in the target environment
before training. Install the sibling source bindings with
`requirements-geometry-local.txt` only in an ABI-compatible environment.
