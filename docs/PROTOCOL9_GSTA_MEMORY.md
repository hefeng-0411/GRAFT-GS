# Protocol 9: exact GSTA activation recomputation

## Incident verdict

The supplied four-rank A800 log is a CUDA capacity failure, not an NCCL fault
and not allocator fragmentation. Rank 3 has 77.49 GiB allocated by PyTorch,
only 55 MiB free at the driver, and only 237.65 MiB reserved-but-unallocated
when a further 68 MiB gather is requested. The same log says that
`expandable_segments` is unsupported on that platform.

The failing source line is only the next allocation after exhaustion. GSTA
does not construct a dense token-by-token attention matrix. It applies a
source-segment softmax to the persistent sparse atlas edges, so a layer has
`O(E D)` rather than `O(V^2)` activation complexity, where `E` is the active
directed edge count and `D` is the combined irrep/path width. The batch-8
pipeline processes variable-size scenes serially, but every scene's autograd
graph remains live until the batch loss is backpropagated. Without
recomputation, the dominant retained edge expansions therefore scale as
`O(B L E D)` for physical object batch `B` and encoder depth `L`.

## Implemented memory architecture

`GSTAConfig.activation_checkpointing` enables non-reentrant
`torch.utils.checkpoint` around the differentiable prepared-graph kernel. The
discrete adjacency is built once. During the original forward the kernel keeps
only its explicit inputs; the large transported, radial-path, attention, and
message intermediates are reconstructed during backward. This removes the
`L E D` retained term. Layer-boundary fields still require `O(L V D)` storage,
and one layer's transient work remains `O(E D)`; claiming literal constant
total memory in depth for a generic non-reversible stack would be incorrect.

Every effective spectral-normalization matrix is resolved once in the original
forward and passed as a differentiable checkpoint input. This is necessary for
physical object batches: later scene forwards advance spectral-normalization
power-iteration buffers before an earlier scene is backpropagated. Calling the
parametrization again during recomputation could otherwise use a different
effective matrix.

For one layer, write

```text
W_eff = spectral_normalize(W_raw, u, v)
y = f(x, geometry, W_eff, theta)
```

The checkpoint recomputes the same deterministic `f` from the captured
`W_eff`. Autograd first evaluates the exact vector-Jacobian products of `f`,
then follows the original differentiable `W_eff` graph back to `W_raw`.
Consequently

```text
dLoss/dW_raw = dLoss/dy * dy/dW_eff * dW_eff/dW_raw
```

is unchanged. GSTA has no stochastic operator, so RNG-state snapshots are
disabled without changing values. Tests advance every spectral buffer with a
second-object forward before the first-object backward and compare all field,
edge-bias, and parameter gradients.

The former `normalize(q) * normalize(k)` score also materialized two complete
edge/head/channel tensors. It is now evaluated as

```text
(q dot k) / (max(norm(q), epsilon) * max(norm(k), epsilon))
```

which is algebraically identical to the former `F.normalize` expression and
retains only reduced edge/head norms. A float64 test checks its values and both
adjoints against the explicit expression.

No `out=` operation or mutation of an autograd input is used. Adding
`.contiguous()` blindly would create another allocation, while differentiable
PyTorch operators generally reject `out=` variants. Dense SDPA/FlashAttention
is also not substituted: GSTA normalizes over each source's irregular sparse
neighbors and transports `l=1/l=2` values through chart connections. Densifying
that graph would change the operator or create the very `O(V^2)` tensor the
implementation avoids. FlashAttention remains available to the upstream
TRELLIS/VGGT components that implement conventional attention.

`empty_cache()` is intentionally absent from the GSTA hot path because it
cannot release live autograd tensors and would synchronize/slow training. The
existing frozen-TRELLIS lifetime boundary remains the valid macroscopic cache
release point. The training start record now reports the inherited allocator
configuration. Disposable batch probes and explicit profiler runs record
memory at every GSTA encoder layer; regular training avoids those synchronized
driver-memory queries.

## Reproducible validation

With PyTorch 2.4.0, four layers, 8,192 vertices, 24,576 directed edges including
self edges, and FP32 fields:

| Mode | Autograd-retained storage | Time (CPU) |
|---|---:|---:|
| Baseline | 580,188,592 bytes (553.31 MiB) | 0.667 s |
| Recomputed | 20,434,064 bytes (19.49 MiB) | 0.874 s |

This is a 96.48% retained-storage reduction. Output relative L2, maximum field
gradient relative L2, and maximum parameter-gradient relative L2 were all
exactly `0.0`. Run the architecture-independent audit with:

```bash
python scripts/benchmark_gsta_memory.py \
  --device cpu --vertices 8192 --layers 4 \
  --output outputs/validation/gsta-retention-cpu.json
```

The current execution container exposes no `/dev/nvidia*`, so no A6000 peak is
claimed here. On an idle A6000 or A100/A800, explicitly isolate one device and
run the allocator-backed acceptance mode:

```bash
CUDA_VISIBLE_DEVICES=2 "$GRAFT_GS_PYTHON" \
  scripts/benchmark_gsta_memory.py \
  --device cuda --vertices 8192 --layers 4 \
  --output outputs/validation/gsta-memory-gpu.json
```

The command fails unless recomputation reduces both retained storage and CUDA
peak allocation while satisfying output/gradient tolerances.

## Exact batch-8 production gate

The server YAML enables recomputation by default. The CLI Boolean flag is
included explicitly below so a different config cannot silently disable it.
Do not export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the logged
A800 stack; it already reports that feature unsupported.

First run a fresh-process, three-step batch-8 probe:

```bash
CUDA_VISIBLE_DEVICES=4,5 bash scripts/launch_a800_6gpu.sh \
  /mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  B 50000 \
  --manifest data_manifests/meshfleet_server.jsonl --split train \
  --object-batch-size 8 --global-object-batch 32 \
  --gsta-activation-checkpointing \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --initialize-from outputs/phase_a/final.pt \
  --batch-probe outputs/validation/phase-b-batch8-probe.json \
  --batch-probe-warmup-steps 1 --batch-probe-measurement-steps 2 \
  --output outputs/phase_b_batch8_probe
```

Admission requires every rank's report, finite optimizer commits, lower peak
allocation than the physical device capacity with the configured headroom, and
no capacity/collective failure. Then run the required 200-step soak using the
same arguments without `--batch-probe`, changing `--steps` to `200` and using a
fresh output directory. Only after both gates pass on the exact production GPU
pool should the 50,000-step output be launched. A6000 and A100/A800 evidence
must be recorded independently; device capacity is never hardcoded or inferred
from the other topology.
