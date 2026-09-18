# radeon-kernels

High-performance AI operators selected from reproducible searches on Radeon GPUs.

## Public runtime

The default GEMM API performs static dispatch only. It loads an approved
precompiled kernel when the complete environment and workload match; otherwise
it executes the manifest's deterministic PyTorch fallback. It never benchmarks,
autotunes, compiles, or calls AI on a user machine.

```python
import radeon_kernels as rk

result = rk.gemm(a, b)
key_cache, value_cache = rk.kv_cache_append(key, value, key_cache, value_cache, slots)
key_cache, value_cache = rk.kv_cache_copy(key_cache, value_cache, block_mapping)
decoded = rk.paged_attention_decode(query, key_cache, value_cache, block_tables, context_lengths)
normalized = rk.rms_norm(x, weight)
normalized, residual_out = rk.add_rms_norm(x, residual, weight)
q_out, k_out = rk.rope(q, k, cos, sin)
activated = rk.silu_mul(gate, up)
probabilities = rk.softmax(logits)
print(rk.explain_last_dispatch())
```

These APIs expose operator semantics, never tuning variants. Published winners
are selected only for an exact reviewed environment and workload; every other
call uses the declared PyTorch fallback.

List published releases, install a compatible signed archive, and verify an
installed pack with:

```bash
rk pack list
rk pack install --operator rms_norm --semantic-version 1.0 \
  --archive rk-rms-norm-<environment>-0.1.0.tar.gz
rk pack verify ~/.local/share/radeon-kernels/packs/<pack-id>
```

Install or extract kernel packs under either of these default locations:

```text
~/.local/share/radeon-kernels/packs/
<python-prefix>/share/radeon-kernels/packs/
```

For an explicit location, set `RADEON_KERNELS_PACK_PATH` to one pack directory
or an OS-path-separated list of parent directories. The runtime verifies the
Ed25519-signed pack manifest, exact ABI, evidence identity, artifact metadata, and SHA-256
before loading the extension.

The public repository contains operator contracts, search logic, dispatch data and
sanitized evidence. Real SSH aliases, machine paths, container locations and raw
results live beside the repository in a private WSL workspace.

## Local control plane

Create this sibling layout:

```text
radeon-kernel-workspace/
|-- .radeon-workspace.yaml
|-- radeon-kernels/
`-- lab-private/config/targets.yaml
```

Start from [`lab/workspace.example.yaml`](lab/workspace.example.yaml) and
[`lab/targets.example.yaml`](lab/targets.example.yaml). Then validate without
contacting a remote machine:

```bash
uv run rk doctor
uv run rk plan
uv run rk plan --target r9700-lab
```

Remote checks use a private workspace SSH trust store. First use requires an
explicit trust decision:

```bash
uv run rk doctor --remote --accept-new-host-keys
```

Later checks omit `--accept-new-host-keys` and require the saved fingerprints to
match.

Only targets listed in the private `targets.yaml` are planned. A CLI selector can
narrow that set but cannot introduce an unconfigured SSH target.

Milestone 5 implements signed pack installation, shared static dispatch, five
foundational LLM operators, dual-architecture evaluation, promotion and target
acceptance. The generic `rk run` remote scheduler remains a later milestone.
See [`docs/product-architecture.md`](docs/product-architecture.md).

Milestone 6.1 adds the stable GEMV / skinny-GEMM contract and public API:

```python
from radeon_kernels import gemv

output = gemv(x, weight)  # x[M, K] @ weight[N, K].T
```

The current manifest contains three approved gfx1201 decode winners. Calls use
a native kernel only when architecture, ABI, dtype and shape match exactly;
all other calls use `torch.mm`. Users never run the development benchmark or
choose a variant.

Milestone 6.2 fixes the paged KV-cache layout as
`[num_blocks, block_size, num_kv_heads, head_dim]`. Append addresses physical
slots, while copy operates on complete source/destination block pairs. Both
APIs mutate and return the cache tensors. The published dispatch contains 22
approved gfx1151 winners and 11 approved gfx1201 winners; unmatched workloads
use the deterministic PyTorch fallback.

Milestone 6.3 adds the stable single-query Paged Attention Decode API backed by
FP32 online-softmax native kernels. The published dispatch contains 13 approved
gfx1151 winners and 12 approved gfx1201 winners; unmatched workloads use the
deterministic PyTorch fallback, and callers never choose a candidate variant.

## GEMM experiment runner

The runner compares a PyTorch ROCm `torch.mm` baseline with several portable
Triton tile configurations. It retains every timing sample and writes its result
atomically so an interrupted run still leaves useful evidence:

```bash
python benchmarks/gemm/benchmark.py \
  --output result.json \
  --target-id local-gpu \
  --run-id manual-run
```

Machine selection, container launch and artifact collection remain private lab
concerns; the runner contains no SSH aliases, registry locations or machine paths.

## Developer Native WMMA GEMM

The explicit development path builds an architecture-specific PyTorch extension
from C++/HIP and rocWMMA. The generated `gfx1151` and `gfx1201` code objects use
hardware `v_wmma_f32_16x16x16_f16` and `v_wmma_f32_16x16x16_bf16` instructions:

```python
from radeon_kernels.ops.gemm.native import gemm, gemm_out

result = gemm(a, b, variant=8)
gemm_out(a, b, output, variant=8)
```

This API currently supports contiguous row-major FP16/BF16 matrices whose M, N
and K dimensions meet the selected tile's alignment constraints. See
[`docs/native-gemm.md`](docs/native-gemm.md) for the architecture background,
variant table and verification procedure. It is not called by the public
`rk.gemm` runtime path.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

## Promotion and signed releases

Verified private evidence is converted into a sanitized immutable proposal. A
human decision is stored separately and binds the exact proposal SHA-256. Only
an approved proposal can enter the publisher, which creates a minimal signed
pack archive plus signed public evidence, approval, compatibility and pack-index
documents. The publisher verifies that the reviewed dispatch entry has not
changed, but it does not edit the dispatch database.

The release private key is kept outside both the public repository and the
shared workspace. The repository contains only [`keys/trusted-keys.json`](keys/trusted-keys.json).
See [`developer/promotion/README.md`](developer/promotion/README.md) for the
review and publication sequence. A failed stability gate cannot be silently
converted into a normal release.

Published indexes now cover GEMM, GEMV, KV-cache append/copy, Paged Attention
Decode, RMSNorm, AddRMSNorm, RoPE, SwiGLU and Softmax. The M5 packs for gfx1151
and gfx1201, the M6.1 GEMV packs for gfx1201, and the M6.2/M6.3 packs for both
architectures passed signed-index installation, exact native winner execution
and declared fallback execution with zero compilation and zero search.

Active development does not rebuild a wheel after every promotion. Target
acceptance may synchronize the source tree and public resources directly;
wheels are built at milestone freezes and public release boundaries.
