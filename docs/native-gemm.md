# Native Radeon WMMA GEMM

## Scope

The native GEMM experiment exposes a Python API backed by a C++/HIP extension.
The extension uses rocWMMA fragments, which lower to architecture-specific AMDGPU
matrix instructions. It does not call rocBLAS, hipBLASLt or `torch.mm` internally.

The initial implementation supports:

- `gfx1151` (RDNA 3.5, Radeon 8060S class);
- `gfx1201` (RDNA 4, Radeon RX 9070/Radeon AI PRO R9700 class);
- contiguous row-major NN GEMM;
- FP16 or BF16 inputs and output with FP32 accumulation;
- dimensions aligned to the selected macro tile and K panel.

## Architecture background

Both targets execute rocWMMA kernels in wave32 mode and provide 16x16x16 WMMA
operations. They do not share the same operand ABI:

- gfx11 WMMA consumes replicated 256-bit A and B operands;
- gfx12 WMMA uses the newer 128-bit operand form and does not require that
  replication.

The source therefore targets the detected architecture exactly through
`PYTORCH_ROCM_ARCH`; a generic or cross-architecture code object is not reused.
Useful primary references are the
[rocWMMA API guide](https://rocm.docs.amd.com/projects/rocWMMA/en/docs-7.14.0/api-reference/api-reference-guide.html),
[Clang AMDGPU builtin reference](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html),
[AMD RDNA 3 WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/), and
[AMD RDNA 4 matrix-core guide](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/).

## Kernel design

Each workgroup cooperatively stages row-major A and B panels from global memory
into LDS. B is transposed while entering LDS so every wave can load the fragment
layout required by WMMA. Two LDS buffers overlap the next global load with the
current matrix multiply-accumulate work.

The search currently spans macro tiles from 64x128 through 256x32 and K panels of
16, 32 and 64. K=32 generally gives the best balance: it halves workgroup barriers
relative to K=16 without the register and LDS pressure observed at K=64.

The implementation is in
`src/radeon_kernels/ops/gemm/native/wmma_gemm.hip`. The Python loader and validated
API are in `src/radeon_kernels/ops/gemm/native.py`.

## Python API

```python
from radeon_kernels.ops.gemm.native import VARIANT_NAMES, gemm, gemm_out

# Allocating form.
c = gemm(a, b, variant=8)

# Fixed-output form for repeated execution and honest kernel timing.
c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
gemm_out(a, b, c, variant=8)
```

The extension is compiled lazily and cached by PyTorch under an architecture-
specific module name. The fixed-output API is important for benchmarking: compare
it with `torch.mm(a, b, out=output)` so allocation is excluded on both sides.

## ISA verification

After building, extract and disassemble the device image rather than assuming an
intrinsic survived optimization:

```bash
llvm-objdump --offloading radeon_kernels_wmma_gfx1201.so
llvm-objdump --mcpu=gfx1201 -d \
  radeon_kernels_wmma_gfx1201.so.0.hipv4-amdgcn-amd-amdhsa--gfx1201 \
  | grep v_wmma
```

A valid FP16/BF16 build must contain the matching
`v_wmma_f32_16x16x16_f16` and `v_wmma_f32_16x16x16_bf16` instructions. Benchmark
promotion still requires correctness, stable timing and the configured confidence
threshold; merely containing WMMA instructions is not a performance claim.

## Current limitations

- There is no edge kernel or padding fallback for arbitrary dimensions.
- Selection is explicit by variant; production dispatch is not yet published.
- Only alpha=1, beta=0 and row-major NN are implemented.
- JIT compilation requires the rocWMMA headers and HIP compiler matching the
  active PyTorch/ROCm environment.
