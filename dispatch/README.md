# Dispatch Manifests

This directory contains reviewed, public dispatch manifests. A manifest may be
added only by the promotion workflow after its evidence, compatibility bounds,
artifact hashes, workload regions, priorities, and fallbacks pass validation.

No development or private-lab result is a published winner merely because it is
present in a run directory.

`gemm-1.0.json` is the first reviewed runtime manifest. M5 adds reviewed
manifests for RMSNorm, AddRMSNorm, RoPE, SwiGLU and Softmax. Entries bind the
architecture, wave size, ROCm ABI, Python ABI, PyTorch build, workload, artifact
digest and evidence. All unmatched calls use the declared `torch` fallback;
runtime code never broadens an entry or searches on a user machine.
