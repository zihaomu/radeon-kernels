from radeon_kernels.ops.swiglu.api import silu_mul
from radeon_kernels.ops.swiglu.reference import silu_mul_reference

__all__ = ["silu_mul", "silu_mul_reference"]
