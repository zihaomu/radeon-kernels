from __future__ import annotations

from typing import Any


def silu_mul_reference(torch_module: Any, gate: Any, up: Any) -> Any:
    gate_fp32 = gate.float()
    result = gate_fp32 * torch_module.sigmoid(gate_fp32) * up.float()
    return result.to(dtype=gate.dtype)
