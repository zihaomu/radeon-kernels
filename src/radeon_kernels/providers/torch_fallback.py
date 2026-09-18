from __future__ import annotations

from typing import Any


def gemm(torch_module: Any, a: Any, b: Any, output: Any | None = None) -> Any:
    if output is None:
        return torch_module.mm(a, b)
    return torch_module.mm(a, b, out=output)
