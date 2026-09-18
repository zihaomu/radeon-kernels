from __future__ import annotations

from typing import Any


def softmax_reference(torch_module: Any, x: Any) -> Any:
    return torch_module.softmax(x.float(), dim=-1).to(dtype=x.dtype)
