from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Mapping


_LAST_DISPATCH: ContextVar[dict[str, Any] | None] = ContextVar(
    "radeon_kernels_last_dispatch", default=None
)


def record_dispatch(value: Mapping[str, Any]) -> None:
    _LAST_DISPATCH.set(deepcopy(dict(value)))


def get_last_dispatch() -> dict[str, Any] | None:
    value = _LAST_DISPATCH.get()
    return deepcopy(value) if value is not None else None


def clear_last_dispatch() -> None:
    _LAST_DISPATCH.set(None)


def explain_last_dispatch() -> str:
    value = get_last_dispatch()
    if value is None:
        return "No Radeon kernel dispatch has been performed in this context."

    selected = value.get("selected") or "none"
    artifact = value.get("artifact") or "none"
    evidence = value.get("evidence_id") or "none"
    fallbacks = value.get("fallbacks", [])
    fallback_text = " -> ".join(fallbacks) if fallbacks else "none"
    return "\n".join(
        (
            f"operator: {value['operator']}",
            f"architecture: {value['architecture']}",
            f"selected: {selected}",
            f"artifact: {artifact}",
            f"evidence: {evidence}",
            f"fallbacks: {fallback_text}",
            f"reason: {value['reason']}",
        )
    )
