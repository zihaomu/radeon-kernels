from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DType(StrEnum):
    FP16 = "fp16"
    BF16 = "bf16"


class Layout(StrEnum):
    NN = "nn"
    NT = "nt"
    TN = "tn"


@dataclass(frozen=True)
class ShapeRange:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum <= 0 or self.maximum < self.minimum:
            raise ValueError("shape range must be positive and ordered")

    def contains(self, value: int) -> bool:
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class GemmWorkload:
    id: str
    dtype: DType
    layout: Layout
    m: ShapeRange
    n: ShapeRange
    k: ShapeRange
    seed: int = 0
    warmups: int = 10
    samples: int = 30
    absolute_tolerance: float = 1e-2
    relative_tolerance: float = 1e-2
    maximum_cv: float = 0.03
    minimum_improvement_pct: float = 3.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("workload id must not be empty")
        if self.warmups < 1 or self.samples < 2:
            raise ValueError("workload requires warmups and at least two samples")
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("tolerances must be non-negative")
        if not 0 < self.maximum_cv < 1:
            raise ValueError("maximum_cv must be between zero and one")
        if self.minimum_improvement_pct < 0:
            raise ValueError("minimum improvement must be non-negative")

