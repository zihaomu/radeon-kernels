from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence


class RankingDecision(StrEnum):
    PROMOTE = "PROMOTE"
    KEEP_CURRENT = "KEEP_CURRENT"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass(frozen=True)
class SampleSummary:
    median_ms: float
    cv: float
    sample_count: int


@dataclass(frozen=True)
class RankingResult:
    decision: RankingDecision
    current: SampleSummary
    challenger: SampleSummary
    median_improvement_pct: float
    confidence_lower_pct: float
    confidence_upper_pct: float


def summarize(samples: Sequence[float]) -> SampleSummary:
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    values = [float(value) for value in samples]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("samples must be positive finite values")
    mean = statistics.fmean(values)
    return SampleSummary(
        median_ms=statistics.median(values),
        cv=statistics.pstdev(values) / mean,
        sample_count=len(values),
    )


def _improvement(current_ms: float, challenger_ms: float) -> float:
    return (current_ms - challenger_ms) / current_ms * 100.0


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_interval(
    current: Sequence[float],
    challenger: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if len(current) != len(challenger):
        raise ValueError("paired samples must have equal length")
    if len(current) < 2:
        raise ValueError("at least two paired samples are required")
    if iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")

    current_values = [float(value) for value in current]
    challenger_values = [float(value) for value in challenger]
    summarize(current_values)
    summarize(challenger_values)
    rng = random.Random(seed)
    improvements: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(len(current_values)) for _ in current_values]
        current_median = statistics.median(current_values[index] for index in indices)
        challenger_median = statistics.median(
            challenger_values[index] for index in indices
        )
        improvements.append(_improvement(current_median, challenger_median))
    improvements.sort()
    return _percentile(improvements, 0.025), _percentile(improvements, 0.975)


def rank_challenger(
    current: Sequence[float],
    challenger: Sequence[float],
    *,
    maximum_cv: float = 0.03,
    minimum_improvement_pct: float = 3.0,
    bootstrap_iterations: int = 2000,
    seed: int = 0,
) -> RankingResult:
    current_summary = summarize(current)
    challenger_summary = summarize(challenger)
    lower, upper = paired_bootstrap_interval(
        current,
        challenger,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    point = _improvement(current_summary.median_ms, challenger_summary.median_ms)

    if current_summary.cv > maximum_cv or challenger_summary.cv > maximum_cv:
        decision = RankingDecision.NEEDS_REVIEW
    elif lower > minimum_improvement_pct:
        decision = RankingDecision.PROMOTE
    elif upper <= 0:
        decision = RankingDecision.KEEP_CURRENT
    else:
        decision = RankingDecision.NEEDS_REVIEW

    return RankingResult(
        decision=decision,
        current=current_summary,
        challenger=challenger_summary,
        median_improvement_pct=point,
        confidence_lower_pct=lower,
        confidence_upper_pct=upper,
    )

