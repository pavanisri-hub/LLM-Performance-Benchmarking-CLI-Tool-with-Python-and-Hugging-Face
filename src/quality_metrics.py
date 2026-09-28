from __future__ import annotations

import re
from typing import NamedTuple


class QualityResult(NamedTuple):
    metric_name: str
    score: float
    total_tokens: int
    unique_tokens: int


def compute_quality(text: str) -> QualityResult:
    words = _tokenize(text)

    if not words:
        return QualityResult(
            metric_name="TTR",
            score=0.0,
            total_tokens=0,
            unique_tokens=0,
        )

    total = len(words)
    unique = len(set(w.lower() for w in words))
    ttr = unique / total

    return QualityResult(
        metric_name="TTR",
        score=round(ttr, 6),
        total_tokens=total,
        unique_tokens=unique,
    )


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s']", " ", text)
    tokens = cleaned.split()
    return [t for t in tokens if len(t) > 0]
