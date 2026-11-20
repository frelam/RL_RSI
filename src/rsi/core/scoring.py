from __future__ import annotations

import math
import re
from typing import Any

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_SPACE_RE = re.compile(r"\s+")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if n <= 0 or k <= 0:
        return 0.0
    c = max(0, min(c, n))
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip().lower())


def score_text(
    prediction: str,
    reference: str | None,
    scorer: str,
    metadata: dict[str, Any] | None = None,
) -> float:
    metadata = metadata or {}
    if reference is None:
        return 0.0
    if scorer == "exact":
        return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0
    if scorer == "contains":
        return 1.0 if normalize_text(reference) in normalize_text(prediction) else 0.0
    if scorer == "regex":
        return 1.0 if re.search(reference, prediction, flags=re.IGNORECASE | re.DOTALL) else 0.0
    if scorer == "numeric":
        target = _NUMBER_RE.findall(reference)
        predicted = _NUMBER_RE.findall(prediction)
        if not target or not predicted:
            return 0.0
        tolerance = float(metadata.get("tolerance", 1e-4))
        try:
            return 1.0 if abs(float(target[-1]) - float(predicted[-1])) <= tolerance else 0.0
        except ValueError:
            return 0.0
    raise ValueError(f"Unknown scorer {scorer!r}")
