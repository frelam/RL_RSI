from __future__ import annotations

from fractions import Fraction
from heapq import heapify, heappop, heappush
from math import ceil, floor, isfinite


def target_shares(weights: dict[str, float]) -> dict[str, Fraction]:
    positive: dict[str, Fraction] = {}
    for group, weight in sorted(weights.items()):
        if not isfinite(weight):
            raise ValueError(f"mixing weight for {group!r} must be finite")
        if weight > 0:
            positive[group] = Fraction(str(weight))
    total = sum(positive.values())
    return {group: weight / total for group, weight in positive.items()}


def largest_ratio_quotas(
    shares: dict[str, Fraction],
    capacities: dict[str, int],
    strict_capacities: dict[str, int],
    *,
    target_total: int,
    min_per_group: int,
    tolerance: Fraction,
) -> dict[str, int]:
    """Maximize the total under absolute share limits and integer group bounds."""
    if not shares or any(capacities[group] < min_per_group for group in shares):
        return {}

    # Each group bounds the total by the volume it can supply at its minimum share.
    lower_shares = {group: max(Fraction(0), share - tolerance) for group, share in shares.items()}
    upper_shares = {group: min(Fraction(1), share + tolerance) for group, share in shares.items()}
    limit = sum(capacities.values())
    if target_total > 0:
        limit = min(limit, target_total)
    for group, share in lower_shares.items():
        if share:
            limit = min(limit, floor(capacities[group] / share))

    # Integer rounding can make one total feasible and its neighbor infeasible.
    minimum_total = max(1, len(shares) * min_per_group)
    for total in range(limit, minimum_total - 1, -1):
        lower = {
            group: max(min_per_group, ceil(total * lower_shares[group])) for group in shares
        }
        upper = {
            group: min(capacities[group], floor(total * upper_shares[group])) for group in shares
        }
        if any(lower[group] > upper[group] for group in shares):
            continue
        if sum(lower.values()) <= total <= sum(upper.values()):
            return _fill_quotas(total, lower, upper, shares, strict_capacities)
    return {}


def _fill_quotas(
    total: int,
    quotas: dict[str, int],
    upper: dict[str, int],
    shares: dict[str, Fraction],
    strict_capacities: dict[str, int],
) -> dict[str, int]:
    # Within the feasible bounds, minimize relaxed samples, then prefer the
    # group furthest below its ideal count. The name breaks ties consistently.
    ideals = {group: total * share for group, share in shares.items()}

    def priority(group: str) -> tuple[bool, Fraction, str]:
        return quotas[group] >= strict_capacities[group], quotas[group] - ideals[group], group

    available = [priority(group) for group in shares if quotas[group] < upper[group]]
    heapify(available)
    for _ in range(total - sum(quotas.values())):
        _, _, group = heappop(available)
        quotas[group] += 1
        if quotas[group] < upper[group]:
            heappush(available, priority(group))
    return quotas
