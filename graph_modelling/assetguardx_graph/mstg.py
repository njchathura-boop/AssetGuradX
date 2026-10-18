"""Finite-difference bookkeeping, not an implemented DeFi loss mechanism."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable


@dataclass(frozen=True)
class MarginalLoss:
    loss_without_shock_usd: Decimal
    loss_with_shock_usd: Decimal
    downside_fraction: Decimal
    usd_per_fraction: Decimal
    method: str
    evidence: tuple[str, ...]

    @property
    def usd_per_one_percent(self):
        return self.usd_per_fraction * Decimal("0.01")


def marginal_loss(
    loss_function: Callable[[Decimal], Decimal],
    *,
    downside_fraction: Decimal,
    method: str,
    evidence: tuple[str, ...],
) -> MarginalLoss:
    if not method or not evidence:
        raise ValueError("a supported mechanism and its evidence must be declared")
    if (
        not isinstance(downside_fraction, Decimal)
        or not downside_fraction.is_finite()
        or not 0 < downside_fraction < 1
    ):
        raise ValueError("shock must be a finite fractional downside between zero and one")
    baseline, shocked = loss_function(Decimal(0)), loss_function(downside_fraction)
    if any(not isinstance(v, Decimal) or not v.is_finite() for v in (baseline, shocked)):
        raise ValueError("loss function must return finite Decimal USD values")
    return MarginalLoss(
        baseline,
        shocked,
        downside_fraction,
        (shocked - baseline) / downside_fraction,
        method,
        evidence,
    )
