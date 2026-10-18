"""Forecast labels and leakage-safe temporal split primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from assetguardx_graph.temporal import utc




@dataclass(frozen=True)
class LabelWindow:
    anchor: datetime
    horizon_hours: int
    information_end: datetime
    onset: bool | None
    max_absolute_deviation: Decimal | None
    downside: Decimal | None
    premium: Decimal | None
    status: str


def label_window(anchor, deviations, *, horizon_hours, threshold, active_at_anchor):
    """Label a future sustained onset; callers must supply verified episode state.

    A breach starting at the final horizon hour needs one extra confirmation
    hour. That extra hour is explicitly included in the purge window, but not
    in severity. Unknown episode state or any missing future hour => unknown.
    Deviations map each hourly boundary to a valid Decimal or None. Stale and
    low-quality observations must be represented as None, never as zero.
    """
    utc(anchor)
    if anchor.minute or anchor.second or anchor.microsecond:
        raise ValueError("label anchor must be a whole hour")
    if type(horizon_hours) is not int or horizon_hours < 1:
        raise ValueError("positive horizon required")
    if not isinstance(threshold, Decimal) or not threshold.is_finite() or threshold <= 0:
        raise ValueError("positive finite threshold required")
    information_end = anchor + timedelta(hours=horizon_hours + 1)
    empty = (anchor, horizon_hours, information_end, None, None, None, None)
    if active_at_anchor is not False:
        return LabelWindow(
            *empty, "already_active" if active_at_anchor is True else "unknown_episode_state"
        )
    values = [deviations.get(anchor + timedelta(hours=i)) for i in range(horizon_hours + 2)]
    if any(v is None or not isinstance(v, Decimal) or not v.is_finite() for v in values):
        return LabelWindow(*empty, "unknown_coverage")
    # A breach at the anchor may be the unconfirmed start of an episode.
    if abs(values[0]) >= threshold:
        return LabelWindow(*empty, "anchor_breach")
    onset = any(
        abs(values[i]) >= threshold and abs(values[i + 1]) >= threshold
        for i in range(1, horizon_hours + 1)
    )
    future = values[1 : horizon_hours + 1]
    return LabelWindow(
        anchor,
        horizon_hours,
        information_end,
        onset,
        max(abs(v) for v in future),
        max(Decimal(0), -min(future)),
        max(Decimal(0), max(future)),
        "valid",
    )


def belongs_to_split(*, anchor, information_end, start, end):
    """Use actual future window (including confirmation/recovery), not horizon alone."""
    for stamp in (anchor, information_end, start, end):
        utc(stamp)
    if end <= start or information_end < anchor:
        raise ValueError("invalid split or label information interval")
    return start <= anchor < end and information_end < end
