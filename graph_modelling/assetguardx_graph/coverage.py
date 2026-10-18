"""Complete hourly flow grids with explicit missingness and source coverage."""

from datetime import datetime, timedelta
from decimal import Decimal

from .temporal import utc


def hours(start: datetime, end: datetime):
    utc(start)
    utc(end)
    if end <= start or any(t.minute or t.second or t.microsecond for t in (start, end)):
        raise ValueError("hour grid requires increasing whole-hour UTC bounds [start, end)")
    current = start
    while current < end:
        yield current
        current += timedelta(hours=1)


def complete_flow_grid(start, end, observations, coverage):
    """Zero is justified ONLY by explicit complete coverage and no flow records.

    Do not use this helper for prices or reserve states. Values are already
    decoded and assigned to right-closed hour boundaries upstream.
    """
    timeline = tuple(hours(start, end))
    if not set(observations) <= set(timeline) or not set(coverage) <= set(timeline):
        raise ValueError("flow/coverage timestamp outside eligible grid")
    output = []
    for stamp in timeline:
        status = coverage.get(stamp, "unavailable")
        if status not in {"complete", "unavailable", "source_failure", "stale"}:
            raise ValueError("unknown coverage status")
        value = observations.get(stamp)
        if value is not None and (not isinstance(value, Decimal) or not value.is_finite()):
            raise ValueError("flow must be a finite Decimal")
        if status == "complete":
            status = "observed" if value is not None else "no_activity"
            value = Decimal(0) if value is None else value
        else:
            value = None
        output.append({"timestamp": stamp, "value": value, "status": status})
    return output
