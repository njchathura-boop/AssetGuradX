"""Canonical historical graph semantics, shared by export, UI and DB checks.

This small in-memory implementation is a bounded-window validator, not a
replacement for TigerGraph or a loader for the entire historical dataset.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import cached_property


SCHEMA_VERSION = "assetguardx-temporal-graph/v1"
VIEWS = {"correlation", "exposure", "mstg"}
QUALITIES = {"valid", "missing", "stale", "unsupported", "source_failure"}
ENTITY_KINDS = {
    "asset",
    "reference_asset",
    "dex_pool",
    "lending_market",
    "staking_contract",
    "wrapping_contract",
    "oracle_feed",
}


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("graph timestamps must be timezone-aware UTC")
    if value.microsecond:
        raise ValueError("graph v1 timestamps require whole-second precision")
    return value.astimezone(timezone.utc)


def canonical(value) -> str:
    def encode(item):
        if isinstance(item, datetime):
            return utc(item).isoformat()
        if isinstance(item, Decimal):
            if not item.is_finite():
                raise ValueError("non-finite graph amount")
            return str(item)
        raise TypeError(type(item).__name__)

    return json.dumps(value, default=encode, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class PeriodNotLoaded(ValueError):
    pass


@dataclass(frozen=True)
class Entity:
    entity_id: str
    kind: str
    name: str

    def __post_init__(self):
        if not self.entity_id or not self.name or self.kind not in ENTITY_KINDS:
            raise ValueError("invalid graph entity")


@dataclass(frozen=True)
class TemporalRecord:
    """A state feature (source == target) or a directional economic relation.

    interval end is exclusive; availability is independently enforced. Explicit
    missing records supersede old observations rather than forward-filling them.
    """

    source: str
    target: str
    relation: str
    view: str
    observed_at: datetime
    available_at: datetime
    valid_until: datetime
    valuation_at: datetime
    amount: Decimal | None
    unit: str
    meaning: str
    method: str
    evidence: tuple[str, ...]
    quality: str = "valid"
    status: str = "observed"

    def __post_init__(self):
        for stamp in (self.observed_at, self.available_at, self.valid_until, self.valuation_at):
            utc(stamp)
        if self.available_at < self.observed_at or self.valid_until <= self.observed_at:
            raise ValueError("invalid observation/availability/validity interval")
        if self.valuation_at > self.available_at:
            raise ValueError("valuation uses future information")
        if self.view not in VIEWS or self.quality not in QUALITIES:
            raise ValueError("unknown graph view or quality")
        if self.status not in {"observed", "reconstructed", "derived", "unavailable"}:
            raise ValueError("scenario/assumed records cannot enter a historical graph")
        if not all(
            (
                self.source,
                self.target,
                self.relation,
                self.unit,
                self.meaning,
                self.method,
                self.evidence,
            )
        ):
            raise ValueError("graph records require direction, meaning, units, method and evidence")
        if self.amount is not None and (
            not isinstance(self.amount, Decimal) or not self.amount.is_finite()
        ):
            raise ValueError("amount must be a finite Decimal or missing")
        if self.quality == "valid" and self.amount is None:
            raise ValueError("valid record requires an amount")
        if self.view == "mstg" and self.unit != "USD_loss_per_fractional_downside_shock":
            raise ValueError("MSTG must use explicit marginal loss units")

    @property
    def key(self):
        return self.source, self.target, self.relation, self.view

    @cached_property
    def content_hash(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class Snapshot:
    release_id: str
    release_hash: str
    timestamp: datetime
    view: str
    nodes: tuple[Entity, ...]
    records: tuple[TemporalRecord, ...]

    def to_dict(self):
        indexes = {node.entity_id: i for i, node in enumerate(self.nodes)}
        return {
            "schema_version": SCHEMA_VERSION,
            "release_id": self.release_id,
            "release_hash": self.release_hash,
            "timestamp": self.timestamp,
            "view": self.view,
            "nodes": [asdict(n) for n in self.nodes],
            "records": [asdict(r) for r in self.records],
            "edge_index": [[indexes[r.source], indexes[r.target]] for r in self.records],
            "freshness_mask": [r.quality == "valid" for r in self.records],
            "age_seconds": [
                int((self.timestamp - r.observed_at).total_seconds()) for r in self.records
            ],
        }

    @cached_property
    def content_hash(self):
        return digest(self.to_dict())


@dataclass(frozen=True)
class GraphRelease:
    release_id: str
    start: datetime
    end: datetime
    nodes: tuple[Entity, ...]
    records: tuple[TemporalRecord, ...]
    provenance: str

    def __post_init__(self):
        utc(self.start)
        utc(self.end)
        if not self.release_id or not self.provenance or self.end <= self.start:
            raise ValueError("release identity, provenance and nonempty window required")
        ids = {node.entity_id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate entity ID")
        keys = set()
        for record in self.records:
            if record.source not in ids or record.target not in ids:
                raise ValueError("record endpoint not declared")
            key = (*record.key, record.observed_at, record.available_at)
            if key in keys:
                raise ValueError("duplicate/conflicting temporal record")
            keys.add(key)
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes, key=lambda n: n.entity_id)))
        object.__setattr__(
            self,
            "records",
            tuple(sorted(self.records, key=lambda r: (*r.key, r.observed_at, r.available_at))),
        )

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "release_id": self.release_id,
            "start": self.start,
            "end": self.end,
            "provenance": self.provenance,
            "nodes": [asdict(n) for n in self.nodes],
            "records": [asdict(r) for r in self.records],
        }

    @cached_property
    def content_hash(self):
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported graph release schema")
        records = []
        for raw in value["records"]:
            row = dict(raw)
            for name in ("observed_at", "available_at", "valid_until", "valuation_at"):
                row[name] = datetime.fromisoformat(row[name])
            row["amount"] = Decimal(row["amount"]) if row["amount"] is not None else None
            row["evidence"] = tuple(row["evidence"])
            records.append(TemporalRecord(**row))
        return cls(
            value["release_id"],
            datetime.fromisoformat(value["start"]),
            datetime.fromisoformat(value["end"]),
            tuple(Entity(**n) for n in value["nodes"]),
            tuple(records),
            value["provenance"],
        )

    def graph_at(self, timestamp: datetime, view: str) -> Snapshot:
        utc(timestamp)
        if not self.start <= timestamp < self.end:
            raise PeriodNotLoaded(f"period not loaded: {timestamp.isoformat()}")
        if view not in VIEWS:
            raise ValueError("unknown graph view")
        selected = {}
        for record in self.records:
            if (
                record.view == view
                and record.observed_at <= timestamp
                and record.available_at <= timestamp
            ):
                prior = selected.get(record.key)
                if prior is None or (record.observed_at, record.available_at) > (
                    prior.observed_at,
                    prior.available_at,
                ):
                    selected[record.key] = record
        # Do not resurrect older valid records after the latest one expires.
        records = tuple(r for _, r in sorted(selected.items()) if timestamp < r.valid_until)
        active_ids = {endpoint for record in records for endpoint in (record.source, record.target)}
        active_nodes = tuple(node for node in self.nodes if node.entity_id in active_ids)
        return Snapshot(self.release_id, self.content_hash, timestamp, view, active_nodes, records)

    def graph_sequence(self, timestamp: datetime, lookback_hours: int, view: str):
        if type(lookback_hours) is not int or lookback_hours < 1:
            raise ValueError("lookback_hours must be positive")
        return tuple(
            self.graph_at(timestamp - timedelta(hours=offset), view)
            for offset in reversed(range(lookback_hours))
        )


def neighbourhood(snapshot: Snapshot, entity_id: str, hops: int = 2):
    if hops not in (1, 2, 3):
        raise ValueError("exploration supports one to three economic hops")
    if entity_id not in {n.entity_id for n in snapshot.nodes}:
        raise ValueError("unknown entity")
    visited = {entity_id}
    for _ in range(hops):
        reached = set(visited)
        for record in snapshot.records:
            if record.quality == "valid" and (record.source in visited or record.target in visited):
                reached.update((record.source, record.target))
        visited = reached
    return tuple(n for n in snapshot.nodes if n.entity_id in visited)


def exposure_changes(before: Snapshot, after: Snapshot):
    if (before.release_hash, before.view) != (
        after.release_hash,
        after.view,
    ) or after.timestamp <= before.timestamp:
        raise ValueError("changes require ordered snapshots of the same release/view")
    left, right = ({r.key: r for r in s.records} for s in (before, after))
    changes = []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        delta = None
        if a and b and a.quality == b.quality == "valid" and a.unit == b.unit:
            delta = b.amount - a.amount
        changes.append(
            {
                "relation_key": key,
                "before": a.amount if a else None,
                "after": b.amount if b else None,
                "delta": delta,
                "status": "comparable" if delta is not None else "unavailable_or_incomparable",
            }
        )
    return changes
