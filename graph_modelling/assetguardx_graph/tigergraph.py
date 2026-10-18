"""Local TigerGraph RESTPP adapter with immutable content-addressed records.

Schema/query installation is a separate explicit operation. No credentials are
stored in graph releases. Large histories must be split into bounded releases.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from assetguardx_graph.temporal import GraphRelease, canonical, digest


class TigerGraphError(RuntimeError):
    pass


def _transport(method, url, token, payload, *, auth_scheme="Bearer"):
    request = Request(
        url,
        method=method,
        data=canonical(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"{auth_scheme} {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=45) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        # Do not leak credentials, query contents or credential-bearing URLs.
        raise TigerGraphError(f"TigerGraph request failed: {type(exc).__name__}") from exc


def _attributes(**values):
    return {key: {"value": value} for key, value in values.items()}


class TigerGraphStore:
    def __init__(
        self, *, base_url="http://localhost:14240/restpp", token=None, transport=_transport
    ):
        parsed = urlparse(base_url)
        allow_remote = os.environ.get("ASSETGUARDX_TIGERGRAPH_ALLOW_REMOTE", "").lower() in {
            "1",
            "true",
            "yes",
        }
        local_host = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        cloud_endpoint = parsed.scheme == "https" and allow_remote
        if (
            parsed.scheme not in {"http", "https"}
            or not (local_host or cloud_endpoint)
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "TigerGraph endpoint must be localhost, or an explicitly allowed HTTPS Cloud URL"
            )
        self.base_url = base_url.rstrip("/")
        bearer_token = token or os.environ.get("ASSETGUARDX_TIGERGRAPH_TOKEN")
        database_secret = os.environ.get("ASSETGUARDX_TIGERGRAPH_SECRET")
        if bearer_token:
            self.token = bearer_token
            self.auth_scheme = "Bearer"
        elif database_secret:
            # Savanna supports a database secret directly on RESTPP endpoints.
            # This avoids putting a long-lived secret into a bearer-token URL
            # exchange; the secret is still read only from the environment.
            self.token = database_secret
            self.auth_scheme = "GSQL-Secret"
        else:
            raise ValueError(
                "ASSETGUARDX_TIGERGRAPH_TOKEN or ASSETGUARDX_TIGERGRAPH_SECRET is required"
            )
        if transport is _transport:
            self.transport = lambda method, url, credential, payload: _transport(
                method,
                url,
                credential,
                payload,
                auth_scheme=self.auth_scheme,
            )
        else:
            self.transport = transport

    def _call(self, method, path, payload=None):
        response = self.transport(method, self.base_url + path, self.token, payload)
        if not isinstance(response, dict) or response.get("error") is not False:
            raise TigerGraphError("TigerGraph returned an error or malformed response")
        results = response.get("results")
        if not isinstance(results, list):
            raise TigerGraphError("TigerGraph response is missing results")
        return results

    def _query(self, release, timestamp):
        params = urlencode(
            {
                "release_id": release.release_id,
                "release_hash": release.content_hash,
                "as_of_epoch": int(timestamp),
            }
        )
        result = self._call("GET", "/query/AssetGuardX/agx_records_v1?" + params)
        merged = {}
        for item in result:
            merged.update(item)
        if "records" not in merged or "releases" not in merged:
            raise TigerGraphError("installed query returned an incompatible schema")
        return merged

    def _upsert(self, payload):
        results = self._call("POST", "/graph/AssetGuardX?ack=all&new_vertex_only=true", payload)
        if not results:
            raise TigerGraphError("upsert did not acknowledge data")
        # `new_vertex_only=true` makes retries idempotent. TigerGraph reports
        # already-present immutable vertices/edges as skipped on a retry; the
        # canonical read-back checks below decide whether the release is truly
        # complete and content-identical.

    @staticmethod
    def _check_records(result, expected):
        actual = []
        for vertex in result["records"]:
            try:
                payload = json.loads(vertex["attributes"]["payload_json"])
                actual.append(digest(payload))
                if vertex["v_id"] != vertex["attributes"]["record_id"]:
                    raise TigerGraphError("record identity mismatch")
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise TigerGraphError("malformed graph record") from exc
        if sorted(actual) != sorted(r.content_hash for r in expected):
            raise TigerGraphError("database records disagree with canonical release")

    def load_graph_release(self, release: GraphRelease, *, batch_size: int = 300):
        if type(batch_size) is not int or not 1 <= batch_size <= 1000:
            raise ValueError("batch_size must be between 1 and 1000")
        # GSQL INT is a signed 32-bit value. 2038-01-19 is a safe query
        # ceiling for this historical project; a year-9999 sentinel overflows.
        current = self._query(release, 2_147_483_647)
        for marker in current["releases"]:
            if marker["attributes"]["content_hash"] != release.content_hash:
                raise TigerGraphError("release_id already identifies different immutable content")
        entities = {
            digest([release.content_hash, n.entity_id]): _attributes(
                entity_id=n.entity_id, kind=n.kind, name=n.name, release_hash=release.content_hash
            )
            for n in release.nodes
        }
        entity_items = list(entities.items())
        for start in range(0, len(entity_items), batch_size):
            self._upsert(
                {"vertices": {"AGXEntity": dict(entity_items[start : start + batch_size])}}
            )
        for start in range(0, len(release.records), batch_size):
            records, edges = {}, {}
            for record in release.records[start : start + batch_size]:
                record_id = digest([release.content_hash, record.content_hash])
                records[record_id] = _attributes(
                    record_id=record_id,
                    release_hash=release.content_hash,
                    observed_epoch=int(record.observed_at.timestamp()),
                    available_epoch=int(record.available_at.timestamp()),
                    payload_json=canonical(asdict(record)),
                )
                edges[record_id] = {
                    "AGX_SOURCE": {
                        "AGXEntity": {digest([release.content_hash, record.source]): {}}
                    },
                    "AGX_TARGET": {
                        "AGXEntity": {digest([release.content_hash, record.target]): {}}
                    },
                }
            self._upsert({"vertices": {"AGXRecord": records}, "edges": {"AGXRecord": edges}})
        self._check_records(self._query(release, 2_147_483_647), release.records)
        # Publish the loaded-window marker only after read-back parity succeeds.
        self._upsert(
            {
                "vertices": {
                    "AGXRelease": {
                        digest(release.release_id): _attributes(
                            release_id=release.release_id,
                            content_hash=release.content_hash,
                            start_epoch=int(release.start.timestamp()),
                            end_epoch=int(release.end.timestamp()),
                            record_count=len(release.records),
                        )
                    }
                }
            }
        )
        result = self._query(release, 2_147_483_647)
        self._require_marker(result, release)
        return release.content_hash

    @staticmethod
    def _require_marker(result, release):
        markers = result["releases"]
        if len(markers) != 1:
            raise TigerGraphError("period not loaded or release not committed")
        attributes = markers[0]["attributes"]
        expected = {
            "release_id": release.release_id,
            "content_hash": release.content_hash,
            "start_epoch": int(release.start.timestamp()),
            "end_epoch": int(release.end.timestamp()),
            "record_count": len(release.records),
        }
        if any(attributes.get(key) != value for key, value in expected.items()):
            raise TigerGraphError("loaded release identity/window disagrees with manifest")

    def graph_at(self, release: GraphRelease, timestamp: datetime, view: str):
        reference = release.graph_at(timestamp, view)  # validates window and inputs
        result = self._query(release, timestamp.timestamp())
        self._require_marker(result, release)
        expected = tuple(
            r for r in release.records if r.observed_at <= timestamp and r.available_at <= timestamp
        )
        self._check_records(result, expected)
        # Reconstruct from DB-returned records, not a hidden alternate graph.
        payload = json.loads(canonical(release.to_dict()))
        payload["records"] = [
            json.loads(v["attributes"]["payload_json"]) for v in result["records"]
        ]
        snapshot = GraphRelease.from_dict(payload).graph_at(timestamp, view)
        snapshot = replace(snapshot, release_hash=release.content_hash)
        if snapshot.content_hash != reference.content_hash:
            raise TigerGraphError("snapshot export/query parity failed")
        return snapshot

    def graph_sequence(self, release, timestamp, lookback_hours, view):
        reference = release.graph_sequence(timestamp, lookback_hours, view)
        return tuple(self.graph_at(release, item.timestamp, view) for item in reference)

    def graph_snapshots(self, release, timestamps, view):
        """Read a release once, then cache exact point-in-time snapshots locally.

        Every returned snapshot is reconstructed from TigerGraph read-back data,
        compared with the canonical release, and filtered by availability at its
        own timestamp. This avoids one REST query per hour while preserving the
        point-in-time rule. Use graph_at for a direct single-time query check.
        """
        timestamps = tuple(timestamps)
        if not timestamps:
            raise ValueError("snapshot cache requires at least one timestamp")
        for timestamp in timestamps:
            release.graph_at(timestamp, view)  # validates UTC, loaded period and view
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("snapshot cache timestamps must be unique")
        latest = max(timestamps)
        result = self._query(release, latest.timestamp())
        self._require_marker(result, release)
        eligible = tuple(
            record for record in release.records
            if record.observed_at <= latest and record.available_at <= latest
        )
        self._check_records(result, eligible)
        db_value = release.to_dict()
        db_value["records"] = [
            json.loads(vertex["attributes"]["payload_json"])
            for vertex in result["records"]
        ]
        database_release = GraphRelease.from_dict(db_value)
        snapshots = []
        for timestamp in timestamps:
            snapshot = database_release.graph_at(timestamp, view)
            snapshot = replace(snapshot, release_hash=release.content_hash)
            expected = release.graph_at(timestamp, view)
            if snapshot.content_hash != expected.content_hash:
                raise TigerGraphError("cached snapshot disagrees with canonical point-in-time graph")
            if any(record.available_at > timestamp for record in snapshot.records):
                raise TigerGraphError("future-available record entered a cached snapshot")
            snapshots.append(snapshot)
        return tuple(snapshots)
