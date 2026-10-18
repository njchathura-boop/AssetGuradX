from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from urllib.parse import parse_qs, urlparse

import pytest

from assetguardx_graph.temporal import (
    Entity,
    TemporalRecord,
    GraphRelease,
    PeriodNotLoaded,
    canonical,
    exposure_changes,
    neighbourhood,
)
from assetguardx_graph.tigergraph import TigerGraphStore, TigerGraphError
from assetguardx_graph.coverage import complete_flow_grid
from assetguardx_graph.mstg import marginal_loss


T = datetime(2023, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


def record(**changes):
    row = TemporalRecord(
        "USDC",
        "pool",
        "pool_balance",
        "exposure",
        T,
        T,
        T + 24 * H,
        T,
        Decimal("100"),
        "USDC",
        "synthetic pool inventory",
        "synthetic_fixture_only",
        ("test:record-1",),
    )
    return replace(row, **changes)


def release(records=None):
    return GraphRelease(
        "synthetic-test-only",
        T,
        T + 24 * H,
        (Entity("pool", "dex_pool", "Synthetic pool"), Entity("USDC", "asset", "USDC")),
        records or (record(),),
        "synthetic fixture; not blockchain evidence",
    )


def test_availability_filter_and_late_arriving_observation():
    early = record()
    delayed = record(observed_at=T + H, available_at=T + 3 * H, amount=Decimal(200))
    graph = release((early, delayed))
    assert graph.graph_at(T + 2 * H, "exposure").records == (early,)
    assert graph.graph_at(T + 3 * H, "exposure").records == (delayed,)
    assert graph.graph_at(T, "mstg").records == ()
    assert graph.nodes[0].entity_id == "USDC"


def test_future_injection_does_not_change_snapshot_values():
    a = release()
    b = release((record(), record(observed_at=T + H, available_at=T + 2 * H, amount=Decimal(999))))
    assert a.graph_at(T, "exposure").records == b.graph_at(T, "exposure").records


def test_expired_latest_state_does_not_resurrect_previous_state():
    graph = release(
        (
            record(),
            record(
                observed_at=T + H,
                available_at=T + H,
                valid_until=T + 2 * H,
                amount=None,
                quality="missing",
            ),
        )
    )
    assert graph.graph_at(T + H, "exposure").to_dict()["freshness_mask"] == [False]
    assert graph.graph_at(T + 3 * H, "exposure").records == ()


def test_missing_window_and_insufficient_context_fail_explicitly():
    graph = release()
    with pytest.raises(PeriodNotLoaded):
        graph.graph_at(T - H, "exposure")
    with pytest.raises(PeriodNotLoaded):
        graph.graph_at(graph.end, "exposure")
    with pytest.raises(PeriodNotLoaded):
        graph.graph_sequence(T, 72, "exposure")


def test_roundtrip_duplicate_and_scenario_rejection():
    graph = release()
    assert GraphRelease.from_dict(json.loads(canonical(graph.to_dict()))) == graph
    with pytest.raises(ValueError, match="duplicate"):
        release((record(), record()))
    with pytest.raises(ValueError, match="scenario"):
        record(status="scenario")
    with pytest.raises(ValueError, match="units"):
        record(view="mstg")


def test_missing_relations_are_not_treated_as_zero_exposure():
    graph = release((record(valid_until=T + H),))
    changes = exposure_changes(graph.graph_at(T, "exposure"), graph.graph_at(T + H, "exposure"))
    assert changes[0]["delta"] is None
    assert changes[0]["after"] is None
    assert len(neighbourhood(graph.graph_at(T, "exposure"), "USDC")) == 2


class FakeTigerGraph:
    """Test double for transport, not evidence that GSQL compiles on a real server."""

    def __init__(self):
        self.vertices = {}

    def __call__(self, method, url, token, payload):
        assert token == "test-token"
        if method == "POST":
            count = 0
            for kind, vertices in payload.get("vertices", {}).items():
                self.vertices.setdefault(kind, {}).update(vertices)
                count += len(vertices)
            return {"error": False, "results": [{"accepted_vertices": count}]}
        params = parse_qs(urlparse(url).query)

        def attrs(values):
            return {key: val["value"] for key, val in values.items()}

        records = []
        for key, values in self.vertices.get("AGXRecord", {}).items():
            attributes = attrs(values)
            if attributes["release_hash"] == params["release_hash"][0] and max(
                attributes["observed_epoch"], attributes["available_epoch"]
            ) <= int(params["as_of_epoch"][0]):
                records.append({"v_id": key, "attributes": attributes})
        markers = [
            {"v_id": key, "attributes": attrs(values)}
            for key, values in self.vertices.get("AGXRelease", {}).items()
            if attrs(values)["release_id"] == params["release_id"][0]
        ]
        return {"error": False, "results": [{"records": records, "releases": markers}]}


def test_tigergraph_idempotence_parity_and_release_collision():
    fake = FakeTigerGraph()
    db = TigerGraphStore(token="test-token", transport=fake)
    graph = release()
    with pytest.raises(TigerGraphError, match="not loaded"):
        db.graph_at(graph, T, "exposure")
    assert db.load_graph_release(graph) == graph.content_hash
    db.load_graph_release(graph)
    assert len(fake.vertices["AGXRecord"]) == 1
    assert db.graph_at(graph, T, "exposure") == graph.graph_at(T, "exposure")
    with pytest.raises(TigerGraphError, match="different immutable"):
        db.load_graph_release(release((record(amount=Decimal(999)),)))
    fake.vertices["AGXRecord"].clear()
    with pytest.raises(TigerGraphError, match="disagree"):
        db.graph_at(graph, T, "exposure")


def test_no_commit_marker_on_failed_graph_verification():
    fake = FakeTigerGraph()

    def bad_transport(method, url, token, payload):
        response = fake(method, url, token, payload)
        if method == "GET":
            response["results"][0]["records"] = []
        return response

    db = TigerGraphStore(token="test-token", transport=bad_transport)
    with pytest.raises(TigerGraphError):
        db.load_graph_release(release())
    assert not fake.vertices.get("AGXRelease")


def test_remote_db_is_not_silently_enabled():
    with pytest.raises(ValueError, match="localhost"):
        TigerGraphStore(base_url="https://public.example", token="test")


def test_cloud_endpoint_requires_explicit_opt_in(monkeypatch):
    with pytest.raises(ValueError, match="explicitly allowed"):
        TigerGraphStore(base_url="https://demo.i.tgcloud.io:443/restpp", token="test")
    monkeypatch.setenv("ASSETGUARDX_TIGERGRAPH_ALLOW_REMOTE", "true")
    db = TigerGraphStore(
        base_url="https://demo.i.tgcloud.io:443/restpp", token="test-token", transport=lambda *_: {}
    )
    assert db.base_url == "https://demo.i.tgcloud.io:443/restpp"


def test_cloud_database_secret_uses_gsql_secret_auth(monkeypatch):
    monkeypatch.setenv("ASSETGUARDX_TIGERGRAPH_ALLOW_REMOTE", "true")
    monkeypatch.delenv("ASSETGUARDX_TIGERGRAPH_TOKEN", raising=False)
    monkeypatch.setenv("ASSETGUARDX_TIGERGRAPH_SECRET", "test-secret")
    db = TigerGraphStore(
        base_url="https://demo.i.tgcloud.io:443/restpp", transport=lambda *_: {}
    )
    assert db.auth_scheme == "GSQL-Secret"


def test_flow_grid_distinguishes_no_activity_missing_and_failure():
    grid = complete_flow_grid(
        T,
        T + 4 * H,
        {T: Decimal(10)},
        {T: "complete", T + H: "complete", T + 3 * H: "source_failure"},
    )
    assert [r["status"] for r in grid] == [
        "observed",
        "no_activity",
        "unavailable",
        "source_failure",
    ]
    assert [r["value"] for r in grid] == [Decimal(10), Decimal(0), None, None]


def test_mstg_preserves_beneficial_negative_response_and_units():
    result = marginal_loss(
        lambda shock: Decimal(50) - Decimal(100) * shock,
        downside_fraction=Decimal("0.01"),
        method="synthetic_linear",
        evidence=("toy-not-real-protocol",),
    )
    assert result.usd_per_fraction == Decimal(-100)
    assert result.usd_per_one_percent == Decimal(-1)
    with pytest.raises(ValueError):
        marginal_loss(
            lambda shock: Decimal(0), downside_fraction=Decimal(0), method="toy", evidence=("toy",)
        )
