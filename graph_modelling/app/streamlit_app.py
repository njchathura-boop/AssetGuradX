"""Read-only historical graph explorer; no live downloads or invented forecasts."""

import json
from datetime import timedelta
from pathlib import Path

import streamlit as st

from assetguardx_graph.cli import read_release
from assetguardx_graph.demo import synthetic_release
from assetguardx_graph.temporal import canonical, neighbourhood


st.set_page_config(page_title="AssetGuard-X", layout="wide")
st.title("AssetGuard-X historical graph explorer")
st.caption("Foundation release: data quality, historical relationships, and evidence.")
mode = st.sidebar.radio("Input", ["Synthetic engineering demo", "Archived graph release"])
try:
    if mode == "Synthetic engineering demo":
        graph = synthetic_release()
        st.warning(
            "SYNTHETIC DATA — not downloaded blockchain data, predictions, or thesis results."
        )
    else:
        source = st.sidebar.text_input("Path to graph release JSON on mounted Google Drive")
        if not source:
            st.info("Select an archived graph release to explore it without a running database.")
            st.stop()
        graph = read_release(Path(source))
        st.info("Canonical archive view. This screen does not certify TigerGraph query parity.")
    total_hours = int((graph.end - graph.start).total_seconds() // 3600)
    index = st.sidebar.number_input(
        "Hour offset", min_value=0, max_value=max(0, total_hours - 1), value=0
    )
    view = st.sidebar.selectbox("Graph representation", ["exposure", "correlation", "mstg"])
    stamp = graph.start + timedelta(hours=index)
    snapshot = graph.graph_at(stamp, view)
    st.write(f"Historical time: {stamp.isoformat()}")
    st.code(
        f"release={graph.release_id}\nrelease_hash={graph.content_hash}\nsnapshot_hash={snapshot.content_hash}"
    )
    quality, linked, evidence = st.tabs(["Data quality", "Linked assets", "Evidence"])
    with quality:
        st.write(
            {
                "entities": len(snapshot.nodes),
                "relations_available_at_time": len(snapshot.records),
                "valid_relations": sum(r.quality == "valid" for r in snapshot.records),
            }
        )
        st.write("A missing relation does not establish zero financial exposure.")
        st.write(graph.provenance)
    with linked:
        entity = st.selectbox("Entity", [n.entity_id for n in snapshot.nodes])
        nodes = neighbourhood(snapshot, entity)
        node_ids = {n.entity_id for n in nodes}
        relations = [r for r in snapshot.records if r.source in node_ids and r.target in node_ids]
        st.dataframe(
            [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation,
                    "amount": str(r.amount),
                    "unit": r.unit,
                    "quality": r.quality,
                }
                for r in relations
            ]
        )
    with evidence:
        st.json(json.loads(canonical(snapshot.to_dict())))
    st.caption(
        "Forecast, intervention and research-results views await validated datasets and models."
    )
except (ValueError, OSError, KeyError) as exc:
    st.error(f"Cannot display release: {exc}")
