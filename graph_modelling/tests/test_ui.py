from pathlib import Path

import pytest


def test_synthetic_explorer_starts_without_network_or_data():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(path)).run(timeout=20)
    assert not app.exception
    assert "SYNTHETIC DATA" in app.warning[0].value
    assert len(app.tabs) == 3
