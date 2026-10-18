import json

import pytest

from assetguardx_graph.cli import immutable_json, main, read_release
from assetguardx_graph.demo import synthetic_release


def test_offline_demo_is_explicitly_synthetic_and_roundtrips(tmp_path, capsys):
    output = tmp_path / "synthetic-release.json"
    assert main(["offline-demo", "--output", str(output)]) == 0
    assert "SYNTHETIC - NOT RESEARCH RESULTS" in capsys.readouterr().out
    assert read_release(output) == synthetic_release()
    assert main(["graph-check", "--input", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["database_verified"] is False


def test_graph_artifacts_are_immutable(tmp_path):
    path = tmp_path / "release.json"
    release = synthetic_release()
    immutable_json(path, release.to_dict())
    immutable_json(path, release.to_dict())
    with pytest.raises(ValueError, match="overwrite immutable"):
        immutable_json(path, {"different": "content"})
