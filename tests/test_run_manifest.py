import json

from gems.run_manifest import build_run_manifest


def test_run_manifest_hashes_config_data_and_artifacts(tmp_path):
    config = tmp_path / "config.yaml"
    data = tmp_path / "data.json"
    artifact = tmp_path / "prediction.bin"
    config.write_text("seed: 7\n")
    data.write_text(json.dumps({"manifest_sha256": "dataset-v1"}))
    artifact.write_bytes(b"prediction")

    manifest = build_run_manifest(
        run_id="exp-001",
        config_path=config,
        data_manifest_path=data,
        git_commit="abc123",
        git_dirty=False,
        artifacts=[artifact],
        hypothesis="test hypothesis",
        command="python train.py",
    )

    assert manifest["run_id"] == "exp-001"
    assert manifest["code"] == {"git_commit": "abc123", "git_dirty": False}
    assert manifest["data_manifest"]["manifest_sha256"] == "dataset-v1"
    assert len(manifest["config"]["sha256"]) == 64
    assert len(manifest["artifacts"][0]["sha256"]) == 64


def test_run_manifest_rejects_invalid_data_manifest_json(tmp_path):
    config = tmp_path / "config.yaml"
    data = tmp_path / "data.json"
    config.write_text("seed: 7\n")
    data.write_text("not json")

    try:
        build_run_manifest(
            run_id="exp-002",
            config_path=config,
            data_manifest_path=data,
            git_commit="abc123",
            git_dirty=False,
        )
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:
        raise AssertionError("expected invalid JSON to be rejected")
