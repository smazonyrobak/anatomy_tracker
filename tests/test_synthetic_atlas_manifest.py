import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from training.synthetic_atlas import AP_MAX_UM, AP_MIN_UM, make_manifest


ROOT = Path(__file__).parents[1]


def test_manifest_augmentation_probabilities_and_ranges():
    manifest = make_manifest(50_000, "train", 73191)
    optical = manifest["flaw_mask"].any(axis=1)
    flaw_count = manifest["flaw_mask"].sum(axis=1)
    occlusion = manifest["occlusion_type"]

    assert optical.mean() == pytest.approx(0.90, abs=0.01)
    assert np.all(flaw_count[optical] >= 1)
    assert np.all(flaw_count[optical] <= 3)
    assert np.all(flaw_count[~optical] == 0)
    assert manifest["warp"].mean() == pytest.approx(0.60, abs=0.01)
    assert np.mean(occlusion == 1) == pytest.approx(0.04, abs=0.005)
    assert np.mean(occlusion == 2) == pytest.approx(0.36, abs=0.01)
    assert np.all((-180.0 <= manifest["rotation_deg"]) & (manifest["rotation_deg"] <= 180.0))
    assert np.all((0.5 <= manifest["scale"]) & (manifest["scale"] <= 1.5))
    assert np.std(manifest["scale"]) == pytest.approx(np.sqrt(1.0 / 12.0), abs=0.01)
    ap_bin_counts, _ = np.histogram(manifest["ap_um"], bins=50, range=(AP_MIN_UM, AP_MAX_UM))
    assert ap_bin_counts.max() - ap_bin_counts.min() <= 1
    assert manifest["ap_um"].min() < AP_MIN_UM + 1.0
    assert manifest["ap_um"].max() > AP_MAX_UM - 1.0


def test_manifest_is_reproducible_and_seed_changes_the_manifest():
    first = make_manifest(256, "test", 49157)
    repeat = make_manifest(256, "test", 49157)
    other = make_manifest(256, "test", 49158)

    assert first.keys() == repeat.keys()
    assert all(np.array_equal(first[name], repeat[name]) for name in first)
    assert not np.array_equal(first["ap_um"], other["ap_um"])
    assert not np.array_equal(first["warp_seed"], other["warp_seed"])
    with pytest.raises(ValueError, match="Unknown split"):
        make_manifest(1, "unknown", 0)


def test_local_model_binary_matches_metadata_and_onnx_contract():
    ort = pytest.importorskip("onnxruntime")
    model = ROOT / "models" / "AtlasPose" / "atlas_pose.onnx"
    metadata_path = model.with_suffix(".json")
    if not model.exists():
        pytest.skip("The release model is an external build asset")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with model.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == metadata["sha256"]
    with (ROOT / "source" / "atlas_pose_runtime.py").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == metadata[
            "preprocessing_source_sha256"
        ]
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "images"
    assert session.get_inputs()[0].shape[1:] == [3, 299, 299]
    assert [output.name for output in session.get_outputs()] == [
        "pose_ap_um_lr_deg_dv_deg",
        "orientation_inverted_logit",
    ]
    prediction, orientation = session.run(None, {"images": np.zeros((1, 3, 299, 299), np.float32)})
    assert prediction.shape == (1, 3)
    assert np.isfinite(prediction).all()
    assert orientation.shape == (1,)
    assert np.isfinite(orientation).all()


def test_pyinstaller_bundles_model_and_metadata():
    spec = (ROOT / "TrajectoryTracker.spec").read_text(encoding="utf-8")
    assert 'ATLAS_POSE_MODELS / "atlas_pose.onnx"' in spec
    assert 'ATLAS_POSE_MODELS / "atlas_pose.json"' in spec
