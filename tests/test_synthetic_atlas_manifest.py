import hashlib
import json
from pathlib import Path

import numpy as np
import nrrd
import pytest
import torch

from source.atlas_pose_runtime import atlas_pose_preprocessing_contract_sha256
from training.synthetic_atlas import (
    AP_MAX_UM,
    AP_MIN_UM,
    APPEARANCE_MANIFEST_KEYS,
    COARSE_ANATOMY_CLASSES,
    GEOMETRY_MANIFEST_KEYS,
    TARGET_CENTER,
    TARGET_SCALE,
    SyntheticAtlas,
    make_manifest,
    paired_appearance_manifest,
)


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
    assert np.bincount(manifest["cohort"], minlength=4) / len(optical) == pytest.approx(
        [0.10, 0.45, 0.35, 0.10], abs=0.001
    )
    clean = manifest["cohort"] == 0
    assert not manifest["warp"][clean].any()
    assert not manifest["occlusion_type"][clean].any()
    assert not manifest["sensor_enabled"][clean].any()
    assert not manifest["anatomy_mix"][clean].any()
    for name in ("tilt_lr_deg", "tilt_dv_deg"):
        assert manifest[name].min() < -34.9
        assert manifest[name].max() > 34.9
        assert np.mean(np.abs(manifest[name]) > 25.0) == pytest.approx(2.0 / 7.0, abs=0.005)
    assert manifest["sensor_enabled"][manifest["cohort"] == 3].mean() > manifest["sensor_enabled"][manifest["cohort"] == 1].mean()


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


def test_paired_appearance_manifest_preserves_geometry_and_resamples_appearance():
    base = make_manifest(256, "train", 73191)
    first = paired_appearance_manifest(base, 8821)
    repeat = paired_appearance_manifest(base, 8821)
    other = paired_appearance_manifest(base, 8822)

    assert not set(GEOMETRY_MANIFEST_KEYS) & set(APPEARANCE_MANIFEST_KEYS)
    assert set(base) == set(GEOMETRY_MANIFEST_KEYS) | set(APPEARANCE_MANIFEST_KEYS)
    assert all(np.array_equal(first[key], base[key]) for key in GEOMETRY_MANIFEST_KEYS)
    assert all(np.array_equal(first[key], repeat[key]) for key in first)
    assert any(not np.array_equal(first[key], base[key]) for key in APPEARANCE_MANIFEST_KEYS)
    assert any(not np.array_equal(first[key], other[key]) for key in APPEARANCE_MANIFEST_KEYS)
    assert np.array_equal(first["occlusion_seed"], base["occlusion_seed"])
    assert np.array_equal(first["damage_mode"], base["damage_mode"])


def test_renderer_is_deterministic_preserves_pose_and_emits_quantized_rgb(tmp_path):
    ap, dv, ml = 420, 64, 64
    yy, xx = np.mgrid[:dv, :ml]
    ellipse = ((xx - 31.5) / 27.0) ** 2 + ((yy - 33.0) / 24.0) ** 2 < 1.0
    average = np.empty((ap, dv, ml), dtype=np.uint16)
    annotation = np.zeros((ap, dv, ml), dtype=np.uint32)
    for section in range(ap):
        phase = section / 23.0
        texture = 8000.0 + 2500.0 * np.cos((xx - 31.5) / 5.0 + phase) + 1800.0 * np.sin((yy - 33.0) / 4.0 - phase)
        average[section] = np.where(ellipse, texture, 0.0).clip(0, 65535).astype(np.uint16)
        annotation[section, ellipse] = 1
        annotation[section, ellipse & (xx < 31)] = 2
        annotation[section, ellipse & (((xx - 31.5) / 10.0) ** 2 + ((yy - 33.0) / 7.0) ** 2 < 1.0)] = 3
    nrrd.write(str(tmp_path / "average_template_25.nrrd"), average)
    nrrd.write(str(tmp_path / "annotation_25.nrrd"), annotation)
    (tmp_path / "query.csv").write_text(
        "id,structure_id_path\n"
        "1,/997/8/567/688/1/\n"
        "2,/997/8/567/688/1089/2/\n"
        "3,/997/1009/3/\n",
        encoding="utf-8",
    )

    manifest = make_manifest(4, "test", 9137)
    manifest["ap_index"][:] = [205.0, 255.0, 305.0, 355.0]
    manifest["tilt_lr_deg"][:] = [-12.0, -4.0, 7.0, 15.0]
    manifest["tilt_dv_deg"][:] = [8.0, -5.0, 4.0, -10.0]
    manifest["rotation_deg"][:] = [0.0, 65.0, -125.0, 178.0]
    manifest["cohort"][:] = [0, 1, 2, 3]
    manifest["occlusion_type"][:] = [0, 1, 2, 2]
    manifest["damage_mode"][:] = [0, 0, 1, 2]

    renderer = SyntheticAtlas(tmp_path, "cpu")
    intact = torch.full((4, 1, 299, 299), 0.5)
    full_mask = torch.ones_like(intact, dtype=torch.bool)
    damaged, visible = renderer._occlude(intact, full_mask, manifest, slice(0, 4))
    assert torch.equal(visible[0], full_mask[0])
    assert torch.all(visible[1:].flatten(1).sum(1) < full_mask[1:].flatten(1).sum(1))
    assert damaged[~visible & full_mask].mean() > 0.005

    first, normalized, targets = renderer.batch(manifest, 0, 4)
    repeat, _, _ = renderer.batch(manifest, 0, 4)
    extended_image, _, _, anatomy = renderer.batch(manifest, 0, 4, return_anatomy=True)
    paired = paired_appearance_manifest(manifest, 5103)
    paired_image, _, _, paired_anatomy = renderer.batch(paired, 0, 4, return_anatomy=True)
    expected = np.column_stack((manifest["ap_um"], manifest["tilt_lr_deg"], manifest["tilt_dv_deg"]))

    assert first.shape == (4, 3, 299, 299)
    assert first.dtype == torch.float32
    assert torch.equal(first, repeat)
    assert torch.equal(first, extended_image)
    assert not torch.equal(first, paired_image)
    assert torch.equal(anatomy, paired_anatomy)
    assert torch.equal(first[:, 0], first[:, 1]) and torch.equal(first[:, 1], first[:, 2])
    assert torch.equal(first.mul(255.0), first.mul(255.0).round())
    assert torch.count_nonzero(first) > 0
    assert np.array_equal(targets.numpy(), expected)
    assert np.allclose(normalized.numpy(), (expected - TARGET_CENTER) / TARGET_SCALE)
    assert len(COARSE_ANATOMY_CLASSES) == 9
    assert anatomy.shape == (4, 299, 299)
    assert anatomy.dtype == torch.int64
    assert set(torch.unique(renderer.coarse_volume).tolist()) == {0.0, 1.0, 2.0, 8.0}
    assert set(torch.unique(anatomy).tolist()) <= set(range(9))
    assert torch.all(anatomy[:, 0, 0] == 0)


def test_local_model_binary_matches_metadata_and_onnx_contract():
    ort = pytest.importorskip("onnxruntime")
    model = ROOT / "models" / "AtlasPose" / "atlas_pose.onnx"
    metadata_path = model.with_suffix(".json")
    if not model.exists():
        pytest.skip("The release model is an external build asset")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with model.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == metadata["sha256"]
    assert metadata["preprocessing_contract_sha256"] == atlas_pose_preprocessing_contract_sha256()
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
    assert '"atlas_pose.onnx", "atlas_pose.json", "RELEASE_REPORT.json", "SEALED_metrics.json"' in spec
    assert "verify_atlas_pose_model_bundle" in spec
