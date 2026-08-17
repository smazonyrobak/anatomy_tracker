from pathlib import Path

import numpy as np
import pytest
import torch

from training.atlas_pose_models_v7 import TILT_MAX_DEG, TILT_MIN_DEG
from training.joint_pose_registration_data import (
    AP_OFFSET_LEVELS_UM,
    TILT_OFFSET_LEVELS_DEG,
    JointSyntheticData,
    load_joint_manifest,
    make_joint_manifest,
    registration_manifest,
    save_joint_manifest,
)
from training.synthetic_registration import (
    BREGMA_AP_INDEX,
    VOXEL_UM,
    SyntheticRegistrationGenerator,
    split_ap_indices,
)


ATLAS = Path(__file__).resolve().parents[1] / "data" / "Allen Brain Atlas 25um"


@pytest.fixture(scope="module")
def generator():
    if not (ATLAS / "average_template_25.nrrd").is_file():
        pytest.skip("Local Allen atlas is not installed")
    return SyntheticRegistrationGenerator(ATLAS, "cpu")


def _assert_manifest_equal(first: dict, second: dict) -> None:
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], np.ndarray):
            assert np.array_equal(first[key], second[key]), key
        else:
            assert first[key] == second[key], key


def test_joint_manifest_is_reproducible_hashed_and_seed_sensitive(generator):
    first = make_joint_manifest(generator, 12, "validation", 8101, "mild", 6)
    repeated = make_joint_manifest(generator, 12, "validation", 8101, "mild", 6)
    changed = make_joint_manifest(generator, 12, "validation", 8102, "mild", 6)

    _assert_manifest_equal(first, repeated)
    assert first["joint_manifest_sha256"] != changed["joint_manifest_sha256"]
    assert first["registration_manifest_sha256"] == registration_manifest(first)["manifest_sha256"]
    assert first["split"] == registration_manifest(first)["split"] == "validation"
    assert first["artifact_stratum"] == registration_manifest(first)["stratum"] == "mild"
    assert np.array_equal(first["hard_negative_ap_levels_um"], AP_OFFSET_LEVELS_UM)
    assert np.array_equal(first["hard_negative_tilt_levels_deg"], TILT_OFFSET_LEVELS_DEG)


def test_initial_and_wrong_poses_are_nonzero_discrete_split_safe_offsets(generator):
    manifest = make_joint_manifest(generator, 64, "train", 8103, "hard", 12)
    base = registration_manifest(manifest)
    split_pool = set(split_ap_indices("train").tolist())
    initial = manifest["initial_pose_offset"]
    wrong = manifest["wrong_candidate_offset"]

    assert initial.shape == (64, 3)
    assert wrong.shape == (64, 12, 3)
    assert np.all(np.count_nonzero(wrong, axis=2) == 1)
    assert np.all(np.any(np.abs(wrong[:, :, 0]) == VOXEL_UM, axis=1))
    assert np.all(np.abs(wrong[:, 0, 0]) == VOXEL_UM)
    assert np.all(wrong[:, 0, 1:] == 0.0)
    assert np.all(np.abs(wrong[:, 1, 1]) == TILT_OFFSET_LEVELS_DEG[0])
    assert np.all(wrong[:, 1, (0, 2)] == 0.0)
    assert np.all(np.abs(wrong[:, 2, 2]) == TILT_OFFSET_LEVELS_DEG[0])
    assert np.all(wrong[:, 2, :2] == 0.0)
    assert np.all(np.any(wrong[:, :, 1] != 0.0, axis=1))
    assert np.all(np.any(wrong[:, :, 2] != 0.0, axis=1))
    assert not manifest["wrong_candidate_dense_target_valid"].any()
    assert set(np.abs(initial[:, 0])) <= set(AP_OFFSET_LEVELS_UM)
    assert set(np.abs(initial[:, 1])) <= set(TILT_OFFSET_LEVELS_DEG)
    assert set(np.abs(initial[:, 2])) <= set(TILT_OFFSET_LEVELS_DEG)
    for axis in (1, 2):
        values = np.abs(wrong[:, :, axis][wrong[:, :, axis] != 0])
        assert set(values) <= set(TILT_OFFSET_LEVELS_DEG)
    ap_values = wrong[:, :, 0][wrong[:, :, 0] != 0]
    assert set(np.abs(ap_values)) <= set(AP_OFFSET_LEVELS_UM)

    true_pose = manifest["true_pose"]
    for poses in (true_pose + initial, true_pose[:, None] + wrong):
        ap = poses[..., 0].reshape(-1)
        indices = np.rint(BREGMA_AP_INDEX - ap / VOXEL_UM).astype(int)
        assert set(indices.tolist()) <= split_pool
        assert np.all((TILT_MIN_DEG <= poses[..., 1]) & (poses[..., 1] <= TILT_MAX_DEG))
        assert np.all((TILT_MIN_DEG <= poses[..., 2]) & (poses[..., 2] <= TILT_MAX_DEG))
    assert set(base["ap_index"].astype(int).tolist()) <= split_pool


def test_raw_pose_view_similarity_is_independent_and_spans_its_full_domain(generator):
    manifest = make_joint_manifest(generator, 2048, "train", 8112, "mild", 3)
    rotation = manifest["pose_view_rotation_deg"]
    scale = manifest["pose_view_scale"]
    assert rotation.dtype == scale.dtype == np.float32
    assert -180.0 <= rotation.min() < -179.8
    assert 179.8 < rotation.max() <= 180.0
    assert 0.5 <= scale.min() < 0.501
    assert 1.499 < scale.max() <= 1.5
    assert abs(float(rotation.mean())) < 0.2
    assert abs(float(scale.mean()) - 1.0) < 0.001
    assert not np.array_equal(
        np.argsort(rotation),
        np.argsort(scale),
    )


def test_train_and_validation_joint_manifests_do_not_share_ap_centers(generator):
    train = make_joint_manifest(generator, 64, "train", 8104, "mild", 6)
    validation = make_joint_manifest(generator, 64, "validation", 8104, "mild", 6)

    def centers(manifest):
        true = manifest["true_pose"]
        poses = np.concatenate(
            (true, true + manifest["initial_pose_offset"], (true[:, None] + manifest["wrong_candidate_offset"]).reshape(-1, 3)),
            axis=0,
        )
        return set(np.rint(BREGMA_AP_INDEX - poses[:, 0] / VOXEL_UM).astype(int))

    assert not centers(train) & centers(validation)


def test_joint_manifest_npz_roundtrip_preserves_types_values_and_hash(generator, tmp_path):
    manifest = make_joint_manifest(generator, 4, "validation", 8105, "clean", 3)
    path = save_joint_manifest(manifest, tmp_path / "joint-manifest.npz")
    loaded = load_joint_manifest(path)
    _assert_manifest_equal(manifest, loaded)
    assert loaded["joint_manifest_sha256"] == manifest["joint_manifest_sha256"]


def test_joint_batch_preserves_exact_true_pair_and_has_no_wrong_plane_flow(generator):
    data = JointSyntheticData(generator)
    manifest = data.make_manifest(1, "validation", 8106, "hard", 3)
    base = registration_manifest(manifest)
    expected = generator.batch(base, qa=True)
    batch = data.batch(manifest, qa=True)
    repeated = data.batch(manifest, qa=True)

    for key in (
        "fixed", "moving", "fixed_mask", "moving_tissue_mask", "fixed_labels",
        "moving_labels", "fixed_to_moving", "moving_to_fixed", "local_velocity",
    ):
        assert torch.equal(batch[key], expected[key]), key
    assert batch["pose_image"].shape == (1, 3, 299, 299)
    assert torch.equal(batch["pose_image"], repeated["pose_image"])
    assert batch["true_pose"].shape == batch["initial_pose"].shape == (1, 3)
    assert batch["wrong_candidate_pose"].shape == (1, 3, 3)
    assert batch["wrong_candidate_fixed"].shape == (1, 3, 1, 320, 464)
    assert batch["wrong_candidate_fixed_mask"].dtype == torch.bool
    assert batch["wrong_candidate_fixed_labels"].dtype == torch.int64
    assert batch["true_dense_target_valid"].all()
    assert not batch["wrong_candidate_dense_target_valid"].any()
    assert not any(
        key.startswith("wrong_candidate_") and ("flow" in key or "velocity" in key)
        for key in batch
    )
    assert batch["artifact_stratum"] == "hard"
    assert batch["artifact_stratum_index"].tolist() == [2]
    assert batch["orientation_inverted_target"].item() == (
        abs(float(manifest["pose_view_total_rotation_deg"][0])) > 90.0
    )
    assert batch["pose_view_rotation_deg"].item() == pytest.approx(
        float(manifest["pose_view_rotation_deg"][0])
    )
    assert batch["pose_view_scale"].item() == pytest.approx(
        float(manifest["pose_view_scale"][0])
    )
    assert batch["pose_view_total_rotation_deg"].item() == pytest.approx(
        float(manifest["pose_view_total_rotation_deg"][0])
    )


def test_orientation_target_uses_wrapped_total_rotation_at_both_thresholds(generator):
    from training.synthetic_registration import _payload_sha256

    data = JointSyntheticData(generator)
    manifest = data.make_manifest(4, "validation", 8113, "clean", 1)
    base = registration_manifest(manifest)
    desired_total = np.asarray((89.9, 90.1, -89.9, -90.1), dtype=np.float32)
    raw = (desired_total - base["rotation_deg"] + 180.0) % 360.0 - 180.0
    manifest["pose_view_rotation_deg"] = raw.astype(np.float32)
    manifest["pose_view_total_rotation_deg"] = (
        (base["rotation_deg"] + raw + 180.0) % 360.0 - 180.0
    ).astype(np.float32)
    manifest["joint_manifest_sha256"] = _payload_sha256(
        {key: value for key, value in manifest.items() if key != "joint_manifest_sha256"}
    )
    batch = data.batch(manifest)
    assert batch["orientation_inverted_target"].tolist() == [False, True, False, True]
    torch.testing.assert_close(
        batch["pose_view_total_rotation_deg"],
        torch.from_numpy(desired_total),
        rtol=0.0,
        atol=5e-5,
    )


def test_rehashed_manifest_cannot_detach_total_pose_rotation(generator):
    from training.synthetic_registration import _payload_sha256

    data = JointSyntheticData(generator)
    manifest = data.make_manifest(1, "validation", 8114, "clean", 1)
    manifest["pose_view_total_rotation_deg"] = np.asarray([0.0], dtype=np.float32)
    manifest["joint_manifest_sha256"] = _payload_sha256(
        {key: value for key, value in manifest.items() if key != "joint_manifest_sha256"}
    )
    with pytest.raises(ValueError, match="total pose-view rotations"):
        data.batch(manifest)


def test_joint_batch_wrong_candidate_atlas_is_the_exact_ccf_renderer(generator):
    data = JointSyntheticData(generator)
    manifest = data.make_manifest(1, "validation", 8107, "clean", 2)
    batch = data.batch(manifest)
    pose = batch["wrong_candidate_pose"].reshape(-1, 3)
    direct = generator.render_planes(
        BREGMA_AP_INDEX - pose[:, 0] / VOXEL_UM,
        pose[:, 1],
        pose[:, 2],
    )
    assert torch.equal(batch["wrong_candidate_fixed"].reshape_as(direct[0]), direct[0])
    assert torch.equal(batch["wrong_candidate_fixed_mask"].reshape_as(direct[1]), direct[1])
    assert torch.equal(batch["wrong_candidate_fixed_labels"].reshape_as(direct[2]), direct[2])


def test_public_pose_renderer_keeps_the_ccf_image_path_differentiable(generator):
    data = JointSyntheticData(generator)
    pose = torch.tensor([[-1450.0, 2.0, -1.5]], requires_grad=True)
    fixed, mask, labels = data.render_pose(pose)
    fixed.square().mean().backward()
    assert fixed.shape == mask.shape == labels.shape == (1, 1, 320, 464)
    assert pose.grad is not None
    assert torch.isfinite(pose.grad).all()
    assert torch.count_nonzero(pose.grad) == 3


def test_tampered_joint_manifest_is_rejected_before_materialization(generator):
    data = JointSyntheticData(generator)
    manifest = data.make_manifest(1, "validation", 8108, "clean", 2)
    manifest["wrong_candidate_offset"] = manifest["wrong_candidate_offset"].copy()
    manifest["wrong_candidate_offset"][0, 0, 0] += 25.0
    with pytest.raises(ValueError, match="joint manifest hash"):
        data.batch(manifest)


def test_rehashed_manifest_cannot_detach_true_pose_or_enable_wrong_dense_targets(generator):
    from training.synthetic_registration import _payload_sha256

    data = JointSyntheticData(generator)
    pose_tamper = data.make_manifest(1, "validation", 8110, "clean", 2)
    pose_tamper["true_pose"] = pose_tamper["true_pose"].copy()
    pose_tamper["true_pose"][0, 0] += 25.0
    pose_tamper["joint_manifest_sha256"] = _payload_sha256(
        {key: value for key, value in pose_tamper.items() if key != "joint_manifest_sha256"}
    )
    with pytest.raises(ValueError, match="true poses"):
        data.batch(pose_tamper)

    supervision_tamper = data.make_manifest(1, "validation", 8111, "clean", 2)
    supervision_tamper["wrong_candidate_dense_target_valid"] = np.ones((1, 2), dtype=np.bool_)
    supervision_tamper["joint_manifest_sha256"] = _payload_sha256(
        {
            key: value
            for key, value in supervision_tamper.items()
            if key != "joint_manifest_sha256"
        }
    )
    with pytest.raises(ValueError, match="cannot carry dense-flow"):
        data.batch(supervision_tamper)


def test_joint_public_api_cannot_open_the_sealed_split(generator):
    with pytest.raises(PermissionError, match="one-shot evaluator"):
        make_joint_manifest(generator, 1, "sealed-test", 8109, "clean", 2)
