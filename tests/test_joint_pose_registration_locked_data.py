from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import cv2
import nrrd
import numpy as np
import pytest
import torch

from training import joint_pose_registration_locked_data as locked


@pytest.fixture(scope="module")
def atlas_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("independent-ccf")
    ap, dv, ml = 420, 18, 22
    ap_axis = np.arange(ap, dtype=np.float32)[:, None, None]
    dv_axis = np.arange(dv, dtype=np.float32)[None, :, None]
    ml_axis = np.arange(ml, dtype=np.float32)[None, None, :]
    average = ap_axis * 0.2 + dv_axis * 2.0 + ml_axis
    labels = np.broadcast_to(
        (np.arange(ap, dtype=np.int64) + 1)[:, None, None], (ap, dv, ml)
    ).copy()
    nrrd.write(str(folder / "average_template_25.nrrd"), average.astype(np.float32))
    nrrd.write(str(folder / "annotation_25.nrrd"), labels.astype(np.int64))
    return folder


@pytest.fixture()
def benchmark(atlas_folder):
    return locked.LockedJointSyntheticBenchmark(atlas_folder, "cpu")


def _rehash(manifest):
    manifest["case_sha256"] = locked._case_hashes(manifest)
    manifest["manifest_sha256"] = locked._payload_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def test_module_does_not_import_any_training_generator():
    path = Path(locked.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "training.synthetic_registration" not in imported
    assert "training.synthetic_atlas" not in imported
    assert "training.joint_pose_registration_data" not in imported


def test_renderer_uses_bregma_216_and_anterior_positive(benchmark):
    _, mask, labels = benchmark.render_planes(
        torch.tensor((locked.BREGMA_AP_INDEX, locked.BREGMA_AP_INDEX - 1.0)),
        torch.zeros(2), torch.zeros(2),
    )
    center_y = benchmark.pad_y + benchmark.volume_shape[1] // 2
    center_x = benchmark.pad_x + benchmark.volume_shape[2] // 2
    assert mask[:, 0, center_y, center_x].all()
    assert labels[0, 0, center_y, center_x] == 217  # AP 0 um -> volume index 216.
    assert labels[1, 0, center_y, center_x] == 216  # AP +25 um -> index 215.


def test_oblique_renderer_matches_analytic_plane_equation(benchmark):
    _, _, labels = benchmark.render_planes(
        torch.tensor((216.0,)), torch.tensor((45.0,)), torch.tensor((0.0,))
    )
    y = benchmark.pad_y + benchmark.volume_shape[1] // 2
    native_center = (benchmark.volume_shape[2] - 1) / 2.0
    for native_x in (2, 8, 15, 20):
        expected_ap = round(216.0 + native_x - native_center)
        assert labels[0, 0, y, benchmark.pad_x + native_x] == expected_ap + 1


def test_public_ap_blocks_are_disjoint_and_guarded():
    development = locked.split_ap_indices("development")
    validation = locked.split_ap_indices("locked-validation")
    local_test = locked._split_indices("sealed-test", allow_sealed=True)
    pools = (development, validation, local_test)
    for first_index, first in enumerate(pools):
        for second in pools[first_index + 1 :]:
            assert not np.intersect1d(first, second).size
            assert (
                np.min(np.abs(first[:, None] - second[None, :]))
                >= locked.AP_GUARD_MIN_INDEX_DISTANCE
            )
    for pool in pools:
        bands = locked._ap_band_indices(pool)
        assert np.array_equal(np.unique(bands), np.arange(locked.AP_BAND_COUNT))
    with pytest.raises(PermissionError, match="local evaluator"):
        locked.split_ap_indices("sealed-test")


def test_locked_test_centers_exclude_frozen_dense_v2_train_and_validation(
    benchmark,
):
    first = int(
        round(locked.BREGMA_AP_INDEX - locked.AP_RANGE_UM[1] / locked.VOXEL_UM)
    )
    last = int(
        round(locked.BREGMA_AP_INDEX - locked.AP_RANGE_UM[0] / locked.VOXEL_UM)
    )
    indices = np.arange(first, last + 1, dtype=np.int32)
    dense_blocks = (indices - first) // 4
    dense_pattern = np.asarray(
        (
            "train", "train", "guard", "validation", "guard",
            "train", "train", "guard", "sealed-test", "guard",
        ),
        dtype=object,
    )
    dense_roles = dense_pattern[dense_blocks % len(dense_pattern)]
    dense_development = indices[np.isin(dense_roles, ("train", "validation"))]
    local_test = locked._split_indices("sealed-test", allow_sealed=True)
    assert not np.intersect1d(local_test, dense_development).size
    assert np.array_equal(
        np.unique(locked._ap_band_indices(local_test)),
        np.arange(locked.AP_BAND_COUNT),
    )
    expected_hash = locked._payload_sha256(
        {
            "contract": {
                "ap_block_width": 4,
                "pattern": tuple(dense_pattern.tolist()),
                "excluded_roles": ("train", "validation"),
                "scope": (
                    "leakage audit for dense-v2/joint generator centers only; "
                    "AtlasPose V7 used the full AP domain, so this benchmark does "
                    "not claim unseen AP anatomy"
                ),
            },
            "excluded_centers": dense_development,
        }
    )
    assert (
        benchmark.contract["dense_v2_train_validation_exclusion_sha256"]
        == expected_hash
    )
    receipt = benchmark.contract["ap_center_exclusion_receipt"]
    assert tuple(receipt["dense_v2_train_validation_centers"]) == tuple(
        dense_development
    )
    assert tuple(receipt["locked_test_centers"]) == tuple(local_test)
    assert receipt["overlap_centers"] == ()
    assert "does not claim unseen AP anatomy" in receipt["scope"]


def test_manifest_hash_is_deterministic_and_seed_sensitive(benchmark):
    first = benchmark.make_manifest(2, "locked-validation", 8102, "severe", 4)
    second = benchmark.make_manifest(2, "locked-validation", 8102, "severe", 4)
    changed = benchmark.make_manifest(2, "locked-validation", 8103, "severe", 4)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["manifest_sha256"] != changed["manifest_sha256"]
    assert np.array_equal(first["case_sha256"], second["case_sha256"])
    assert not np.array_equal(first["case_sha256"], changed["case_sha256"])
    assert hashlib.sha256(first["pose"].tobytes()).hexdigest() == hashlib.sha256(
        second["pose"].tobytes()
    ).hexdigest()


def test_manifest_balances_every_declared_500um_ap_band(benchmark):
    manifest = benchmark.make_manifest(103, "locked-validation", 8172, "moderate", 4)
    counts = np.asarray(manifest["ap_band_counts"])
    assert counts.sum() == 103
    assert counts.max() - counts.min() == 1
    assert np.array_equal(
        counts,
        np.bincount(manifest["ap_band_index"], minlength=locked.AP_BAND_COUNT),
    )
    assert np.all(counts >= 10)
    rendered = locked.BREGMA_AP_INDEX - manifest["pose"][:, 0] / locked.VOXEL_UM
    assert rendered.min() >= locked.BREGMA_AP_INDEX - locked.AP_RANGE_UM[1] / locked.VOXEL_UM
    assert rendered.max() <= locked.BREGMA_AP_INDEX - locked.AP_RANGE_UM[0] / locked.VOXEL_UM


def test_severity_names_and_ranges_match_the_frozen_four_strata(benchmark):
    assert tuple(locked.SEVERITIES) == ("clean", "mild", "moderate", "severe")
    for name in (
        "control", "radial", "anisotropic", "shear", "refiner_rotation",
        "pose_view_rotation", "translation", "gamma", "noise", "damage",
    ):
        values = [locked.SEVERITIES[level][name] for level in locked.SEVERITIES]
        assert values == sorted(values)
    for field in ("refiner_scale", "pose_view_scale"):
        scale_ranges = [locked.SEVERITIES[level][field] for level in locked.SEVERITIES]
        assert [value[0] for value in scale_ranges] == sorted(
            (value[0] for value in scale_ranges), reverse=True
        )
        assert [value[1] for value in scale_ranges] == sorted(
            value[1] for value in scale_ranges
        )
    assert locked.SEVERITIES["severe"]["refiner_rotation"] == 15.0
    assert locked.SEVERITIES["severe"]["refiner_scale"] == (0.86, 1.14)
    assert locked.SEVERITIES["severe"]["pose_view_rotation"] == 180.0
    assert locked.SEVERITIES["severe"]["pose_view_scale"] == (0.50, 1.50)
    with pytest.raises(ValueError, match="severity"):
        benchmark.make_manifest(1, "development", 7, "hard")


def test_pose_view_nuisance_is_separate_from_bounded_refiner_residual(benchmark):
    manifest = benchmark.make_manifest(512, "locked-validation", 33107, "severe", 4)
    assert np.abs(manifest["refiner_rotation_deg"]).max() <= 15.0
    assert manifest["refiner_scale"].min() >= 0.86
    assert manifest["refiner_scale"].max() <= 1.14
    assert manifest["pose_view_rotation_deg"].min() < -150.0
    assert manifest["pose_view_rotation_deg"].max() > 150.0
    assert manifest["pose_view_scale"].min() < 0.60
    assert manifest["pose_view_scale"].max() > 1.40

    pair = benchmark.batch(
        benchmark.make_manifest(1, "locked-validation", 33108, "severe", 2)
    )
    assert pair["pose_view"].shape == pair["moving"].shape
    assert pair["pose_view_mask"].shape == pair["moving_visible_mask"].shape
    assert not pair["pose_view_dense_target_valid"].any()
    assert pair["refiner_to_pose_view_map"].shape == pair["fixed_to_moving"].shape
    assert pair["pose_view_to_refiner_map"].shape == pair["moving_to_fixed"].shape


def test_hard_negatives_are_distinct_in_domain_and_have_no_dense_target(benchmark):
    pair = benchmark.generate(1, "locked-validation", 7204, "clean", 8)
    pose = pair["pose"][:, None]
    assert (pair["negative_pose"] != pose).any(dim=2).all()
    assert not pair["negative_dense_target_valid"].any()
    assert pair["negative_pose"][:, :, 0].min() >= locked.AP_RANGE_UM[0]
    assert pair["negative_pose"][:, :, 0].max() <= locked.AP_RANGE_UM[1]
    assert pair["negative_pose"][:, :, 1:].abs().max() <= 35.0
    adjacent = (
        (pair["negative_pose_offset"][:, :, 0].abs() == locked.VOXEL_UM)
        & (pair["negative_pose_offset"][:, :, 1:] == 0.0).all(dim=2)
    )
    assert adjacent.any(dim=1).all()
    adjacent_lr = (
        (pair["negative_pose_offset"][:, :, 0] == 0.0)
        & (pair["negative_pose_offset"][:, :, 1].abs() == locked.NEGATIVE_TILT_DEG[0])
        & (pair["negative_pose_offset"][:, :, 2] == 0.0)
    )
    adjacent_dv = (
        (pair["negative_pose_offset"][:, :, :2] == 0.0).all(dim=2)
        & (pair["negative_pose_offset"][:, :, 2].abs() == locked.NEGATIVE_TILT_DEG[0])
    )
    assert adjacent_lr.any(dim=1).all()
    assert adjacent_dv.any(dim=1).all()
    negative_centers = torch.round(
        locked.BREGMA_AP_INDEX - pair["negative_pose"][:, :, 0] / locked.VOXEL_UM
    ).numpy().astype(np.int32)
    assert np.isin(
        negative_centers, locked.split_ap_indices("locked-validation")
    ).all()


def test_forward_inverse_cycle_and_positive_jacobians(benchmark):
    pair = benchmark.generate(1, "locked-validation", 8713, "severe", 3)
    forward_jacobian = locked.jacobian_determinant(pair["fixed_to_moving"])
    inverse_jacobian = locked.jacobian_determinant(pair["moving_to_fixed"])
    cells = (
        pair["fixed_mask"][:, :, :-1, :-1] | pair["fixed_mask"][:, :, :-1, 1:]
        | pair["fixed_mask"][:, :, 1:, :-1] | pair["fixed_mask"][:, :, 1:, 1:]
    )[:, 0]
    assert forward_jacobian[cells].min() > 0.0
    moving_cells = (
        pair["moving_tissue_mask"][:, :, :-1, :-1]
        | pair["moving_tissue_mask"][:, :, :-1, 1:]
        | pair["moving_tissue_mask"][:, :, 1:, :-1]
        | pair["moving_tissue_mask"][:, :, 1:, 1:]
    )[:, 0]
    assert inverse_jacobian[moving_cells].min() > 0.0
    cycle = locked.compose_pixel_maps(pair["fixed_to_moving"], pair["moving_to_fixed"])
    identity = locked._identity_grid(1, *locked.MODEL_SHAPE, torch.device("cpu"))
    error = torch.linalg.vector_norm(cycle - identity, dim=1)[pair["moving_tissue_mask"][:, 0]]
    assert torch.quantile(error, 0.95) < 0.75


def test_radial_anisotropic_and_shear_velocity_families_are_independent():
    zero_radial = torch.tensor(((0.0, 0.0, 0.35, 0.0),))
    zero_anisotropic = torch.tensor(((0.0, 0.0, 0.35, 0.4, 0.0, 0.0),))
    zero_shear = torch.tensor(((0.0, 0.0, 0.35, -0.3, 0.0),))
    radial = zero_radial.clone()
    radial[:, 3] = 4.0
    anisotropic = zero_anisotropic.clone()
    anisotropic[:, 4:] = torch.tensor((4.0, -2.0))
    shear = zero_shear.clone()
    shear[:, 4] = 4.0
    shape = (41, 53)

    radial_field = locked._independent_local_velocity(
        radial, zero_anisotropic, zero_shear, shape
    )
    anisotropic_field = locked._independent_local_velocity(
        zero_radial, anisotropic, zero_shear, shape
    )
    shear_field = locked._independent_local_velocity(
        zero_radial, zero_anisotropic, shear, shape
    )
    for field in (radial_field, anisotropic_field, shear_field):
        assert field.shape == (1, 2, *shape)
        assert torch.isfinite(field).all()
        assert torch.count_nonzero(field) > 0
    assert not torch.allclose(radial_field, anisotropic_field)
    assert not torch.allclose(radial_field, shear_field)
    assert not torch.allclose(anisotropic_field, shear_field)


def test_missing_tissue_is_excluded_from_correspondence(benchmark):
    manifest = benchmark.make_manifest(1, "locked-validation", 9917, "mild", 2)
    manifest["missing_enabled"][:] = True
    manifest["tear_enabled"][:] = True
    manifest["occlusion_enabled"][:] = True
    manifest["occlusion"][:] = np.asarray((0.0, 0.0, 0.5, 0.5, 0.0), np.float32)
    _rehash(manifest)
    pair = benchmark.batch(manifest)
    assert pair["moving_damage_mask"].any()
    assert not torch.any(pair["moving_visible_mask"] & pair["moving_damage_mask"])
    expected = pair["fixed_mask"] & (
        locked.sample_at(
            pair["moving_visible_mask"].float(), pair["fixed_to_moving"], "nearest"
        ) > 0.5
    )
    assert torch.equal(pair["fixed_visible_mask"], expected)
    assert torch.any(pair["fixed_mask"] & ~pair["fixed_visible_mask"])


def test_clean_appearance_is_exactly_clean_and_repeatable(benchmark):
    manifest = benchmark.make_manifest(1, "locked-validation", 13117, "clean", 2)
    first = benchmark.batch(manifest, qa=True)
    second = benchmark.batch(manifest, qa=True)
    torch.testing.assert_close(
        first["moving"][first["moving_tissue_mask"]],
        first["moving_clean"][first["moving_tissue_mask"]], rtol=0.0, atol=0.0,
    )
    assert not first["moving"][~first["moving_tissue_mask"]].any()
    assert torch.equal(first["moving"], second["moving"])
    assert not first["moving_damage_mask"].any()
    assert not first["moving_optical_artifact_mask"].any()


def test_vignettes_seams_specks_scratches_and_bubbles_are_visible_but_valid(benchmark):
    manifest = benchmark.make_manifest(1, "locked-validation", 14117, "clean", 2)
    manifest["tile_strength"][:] = 0.22
    _rehash(manifest)
    tiled = benchmark.batch(manifest, qa=True)
    tile_difference = (tiled["moving"] - tiled["moving_clean"]).abs()[
        tiled["moving_tissue_mask"]
    ]
    assert float(tile_difference.mean()) > 0.001
    assert not tiled["moving_optical_artifact_mask"].any()

    manifest["speck_density"][:] = 0.0008
    manifest["scratch_enabled"][:] = True
    manifest["scratch"][:] = np.asarray((0.4, 0.0, 0.006, 0.75), np.float32)
    manifest["bubble_enabled"][:] = True
    manifest["bubble"][:] = np.asarray((0.0, 0.0, 0.20, 0.010, 0.55), np.float32)
    _rehash(manifest)
    pair = benchmark.batch(manifest, qa=True)
    repeated = benchmark.batch(manifest, qa=True)
    assert torch.equal(pair["moving"], repeated["moving"])
    assert pair["moving_optical_artifact_mask"].any()
    assert pair["moving_speck_mask"].any()
    assert pair["moving_scratch_mask"].any()
    assert pair["moving_bubble_mask"].any()
    assert not pair["moving_damage_mask"].any()
    difference = (pair["moving"] - pair["moving_clean"]).abs()[pair["moving_tissue_mask"]]
    assert float(difference.mean()) > 0.005
    assert float(torch.quantile(difference, 0.99)) > 0.10


def test_polygon_edge_loss_and_edge_to_edge_blackout_are_invalid(benchmark):
    manifest = benchmark.make_manifest(1, "locked-validation", 15117, "clean", 2)
    manifest["polygon_enabled"][:] = True
    manifest["polygon_xy"][:] = np.asarray(
        ((-0.35, -0.25), (0.25, -0.30), (0.40, 0.05),
         (0.10, 0.35), (-0.40, 0.20), (-0.50, -0.05)),
        np.float32,
    )
    manifest["edge_loss_enabled"][:] = True
    manifest["edge_loss_side"][:] = 0
    manifest["edge_loss"][:] = np.asarray((0.0, 0.8, 1.2, 0.0), np.float32)
    manifest["blackout_enabled"][:] = True
    manifest["blackout"][:] = np.asarray((0.0, 0.0, 0.06), np.float32)
    _rehash(manifest)
    pair = benchmark.batch(manifest)
    assert pair["moving_polygon_mask"].any()
    assert pair["moving_edge_loss_mask"].any()
    assert pair["moving_blackout_mask"].any()
    assert torch.all(pair["moving_edge_loss_mask"] <= pair["moving_damage_mask"])
    assert torch.all(pair["moving_blackout_mask"] <= pair["moving_damage_mask"])
    assert torch.all(pair["moving_polygon_mask"] <= pair["moving_damage_mask"])
    assert not torch.any(pair["moving_visible_mask"] & pair["moving_damage_mask"])
    assert torch.all(pair["moving"][pair["moving_blackout_mask"]] == 0.0)


def test_artifact_parameter_tampering_breaks_exact_manifest_hash(benchmark):
    manifest = benchmark.make_manifest(1, "locked-validation", 16117, "mild", 2)
    manifest["speck_density"][0] += np.float32(0.0001)
    with pytest.raises(ValueError, match="manifest hash"):
        benchmark.batch(manifest)


def test_sealed_generation_requires_and_consumes_private_capability(atlas_folder):
    benchmark = locked.LockedJointSyntheticBenchmark(atlas_folder, "cpu")
    with pytest.raises(PermissionError, match="local qualification"):
        benchmark.make_manifest(1, "sealed-test", 101, "clean")
    with pytest.raises(PermissionError, match="local qualification"):
        benchmark.generate(1, "sealed-test", 101, "clean")
    with pytest.raises(PermissionError, match="capability"):
        benchmark.generate_sealed_once(1, 101, "clean", _capability=object())
    pair = benchmark.generate_sealed_once(
        1, 101, "clean", _capability=locked._SEALED_EVALUATOR_CAPABILITY
    )
    assert pair["pose"].shape == (1, 3)
    with pytest.raises(PermissionError, match="already been consumed"):
        benchmark.generate_sealed_once(
            1, 102, "clean", _capability=locked._SEALED_EVALUATOR_CAPABILITY
        )


def test_balanced_locked_manifests_have_within_case_reference_challenge_pairs(
    atlas_folder,
):
    benchmark = locked.LockedJointSyntheticBenchmark(atlas_folder, "cpu")
    manifests = benchmark.make_balanced_sealed_manifests_once(
        19017,
        20,
        3,
        _capability=locked._SEALED_EVALUATOR_CAPABILITY,
    )
    all_pair_ids = []
    for severity, manifest in manifests.items():
        assert manifest["severity"] == severity
        assert tuple(manifest["ap_band_counts"]) == (2,) * locked.AP_BAND_COUNT
        assert len(np.unique(manifest["pair_id"])) == 20
        all_pair_ids.extend(manifest["pair_id"].tolist())
        batch = benchmark._batch(manifest, qa=False)
        assert np.array_equal(batch["pair_id"], manifest["pair_id"])
        assert batch["reference_moving_raw_uint8"].shape == batch[
            "moving_raw_uint8"
        ].shape
        assert batch["reference_pose_view_raw_uint8"].shape == batch[
            "pose_view_raw_uint8"
        ].shape
        assert torch.equal(
            batch["reference_moving_model_mask"], batch["moving_tissue_mask"]
        )
        assert torch.equal(
            batch["reference_pose_view_mask"],
            locked.sample_at(
                batch["moving_tissue_mask"].float(),
                batch["pose_view_to_refiner_map"],
                "nearest",
            ) > 0.5,
        )
        limit = locked.SEVERITIES[severity]
        assert np.abs(manifest["refiner_rotation_deg"]).max() <= limit[
            "refiner_rotation"
        ]
        assert np.abs(manifest["pose_view_rotation_deg"]).max() <= limit[
            "pose_view_rotation"
        ]
    assert len(set(all_pair_ids)) == 4 * 20

    severe = benchmark._batch(manifests["severe"], qa=False)
    assert not torch.equal(
        severe["reference_moving_raw_uint8"], severe["moving_raw_uint8"]
    )


def test_exact_predictions_score_without_training_evaluator(benchmark):
    manifest = benchmark.make_manifest(1, "locked-validation", 11209, "mild", 2)
    manifest["pose"][:, 1:] = 0.0
    _rehash(manifest)
    pair = benchmark.batch(manifest)
    metrics = benchmark.evaluate_predictions(
        pair,
        {
            "pose": pair["pose"],
            "map_pose": pair["pose"],
            "map_space": "source-model-canvas",
            "source_shape": locked.MODEL_SHAPE,
            "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
            "fixed_to_source_model": pair["fixed_to_moving"],
            "source_model_to_fixed": pair["moving_to_fixed"],
        },
    )
    assert metrics["ap_mae_um"] == metrics["lr_mae_deg"] == metrics["dv_mae_deg"] == 0.0
    assert metrics["warp_only_endpoint_mean_px"] == metrics["warp_only_endpoint_p95_px"] == 0.0
    assert metrics["warp_only_negative_jacobian_fraction"] == 0.0
    assert metrics["warp_only_visible_region_correspondence"] > 0.99
    assert metrics["warp_only_macro_region_dice"] > 0.99
    assert metrics["end_to_end_visible_region_correspondence"] == metrics[
        "true_pose_source_model_correspondence_ceiling"
    ]
    assert metrics["end_to_end_interior_region_correspondence"] >= metrics[
        "end_to_end_visible_region_correspondence"
    ]


def test_wrong_pose_maps_are_scored_end_to_end_without_cross_plane_epe(benchmark):
    pair = benchmark.generate(1, "locked-validation", 11209, "clean", 2)
    correct = benchmark.evaluate_predictions(
        pair,
        {
            "pose": pair["pose"],
            "map_pose": pair["pose"],
            "map_space": "source-model-canvas",
            "source_shape": locked.MODEL_SHAPE,
            "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
            "fixed_to_source_model": pair["fixed_to_moving"],
            "source_model_to_fixed": pair["moving_to_fixed"],
        },
    )
    wrong_pose = pair["pose"].clone()
    wrong_pose[:, 0] += 25.0
    identity = locked._identity_grid(1, *locked.MODEL_SHAPE, torch.device("cpu"))
    wrong = benchmark.evaluate_predictions(
        pair,
        {
            "pose": wrong_pose,
            "map_pose": wrong_pose,
            "map_space": "source-model-canvas",
            "source_shape": locked.MODEL_SHAPE,
            "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
            "fixed_to_source_model": identity,
            "source_model_to_fixed": identity,
        },
    )
    assert wrong["ap_mae_um"] == 25.0
    assert wrong["end_to_end_visible_region_correspondence"] < correct[
        "end_to_end_visible_region_correspondence"
    ]
    assert not any(key.startswith("warp_only_endpoint") for key in wrong)


def test_final_maps_cannot_be_declared_at_a_different_pose(benchmark):
    pair = benchmark.generate(1, "locked-validation", 18209, "clean", 2)
    with pytest.raises(ValueError, match="same predicted pose"):
        benchmark.evaluate_predictions(
            pair,
            {
                "pose": pair["pose"],
                "map_pose": pair["pose"] + torch.tensor((25.0, 0.0, 0.0)),
                "map_space": "source-model-canvas",
                "source_shape": locked.MODEL_SHAPE,
                "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
                "fixed_to_source_model": pair["fixed_to_moving"],
                "source_model_to_fixed": pair["moving_to_fixed"],
            },
        )


def test_final_map_pose_and_raw_space_are_mandatory(benchmark):
    pair = benchmark.generate(1, "locked-validation", 19209, "clean", 2)
    base = {
        "pose": pair["pose"],
        "fixed_to_source_model": pair["fixed_to_moving"],
        "source_model_to_fixed": pair["moving_to_fixed"],
        "source_shape": locked.MODEL_SHAPE,
        "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
    }
    with pytest.raises(ValueError, match="map_pose is required"):
        benchmark.evaluate_predictions(pair, {**base, "map_space": "source-model-canvas"})
    with pytest.raises(ValueError, match="source model"):
        benchmark.evaluate_predictions(pair, {**base, "map_pose": pair["pose"]})
    without_receipt = {key: value for key, value in base.items() if key != "refiner_preprocessing"}
    with pytest.raises(ValueError, match="preprocessing receipt"):
        benchmark.evaluate_predictions(
            pair,
            {**without_receipt, "map_pose": pair["pose"], "map_space": "source-model-canvas"},
        )


@pytest.mark.parametrize(
    "partial",
    (
        {"exact_plane_source_model_to_fixed": "moving_to_fixed"},
        {"exact_plane_pose": "pose"},
    ),
)
def test_partial_exact_plane_receipts_are_rejected(benchmark, partial):
    pair = benchmark.generate(1, "locked-validation", 19709, "clean", 2)
    prediction = {
        "pose": pair["pose"],
        "map_pose": pair["pose"],
        "map_space": "source-model-canvas",
        "source_shape": locked.MODEL_SHAPE,
        "refiner_preprocessing": locked.REFINER_PREPROCESSING_CONTRACT,
        "fixed_to_source_model": pair["fixed_to_moving"],
        "source_model_to_fixed": pair["moving_to_fixed"],
    }
    prediction.update({name: pair[key] for name, key in partial.items()})
    with pytest.raises(ValueError, match="requires maps and their exact-plane pose"):
        benchmark.evaluate_predictions(pair, prediction)


def test_independent_candidate_mask_affine_matches_runtime_contract():
    from source.atlas_pose_runtime import brain_mask_affine

    target = np.zeros(locked.MODEL_SHAPE, np.uint8)
    cv2.ellipse(target, (232, 170), (150, 92), 0, 0, 360, 1, -1)
    cv2.rectangle(target, (180, 55), (214, 105), 1, -1)
    raw_transform = cv2.getRotationMatrix2D((231.5, 159.5), 37.0, 0.63)
    raw_transform[:, 2] += (11.0, -7.0)
    source = cv2.warpAffine(
        target, raw_transform, (locked.MODEL_SHAPE[1], locked.MODEL_SHAPE[0]),
        flags=cv2.INTER_NEAREST,
    )
    independent = locked.candidate_pose_mask_affine(source, target)
    reference = brain_mask_affine(source, target)
    np.testing.assert_allclose(independent, reference, rtol=0.0, atol=1e-10)
    mapped = cv2.warpAffine(
        source, independent[:2], (locked.MODEL_SHAPE[1], locked.MODEL_SHAPE[0]),
        flags=cv2.INTER_NEAREST,
    )
    intersection = np.count_nonzero((mapped > 0) & (target > 0))
    union = np.count_nonzero((mapped > 0) | (target > 0))
    assert intersection / union > 0.97


def test_candidate_preprocessing_removes_raw_roll_and_scale_before_refiner(benchmark):
    manifest = benchmark.make_manifest(1, "locked-validation", 20209, "severe", 2)
    manifest["control_velocity_xy"][:] = 0.0
    manifest["radial_velocity"][:, 3] = 0.0
    manifest["anisotropic_velocity"][:, 4:] = 0.0
    manifest["shear_velocity"][:, 4] = 0.0
    manifest["refiner_rotation_deg"][:] = 15.0
    manifest["refiner_scale"][:] = 0.86
    manifest["translation_xy"][:] = (12.0, -8.0)
    for name in (
        "tear_enabled", "missing_enabled", "occlusion_enabled",
        "polygon_enabled", "edge_loss_enabled", "blackout_enabled",
        "scratch_enabled", "bubble_enabled",
    ):
        manifest[name][:] = False
    manifest["speck_density"][:] = 0.0
    _rehash(manifest)
    pair = benchmark.batch(manifest)
    prepared = benchmark.prepare_refiner_inputs(pair, pair["pose"])
    mapped, target = prepared["aligned_moving_mask"], prepared["fixed_mask"]
    intersection = (mapped & target).sum()
    union = (mapped | target).sum()
    assert float(intersection / union) > 0.87  # Tiny analytic mask is interpolation-limited.
    assert torch.allclose(prepared["map_pose"], pair["pose"])
    fixed_to_aligned = locked._apply_homography(
        pair["fixed_to_moving"], prepared["raw_to_aligned"]
    )
    aligned_grid = locked._identity_grid(1, *locked.MODEL_SHAPE, torch.device("cpu"))
    aligned_to_raw_grid = locked._apply_homography(
        aligned_grid, prepared["aligned_to_raw"]
    )
    aligned_to_fixed = locked.sample_at(pair["moving_to_fixed"], aligned_to_raw_grid)
    fixed_to_source, source_to_fixed = benchmark.compose_refiner_maps_to_source_model(
        fixed_to_aligned, aligned_to_fixed, prepared["raw_to_aligned"]
    )
    torch.testing.assert_close(
        fixed_to_source, pair["fixed_to_moving"], rtol=0.0, atol=1e-4
    )
    inverse_error = torch.linalg.vector_norm(
        source_to_fixed - pair["moving_to_fixed"], dim=1
    )[pair["moving_tissue_mask"][:, 0]]
    assert torch.quantile(inverse_error, 0.95) < 0.15


@pytest.mark.parametrize("negative_count", (1, 2, 3))
def test_minimal_negative_sets_reserve_adjacent_axes(benchmark, negative_count):
    manifest = benchmark.make_manifest(
        8, "locked-validation", 21209 + negative_count, "clean", negative_count
    )
    offsets = manifest["negative_pose_offset"]
    assert np.any(
        (np.abs(offsets[:, :, 0]) == locked.VOXEL_UM)
        & (offsets[:, :, 1:] == 0.0).all(axis=2), axis=1
    ).all()
    if negative_count >= 2:
        assert np.any(
            (offsets[:, :, 0] == 0.0)
            & (np.abs(offsets[:, :, 1]) == locked.NEGATIVE_TILT_DEG[0])
            & (offsets[:, :, 2] == 0.0), axis=1
        ).all()
    if negative_count >= 3:
        assert np.any(
            (offsets[:, :, :2] == 0.0).all(axis=2)
            & (np.abs(offsets[:, :, 2]) == locked.NEGATIVE_TILT_DEG[0]), axis=1
        ).all()


def test_refiner_preprocessing_uses_outline_raw_uint8_and_runtime_feather(benchmark):
    manifest = benchmark.make_manifest(1, "locked-validation", 22209, "clean", 3)
    manifest["blackout_enabled"][:] = True
    manifest["blackout"][:] = np.asarray((0.0, 0.0, 0.08), np.float32)
    _rehash(manifest)
    pair = benchmark.batch(manifest)
    prepared = benchmark.prepare_refiner_inputs(pair, pair["pose"])

    matrix = prepared["raw_to_aligned"][0].numpy()
    expected_raw = cv2.warpAffine(
        pair["moving_raw_uint8"][0, 0].numpy(), matrix[:2],
        (locked.MODEL_SHAPE[1], locked.MODEL_SHAPE[0]), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    expected_mask = cv2.warpAffine(
        pair["moving_model_mask"][0, 0].numpy().astype(np.uint8), matrix[:2],
        (locked.MODEL_SHAPE[1], locked.MODEL_SHAPE[0]), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    expected = expected_raw.astype(np.float32) / 255.0 * locked.numpy_cosine_mask_feather(
        expected_mask
    )
    np.testing.assert_array_equal(prepared["aligned_moving_raw_uint8"][0, 0].numpy(), expected_raw)
    np.testing.assert_array_equal(prepared["aligned_moving_mask"][0, 0].numpy(), expected_mask)
    np.testing.assert_allclose(prepared["aligned_moving"][0, 0].numpy(), expected, rtol=0.0, atol=0.0)
    assert prepared["dense_preprocessing_contract"] == locked.PREPROCESSING_CONTRACT_V2
    assert prepared["dense_mask_contract_sha256"] == locked.MASK_CONTRACT_SHA256
    assert torch.any(pair["moving_blackout_mask"] & pair["moving_model_mask"])
    assert not torch.any(pair["moving_blackout_mask"] & pair["moving_visible_mask"])


def test_outline_mask_excludes_missing_edges_but_keeps_internal_damage(benchmark):
    manifest = benchmark.make_manifest(1, "locked-validation", 23209, "clean", 3)
    manifest["missing_enabled"][:] = True
    manifest["missing"][:] = np.asarray((0.0, 0.5, 0.5), np.float32)
    manifest["edge_loss_enabled"][:] = True
    manifest["edge_loss_side"][:] = 0
    manifest["edge_loss"][:] = np.asarray((0.0, 0.8, 1.0, 0.0), np.float32)
    manifest["occlusion_enabled"][:] = True
    manifest["occlusion"][:] = np.asarray((0.0, 0.0, 0.35, 0.35, 0.0), np.float32)
    _rehash(manifest)
    pair = benchmark.batch(manifest)
    removed_edge = pair["moving_tissue_mask"] & ~pair["moving_model_mask"]
    assert removed_edge.any()
    assert torch.any(pair["moving_damage_mask"] & pair["moving_model_mask"])
    assert torch.all(pair["moving_model_mask"] <= pair["moving_tissue_mask"])
