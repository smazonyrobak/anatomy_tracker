import copy
from collections.abc import Mapping

import numpy as np
import pytest
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_pose_v2 as pose_v2
import training.arbitrary_plane_realization_v2 as realization_v2
from training.arbitrary_plane_acquisition_v2 import (
    prepare_arbitrary_plane_acquisition_context_v2,
)
from training.arbitrary_plane_geometry import (
    allen_index_to_physical_um_points,
    allen_index_to_physical_um_vectors,
    frame_to_physical_ouv,
    frame_to_rotation_6d,
    inplane_basis_to_parameters,
    physical_ouv_to_frame,
    positive_inplane_basis,
    quicknii_to_allen_points,
    quicknii_to_allen_vectors,
    rotation_6d_to_frame,
)
from training.arbitrary_plane_manifest import canonicalize_plane
from training.arbitrary_plane_support import build_annotation_support_index


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return copy.deepcopy(value)


@pytest.fixture(scope="module")
def prepared_context():
    annotation = np.zeros((9, 8, 7), dtype=np.uint16)
    annotation[1:8, 1:7, 1:6] = 3
    ap, dv, ml = np.indices(annotation.shape)
    scalar = (2 * ap + 3 * dv + 5 * ml).astype(np.float32)
    support = build_annotation_support_index(
        annotation,
        atlas_id="pose-v2-fixture",
        atlas_version="v1",
        source_uri="file:///fixture/pose-annotation.nrrd",
        source_sha256="7" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(11.0, 17.0, 29.0),
        origin_um=(-71.0, 23.0, 107.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return prepare_arbitrary_plane_acquisition_context_v2(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/pose-template.nrrd",
        scalar_source_sha256="8" * 64,
        template_decoder="pose-v2 fixture",
        annotation_decoder="pose-v2 fixture",
    )


def _source_pose(context):
    support_origin = np.asarray(
        context["receipt"]["global_reference_fov"][
            "support_origin_ap_dv_ml_um"
        ],
        dtype=np.float64,
    )
    frame = rotation_6d_to_frame(
        torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    )
    basis = positive_inplane_basis(
        torch.log(torch.tensor([620.0, 470.0], dtype=torch.float64)),
        torch.tensor(0.21, dtype=torch.float64),
    )
    center = torch.from_numpy(support_origin) + 85.0 * frame[:, 2] + 31.0 * frame[:, 0]
    return frame_to_physical_ouv(center, frame, basis).detach().numpy().reshape(3, 3)


def _final_realization(context, horizontal=False, vertical=False):
    parent_shape = (10, 12)
    top_left = (2, 3)
    output_shape = (6, 7)
    parent_ouv = _source_pose(context)
    cropped, model = realization_v2._crop_and_reflect_ouv(
        parent_ouv,
        parent_shape,
        top_left,
        output_shape,
        horizontal,
        vertical,
    )
    frame_arrays = {
        "full_raster_best_fit_physical_ouv_ap_dv_ml_um_float64": parent_ouv,
        "cropped_pre_reflection_physical_ouv_ap_dv_ml_um_float64": cropped,
        "model_raster_physical_ouv_ap_dv_ml_um_float64": model,
    }
    frame_transform = {
        "quicknii_pixel_contract": "O + (x/W) U + (y/H) V; no half-pixel term",
        "crop_is_upstream_observation_crop": True,
        "crop_window_id": "pose-v2-crop",
        "parent_shape_h_w": list(parent_shape),
        "top_left_y_x": list(top_left),
        "output_shape_h_w": list(output_shape),
        "horizontal_reflection": horizontal,
        "vertical_reflection": vertical,
        "reflection_order": ["horizontal", "vertical"],
        "reflection_is_raster_reparameterization_not_physical_mirror": True,
        "crop_formula": "Oc=O+(left/W)U+(top/H)V; Uc=(w/W)U; Vc=(h/H)V",
        "horizontal_reflection_formula": "O'=O+((w-1)/w)U; U'=-U",
        "vertical_reflection_formula": "O'=O+((h-1)/h)V; V'=-V",
        "parent_subject_centre_plane_fit_id": "pose-v2-fit",
        "arrays": frame_arrays,
        "array_receipts": {
            name: acquisition._array_receipt(value)
            for name, value in frame_arrays.items()
        },
    }
    frame_transform["frame_transform_id"] = acquisition._payload_sha256(
        {key: value for key, value in frame_transform.items() if key != "arrays"}
    )
    height, width = output_shape
    channels = np.zeros((3, height, width), dtype=np.float32)
    targets = {}
    for name in realization_v2._TARGET_ARRAY_KEYS:
        if "coordinates" in name:
            targets[name] = np.zeros((height, width, 3), dtype=np.float64)
        elif "weight" in name:
            targets[name] = np.zeros((height, width), dtype=np.float32)
        elif "label" in name:
            targets[name] = np.zeros((height, width), dtype=np.int64)
        else:
            targets[name] = np.zeros((height, width), dtype=bool)
    factor_arrays = {
        name: np.zeros((height, width, 2 if "index_yx" in name else 3), dtype=np.float64)
        for name in realization_v2._FACTOR_ARRAY_KEYS
    }
    prepared_receipt = acquisition._json_value(context["receipt"])
    upstream = {
        "v2_context_sha256": context["v2_context_sha256"],
        "live_receipt_bindings": {
            "prepared_context": {
                "receipt_payload": prepared_receipt,
                "receipt_sha256": acquisition._payload_sha256(prepared_receipt),
            }
        },
    }
    paired_modes = {
        name: {"descendant_id": f"{name}-id", "brush_available": name != "smart-brush-absent"}
        for name in realization_v2.TRAINABLE_INPUT_MODES
    }
    artifact = {
        "schema_version": realization_v2.SYNTHETIC_REALIZATION_V2_SCHEMA,
        "algorithm": realization_v2.SYNTHETIC_REALIZATION_V2_ALGORITHM,
        "implementation_source_sha256": realization_v2._source_hashes(),
        "implementation_source_sha256_canonicalization": acquisition.V2_SOURCE_SHA256_CANONICALIZATION,
        "runtime_dependencies": {"numpy_version": np.__version__},
        "asset_dependencies": {
            "learned_checkpoint_dependencies": [],
            "pretrained_feature_dependencies": [],
            "previous_model_dependencies": [],
        },
        "upstream_reference": upstream,
        "provenance": {
            "root_seed_uint64": "0x415154564f320001",
            "split": "train",
            "split_index": 2,
            "animal_index": 3,
            "animal_id": "pose-v2-animal",
            "section_index": 4,
            "observation_index": 5,
            "realization_index": 6,
        },
        "rng_sources": {},
        "mode_selection": {"selected_mode": "smart-brush-absent"},
        "paired_mode_sensitivity_reference": {
            "acquired_observation_id": "pose-v2-observation",
            "frame_transform_id": frame_transform["frame_transform_id"],
            "horizontal_reflection": horizontal,
            "vertical_reflection": vertical,
            "trainable_modes": paired_modes,
            "raw_exclusion_reference": {"trainable": False},
            "emitted_training_row_count": 1,
        },
        "frame_transform": frame_transform,
        "model_input": {
            "channel_names": list(realization_v2.MODEL_INPUT_CHANNEL_NAMES),
            "channels_float32": channels,
            "channels_array_receipt": acquisition._array_receipt(channels),
            "spatial_shape_h_w": list(output_shape),
            "strict_allowlist": list(realization_v2.MODEL_INPUT_CHANNEL_NAMES),
        },
        "targets": targets,
        "target_array_receipts": {
            name: acquisition._array_receipt(value) for name, value in targets.items()
        },
        "target_policy": {"pose_target_policy": "fixture"},
        "factor_truth": {
            "arrays": factor_arrays,
            "array_receipts": {
                name: acquisition._array_receipt(value)
                for name, value in factor_arrays.items()
            },
        },
        "training_row_id": "pose-v2-training-row",
    }
    artifact["synthetic_realization_id"] = acquisition._payload_sha256(
        realization_v2._identity_payload(artifact)
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        realization_v2.synthetic_realization_receipt_v2(artifact)
    )
    return artifact


def test_public_physical_ouv_state_round_trip_is_differentiable():
    center = torch.tensor([91.0, -37.0, 218.0], dtype=torch.float64, requires_grad=True)
    raw_rotation = torch.tensor(
        [0.3, 1.1, -0.7, -0.9, 0.2, 0.8], dtype=torch.float64, requires_grad=True
    )
    log_diagonal = torch.log(
        torch.tensor([731.0, 419.0], dtype=torch.float64)
    ).detach().requires_grad_()
    shear = torch.tensor(-0.27, dtype=torch.float64, requires_grad=True)
    frame = rotation_6d_to_frame(raw_rotation)
    basis = positive_inplane_basis(log_diagonal, shear)
    ouv = frame_to_physical_ouv(center, frame, basis)
    recovered_center, recovered_frame, recovered_basis = physical_ouv_to_frame(ouv)
    recovered_rotation = frame_to_rotation_6d(recovered_frame)
    recovered_log, recovered_shear = inplane_basis_to_parameters(recovered_basis)
    assert torch.allclose(
        frame_to_physical_ouv(recovered_center, recovered_frame, recovered_basis),
        ouv,
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(rotation_6d_to_frame(recovered_rotation), recovered_frame)
    assert torch.allclose(
        positive_inplane_basis(recovered_log, recovered_shear), recovered_basis
    )
    (recovered_center.square().sum() + recovered_frame.square().sum() + recovered_basis.square().sum()).backward()
    for value in (center.grad, raw_rotation.grad, log_diagonal.grad, shear.grad):
        assert value is not None and torch.isfinite(value).all()


def test_pose_truth_exact_replay_strict_tamper_and_no_probability_claim(prepared_context):
    final = _final_realization(prepared_context)
    truth = pose_v2.make_arbitrary_plane_pose_truth_v2(final, prepared_context)
    replayed = pose_v2.replay_arbitrary_plane_pose_truth_v2(
        truth, final, prepared_context
    )
    pose_v2.verify_arbitrary_plane_pose_truth_v2(truth, final, prepared_context)
    assert pose_v2.finite_plane_pose_truth_receipt_v2(truth) == (
        pose_v2.finite_plane_pose_truth_receipt_v2(replayed)
    )
    assert truth["scope"]["posterior_or_probability_claim"] is False
    assert truth["scope"]["calibrated_uncertainty_claim"] is False
    assert all(not value for value in truth["asset_dependencies"].values())
    assert pose_v2._count_synthetic_realization_ids(truth) == 1
    for name in truth["arrays"]:
        assert np.array_equal(truth["arrays"][name], replayed["arrays"][name])

    changed = _thaw(truth)
    changed["arrays"][
        "actual_plane_normal_and_signed_offset_um_float64"
    ][3] += 1.0
    changed["array_receipts"] = pose_v2._array_receipts(changed["arrays"])
    changed["finite_plane_pose_truth_id"] = acquisition._payload_sha256(
        pose_v2._identity_payload(changed)
    )
    changed["receipt_sha256"] = acquisition._payload_sha256(
        pose_v2.finite_plane_pose_truth_receipt_v2(changed)
    )
    with pytest.raises(ValueError):
        pose_v2.verify_arbitrary_plane_pose_truth_v2(
            changed, final, prepared_context
        )

    changed_final = _thaw(final)
    changed_final["frame_transform"]["arrays"][
        "model_raster_physical_ouv_ap_dv_ml_um_float64"
    ][0, 0] += 0.5
    with pytest.raises(ValueError):
        pose_v2.make_arbitrary_plane_pose_truth_v2(changed_final, prepared_context)


def test_crop_and_all_reflections_preserve_plane_and_reconstruct_model_ouv(prepared_context):
    truths = {}
    for horizontal in (False, True):
        for vertical in (False, True):
            final = _final_realization(prepared_context, horizontal, vertical)
            truth = pose_v2.make_arbitrary_plane_pose_truth_v2(final, prepared_context)
            pose_v2.verify_arbitrary_plane_pose_truth_v2(
                truth, final, prepared_context
            )
            truths[(horizontal, vertical)] = truth
            assert truth["reflection_state"]["horizontal_reflection"] is horizontal
            assert truth["reflection_state"]["vertical_reflection"] is vertical
            assert np.array_equal(
                truth["arrays"][
                    "model_raster_physical_ouv_ap_dv_ml_um_float64"
                ],
                final["frame_transform"]["arrays"][
                    "model_raster_physical_ouv_ap_dv_ml_um_float64"
                ],
            )
            model = truth["arrays"][
                "model_raster_physical_ouv_ap_dv_ml_um_float64"
            ]
            quicknii = truth["arrays"][
                "model_raster_quicknii_ouv_ml_ap_dv_float64"
            ]
            contract = truth["coordinate_contract"]
            restored = np.stack(
                (
                    allen_index_to_physical_um_points(
                        quicknii_to_allen_points(
                            torch.from_numpy(np.array(quicknii[0], copy=True)),
                            tuple(contract["atlas_shape_ap_dv_ml"]),
                        ),
                        tuple(contract["physical_origin_ap_dv_ml_um"]),
                        tuple(contract["voxel_size_ap_dv_ml_um"]),
                    ).numpy(),
                    allen_index_to_physical_um_vectors(
                        quicknii_to_allen_vectors(
                            torch.from_numpy(np.array(quicknii[1], copy=True))
                        ),
                        tuple(contract["voxel_size_ap_dv_ml_um"]),
                    ).numpy(),
                    allen_index_to_physical_um_vectors(
                        quicknii_to_allen_vectors(
                            torch.from_numpy(np.array(quicknii[2], copy=True))
                        ),
                        tuple(contract["voxel_size_ap_dv_ml_um"]),
                    ).numpy(),
                )
            )
            assert np.allclose(restored, model, rtol=2e-13, atol=2e-12)
            model_center, model_frame, _ = physical_ouv_to_frame(
                torch.from_numpy(np.array(model, copy=True).reshape(9))
            )
            support_origin = truth["arrays"][
                "support_origin_ap_dv_ml_um_float64"
            ]
            model_offset = float(
                model_frame[:, 2].numpy() @ (model_center.numpy() - support_origin)
            )
            canonical_model = canonicalize_plane(
                model_frame[:, 2].numpy(), model_offset
            )
            canonical_truth = truth["arrays"][
                "canonical_plane_normal_and_signed_offset_um_float64"
            ]
            assert np.allclose(canonical_model[0], canonical_truth[:3], atol=2e-12)
            assert np.isclose(canonical_model[1], canonical_truth[3], atol=2e-10)

    reference = truths[(False, False)]["arrays"]
    invariant = pose_v2._ARRAY_KEYS - {
        "model_raster_physical_ouv_ap_dv_ml_um_float64",
        "model_raster_quicknii_ouv_ml_ap_dv_float64",
    }
    for truth in truths.values():
        for name in invariant:
            assert np.array_equal(reference[name], truth["arrays"][name])


def test_coupled_antipodal_serialization_gauge_and_transport(prepared_context):
    truth = pose_v2.make_arbitrary_plane_pose_truth_v2(
        _final_realization(prepared_context), prepared_context
    )
    arrays = truth["arrays"]
    actual = arrays["actual_plane_normal_and_signed_offset_um_float64"]
    canonical = arrays["canonical_plane_normal_and_signed_offset_um_float64"]
    sign = truth["plane_serialization"]["actual_to_canonical_sign"]
    assert sign == -1
    assert np.array_equal(canonical, sign * actual)
    actual_gauge = arrays["actual_tangent_gauge_ap_dv_ml_float64"]
    canonical_gauge = arrays["canonical_tangent_gauge_ap_dv_ml_float64"]
    transport = arrays[
        "actual_to_canonical_tangent_offset_transport_float64"
    ]
    coordinates = np.asarray([0.31, -0.47, 29.0])
    transported = transport @ coordinates
    assert np.allclose(
        canonical_gauge @ transported[:2],
        sign * (actual_gauge @ coordinates[:2]),
        atol=2e-12,
    )
    assert transported[2] == sign * coordinates[2]
    frame = arrays["proper_frame_ap_dv_ml_float64"]
    rotation = arrays["rotation_6d_ap_dv_ml_float64"]
    assert np.allclose(
        rotation_6d_to_frame(torch.from_numpy(np.array(rotation, copy=True))).numpy(),
        frame,
        atol=2e-12,
    )
