from collections.abc import Mapping
import os

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_observation_v2 as observation
import training.arbitrary_plane_realization_v2 as realization
import training.arbitrary_plane_section_processing_v2 as processing
import training.arbitrary_plane_subject_slab_v2 as subject_slab
import training.arbitrary_plane_support_resolution_v2 as support_resolution_stage
from training.arbitrary_plane_support import build_annotation_support_index
from training.arbitrary_plane_subject_deformation_v2 import (
    sample_animal_subject_deformation_plan_v2,
)


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    return value


def _flip(array, horizontal, vertical):
    result = np.asarray(array)
    if horizontal:
        result = result[:, ::-1, ...]
    if vertical:
        result = result[::-1, :, ...]
    return np.ascontiguousarray(result)


def _count_final_ids(value):
    if isinstance(value, Mapping):
        return int("synthetic_realization_id" in value) + sum(
            _count_final_ids(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(_count_final_ids(item) for item in value)
    return 0


@pytest.fixture
def chain(monkeypatch):
    parent_shape = (5, 6)
    top_left = (1, 2)
    output_shape = (3, 3)
    origin = np.asarray([10.0, 20.0, 30.0])
    edge_u = np.asarray([42.0, 6.0, 3.0])
    edge_v = np.asarray([3.0, 30.0, 9.0])
    parent_ouv = np.stack((origin, edge_u, edge_v))
    y, x = np.indices(parent_shape, dtype=np.float64)
    source_index = np.stack(
        (
            y + 0.06 * np.sin((x + 1.0) / 2.0),
            x + 0.08 * np.cos((y + 1.0) / 2.0),
        ),
        -1,
    )
    nominal_at_pullback = (
        origin
        + (source_index[..., 1] / parent_shape[1])[..., None] * edge_u
        + (source_index[..., 0] / parent_shape[0])[..., None] * edge_v
    )
    animal = np.stack(
        (
            0.05 * source_index[..., 0] * source_index[..., 1],
            0.03 * source_index[..., 0] ** 2,
            -0.04 * source_index[..., 1] ** 2,
        ),
        -1,
    )
    exact_parent = nominal_at_pullback + animal
    top, left = top_left
    height, width = output_shape
    window = np.s_[top : top + height, left : left + width]
    exact = np.ascontiguousarray(exact_parent[window], dtype=np.float64)
    raw = (np.arange(height * width, dtype=np.float32).reshape(output_shape) + 1) / 10
    accurate_mask = np.asarray(
        [[False, True, False], [True, True, True], [False, True, False]]
    )
    imperfect_mask = np.asarray(
        [[True, True, False], [True, False, True], [False, True, True]]
    )
    zeros = np.zeros(output_shape, dtype=bool)

    def descendant(name, mask, available, trainable, identifier):
        image = np.where(mask, raw, np.float32(0)) if available else raw.copy()
        error = mask ^ accurate_mask if name == "smart-brush-imperfect" else zeros
        return {
            "trainable": trainable,
            "brush_available": available,
            "descendant_id": identifier,
            "arrays": {
                "model_input_image_float32": image.astype(np.float32),
                "selected_input_mask": mask.copy(),
                "brush_mask_error_mask": error.copy(),
            },
        }

    tissue = np.ones(output_shape, dtype=bool)
    tissue[0, 0] = False
    valid = tissue.copy()
    valid[-1, -1] = False
    weight = valid.astype(np.float32)
    labels = np.arange(height * width, dtype=np.int64).reshape(output_shape) + 1
    observation_arrays = {
        "source_label_ground_truth_crop_int64": labels,
        "source_tissue_ground_truth_mask": tissue,
        "source_correspondence_domain_mask": np.ones(output_shape, dtype=bool),
        "source_dense_correspondence_weight_float32": weight,
        "source_dense_correspondence_abstention_mask": ~valid,
        "processed_mapped_ccf_physical_coordinates_crop_float64": exact,
        "processed_bilinear_domain_valid_mask": np.ones(output_shape, dtype=bool),
        "processed_nearest_domain_valid_mask": np.ones(output_shape, dtype=bool),
        "processed_dense_coordinate_valid_mask": np.ones(output_shape, dtype=bool),
        "physical_loss_mask": np.zeros(output_shape, dtype=bool),
        "occlusion_mask": np.zeros(output_shape, dtype=bool),
        "appearance_artifact_mask": np.zeros(output_shape, dtype=bool),
        "damage_union_mask": np.zeros(output_shape, dtype=bool),
        "observable_footprint_mask": tissue,
        "observation_invalid_mask": np.zeros(output_shape, dtype=bool),
        "outside_correspondence_domain_mask": np.zeros(output_shape, dtype=bool),
        "valid_correspondence_mask": valid,
        "valid_correspondence_weight_float32": weight,
    }
    observation = {
        "provenance": {
            "root_seed_uint64": "0x415154564f320001",
            "split": "train",
            "split_index": 8,
            "animal_index": 4,
            "animal_id": "animal-label-does-not-enter-rng",
            "section_index": 2,
            "observation_index": 3,
        },
        "modality": "brightfield-nissl-like",
        "crop_window": {
            "parent_shape_h_w": list(parent_shape),
            "top_left_y_x": list(top_left),
            "output_shape_h_w": list(output_shape),
        },
        "crop_window_id": "crop-window-id",
        "observation_bundle_id": "observation-bundle-id",
        "receipt_sha256": "observation-receipt",
        "acquired_observation_id": "acquired-observation-id",
        "arrays": observation_arrays,
        "descendants": {
            "raw": descendant("raw", zeros, False, False, "raw-id"),
            "smart-brush-accurate": descendant(
                "smart-brush-accurate", accurate_mask, True, True, "accurate-id"
            ),
            "smart-brush-imperfect": descendant(
                "smart-brush-imperfect", imperfect_mask, True, True, "imperfect-id"
            ),
            "smart-brush-absent": descendant(
                "smart-brush-absent", zeros, False, True, "absent-id"
            ),
        },
    }
    receipt_payloads = {
        "prepared_context": {"stage": "prepared-context"},
        "support_resolution": {"stage": "support-resolution"},
        "precursor": {"stage": "precursor"},
        "subject_slab": {"stage": "subject-slab"},
        "section_processing_plan": {"stage": "section-plan"},
        "section_processing_render": {"stage": "section-render"},
        "observation_bundle": {"stage": "observation"},
    }
    prepared_context = {
        "v2_context_sha256": "context-id",
        "receipt": receipt_payloads["prepared_context"],
    }
    precursor = {
        "slab_render_id": "precursor-id",
        "receipt_sha256": realization.acquisition._payload_sha256(
            receipt_payloads["precursor"]
        ),
    }
    support_resolution = {
        "resolution": {
            "status": "accepted",
            "support_resolution_plan_id": "support-plan-id",
            "subject_support_resolution_id": "support-resolution-id",
            "accepted_attempt_index": 1,
            "accepted_precursor_reference": {
                "slab_render_id": precursor["slab_render_id"],
                "receipt_sha256": precursor["receipt_sha256"],
            },
            "accepted_probe_reference": {
                "subject_centre_support_probe_id": "support-probe-id",
                "receipt_sha256": "support-probe-receipt",
            },
            "configuration": {
                "master_root_seed_uint64": "0x535550504f525401",
                "split_index": 8,
                "animal_index": 4,
                "section_index": 2,
                "plane_stratum": "general_oblique",
                "nominal_cut_thickness_um": 10.0,
                "axial_step_um_max": 10.0,
                "parent_shape_h_w": list(parent_shape),
                "max_attempts": 4,
            },
            "lineage": {
                "split": "train",
                "animal_id": observation["provenance"]["animal_id"],
                "animal_index": 4,
                "specimen_id": "specimen-4",
                "experiment_id": "experiment-4",
            },
            "receipt_sha256": realization.acquisition._payload_sha256(
                receipt_payloads["support_resolution"]
            ),
        },
        "accepted_precursor": precursor,
        "accepted_probe": {"subject_centre_support_probe_id": "support-probe-id"},
    }
    subject_slab = {
        "subject_slab_render_id": "subject-slab-id",
        "receipt_sha256": realization.acquisition._payload_sha256(
            receipt_payloads["subject_slab"]
        ),
        "coordinate_map": {
            "subject_coordinate_map_id": "subject-coordinate-map-id",
            "deformation_reference": {
                "mode": "nonidentity",
                "synthetic_animal_id": "synthetic-animal-id",
            },
            "centre_plane_fit": {
                "subject_centre_plane_fit_id": "centre-plane-fit-id",
                "output_shape_h_w": list(parent_shape),
                "arrays": {
                    "physical_ouv_ap_dv_ml_um_float64": parent_ouv.reshape(-1)
                },
            },
        },
    }
    section_plan = {
        "section_processing_plan_id": "section-plan-id",
        "section_processing_realization_id": "section-realization-id",
        "receipt_sha256": realization.acquisition._payload_sha256(
            receipt_payloads["section_processing_plan"]
        ),
        "provenance": {
            "split": "train",
            "animal_index": 4,
            "animal_id": observation["provenance"]["animal_id"],
            "section_index": 2,
        },
    }
    processed_render = {
        "section_processing_render_id": "section-render-id",
        "receipt_sha256": realization.acquisition._payload_sha256(
            receipt_payloads["section_processing_render"]
        ),
        "state": {"source_index_yx": source_index},
    }
    calls = {"observation": [], "support_resolution": [], "accepted_slab": []}

    def verified(artifact, *args, **kwargs):
        calls["observation"].append((artifact, args, kwargs))
        if artifact["observation_bundle_id"] != "observation-bundle-id":
            raise ValueError("upstream observation changed")

    def verified_support(bundle, context, *, subject_plan, **kwargs):
        calls["support_resolution"].append((bundle, context, subject_plan, kwargs))
        resolution = bundle["resolution"]
        config = resolution["configuration"]
        lineage = resolution["lineage"]
        expected = {
            "master_root_seed": config["master_root_seed_uint64"],
            "split": lineage["split"],
            "split_index": config["split_index"],
            "animal_index": config["animal_index"],
            "animal_id": lineage["animal_id"],
            "section_index": config["section_index"],
            "plane_stratum": config["plane_stratum"],
            "nominal_cut_thickness_um": config["nominal_cut_thickness_um"],
            "specimen_id": lineage["specimen_id"],
            "experiment_id": lineage["experiment_id"],
            "axial_step_um_max": config["axial_step_um_max"],
            "parent_shape_h_w": tuple(config["parent_shape_h_w"]),
            "max_attempts": config["max_attempts"],
            "batch_size": None,
            "subject_to_ccf_mapper": verified_mapper,
        }
        if (
            resolution["subject_support_resolution_id"] != "support-resolution-id"
            or kwargs != expected
        ):
            raise ValueError("support-resolution stub changed")

    def verified_accepted_slab(bundle, artifact):
        calls["accepted_slab"].append((bundle, artifact))
        if artifact["subject_slab_render_id"] != "subject-slab-id":
            raise ValueError("accepted subject slab stub changed")

    verified_mapper = object()
    monkeypatch.setattr(
        realization,
        "_verified_subject_to_ccf_mapper_for_context_v2",
        lambda context, subject_plan: verified_mapper,
    )
    monkeypatch.setattr(
        realization,
        "_verify_arbitrary_plane_observation_with_mapper_v2",
        verified,
    )
    monkeypatch.setattr(
        realization,
        "_verify_subject_support_resolution_with_mapper_v2",
        verified_support,
    )
    monkeypatch.setattr(
        realization,
        "verify_accepted_subject_slab_matches_support_resolution_v2",
        verified_accepted_slab,
    )
    monkeypatch.setattr(
        realization,
        "subject_support_resolution_receipt_v2",
        lambda artifact: receipt_payloads["support_resolution"],
    )
    monkeypatch.setattr(
        realization,
        "_precursor_contract_and_receipt",
        lambda artifact: ("test", receipt_payloads["precursor"]),
    )
    monkeypatch.setattr(
        realization,
        "subject_slab_render_receipt_v2",
        lambda artifact: receipt_payloads["subject_slab"],
    )
    monkeypatch.setattr(
        realization,
        "section_processing_plan_receipt_v2",
        lambda artifact: receipt_payloads["section_processing_plan"],
    )
    monkeypatch.setattr(
        realization,
        "section_processing_render_receipt_v2",
        lambda artifact: receipt_payloads["section_processing_render"],
    )
    monkeypatch.setattr(
        realization,
        "observation_bundle_receipt_v2",
        lambda artifact: receipt_payloads["observation_bundle"],
    )
    observation["receipt_sha256"] = realization.acquisition._payload_sha256(
        receipt_payloads["observation_bundle"]
    )
    return {
        "prepared_context": prepared_context,
        "support_resolution": support_resolution,
        "precursor": precursor,
        "subject_slab": subject_slab,
        "section_plan": section_plan,
        "processed_render": processed_render,
        "observation": observation,
        "subject_plan": None,
        "parent_ouv": parent_ouv,
        "calls": calls,
    }


def _make(chain, index):
    return realization.make_arbitrary_plane_realization_v2(
        chain["prepared_context"],
        chain["support_resolution"],
        chain["precursor"],
        chain["subject_slab"],
        chain["section_plan"],
        chain["processed_render"],
        chain["observation"],
        subject_plan=chain["subject_plan"],
        realization_index=index,
    )


def _verify(chain, result):
    realization.verify_arbitrary_plane_realization_v2(
        result,
        chain["prepared_context"],
        chain["support_resolution"],
        chain["precursor"],
        chain["subject_slab"],
        chain["section_plan"],
        chain["processed_render"],
        chain["observation"],
        subject_plan=chain["subject_plan"],
    )


def test_exact_input_allowlist_upstream_verification_replay_and_tamper(chain):
    result = _make(chain, 0)
    assert all(len(calls) == 1 for calls in chain["calls"].values())
    _verify(chain, result)
    assert all(len(calls) == 2 for calls in chain["calls"].values())
    model_input = result["model_input"]
    assert set(model_input) == {
        "channel_names",
        "channels_float32",
        "channels_array_receipt",
        "spatial_shape_h_w",
        "strict_allowlist",
    }
    assert tuple(model_input["channel_names"]) == realization.MODEL_INPUT_CHANNEL_NAMES
    assert model_input["channels_float32"].dtype == np.float32
    assert result["target_policy"]["selected_input_mask_may_gate_any_loss"] is False
    assert result["mode_selection"]["selected_mode"] in realization.TRAINABLE_INPUT_MODES
    assert result["mode_selection"]["raw_mode_trainable"] is False
    assert result["mode_selection"]["emitted_training_row_count"] == 1
    paired_reference = result["paired_mode_sensitivity_reference"]
    assert set(paired_reference["trainable_modes"]) == set(
        realization.TRAINABLE_INPUT_MODES
    )
    assert paired_reference["emitted_training_row_count"] == 1
    assert paired_reference["raw_exclusion_reference"] == {
        "descendant_id": chain["observation"]["descendants"]["raw"][
            "descendant_id"
        ],
        "trainable": False,
        "equivalent_trainable_mode": "smart-brush-absent",
    }
    assert _count_final_ids(result) == 1
    assert result["receipt_sha256"] == realization.acquisition._payload_sha256(
        realization.synthetic_realization_receipt_v2(result)
    )
    bindings = result["upstream_reference"]["live_receipt_bindings"]
    assert set(bindings) == {
        "prepared_context",
        "support_resolution",
        "precursor",
        "subject_slab",
        "section_processing_plan",
        "section_processing_render",
        "observation_bundle",
    }
    assert all(
        item["receipt_sha256"]
        == realization.acquisition._payload_sha256(
            realization.acquisition._json_value(item["receipt_payload"])
        )
        for item in bindings.values()
    )

    replay = realization.replay_arbitrary_plane_realization_v2(
        result,
        chain["prepared_context"],
        chain["support_resolution"],
        chain["precursor"],
        chain["subject_slab"],
        chain["section_plan"],
        chain["processed_render"],
        chain["observation"],
        subject_plan=chain["subject_plan"],
    )
    assert realization.synthetic_realization_receipt_v2(
        replay
    ) == realization.synthetic_realization_receipt_v2(result)

    tampered = _thaw(result)
    tampered["model_input"]["channels_float32"][0, 0, 0] += 1.0
    with pytest.raises(ValueError):
        _verify(chain, tampered)
    tampered = _thaw(result)
    tampered["targets"]["valid_correspondence_mask"][0, 0] ^= True
    with pytest.raises(ValueError):
        _verify(chain, tampered)
    upstream = _thaw(chain["observation"])
    upstream["observation_bundle_id"] = "tampered"
    with pytest.raises(ValueError):
        realization.make_arbitrary_plane_realization_v2(
            chain["prepared_context"],
            chain["support_resolution"],
            chain["precursor"],
            chain["subject_slab"],
            chain["section_plan"],
            chain["processed_render"],
            upstream,
            subject_plan=None,
            realization_index=0,
        )
    upstream = _thaw(chain["support_resolution"])
    upstream["resolution"]["subject_support_resolution_id"] = "tampered"
    with pytest.raises(ValueError):
        realization.make_arbitrary_plane_realization_v2(
            chain["prepared_context"],
            upstream,
            chain["precursor"],
            chain["subject_slab"],
            chain["section_plan"],
            chain["processed_render"],
            chain["observation"],
            subject_plan=None,
            realization_index=0,
        )
    upstream = _thaw(chain["support_resolution"])
    upstream["resolution"]["configuration"]["split_index"] += 1
    with pytest.raises(ValueError, match="scheduling lineage"):
        realization.make_arbitrary_plane_realization_v2(
            chain["prepared_context"],
            upstream,
            chain["precursor"],
            chain["subject_slab"],
            chain["section_plan"],
            chain["processed_render"],
            chain["observation"],
            subject_plan=None,
            realization_index=0,
        )
    upstream = _thaw(chain["precursor"])
    upstream["receipt_sha256"] = "tampered"
    with pytest.raises(ValueError):
        realization.make_arbitrary_plane_realization_v2(
            chain["prepared_context"],
            chain["support_resolution"],
            upstream,
            chain["subject_slab"],
            chain["section_plan"],
            chain["processed_render"],
            chain["observation"],
            subject_plan=None,
            realization_index=0,
        )


def test_crop_and_all_reflections_match_quicknii_and_reconstruct_exact_map(chain):
    indices = {}
    for index in range(128):
        choice = realization.sample_synthetic_realization_choice_v2(
            chain["observation"], index
        )
        key = (choice["horizontal_reflection"], choice["vertical_reflection"])
        indices.setdefault(key, index)
    assert set(indices) == {(False, False), (True, False), (False, True), (True, True)}

    parent_ouv = chain["parent_ouv"]
    origin, edge_u, edge_v = parent_ouv
    parent_height, parent_width = chain["observation"]["crop_window"][
        "parent_shape_h_w"
    ]
    top, left = chain["observation"]["crop_window"]["top_left_y_x"]
    height, width = chain["observation"]["crop_window"]["output_shape_h_w"]
    exact = chain["observation"]["arrays"][
        "processed_mapped_ccf_physical_coordinates_crop_float64"
    ]
    for (horizontal, vertical), index in indices.items():
        result = _make(chain, index)
        expected = np.stack(
            (
                origin + (left / parent_width) * edge_u + (top / parent_height) * edge_v,
                (width / parent_width) * edge_u,
                (height / parent_height) * edge_v,
            )
        )
        if horizontal:
            expected[0] += ((width - 1) / width) * expected[1]
            expected[1] *= -1
        if vertical:
            expected[0] += ((height - 1) / height) * expected[2]
            expected[2] *= -1
        assert np.array_equal(
            result["frame_transform"]["arrays"][
                "model_raster_physical_ouv_ap_dv_ml_um_float64"
            ],
            expected,
        )
        assert np.array_equal(
            result["targets"][
                "processed_mapped_ccf_physical_coordinates_crop_float64"
            ],
            _flip(exact, horizontal, vertical),
        )
        factor = result["factor_truth"]["arrays"]
        finite = np.isfinite(exact).all(axis=-1)
        assert np.allclose(
            factor["nominal_physical_map_ap_dv_ml_um_float64"][finite]
            + factor["composed_coordinate_residual_ap_dv_ml_um_float64"][finite],
            result["targets"][
                "processed_mapped_ccf_physical_coordinates_crop_float64"
            ][finite],
            rtol=0.0,
            atol=1e-10,
        )
        mode = result["mode_selection"]["selected_mode"]
        descendant = chain["observation"]["descendants"][mode]
        assert np.array_equal(
            result["model_input"]["channels_float32"][0],
            _flip(
                descendant["arrays"]["model_input_image_float32"],
                horizontal,
                vertical,
            ),
        )


def test_paired_modes_share_observation_reflection_targets_and_raw_never_emits(chain):
    groups = {}
    for index in range(512):
        choice = realization.sample_synthetic_realization_choice_v2(
            chain["observation"], index
        )
        key = (choice["horizontal_reflection"], choice["vertical_reflection"])
        groups.setdefault(key, {}).setdefault(choice["selected_mode"], index)
    paired = next(group for group in groups.values() if len(group) == 3)
    results = {mode: _make(chain, index) for mode, index in paired.items()}
    assert set(results) == set(realization.TRAINABLE_INPUT_MODES)
    reference = results["smart-brush-accurate"]
    for result in results.values():
        assert result["upstream_reference"]["acquired_observation_id"] == (
            reference["upstream_reference"]["acquired_observation_id"]
        )
        assert np.array_equal(
            result["frame_transform"]["arrays"][
                "model_raster_physical_ouv_ap_dv_ml_um_float64"
            ],
            reference["frame_transform"]["arrays"][
                "model_raster_physical_ouv_ap_dv_ml_um_float64"
            ],
        )
        for name in realization._TARGET_ARRAY_KEYS - {
            "selected_brush_mask_error_mask"
        }:
            assert np.array_equal(result["targets"][name], reference["targets"][name])
        assert result["mode_selection"]["selected_mode"] != "raw"
        assert result["mode_selection"]["raw_mode_trainable"] is False
        assert result["mode_selection"]["emitted_training_row_count"] == 1
        assert (
            result["paired_mode_sensitivity_reference"]
            == reference["paired_mode_sensitivity_reference"]
        )
    assert len({result["training_row_id"] for result in results.values()}) == 3
    for mode, descendant in chain["observation"]["descendants"].items():
        if mode not in realization.TRAINABLE_INPUT_MODES:
            continue
        stored = reference["paired_mode_sensitivity_reference"]["trainable_modes"][
            mode
        ]
        horizontal = reference["frame_transform"]["horizontal_reflection"]
        vertical = reference["frame_transform"]["vertical_reflection"]
        assert realization.acquisition._json_value(
            stored["reflected_model_input_image_receipt"]
        ) == realization.acquisition._json_value(
            realization.acquisition._array_receipt(
                _flip(
                    descendant["arrays"]["model_input_image_float32"],
                    horizontal,
                    vertical,
                )
            )
        )
        assert realization.acquisition._json_value(
            stored["reflected_selected_input_mask_receipt"]
        ) == realization.acquisition._json_value(
            realization.acquisition._array_receipt(
                _flip(
                    descendant["arrays"]["selected_input_mask"],
                    horizontal,
                    vertical,
                )
            )
        )
    assert all(
        result["mode_selection"]["selected_descendant_id"]
        != result["mode_selection"]["raw_descendant_id"]
        for result in results.values()
    )


def test_rng_choice_is_independent_of_animal_label_strings(chain):
    relabelled = _thaw(chain["observation"])
    relabelled["provenance"]["animal_id"] = "completely-different-label"
    for index in range(32):
        assert realization.sample_synthetic_realization_choice_v2(
            relabelled, index
        ) == realization.sample_synthetic_realization_choice_v2(
            chain["observation"], index
        )
    assert "animal_id" not in realization.signature(
        realization.derive_synthetic_realization_seed_v2
    ).parameters


def test_rng_accepts_any_nonempty_split_name(chain):
    observation = _thaw(chain["observation"])
    observation["provenance"]["split"] = "untouched-final-test-animals"
    assert realization.sample_synthetic_realization_choice_v2(
        observation, 9
    ) == realization.sample_synthetic_realization_choice_v2(observation, 9)
    with pytest.raises(ValueError):
        realization.derive_synthetic_realization_seed_v2(
            0,
            "",
            0,
            0,
            0,
            0,
            0,
            "trainable-input-mode",
        )


@pytest.mark.parametrize("value", [1.0, 1.9, True, np.bool_(False)])
def test_final_rng_rejects_noninteger_schedule_coordinates(value):
    with pytest.raises((TypeError, ValueError)):
        realization.derive_synthetic_realization_seed_v2(
            7, "development", value, 2, 3, 4, 5, "mode"
        )


@pytest.mark.parametrize("value", [1.0, 1.9, True, np.bool_(False)])
def test_final_realization_rejects_noninteger_realization_index(chain, value):
    with pytest.raises((TypeError, ValueError)):
        _make(chain, value)


@pytest.mark.skipif(
    os.environ.get("RUN_REALIZATION_V2_INTEGRATION") != "1",
    reason="deferred authenticated-chain integration run",
)
def test_real_authenticated_nonidentity_chain_replay_and_receipt_binding():
    annotation = np.zeros((17, 15, 13), dtype=np.uint16)
    annotation[2:15, 3:13, 1:11] = 7
    annotation[7:11, 4:8, 7:11] = 19
    ap, dv, ml = np.indices(annotation.shape)
    scalar = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.float32)
    support = build_annotation_support_index(
        annotation,
        atlas_id="realization-chain-fixture",
        atlas_version="fixture-v1",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256="3" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(11.0, 17.0, 29.0),
        origin_um=(-71.0, 23.0, 107.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    context = acquisition.prepare_arbitrary_plane_acquisition_context_v2(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="fixture decoder",
        annotation_decoder="fixture decoder",
    )
    lower = np.asarray(support["origin_um"], dtype=np.float64)
    upper = lower + np.asarray(support["annotation_shape"], dtype=np.float64) * np.asarray(
        support["voxel_size_um"], dtype=np.float64
    )
    subject_plan = sample_animal_subject_deformation_plan_v2(
        lower,
        upper,
        root_seed="0x47454e4552494303",
        split="train",
        animal_index=5,
        animal_id="animal-five",
        ccf_context_sha256=context["v2_context_sha256"],
        coarse_spacing_um=500.0,
        fine_spacing_um=250.0,
        coarse_padding_um=2000.0,
        fine_padding_um=1000.0,
        smoothing_sigma_knots=0.7,
        a0_um=5.0,
        max_local_displacement_um=100.0,
        speed_l2_bound_um_max=100.0,
        minimum_halo_um=0.0,
        integration_steps=2,
    )
    assert subject_plan["resolved_config"]["deformation_stratum"] == "standard"
    assert subject_plan["realization"]["accepted_amplitude_um"] > 0.0
    support_bundle = support_resolution_stage.resolve_subject_support_v2(
        context,
        subject_plan=subject_plan,
        master_root_seed="0x5355505245534f4c",
        split="train",
        split_index=3,
        animal_index=5,
        animal_id="animal-five",
        section_index=41,
        plane_stratum="general_oblique",
        nominal_cut_thickness_um=55.0,
        specimen_id="specimen-five-a",
        experiment_id="experiment-five",
        parent_shape_h_w=(256, 256),
    )
    assert support_bundle["resolution"]["status"] == "accepted"
    precursor = support_bundle["accepted_precursor"]
    subject = subject_slab.make_subject_slab_render_v2(
        context, precursor, subject_plan=subject_plan
    )
    assert subject["coordinate_map"]["deformation_reference"]["mode"] == (
        "accepted-subject-deformation"
    )
    assert subject["synthetic_animal_id"] == subject_plan["synthetic_animal_id"]
    coordinate = subject["coordinate_map"]
    centre_index = coordinate["kernel"]["centre_index"]
    subject_physical = coordinate["arrays"][
        "subject_physical_coordinates_ap_dv_ml_um_float64"
    ][centre_index]
    pitch, _, _ = processing._orthogonal_section_pixel_metric(subject_physical)
    section_plan = processing.sample_section_processing_plan_v2(
        tuple(subject_physical.shape[:2]),
        tuple(pitch),
        root_seed="0x5245414c495a4502",
        split="train",
        animal_index=5,
        section_index=41,
        animal_id="animal-five",
        section_id="integration-section",
        deformation_mode="standard",
    )
    assert section_plan["resolved_config"]["deformation_mode"] == "standard"
    assert section_plan["realization"]["accepted_amplitude_um"] > 0.0
    section_render = processing.make_section_processing_render_v2(
        subject,
        section_plan,
        context,
        precursor,
        subject_plan=subject_plan,
    )
    assert section_render["identity_reference_path"] is False
    observation_bundle = observation.make_arbitrary_plane_observation_v2(
        section_render,
        subject,
        section_plan,
        context,
        precursor,
        subject_plan=subject_plan,
        root_seed="0x5245414c495a4503",
        split="train",
        split_index=3,
        animal_index=5,
        animal_id="animal-five",
        section_index=41,
        observation_index=2,
        modality="brightfield-nissl-like",
    )
    result = realization.make_arbitrary_plane_realization_v2(
        context,
        support_bundle,
        precursor,
        subject,
        section_plan,
        section_render,
        observation_bundle,
        subject_plan=subject_plan,
        realization_index=7,
    )
    replay = realization.replay_arbitrary_plane_realization_v2(
        result,
        context,
        support_bundle,
        precursor,
        subject,
        section_plan,
        section_render,
        observation_bundle,
        subject_plan=subject_plan,
    )
    realization.verify_arbitrary_plane_realization_v2(
        result,
        context,
        support_bundle,
        precursor,
        subject,
        section_plan,
        section_render,
        observation_bundle,
        subject_plan=subject_plan,
    )
    factor = result["factor_truth"]["arrays"]
    animal_residual = factor[
        "animal_residual_at_section_pullback_ap_dv_ml_um_float64"
    ]
    section_displacement = factor[
        "section_plane_displacement_ap_dv_ml_um_float64"
    ]
    assert np.any(np.isfinite(animal_residual) & (animal_residual != 0.0))
    assert np.any(section_displacement != 0.0)
    assert realization.synthetic_realization_receipt_v2(
        replay
    ) == realization.synthetic_realization_receipt_v2(result)
    assert result["receipt_sha256"] == acquisition._payload_sha256(
        realization.synthetic_realization_receipt_v2(result)
    )
    for binding in result["upstream_reference"]["live_receipt_bindings"].values():
        assert binding["receipt_sha256"] == acquisition._payload_sha256(
            acquisition._json_value(binding["receipt_payload"])
        )
    assert result["upstream_reference"]["live_receipt_bindings"][
        "observation_bundle"
    ]["receipt_sha256"] == observation_bundle["receipt_sha256"]
