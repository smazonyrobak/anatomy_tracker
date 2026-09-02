import copy

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_batch_v3 as batch_v3
import training.arbitrary_plane_finite_joint_curriculum_v5 as curriculum
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_training_row_v3 as training_row_v3
from training.arbitrary_plane_rendered_generator import prepare_finite_render_context
from training.arbitrary_plane_support import build_annotation_support_index


@pytest.fixture(scope="module")
def prepared_context():
    annotation = np.zeros((17, 15, 13), dtype=np.uint16)
    annotation[2:15, 3:13, 1:11] = 7
    annotation[6:11, 6:10, 4:8] = 19
    ap, dv, ml = np.indices(annotation.shape)
    template = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.uint16)
    support = build_annotation_support_index(
        annotation,
        atlas_id="fixture-ccf",
        atlas_version="fixture-v5",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256="3" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(11.0, 17.0, 29.0),
        origin_um=(-71.0, 23.0, 107.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return prepare_finite_render_context(
        template,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="fixture",
        annotation_decoder="fixture",
    )


def _row(prepared_context, **changes):
    config = {
        "root_seed": 2**63 + 701,
        "sample_index": 4,
        "output_shape_h_w": (39, 43),
        "selected_mode": "smart-brush-accurate",
        "reflection_state": "none",
        "amplitude_band": "mild",
        "animal_id": "finite-v5-animal",
        "specimen_id": "finite-v5-specimen",
        "experiment_id": "finite-v5-experiment",
        "synthetic_animal_id": "finite-v5-synthetic-animal",
        "section_id": "finite-v5-section",
        "margin_um": (13.0, 17.0),
        "minimum_brain_pixels": 120,
        "nominal_cut_thickness_um": 55.0,
    }
    config.update(changes)
    return curriculum.make_finite_joint_curriculum_training_row_v5(
        prepared_context, **config
    )


def test_row_is_strict_v4_slab_bound_replayable_and_model_independent(
    prepared_context,
):
    row = _row(prepared_context)
    upstream = row["upstream_reference"]
    support = upstream["support_supervision_contract"]

    assert psf_v4.verify_training_row_v4(
        row, capability=psf_v4.finite_psf_model_capability_v4()
    )
    assert curriculum.verify_finite_joint_curriculum_training_row_v5(
        row, prepared_context
    )
    assert row["schema_version"] == psf_v4.TRAINING_ROW_V4_SCHEMA
    assert set(row["arrays"]) == training_row_v3._ARRAY_KEYS
    assert row["finite_psf_contract"]["nominal_cut_thickness_um"] == 55.0
    assert row["finite_psf_contract"]["axial_sample_count"] == 9
    assert row["finite_psf_contract"]["slab_observation_v4_receipt_sha256"] == (
        upstream["slab_observation_v4_receipt_sha256"]
    )
    assert row["finite_psf_contract"]["finite_psf_sha256"] == upstream[
        "finite_psf_sha256"
    ]
    assert upstream["centre_plane_targets_receipt_sha256"] == upstream[
        "finite_slab_identity"
    ]["centre_plane_targets_receipt_sha256"]
    assert support["point_pose_metric"] == (
        "post-G1 sum(source_slab_brain_occupancy_float32)"
    )
    assert support["post_g1_slab_effective_brain_pixel_mass"] == next(
        iter(support["paired_post_g1_slab_effective_brain_pixel_mass"].values())
    )
    assert support["point_pose_supervision_weight"] == float(
        support["post_g1_slab_effective_brain_pixel_mass"] >= 120
    )
    assert support["dense_deformation_supervision_weight"] == float(
        support["center_gauge_support_pixel_count"] >= 120
        and support["post_g3_dense_correspondence_weight_mass"] > 0.0
        and not upstream["selected_g1_accepted_attempt"]["identity_path"]
    )
    assert np.array_equal(
        row["arrays"]["target_correspondence_abstention_mask"],
        row["arrays"]["target_correspondence_weight_float32"] <= 0.0,
    )
    assert all(
        row[name] == []
        for name in (
            "prior_model_dependencies",
            "prior_feature_dependencies",
            "prior_pseudolabel_dependencies",
        )
    )
    assert all(
        upstream[name] == []
        for name in (
            "prior_model_weight_dependencies",
            "prior_feature_dependencies",
            "prior_pseudolabel_dependencies",
        )
    )
    assert row["lineage"] == {
        "animal_id": "finite-v5-animal",
        "specimen_id": "finite-v5-specimen",
        "experiment_id": "finite-v5-experiment",
        "synthetic_animal_id": "finite-v5-synthetic-animal",
        "section_id": "finite-v5-section",
        "split": "development",
    }


def test_horizontal_reflection_preserves_physical_pose_and_reflects_all_targets(
    prepared_context,
):
    plain = _row(prepared_context)
    reflected = _row(prepared_context, reflection_state="horizontal")
    width = plain["arrays"]["model_input_channels_float32"].shape[1]

    assert plain["source_observation_receipt_sha256"] == reflected[
        "source_observation_receipt_sha256"
    ]
    assert plain["proper_physical_pose_unchanged"] == reflected[
        "proper_physical_pose_unchanged"
    ]
    for name in (
        "model_input_channels_float32",
        "source_label_ground_truth_canvas_int64",
        "source_tissue_ground_truth_mask",
        "target_ccf_coordinates_ap_dv_ml_um_float64",
        "target_valid_correspondence_mask",
        "target_correspondence_weight_float32",
        "target_correspondence_abstention_mask",
        "truth_section_deformation_valid_mask",
    ):
        assert np.array_equal(
            reflected["arrays"][name], np.flip(plain["arrays"][name], axis=1)
        )
    expected_map = np.flip(
        plain["arrays"]["truth_section_pullback_map_yx_px_float64"], axis=1
    ).copy()
    expected_map[..., 1] = width - 1.0 - expected_map[..., 1]
    expected_velocity = np.flip(
        plain["arrays"][
            "truth_section_pullback_stationary_velocity_yx_px_float64"
        ],
        axis=1,
    ).copy()
    expected_velocity[..., 1] *= -1.0
    assert np.array_equal(
        reflected["arrays"]["truth_section_pullback_map_yx_px_float64"],
        expected_map,
    )
    assert np.array_equal(
        reflected["arrays"][
            "truth_section_pullback_stationary_velocity_yx_px_float64"
        ],
        expected_velocity,
    )


def test_default_production_thickness_is_independently_seeded_and_replays(
    prepared_context,
):
    row = _row(
        prepared_context,
        root_seed=2**63 + 733,
        sample_index=5,
        nominal_cut_thickness_um=None,
        section_id="finite-v5-seeded-thickness-section",
    )
    upstream = row["upstream_reference"]
    selection = upstream["thickness_selection"]
    numeric = row["numeric_rng_provenance"]

    assert selection["selection_mode"] == "independent-seeded-uniform"
    assert selection["thickness_seed_uint64"] == numeric[
        "finite_thickness_seed_uint64"
    ]
    assert numeric["finite_thickness_seed_uint64"] != numeric[
        "finite_render_seed_uint64"
    ]
    assert 25.0 <= selection["nominal_cut_thickness_um"] <= 100.0
    assert row["finite_psf_contract"]["nominal_cut_thickness_um"] == selection[
        "nominal_cut_thickness_um"
    ]
    assert curriculum.verify_finite_joint_curriculum_training_row_v5(
        row, prepared_context
    )


def test_canonical_finalizer_and_replay_reject_array_psf_and_source_tamper(
    prepared_context,
):
    row = _row(prepared_context)
    changed = copy.deepcopy(row)
    changed["arrays"]["model_input_channels_float32"][0, 0, 0] += 0.25
    with pytest.raises(ValueError, match="receipt or arrays"):
        psf_v4.verify_training_row_v4(changed)

    changed = copy.deepcopy(row)
    changed["finite_psf_contract"]["axial_offsets_um"][0] += 1.0
    changed["receipt_sha256"] = acquisition._payload_sha256(
        psf_v4.training_row_receipt_v4(changed)
    )
    with pytest.raises(ValueError, match="schedule"):
        psf_v4.verify_training_row_v4(changed)

    changed = copy.deepcopy(row)
    changed["upstream_reference"]["slab_observation_id"] = "0" * 64
    changed["receipt_sha256"] = acquisition._payload_sha256(
        psf_v4.training_row_receipt_v4(changed)
    )
    assert psf_v4.verify_training_row_v4(changed)
    with pytest.raises(ValueError, match="does not replay exactly"):
        curriculum.verify_finite_joint_curriculum_training_row_v5(
            changed, prepared_context
        )


def test_no_drop_reuses_exact_parent_and_slab_across_retries_and_fallback(
    prepared_context, monkeypatch
):
    original = curriculum.make_arbitrary_plane_synthetic_realization
    observed = []

    def reject_nonidentity(parent, support, **kwargs):
        block = kwargs["slab_observation_v4"]
        observed.append(
            (
                id(parent),
                id(block),
                parent["finite_plane_render_id"],
                block["slab_observation_id"],
                kwargs["config_overrides"]["g1"]["identity_probability"],
            )
        )
        if kwargs["config_overrides"]["g1"]["identity_probability"] == 0.0:
            raise ValueError(
                "no G1 realization passed every predeclared topology, cycle, displacement, and FOV gate"
            )
        return original(parent, support, **kwargs)

    monkeypatch.setattr(
        curriculum, "make_arbitrary_plane_synthetic_realization", reject_nonidentity
    )
    row = curriculum.make_finite_joint_curriculum_training_rows_v5(
        prepared_context,
        root_seed=2**63 + 909,
        start_index=0,
        row_count=1,
        output_shape_h_w=(39, 43),
        identity_prefix="finite-no-drop-v5",
        sections_per_animal=1,
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=120,
        maximum_joint_rejection_attempts=2,
        nominal_cut_thickness_um=55.0,
    )[0]
    history = row["upstream_reference"]["joint_rejection_history"]
    support = row["upstream_reference"]["support_supervision_contract"]

    assert len(history) == 2
    assert len({item[0] for item in observed}) == 1
    assert len({item[1] for item in observed}) == 1
    assert len({item[2] for item in observed}) == 1
    assert len({item[3] for item in observed}) == 1
    assert [item[4] for item in observed[:2]] == [0.0, 0.0]
    assert all(item[4] == 1.0 for item in observed[2:])
    assert all(
        item["finite_parent_identity"]
        == row["upstream_reference"]["finite_parent_identity"]
        and item["finite_slab_identity"]
        == row["upstream_reference"]["finite_slab_identity"]
        for item in history
    )
    assert row["upstream_reference"]["deformation_censoring_contract"][
        "status"
    ] == curriculum.IDENTITY_FALLBACK_CENSOR_STATUS
    assert row["upstream_reference"]["selected_synthetic_generator_binding"][
        "resolved_config"
    ]["synthetic_stratum"] == "low-information-stress"
    assert support["dense_deformation_supervision_weight"] == 0.0
    assert np.count_nonzero(
        row["arrays"][
            "truth_section_pullback_stationary_velocity_yx_px_float64"
        ]
    ) == 0


def test_marginal_observation_is_retained_with_zero_loss_weights(prepared_context):
    row = _row(
        prepared_context,
        root_seed=2**63 + 777,
        sample_index=8,
        minimum_brain_pixels=10**6,
        section_id="finite-v5-marginal-section",
    )
    support = row["upstream_reference"]["support_supervision_contract"]

    assert support["continuous_plane_sample_retained"] is True
    assert support["point_pose_supervision_weight"] == 0.0
    assert support["dense_deformation_supervision_weight"] == 0.0
    assert row["upstream_reference"]["deformation_censoring_contract"][
        "status"
    ] == curriculum.MARGINAL_CENTRE_GAUGE_CENSOR_STATUS
    assert np.count_nonzero(
        row["arrays"][
            "truth_section_pullback_stationary_velocity_yx_px_float64"
        ]
    ) == 0


def test_dense_censor_uses_post_g3_weight_mass_not_slab_presence():
    shape = (20, 20)
    occupancy = np.full(shape, 0.5, dtype=np.float32)
    center = np.ones(shape, dtype=bool)

    def paired(weight):
        arrays = {
            "source_slab_brain_occupancy_float32": occupancy.copy(),
            "source_map_domain_mask": np.ones(shape, dtype=bool),
            "source_clean_tissue_mask": center.copy(),
            "source_dense_correspondence_weight_float32": np.full(
                shape, weight, dtype=np.float32
            ),
            "source_dense_correspondence_abstention_mask": np.full(
                shape, weight <= 0.0, dtype=bool
            ),
        }
        return {
            mode: {
                "support_supervision": {
                    "point_pose_evidence_effective_brain_pixel_mass": 200.0
                },
                "arrays": copy.deepcopy(arrays),
            }
            for mode in curriculum.pose_curriculum.MODE_TO_OUTLINE
        }

    censored = curriculum._supervision_evidence(paired(0.0), 120, False)
    supervised = curriculum._supervision_evidence(paired(0.25), 120, False)

    assert censored["point_pose_supervision_weight"] == 1.0
    assert censored["center_gauge_support_identifiable"] is True
    assert censored["post_g3_dense_correspondence_weight_mass"] == 0.0
    assert censored["dense_deformation_supervision_weight"] == 0.0
    assert supervised["post_g3_dense_correspondence_weight_mass"] == 100.0
    assert supervised["dense_deformation_supervision_weight"] == 1.0


def test_batch_cycles_all_brush_modes_and_preserves_animal_grouping(
    prepared_context,
):
    rows = curriculum.make_finite_joint_curriculum_training_rows_v5(
        prepared_context,
        root_seed=2**63 + 811,
        start_index=0,
        row_count=3,
        output_shape_h_w=(39, 43),
        identity_prefix="finite-cycle-v5",
        sections_per_animal=2,
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=120,
        maximum_joint_rejection_attempts=4,
        nominal_cut_thickness_um=55.0,
    )

    assert [row["selected_mode"] for row in rows] == list(
        training_row_v3.TRAINABLE_MODES
    )
    assert rows[0]["lineage"]["animal_id"] == rows[1]["lineage"]["animal_id"]
    assert rows[1]["lineage"]["animal_id"] != rows[2]["lineage"]["animal_id"]
    assert len({row["training_row_id"] for row in rows}) == 3
    assert all(psf_v4.verify_training_row_v4(row) for row in rows)


def test_v4_batch_consumer_receives_exact_row_schedule_and_loss_gates(
    prepared_context,
):
    row = _row(prepared_context)
    support = prepared_context["support_index"]
    converted = batch_v3.training_row_to_tensors_v3(
        row,
        atlas_shape_ap_dv_ml=support["annotation_shape"],
        origin_ap_dv_ml_um=support["origin_um"],
        voxel_size_ap_dv_ml_um=support["voxel_size_um"],
        finite_psf_capability=psf_v4.finite_psf_model_capability_v4(),
    )
    tensors = converted["tensors"]
    supervision = row["upstream_reference"]["support_supervision_contract"]

    assert np.array_equal(
        tensors["axial_offsets_um"].numpy()[0],
        np.asarray(row["finite_psf_contract"]["axial_offsets_um"], np.float32),
    )
    assert np.array_equal(
        tensors["axial_weights"].numpy()[0],
        np.asarray(row["finite_psf_contract"]["axial_weights"], np.float32),
    )
    assert tensors["pose_supervision_weight"].item() == supervision[
        "point_pose_supervision_weight"
    ]
    assert tensors["dense_deformation_supervision_weight"].item() == supervision[
        "dense_deformation_supervision_weight"
    ]
