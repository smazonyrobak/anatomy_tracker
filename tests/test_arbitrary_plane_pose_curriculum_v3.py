import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_pose_curriculum_v3 as pose_curriculum
import training.arbitrary_plane_row_cache_v3 as row_cache
import training.arbitrary_plane_training_row_v3 as training_row
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
        atlas_version="fixture-v1",
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
        template_decoder="pynrrd 1.1.3",
        template_index_order="F",
        annotation_decoder="pynrrd 1.1.3",
        annotation_index_order="F",
    )


@pytest.fixture(scope="module")
def six_rows(prepared_context):
    return pose_curriculum.make_pose_curriculum_training_rows_v3(
        prepared_context,
        root_seed=2**63 + 55,
        start_index=0,
        row_count=6,
        output_shape_h_w=(47, 53),
        identity_prefix="pose-v3",
        sections_per_animal=2,
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=64,
    )


def _identity_yx(height, width):
    y, x = np.mgrid[:height, :width]
    return np.stack((y, x), axis=-1).astype(np.float64)


def test_exact_replay_pose_gauge_and_complete_plane_measure(prepared_context, six_rows):
    row = six_rows[0]
    replay = pose_curriculum.replay_pose_curriculum_training_row_v3(
        row, prepared_context
    )
    assert pose_curriculum.verify_pose_curriculum_training_row_v3(
        row, prepared_context
    )
    assert row_cache.verify_cached_training_row_v3(row)
    assert row["receipt_sha256"] == replay["receipt_sha256"]
    assert all(
        np.array_equal(row["arrays"][name], replay["arrays"][name])
        and row["arrays"][name].dtype == replay["arrays"][name].dtype
        for name in training_row._ARRAY_KEYS
    )

    height, width = row["arrays"]["source_tissue_ground_truth_mask"].shape
    assert np.array_equal(
        row["arrays"]["truth_section_pullback_map_yx_px_float64"],
        _identity_yx(height, width),
    )
    assert not row["arrays"][
        "truth_section_pullback_stationary_velocity_yx_px_float64"
    ].any()
    assert row["upstream_reference"]["g1_identity_forced"] is True
    assert row["upstream_reference"]["effective_pose_source_key"] == (
        "parent['geometry']['effective_quicknii_ouv_ml_ap_dv']"
    )
    assert np.array_equal(
        row["canonical_effective_quicknii_ouv_float64"],
        row["upstream_reference"]["effective_quicknii_ouv_ml_ap_dv"],
    )
    assert row["deformation_pose_gauge_reference"]["projection_weighting"] == (
        row_cache.DEFORMATION_GAUGE_PROJECTION_WEIGHTING
    )
    measure = row["upstream_reference"]["plane_sampling_measure"]
    assert "haar-uniform rp2" in measure["orientation"].lower()
    assert "length-uniform" in measure["reference_offset"].lower()
    support_contract = row["upstream_reference"]["support_supervision_contract"]
    assert support_contract["continuous_plane_sample_retained"] is True
    assert support_contract["pose_redrawn_for_raster_support"] is False
    assert support_contract["raster_brain_pixel_count"] == row[
        "upstream_reference"
    ]["brain_pixel_count"]
    assert support_contract["point_pose_supervision_identifiable"] is (
        row["upstream_reference"]["brain_pixel_count"] >= 64
    )
    assert "unconditioned" in measure["conditioning"]
    assert row["upstream_reference"]["render_thickness_scope"] == (
        pose_curriculum.SINGLE_PLANE_RENDER_SCOPE
    )
    provenance = row["upstream_reference"]["finite_parent_provenance"]
    assert provenance["animal_id"] == row["lineage"]["animal_id"]
    assert provenance["specimen_id"] == row["lineage"]["specimen_id"]
    assert provenance["experiment_id"] == row["lineage"]["experiment_id"]
    assert row["upstream_reference"]["finite_parent_generator_binding"][
        "resolved_config"
    ]["root_seed"] == row["numeric_rng_provenance"]["finite_render_seed_uint64"]
    assert row["upstream_reference"]["selected_synthetic_generator_binding"][
        "resolved_config"
    ]["root_seed"] == row["numeric_rng_provenance"][
        "appearance_damage_seed_uint64"
    ]


def test_modes_backgrounds_reflection_and_lineage_are_separate(six_rows):
    assert [row["selected_mode"] for row in six_rows] == [
        *training_row.TRAINABLE_MODES,
        *training_row.TRAINABLE_MODES,
    ]
    assert [row["reflection_state"] for row in six_rows] == [
        "none",
        "none",
        "none",
        "horizontal",
        "horizontal",
        "horizontal",
    ]
    for row in six_rows:
        channels = row["arrays"]["model_input_channels_float32"]
        available = row["selected_mode"] != "smart-brush-absent"
        assert np.all(channels[..., 2] == float(available))
        assert bool(channels[..., 1].any()) is available
        assert row["upstream_reference"]["selected_black_exterior_exact"] == (
            True if available else None
        )
        assert row["proper_physical_pose_unchanged"] == row[
            "canonical_effective_quicknii_ouv_float64"
        ]
        if row["reflection_state"] == "horizontal":
            assert row["observed_effective_quicknii_ouv_float64"] != row[
                "canonical_effective_quicknii_ouv_float64"
            ]

    assert six_rows[0]["lineage"]["animal_id"] == six_rows[1]["lineage"]["animal_id"]
    assert six_rows[0]["lineage"]["section_id"] != six_rows[1]["lineage"]["section_id"]
    assert six_rows[1]["lineage"]["animal_id"] != six_rows[2]["lineage"]["animal_id"]
    assert len(
        {
            row["upstream_reference"]["selected_stage_realization_ids"]["g2"]
            for row in six_rows
        }
    ) == len(six_rows)
    assert all(
        row[name] == []
        for row in six_rows
        for name in (
            "prior_model_dependencies",
            "prior_feature_dependencies",
            "prior_pseudolabel_dependencies",
        )
    )


def test_standard_i_drive_row_cache_accepts_exact_generator_binding(
    prepared_context, six_rows
):
    finite_parent_source_commit = six_rows[0]["upstream_reference"][
        "finite_parent_generator_binding"
    ]["implementation"]["source_commit"]
    config = pose_curriculum.pose_curriculum_generation_config_v3(
        prepared_context,
        root_seed=2**63 + 55,
        start_index=0,
        row_count=6,
        output_shape_h_w=(47, 53),
        identity_prefix="pose-v3",
        sections_per_animal=2,
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=64,
        finite_parent_generator_source_commit=finite_parent_source_commit,
    )
    binding = pose_curriculum.pose_curriculum_generator_binding_v3(config)
    base = Path("I:/AnatomyTracker/tmp")
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pose-row-cache-", dir=base) as directory:
        row_cache.initialize_training_row_cache_v3(
            directory,
            generator_binding=binding,
            generation_config=config,
            seed_record={"root_seed_uint64": "0x8000000000000037"},
        )
        manifest = row_cache.append_training_rows_v3(directory, six_rows[:2])
        loaded = row_cache.load_training_rows_v3(directory)

    assert manifest["row_count"] == 2
    assert [row["training_row_id"] for row in loaded] == [
        row["training_row_id"] for row in six_rows[:2]
    ]
    assert binding["prior_model_weight_dependencies"] == []
    assert binding["prior_feature_dependencies"] == []
    assert binding["prior_pseudolabel_dependencies"] == []
    assert config["finite_parent_generator_source_commit"] == finite_parent_source_commit
    assert config["required_runner_psf"] == {
        "axial_offsets_um": [0.0],
        "axial_weights": [1.0],
        "interpretation": "single centre-plane sample matching the direct curriculum raster",
    }
    runner_config = pose_curriculum.single_plane_curriculum_runner_config_v3(
        {
            "target_applied_steps": 5,
            "axial_offsets_um": [-25.0, 0.0, 25.0],
            "axial_weights": [0.25, 0.5, 0.25],
        }
    )
    assert runner_config == {
        "target_applied_steps": 5,
        "axial_offsets_um": [0.0],
        "axial_weights": [1.0],
    }


def test_tamper_rejection_and_recorded_deterministic_parent_retry(
    prepared_context, six_rows, monkeypatch
):
    changed_array = copy.deepcopy(six_rows[0])
    changed_array["arrays"]["model_input_channels_float32"][0, 0, 0] += 0.25
    with pytest.raises(ValueError, match="unauthenticated"):
        row_cache.verify_cached_training_row_v3(changed_array)

    changed_source = copy.deepcopy(six_rows[0])
    source_name = next(
        iter(changed_source["upstream_reference"]["implementation_source_sha256"])
    )
    changed_source["upstream_reference"]["implementation_source_sha256"][source_name] = (
        "0" * 64
    )
    changed_source["receipt_sha256"] = acquisition._payload_sha256(
        training_row.training_row_receipt_v3(changed_source)
    )
    assert row_cache.verify_cached_training_row_v3(changed_source)
    with pytest.raises(ValueError, match="source binding changed"):
        pose_curriculum.verify_pose_curriculum_training_row_v3(
            changed_source, prepared_context
        )

    original = pose_curriculum.make_pose_curriculum_training_row_v3

    def one_forced_rejection(context, **kwargs):
        if kwargs["plane_attempt_number"] == 0:
            raise ValueError(pose_curriculum.PARENT_GEOMETRY_REJECTION)
        return original(context, **kwargs)

    monkeypatch.setattr(
        pose_curriculum,
        "make_pose_curriculum_training_row_v3",
        one_forced_rejection,
    )
    retried = pose_curriculum.make_pose_curriculum_training_rows_v3(
        prepared_context,
        root_seed=2**63 + 91,
        start_index=300,
        row_count=1,
        output_shape_h_w=(47, 53),
        identity_prefix="retry-v3",
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=64,
        maximum_parent_geometry_retries=8,
    )[0]
    attempt = retried["numeric_rng_provenance"]["plane_attempt_number"]
    history = retried["upstream_reference"]["plane_parent_rejection_history"]
    assert attempt >= 1
    assert len(history) == attempt
    assert history[0] == {
        "attempt_index": 0,
        "derived_plane_sample_index": pose_curriculum.plane_attempt_index_v3(
            2**63 + 91, 300, 0
        ),
        "finite_render_seed_uint64": (
            f"0x{pose_curriculum._derived_seed(2**63 + 91, 300, 'finite-render/plane-attempt-0'):016x}"
        ),
        "appearance_damage_seed_uint64": (
            f"0x{pose_curriculum._derived_seed(2**63 + 91, 300, 'appearance-damage/plane-attempt-0'):016x}"
        ),
        "error_type": "ValueError",
        "stage": "finite-parent-verification",
        "reason": pose_curriculum.PARENT_GEOMETRY_REJECTION,
    }
    assert retried["lineage"]["section_id"] == "retry-v3-section-00000300"
    assert row_cache.verify_cached_training_row_v3(retried)
    assert pose_curriculum.verify_pose_curriculum_training_row_v3(
        retried, prepared_context
    )
    assert retried["numeric_rng_provenance"]["finite_render_seed_uint64"] == (
        history[0]["finite_render_seed_uint64"]
    )
    assert retried["numeric_rng_provenance"]["derived_plane_sample_index"] == (
        history[0]["derived_plane_sample_index"]
    )

    outline_reason = "imperfect outline did not meet its predeclared IoU gate"
    first_synthetic_seed = pose_curriculum._derived_seed(
        2**63 + 93, 301, "appearance-damage/plane-attempt-0"
    )
    original_synthetic = pose_curriculum.make_arbitrary_plane_synthetic_realization
    parent_plane_ids = []

    def one_forced_outline_rejection(parent, support, **kwargs):
        parent_plane_ids.append(parent["plane_realization_id"])
        if kwargs["root_seed"] == first_synthetic_seed:
            raise ValueError(outline_reason)
        return original_synthetic(parent, support, **kwargs)

    monkeypatch.setattr(
        pose_curriculum,
        "make_arbitrary_plane_synthetic_realization",
        one_forced_outline_rejection,
    )
    monkeypatch.setattr(
        pose_curriculum,
        "make_pose_curriculum_training_row_v3",
        original,
    )
    synthetic_retry = pose_curriculum.make_pose_curriculum_training_rows_v3(
        prepared_context,
        root_seed=2**63 + 93,
        start_index=301,
        row_count=1,
        output_shape_h_w=(47, 53),
        identity_prefix="retry-outline-v3",
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=64,
        maximum_parent_geometry_retries=8,
    )[0]
    assert synthetic_retry["upstream_reference"]["plane_parent_rejection_history"][
        0
    ]["stage"] == "synthetic-outline"
    assert pose_curriculum.verify_pose_curriculum_training_row_v3(
        synthetic_retry, prepared_context
    )
    assert len(parent_plane_ids) >= 5
    assert len(set(parent_plane_ids)) == 1

    changed_history = copy.deepcopy(synthetic_retry)
    changed_history["upstream_reference"]["adapter_configuration"][
        "plane_parent_rejection_history"
    ][0]["stage"] = "synthetic-g3"
    changed_history["upstream_reference"]["plane_parent_rejection_history"][0][
        "stage"
    ] = "synthetic-g3"
    changed_history["receipt_sha256"] = acquisition._payload_sha256(
        training_row.training_row_receipt_v3(changed_history)
    )
    assert row_cache.verify_cached_training_row_v3(changed_history)
    with pytest.raises(ValueError, match="history is not canonical"):
        pose_curriculum.verify_pose_curriculum_training_row_v3(
            changed_history, prepared_context
        )


def test_chunked_generation_uses_global_logical_index_cycle(
    prepared_context, six_rows
):
    chunk = pose_curriculum.make_pose_curriculum_training_rows_v3(
        prepared_context,
        root_seed=2**63 + 55,
        start_index=4,
        row_count=2,
        output_shape_h_w=(47, 53),
        identity_prefix="pose-v3",
        sections_per_animal=2,
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=64,
    )
    assert [row["selected_mode"] for row in chunk] == [
        training_row.TRAINABLE_MODES[1],
        training_row.TRAINABLE_MODES[2],
    ]
    assert [row["reflection_state"] for row in chunk] == [
        "horizontal",
        "horizontal",
    ]
    assert [row["receipt_sha256"] for row in chunk] == [
        row["receipt_sha256"] for row in six_rows[4:]
    ]
