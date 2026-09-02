import copy

import numpy as np
import pytest

import training.arbitrary_plane_finite_pose_curriculum_v4 as finite_pose
import training.arbitrary_plane_psf_v4 as psf_v4
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


def _one(prepared_context, **changes):
    arguments = {
        "root_seed": 2**63 + 704,
        "sample_index": 5,
        "output_shape_h_w": (31, 35),
        "selected_mode": "smart-brush-imperfect",
        "reflection_state": "horizontal",
        "animal_id": "finite-pose-animal-1",
        "specimen_id": "finite-pose-specimen-1",
        "experiment_id": "finite-pose-experiment-1",
        "synthetic_animal_id": "finite-pose-synthetic-animal-1",
        "section_id": "finite-pose-section-5",
        "split": "development",
        "margin_um": (13.0, 17.0),
        "minimum_brain_pixels": 32,
        "maximum_pose_rejection_attempts": 4,
    }
    arguments.update(changes)
    return finite_pose.make_finite_pose_curriculum_training_row_v4(
        prepared_context, **arguments
    )


@pytest.fixture(scope="module")
def row(prepared_context):
    return _one(prepared_context)


def test_exact_finite_pose_row_replay_and_loss_contract(prepared_context, row):
    replay = finite_pose.replay_finite_pose_curriculum_training_row_v4(
        row, prepared_context
    )
    assert finite_pose.verify_finite_pose_curriculum_training_row_v4(
        row, prepared_context
    )
    assert psf_v4.verify_training_row_v4(
        row, capability=psf_v4.finite_psf_model_capability_v4()
    )
    assert row["schema_version"] == psf_v4.TRAINING_ROW_V4_SCHEMA
    assert row["receipt_sha256"] == replay["receipt_sha256"]
    assert all(
        row["arrays"][name].dtype == replay["arrays"][name].dtype
        and np.array_equal(row["arrays"][name], replay["arrays"][name])
        for name in training_row._ARRAY_KEYS
    )
    psf = row["finite_psf_contract"]
    assert psf["render_mode"] == "finite_boxcar"
    assert psf["axial_sample_count"] == 9
    assert 25.0 <= psf["nominal_cut_thickness_um"] <= 100.0
    assert psf["slab_observation_v4_receipt_sha256"] == row[
        "upstream_reference"
    ]["slab_observation_v4_receipt_sha256"]
    support = row["upstream_reference"]["support_supervision_contract"]
    assert support["point_pose_supervision_evidence_metric"] == (
        "post-G1 sum(source_slab_brain_occupancy_float32)"
    )
    assert support["point_pose_supervision_weight"] == float(
        support["post_g1_point_pose_evidence_effective_brain_pixel_mass"] >= 32
    )
    assert support["dense_deformation_supervision_weight"] == 0.0
    assert not row["arrays"][
        "truth_section_pullback_stationary_velocity_yx_px_float64"
    ].any()
    assert np.all(
        row["arrays"]["target_correspondence_weight_float32"][
            row["arrays"]["target_correspondence_abstention_mask"]
        ]
        == 0.0
    )


def test_seeded_slab_is_attempt_mode_and_reflection_independent(prepared_context):
    common = {
        "root_seed": 2**63 + 710,
        "start_index": 9,
        "row_count": 3,
        "output_shape_h_w": (31, 35),
        "identity_prefix": "finite-pose-cycle",
        "sections_per_animal": 3,
        "margin_um": (13.0, 17.0),
        "minimum_brain_pixels": 32,
        "maximum_pose_rejection_attempts": 4,
    }
    rows = finite_pose.make_finite_pose_curriculum_training_rows_v4(
        prepared_context, **common
    )
    assert [item["selected_mode"] for item in rows] == list(
        training_row.TRAINABLE_MODES
    )
    assert all(item["reflection_state"] == "horizontal" for item in rows)
    for item in rows:
        reference = item["upstream_reference"]["finite_slab_reference"]
        assert reference["thickness_seed_uint64"] == item[
            "numeric_rng_provenance"
        ]["finite_slab_thickness_seed_uint64"]
        assert reference["slab_parent_sample_index"] == item[
            "numeric_rng_provenance"
        ]["derived_plane_sample_index"]
        channels = item["arrays"]["model_input_channels_float32"]
        available = item["selected_mode"] != "smart-brush-absent"
        assert np.all(channels[..., 2] == float(available))
        assert bool(channels[..., 1].any()) is available

    base = _one(
        prepared_context,
        root_seed=2**63 + 712,
        sample_index=13,
        selected_mode="smart-brush-accurate",
        reflection_state="none",
        animal_id="same-animal",
        specimen_id="same-specimen",
        experiment_id="same-experiment",
        synthetic_animal_id="same-synthetic",
        section_id="same-section",
    )
    changed_view = _one(
        prepared_context,
        root_seed=2**63 + 712,
        sample_index=13,
        selected_mode="smart-brush-absent",
        reflection_state="horizontal",
        animal_id="same-animal",
        specimen_id="same-specimen",
        experiment_id="same-experiment",
        synthetic_animal_id="same-synthetic",
        section_id="same-section",
    )
    assert base["upstream_reference"]["finite_slab_reference"] == changed_view[
        "upstream_reference"
    ]["finite_slab_reference"]
    assert base["finite_psf_contract"] == changed_view["finite_psf_contract"]


def test_retry_reuses_exact_parent_and_slab(prepared_context, monkeypatch):
    reason = "imperfect outline did not meet its predeclared IoU gate"
    root_seed = 2**63 + 721
    sample_index = 17
    rejected_seed = finite_pose._derived_seed(
        root_seed, sample_index, "appearance-damage/attempt-0"
    )
    original = finite_pose.make_arbitrary_plane_synthetic_realization
    observed = []

    def reject_first(parent, support, **kwargs):
        slab = kwargs["slab_observation_v4"]
        observed.append(
            (
                parent["finite_render_receipt_sha256"],
                slab["receipt_sha256"],
                slab["finite_psf"]["finite_psf_sha256"],
            )
        )
        if kwargs["root_seed"] == rejected_seed:
            raise ValueError(reason)
        return original(parent, support, **kwargs)

    monkeypatch.setattr(
        finite_pose, "make_arbitrary_plane_synthetic_realization", reject_first
    )
    retried = finite_pose.make_finite_pose_curriculum_training_rows_v4(
        prepared_context,
        root_seed=root_seed,
        start_index=sample_index,
        row_count=1,
        output_shape_h_w=(31, 35),
        identity_prefix="finite-pose-retry",
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=32,
        maximum_pose_rejection_attempts=4,
    )[0]
    history = retried["upstream_reference"]["pose_rejection_history"]
    assert history[0]["reason"] == reason
    assert len(set(observed)) == 1
    assert history[0]["finite_parent_identity"] == retried[
        "upstream_reference"
    ]["finite_parent_identity"]
    assert history[0]["finite_slab_identity"] == retried[
        "upstream_reference"
    ]["adapter_configuration"]["finite_slab_identity"]


def test_marginal_plane_is_retained_and_tamper_is_rejected(prepared_context):
    marginal = _one(
        prepared_context,
        root_seed=2**63 + 731,
        sample_index=23,
        reflection_state="none",
        minimum_brain_pixels=31 * 35 + 1,
    )
    support = marginal["upstream_reference"]["support_supervision_contract"]
    assert support["point_pose_supervision_weight"] == 0.0
    assert support["dense_deformation_supervision_weight"] == 0.0
    assert "retained" in support["marginal_observation_role"]
    assert marginal["upstream_reference"]["pose_rejection_history"] == []
    assert finite_pose.verify_finite_pose_curriculum_training_row_v4(
        marginal, prepared_context
    )

    changed = copy.deepcopy(marginal)
    changed["finite_psf_contract"]["nominal_cut_thickness_um"] += 1.0
    with pytest.raises(ValueError):
        psf_v4.verify_training_row_v4(changed)

    changed = copy.deepcopy(marginal)
    changed["upstream_reference"]["slab_observation_v4_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        psf_v4.verify_training_row_v4(changed)


def test_bounded_exhaustion_uses_fresh_no_drop_fallback(
    prepared_context, monkeypatch
):
    reason = "G3 realization failed all deterministic damage/visibility rejection attempts"
    original = finite_pose.make_arbitrary_plane_synthetic_realization
    observed = []

    def reject_until_gate_bypass(parent, support, **kwargs):
        slab = kwargs["slab_observation_v4"]
        observed.append(
            (
                parent["finite_render_receipt_sha256"],
                slab["receipt_sha256"],
            )
        )
        if "ordinary_minimum_clean_brain_pixels_floor" not in kwargs[
            "config_overrides"
        ]:
            raise ValueError(reason)
        return original(parent, support, **kwargs)

    monkeypatch.setattr(
        finite_pose,
        "make_arbitrary_plane_synthetic_realization",
        reject_until_gate_bypass,
    )
    fallback = finite_pose.make_finite_pose_curriculum_training_rows_v4(
        prepared_context,
        root_seed=2**63 + 741,
        start_index=29,
        row_count=1,
        output_shape_h_w=(31, 35),
        identity_prefix="finite-pose-no-drop",
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=1,
        maximum_pose_rejection_attempts=2,
    )[0]
    assert fallback["upstream_reference"]["no_drop_fallback"] is True
    assert fallback["upstream_reference"]["no_drop_fallback_reason"] == (
        finite_pose.NO_DROP_FALLBACK_REASON
    )
    assert fallback["numeric_rng_provenance"]["pose_attempt_number"] == 2
    assert len(fallback["upstream_reference"]["pose_rejection_history"]) == 2
    assert len(set(observed)) == 1
    assert fallback["upstream_reference"]["support_supervision_contract"][
        "dense_deformation_supervision_weight"
    ] == 0.0


def test_config_and_binding_are_random_only(prepared_context):
    config = finite_pose.finite_pose_curriculum_generation_config_v4(
        prepared_context,
        root_seed=2**63 + 704,
        start_index=0,
        row_count=4,
        output_shape_h_w=(31, 35),
        identity_prefix="finite-pose",
    )
    binding = finite_pose.finite_pose_curriculum_generator_binding_v4(config)
    assert config["schema_version"] == finite_pose.FINITE_POSE_CURRICULUM_V4_SCHEMA
    assert config["finite_psf_render_mode"] == "finite_boxcar"
    assert config["finite_psf_capability"] == psf_v4.finite_psf_model_capability_v4()
    assert "never redraw" in config["pose_acceptance"]
    assert binding["generator_ids"] == [
        finite_pose.FINITE_POSE_CURRICULUM_V4_ALGORITHM
    ]
    assert all(
        value == []
        for value in (
            binding["prior_model_weight_dependencies"],
            binding["prior_feature_dependencies"],
            binding["prior_pseudolabel_dependencies"],
        )
    )
