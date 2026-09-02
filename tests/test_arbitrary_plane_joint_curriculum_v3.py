import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_deformation_gauge_v4 as deformation_gauge
import training.arbitrary_plane_joint_curriculum_v3 as joint_curriculum
import training.arbitrary_plane_pose_curriculum_v3 as pose_curriculum
import training.arbitrary_plane_row_cache_v3 as row_cache
import training.arbitrary_plane_training_row_v3 as training_row
from training.arbitrary_plane_rendered_generator import prepare_finite_render_context
from training.arbitrary_plane_rendered_generator import (
    effective_renderer_sampling_arrays,
)
from training.arbitrary_plane_synthetic_ops import bilinear_sample_scalar
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
    return joint_curriculum.make_joint_curriculum_training_rows_v3(
        prepared_context,
        root_seed=2**63 + 155,
        start_index=0,
        row_count=6,
        output_shape_h_w=(47, 53),
        identity_prefix="joint-v3",
        sections_per_animal=2,
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=320,
    )


@pytest.fixture(scope="module")
def composite_curriculum(prepared_context, six_rows):
    joint_config = joint_curriculum.joint_curriculum_generation_config_v3(
        prepared_context,
        root_seed=2**63 + 155,
        start_index=0,
        row_count=6,
        output_shape_h_w=(47, 53),
        identity_prefix="joint-v3",
        sections_per_animal=2,
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=320,
    )
    pose_config = pose_curriculum.pose_curriculum_generation_config_v3(
        prepared_context,
        root_seed=2**63 + 55,
        start_index=0,
        row_count=1,
        output_shape_h_w=(47, 53),
        identity_prefix="pose-v3",
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=64,
    )
    pose_row = pose_curriculum.make_pose_curriculum_training_rows_v3(
        prepared_context,
        root_seed=2**63 + 55,
        start_index=0,
        row_count=1,
        output_shape_h_w=(47, 53),
        identity_prefix="pose-v3",
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=64,
    )[0]
    config = joint_curriculum.composite_curriculum_generation_config_v3(
        pose_config, joint_config
    )
    return {
        "pose_config": pose_config,
        "joint_config": joint_config,
        "config": config,
        "binding": joint_curriculum.composite_curriculum_generator_binding_v3(
            config
        ),
        "rows": [pose_row, *six_rows],
    }


def _cell_jacobian(map_yx):
    xy = np.asarray(map_yx, dtype=np.float64)[..., ::-1]
    origin = xy[:-1, :-1]
    du = xy[:-1, 1:] - origin
    dv = xy[1:, :-1] - origin
    return du[..., 0] * dv[..., 1] - du[..., 1] * dv[..., 0]


def test_nonzero_affine_free_deformation_positive_jacobian_and_exact_replay(
    prepared_context, six_rows
):
    row = six_rows[0]
    arrays = row["arrays"]
    velocity = arrays[
        "truth_section_pullback_stationary_velocity_yx_px_float64"
    ]
    pullback = arrays["truth_section_pullback_map_yx_px_float64"]
    height, width = velocity.shape[:2]
    y, x = np.mgrid[:height, :width]
    identity = np.stack((y, x), axis=-1).astype(np.float64)

    assert np.any(velocity != 0.0)
    assert not np.array_equal(pullback, identity)
    assert _cell_jacobian(pullback).min() > 0.0
    g1 = row["upstream_reference"]["selected_g1_accepted_attempt"]
    assert g1["identity_path"] is False
    assert g1["topology_metrics"]["forward_jacobian_min"] > 0.0
    assert g1["topology_metrics"]["inverse_jacobian_min"] > 0.0
    gauge = row["upstream_reference"][
        "direct_deformation_target_certification_summary"
    ]
    assert gauge["target_direction"] == deformation_gauge.TARGET_DIRECTION
    assert gauge["integration_steps"] == 7
    assert gauge["diagnostics"]["valid_certification_error_max_px"] == 0.0
    assert (
        gauge["diagnostics"]["uniform_canvas_affine_coefficient_max_abs"]
        < 1e-6
    )
    assert gauge["diagnostics"]["parent_pose_adjustment_max_abs"] == 0.0
    assert row["upstream_reference"]["render_thickness_scope"] == (
        "single centre-plane finite-FOV raster; no through-plane PSF integration"
    )
    assert row["canonical_effective_quicknii_ouv_float64"] == row[
        "upstream_reference"
    ]["effective_quicknii_ouv_ml_ap_dv_before_gauge"]
    assert joint_curriculum.verify_joint_curriculum_training_row_v3(
        row, prepared_context
    )
    assert row_cache.verify_cached_training_row_v3(row)


def test_joint_target_direction_and_ccf_follow_source_to_fixed_pullback(
    prepared_context, monkeypatch
):
    captured = {}
    original = joint_curriculum.make_arbitrary_plane_synthetic_realization

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        captured[result["outline"]["parameters"]["mode"]] = result
        return result

    monkeypatch.setattr(
        joint_curriculum, "make_arbitrary_plane_synthetic_realization", record
    )
    row = joint_curriculum.make_joint_curriculum_training_row_v4(
        prepared_context,
        root_seed=2**63 + 155,
        sample_index=0,
        output_shape_h_w=(47, 53),
        selected_mode="smart-brush-accurate",
        reflection_state="none",
        amplitude_band="mild",
        animal_id="direction-animal",
        specimen_id="direction-specimen",
        experiment_id="direction-experiment",
        synthetic_animal_id="direction-synthetic-animal",
        section_id="direction-section",
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=320,
    )
    selected = captured[
        pose_curriculum.MODE_TO_OUTLINE["smart-brush-accurate"]
    ]
    source = selected["arrays"]
    expected_map = np.moveaxis(source["source_to_fixed_map"], 0, -1)[..., ::-1]
    wrong_map = np.moveaxis(source["fixed_to_source_map"], 0, -1)[..., ::-1]
    expected_velocity = -np.moveaxis(source["velocity_xy_px"], 0, -1)[..., ::-1]
    assert np.array_equal(
        row["arrays"]["truth_section_pullback_map_yx_px_float64"],
        expected_map.astype(np.float64),
    )
    assert not np.array_equal(
        row["arrays"]["truth_section_pullback_map_yx_px_float64"],
        wrong_map.astype(np.float64),
    )
    assert np.array_equal(
        row["arrays"][
            "truth_section_pullback_stationary_velocity_yx_px_float64"
        ],
        expected_velocity.astype(np.float64),
    )

    parent = selected["finite_parent"]
    support = prepared_context["support_index"]
    fixed = effective_renderer_sampling_arrays(
        parent["geometry"],
        tuple(int(value) for value in support["annotation_shape"]),
        origin_ap_dv_ml_um=tuple(support["origin_um"]),
        voxel_size_ap_dv_ml_um=tuple(support["voxel_size_um"]),
    )["coordinate_raster_allen_index_float32"]
    source_allen = np.stack(
        [
            bilinear_sample_scalar(fixed[..., axis], source["source_to_fixed_map"])
            for axis in range(3)
        ],
        axis=-1,
    ).astype(np.float32)
    expected_ccf = (
        np.asarray(support["origin_um"], np.float32)
        + (source_allen + np.float32(0.5))
        * np.asarray(support["voxel_size_um"], np.float32)
    ).astype(np.float32)
    expected_ccf[~source["source_map_domain_mask"]] = 0.0
    assert np.array_equal(
        row["arrays"]["target_ccf_coordinates_ap_dv_ml_um_float64"],
        expected_ccf.astype(np.float64),
    )


def test_modes_reflections_amplitude_bands_and_provenance_cycle(six_rows):
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
    assert [
        row["upstream_reference"]["deformation_amplitude_band"] for row in six_rows
    ] == ["mild", "moderate", "mild", "moderate", "mild", "moderate"]
    for row in six_rows:
        channels = row["arrays"]["model_input_channels_float32"]
        available = row["selected_mode"] != "smart-brush-absent"
        assert np.all(channels[..., 2] == float(available))
        assert row["upstream_reference"]["selected_black_exterior_exact"] == (
            True if available else None
        )
        support_contract = row["upstream_reference"][
            "support_supervision_contract"
        ]
        assert support_contract["continuous_plane_sample_retained"] is True
        assert support_contract["pose_redrawn_for_raster_support"] is False
        has_deformation = bool(
            np.any(
                row["arrays"][
                    "truth_section_pullback_stationary_velocity_yx_px_float64"
                ]
                != 0.0
            )
        )
        assert has_deformation is support_contract[
            "dense_deformation_supervision_identifiable"
        ]
        assert row["upstream_reference"]["g1_nonidentity_forced"] is has_deformation
        assert row["upstream_reference"]["marginal_support_identity_forced"] is (
            not has_deformation
        )
        assert "haar-uniform rp2" in row["upstream_reference"][
            "plane_sampling_measure"
        ]["orientation"].lower()
        assert row["proper_physical_pose_unchanged"] == row[
            "canonical_effective_quicknii_ouv_float64"
        ]
        assert all(
            row[name] == []
            for name in (
                "prior_model_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    assert six_rows[0]["lineage"]["animal_id"] == six_rows[1]["lineage"]["animal_id"]
    assert six_rows[0]["lineage"]["section_id"] != six_rows[1]["lineage"]["section_id"]
    assert six_rows[1]["lineage"]["animal_id"] != six_rows[2]["lineage"]["animal_id"]


def test_standard_i_drive_row_cache_accepts_composite_pose_and_joint_rows(
    composite_curriculum,
):
    config = composite_curriculum["config"]
    binding = composite_curriculum["binding"]
    rows = composite_curriculum["rows"]
    base = Path("I:/AnatomyTracker/tmp")
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="joint-row-cache-", dir=base) as directory:
        row_cache.initialize_training_row_cache_v3(
            directory,
            generator_binding=binding,
            generation_config=config,
            seed_record={
                "pose_root_seed_uint64": "0x8000000000000037",
                "joint_root_seed_uint64": "0x800000000000009b",
            },
        )
        manifest = row_cache.append_training_rows_v3(directory, rows)
        frozen = row_cache.freeze_training_row_cache_v3(directory)
        audit = row_cache.audit_training_row_cache_v3(directory)
        loaded = row_cache.load_training_rows_v3(directory)

    assert manifest["row_count"] == 7
    assert frozen["row_count"] == audit["row_count"] == 7
    assert [row["training_row_id"] for row in loaded] == [
        row["training_row_id"] for row in rows
    ]
    assert [record["composite_component"] for record in manifest["rows"]] == [
        "identity_pose_curriculum",
        *(["nonidentity_joint_curriculum"] * 6),
    ]
    assert set(binding["generator_ids"]) == {
        pose_curriculum.POSE_CURRICULUM_V3_ALGORITHM,
        joint_curriculum.JOINT_CURRICULUM_V3_ALGORITHM,
    }
    assert set(binding["source_sha256"]) >= set(
        pose_curriculum.pose_curriculum_generator_binding_v3(
            composite_curriculum["pose_config"]
        )["source_sha256"]
    )
    assert config["component_row_counts"] == {
        "identity_pose_curriculum": 1,
        "nonidentity_joint_curriculum": 6,
    }
    assert binding["prior_model_weight_dependencies"] == []
    assert binding["prior_feature_dependencies"] == []
    assert binding["prior_pseudolabel_dependencies"] == []
    assert composite_curriculum["joint_config"][
        "finite_parent_generator_source_commit"
    ] is None
    assert composite_curriculum["joint_config"]["required_runner_psf"] == composite_curriculum["pose_config"][
        "required_runner_psf"
    ] == {
        "axial_offsets_um": [0.0],
        "axial_weights": [1.0],
        "interpretation": "single centre-plane sample matching the direct curriculum raster",
    }


def _new_composite_cache(composite_curriculum, prefix):
    base = Path("I:/AnatomyTracker/tmp")
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
    row_cache.initialize_training_row_cache_v3(
        directory,
        generator_binding=composite_curriculum["binding"],
        generation_config=composite_curriculum["config"],
        seed_record={
            "pose_root_seed_uint64": "0x8000000000000037",
            "joint_root_seed_uint64": "0x800000000000009b",
        },
    )
    return directory


def _reauthenticated_row(row):
    row["receipt_sha256"] = acquisition._payload_sha256(
        training_row.training_row_receipt_v3(row)
    )
    return row


@pytest.mark.parametrize("tamper", ["algorithm", "source", "config"])
def test_composite_cache_rejects_wrong_algorithm_source_and_config(
    composite_curriculum, tamper
):
    changed = copy.deepcopy(composite_curriculum["rows"][0])
    if tamper == "algorithm":
        changed["upstream_reference"]["algorithm"] = (
            joint_curriculum.JOINT_CURRICULUM_V3_ALGORITHM
        )
    elif tamper == "source":
        source_name = next(
            iter(changed["upstream_reference"]["implementation_source_sha256"])
        )
        changed["upstream_reference"]["implementation_source_sha256"][
            source_name
        ] = "0" * 64
    else:
        changed["upstream_reference"]["adapter_configuration"][
            "minimum_brain_pixels"
        ] += 1
    changed = _reauthenticated_row(changed)
    directory = _new_composite_cache(
        composite_curriculum, f"joint-row-cache-{tamper}-"
    )
    with pytest.raises(ValueError, match="composite row"):
        row_cache.append_training_rows_v3(directory, [changed])


def test_composite_cache_rejects_wrong_order_and_incomplete_counts(
    composite_curriculum,
):
    wrong_order = _new_composite_cache(composite_curriculum, "joint-wrong-order-")
    with pytest.raises(ValueError, match="composite row"):
        row_cache.append_training_rows_v3(
            wrong_order, [composite_curriculum["rows"][1]]
        )

    incomplete = _new_composite_cache(composite_curriculum, "joint-incomplete-")
    row_cache.append_training_rows_v3(
        incomplete, composite_curriculum["rows"][:2]
    )
    with pytest.raises(ValueError, match="component counts"):
        row_cache.audit_training_row_cache_v3(incomplete)
    with pytest.raises(ValueError, match="component counts"):
        row_cache.freeze_training_row_cache_v3(incomplete)


def test_composite_cache_rejects_generation_config_binding_mismatch(
    composite_curriculum,
):
    config = copy.deepcopy(composite_curriculum["config"])
    config["component_generation_configs"]["identity_pose_curriculum"][
        "minimum_brain_pixels"
    ] += 1
    base = Path("I:/AnatomyTracker/tmp")
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="joint-config-mismatch-", dir=base) as directory:
        with pytest.raises(ValueError, match="differs from its binding"):
            row_cache.initialize_training_row_cache_v3(
                directory,
                generator_binding=composite_curriculum["binding"],
                generation_config=config,
                seed_record={"root_seed_uint64": "0x1"},
            )


def test_composite_cache_rejects_exact_parent_source_commit_mismatch(
    composite_curriculum,
):
    changed = copy.deepcopy(composite_curriculum["rows"][0])
    changed["upstream_reference"]["adapter_configuration"][
        "finite_parent_generator_source_commit"
    ] = "wrong-source-commit"
    changed = _reauthenticated_row(changed)
    directory = _new_composite_cache(composite_curriculum, "joint-parent-source-")
    with pytest.raises(ValueError, match="adapter differs"):
        row_cache.append_training_rows_v3(directory, [changed])


def test_composite_streaming_audit_rejects_fully_reauthenticated_row_tamper(
    composite_curriculum,
):
    directory = _new_composite_cache(composite_curriculum, "joint-audit-tamper-")
    manifest = row_cache.append_training_rows_v3(
        directory, composite_curriculum["rows"]
    )
    record = manifest["rows"][0]
    metadata_path = directory / record["metadata_relative_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["upstream_reference"]["adapter_configuration"][
        "minimum_brain_pixels"
    ] += 1
    metadata["receipt_sha256"] = acquisition._payload_sha256(
        training_row.training_row_receipt_v3(metadata)
    )
    row_cache._atomic_json(metadata_path, metadata)
    payload = row_cache._manifest_payload(manifest)
    payload["rows"][0]["training_row_receipt_sha256"] = metadata[
        "receipt_sha256"
    ]
    payload["rows"][0]["metadata_file_sha256"] = row_cache._file_sha256(
        metadata_path
    )
    row_cache._atomic_json(
        directory / "manifest.json", row_cache._with_manifest_receipt(payload)
    )
    with pytest.raises(ValueError, match="adapter differs"):
        row_cache.audit_training_row_cache_v3(directory)


def test_tamper_rejection_and_exact_authenticated_retry_replay(
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
        joint_curriculum.verify_joint_curriculum_training_row_v3(
            changed_source, prepared_context
        )

    original = joint_curriculum.make_joint_curriculum_training_row_v3

    def one_forced_rejection(context, **kwargs):
        if kwargs["joint_attempt_number"] == 0:
            raise ValueError(joint_curriculum.GAUGE_RECOMPOSITION_REJECTION)
        return original(context, **kwargs)

    monkeypatch.setattr(
        joint_curriculum,
        "make_joint_curriculum_training_row_v3",
        one_forced_rejection,
    )
    retried = joint_curriculum.make_joint_curriculum_training_rows_v3(
        prepared_context,
        root_seed=2**63 + 191,
        start_index=300,
        row_count=1,
        output_shape_h_w=(47, 53),
        identity_prefix="joint-retry-v3",
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=320,
        maximum_joint_rejection_attempts=8,
    )[0]
    attempt = retried["numeric_rng_provenance"]["joint_attempt_number"]
    history = retried["upstream_reference"]["joint_rejection_history"]
    assert attempt >= 1
    assert len(history) == attempt
    assert history[0]["stage"] == "deformation-gauge"
    assert history[0]["reason"] == joint_curriculum.GAUGE_RECOMPOSITION_REJECTION
    assert history[0]["derived_plane_sample_index"] == (
        joint_curriculum.joint_attempt_index_v3(2**63 + 191, 300, 0)
    )
    assert retried["numeric_rng_provenance"]["finite_render_seed_uint64"] == history[
        0
    ]["finite_render_seed_uint64"]
    assert retried["numeric_rng_provenance"]["derived_plane_sample_index"] == history[
        0
    ]["derived_plane_sample_index"]
    assert retried["lineage"]["section_id"] == "joint-retry-v3-section-00000300"
    assert joint_curriculum.verify_joint_curriculum_training_row_v3(
        retried, prepared_context
    )


def test_chunked_joint_generation_uses_global_logical_index_cycle(
    prepared_context, six_rows
):
    chunk = joint_curriculum.make_joint_curriculum_training_rows_v3(
        prepared_context,
        root_seed=2**63 + 155,
        start_index=4,
        row_count=2,
        output_shape_h_w=(47, 53),
        identity_prefix="joint-v3",
        sections_per_animal=2,
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=320,
    )
    assert [row["selected_mode"] for row in chunk] == [
        training_row.TRAINABLE_MODES[1],
        training_row.TRAINABLE_MODES[2],
    ]
    assert [row["reflection_state"] for row in chunk] == [
        "horizontal",
        "horizontal",
    ]
    assert [
        row["upstream_reference"]["deformation_amplitude_band"] for row in chunk
    ] == ["mild", "moderate"]
    assert [row["receipt_sha256"] for row in chunk] == [
        row["receipt_sha256"] for row in six_rows[4:]
    ]


def test_v4_no_drop_fallback_preserves_exact_parent_and_never_relabels_failure(
    prepared_context, composite_curriculum, monkeypatch
):
    original = joint_curriculum.make_arbitrary_plane_synthetic_realization
    parent_observations = []

    def reject_nonidentity(parent, *args, **kwargs):
        parent_observations.append(
            {
                "object_id": id(parent),
                "finite_plane_render_id": parent["finite_plane_render_id"],
                "finite_render_receipt_sha256": parent[
                    "finite_render_receipt_sha256"
                ],
                "identity_probability": kwargs["config_overrides"]["g1"][
                    "identity_probability"
                ],
            }
        )
        if kwargs["config_overrides"]["g1"]["identity_probability"] == 0.0:
            raise ValueError(
                "no G1 realization passed every predeclared topology, cycle, displacement, and FOV gate"
            )
        return original(parent, *args, **kwargs)

    monkeypatch.setattr(
        joint_curriculum,
        "make_arbitrary_plane_synthetic_realization",
        reject_nonidentity,
    )
    rows = joint_curriculum.make_joint_curriculum_training_rows_v4(
        prepared_context,
        root_seed=2**63 + 919,
        start_index=0,
        row_count=2,
        output_shape_h_w=(47, 53),
        identity_prefix="joint-no-drop-v4",
        sections_per_animal=2,
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=320,
        maximum_joint_rejection_attempts=2,
    )
    batch_parent_observations = copy.deepcopy(parent_observations)

    assert len(rows) == 2
    assert [row["lineage"]["section_id"] for row in rows] == [
        "joint-no-drop-v4-section-00000000",
        "joint-no-drop-v4-section-00000001",
    ]
    config = joint_curriculum.joint_curriculum_generation_config_v4(
        prepared_context,
        root_seed=2**63 + 919,
        start_index=0,
        row_count=2,
        output_shape_h_w=(47, 53),
        identity_prefix="joint-no-drop-v4",
        sections_per_animal=2,
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=320,
        maximum_joint_rejection_attempts=2,
    )
    assert config["joint_no_drop_policy"] == joint_curriculum.JOINT_NO_DROP_POLICY
    assert config["fallback_attempt_index"] == 2
    assert config["rejected_nonidentity_image_relabeling_allowed"] is False
    assert config["deformation_censor_statuses"]["bounded_retry_fallback"] == (
        joint_curriculum.IDENTITY_FALLBACK_CENSOR_STATUS
    )
    composite_config = joint_curriculum.composite_curriculum_generation_config_v3(
        composite_curriculum["pose_config"], config
    )
    composite_binding = (
        joint_curriculum.composite_curriculum_generator_binding_v3(
            composite_config
        )
    )
    base = Path("I:/AnatomyTracker/tmp")
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="joint-no-drop-cache-", dir=base
    ) as directory:
        row_cache.initialize_training_row_cache_v3(
            directory,
            generator_binding=composite_binding,
            generation_config=composite_config,
            seed_record={
                "pose_root_seed_uint64": "0x8000000000000037",
                "joint_root_seed_uint64": "0x8000000000000397",
            },
        )
        manifest = row_cache.append_training_rows_v3(
            directory,
            [composite_curriculum["rows"][0], *rows],
        )
    assert [item["composite_component"] for item in manifest["rows"]] == [
        "identity_pose_curriculum",
        "nonidentity_joint_curriculum",
        "nonidentity_joint_curriculum",
    ]
    changed_dense = copy.deepcopy(rows[0])
    changed_effective = changed_dense["upstream_reference"][
        "adapter_configuration"
    ]["effective_dense_support"]
    changed_effective["effective_dense_deformation_supervision_identifiable"] = True
    changed_effective["effective_dense_deformation_supervision_weight"] = 1.0
    changed_dense["upstream_reference"]["deformation_censoring_contract"][
        "effective_dense_support"
    ] = copy.deepcopy(changed_effective)
    changed_dense["upstream_reference"]["support_supervision_contract"][
        "effective_dense_support"
    ] = copy.deepcopy(changed_effective)
    changed_dense["upstream_reference"]["support_supervision_contract"][
        "dense_deformation_supervision_identifiable"
    ] = True
    changed_dense["upstream_reference"]["support_supervision_contract"][
        "dense_deformation_supervision_weight"
    ] = 1.0
    changed_dense = _reauthenticated_row(changed_dense)
    with tempfile.TemporaryDirectory(
        prefix="joint-no-drop-tamper-cache-", dir=base
    ) as directory:
        row_cache.initialize_training_row_cache_v3(
            directory,
            generator_binding=composite_binding,
            generation_config=composite_config,
            seed_record={
                "pose_root_seed_uint64": "0x8000000000000037",
                "joint_root_seed_uint64": "0x8000000000000397",
            },
        )
        with pytest.raises(ValueError, match="supervision weights"):
            row_cache.append_training_rows_v3(
                directory,
                [composite_curriculum["rows"][0], changed_dense],
            )
    for row in rows:
        upstream = row["upstream_reference"]
        adapter = upstream["adapter_configuration"]
        numeric = row["numeric_rng_provenance"]
        support = upstream["support_supervision_contract"]
        censor = upstream["deformation_censoring_contract"]
        parent_identity = upstream["finite_parent_identity"]
        history = upstream["joint_rejection_history"]
        height, width = row["arrays"][
            "truth_section_pullback_map_yx_px_float64"
        ].shape[:2]
        y, x = np.mgrid[:height, :width]
        identity = np.stack((y, x), axis=-1).astype(np.float64)

        assert len(history) == 2
        assert all(
            item["finite_parent_identity"] == parent_identity
            and item["finite_parent_request"] == {
                key: parent_identity[key]
                for key in (
                    "logical_root_seed_uint64",
                    "logical_sample_index",
                    "derived_plane_sample_index",
                    "finite_render_seed_uint64",
                    "lineage_ids",
                )
            }
            and item["requested_deformation_amplitude_band"]
            == adapter["amplitude_band"]
            for item in history
        )
        assert adapter["identity_g1_pose_only_fallback"] is True
        assert adapter["joint_attempt_number"] == 2
        assert adapter["fallback_attempt_number"] == 2
        assert adapter["fallback_synthetic_seed_uint64"] == numeric[
            "synthetic_seed_uint64"
        ]
        assert numeric["synthetic_seed_uint64"] not in {
            item["synthetic_seed_uint64"] for item in history
        }
        assert censor["status"] == (
            joint_curriculum.IDENTITY_FALLBACK_CENSOR_STATUS
        )
        assert censor["reason"] == (
            joint_curriculum.NONIDENTITY_RETRY_EXHAUSTION_CENSOR_REASON
        )
        assert censor["fresh_identity_g1_realization"] is True
        assert censor["rejected_nonidentity_image_relabeling_allowed"] is False
        assert support["point_pose_supervision_weight"] == 1.0
        assert support["dense_deformation_supervision_weight"] == 0.0
        assert np.count_nonzero(
            row["arrays"][
                "truth_section_pullback_stationary_velocity_yx_px_float64"
            ]
        ) == 0
        assert np.array_equal(
            row["arrays"]["truth_section_pullback_map_yx_px_float64"],
            identity,
        )
        assert row["canonical_effective_quicknii_ouv_float64"] == upstream[
            "effective_quicknii_ouv_ml_ap_dv_before_gauge"
        ]
        assert joint_curriculum.verify_joint_curriculum_training_row_v4(
            row, prepared_context
        )

    for row in rows:
        parent_id = row["upstream_reference"]["finite_plane_render_id"]
        observations = [
            item for item in batch_parent_observations
            if item["finite_plane_render_id"] == parent_id
        ]
        assert len(observations) >= 5
        assert len({item["object_id"] for item in observations}) == 1
        assert len({item["finite_render_receipt_sha256"] for item in observations}) == 1
        assert [item["identity_probability"] for item in observations[:2]] == [
            0.0,
            0.0,
        ]
        assert all(
            item["identity_probability"] == 1.0 for item in observations[2:]
        )

    tampered = copy.deepcopy(rows[0])
    tampered["upstream_reference"]["joint_rejection_history"][0][
        "finite_parent_identity"
    ]["finite_plane_render_id"] = "0" * 64
    tampered = _reauthenticated_row(tampered)
    with pytest.raises(ValueError, match="rejection history|finite parent"):
        joint_curriculum.verify_joint_curriculum_training_row_v4(
            tampered, prepared_context
        )

    tampered = copy.deepcopy(rows[0])
    tampered["upstream_reference"]["adapter_configuration"][
        "deformation_censor_status"
    ] = "tampered-censor-status"
    tampered["upstream_reference"]["deformation_censoring_contract"][
        "status"
    ] = "tampered-censor-status"
    tampered = _reauthenticated_row(tampered)
    with pytest.raises(ValueError, match="censor status"):
        joint_curriculum.verify_joint_curriculum_training_row_v4(
            tampered, prepared_context
        )


def test_v4_direct_success_remains_bit_exact_and_one_over_one(
    prepared_context, six_rows
):
    direct = joint_curriculum.make_joint_curriculum_training_row_v4(
        prepared_context,
        root_seed=2**63 + 155,
        sample_index=0,
        output_shape_h_w=(47, 53),
        selected_mode=training_row.TRAINABLE_MODES[0],
        reflection_state=training_row.REFLECTION_STATES[0],
        amplitude_band="mild",
        animal_id="joint-v3-animal-00000000",
        specimen_id="joint-v3-specimen-00000000",
        experiment_id="joint-v3-experiment-00000000",
        synthetic_animal_id="joint-v3-synthetic-animal-00000000",
        section_id="joint-v3-section-00000000",
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=320,
    )
    cached = six_rows[0]
    assert all(
        np.array_equal(direct["arrays"][name], cached["arrays"][name])
        for name in training_row._ARRAY_KEYS
    )
    support = direct["upstream_reference"]["support_supervision_contract"]
    censor = direct["upstream_reference"]["deformation_censoring_contract"]
    assert support["point_pose_supervision_weight"] == 1.0
    assert support["dense_deformation_supervision_weight"] == 1.0
    assert censor["status"] == joint_curriculum.UNCENSORED_DEFORMATION_STATUS
    assert censor["reason"] is None
    assert censor["identity_g1_pose_only_fallback"] is False


def test_v4_marginal_support_remains_zero_over_zero(prepared_context):
    row = joint_curriculum.make_joint_curriculum_training_row_v4(
        prepared_context,
        root_seed=2**63 + 921,
        sample_index=7,
        output_shape_h_w=(47, 53),
        selected_mode="smart-brush-accurate",
        reflection_state="none",
        amplitude_band="moderate",
        animal_id="marginal-animal",
        specimen_id="marginal-specimen",
        experiment_id="marginal-experiment",
        synthetic_animal_id="marginal-synthetic-animal",
        section_id="marginal-section",
        margin_um=(13.0, 17.0),
        minimum_brain_pixels=47 * 53 + 1,
    )
    support = row["upstream_reference"]["support_supervision_contract"]
    censor = row["upstream_reference"]["deformation_censoring_contract"]
    assert support["point_pose_supervision_weight"] == 0.0
    assert support["dense_deformation_supervision_weight"] == 0.0
    assert support["effective_dense_support"][
        "effective_dense_deformation_supervision_weight"
    ] == 0.0
    assert censor["status"] == joint_curriculum.MARGINAL_SUPPORT_CENSOR_STATUS
    assert censor["identity_g1_pose_only_fallback"] is False
    assert joint_curriculum.verify_joint_curriculum_training_row_v4(
        row, prepared_context
    )
