import copy

import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_deformation_gauge_v4 as gauge_v4
import training.arbitrary_plane_finite_composite_v4 as composite_v4
import training.arbitrary_plane_finite_joint_curriculum_v5 as joint_v5
import training.arbitrary_plane_finite_pose_curriculum_v4 as pose_v4
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_row_cache_v4 as cache_v4
import test_arbitrary_plane_row_cache_v4 as cache_fixture


def _configs(pose_count=2, joint_count=3):
    capability = psf_v4.finite_psf_model_capability_v4()
    common = {
        "prepared_context_sha256": "a" * 64,
        "support_index_sha256": "b" * 64,
        "output_shape_h_w": [8, 8],
        "split": "development-finite-composite-v4",
        "stratum": "reference",
        "margin_u_v_um": [0.0, 0.0],
        "sections_per_animal": 4,
        "maximum_rejection_attempts": 64,
        "finite_parent_generator_source_commit": "1" * 40,
        "finite_slab_generator_source_commit": "1" * 40,
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    pose = {
        **common,
        "schema_version": pose_v4.FINITE_POSE_CURRICULUM_V4_SCHEMA,
        "algorithm": pose_v4.FINITE_POSE_CURRICULUM_V4_ALGORITHM,
        "row_count": pose_count,
        "root_seed_uint64": "0x0000000000000101",
        "start_index": 7,
        "identity_prefix": "finite-pose",
        "minimum_brain_pixels": 64,
        "maximum_pose_rejection_attempts": 16,
        "finite_psf_render_mode": "finite_boxcar",
        "finite_psf_capability": capability,
    }
    joint = {
        **common,
        "schema_version": joint_v5.FINITE_JOINT_CURRICULUM_V5_SCHEMA,
        "algorithm": joint_v5.FINITE_JOINT_CURRICULUM_V5_ALGORITHM,
        "row_count": joint_count,
        "root_seed_uint64": "0x0000000000000202",
        "start_index": 11,
        "identity_prefix": "finite-joint",
        "minimum_brain_pixels": 320,
        "maximum_joint_rejection_attempts": 16,
        "amplitude_band_cycle": ["mild", "moderate"],
        "render_mode": "finite_boxcar",
        "nominal_cut_thickness_um": None,
        "finite_psf_model_capability": capability,
    }
    return pose, joint


def _row(index, config):
    row = cache_fixture._row(index, 25.0 + index)
    row["lineage"]["split"] = config["split"]
    row["upstream_reference"]["schema_version"] = config["schema_version"]
    row["upstream_reference"]["algorithm"] = config["algorithm"]
    row["upstream_reference"]["implementation_source_sha256"] = (
        pose_v4._source_sha256()
        if config["schema_version"] == pose_v4.FINITE_POSE_CURRICULUM_V4_SCHEMA
        else joint_v5._source_sha256()
    )
    row["deformation_pose_gauge_reference"] = {
        **gauge_v4.direct_deformation_target_contract_v4(),
        "direct_deformation_target_id": acquisition_v2._payload_sha256(
            {"gauge": index}
        ),
        "receipt_sha256": acquisition_v2._payload_sha256(
            {"gauge-receipt": index}
        ),
    }
    row["receipt_sha256"] = acquisition_v2._payload_sha256(
        psf_v4.training_row_receipt_v4(row)
    )
    return row


def test_composite_config_and_binding_are_distinct_strict_v4_contracts():
    pose, joint = _configs()
    config = composite_v4.make_finite_composite_generation_config_v4(
        pose, joint
    )
    binding = composite_v4.make_finite_composite_generator_binding_v4(
        config,
        generation_run_id="finite-composite-v4-test",
        source_commit="2" * 40,
    )
    assert config["row_count"] == 5
    assert config["row_order_policy"] == (
        composite_v4.FINITE_COMPOSITE_ROW_ORDER_POLICY
    )
    assert config["finite_psf_run_contract"]["axial_sample_count"] == 9
    assert binding["schema_version"] == cache_v4.GENERATOR_BINDING_V4_SCHEMA
    assert binding["generator_ids"] == sorted(config["generator_ids"])
    assert cache_v4.verify_generator_binding_v4(binding)
    assert binding["prior_model_weight_dependencies"] == []

    changed = copy.deepcopy(joint)
    changed["render_mode"] = "centre_plane_ablation"
    with pytest.raises(ValueError, match="incompatible"):
        composite_v4.make_finite_composite_generation_config_v4(pose, changed)


def test_composite_range_crosses_pose_joint_boundary_without_drop(monkeypatch):
    pose, joint = _configs(pose_count=2, joint_count=3)
    config = composite_v4.make_finite_composite_generation_config_v4(
        pose, joint
    )

    def fake_rows(prepared_context, component, start_index, row_count):
        return [
            (component["schema_version"], start_index + offset)
            for offset in range(row_count)
        ]

    monkeypatch.setattr(composite_v4, "_component_rows", fake_rows)
    rows = composite_v4.make_finite_composite_training_rows_v4(
        object(), config, start_index=1, row_count=3
    )
    assert rows == [
        (pose_v4.FINITE_POSE_CURRICULUM_V4_SCHEMA, 8),
        (joint_v5.FINITE_JOINT_CURRICULUM_V5_SCHEMA, 11),
        (joint_v5.FINITE_JOINT_CURRICULUM_V5_SCHEMA, 12),
    ]


def test_composite_cache_resumes_into_one_ordered_frozen_v4_cache(
    tmp_path, monkeypatch
):
    pose, joint = _configs(pose_count=1, joint_count=1)
    config = composite_v4.make_finite_composite_generation_config_v4(
        pose, joint
    )
    binding = composite_v4.make_finite_composite_generator_binding_v4(
        config,
        generation_run_id="finite-composite-v4-resume-test",
        source_commit="3" * 40,
    )

    def fake_rows(prepared_context, component, start_index, row_count):
        base = 0 if component["schema_version"] == pose["schema_version"] else 1
        return [_row(base + offset, component) for offset in range(row_count)]

    monkeypatch.setattr(composite_v4, "_component_rows", fake_rows)
    root = tmp_path / "finite-composite-cache"
    manifest, audit = composite_v4.resume_finite_composite_cache_v4(
        root, object(), binding, chunk_size=1
    )
    assert manifest["status"] == cache_v4.FROZEN_CACHE_STATUS
    assert audit["row_count"] == 2
    loaded = cache_v4.load_training_rows_v4(root)
    assert [row["upstream_reference"]["algorithm"] for row in loaded] == [
        pose_v4.FINITE_POSE_CURRICULUM_V4_ALGORITHM,
        joint_v5.FINITE_JOINT_CURRICULUM_V5_ALGORITHM,
    ]
    resumed, second_audit = composite_v4.resume_finite_composite_cache_v4(
        root, object(), binding, chunk_size=1
    )
    assert resumed == manifest
    assert second_audit == audit
