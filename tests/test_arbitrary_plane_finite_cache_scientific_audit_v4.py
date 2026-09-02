import copy
from pathlib import Path

import numpy as np
import pytest

import training.arbitrary_plane_finite_cache_scientific_audit_v4 as audit_v4
import training.arbitrary_plane_finite_joint_curriculum_v5 as joint_v5
import training.arbitrary_plane_finite_pose_curriculum_v4 as pose_v4
import training.arbitrary_plane_row_cache_v4 as cache_v4


def _sha(character):
    return character * 64


def _generation_config(row_count=6):
    return {
        "schema_version": audit_v4.COMPOSITE_GENERATOR_V4_SCHEMA,
        "algorithm": "finite-pose-v4-then-finite-joint-v5-single-homogeneous-cache/v4",
        "row_count": row_count,
        "marginal_or_empty_row_policy": (
            "retain every logical row exactly once; zero supervision weights remain zero"
        ),
        "component_row_counts": {
            audit_v4.POSE_COMPONENT: row_count // 2,
            audit_v4.JOINT_COMPONENT: row_count // 2,
        },
        "component_generation_configs": {
            audit_v4.POSE_COMPONENT: {
                "row_count": row_count // 2,
                "start_index": 0,
                "plane_domain": "all brain-intersecting planes",
                "thickness_selection": "independent seeded uniform continuous 25-100 um",
            },
            audit_v4.JOINT_COMPONENT: {
                "row_count": row_count // 2,
                "start_index": 0,
                "plane_domain": "all continuous brain-intersecting arbitrary planes",
                "slab_policy": "one independent thickness descendant per parent",
            },
        },
    }


def _manifest(split, row_count=6):
    return {
        "status": cache_v4.FROZEN_CACHE_STATUS,
        "row_count": row_count,
        "generation_config": _generation_config(row_count),
        "receipt_sha256": _sha("a" if split == "train" else "b"),
        "generator_binding": {
            "receipt_sha256": _sha("c" if split == "train" else "d")
        },
        "finite_psf_run_contract": {
            "receipt_sha256": _sha("e" if split == "train" else "f")
        },
    }


def _cache_audit(split):
    return {
        "receipt_sha256": _sha("1" if split == "train" else "2"),
        "ordered_training_row_receipts_sha256": _sha(
            "3" if split == "train" else "4"
        ),
    }


def _ouv_ml_ap_dv(index):
    normals_ap_dv_ml = (
        np.array([1.0, 1.0 + index, 2.0]),
        np.array([-1.0, 2.0, 1.0 + index]),
        np.array([2.0 + index, -1.0, 1.0]),
    )
    normal = normals_ap_dv_ml[index % len(normals_ap_dv_ml)]
    normal /= np.linalg.norm(normal)
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    axis_u = reference - np.dot(reference, normal) * normal
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    origin = normal * (100.0 + index)
    return np.stack((origin, axis_u, axis_v))[:, [2, 0, 1]].tolist()


def _row(split, component, index, mode, band="mild", point_weight=1.0):
    pose = component == audit_v4.POSE_COMPONENT
    schema = (
        pose_v4.FINITE_POSE_CURRICULUM_V4_SCHEMA
        if pose
        else joint_v5.FINITE_JOINT_CURRICULUM_V5_SCHEMA
    )
    shape = (4, 4)
    velocity = np.zeros((*shape, 2), dtype=np.float64)
    if not pose:
        velocity[..., 0] = 0.25 + index * 0.01
    dense_weight = 0.0 if pose else 1.0
    pre_mass = 10.0 + index
    post_mass = pre_mass if pose else pre_mass - 0.5
    dense_mass = 8.0 + index
    support = {
        "continuous_plane_sample_retained": True,
        "pre_g1_slab_effective_brain_pixel_mass": pre_mass,
        "point_pose_supervision_weight": point_weight,
        "dense_deformation_supervision_weight": dense_weight,
    }
    if pose:
        support.update(
            {
                "pose_redrawn_for_raster_or_slab_support": False,
                "post_g1_point_pose_evidence_effective_brain_pixel_mass": post_mass,
                "post_g3_dense_effective_supervision_mass": dense_mass,
            }
        )
    else:
        support.update(
            {
                "pose_redrawn_for_raster_support": False,
                "finite_parent_reused_across_retries": True,
                "finite_slab_reused_across_retries": True,
                "post_g1_slab_effective_brain_pixel_mass": post_mass,
                "post_g3_dense_correspondence_weight_mass": dense_mass,
                "dense_deformation_supervision_identifiable": True,
                "center_gauge_support_identifiable": True,
            }
        )
    available = 0.0 if mode == "smart-brush-absent" else 1.0
    image = np.full(shape, 0.1 + index * 0.05, dtype=np.float32)
    channels = np.stack(
        (image, np.zeros(shape, np.float32), np.full(shape, available, np.float32)),
        axis=-1,
    )
    arrays = {
        "model_input_channels_float32": channels,
        "source_label_ground_truth_canvas_int64": np.ones(shape, np.int64),
        "source_tissue_ground_truth_mask": np.ones(shape, bool),
        "target_ccf_coordinates_ap_dv_ml_um_float64": np.zeros((*shape, 3)),
        "target_valid_correspondence_mask": np.ones(shape, bool),
        "target_correspondence_weight_float32": np.full(
            shape, dense_mass / np.prod(shape), np.float32
        ),
        "target_correspondence_abstention_mask": np.zeros(shape, bool),
        "truth_section_pullback_map_yx_px_float64": np.zeros((*shape, 2)),
        "truth_section_pullback_stationary_velocity_yx_px_float64": velocity,
        "truth_section_deformation_valid_mask": np.ones(shape, bool),
    }
    stage_base = 10 + index + (0 if split == "train" else 20)
    upstream = {
        "finite_parent_identity": {
            "plane_realization_id": f"{stage_base + 400:064x}",
            "finite_plane_render_id": f"{stage_base + 401:064x}",
            "finite_parent_provenance_sha256": f"{stage_base + 402:064x}",
        },
        "finite_plane_render_id": f"{stage_base + 401:064x}",
        "finite_parent_provenance_sha256": f"{stage_base + 402:064x}",
        "plane_sampling_measure": {
            "orientation": "orientation-balanced Haar-uniform RP2 normal",
            "conditioning": "unconditioned by the finite raster",
            "roll": "independent uniform [0,2pi)",
            "reference_offset": "length-uniform over support intervals",
        },
        "support_supervision_contract": support,
        "selected_black_exterior_exact": (
            None if mode == "smart-brush-absent" else True
        ),
        "selected_stage_realization_ids": {
            "g1": f"{stage_base:064x}",
            "g2": f"{stage_base + 100:064x}",
            "g3": f"{stage_base + 200:064x}",
            "outline": f"{stage_base + 300:064x}",
        },
        "direct_deformation_target_certification_summary": {
            "diagnostics": {
                "uniform_canvas_affine_coefficient_max_abs": 0.0,
                "valid_certification_error_max_px": 0.0,
            }
        },
    }
    if pose:
        upstream["g1_identity_forced"] = True
        upstream["finite_slab_reference"] = {
            "thickness_seed_uint64": f"0x{index + 101:016x}"
        }
    else:
        upstream["deformation_amplitude_band"] = band
        upstream["requested_deformation_amplitude_band"] = band
        upstream["g1_overrides"] = {
            "target_rms_displacement_over_D": list(
                joint_v5.DEFORMATION_AMPLITUDE_BANDS[band]
            )
        }
        upstream["finite_slab_identity"] = {
            "independent_thickness_seed_uint64": f"0x{index + 101:016x}"
        }
        upstream["deformation_censoring_contract"] = {
            "status": joint_v5.UNCENSORED_DEFORMATION_STATUS
        }
    thickness = 25.0 + index * 5.0
    finite_psf = {
        "render_mode": "finite_boxcar",
        "family": "boxcar",
        "axial_sample_count": 9,
        "axial_integer_masses": audit_v4.EXPECTED_TRAPEZOID_MASSES,
        "axial_weights": audit_v4.EXPECTED_TRAPEZOID_WEIGHTS,
        "sampling_direction": "canonical physical AP-DV-ML arbitrary-plane normal",
        "normalization": "global unit-mass PSF; no per-pixel in-bounds renormalization",
        "outside_atlas_rule": "zero padding before global weighted sum",
        "nominal_cut_thickness_um": thickness,
        "axial_step_um": thickness / 8.0,
        "axial_offsets_um": np.linspace(-thickness / 2.0, thickness / 2.0, 9).tolist(),
    }
    numeric = {
        "schema_version": schema,
        "sample_index": index,
        "finite_render_seed_uint64": f"0x{index + 1:016x}",
    }
    numeric[
        "finite_slab_thickness_seed_uint64"
        if pose
        else "finite_thickness_seed_uint64"
    ] = f"0x{index + 101:016x}"
    prefix = f"{split}-{component}-{index}"
    return {
        "canonical_effective_quicknii_ouv_float64": _ouv_ml_ap_dv(index),
        "numeric_rng_provenance": numeric,
        "upstream_reference": upstream,
        "finite_psf_contract": finite_psf,
        "selected_mode": mode,
        "arrays": arrays,
        "lineage": {
            "animal_id": prefix + "-animal",
            "synthetic_animal_id": prefix + "-synthetic",
            "specimen_id": prefix + "-specimen",
            "experiment_id": prefix + "-experiment",
            "section_id": prefix + "-section",
            "split": split,
        },
    }


def _rows(split):
    modes = sorted(audit_v4.EXPECTED_BRUSH_MODES)
    pose_rows = [
        _row(
            split,
            audit_v4.POSE_COMPONENT,
            index,
            modes[index],
            point_weight=0.0 if index == 2 else 1.0,
        )
        for index in range(3)
    ]
    joint_rows = [
        _row(
            split,
            audit_v4.JOINT_COMPONENT,
            index,
            modes[(index + 1) % 3],
            band="mild" if index != 1 else "moderate",
        )
        for index in range(3)
    ]
    return pose_rows + joint_rows


@pytest.fixture
def synthetic_cache_pair(monkeypatch):
    manifests = {
        "train-cache": _manifest("train"),
        "dev-cache": _manifest("development"),
    }
    audits = {
        "train-cache": _cache_audit("train"),
        "dev-cache": _cache_audit("development"),
    }
    rows = {"train-cache": _rows("train"), "dev-cache": _rows("development")}
    monkeypatch.setattr(
        audit_v4.cache_v4,
        "load_training_row_cache_manifest_v4",
        lambda path: manifests[str(path)],
    )
    monkeypatch.setattr(
        audit_v4.cache_v4,
        "audit_training_row_cache_v4",
        lambda path: audits[str(path)],
    )
    monkeypatch.setattr(
        audit_v4.cache_v4,
        "load_training_rows_v4",
        lambda path: rows[str(path)],
    )
    return manifests, audits, rows


def test_receipt_bound_pair_audit_reports_contracts_without_performance_gate(
    synthetic_cache_pair,
):
    report = audit_v4.audit_finite_cache_pair_scientific_v4(
        "train-cache", "dev-cache"
    )
    assert audit_v4.verify_finite_cache_scientific_audit_v4(report)
    assert report["all_contractual_gates_passed"]
    assert not report["scope"]["public_benchmark_accessed"]
    assert not report["scope"]["performance_thresholds_evaluated"]
    training = report["cache_reports"]["training"]
    assert training["logical_rows"]["zero_point_pose_weight_rows_retained"] == 1
    assert training["logical_rows"]["zero_dense_weight_rows_retained"] == 3
    assert training["rp2_plane_coverage_descriptive_only"][
        "empirical_coverage_is_not_a_gate"
    ]
    assert training["appearance_damage_and_brush"]["brush_mode_counts"] == {
        "smart-brush-absent": 2,
        "smart-brush-accurate": 2,
        "smart-brush-imperfect": 2,
    }


def test_identity_overlap_is_a_contract_failure(synthetic_cache_pair):
    _, _, rows = synthetic_cache_pair
    rows["dev-cache"][0]["lineage"]["animal_id"] = rows["train-cache"][0][
        "lineage"
    ]["animal_id"]
    with pytest.raises(ValueError, match="animal_id overlap"):
        audit_v4.audit_finite_cache_pair_scientific_v4("train-cache", "dev-cache")


def test_dropped_logical_index_and_restricted_plane_measure_fail(
    synthetic_cache_pair,
):
    _, _, rows = synthetic_cache_pair
    rows["train-cache"][1]["numeric_rng_provenance"]["sample_index"] = 0
    with pytest.raises(ValueError, match="gap or duplicate"):
        audit_v4.audit_finite_cache_pair_scientific_v4("train-cache", "dev-cache")
    rows["train-cache"] = _rows("train")
    rows["train-cache"][0]["upstream_reference"]["plane_sampling_measure"][
        "orientation"
    ] = "coronal-only"
    with pytest.raises(ValueError, match="RP2/Haar"):
        audit_v4.audit_finite_cache_pair_scientific_v4("train-cache", "dev-cache")


def test_report_receipt_tamper_and_non_i_output_are_rejected(synthetic_cache_pair):
    report = audit_v4.audit_finite_cache_pair_scientific_v4(
        "train-cache", "dev-cache"
    )
    changed = copy.deepcopy(report)
    changed["scope"]["public_benchmark_accessed"] = True
    with pytest.raises(ValueError, match="receipt or gate"):
        audit_v4.verify_finite_cache_scientific_audit_v4(changed)
    with pytest.raises(ValueError, match="only on I"):
        audit_v4.audit_finite_cache_pair_scientific_v4(
            "train-cache",
            "dev-cache",
            output_json_path=Path("C:/forbidden/audit.json"),
        )
