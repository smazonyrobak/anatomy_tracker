"""Receipt-bound scientific audit for paired finite-thickness v4 caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

import training.arbitrary_plane_finite_joint_curriculum_v5 as joint_v5
import training.arbitrary_plane_finite_pose_curriculum_v4 as pose_v4
import training.arbitrary_plane_row_cache_v4 as cache_v4


FINITE_CACHE_SCIENTIFIC_AUDIT_V4_SCHEMA = (
    "anatomy-tracker.finite-cache-scientific-audit/v4"
)
COMPOSITE_GENERATOR_V4_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-finite-composite-curriculum/v4"
)
POSE_COMPONENT = "finite_identity_pose_curriculum"
JOINT_COMPONENT = "finite_nonidentity_joint_curriculum"
EXPECTED_BRUSH_MODES = {
    "smart-brush-accurate",
    "smart-brush-imperfect",
    "smart-brush-absent",
}
EXPECTED_TRAPEZOID_MASSES = [1, 2, 2, 2, 2, 2, 2, 2, 1]
EXPECTED_TRAPEZOID_WEIGHTS = [value / 16.0 for value in EXPECTED_TRAPEZOID_MASSES]
IDENTITY_POSE_SCHEMA = pose_v4.FINITE_POSE_CURRICULUM_V4_SCHEMA
NONIDENTITY_JOINT_SCHEMA = joint_v5.FINITE_JOINT_CURRICULUM_V5_SCHEMA
CONTRACTUAL_GATE_NAMES = (
    "cache-authentication",
    "no-dropped-logical-rows",
    "arbitrary-plane-parent",
    "finite-s9-trapezoid-psf",
    "independent-thickness-lineage",
    "pose-joint-strata",
    "smart-brush-optional",
    "appearance-damage-lineage",
    "occupancy-mass-lineage",
    "finite-values",
    "development-identity-disjointness",
    "internal-development-only-scope",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    return value


def _canonical_json(value):
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_json(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require(condition, gate, detail):
    if not condition:
        raise ValueError(f"finite scientific audit gate {gate!r} failed: {detail}")


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0, "descriptive-statistics", "empty numeric series")
    _require(np.isfinite(array).all(), "finite-values", "nonfinite descriptive value")
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.5)),
        "mean": float(array.mean()),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _counter(values):
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def _plane_coordinates(row):
    ouv_ml_ap_dv = np.asarray(
        row["canonical_effective_quicknii_ouv_float64"], dtype=np.float64
    ).reshape(3, 3)
    _require(
        np.isfinite(ouv_ml_ap_dv).all(),
        "finite-values",
        "nonfinite canonical plane geometry",
    )
    ouv = ouv_ml_ap_dv[:, [1, 2, 0]]
    origin, axis_u, axis_v = ouv
    normal = np.cross(axis_u, axis_v)
    normal_norm = float(np.linalg.norm(normal))
    axis_u_norm = float(np.linalg.norm(axis_u))
    _require(
        normal_norm > 0.0 and axis_u_norm > 0.0,
        "arbitrary-plane-parent",
        "degenerate plane basis",
    )
    normal /= normal_norm
    first = int(np.flatnonzero(np.abs(normal) > 1.0e-12)[0])
    sign = 1.0 if normal[first] > 0.0 else -1.0
    normal *= sign
    offset_um = float(np.dot(normal, origin))
    axis_u /= axis_u_norm
    reference_axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    tangent_1 = reference_axis - np.dot(reference_axis, normal) * normal
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    roll_deg = math.degrees(
        math.atan2(float(np.dot(axis_u, tangent_2)), float(np.dot(axis_u, tangent_1)))
    ) % 360.0
    return normal, offset_um, roll_deg


def _component_name(row):
    schema = row["numeric_rng_provenance"]["schema_version"]
    if schema == IDENTITY_POSE_SCHEMA:
        return POSE_COMPONENT
    if schema == NONIDENTITY_JOINT_SCHEMA:
        return JOINT_COMPONENT
    raise ValueError(
        "finite scientific audit gate 'pose-joint-strata' failed: "
        f"unknown curriculum schema {schema!r}"
    )


def _support(row):
    return row["upstream_reference"]["support_supervision_contract"]


def _pre_post_masses(row):
    support = _support(row)
    pre = float(support["pre_g1_slab_effective_brain_pixel_mass"])
    if _component_name(row) == POSE_COMPONENT:
        post = float(
            support["post_g1_point_pose_evidence_effective_brain_pixel_mass"]
        )
        dense = float(support["post_g3_dense_effective_supervision_mass"])
    else:
        post = float(support["post_g1_slab_effective_brain_pixel_mass"])
        dense = float(support["post_g3_dense_correspondence_weight_mass"])
    return pre, post, dense


def _thickness_seed(row):
    numeric = row["numeric_rng_provenance"]
    if _component_name(row) == POSE_COMPONENT:
        return numeric["finite_slab_thickness_seed_uint64"]
    return numeric["finite_thickness_seed_uint64"]


def _logical_row_gate(manifest, rows):
    config = manifest["generation_config"]
    _require(
        config.get("schema_version") == COMPOSITE_GENERATOR_V4_SCHEMA,
        "no-dropped-logical-rows",
        "cache is not the namespaced finite composite curriculum",
    )
    _require(
        manifest["status"] == cache_v4.FROZEN_CACHE_STATUS,
        "cache-authentication",
        "scientific audit requires an immutable frozen cache",
    )
    _require(
        manifest["row_count"] == config["row_count"] == len(rows),
        "no-dropped-logical-rows",
        "manifest, generation config, and loaded row counts differ",
    )
    _require(
        "retain every logical row exactly once"
        in config.get("marginal_or_empty_row_policy", ""),
        "no-dropped-logical-rows",
        "marginal/empty no-drop policy is absent",
    )
    component_configs = config["component_generation_configs"]
    declared_counts = config["component_row_counts"]
    actual_names = [_component_name(row) for row in rows]
    for name in (POSE_COMPONENT, JOINT_COMPONENT):
        component = component_configs[name]
        expected_count = int(declared_counts[name])
        selected = [
            row for row, actual_name in zip(rows, actual_names) if actual_name == name
        ]
        _require(
            expected_count == component["row_count"] == len(selected),
            "no-dropped-logical-rows",
            f"declared and loaded {name} counts differ",
        )
        start = int(component["start_index"])
        indices = sorted(
            int(row["numeric_rng_provenance"]["sample_index"]) for row in selected
        )
        _require(
            indices == list(range(start, start + expected_count)),
            "no-dropped-logical-rows",
            f"{name} logical sample indices contain a gap or duplicate",
        )
    return actual_names


def _audit_rows(label, manifest, cache_audit, rows):
    component_names = _logical_row_gate(manifest, rows)
    config = manifest["generation_config"]
    components = config["component_generation_configs"]
    _require(
        all(
            "all" in component["plane_domain"].lower()
            and "brain-intersecting" in component["plane_domain"].lower()
            for component in components.values()
        ),
        "arbitrary-plane-parent",
        "component plane domain is not all brain-intersecting planes",
    )

    normals = []
    offsets = []
    rolls = []
    thicknesses = []
    steps = []
    pre_masses = []
    post_masses = []
    dense_masses = []
    velocity_rms = []
    velocity_max = []
    missing_fractions = []
    image_means = []
    image_stds = []
    image_zero_fractions = []
    g2_ids = []
    g3_ids = []
    thickness_seeds = []
    finite_render_seeds = []
    point_weights = []
    dense_weights = []
    joint_censor_statuses = []
    joint_amplitude_bands = []
    gate_diagnostics = []

    for row_index, row in enumerate(rows):
        upstream = row["upstream_reference"]
        support = _support(row)
        arrays = row["arrays"]
        component = component_names[row_index]
        measure = upstream["plane_sampling_measure"]
        parent_identity = upstream["finite_parent_identity"]
        _require(
            "Haar-uniform RP2" in measure["orientation"]
            and "unconditioned" in measure["conditioning"]
            and "independent uniform" in measure["roll"]
            and "length-uniform" in measure["reference_offset"],
            "arbitrary-plane-parent",
            "row plane measure is restricted or no longer RP2/Haar",
        )
        _require(
            all(
                isinstance(parent_identity.get(name), str)
                and len(parent_identity[name]) == 64
                for name in (
                    "plane_realization_id",
                    "finite_plane_render_id",
                    "finite_parent_provenance_sha256",
                )
            )
            and parent_identity["finite_plane_render_id"]
            == upstream["finite_plane_render_id"]
            and parent_identity["finite_parent_provenance_sha256"]
            == upstream["finite_parent_provenance_sha256"],
            "arbitrary-plane-parent",
            "authenticated brain-intersecting finite parent identity is absent",
        )
        _require(
            support["continuous_plane_sample_retained"] is True,
            "no-dropped-logical-rows",
            "continuous parent was not retained",
        )
        if component == POSE_COMPONENT:
            _require(
                support["pose_redrawn_for_raster_or_slab_support"] is False,
                "arbitrary-plane-parent",
                "pose row was redrawn for observed support",
            )
        else:
            _require(
                support["pose_redrawn_for_raster_support"] is False
                and support["finite_parent_reused_across_retries"] is True
                and support["finite_slab_reused_across_retries"] is True,
                "arbitrary-plane-parent",
                "joint row parent/slab changed during retries",
            )
        normal, offset, roll = _plane_coordinates(row)
        normals.append(normal)
        offsets.append(offset)
        rolls.append(roll)

        psf = row["finite_psf_contract"]
        _require(
            psf["render_mode"] == "finite_boxcar"
            and psf["family"] == "boxcar"
            and psf["axial_sample_count"] == 9
            and psf["axial_integer_masses"] == EXPECTED_TRAPEZOID_MASSES
            and np.allclose(
                psf["axial_weights"], EXPECTED_TRAPEZOID_WEIGHTS, rtol=0.0, atol=0.0
            )
            and psf["sampling_direction"]
            == "canonical physical AP-DV-ML arbitrary-plane normal"
            and psf["normalization"]
            == "global unit-mass PSF; no per-pixel in-bounds renormalization"
            and psf["outside_atlas_rule"]
            == "zero padding before global weighted sum",
            "finite-s9-trapezoid-psf",
            "row does not use the exact production finite slab operator",
        )
        thickness = float(psf["nominal_cut_thickness_um"])
        step = float(psf["axial_step_um"])
        _require(
            25.0 <= thickness <= 100.0
            and step <= 12.5
            and math.isclose(step, thickness / 8.0, rel_tol=0.0, abs_tol=1.0e-12)
            and np.allclose(
                psf["axial_offsets_um"],
                np.linspace(-thickness / 2.0, thickness / 2.0, 9),
                rtol=0.0,
                atol=1.0e-12,
            ),
            "finite-s9-trapezoid-psf",
            "row thickness, offsets, or axial step violates the production schedule",
        )
        thicknesses.append(thickness)
        steps.append(step)
        thickness_seed = _thickness_seed(row)
        finite_seed = row["numeric_rng_provenance"]["finite_render_seed_uint64"]
        slab_seed = (
            upstream["finite_slab_reference"]["thickness_seed_uint64"]
            if component == POSE_COMPONENT
            else upstream["finite_slab_identity"][
                "independent_thickness_seed_uint64"
            ]
        )
        _require(
            isinstance(thickness_seed, str)
            and thickness_seed.startswith("0x")
            and thickness_seed != finite_seed
            and thickness_seed == slab_seed,
            "independent-thickness-lineage",
            "thickness seed is absent or aliases the finite parent seed",
        )
        thickness_seeds.append(thickness_seed)
        finite_render_seeds.append(finite_seed)

        pre_mass, post_mass, dense_mass = _pre_post_masses(row)
        _require(
            np.isfinite([pre_mass, post_mass, dense_mass]).all()
            and min(pre_mass, post_mass, dense_mass) >= 0.0,
            "occupancy-mass-lineage",
            "pre/post-G1 or post-G3 mass is nonfinite or negative",
        )
        pre_masses.append(pre_mass)
        post_masses.append(post_mass)
        dense_masses.append(dense_mass)
        point_weight = float(support["point_pose_supervision_weight"])
        dense_weight = float(support["dense_deformation_supervision_weight"])
        _require(
            point_weight in (0.0, 1.0) and dense_weight in (0.0, 1.0),
            "pose-joint-strata",
            "supervision scalar is not binary",
        )
        point_weights.append(point_weight)
        dense_weights.append(dense_weight)
        recomputed_dense_mass = float(
            np.asarray(
                arrays["target_correspondence_weight_float32"], dtype=np.float64
            ).sum()
        )
        _require(
            dense_mass == recomputed_dense_mass,
            "occupancy-mass-lineage",
            "post-G3 dense mass differs from the persisted correspondence weights",
        )

        velocity = np.asarray(
            arrays["truth_section_pullback_stationary_velocity_yx_px_float64"],
            dtype=np.float64,
        )
        valid = np.asarray(arrays["truth_section_deformation_valid_mask"], dtype=bool)
        vector_norm = np.linalg.norm(velocity[valid], axis=-1)
        velocity_rms.append(
            float(np.sqrt(np.mean(np.square(vector_norm)))) if vector_norm.size else 0.0
        )
        velocity_max.append(float(vector_norm.max()) if vector_norm.size else 0.0)
        if component == POSE_COMPONENT:
            _require(
                upstream["g1_identity_forced"] is True
                and dense_weight == 0.0
                and not np.any(velocity != 0.0),
                "pose-joint-strata",
                "pose component is not exact identity-G1 with censored dense loss",
            )
        else:
            band = upstream["deformation_amplitude_band"]
            censor = upstream["deformation_censoring_contract"]
            status = censor["status"]
            joint_amplitude_bands.append(band)
            joint_censor_statuses.append(status)
            _require(
                band in joint_v5.DEFORMATION_AMPLITUDE_BANDS
                and upstream["requested_deformation_amplitude_band"] == band
                and upstream["g1_overrides"]["target_rms_displacement_over_D"]
                == list(joint_v5.DEFORMATION_AMPLITUDE_BANDS[band]),
                "pose-joint-strata",
                "joint deformation amplitude request differs from its G1 realization band",
            )
            if status == joint_v5.UNCENSORED_DEFORMATION_STATUS:
                _require(
                    dense_weight == 1.0
                    and support["dense_deformation_supervision_identifiable"] is True
                    and support["center_gauge_support_identifiable"] is True
                    and dense_mass > 0.0
                    and np.any(velocity != 0.0),
                    "pose-joint-strata",
                    "uncensored joint row is not identifiable nonidentity deformation",
                )
            else:
                _require(
                    dense_weight == 0.0
                    and support["dense_deformation_supervision_identifiable"] is False,
                    "pose-joint-strata",
                    "censored joint row retained nonzero dense supervision",
                )

        mode = row["selected_mode"]
        available = np.asarray(
            arrays["model_input_channels_float32"], dtype=np.float64
        )[..., 2]
        expected_available = 0.0 if mode == "smart-brush-absent" else 1.0
        expected_black = None if mode == "smart-brush-absent" else True
        _require(
            mode in EXPECTED_BRUSH_MODES
            and np.all(available == expected_available)
            and upstream["selected_black_exterior_exact"] is expected_black,
            "smart-brush-optional",
            "brush availability or black-exterior contract changed",
        )
        stage_ids = upstream["selected_stage_realization_ids"]
        _require(
            set(stage_ids) == {"g1", "g2", "g3", "outline"}
            and all(isinstance(value, str) and len(value) == 64 for value in stage_ids.values()),
            "appearance-damage-lineage",
            "authenticated appearance/damage stage IDs are absent",
        )
        g2_ids.append(stage_ids["g2"])
        g3_ids.append(stage_ids["g3"])
        image = np.asarray(arrays["model_input_channels_float32"], dtype=np.float64)[
            ..., 0
        ]
        image_means.append(float(image.mean()))
        image_stds.append(float(image.std()))
        image_zero_fractions.append(float(np.mean(image == 0.0)))
        labels = np.asarray(arrays["source_label_ground_truth_canvas_int64"])
        tissue = np.asarray(arrays["source_tissue_ground_truth_mask"], dtype=bool)
        source_brain = labels != 0
        denominator = int(source_brain.sum())
        missing_fractions.append(
            float(np.sum(source_brain & ~tissue) / denominator) if denominator else 0.0
        )

        for name, value in arrays.items():
            _require(
                np.isfinite(np.asarray(value)).all(),
                "finite-values",
                f"row {row_index} array {name!r} contains a nonfinite value",
            )
        diagnostics = upstream["direct_deformation_target_certification_summary"][
            "diagnostics"
        ]
        _require(
            all(np.isfinite(float(value)) for value in diagnostics.values()),
            "finite-values",
            "deformation gauge diagnostics are nonfinite",
        )
        gate_diagnostics.append(diagnostics)

    _require(
        set(row["selected_mode"] for row in rows) == EXPECTED_BRUSH_MODES,
        "smart-brush-optional",
        "cache does not include accurate, imperfect, and absent brush modes",
    )
    _require(
        set(joint_amplitude_bands) == set(joint_v5.DEFORMATION_AMPLITUDE_BANDS),
        "pose-joint-strata",
        "declared mild/moderate joint deformation strata are incomplete",
    )
    _require(
        all(
            "independent" in component.get("thickness_selection", component.get("slab_policy", ""))
            for component in components.values()
        ),
        "independent-thickness-lineage",
        "component configuration does not declare independent thickness sampling",
    )

    normals = np.asarray(normals, dtype=np.float64)
    dominant_axes = np.argmax(np.abs(normals), axis=1)
    signs = ["+" if value >= 0.0 else "-" for value in normals[:, 1:3].prod(axis=1)]
    return {
        "cache_binding": {
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "generator_binding_receipt_sha256": manifest["generator_binding"][
                "receipt_sha256"
            ],
            "finite_psf_run_contract_receipt_sha256": manifest[
                "finite_psf_run_contract"
            ]["receipt_sha256"],
            "cache_audit_receipt_sha256": cache_audit["receipt_sha256"],
            "ordered_training_row_receipts_sha256": cache_audit[
                "ordered_training_row_receipts_sha256"
            ],
            "row_count": len(rows),
        },
        "logical_rows": {
            "component_counts": _counter(component_names),
            "zero_point_pose_weight_rows_retained": int(np.sum(np.asarray(point_weights) == 0.0)),
            "zero_dense_weight_rows_retained": int(np.sum(np.asarray(dense_weights) == 0.0)),
            "no_drop_fallback_rows_retained": int(
                sum(bool(row["upstream_reference"].get("no_drop_fallback", False)) for row in rows)
            ),
        },
        "rp2_plane_coverage_descriptive_only": {
            "coordinate_order": "AP-DV-ML",
            "antipodal_canonicalization": "first nonzero normal component positive",
            "normal_component_ap": _summary(normals[:, 0]),
            "normal_component_dv": _summary(normals[:, 1]),
            "normal_component_ml": _summary(normals[:, 2]),
            "dominant_axis_counts": {
                "AP": int(np.sum(dominant_axes == 0)),
                "DV": int(np.sum(dominant_axes == 1)),
                "ML": int(np.sum(dominant_axes == 2)),
            },
            "secondary_component_product_sign_counts": _counter(signs),
            "signed_offset_um": _summary(offsets),
            "roll_degrees_0_360": _summary(rolls),
            "empirical_coverage_is_not_a_gate": True,
        },
        "finite_psf": {
            "nominal_cut_thickness_um": _summary(thicknesses),
            "axial_step_um": _summary(steps),
            "unique_thickness_seed_count": len(set(thickness_seeds)),
            "unique_finite_parent_seed_count": len(set(finite_render_seeds)),
            "thickness_seed_parent_seed_collision_count": int(
                sum(a == b for a, b in zip(thickness_seeds, finite_render_seeds))
            ),
        },
        "supervision": {
            "point_pose_weight_counts": _counter(point_weights),
            "dense_deformation_weight_counts": _counter(dense_weights),
            "pre_g1_slab_occupancy_mass": _summary(pre_masses),
            "post_g1_slab_occupancy_mass": _summary(post_masses),
            "post_g3_dense_weight_mass": _summary(dense_masses),
            "joint_deformation_amplitude_band_counts": _counter(joint_amplitude_bands),
            "joint_deformation_censor_status_counts": _counter(joint_censor_statuses),
            "deformation_velocity_rms_px": _summary(velocity_rms),
            "deformation_velocity_max_px": _summary(velocity_max),
            "gauge_uniform_affine_coefficient_max_abs": _summary(
                [
                    item["uniform_canvas_affine_coefficient_max_abs"]
                    for item in gate_diagnostics
                ]
            ),
            "gauge_certification_error_max_px": _summary(
                [
                    item["valid_certification_error_max_px"]
                    for item in gate_diagnostics
                ]
            ),
        },
        "appearance_damage_and_brush": {
            "brush_mode_counts": _counter(row["selected_mode"] for row in rows),
            "unique_g2_appearance_realization_count": len(set(g2_ids)),
            "unique_g3_damage_realization_count": len(set(g3_ids)),
            "image_mean": _summary(image_means),
            "image_std": _summary(image_stds),
            "image_exact_zero_fraction": _summary(image_zero_fractions),
            "missing_tissue_fraction_within_nonzero_source_label": _summary(
                missing_fractions
            ),
            "empirical_background_and_damage_diversity_are_not_gates": True,
        },
        "identity_sets": {
            name: sorted({str(row["lineage"][name]) for row in rows})
            for name in (
                "animal_id",
                "synthetic_animal_id",
                "specimen_id",
                "experiment_id",
                "section_id",
            )
        },
        "split_label_counts": _counter(row["lineage"]["split"] for row in rows),
        "label": label,
    }


def audit_finite_cache_pair_scientific_v4(
    training_cache_directory,
    development_cache_directory,
    *,
    output_json_path=None,
):
    """Authenticate and scientifically audit paired internal finite caches."""
    manifests = {}
    cache_audits = {}
    rows = {}
    for label, directory in (
        ("training", training_cache_directory),
        ("internal_development", development_cache_directory),
    ):
        manifests[label] = cache_v4.load_training_row_cache_manifest_v4(directory)
        cache_audits[label] = cache_v4.audit_training_row_cache_v4(directory)
        rows[label] = cache_v4.load_training_rows_v4(directory)

    split_reports = {
        label: _audit_rows(label, manifests[label], cache_audits[label], rows[label])
        for label in manifests
    }
    disjointness = {}
    for name in (
        "animal_id",
        "synthetic_animal_id",
        "specimen_id",
        "experiment_id",
        "section_id",
    ):
        training_ids = set(split_reports["training"]["identity_sets"][name])
        development_ids = set(
            split_reports["internal_development"]["identity_sets"][name]
        )
        overlap = sorted(training_ids & development_ids)
        _require(
            not overlap,
            "development-identity-disjointness",
            f"training/development {name} overlap: {overlap}",
        )
        disjointness[name] = {
            "training_unique_count": len(training_ids),
            "internal_development_unique_count": len(development_ids),
            "intersection": overlap,
            "passed": True,
        }
    _require(
        set(split_reports["training"]["split_label_counts"]) == {"train"}
        and set(split_reports["internal_development"]["split_label_counts"])
        == {"development"},
        "development-identity-disjointness",
        "cache roles are not exact train and internal development splits",
    )

    payload = {
        "schema_version": FINITE_CACHE_SCIENTIFIC_AUDIT_V4_SCHEMA,
        "scope": {
            "data_roles": ["development-training", "internal-development"],
            "public_benchmark_accessed": False,
            "external_validation_accessed": False,
            "final_test_accessed": False,
            "performance_thresholds_evaluated": False,
        },
        "cache_reports": split_reports,
        "training_internal_development_identity_disjointness": disjointness,
        "contractual_gates": {
            name: {"passed": True} for name in CONTRACTUAL_GATE_NAMES
        },
        "all_contractual_gates_passed": True,
    }
    report = {**payload, "receipt_sha256": _hash_json(payload)}
    if output_json_path is not None:
        target = Path(output_json_path).resolve()
        _require(
            os.path.splitdrive(str(target))[0].upper() == "I:",
            "internal-development-only-scope",
            "audit report must be written only on I:",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(_canonical_json(report) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    return report


def verify_finite_cache_scientific_audit_v4(report):
    """Verify the JSON receipt and the explicit all-pass gate declaration."""
    if not isinstance(report, dict) or report.get("schema_version") != (
        FINITE_CACHE_SCIENTIFIC_AUDIT_V4_SCHEMA
    ):
        raise ValueError("finite cache scientific audit schema changed")
    payload = {key: value for key, value in report.items() if key != "receipt_sha256"}
    if (
        report.get("receipt_sha256") != _hash_json(payload)
        or report.get("all_contractual_gates_passed") is not True
        or set(report.get("contractual_gates", {})) != set(CONTRACTUAL_GATE_NAMES)
        or any(
            gate.get("passed") is not True
            for gate in report.get("contractual_gates", {}).values()
        )
        or report.get("scope")
        != {
            "data_roles": ["development-training", "internal-development"],
            "public_benchmark_accessed": False,
            "external_validation_accessed": False,
            "final_test_accessed": False,
            "performance_thresholds_evaluated": False,
        }
    ):
        raise ValueError("finite cache scientific audit receipt or gate status changed")
    return True


__all__ = [
    "FINITE_CACHE_SCIENTIFIC_AUDIT_V4_SCHEMA",
    "audit_finite_cache_pair_scientific_v4",
    "verify_finite_cache_scientific_audit_v4",
]
