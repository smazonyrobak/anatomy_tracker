"""Small development-only qualification of subject-deformed finite-slab quadrature."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_subject_deformation_v2 as deformation
import training.arbitrary_plane_subject_slab_v2 as subject_slab
import training.arbitrary_plane_support_resolution_v2 as support_resolution
import training.arbitrary_plane_synthetic_generator_v2 as slab


SUBJECT_DEFORMED_SLAB_QUALIFICATION_V2_SCHEMA = (
    "anatomy-tracker.subject-deformed-slab-qualification/v2"
)
SUBJECT_DEFORMED_SLAB_CASE_V2_SCHEMA = (
    "anatomy-tracker.subject-deformed-slab-qualification-case/v2"
)
COARSE_AXIAL_STEP_UM_MAX = 12.5
REFINED_AXIAL_STEP_UM_MAX = 6.25
NOMINAL_CUT_THICKNESS_UM = 55.0
MAX_SUPPORT_ATTEMPTS = 8
THRESHOLDS = MappingProxyType({
    "continuous_mean_absolute_error_max": 0.02,
    "continuous_absolute_error_p99_max": 0.10,
    "categorical_disagreement_fraction_max": 0.02,
})
_ANIMAL_ROWS = (
    (
        1101,
        "development-subject-deformed-animal-1101",
        "development-specimen-1101",
        "subject-deformed-slab-development-experiment",
        "0x534451a11a1a0001",
    ),
    (
        1102,
        "development-subject-deformed-animal-1102",
        "development-specimen-1102",
        "subject-deformed-slab-development-experiment",
        "0x534451a11a1a0002",
    ),
)
_CASE_ROWS = (
    ("sdq-reference-a", 1101, 0, 0, "reference", "0x5344515aab000001"),
    ("sdq-near-ap-a", 1101, 1, 1, "near_AP", "0x5344515aab000002"),
    ("sdq-oblique-a", 1101, 2, 2, "general_oblique", "0x5344515aab000003"),
    ("sdq-near-dv-b", 1102, 3, 3, "near_DV", "0x5344515aab000004"),
    ("sdq-near-ml-b", 1102, 4, 4, "near_ML", "0x5344515aab000005"),
    ("sdq-edge-b", 1102, 5, 5, "edge_or_partial", "0x5344515aab000006"),
)
_DEFORMATION_PARAMETERS = MappingProxyType({
    "deformation_stratum": "standard",
    "coarse_spacing_um": 1000.0,
    "fine_spacing_um": 500.0,
    "coarse_padding_um": 4000.0,
    "fine_padding_um": 2000.0,
    "smoothing_sigma_knots": 1.0,
    "coarse_weight": 0.75,
    "fine_weight": 0.25,
    "a0_um": 125.0,
    "global_log_scale_half_range": 0.03,
    "integration_steps": 8,
    "local_jacobian_det_min": 0.50,
    "local_jacobian_det_max": 2.00,
    "composed_jacobian_det_floor": 0.25,
    "cycle_max_um": 2.5,
    "max_local_displacement_um": 750.0,
    "component_derivative_abs_max": 1.0,
    "gradient_frobenius_bound_max": 2.0,
    "divergence_abs_bound_max": 1.5,
    "speed_l2_bound_um_max": 750.0,
    "minimum_halo_um": 100.0,
    "post_float32_affine_residual_max": 1.0e-6,
})
_METRIC_KEYS = frozenset({
    "normalized_scalar_mae",
    "normalized_scalar_absolute_error_p99",
    "support_mass_mae",
    "support_mass_absolute_error_p99",
    "slab_label_purity_mae",
    "slab_label_purity_absolute_error_p99",
    "centre_label_support_weight_mae",
    "centre_label_support_weight_absolute_error_p99",
    "dense_correspondence_weight_mae",
    "dense_correspondence_weight_absolute_error_p99",
    "slab_modal_annotation_disagreement_fraction",
    "slab_observable_support_mask_disagreement_fraction",
    "dense_correspondence_abstention_disagreement_fraction",
})
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "subject_deformed_slab_qualification_v2.py",
    "arbitrary_plane_support_resolution_v2.py",
    "arbitrary_plane_subject_slab_v2.py",
    "arbitrary_plane_subject_section_v2.py",
    "arbitrary_plane_subject_deformation_v2.py",
    "arbitrary_plane_synthetic_generator_v2.py",
    "arbitrary_plane_acquisition_v2.py",
)


def subject_deformed_slab_qualification_panel_v2() -> dict[str, object]:
    animals = [
        {
            "animal_index": row[0],
            "animal_id": row[1],
            "specimen_id": row[2],
            "experiment_id": row[3],
            "subject_deformation_root_seed_uint64": row[4],
        }
        for row in _ANIMAL_ROWS
    ]
    cases = [
        {
            "case_id": row[0],
            "animal_index": row[1],
            "split_index": row[2],
            "section_index": row[3],
            "plane_stratum": row[4],
            "support_master_root_seed_uint64": row[5],
        }
        for row in _CASE_ROWS
    ]
    return {
        "split": "development",
        "role": "small predeclared engineering qualification; not a benchmark",
        "animals": animals,
        "cases": cases,
    }


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _runner_source_sha256() -> str:
    return acquisition._normalized_text_sha256(
        _SOURCE_ROOT / "run_subject_deformed_slab_qualification_v2.py"
    )


def _contains_final_id(value: object) -> bool:
    if isinstance(value, Mapping):
        return "synthetic_realization_id" in value or any(
            _contains_final_id(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_final_id(item) for item in value)
    return False


def _receipt_binding(
    receipt_payload: Mapping[str, object], expected_sha256: str | None = None
) -> dict[str, object]:
    payload = acquisition._json_value(receipt_payload)
    digest = acquisition._payload_sha256(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("live receipt payload does not match its stored hash")
    return {"receipt_payload": payload, "receipt_sha256": digest}


def _inner_receipt_binding(
    public_receipt: Mapping[str, object], expected_sha256: str
) -> dict[str, object]:
    receipt = acquisition._json_value(public_receipt)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != expected_sha256:
        raise ValueError("inner receipt hash does not match its artifact")
    return _receipt_binding(receipt, expected_sha256)


def _context_reference(prepared_context: Mapping[str, object]) -> dict[str, object]:
    binding = _receipt_binding(prepared_context["receipt"])
    if binding["receipt_sha256"] != prepared_context["v2_context_sha256"]:
        raise ValueError("prepared context receipt changed")
    return {
        "v2_context_sha256": prepared_context["v2_context_sha256"],
        "receipt": binding,
    }


def _context_bounds(prepared_context: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    support = acquisition._context_support(prepared_context)
    lower = np.asarray(support["origin_um"], dtype=np.float64)
    upper = lower + np.asarray(support["annotation_shape"], dtype=np.float64) * np.asarray(
        support["voxel_size_um"], dtype=np.float64
    )
    return lower, upper


def _qualification_configuration() -> dict[str, object]:
    return {
        "coarse_axial_step_um_max": COARSE_AXIAL_STEP_UM_MAX,
        "refined_axial_step_um_max": REFINED_AXIAL_STEP_UM_MAX,
        "nominal_cut_thickness_um": NOMINAL_CUT_THICKNESS_UM,
        "maximum_support_attempts": MAX_SUPPORT_ATTEMPTS,
        "parent_shape_h_w": [256, 256],
        "subject_deformation_parameters": acquisition._json_value(
            _DEFORMATION_PARAMETERS
        ),
        "thresholds": acquisition._json_value(THRESHOLDS),
    }


def _metric_pass(metrics: Mapping[str, float]) -> bool:
    return bool(
        set(metrics) == _METRIC_KEYS
        and all(math.isfinite(float(value)) for value in metrics.values())
        and all(
            value <= THRESHOLDS["continuous_mean_absolute_error_max"]
            for name, value in metrics.items()
            if name.endswith("_mae")
        )
        and all(
            value <= THRESHOLDS["continuous_absolute_error_p99_max"]
            for name, value in metrics.items()
            if name.endswith("_p99")
        )
        and all(
            value <= THRESHOLDS["categorical_disagreement_fraction_max"]
            for name, value in metrics.items()
            if name.endswith("_disagreement_fraction")
        )
    )


def _comparison_metrics(
    coarse_raster: Mapping[str, object],
    refined_raster: Mapping[str, object],
    scalar_range: float,
) -> tuple[dict[str, float], int]:
    coarse_support = np.asarray(coarse_raster["slab_brain_occupancy"], np.float64)
    refined_support = np.asarray(refined_raster["slab_brain_occupancy"], np.float64)
    union = (coarse_support > 0.0) | (refined_support > 0.0)
    if not union.any() or not math.isfinite(scalar_range) or scalar_range <= 0.0:
        raise ValueError("subject slab comparison has no support or scalar range")

    def errors(name: str, denominator: float = 1.0) -> tuple[float, float]:
        values = np.abs(
            np.asarray(coarse_raster[name], np.float64)[union]
            - np.asarray(refined_raster[name], np.float64)[union]
        ) / denominator
        return float(values.mean()), float(np.quantile(values, 0.99, method="linear"))

    scalar_mae, scalar_p99 = errors("scalar", scalar_range)
    support_mae, support_p99 = errors("slab_brain_occupancy")
    purity_mae, purity_p99 = errors("slab_label_purity")
    centre_mae, centre_p99 = errors("centre_label_support_weight")
    coarse_dense = coarse_raster["slab_supervision_weight_or_abstention"]
    refined_dense = refined_raster["slab_supervision_weight_or_abstention"]
    dense_error = np.abs(
        np.asarray(coarse_dense["dense_correspondence_weight"], np.float64)[union]
        - np.asarray(refined_dense["dense_correspondence_weight"], np.float64)[union]
    )
    metrics = {
        "normalized_scalar_mae": scalar_mae,
        "normalized_scalar_absolute_error_p99": scalar_p99,
        "support_mass_mae": support_mae,
        "support_mass_absolute_error_p99": support_p99,
        "slab_label_purity_mae": purity_mae,
        "slab_label_purity_absolute_error_p99": purity_p99,
        "centre_label_support_weight_mae": centre_mae,
        "centre_label_support_weight_absolute_error_p99": centre_p99,
        "dense_correspondence_weight_mae": float(dense_error.mean()),
        "dense_correspondence_weight_absolute_error_p99": float(
            np.quantile(dense_error, 0.99, method="linear")
        ),
        "slab_modal_annotation_disagreement_fraction": float(
            np.mean(
                np.asarray(coarse_raster["slab_modal_annotation"])[union]
                != np.asarray(refined_raster["slab_modal_annotation"])[union]
            )
        ),
        "slab_observable_support_mask_disagreement_fraction": float(
            np.mean(
                np.asarray(coarse_raster["slab_observable_support_mask"])[union]
                != np.asarray(refined_raster["slab_observable_support_mask"])[union]
            )
        ),
        "dense_correspondence_abstention_disagreement_fraction": float(
            np.mean(
                np.asarray(coarse_dense["abstention_mask"])[union]
                != np.asarray(refined_dense["abstention_mask"])[union]
            )
        ),
    }
    return metrics, int(union.sum())


def _centre_invariance(
    coarse: Mapping[str, object], refined: Mapping[str, object]
) -> dict[str, object]:
    def arrays(
        stage: Mapping[str, object],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        centre = int(stage["coordinate_map"]["kernel"]["centre_index"])
        return (
            np.asarray(
                stage["coordinate_map"]["arrays"][
                    "mapped_allen_index_coordinates_float32"
                ][centre]
            ),
            np.asarray(
                stage["coordinate_map"]["arrays"][
                    "mapped_ccf_physical_coordinates_ap_dv_ml_um_float64"
                ][centre]
            ),
            np.asarray(stage["sample_arrays"]["annotation_samples_int64"][centre]),
            np.asarray(stage["raster"]["centre_plane_support_mask"]),
        )

    coarse_arrays = arrays(coarse)
    refined_arrays = arrays(refined)
    names = (
        "mapped_allen_coordinates",
        "mapped_physical_coordinates",
        "annotations",
        "support",
    )
    result = {}
    all_exact = True
    for name, left, right in zip(names, coarse_arrays, refined_arrays):
        exact = bool(np.array_equal(left, right))
        result[name] = {
            "coarse_receipt": acquisition._array_receipt(left),
            "refined_receipt": acquisition._array_receipt(right),
            "byte_identical": exact,
        }
        all_exact &= exact
    result["all_centre_targets_byte_identical"] = all_exact
    return result


def _make_subject_plan_with_mapper(
    prepared_context: Mapping[str, object], animal: Mapping[str, object]
) -> tuple[Mapping[str, object], object]:
    lower, upper = _context_bounds(prepared_context)
    plan = deformation.sample_animal_subject_deformation_plan_v2(
        lower,
        upper,
        root_seed=animal["subject_deformation_root_seed_uint64"],
        split="development",
        animal_index=animal["animal_index"],
        animal_id=animal["animal_id"],
        ccf_context_sha256=prepared_context["v2_context_sha256"],
        **_DEFORMATION_PARAMETERS,
    )
    subject_to_ccf_mapper = deformation._verified_subject_to_ccf_mapper_v2(
        plan,
        expected_ccf_context_sha256=prepared_context["v2_context_sha256"],
        expected_full_ccf_lower_um=lower,
        expected_full_ccf_upper_um=upper,
    )
    return plan, subject_to_ccf_mapper


def _make_subject_plan(
    prepared_context: Mapping[str, object], animal: Mapping[str, object]
) -> Mapping[str, object]:
    plan, _ = _make_subject_plan_with_mapper(prepared_context, animal)
    return plan


def _animal_record(
    animal: Mapping[str, object], plan: Mapping[str, object]
) -> dict[str, object]:
    receipt = _inner_receipt_binding(
        deformation.subject_deformation_plan_receipt_v2(plan),
        plan["receipt_sha256"],
    )
    return {
        "animal_manifest": acquisition._json_value(animal),
        "subject_deformation_plan_id": plan["subject_deformation_plan_id"],
        "subject_deformation_realization_id": plan[
            "subject_deformation_realization_id"
        ],
        "synthetic_animal_id": plan["synthetic_animal_id"],
        "resolved_config": acquisition._json_value(plan["resolved_config"]),
        "rng_sources": acquisition._json_value(plan["rng_sources"]),
        "receipt": receipt,
    }


def _verify_support_resolution(
    bundle: Mapping[str, object],
    prepared_context: Mapping[str, object],
    subject_plan: Mapping[str, object],
    subject_to_ccf_mapper,
    *,
    batch_size: int | None,
) -> None:
    resolution = bundle["resolution"]
    config = resolution["configuration"]
    lineage = resolution["lineage"]
    support_resolution._verify_subject_support_resolution_with_mapper_v2(
        bundle,
        prepared_context,
        subject_plan=subject_plan,
        master_root_seed=config["master_root_seed_uint64"],
        split=lineage["split"],
        split_index=config["split_index"],
        animal_index=config["animal_index"],
        animal_id=lineage["animal_id"],
        section_index=config["section_index"],
        plane_stratum=config["plane_stratum"],
        nominal_cut_thickness_um=config["nominal_cut_thickness_um"],
        specimen_id=lineage["specimen_id"],
        experiment_id=lineage["experiment_id"],
        axial_step_um_max=config["axial_step_um_max"],
        parent_shape_h_w=tuple(config["parent_shape_h_w"]),
        max_attempts=config["max_attempts"],
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )


def _artifact_reference(
    artifact: Mapping[str, object], receipt_payload: Mapping[str, object], id_names: tuple[str, ...]
) -> dict[str, object]:
    return {
        **{name: artifact[name] for name in id_names},
        "receipt": _receipt_binding(receipt_payload, artifact["receipt_sha256"]),
    }


def _finalize_case(payload: dict[str, object]) -> dict[str, object]:
    payload["passed"] = bool(
        payload["centre_invariance"]["all_centre_targets_byte_identical"]
        and payload["same_pose_attempt"]
        and _metric_pass(payload["metrics"])
    )
    payload["case_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def _evaluate_case(
    prepared_context: Mapping[str, object],
    animal: Mapping[str, object],
    subject_plan: Mapping[str, object],
    subject_to_ccf_mapper,
    case_spec: Mapping[str, object],
    *,
    batch_size: int | None,
) -> dict[str, object]:
    bundle = support_resolution._resolve_subject_support_with_mapper_v2(
        prepared_context,
        subject_plan=subject_plan,
        master_root_seed=case_spec["support_master_root_seed_uint64"],
        split="development",
        split_index=case_spec["split_index"],
        animal_index=animal["animal_index"],
        animal_id=animal["animal_id"],
        section_index=case_spec["section_index"],
        plane_stratum=case_spec["plane_stratum"],
        nominal_cut_thickness_um=NOMINAL_CUT_THICKNESS_UM,
        specimen_id=animal["specimen_id"],
        experiment_id=animal["experiment_id"],
        axial_step_um_max=COARSE_AXIAL_STEP_UM_MAX,
        max_attempts=MAX_SUPPORT_ATTEMPTS,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    _verify_support_resolution(
        bundle,
        prepared_context,
        subject_plan,
        subject_to_ccf_mapper,
        batch_size=batch_size,
    )
    resolution = bundle["resolution"]
    if resolution["status"] != "accepted":
        raise ValueError("predeclared subject-deformed slab case exhausted support attempts")
    coarse_precursor = bundle["accepted_precursor"]
    attempt = resolution["attempts"][resolution["accepted_attempt_index"]]
    config = resolution["configuration"]
    lineage = resolution["lineage"]
    refined_precursor = slab.make_v2_generic_global_reference_slab_render(
        prepared_context,
        lineage["split"],
        attempt["attempt_seed"]["attempt_root_seed_uint64"],
        config["section_index"],
        config["plane_stratum"],
        nominal_cut_thickness_um=config["nominal_cut_thickness_um"],
        axial_step_um_max=REFINED_AXIAL_STEP_UM_MAX,
        parent_shape_h_w=tuple(config["parent_shape_h_w"]),
        animal_id=lineage["animal_id"],
        animal_index=lineage["animal_index"],
        specimen_id=lineage["specimen_id"],
        experiment_id=lineage["experiment_id"],
    )
    slab.verify_v2_generic_global_reference_slab_render(
        refined_precursor, prepared_context
    )
    same_pose = bool(
        coarse_precursor["v2_plane_realization_id"]
        == refined_precursor["v2_plane_realization_id"]
        and coarse_precursor["centre_plane_render_id"]
        == refined_precursor["centre_plane_render_id"]
    )
    if not same_pose:
        raise ValueError("coarse and refined slabs do not share the accepted pose attempt")
    coarse = subject_slab._make_subject_slab_render_with_mapper_v2(
        prepared_context,
        coarse_precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    refined = subject_slab._make_subject_slab_render_with_mapper_v2(
        prepared_context,
        refined_precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    support_resolution.verify_accepted_subject_slab_matches_support_resolution_v2(
        bundle, coarse
    )
    for stage, precursor in ((coarse, coarse_precursor), (refined, refined_precursor)):
        subject_slab._verify_subject_slab_render_with_mapper_v2(
            stage,
            prepared_context,
            precursor,
            subject_plan=subject_plan,
            batch_size=batch_size,
            subject_to_ccf_mapper=subject_to_ccf_mapper,
        )
    centre = _centre_invariance(coarse, refined)
    if not centre["all_centre_targets_byte_identical"]:
        raise ValueError("axial refinement changed an immutable centre target")
    scalar_tensor = prepared_context["opaque_v1_context"]["scalar_tensor"]
    scalar_range = max(
        float(scalar_tensor.max().item() - scalar_tensor.min().item()), 1.0
    )
    metrics, union_count = _comparison_metrics(
        coarse["raster"], refined["raster"], scalar_range
    )
    support_receipt = support_resolution.subject_support_resolution_receipt_v2(
        resolution
    )
    payload = {
        "schema_version": SUBJECT_DEFORMED_SLAB_CASE_V2_SCHEMA,
        "case_spec": acquisition._json_value(case_spec),
        "animal_reference": {
            "animal_id": animal["animal_id"],
            "animal_index": animal["animal_index"],
            "specimen_id": animal["specimen_id"],
            "experiment_id": animal["experiment_id"],
            "subject_deformation_plan_id": subject_plan[
                "subject_deformation_plan_id"
            ],
            "subject_deformation_realization_id": subject_plan[
                "subject_deformation_realization_id"
            ],
            "synthetic_animal_id": subject_plan["synthetic_animal_id"],
        },
        "support_resolution_reference": {
            "support_resolution_plan_id": resolution[
                "support_resolution_plan_id"
            ],
            "subject_support_resolution_id": resolution[
                "subject_support_resolution_id"
            ],
            "accepted_attempt_index": resolution["accepted_attempt_index"],
            "configuration": acquisition._json_value(config),
            "lineage": acquisition._json_value(lineage),
            "accepted_attempt_seed": acquisition._json_value(attempt["attempt_seed"]),
            "accepted_precursor_reference": acquisition._json_value(
                resolution["accepted_precursor_reference"]
            ),
            "accepted_probe_reference": acquisition._json_value(
                resolution["accepted_probe_reference"]
            ),
            "receipt": _receipt_binding(
                support_receipt, resolution["receipt_sha256"]
            ),
        },
        "precursor_references": {
            "coarse": _artifact_reference(
                coarse_precursor,
                slab.v2_generic_slab_render_receipt(coarse_precursor),
                ("v2_plane_realization_id", "centre_plane_render_id", "slab_render_id"),
            ),
            "refined": _artifact_reference(
                refined_precursor,
                slab.v2_generic_slab_render_receipt(refined_precursor),
                ("v2_plane_realization_id", "centre_plane_render_id", "slab_render_id"),
            ),
        },
        "subject_slab_references": {
            "coarse": _artifact_reference(
                coarse,
                subject_slab.subject_slab_render_receipt_v2(coarse),
                ("subject_coordinate_map_id", "subject_slab_render_id"),
            ),
            "refined": _artifact_reference(
                refined,
                subject_slab.subject_slab_render_receipt_v2(refined),
                ("subject_coordinate_map_id", "subject_slab_render_id"),
            ),
        },
        "same_pose_attempt": same_pose,
        "axial_steps_um_max": {
            "coarse": COARSE_AXIAL_STEP_UM_MAX,
            "refined": REFINED_AXIAL_STEP_UM_MAX,
        },
        "centre_invariance": centre,
        "union_nonzero_support_pixel_count": union_count,
        "authenticated_scalar_range_denominator": scalar_range,
        "metrics": metrics,
        "thresholds": acquisition._json_value(THRESHOLDS),
    }
    return _finalize_case(payload)


def _run_predeclared_panel_v2(
    prepared_context: Mapping[str, object], *, batch_size: int | None
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    panel = subject_deformed_slab_qualification_panel_v2()
    animals = {animal["animal_index"]: animal for animal in panel["animals"]}
    plans_and_mappers = {
        index: _make_subject_plan_with_mapper(prepared_context, animal)
        for index, animal in animals.items()
    }
    plans = {index: value[0] for index, value in plans_and_mappers.items()}
    subject_to_ccf_mappers = {
        index: value[1] for index, value in plans_and_mappers.items()
    }
    animal_records = [
        _animal_record(animal, plans[animal["animal_index"]])
        for animal in panel["animals"]
    ]
    cases = [
        _evaluate_case(
            prepared_context,
            animals[case["animal_index"]],
            plans[case["animal_index"]],
            subject_to_ccf_mappers[case["animal_index"]],
            case,
            batch_size=batch_size,
        )
        for case in panel["cases"]
    ]
    return animal_records, cases


def _assemble_report(
    prepared_context: Mapping[str, object],
    animals: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    payload = {
        "schema_version": SUBJECT_DEFORMED_SLAB_QUALIFICATION_V2_SCHEMA,
        "claim_scope": (
            "small subject-deformed development qualification of 12.5-to-6.25-um "
            "finite-slab quadrature only; no model scoring, benchmark, or final animals"
        ),
        "cohort_policy": {
            "development_only": True,
            "benchmark_animals_used": False,
            "final_test_animals_used": False,
            "full_benchmark": False,
        },
        "panel": subject_deformed_slab_qualification_panel_v2(),
        "configuration": _qualification_configuration(),
        "context_reference": _context_reference(prepared_context),
        "animals": animals,
        "cases": cases,
        "animal_count": len(animals),
        "case_count": len(cases),
        "all_cases_passed": bool(cases and all(case["passed"] for case in cases)),
        "learned_dependencies": {
            "learned_checkpoint_dependencies": [],
            "previous_model_dependencies": [],
            "pretrained_feature_dependencies": [],
            "learned_style_model_dependencies": [],
        },
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": (
            acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        ),
        "runner_source_sha256": _runner_source_sha256(),
    }
    if _contains_final_id(payload):
        raise ValueError("qualification input issued a premature synthetic realization ID")
    payload["qualification_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def evaluate_subject_deformed_slab_qualification_v2(
    prepared_context: Mapping[str, object], *, batch_size: int | None = None
) -> dict[str, object]:
    """Evaluate the frozen six-case/two-animal development panel."""
    acquisition._validate_v2_context(prepared_context)
    animals, cases = _run_predeclared_panel_v2(
        prepared_context, batch_size=batch_size
    )
    return _assemble_report(prepared_context, animals, cases)


def _evaluate_subject_deformed_slab_qualification_with_capability_v2(
    prepared_context: Mapping[str, object], *, batch_size: int | None = None
):
    report = acquisition._freeze_value(
        evaluate_subject_deformed_slab_qualification_v2(
            prepared_context, batch_size=batch_size
        )
    )

    def require_exact_live_report(candidate: Mapping[str, object]):
        if candidate is not report:
            raise ValueError("live qualification capability does not match the exact report")
        return report

    return report, require_exact_live_report


_REPORT_KEYS = {
    "schema_version",
    "claim_scope",
    "cohort_policy",
    "panel",
    "configuration",
    "context_reference",
    "animals",
    "cases",
    "animal_count",
    "case_count",
    "all_cases_passed",
    "learned_dependencies",
    "implementation_source_sha256",
    "implementation_source_sha256_canonicalization",
    "runner_source_sha256",
    "qualification_receipt_sha256",
}
_ANIMAL_KEYS = {
    "animal_manifest",
    "subject_deformation_plan_id",
    "subject_deformation_realization_id",
    "synthetic_animal_id",
    "resolved_config",
    "rng_sources",
    "receipt",
}
_CASE_KEYS = {
    "schema_version",
    "case_spec",
    "animal_reference",
    "support_resolution_reference",
    "precursor_references",
    "subject_slab_references",
    "same_pose_attempt",
    "axial_steps_um_max",
    "centre_invariance",
    "union_nonzero_support_pixel_count",
    "authenticated_scalar_range_denominator",
    "metrics",
    "thresholds",
    "passed",
    "case_receipt_sha256",
}


def _verify_binding(binding: Mapping[str, object]) -> bool:
    return bool(
        set(binding) == {"receipt_payload", "receipt_sha256"}
        and binding["receipt_sha256"]
        == acquisition._payload_sha256(
            acquisition._json_value(binding["receipt_payload"])
        )
    )


def _case_semantics(case: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    centre = case.get("centre_invariance", {})
    centre_names = (
        "mapped_allen_coordinates",
        "mapped_physical_coordinates",
        "annotations",
        "support",
    )
    centre_valid = (
        set(centre) == {*centre_names, "all_centre_targets_byte_identical"}
        and centre.get("all_centre_targets_byte_identical") is True
        and all(
            set(centre.get(name, {}))
            == {"coarse_receipt", "refined_receipt", "byte_identical"}
            and centre[name]["byte_identical"] is True
            and centre[name]["coarse_receipt"] == centre[name]["refined_receipt"]
            for name in centre_names
        )
    )
    bindings = [case.get("support_resolution_reference", {}).get("receipt", {})]
    for group in ("precursor_references", "subject_slab_references"):
        bindings.extend(
            case.get(group, {}).get(level, {}).get("receipt", {})
            for level in ("coarse", "refined")
        )
    payload = {
        key: value for key, value in case.items() if key != "case_receipt_sha256"
    }
    metrics = case.get("metrics", {})
    return bool(
        set(case) == _CASE_KEYS
        and case.get("schema_version") == SUBJECT_DEFORMED_SLAB_CASE_V2_SCHEMA
        and case.get("case_spec") == expected
        and case.get("same_pose_attempt") is True
        and case.get("axial_steps_um_max")
        == {"coarse": COARSE_AXIAL_STEP_UM_MAX, "refined": REFINED_AXIAL_STEP_UM_MAX}
        and centre_valid
        and case.get("union_nonzero_support_pixel_count", 0) > 0
        and case.get("authenticated_scalar_range_denominator", 0.0) > 0.0
        and case.get("thresholds") == THRESHOLDS
        and _metric_pass(metrics)
        and case.get("passed") is True
        and all(_verify_binding(binding) for binding in bindings)
        and case.get("case_receipt_sha256") == acquisition._payload_sha256(payload)
    )


def _case_lineage_semantics(
    case: Mapping[str, object],
    expected: Mapping[str, object],
    animal: Mapping[str, object],
    animal_record: Mapping[str, object],
) -> bool:
    reference = case.get("animal_reference", {})
    support = case.get("support_resolution_reference", {})
    config = support.get("configuration", {})
    lineage = support.get("lineage", {})
    seed = support.get("accepted_attempt_seed", {})
    precursors = case.get("precursor_references", {})
    subject_slabs = case.get("subject_slab_references", {})
    coarse_precursor = precursors.get("coarse", {})
    refined_precursor = precursors.get("refined", {})
    coarse_precursor_payload = coarse_precursor.get("receipt", {}).get(
        "receipt_payload", {}
    )
    refined_precursor_payload = refined_precursor.get("receipt", {}).get(
        "receipt_payload", {}
    )
    coarse_precursor_config = coarse_precursor_payload.get("generator", {}).get(
        "resolved_config", {}
    )
    refined_precursor_config = refined_precursor_payload.get("generator", {}).get(
        "resolved_config", {}
    )
    coarse_slab_coordinate = (
        subject_slabs.get("coarse", {})
        .get("receipt", {})
        .get("receipt_payload", {})
        .get("coordinate_identity_payload", {})
    )
    refined_slab_coordinate = (
        subject_slabs.get("refined", {})
        .get("receipt", {})
        .get("receipt_payload", {})
        .get("coordinate_identity_payload", {})
    )
    return bool(
        reference
        == {
            "animal_id": animal["animal_id"],
            "animal_index": animal["animal_index"],
            "specimen_id": animal["specimen_id"],
            "experiment_id": animal["experiment_id"],
            "subject_deformation_plan_id": animal_record[
                "subject_deformation_plan_id"
            ],
            "subject_deformation_realization_id": animal_record[
                "subject_deformation_realization_id"
            ],
            "synthetic_animal_id": animal_record["synthetic_animal_id"],
        }
        and config.get("split_index") == expected["split_index"]
        and config.get("animal_index") == expected["animal_index"]
        and config.get("section_index") == expected["section_index"]
        and config.get("plane_stratum") == expected["plane_stratum"]
        and config.get("nominal_cut_thickness_um") == NOMINAL_CUT_THICKNESS_UM
        and config.get("axial_step_um_max") == COARSE_AXIAL_STEP_UM_MAX
        and config.get("parent_shape_h_w") == [256, 256]
        and config.get("max_attempts") == MAX_SUPPORT_ATTEMPTS
        and lineage
        == {
            "split": "development",
            "animal_id": animal["animal_id"],
            "animal_index": animal["animal_index"],
            "specimen_id": animal["specimen_id"],
            "experiment_id": animal["experiment_id"],
        }
        and seed.get("master_root_seed_uint64")
        == expected["support_master_root_seed_uint64"]
        and seed.get("split_index") == expected["split_index"]
        and seed.get("animal_index") == expected["animal_index"]
        and seed.get("section_index") == expected["section_index"]
        and seed.get("attempt_index") == support.get("accepted_attempt_index")
        and bool(seed.get("attempt_root_seed_uint64"))
        and isinstance(support.get("accepted_attempt_index"), int)
        and support["accepted_attempt_index"] >= 0
        and support.get("accepted_precursor_reference", {}).get("slab_render_id")
        == coarse_precursor.get("slab_render_id")
        and support.get("accepted_precursor_reference", {}).get("receipt_sha256")
        == coarse_precursor.get("receipt", {}).get("receipt_sha256")
        and coarse_precursor.get("v2_plane_realization_id")
        == refined_precursor.get("v2_plane_realization_id")
        and coarse_precursor.get("centre_plane_render_id")
        == refined_precursor.get("centre_plane_render_id")
        and coarse_precursor_config.get("root_seed_uint64")
        == seed["attempt_root_seed_uint64"]
        and refined_precursor_config.get("root_seed_uint64")
        == seed["attempt_root_seed_uint64"]
        and coarse_precursor_config.get("sample_index") == expected["section_index"]
        and refined_precursor_config.get("sample_index") == expected["section_index"]
        and coarse_precursor_config.get("plane_stratum")
        == expected["plane_stratum"]
        and refined_precursor_config.get("plane_stratum")
        == expected["plane_stratum"]
        and coarse_precursor_payload.get("slab_recipe", {}).get(
            "axial_step_um_max"
        )
        == COARSE_AXIAL_STEP_UM_MAX
        and refined_precursor_payload.get("slab_recipe", {}).get(
            "axial_step_um_max"
        )
        == REFINED_AXIAL_STEP_UM_MAX
        and coarse_slab_coordinate.get("deformation_reference")
        == refined_slab_coordinate.get("deformation_reference")
        and coarse_slab_coordinate.get("synthetic_animal_id")
        == animal_record["synthetic_animal_id"]
        and refined_slab_coordinate.get("synthetic_animal_id")
        == animal_record["synthetic_animal_id"]
    )


def verify_subject_deformed_slab_qualification_v2(
    report: Mapping[str, object],
    prepared_context: Mapping[str, object],
    *,
    batch_size: int | None = None,
) -> None:
    """Strictly verify source, provenance, raw metrics, receipts, and replay."""
    acquisition._validate_v2_context(prepared_context)
    panel = subject_deformed_slab_qualification_panel_v2()
    animals = report.get("animals", [])
    cases = report.get("cases", [])
    payload = {
        key: value
        for key, value in report.items()
        if key != "qualification_receipt_sha256"
    }
    animal_semantics = bool(
        isinstance(animals, list)
        and len(animals) == len(panel["animals"])
        and all(
            set(record) == _ANIMAL_KEYS
            and record["animal_manifest"] == expected
            and _verify_binding(record["receipt"])
            for record, expected in zip(animals, panel["animals"])
        )
    )
    case_semantics = bool(
        isinstance(cases, list)
        and len(cases) == len(panel["cases"])
        and len({record["synthetic_animal_id"] for record in animals}) == 2
        and all(
            _case_semantics(case, expected)
            and _case_lineage_semantics(
                case,
                expected,
                next(
                    animal
                    for animal in panel["animals"]
                    if animal["animal_index"] == expected["animal_index"]
                ),
                next(
                    record
                    for record in animals
                    if record["animal_manifest"]["animal_index"]
                    == expected["animal_index"]
                ),
            )
            for case, expected in zip(cases, panel["cases"])
        )
    )
    if (
        set(report) != _REPORT_KEYS
        or report.get("schema_version")
        != SUBJECT_DEFORMED_SLAB_QUALIFICATION_V2_SCHEMA
        or report.get("claim_scope")
        != (
            "small subject-deformed development qualification of 12.5-to-6.25-um "
            "finite-slab quadrature only; no model scoring, benchmark, or final animals"
        )
        or report.get("cohort_policy")
        != {
            "development_only": True,
            "benchmark_animals_used": False,
            "final_test_animals_used": False,
            "full_benchmark": False,
        }
        or report.get("panel") != panel
        or report.get("configuration") != _qualification_configuration()
        or report.get("context_reference") != _context_reference(prepared_context)
        or report.get("animal_count") != 2
        or report.get("case_count") != 6
        or not animal_semantics
        or not case_semantics
        or report.get("all_cases_passed") is not True
        or any(report.get("learned_dependencies", {}).values())
        or set(report.get("learned_dependencies", {}))
        != {
            "learned_checkpoint_dependencies",
            "previous_model_dependencies",
            "pretrained_feature_dependencies",
            "learned_style_model_dependencies",
        }
        or report.get("implementation_source_sha256") != _source_hashes()
        or report.get("implementation_source_sha256_canonicalization")
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or report.get("runner_source_sha256") != _runner_source_sha256()
        or report.get("qualification_receipt_sha256")
        != acquisition._payload_sha256(payload)
        or _contains_final_id(report)
    ):
        raise ValueError("subject-deformed slab qualification is invalid or did not pass")
    replay = evaluate_subject_deformed_slab_qualification_v2(
        prepared_context, batch_size=batch_size
    )
    if acquisition._canonical_json(report) != acquisition._canonical_json(replay):
        raise ValueError("subject-deformed slab qualification replay does not match")


def save_subject_deformed_slab_qualification_v2(
    path: str | Path, report: Mapping[str, object]
) -> None:
    payload = {
        key: value
        for key, value in report.items()
        if key != "qualification_receipt_sha256"
    }
    if report.get("qualification_receipt_sha256") != acquisition._payload_sha256(
        payload
    ):
        raise ValueError("subject-deformed slab qualification receipt does not match")
    Path(path).write_text(
        acquisition._canonical_json(report) + "\n", encoding="utf-8", newline="\n"
    )
