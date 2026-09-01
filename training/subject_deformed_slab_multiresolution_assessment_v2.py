"""Fixed-case, threshold-free multiresolution finite-slab assessment."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_subject_deformation_v2 as deformation
import training.arbitrary_plane_subject_slab_v2 as subject_slab
import training.arbitrary_plane_synthetic_generator_v2 as slab
import training.slab_refinement_gate_status_v2 as gate_status
import training.subject_deformed_slab_qualification_v2 as legacy


PLAN_SCHEMA = "anatomy-tracker.subject-slab-fixed-case-multiresolution-plan/v2"
ASSESSMENT_SCHEMA = "anatomy-tracker.subject-slab-fixed-case-multiresolution-assessment/v2"
AXIAL_STEPS_UM_MAX = (12.5, 6.25, 3.125, 1.5625)
ARM_NAMES = ("same_nonidentity_subject_deformation", "identity_control")
STRATUM_NAMES = ("stable_interior", "support_boundary", "atlas_label_boundary")
METRIC_FAMILIES = (
    "smooth_scalar",
    "occupancy_and_label_mass",
    "dense_correspondence",
    "categorical",
)
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "subject_deformed_slab_multiresolution_assessment_v2.py",
    "subject_deformed_slab_qualification_v2.py",
    "arbitrary_plane_subject_slab_v2.py",
    "arbitrary_plane_subject_deformation_v2.py",
    "arbitrary_plane_synthetic_generator_v2.py",
    "arbitrary_plane_acquisition_v2.py",
    "slab_refinement_gate_status_v2.py",
)


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _step_key(step: float) -> str:
    return f"{step:g}"


def _receipt_matches(payload: Mapping[str, object], receipt_name: str) -> bool:
    return payload.get(receipt_name) == acquisition._payload_sha256(
        acquisition._json_value(
            {key: value for key, value in payload.items() if key != receipt_name}
        )
    )


def _validate_failed_legacy_report_structure_v2(
    report: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> None:
    acquisition._validate_v2_context(prepared_context)
    if (
        report.get("context_reference", {}).get("v2_context_sha256")
        != prepared_context["v2_context_sha256"]
        or report.get("all_cases_passed") is not False
        or not _receipt_matches(report, "qualification_receipt_sha256")
        or not any(case.get("passed") is False for case in report.get("cases", []))
    ):
        raise ValueError("failed subject-slab report is not authentic rejected evidence")


def _verify_failed_legacy_report(
    report: Mapping[str, object],
    prepared_context: Mapping[str, object],
    *,
    batch_size: int | None,
) -> None:
    _validate_failed_legacy_report_structure_v2(report, prepared_context)
    replay = legacy.evaluate_subject_deformed_slab_qualification_v2(
        prepared_context, batch_size=batch_size
    )
    if acquisition._canonical_json(report) != acquisition._canonical_json(replay):
        raise ValueError("failed subject-slab report does not replay exactly")


def _make_fixed_case_multiresolution_plan_from_validated_report_v2(
    failed_report: Mapping[str, object],
) -> dict[str, object]:
    selected_index, selected = next(
        (index, case)
        for index, case in enumerate(failed_report["cases"])
        if case["passed"] is False
    )
    animal_index = selected["case_spec"]["animal_index"]
    animal_record = next(
        record
        for record in failed_report["animals"]
        if record["animal_manifest"]["animal_index"] == animal_index
    )
    coarse = selected["precursor_references"]["coarse"]
    refined = selected["precursor_references"]["refined"]
    if (
        coarse["v2_plane_realization_id"] != refined["v2_plane_realization_id"]
        or coarse["centre_plane_render_id"] != refined["centre_plane_render_id"]
        or selected["same_pose_attempt"] is not True
        or any(case["passed"] is not True for case in failed_report["cases"][:selected_index])
    ):
        raise ValueError("selected case is not the exact first failed pose attempt")
    payload = {
        "schema_version": PLAN_SCHEMA,
        "role": "fixed-case numerical assessment only; not a qualification or benchmark",
        "rejected_legacy_gate_contract": gate_status.legacy_gate_contract(),
        "failed_report_reference": {
            "schema_version": failed_report["schema_version"],
            "v2_context_sha256": failed_report["context_reference"][
                "v2_context_sha256"
            ],
            "qualification_receipt_sha256": failed_report[
                "qualification_receipt_sha256"
            ],
            "case_count": failed_report["case_count"],
        },
        "selected_first_failure": {
            "case_order_index": selected_index,
            "case_receipt_sha256": selected["case_receipt_sha256"],
            "case_spec": acquisition._json_value(selected["case_spec"]),
            "animal_reference": acquisition._json_value(
                selected["animal_reference"]
            ),
            "animal_manifest": acquisition._json_value(
                animal_record["animal_manifest"]
            ),
            "subject_deformation_reference": {
                "subject_deformation_plan_id": animal_record[
                    "subject_deformation_plan_id"
                ],
                "subject_deformation_realization_id": animal_record[
                    "subject_deformation_realization_id"
                ],
                "synthetic_animal_id": animal_record["synthetic_animal_id"],
                "receipt": acquisition._json_value(animal_record["receipt"]),
            },
            "support_attempt_binding": acquisition._json_value(
                selected["support_resolution_reference"]
            ),
            "pose_binding": {
                "v2_plane_realization_id": coarse["v2_plane_realization_id"],
                "centre_plane_render_id": coarse["centre_plane_render_id"],
                "legacy_12.5_um_precursor_reference": acquisition._json_value(coarse),
                "legacy_6.25_um_precursor_reference": acquisition._json_value(refined),
            },
            "authenticated_scalar_range_denominator": selected[
                "authenticated_scalar_range_denominator"
            ],
        },
        "render_contract": {
            "redraw_allowed": False,
            "same_pose_and_support_attempt_at_every_level": True,
            "axial_steps_um_max": list(AXIAL_STEPS_UM_MAX),
            "arms": {
                "same_nonidentity_subject_deformation": (
                    "reuse the bound subject-deformation plan exactly"
                ),
                "identity_control": "reuse the same precursor pose with subject_plan=None",
            },
            "raw_array_groups_required": {
                "precursor_raster": sorted(subject_slab._REDUCED_ARRAY_KEYS),
                "subject_coordinate_map": sorted(subject_slab._COORDINATE_ARRAY_KEYS),
                "subject_samples": [
                    "annotation_samples_int64",
                    "scalar_samples_float32",
                ],
                "subject_raster": sorted(subject_slab._REDUCED_ARRAY_KEYS),
            },
        },
        "analysis_contract": {
            "provisional_reference_axial_step_um_max": 1.5625,
            "metric_families": list(METRIC_FAMILIES),
            "strata": {
                "stable_interior": (
                    "support present at every level, no axial or spatial support edge, "
                    "and no axial, spatial, or between-level label boundary"
                ),
                "support_boundary": (
                    "axial zero/nonzero mixing, between-level occupancy change, or a "
                    "four-neighbour tissue-support edge"
                ),
                "atlas_label_boundary": (
                    "nonzero label mixing axially, between levels, or across four-neighbours, "
                    "after support-boundary pixels are assigned"
                ),
            },
            "acceptance_thresholds": None,
            "scientific_decision": "measurement_only_pending_threshold_predeclaration",
        },
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": (
            acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        ),
    }
    payload["plan_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def make_fixed_case_multiresolution_plan_v2(
    failed_report: Mapping[str, object],
    prepared_context: Mapping[str, object],
    *,
    batch_size: int | None = None,
) -> dict[str, object]:
    """Bind the first failed legacy case after an independent full report replay."""
    _verify_failed_legacy_report(
        failed_report, prepared_context, batch_size=batch_size
    )
    return _make_fixed_case_multiresolution_plan_from_validated_report_v2(
        failed_report
    )


def _make_fixed_case_multiresolution_plan_from_live_report_v2(
    failed_report: Mapping[str, object],
    prepared_context: Mapping[str, object],
    *,
    live_report_capability,
) -> dict[str, object]:
    exact_report = live_report_capability(failed_report)
    _validate_failed_legacy_report_structure_v2(exact_report, prepared_context)
    return _make_fixed_case_multiresolution_plan_from_validated_report_v2(
        exact_report
    )


def verify_fixed_case_multiresolution_plan_v2(
    plan: Mapping[str, object],
    failed_report: Mapping[str, object],
    prepared_context: Mapping[str, object],
    *,
    batch_size: int | None = None,
) -> None:
    expected = make_fixed_case_multiresolution_plan_v2(
        failed_report, prepared_context, batch_size=batch_size
    )
    if acquisition._canonical_json(plan) != acquisition._canonical_json(expected):
        raise ValueError("fixed-case multiresolution plan does not match")


def _verify_fixed_case_multiresolution_plan_structure_v2(
    plan: Mapping[str, object],
    failed_report: Mapping[str, object],
    prepared_context: Mapping[str, object],
) -> None:
    _validate_failed_legacy_report_structure_v2(failed_report, prepared_context)
    expected = _make_fixed_case_multiresolution_plan_from_validated_report_v2(
        failed_report
    )
    if acquisition._canonical_json(plan) != acquisition._canonical_json(expected):
        raise ValueError("fixed-case multiresolution plan structure does not match")


def _verify_plan_receipt(plan: Mapping[str, object]) -> None:
    analysis = plan.get("analysis_contract", {})
    render = plan.get("render_contract", {})
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or not _receipt_matches(plan, "plan_receipt_sha256")
        or render.get("redraw_allowed") is not False
        or render.get("axial_steps_um_max") != list(AXIAL_STEPS_UM_MAX)
        or set(render.get("arms", {})) != set(ARM_NAMES)
        or analysis.get("metric_families") != list(METRIC_FAMILIES)
        or set(analysis.get("strata", {})) != set(STRATUM_NAMES)
        or analysis.get("acceptance_thresholds") is not None
        or analysis.get("scientific_decision")
        != "measurement_only_pending_threshold_predeclaration"
        or plan.get("implementation_source_sha256") != _source_hashes()
        or plan.get("implementation_source_sha256_canonicalization")
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
    ):
        raise ValueError("fixed-case multiresolution plan contract is invalid")


def _subject_plan_matches(plan: Mapping[str, object], subject_plan: Mapping[str, object]) -> bool:
    expected = plan["selected_first_failure"]["subject_deformation_reference"]
    receipt = acquisition._json_value(
        deformation.subject_deformation_plan_receipt_v2(subject_plan)
    )
    claimed = receipt.pop("receipt_sha256")
    return bool(
        subject_plan.get("resolved_config", {}).get("deformation_stratum") != "identity"
        and subject_plan.get("subject_deformation_plan_id")
        == expected["subject_deformation_plan_id"]
        and subject_plan.get("subject_deformation_realization_id")
        == expected["subject_deformation_realization_id"]
        and subject_plan.get("synthetic_animal_id") == expected["synthetic_animal_id"]
        and claimed == expected["receipt"]["receipt_sha256"]
        and receipt == expected["receipt"]["receipt_payload"]
    )


def _render_fixed_case_multiresolution_with_mapper_v2(
    prepared_context: Mapping[str, object],
    plan: Mapping[str, object],
    subject_plan: Mapping[str, object],
    *,
    batch_size: int | None = None,
    subject_to_ccf_mapper,
) -> dict[str, object]:
    """Render the frozen eight records; callers retain the returned raw artifacts."""
    _verify_plan_receipt(plan)
    selected = plan["selected_first_failure"]
    support = selected["support_attempt_binding"]
    config = support["configuration"]
    lineage = support["lineage"]
    seed = support["accepted_attempt_seed"]
    pose = selected["pose_binding"]
    if (
        prepared_context.get("v2_context_sha256")
        != plan["failed_report_reference"]["v2_context_sha256"]
        or not _subject_plan_matches(plan, subject_plan)
    ):
        raise ValueError("prepared context or nonidentity subject plan does not match")
    precursors = {}
    renders = {arm: {} for arm in ARM_NAMES}
    for step in AXIAL_STEPS_UM_MAX:
        key = _step_key(step)
        precursor = slab.make_v2_generic_global_reference_slab_render(
            prepared_context,
            lineage["split"],
            seed["attempt_root_seed_uint64"],
            config["section_index"],
            config["plane_stratum"],
            nominal_cut_thickness_um=config["nominal_cut_thickness_um"],
            axial_step_um_max=step,
            parent_shape_h_w=tuple(config["parent_shape_h_w"]),
            animal_id=lineage["animal_id"],
            animal_index=lineage["animal_index"],
            specimen_id=lineage["specimen_id"],
            experiment_id=lineage["experiment_id"],
        )
        slab.verify_v2_generic_global_reference_slab_render(
            precursor, prepared_context
        )
        if (
            precursor["v2_plane_realization_id"] != pose["v2_plane_realization_id"]
            or precursor["centre_plane_render_id"] != pose["centre_plane_render_id"]
        ):
            raise ValueError("multiresolution precursor changed the bound pose")
        legacy_name = {
            12.5: "legacy_12.5_um_precursor_reference",
            6.25: "legacy_6.25_um_precursor_reference",
        }.get(step)
        if legacy_name is not None:
            expected = pose[legacy_name]
            if (
                precursor["slab_render_id"] != expected["slab_render_id"]
                or precursor["receipt_sha256"] != expected["receipt"]["receipt_sha256"]
                or acquisition._json_value(
                    slab.v2_generic_slab_render_receipt(precursor)
                )
                != expected["receipt"]["receipt_payload"]
            ):
                raise ValueError("legacy precursor does not reproduce exactly")
        precursors[key] = precursor
        for arm, arm_plan in (
            ("same_nonidentity_subject_deformation", subject_plan),
            ("identity_control", None),
        ):
            arm_mapper = (
                subject_to_ccf_mapper if arm_plan is subject_plan else None
            )
            artifact = subject_slab._make_subject_slab_render_with_mapper_v2(
                prepared_context,
                precursor,
                subject_plan=arm_plan,
                batch_size=batch_size,
                subject_to_ccf_mapper=arm_mapper,
            )
            subject_slab._verify_subject_slab_render_with_mapper_v2(
                artifact,
                prepared_context,
                precursor,
                subject_plan=arm_plan,
                batch_size=batch_size,
                subject_to_ccf_mapper=arm_mapper,
            )
            renders[arm][key] = artifact
    return {"precursors": precursors, "renders": renders}


def render_fixed_case_multiresolution_v2(
    prepared_context: Mapping[str, object],
    plan: Mapping[str, object],
    subject_plan: Mapping[str, object],
    *,
    batch_size: int | None = None,
) -> dict[str, object]:
    lower, upper = legacy._context_bounds(prepared_context)
    subject_to_ccf_mapper = deformation._verified_subject_to_ccf_mapper_v2(
        subject_plan,
        expected_ccf_context_sha256=prepared_context["v2_context_sha256"],
        expected_full_ccf_lower_um=lower,
        expected_full_ccf_upper_um=upper,
    )
    return _render_fixed_case_multiresolution_with_mapper_v2(
        prepared_context,
        plan,
        subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )


def _checked_receipts(
    arrays: Mapping[str, np.ndarray], receipts: Mapping[str, object], names: set[str]
) -> dict[str, object]:
    if set(arrays) != names or set(receipts) != names:
        raise ValueError("raw array group does not match the frozen contract")
    expected = {name: acquisition._array_receipt(arrays[name]) for name in names}
    if receipts != expected:
        raise ValueError("raw array receipt does not match its live array")
    return acquisition._json_value(expected)


def _raw_array_receipts(
    precursor: Mapping[str, object], artifact: Mapping[str, object]
) -> dict[str, object]:
    precursor_arrays = slab._slab_arrays(precursor["raster"])
    coordinate_arrays = artifact["coordinate_map"]["arrays"]
    sample_arrays = artifact["sample_arrays"]
    raster_arrays = subject_slab._reduced_arrays(artifact["raster"])
    return {
        "precursor_raster": _checked_receipts(
            precursor_arrays,
            precursor["raster"]["array_receipts"],
            set(subject_slab._REDUCED_ARRAY_KEYS),
        ),
        "subject_coordinate_map": _checked_receipts(
            coordinate_arrays,
            artifact["coordinate_map"]["array_receipts"],
            set(subject_slab._COORDINATE_ARRAY_KEYS),
        ),
        "subject_samples": _checked_receipts(
            sample_arrays,
            artifact["sample_array_receipts"],
            {"scalar_samples_float32", "annotation_samples_int64"},
        ),
        "subject_raster": _checked_receipts(
            raster_arrays,
            artifact["raster_array_receipts"],
            set(subject_slab._REDUCED_ARRAY_KEYS),
        ),
    }


def _neighbour_boundary(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    boundary = np.zeros(support.shape, dtype=bool)
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left = tuple(left)
        right = tuple(right)
        changed = (support[left] != support[right]) | (
            support[left] & support[right] & (values[left] != values[right])
        )
        boundary[left] |= changed
        boundary[right] |= changed
    return boundary


def _strata(artifacts: Mapping[str, Mapping[str, object]]) -> dict[str, np.ndarray]:
    rasters = {
        key: subject_slab._reduced_arrays(artifact["raster"])
        for key, artifact in artifacts.items()
    }
    reference = rasters[_step_key(1.5625)]
    occupancy = np.stack(
        [rasters[_step_key(step)]["slab_brain_occupancy"] for step in AXIAL_STEPS_UM_MAX]
    )
    modal = np.stack(
        [rasters[_step_key(step)]["slab_modal_annotation"] for step in AXIAL_STEPS_UM_MAX]
    )
    support_union = np.any(occupancy > 0.0, axis=0)
    reference_samples = np.asarray(
        artifacts[_step_key(1.5625)]["sample_arrays"]["annotation_samples_int64"]
    )
    nonzero = reference_samples != 0
    axial_support_boundary = nonzero.any(axis=0) & ~nonzero.all(axis=0)
    support_boundary = support_union & (
        axial_support_boundary
        | np.any(occupancy != occupancy[-1], axis=0)
        | _neighbour_boundary(support_union.astype(np.uint8), support_union)
    )
    minimum = np.where(nonzero, reference_samples, np.iinfo(np.int64).max).min(axis=0)
    maximum = np.where(nonzero, reference_samples, 0).max(axis=0)
    axial_label_boundary = nonzero.any(axis=0) & (minimum != maximum)
    spatial_label_boundary = _neighbour_boundary(
        reference["slab_modal_annotation"], support_union
    )
    label_boundary = support_union & ~support_boundary & (
        axial_label_boundary
        | np.any(modal != modal[-1], axis=0)
        | spatial_label_boundary
    )
    return {
        "stable_interior": support_union & ~support_boundary & ~label_boundary,
        "support_boundary": support_boundary,
        "atlas_label_boundary": label_boundary,
    }


def _summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "absolute_error_p99": None, "maximum": None}
    return {
        "mean": float(values.mean()),
        "absolute_error_p99": float(np.quantile(values, 0.99, method="linear")),
        "maximum": float(values.max()),
    }


def _measurements(
    artifacts: Mapping[str, Mapping[str, object]], scalar_range: float
) -> dict[str, object]:
    masks = _strata(artifacts)
    reference = subject_slab._reduced_arrays(
        artifacts[_step_key(1.5625)]["raster"]
    )
    result = {
        "strata": {
            name: {
                "pixel_count": int(mask.sum()),
                "mask_receipt": acquisition._array_receipt(mask),
            }
            for name, mask in masks.items()
        },
        "comparisons_to_1.5625_um": {},
    }
    for step in AXIAL_STEPS_UM_MAX:
        current = subject_slab._reduced_arrays(
            artifacts[_step_key(step)]["raster"]
        )
        per_stratum = {}
        for name, mask in masks.items():
            def error(array_name: str, denominator: float = 1.0) -> dict[str, float | None]:
                values = np.abs(
                    np.asarray(current[array_name], np.float64)[mask]
                    - np.asarray(reference[array_name], np.float64)[mask]
                ) / denominator
                return _summary(values)

            def disagreement(array_name: str) -> float | None:
                values = current[array_name][mask] != reference[array_name][mask]
                return None if values.size == 0 else float(values.mean())

            per_stratum[name] = {
                "pixel_count": int(mask.sum()),
                "smooth_scalar": {
                    "normalized_absolute_error": error("scalar", scalar_range)
                },
                "occupancy_and_label_mass": {
                    "slab_brain_occupancy_absolute_error": error(
                        "slab_brain_occupancy"
                    ),
                    "slab_label_purity_absolute_error": error("slab_label_purity"),
                    "centre_label_support_weight_absolute_error": error(
                        "centre_label_support_weight"
                    ),
                },
                "dense_correspondence": {
                    "weight_absolute_error": error("dense_correspondence_weight")
                },
                "categorical": {
                    "slab_modal_annotation_disagreement_fraction": disagreement(
                        "slab_modal_annotation"
                    ),
                    "slab_observable_support_disagreement_fraction": disagreement(
                        "slab_observable_support_mask"
                    ),
                    "dense_abstention_disagreement_fraction": disagreement(
                        "dense_correspondence_abstention_mask"
                    ),
                },
            }
        result["comparisons_to_1.5625_um"][_step_key(step)] = per_stratum
    return result


def _centre_invariance(
    artifacts: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    names = (
        "mapped_allen_coordinates",
        "mapped_physical_coordinates",
        "annotations",
        "support",
    )
    arrays = {}
    for step in AXIAL_STEPS_UM_MAX:
        key = _step_key(step)
        artifact = artifacts[key]
        centre = int(artifact["coordinate_map"]["kernel"]["centre_index"])
        arrays[key] = (
            artifact["coordinate_map"]["arrays"][
                "mapped_allen_index_coordinates_float32"
            ][centre],
            artifact["coordinate_map"]["arrays"][
                "mapped_ccf_physical_coordinates_ap_dv_ml_um_float64"
            ][centre],
            artifact["sample_arrays"]["annotation_samples_int64"][centre],
            artifact["raster"]["centre_plane_support_mask"],
        )
    first = arrays[_step_key(AXIAL_STEPS_UM_MAX[0])]
    result = {}
    all_exact = True
    for index, name in enumerate(names):
        exact = all(
            np.array_equal(first[index], arrays[_step_key(step)][index])
            for step in AXIAL_STEPS_UM_MAX[1:]
        )
        result[name] = {
            "byte_identical_across_all_levels": bool(exact),
            "receipts_by_axial_step_um_max": {
                _step_key(step): acquisition._array_receipt(
                    arrays[_step_key(step)][index]
                )
                for step in AXIAL_STEPS_UM_MAX
            },
        }
        all_exact &= exact
    result["all_centre_targets_byte_identical_across_all_levels"] = all_exact
    return result


def assemble_fixed_case_multiresolution_assessment_v2(
    plan: Mapping[str, object], rendered: Mapping[str, object]
) -> dict[str, object]:
    """Bind raw receipts and threshold-free measurements for all eight renders."""
    _verify_plan_receipt(plan)
    expected_keys = {_step_key(step) for step in AXIAL_STEPS_UM_MAX}
    if (
        set(rendered) != {"precursors", "renders"}
        or set(rendered["precursors"]) != expected_keys
        or set(rendered["renders"]) != set(ARM_NAMES)
        or any(set(rendered["renders"][arm]) != expected_keys for arm in ARM_NAMES)
    ):
        raise ValueError("multiresolution render set is incomplete or has extras")
    records = []
    centre = {}
    measurements = {}
    scalar_range = float(
        plan["selected_first_failure"]["authenticated_scalar_range_denominator"]
    )
    for arm in ARM_NAMES:
        artifacts = rendered["renders"][arm]
        centre[arm] = _centre_invariance(artifacts)
        if not centre[arm]["all_centre_targets_byte_identical_across_all_levels"]:
            raise ValueError("centre targets changed across axial resolutions")
        measurements[arm] = _measurements(artifacts, scalar_range)
        for step in AXIAL_STEPS_UM_MAX:
            key = _step_key(step)
            precursor = rendered["precursors"][key]
            artifact = artifacts[key]
            if (
                float(precursor["slab_recipe"]["axial_step_um_max"]) != step
                or artifact["identity_reference_path"] != (arm == "identity_control")
                or artifact["precursor_reference"]["slab_render_id"]
                != precursor["slab_render_id"]
            ):
                raise ValueError("render arm, level, or precursor binding changed")
            raw = _raw_array_receipts(precursor, artifact)
            record = {
                "arm": arm,
                "axial_step_um_max": step,
                "v2_plane_realization_id": precursor["v2_plane_realization_id"],
                "centre_plane_render_id": precursor["centre_plane_render_id"],
                "precursor_slab_render_id": precursor["slab_render_id"],
                "precursor_receipt_sha256": precursor["receipt_sha256"],
                "subject_coordinate_map_id": artifact["subject_coordinate_map_id"],
                "subject_slab_render_id": artifact["subject_slab_render_id"],
                "subject_slab_receipt_sha256": artifact["receipt_sha256"],
                "raw_array_receipts": raw,
                "raw_array_retention_required": True,
            }
            record["record_receipt_sha256"] = acquisition._payload_sha256(record)
            records.append(record)
    pose = plan["selected_first_failure"]["pose_binding"]
    if any(
        record["v2_plane_realization_id"] != pose["v2_plane_realization_id"]
        or record["centre_plane_render_id"] != pose["centre_plane_render_id"]
        for record in records
    ):
        raise ValueError("an assessment render changed the selected pose")
    payload = {
        "schema_version": ASSESSMENT_SCHEMA,
        "role": "threshold-free fixed-case convergence measurement; not qualification",
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "render_records": records,
        "centre_invariance_by_arm": centre,
        "stratified_measurements_by_arm": measurements,
        "provisional_reference_axial_step_um_max": 1.5625,
        "acceptance_thresholds": None,
        "scientific_decision": "not_evaluated_pending_threshold_predeclaration",
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": (
            acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        ),
    }
    payload["assessment_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def verify_fixed_case_multiresolution_assessment_v2(
    assessment: Mapping[str, object],
    plan: Mapping[str, object],
    rendered: Mapping[str, object],
) -> None:
    expected = assemble_fixed_case_multiresolution_assessment_v2(plan, rendered)
    if acquisition._canonical_json(assessment) != acquisition._canonical_json(expected):
        raise ValueError("fixed-case multiresolution assessment does not match raw arrays")
