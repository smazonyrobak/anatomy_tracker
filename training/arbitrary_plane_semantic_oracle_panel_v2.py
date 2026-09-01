"""Development-only 24-case arbitrary-plane semantic-oracle panel."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_candidate_bank_v2 as candidate_bank_v2
import training.arbitrary_plane_observation_v2 as observation_v2
import training.arbitrary_plane_pose_v2 as pose_v2
import training.arbitrary_plane_realization_v2 as realization_v2
import training.arbitrary_plane_section_processing_v2 as section_processing_v2
import training.arbitrary_plane_semantic_oracle_v2 as semantic_oracle_v2
import training.arbitrary_plane_semantic_oracle_null_gate_v2 as null_gate_v2
import training.arbitrary_plane_subject_deformation_v2 as subject_deformation_v2
import training.arbitrary_plane_subject_slab_v2 as subject_slab_v2
import training.arbitrary_plane_support_resolution_v2 as support_resolution_v2


SEMANTIC_ORACLE_PANEL_V2_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-semantic-oracle-development-panel/v2"
)
SEMANTIC_ORACLE_PANEL_CASE_V2_SCHEMA = (
    "anatomy-tracker.arbitrary-plane-semantic-oracle-development-case/v2"
)
SEMANTIC_ORACLE_PANEL_V2_ALGORITHM = (
    "four-synthetic-animals-by-six-frozen-plane-strata/v2"
)
DEVELOPMENT_SPLIT = "development"
PLANE_STRATA = acquisition.V2_GENERIC_PLANE_STRATA
TRAINABLE_INPUT_MODES = realization_v2.TRAINABLE_INPUT_MODES
MODALITIES = observation_v2.MODALITIES
NOMINAL_CUT_THICKNESS_UM = 55.0
AXIAL_STEP_UM_MAX = 12.5
PARENT_SHAPE_H_W = (256, 256)
MAXIMUM_SUPPORT_ATTEMPTS = 8
MAXIMUM_MODE_INDEX_SEARCH = 64
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_semantic_oracle_panel_v2.py",
    "run_arbitrary_plane_semantic_oracle_panel_v2.py",
    "arbitrary_plane_semantic_oracle_v2.py",
    "arbitrary_plane_semantic_oracle_null_gate_v2.py",
    "arbitrary_plane_candidate_bank_v2.py",
    "arbitrary_plane_pose_v2.py",
    "arbitrary_plane_realization_v2.py",
    "arbitrary_plane_observation_v2.py",
    "arbitrary_plane_section_processing_v2.py",
    "arbitrary_plane_subject_slab_v2.py",
    "arbitrary_plane_support_resolution_v2.py",
    "arbitrary_plane_subject_deformation_v2.py",
    "arbitrary_plane_synthetic_generator_v2.py",
    "arbitrary_plane_acquisition_v2.py",
)
_ANIMALS = (
    (2101, "development-oracle-animal-2101", "development-oracle-specimen-2101"),
    (2102, "development-oracle-animal-2102", "development-oracle-specimen-2102"),
    (2103, "development-oracle-animal-2103", "development-oracle-specimen-2103"),
    (2104, "development-oracle-animal-2104", "development-oracle-specimen-2104"),
)
_EXPERIMENT_ID = "arbitrary-plane-semantic-oracle-development-panel-v2"
_ROOT_SEED_BASES = {
    "subject_deformation": 0x4F5241434C450100,
    "support_resolution": 0x4F5241434C450200,
    "section_processing": 0x4F5241434C450300,
    "observation": 0x4F5241434C450400,
    "candidate_bank": 0x4F5241434C450500,
    "semantic_null": 0x4F5241434C450600,
}


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _root_seed(field: str, numeric_index: int) -> str:
    value = _ROOT_SEED_BASES[field] + int(numeric_index)
    if not 0 <= value < 2**64:
        raise ValueError("panel root seed is outside uint64")
    return f"0x{value:016x}"


def _realization_index_for_mode(
    *,
    observation_root_seed: str,
    split_index: int,
    animal_index: int,
    section_index: int,
    observation_index: int,
    selected_mode: str,
) -> int:
    if selected_mode not in TRAINABLE_INPUT_MODES:
        raise ValueError("panel input mode is not trainable")
    for realization_index in range(MAXIMUM_MODE_INDEX_SEARCH):
        seed = realization_v2.derive_synthetic_realization_seed_v2(
            observation_root_seed,
            DEVELOPMENT_SPLIT,
            split_index,
            animal_index,
            section_index,
            observation_index,
            realization_index,
            "trainable-input-mode",
        )
        mode_index = int(
            np.random.Generator(np.random.PCG64DXSM(seed)).integers(
                len(TRAINABLE_INPUT_MODES)
            )
        )
        if TRAINABLE_INPUT_MODES[mode_index] == selected_mode:
            return realization_index
    raise RuntimeError("bounded deterministic realization-mode search exhausted")


def _panel_payload_v2() -> dict[str, object]:
    animals = []
    cases = []
    for animal_slot, (animal_index, animal_id, specimen_id) in enumerate(_ANIMALS):
        animals.append(
            {
                "animal_slot": animal_slot,
                "animal_index": animal_index,
                "animal_id": animal_id,
                "specimen_id": specimen_id,
                "experiment_id": _EXPERIMENT_ID,
                "subject_deformation_root_seed_uint64": _root_seed(
                    "subject_deformation", animal_slot
                ),
            }
        )
        for stratum_index, plane_stratum in enumerate(PLANE_STRATA):
            case_index = animal_slot * len(PLANE_STRATA) + stratum_index
            split_index = case_index
            section_index = case_index
            observation_index = 0
            selected_mode = TRAINABLE_INPUT_MODES[
                (animal_slot + stratum_index) % len(TRAINABLE_INPUT_MODES)
            ]
            observation_seed = _root_seed("observation", case_index)
            realization_index = _realization_index_for_mode(
                observation_root_seed=observation_seed,
                split_index=split_index,
                animal_index=animal_index,
                section_index=section_index,
                observation_index=observation_index,
                selected_mode=selected_mode,
            )
            cases.append(
                {
                    "case_index": case_index,
                    "case_id": (
                        f"oracle-development-a{animal_slot + 1}-"
                        f"{plane_stratum.lower()}"
                    ),
                    "split": DEVELOPMENT_SPLIT,
                    "split_index": split_index,
                    "animal_slot": animal_slot,
                    "animal_index": animal_index,
                    "animal_id": animal_id,
                    "specimen_id": specimen_id,
                    "experiment_id": _EXPERIMENT_ID,
                    "section_index": section_index,
                    "section_id": f"development-oracle-section-{case_index:02d}",
                    "observation_index": observation_index,
                    "plane_stratum": plane_stratum,
                    "modality": MODALITIES[(animal_slot + stratum_index) % len(MODALITIES)],
                    "selected_trainable_input_mode": selected_mode,
                    "realization_index": realization_index,
                    "root_seeds_uint64": {
                        "support_resolution": _root_seed(
                            "support_resolution", case_index
                        ),
                        "section_processing": _root_seed(
                            "section_processing", case_index
                        ),
                        "observation": observation_seed,
                        "candidate_bank": _root_seed("candidate_bank", case_index),
                        "semantic_null": _root_seed("semantic_null", case_index),
                    },
                }
            )
    return {
        "schema_version": SEMANTIC_ORACLE_PANEL_V2_SCHEMA,
        "algorithm": SEMANTIC_ORACLE_PANEL_V2_ALGORITHM,
        "claim_scope": (
            "small predeclared model-free semantic pose-ranking engineering panel; "
            "not a benchmark, final test, learned-model evaluation, posterior, or "
            "calibrated-uncertainty claim"
        ),
        "cohort_policy": {
            "development_only": True,
            "synthetic_animals_only": True,
            "authenticated_atlas_reference_allowed": True,
            "public_benchmark_or_deepslice_cases_used": False,
            "benchmark_animals_used": False,
            "final_test_animals_used": False,
            "real_lab_histology_used": False,
            "animal_is_statistical_unit_for_future_validation": True,
        },
        "configuration": {
            "plane_strata": list(PLANE_STRATA),
            "animal_count": len(animals),
            "case_count": len(cases),
            "one_case_per_animal_and_stratum": True,
            "nominal_cut_thickness_um": NOMINAL_CUT_THICKNESS_UM,
            "axial_step_um_max": AXIAL_STEP_UM_MAX,
            "parent_shape_h_w": list(PARENT_SHAPE_H_W),
            "maximum_support_attempts": MAXIMUM_SUPPORT_ATTEMPTS,
            "subject_deformation_mode": "standard",
            "section_processing_deformation_mode": "standard",
            "candidate_count_per_case": 40,
            "case_level_gate_evaluated_here": False,
            "aggregate_failure_adverse_gate_summary_required": True,
            "full_benchmark_deferred": True,
        },
        "input_mode_policy": {
            "trainable_modes": list(TRAINABLE_INPUT_MODES),
            "scheduled_count_per_trainable_mode": 8,
            "each_animal_count_per_trainable_mode": 2,
            "raw_mode_trainable": False,
            "raw_mode_scheduled": False,
            "smart_brush_is_optional_model_side_information_only": True,
            "brush_masks_or_errors_used_by_semantic_scorer": False,
        },
        "rng_policy": {
            "root_seed_bases_uint64": {
                name: f"0x{value:016x}" for name, value in _ROOT_SEED_BASES.items()
            },
            "root_seed_derivation": "fixed domain base plus numeric animal-slot or case-index",
            "dynamic_coordinates": [
                "split_index",
                "animal_index",
                "section_index",
                "observation_index",
                "realization_index",
            ],
            "excluded_from_all_rng": [
                "animal_id",
                "specimen_id",
                "experiment_id",
                "section_id",
                "artifact_ids",
            ],
            "realization_mode_policy": (
                "first index below 64 whose existing domain-separated final-stage RNG "
                "selects the predeclared trainable mode; no image, target, or support access"
            ),
        },
        "asset_dependencies": {
            "learned_checkpoint_dependencies": [],
            "previous_model_dependencies": [],
            "pretrained_feature_dependencies": [],
            "learned_style_model_dependencies": [],
        },
        "failure_policy": {
            "all_scheduled_cases_saved": True,
            "pipeline_exceptions_saved_as_failures": True,
            "zero_support_truth_saved_and_counted_as_failure_by_later_gates": True,
            "redraw_after_semantic_oracle_zero_support": False,
        },
        "persistence_policy": {
            "one_immutable_atomic_json_record_per_scheduled_case": True,
            "one_immutable_atomic_failure_adverse_gate_summary": True,
            "receipt_verified_resume_without_reexecution": True,
            "opt_in_strict_regeneration_and_byte_equivalent_replay": True,
            "gate_freeze_requires_all_cases_live_verified_in_one_execution": True,
            "preexisting_case_requires_strict_replay_for_first_gate_freeze": True,
            "changed_existing_outputs_overwritten": False,
        },
        "failure_adverse_gate_summary_inputs": {
            "raw_semantic_score_vectors": True,
            "raw_id_agreement_vectors": True,
            "mask_dice_vectors": True,
            "rank_and_tie_fields": True,
            "coverage_and_zero_support_fields": True,
            "selected_pose_error_fields": True,
            "candidate_summaries": True,
            "exact_control_evidence": True,
            "animal_specimen_experiment_and_plane_stratum": True,
            "source_config_and_stage_receipts": True,
            "shape_preserving_null_result": True,
            "null_generated_while_live_candidate_and_target_inputs_exist": True,
        },
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": (
            acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        ),
        "animals": animals,
        "cases": cases,
    }


def arbitrary_plane_semantic_oracle_development_panel_v2() -> dict[str, object]:
    """Return the frozen 4-animal by 6-stratum development schedule."""
    payload = _panel_payload_v2()
    return {
        **payload,
        "panel_receipt_sha256": acquisition._payload_sha256(payload),
    }


def verify_arbitrary_plane_semantic_oracle_development_panel_v2(
    panel: Mapping[str, object],
) -> None:
    expected = arbitrary_plane_semantic_oracle_development_panel_v2()
    if acquisition._canonical_json(acquisition._json_value(panel)) != acquisition._canonical_json(
        expected
    ):
        raise ValueError("semantic-oracle development panel contract changed")


def _context_bounds(prepared_context: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    support = acquisition._context_support(prepared_context)
    lower = np.asarray(support["origin_um"], dtype=np.float64)
    upper = lower + np.asarray(support["annotation_shape"], dtype=np.float64) * np.asarray(
        support["voxel_size_um"], dtype=np.float64
    )
    return lower, upper


def _make_development_panel_subject_plan_with_mapper_v2(
    prepared_context: Mapping[str, object], animal_spec: Mapping[str, object]
) -> tuple[Mapping[str, object], object]:
    """Make one independently randomized development synthetic-animal deformation."""
    lower, upper = _context_bounds(prepared_context)
    plan = subject_deformation_v2.sample_animal_subject_deformation_plan_v2(
        lower,
        upper,
        root_seed=animal_spec["subject_deformation_root_seed_uint64"],
        split=DEVELOPMENT_SPLIT,
        animal_index=animal_spec["animal_index"],
        animal_id=animal_spec["animal_id"],
        ccf_context_sha256=prepared_context["v2_context_sha256"],
        deformation_stratum="standard",
    )
    subject_to_ccf_mapper = subject_deformation_v2._verified_subject_to_ccf_mapper_v2(
        plan,
        expected_ccf_context_sha256=prepared_context["v2_context_sha256"],
        expected_full_ccf_lower_um=lower,
        expected_full_ccf_upper_um=upper,
    )
    return plan, subject_to_ccf_mapper


def make_development_panel_subject_plan_v2(
    prepared_context: Mapping[str, object], animal_spec: Mapping[str, object]
) -> Mapping[str, object]:
    plan, _ = _make_development_panel_subject_plan_with_mapper_v2(
        prepared_context, animal_spec
    )
    return plan


def _stage_references(
    subject_plan: Mapping[str, object],
    support_bundle: Mapping[str, object],
    precursor: Mapping[str, object],
    subject_slab: Mapping[str, object],
    section_plan: Mapping[str, object],
    section_render: Mapping[str, object],
    observation: Mapping[str, object],
    final: Mapping[str, object],
    pose_truth: Mapping[str, object],
    candidate_bank: Mapping[str, object],
    oracle_result: Mapping[str, object],
    null_result: Mapping[str, object],
) -> dict[str, object]:
    resolution = support_bundle["resolution"]
    return {
        "subject_deformation": {
            "subject_deformation_plan_id": subject_plan["subject_deformation_plan_id"],
            "subject_deformation_realization_id": subject_plan[
                "subject_deformation_realization_id"
            ],
            "synthetic_animal_id": subject_plan["synthetic_animal_id"],
            "receipt_sha256": subject_plan["receipt_sha256"],
        },
        "support_resolution": {
            "support_resolution_plan_id": resolution["support_resolution_plan_id"],
            "subject_support_resolution_id": resolution[
                "subject_support_resolution_id"
            ],
            "receipt_sha256": resolution["receipt_sha256"],
        },
        "precursor": {
            "v2_plane_realization_id": precursor["v2_plane_realization_id"],
            "slab_render_id": precursor["slab_render_id"],
            "receipt_sha256": precursor["receipt_sha256"],
        },
        "subject_slab": {
            "subject_coordinate_map_id": subject_slab["subject_coordinate_map_id"],
            "subject_slab_render_id": subject_slab["subject_slab_render_id"],
            "receipt_sha256": subject_slab["receipt_sha256"],
        },
        "section_processing_plan": {
            "section_processing_plan_id": section_plan["section_processing_plan_id"],
            "section_processing_realization_id": section_plan[
                "section_processing_realization_id"
            ],
            "receipt_sha256": section_plan["receipt_sha256"],
        },
        "section_processing_render": {
            "section_processing_render_id": section_render[
                "section_processing_render_id"
            ],
            "receipt_sha256": section_render["receipt_sha256"],
        },
        "observation": {
            "observation_plan_id": observation["observation_plan_id"],
            "observation_bundle_id": observation["observation_bundle_id"],
            "receipt_sha256": observation["receipt_sha256"],
        },
        "final_realization": {
            "synthetic_realization_id": final["synthetic_realization_id"],
            "training_row_id": final["training_row_id"],
            "receipt_sha256": final["receipt_sha256"],
        },
        "pose_truth": {
            "finite_plane_pose_truth_id": pose_truth["finite_plane_pose_truth_id"],
            "receipt_sha256": pose_truth["receipt_sha256"],
        },
        "candidate_bank": {
            "candidate_bank_id": candidate_bank["candidate_bank_id"],
            "receipt_sha256": candidate_bank["receipt_sha256"],
        },
        "semantic_oracle": {
            "semantic_oracle_result_id": oracle_result["semantic_oracle_result_id"],
            "receipt_sha256": oracle_result["receipt_sha256"],
        },
        "semantic_null": {
            "semantic_null_result_id": null_result["semantic_null_result_id"],
            "receipt_sha256": null_result["receipt_sha256"],
        },
    }


def _resolved_configurations(
    subject_plan: Mapping[str, object],
    support_bundle: Mapping[str, object],
    section_plan: Mapping[str, object],
    observation: Mapping[str, object],
    final: Mapping[str, object],
    candidate_bank: Mapping[str, object],
    oracle_result: Mapping[str, object],
    null_result: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, str]]:
    configurations = acquisition._json_value(
        {
            "subject_deformation": subject_plan["resolved_config"],
            "support_resolution": support_bundle["resolution"]["configuration"],
            "section_processing": section_plan["resolved_config"],
            "observation": {
                "provenance": observation["provenance"],
                "modality": observation["modality"],
            },
            "final_realization": {
                "provenance": final["provenance"],
                "mode_selection": final["mode_selection"],
            },
            "candidate_bank": {
                "rng_contract": candidate_bank["rng_contract"],
                "schedule": candidate_bank["schedule"],
            },
            "semantic_oracle": {
                "scope": oracle_result["scope"],
                "scorer_input_contract": oracle_result["scorer_input_contract"],
            },
            "semantic_null": {
                "scope": null_result["scope"],
                "rng_contract": null_result["rng_contract"],
            },
        }
    )
    return configurations, {
        name: acquisition._payload_sha256(value)
        for name, value in configurations.items()
    }


def _evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2(
    prepared_context: Mapping[str, object],
    panel: Mapping[str, object],
    case_spec: Mapping[str, object],
    subject_plan: Mapping[str, object],
    *,
    batch_size: int | None = None,
    subject_to_ccf_mapper,
) -> dict[str, object]:
    """Run one scheduled case through the random-only v2 generator and oracle."""
    verify_arbitrary_plane_semantic_oracle_development_panel_v2(panel)
    if acquisition._json_value(case_spec) not in panel["cases"]:
        raise ValueError("case is not in the frozen development panel")
    support_bundle = support_resolution_v2._resolve_subject_support_with_mapper_v2(
        prepared_context,
        subject_plan=subject_plan,
        master_root_seed=case_spec["root_seeds_uint64"]["support_resolution"],
        split=case_spec["split"],
        split_index=case_spec["split_index"],
        animal_index=case_spec["animal_index"],
        animal_id=case_spec["animal_id"],
        section_index=case_spec["section_index"],
        plane_stratum=case_spec["plane_stratum"],
        nominal_cut_thickness_um=NOMINAL_CUT_THICKNESS_UM,
        specimen_id=case_spec["specimen_id"],
        experiment_id=case_spec["experiment_id"],
        axial_step_um_max=AXIAL_STEP_UM_MAX,
        parent_shape_h_w=PARENT_SHAPE_H_W,
        max_attempts=MAXIMUM_SUPPORT_ATTEMPTS,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    if support_bundle["resolution"]["status"] != "accepted":
        raise RuntimeError("predeclared case exhausted its bounded support attempts")
    precursor = support_bundle["accepted_precursor"]
    subject_slab = subject_slab_v2._make_subject_slab_render_with_mapper_v2(
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    centre_index = int(subject_slab["coordinate_map"]["kernel"]["centre_index"])
    subject_physical = np.asarray(
        subject_slab["coordinate_map"]["arrays"][
            "subject_physical_coordinates_ap_dv_ml_um_float64"
        ][centre_index]
    )
    pixel_pitch_y_x_um, _, _ = section_processing_v2._orthogonal_section_pixel_metric(
        subject_physical
    )
    section_plan = section_processing_v2.sample_section_processing_plan_v2(
        tuple(subject_physical.shape[:2]),
        tuple(pixel_pitch_y_x_um),
        root_seed=case_spec["root_seeds_uint64"]["section_processing"],
        split=case_spec["split"],
        animal_index=case_spec["animal_index"],
        section_index=case_spec["section_index"],
        animal_id=case_spec["animal_id"],
        section_id=case_spec["section_id"],
        deformation_mode="standard",
    )
    section_render = section_processing_v2._make_section_processing_render_with_mapper_v2(
        subject_slab,
        section_plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    observation = observation_v2._make_arbitrary_plane_observation_with_mapper_v2(
        section_render,
        subject_slab,
        section_plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        root_seed=case_spec["root_seeds_uint64"]["observation"],
        split=case_spec["split"],
        split_index=case_spec["split_index"],
        animal_index=case_spec["animal_index"],
        animal_id=case_spec["animal_id"],
        section_index=case_spec["section_index"],
        observation_index=case_spec["observation_index"],
        modality=case_spec["modality"],
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    final = realization_v2._make_arbitrary_plane_realization_with_mapper_v2(
        prepared_context,
        support_bundle,
        precursor,
        subject_slab,
        section_plan,
        section_render,
        observation,
        subject_plan=subject_plan,
        realization_index=case_spec["realization_index"],
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    if final["mode_selection"]["selected_mode"] != case_spec[
        "selected_trainable_input_mode"
    ]:
        raise RuntimeError("final-stage RNG did not select the predeclared input mode")
    pose_truth = pose_v2.make_arbitrary_plane_pose_truth_v2(final, prepared_context)
    candidate_bank = candidate_bank_v2.make_arbitrary_plane_candidate_bank_v2(
        pose_truth,
        final,
        prepared_context,
        candidate_root_seed=case_spec["root_seeds_uint64"]["candidate_bank"],
    )
    oracle_result = semantic_oracle_v2.make_arbitrary_plane_semantic_oracle_result_v2(
        candidate_bank, pose_truth, final, prepared_context
    )
    semantic_oracle_v2.verify_arbitrary_plane_semantic_oracle_result_v2(
        oracle_result, candidate_bank, pose_truth, final, prepared_context
    )
    null_result = null_gate_v2.make_arbitrary_plane_semantic_null_result_v2(
        oracle_result,
        candidate_bank,
        pose_truth,
        final,
        prepared_context,
        null_root_seed=case_spec["root_seeds_uint64"]["semantic_null"],
    )
    null_gate_v2.verify_arbitrary_plane_semantic_null_result_v2(
        null_result,
        oracle_result,
        candidate_bank,
        pose_truth,
        final,
        prepared_context,
    )
    lineage = oracle_result["provenance"]["source_lineage"]
    if acquisition._json_value(lineage) != {
        "support_resolution_plan_id": support_bundle["resolution"][
            "support_resolution_plan_id"
        ],
        "support_resolution_receipt_sha256": support_bundle["resolution"][
            "receipt_sha256"
        ],
        "split": case_spec["split"],
        "split_index": case_spec["split_index"],
        "animal_id": case_spec["animal_id"],
        "animal_index": case_spec["animal_index"],
        "specimen_id": case_spec["specimen_id"],
        "experiment_id": case_spec["experiment_id"],
        "section_index": case_spec["section_index"],
        "plane_stratum": case_spec["plane_stratum"],
    }:
        raise RuntimeError("oracle result did not promote the scheduled source lineage")
    configurations, configuration_receipts = _resolved_configurations(
        subject_plan,
        support_bundle,
        section_plan,
        observation,
        final,
        candidate_bank,
        oracle_result,
        null_result,
    )
    payload = {
        "schema_version": SEMANTIC_ORACLE_PANEL_CASE_V2_SCHEMA,
        "status": "complete",
        "case_spec": acquisition._json_value(case_spec),
        "case_spec_receipt_sha256": acquisition._payload_sha256(
            acquisition._json_value(case_spec)
        ),
        "panel_receipt_sha256": panel["panel_receipt_sha256"],
        "source_receipts": _source_hashes(),
        "resolved_stage_configurations": configurations,
        "resolved_stage_configuration_receipts": configuration_receipts,
        "stage_references": _stage_references(
            subject_plan,
            support_bundle,
            precursor,
            subject_slab,
            section_plan,
            section_render,
            observation,
            final,
            pose_truth,
            candidate_bank,
            oracle_result,
            null_result,
        ),
        "selected_trainable_input_mode": final["mode_selection"]["selected_mode"],
        "raw_mode_trainable": False,
        "semantic_oracle_result": acquisition._json_value(oracle_result),
        "semantic_null_result": acquisition._json_value(null_result),
        "later_gate_policy": {
            "included_in_all_scheduled_denominators": True,
            "evaluable": bool(oracle_result["coverage"]["evaluable"]),
            "zero_support_counts_as_top1_and_top3_failure": True,
            "gate_evaluated_here": False,
        },
    }
    payload["case_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def evaluate_arbitrary_plane_semantic_oracle_development_case_v2(
    prepared_context: Mapping[str, object],
    panel: Mapping[str, object],
    case_spec: Mapping[str, object],
    subject_plan: Mapping[str, object],
    *,
    batch_size: int | None = None,
) -> dict[str, object]:
    lower, upper = _context_bounds(prepared_context)
    subject_to_ccf_mapper = subject_deformation_v2._verified_subject_to_ccf_mapper_v2(
        subject_plan,
        expected_ccf_context_sha256=prepared_context["v2_context_sha256"],
        expected_full_ccf_lower_um=lower,
        expected_full_ccf_upper_um=upper,
    )
    return _evaluate_arbitrary_plane_semantic_oracle_development_case_with_mapper_v2(
        prepared_context,
        panel,
        case_spec,
        subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )


def make_arbitrary_plane_semantic_oracle_failure_record_v2(
    panel: Mapping[str, object],
    case_spec: Mapping[str, object],
    error: Exception,
) -> dict[str, object]:
    payload = {
        "schema_version": SEMANTIC_ORACLE_PANEL_CASE_V2_SCHEMA,
        "status": "failed",
        "case_spec": acquisition._json_value(case_spec),
        "case_spec_receipt_sha256": acquisition._payload_sha256(
            acquisition._json_value(case_spec)
        ),
        "panel_receipt_sha256": panel["panel_receipt_sha256"],
        "source_receipts": _source_hashes(),
        "failure": {
            "exception_type": type(error).__name__,
            "message": str(error),
            "scheduled_case_retained": True,
            "redraw_or_replacement_case_created": False,
        },
        "later_gate_policy": {
            "included_in_all_scheduled_denominators": True,
            "evaluable": False,
            "top1_and_top3_success": False,
            "gate_evaluated_here": False,
        },
    }
    payload["case_receipt_sha256"] = acquisition._payload_sha256(payload)
    return payload


def verify_arbitrary_plane_semantic_oracle_case_record_v2(
    record: Mapping[str, object],
    panel: Mapping[str, object],
    case_spec: Mapping[str, object],
) -> None:
    verify_arbitrary_plane_semantic_oracle_development_panel_v2(panel)
    value = acquisition._json_value(record)
    receipt = value.pop("case_receipt_sha256", None)
    common = bool(
        value.get("schema_version") == SEMANTIC_ORACLE_PANEL_CASE_V2_SCHEMA
        and value.get("case_spec") == acquisition._json_value(case_spec)
        and value.get("case_spec_receipt_sha256")
        == acquisition._payload_sha256(acquisition._json_value(case_spec))
        and value.get("panel_receipt_sha256") == panel["panel_receipt_sha256"]
        and value.get("source_receipts") == _source_hashes()
        and receipt == acquisition._payload_sha256(value)
        and value.get("later_gate_policy", {}).get(
            "included_in_all_scheduled_denominators"
        )
        is True
        and value.get("later_gate_policy", {}).get("gate_evaluated_here") is False
    )
    if not common:
        raise ValueError("semantic-oracle development case receipt or lineage changed")
    if value.get("status") == "failed":
        if (
            set(value)
            != {
                "schema_version",
                "status",
                "case_spec",
                "case_spec_receipt_sha256",
                "panel_receipt_sha256",
                "source_receipts",
                "failure",
                "later_gate_policy",
            }
            or value["failure"].get("scheduled_case_retained") is not True
            or value["failure"].get("redraw_or_replacement_case_created") is not False
            or value["later_gate_policy"].get("top1_and_top3_success") is not False
        ):
            raise ValueError("failed development case policy changed")
        return
    if value.get("status") != "complete" or set(value) != {
        "schema_version",
        "status",
        "case_spec",
        "case_spec_receipt_sha256",
        "panel_receipt_sha256",
        "source_receipts",
        "resolved_stage_configurations",
        "resolved_stage_configuration_receipts",
        "stage_references",
        "selected_trainable_input_mode",
        "raw_mode_trainable",
        "semantic_oracle_result",
        "semantic_null_result",
        "later_gate_policy",
    }:
        raise ValueError("completed development case structure changed")
    oracle_result = value["semantic_oracle_result"]
    null_result = value["semantic_null_result"]
    score_arrays = oracle_result.get("scores", {}).get("arrays", {})
    score_receipts = oracle_result.get("scores", {}).get("array_receipts", {})
    lineage = oracle_result.get("provenance", {}).get("source_lineage", {})
    expected_lineage = {
        "split": case_spec["split"],
        "split_index": case_spec["split_index"],
        "animal_id": case_spec["animal_id"],
        "animal_index": case_spec["animal_index"],
        "specimen_id": case_spec["specimen_id"],
        "experiment_id": case_spec["experiment_id"],
        "section_index": case_spec["section_index"],
        "plane_stratum": case_spec["plane_stratum"],
    }
    expected_configuration_names = {
        "subject_deformation",
        "support_resolution",
        "section_processing",
        "observation",
        "final_realization",
        "candidate_bank",
        "semantic_oracle",
        "semantic_null",
    }
    expected_stage_names = {
        "subject_deformation",
        "support_resolution",
        "precursor",
        "subject_slab",
        "section_processing_plan",
        "section_processing_render",
        "observation",
        "final_realization",
        "pose_truth",
        "candidate_bank",
        "semantic_oracle",
        "semantic_null",
    }
    oracle_identity = semantic_oracle_v2._identity_payload(oracle_result)
    oracle_receipt = semantic_oracle_v2.arbitrary_plane_semantic_oracle_result_receipt_v2(
        oracle_result
    )
    if (
        value["selected_trainable_input_mode"]
        != case_spec["selected_trainable_input_mode"]
        or value["raw_mode_trainable"] is not False
        or oracle_result.get("schema_version")
        != semantic_oracle_v2.SEMANTIC_ORACLE_RESULT_V2_SCHEMA
        or set(oracle_result) != semantic_oracle_v2._RESULT_KEYS
        or oracle_result.get("semantic_oracle_result_id")
        != acquisition._payload_sha256(oracle_identity)
        or oracle_result.get("receipt_sha256")
        != acquisition._payload_sha256(oracle_receipt)
        or any(lineage.get(name) != expected for name, expected in expected_lineage.items())
        or set(score_arrays) != semantic_oracle_v2._SCORE_ARRAY_KEYS
        or set(score_receipts) != semantic_oracle_v2._SCORE_ARRAY_KEYS
        or oracle_result.get("scope", {}).get("benchmark_or_final_test_claim") is not False
        or oracle_result.get("scope", {}).get("posterior_or_probability_claim") is not False
        or oracle_result.get("scope", {}).get("valid_correspondence_weight_used") is not False
        or set(oracle_result.get("exact_controls", {}))
        != set(semantic_oracle_v2.EXACT_CONTROL_NAMES)
        or not all(
            control.get("passed") is True
            for control in oracle_result.get("exact_controls", {}).values()
        )
        or oracle_result.get("candidate_reference", {}).get("candidate_count") != 40
        or value["later_gate_policy"].get("evaluable")
        is not bool(oracle_result.get("coverage", {}).get("evaluable"))
        or set(value["resolved_stage_configurations"])
        != expected_configuration_names
        or set(value["resolved_stage_configuration_receipts"])
        != expected_configuration_names
        or set(value["stage_references"]) != expected_stage_names
        or value["stage_references"]["semantic_oracle"].get(
            "semantic_oracle_result_id"
        )
        != oracle_result.get("semantic_oracle_result_id")
        or value["stage_references"]["semantic_oracle"].get("receipt_sha256")
        != oracle_result.get("receipt_sha256")
        or not all(null_gate_v2._null_self_audit(null_result).values())
        or null_result.get("provenance") != oracle_result.get("provenance")
        or null_result.get("rng_contract", {}).get("null_root_seed_uint64")
        != case_spec["root_seeds_uint64"]["semantic_null"]
        or null_result.get("scope", {}).get("model_training_or_benchmark_claim")
        is not False
        or null_result.get("scope", {}).get("posterior_or_probability_claim")
        is not False
        or null_result.get("upstream_reference", {}).get(
            "semantic_oracle_result_id"
        )
        != oracle_result.get("semantic_oracle_result_id")
        or null_result.get("upstream_reference", {}).get(
            "semantic_oracle_result_receipt_sha256"
        )
        != oracle_result.get("receipt_sha256")
        or value["stage_references"]["semantic_null"].get(
            "semantic_null_result_id"
        )
        != null_result.get("semantic_null_result_id")
        or value["stage_references"]["semantic_null"].get("receipt_sha256")
        != null_result.get("receipt_sha256")
        or any(
            not isinstance(stage.get("receipt_sha256"), str)
            or len(stage["receipt_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in stage["receipt_sha256"])
            for stage in value["stage_references"].values()
        )
    ):
        raise ValueError("completed development case oracle contract changed")
    for name, control in oracle_result["exact_controls"].items():
        evidence_payload = {
            "control": name,
            "passed": control.get("passed"),
            "evidence": control.get("evidence"),
        }
        if (
            control.get("control") != name
            or control.get("evidence_receipt_sha256")
            != acquisition._payload_sha256(evidence_payload)
        ):
            raise ValueError("saved development oracle control evidence changed")
    for name, raw in score_arrays.items():
        array = np.asarray(raw, dtype=np.float64)
        if (
            array.shape != (40,)
            or not np.isfinite(array).all()
            or np.any((array < 0.0) | (array > 1.0))
            or score_receipts[name] != acquisition._array_receipt(array)
        ):
            raise ValueError("saved development oracle score vector changed")
    for name, configuration in value["resolved_stage_configurations"].items():
        if value["resolved_stage_configuration_receipts"].get(name) != acquisition._payload_sha256(
            configuration
        ):
            raise ValueError("saved development stage configuration changed")


def adapt_development_panel_records_to_semantic_gate_v2(
    panel: Mapping[str, object],
    records: list[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Project rich panel records onto the frozen four-/five-field gate schemas."""
    verify_arbitrary_plane_semantic_oracle_development_panel_v2(panel)
    cases = list(panel["cases"])
    by_id: dict[str, Mapping[str, object]] = {}
    planned = [
        {
            "case_id": case["case_id"],
            "plane_stratum": case["plane_stratum"],
            "animal_index": case["animal_index"],
            "animal_id": acquisition._json_value(case["animal_id"]),
        }
        for case in cases
    ]
    case_by_id = {case["case_id"]: case for case in cases}
    for record in records:
        case_id = record.get("case_spec", {}).get("case_id")
        if case_id not in case_by_id or case_id in by_id:
            raise ValueError("panel records contain an unplanned or duplicate case")
        verify_arbitrary_plane_semantic_oracle_case_record_v2(
            record, panel, case_by_id[case_id]
        )
        by_id[case_id] = record
    if set(by_id) != set(case_by_id):
        raise ValueError("panel-to-gate adapter requires every scheduled case record")
    gate_records = []
    for case in cases:
        record = by_id[case["case_id"]]
        if record["status"] == "complete":
            gate_records.append(
                {
                    "case_id": case["case_id"],
                    "status": "completed",
                    "primary_result": acquisition._json_value(
                        record["semantic_oracle_result"]
                    ),
                    "null_result": acquisition._json_value(
                        record["semantic_null_result"]
                    ),
                    "failure": None,
                }
            )
        else:
            gate_records.append(
                {
                    "case_id": case["case_id"],
                    "status": "execution_failure",
                    "primary_result": None,
                    "null_result": None,
                    "failure": {
                        "stage": "panel-case-pipeline",
                        "reason": record["failure"]["message"],
                        "exception_type": record["failure"]["exception_type"],
                    },
                }
            )
    return planned, gate_records
