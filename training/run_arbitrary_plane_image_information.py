"""Run the frozen, model-free arbitrary-plane image-information pilot.

The module deliberately contains no learned component.  The public runner first
authenticates the repository and the completed semantic-oracle output, replays all
64 accepted cases without rendering candidate intensities, and only then creates
an output directory and evaluates the predeclared image-score schedule.
"""

from __future__ import annotations

import copy
import ast
import hashlib
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from training.arbitrary_plane_finite_candidates import (
    finite_candidate_bank_receipt,
    make_arbitrary_plane_finite_candidate_bank_from_context,
    transport_finite_candidate_pose,
)
from training.arbitrary_plane_image_candidate_scalar import render_candidate_bank_scalars
from training.arbitrary_plane_image_information import (
    IMAGE_INFORMATION_ALGORITHM,
    common_lattice_map_yx,
    constant_within_support_null,
    dewarp_target_float32,
    dewarp_target_for_scoring,
    mind_parameters,
    rank_candidate_scores,
    resample_common_lattice_intensity,
    resample_common_lattice_support,
    scale_candidate_raster,
    score_mind_candidates,
    score_support_penalized_mind_candidates,
    target_score_masks,
)
from training.arbitrary_plane_image_secondary import (
    hog_boundary_ring_weights,
    hog_complete_block_mask,
    ngf_evaluation_domain,
    score_hog_candidates,
    score_ngf_candidates,
)
from training.arbitrary_plane_rendered_generator import (
    effective_renderer_sampling_arrays,
    finite_render_receipt,
    make_finite_arbitrary_plane_render_from_context,
)
from training.arbitrary_plane_semantic_oracle import build_oracle_target, rp2_plane_error
from training.arbitrary_plane_synthetic_generator import (
    ABSENT_OUTLINE,
    ACCURATE_OUTLINE,
    IMPERFECT_OUTLINE,
    make_arbitrary_plane_synthetic_realization,
    synthetic_realization_receipt,
)
from training.run_arbitrary_plane_semantic_oracle import (
    _array_receipt,
    _effective_ouv,
    _finite_point_error_with_evidence,
    case_seed_lineage,
    load_allen_contexts,
    shuffled_case_cycle,
    verify_written_result as verify_semantic_written_result,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCHEMA = "anatomy-tracker.arbitrary-plane-image-information-run/v1"
PREFLIGHT_COMMIT = "61e179aa7abb955dacc48410614992cff3e5e4b3"
SCORER_COMMIT = "c45105d74f2e5aabd45fd5fe957f463c5008bea9"
SOURCE_PARENT_COMMIT = "ee9c72cec7980cb014721cb5e69a8fee0a7c1a27"
FROZEN_SEMANTIC_SOURCE_COMMIT = "27c4ba644c7b75ab6f676d944e917811df961b05"
EXPECTED_BRANCH = "codex/joint-registration"
CASE_COUNT = 64
CANDIDATE_COUNT = 40
OUTPUT_SHAPE = (192, 256)
MARGIN_UM = (250.0, 250.0)
OUTLINE_MODES = (ACCURATE_OUTLINE, IMPERFECT_OUTLINE, ABSENT_OUTLINE)
ORIENTATION_COUNTS = {"near_AP": 12, "near_DV": 12, "near_ML": 12, "general_oblique": 28}
FROZEN_POOLED_MEMBERSHIP = {
    "challenging_appearance": (
        4, 5, 6, 7, 8, 11, 15, 21, 24, 25, 26, 28, 29, 33, 36, 43, 45,
        51, 52, 53, 55, 60,
    ),
    "damaged": (
        0, 1, 9, 11, 12, 13, 14, 15, 17, 22, 23, 30, 32, 33, 36, 37, 39,
        44, 45, 47, 49, 51, 52, 53, 56, 58, 63,
    ),
}
FROZEN_SEMANTIC_OUTPUT = ROOT / "build" / "arbitrary_plane_semantic_oracle_raw_27c4ba6"
FROZEN_SEMANTIC_RESULT_FILE_SHA256 = "dee90febea4faf9a8ae92c27eea7907cf0ecdc27e6a7e7be19e59023df29794a"
FROZEN_SEMANTIC_CONFIG_PAYLOAD_SHA256 = "72312591a83e6c6250399017bb093fb323ff12b6ed745b2baba19704f94fe9b2"
FROZEN_SEMANTIC_INVENTORY_SHA256 = "3c611f6296e950e7c1b3582e55268838e1922c595775b0217d697d8350191cd3"
FROZEN_SEMANTIC_FILE_COUNT = 514
FROZEN_SEMANTIC_TOTAL_BYTES = 63_939_690
SUPPORT_INDEX_SHA256 = "0f6d8325ff1aa1965b28e4c5edb1327ae58129a78fe41d6f2ccf1896f449f65c"
PREPARED_RENDER_CONTEXT_SHA256 = "04c387c1a778bfd0962cea972047b5eadc84f8cd2bce92ff74e9656f0af63b47"
PREPARED_CANDIDATE_CONTEXT_SHA256 = "9e7e241b1b72bee571dffad53ea817d8221564c5827837f053490d29370eed88"
DECODED_TEMPLATE_ARRAY_SHA256 = "92c1441a9a025d83199bcf3321fb8fb4b519f44030ab44d11f751f64e9516bce"
SCALAR_CONVERSION_ARRAY_SHA256 = "b7ad03fa18551a980232a1558be8b473581fbf162d2df3669337451858fa02fa"
DECODED_ANNOTATION_ARRAY_SHA256 = "d691efd938b9e4694e7b939aa5e71efc68441b6a25aa31f0caea51a6d26b6c8c"
ATLAS_TEMPLATE_URI = "data/Allen Brain Atlas 25um/average_template_25.nrrd"
ATLAS_ANNOTATION_URI = "data/Allen Brain Atlas 25um/annotation_25.nrrd"
ATLAS_TEMPLATE_SHA256 = "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
ATLAS_ANNOTATION_SHA256 = "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
MASK_COMPACT_SHA256 = "447e3c6618a74f123a3f33c6fbae55d0981cd132ef6a305c5c162899a730613f"
MASK_NAMED_SHA256 = "6690b32790b2d8b7e597f51cc0f01cde88f3df1c79b6f65723b348a2a1b31210"
MASK_NAMES = ("map_safe", "visible", "core", "context", "boundary_ring")
PRIMARY_MINIMUM_PIXELS = {"core": 128, "context": 512}
TIE_TOLERANCE = 1.0e-12
DEFAULT_CHUNK_SIZE = 8
CHUNK_CONTROL_SIZES = (1, 8, 40)
PERMUTATION = tuple((7 * index + 3) % CANDIDATE_COUNT for index in range(CANDIDATE_COUNT))
INVERSE_PERMUTATION = tuple(
    (23 * (index - 3)) % CANDIDATE_COUNT for index in range(CANDIDATE_COUNT)
)
AFFINE_CASES = (0, 17, 37, 63)
AFFINE_TRANSFORMS = (
    ("positive-affine-low", 0.7, 0.1),
    ("positive-affine-high", 1.2, 0.0),
    ("polarity-inversion", -1.0, 1.0),
)
INTERPRETATION = (
    "model-free synthetic image-information finite-bank development gate only; "
    "not training, benchmarking, qualification, uncertainty calibration, or final-test evaluation"
)

PREFLIGHT_RELATIVE_PATH = "publication/arbitrary_plane_image_information_preflight.yaml"
RUNNER_RELATIVE_PATH = "training/run_arbitrary_plane_image_information.py"
SCORER_RELATIVE_PATHS = (
    "training/arbitrary_plane_image_information.py",
    "training/arbitrary_plane_image_secondary.py",
    "training/arbitrary_plane_image_candidate_scalar.py",
    "tests/test_arbitrary_plane_image_information.py",
    "tests/test_arbitrary_plane_image_secondary.py",
)
RUNNER_TEST_RELATIVE_PATH = "tests/test_run_arbitrary_plane_image_information.py"
FROZEN_DEPENDENCY_PATHS = (
    "training/run_arbitrary_plane_semantic_oracle.py",
    "training/arbitrary_plane_finite_candidates.py",
    "training/arbitrary_plane_semantic_oracle.py",
    "training/arbitrary_plane_rendered_generator.py",
    "training/arbitrary_plane_synthetic_generator.py",
    "training/arbitrary_plane_support.py",
    "training/arbitrary_plane_geometry.py",
    "training/arbitrary_plane_manifest.py",
    "training/arbitrary_plane_synthetic_ops.py",
    "training/arbitrary_plane_synthetic_observation.py",
    "publication/arbitrary_plane_oracle_pose_ranking_preflight.yaml",
    "publication/arbitrary_plane_synthetic_preflight.yaml",
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    PREFLIGHT_RELATIVE_PATH,
    *SCORER_RELATIVE_PATHS,
    *FROZEN_DEPENDENCY_PATHS,
)

NATIVE_OUTLINE_SLOT_SCHEDULE = tuple(
    (descriptor, domain)
    for descriptor, domains in (
        ("MIND", ("core", "context")),
        ("constant-within-support-MIND-null", ("core", "context")),
        ("support-penalized-MIND", ("core", "context")),
        ("HOG", ("core", "context", "boundary_ring")),
        ("normalized-gradient-like", ("core", "context", "boundary_ring")),
    )
    for domain in domains
)
SHUFFLED_OUTLINE_SLOT_SCHEDULE = tuple(
    (descriptor, domain)
    for descriptor in ("MIND", "constant-within-support-MIND-null")
    for domain in ("core", "context")
)
NATIVE_SLOT_SCHEDULE = tuple(
    (descriptor, domain, outline)
    for outline in OUTLINE_MODES
    for descriptor, domain in NATIVE_OUTLINE_SLOT_SCHEDULE
)
SHUFFLED_SLOT_SCHEDULE = tuple(
    (descriptor, domain, outline)
    for outline in OUTLINE_MODES
    for descriptor, domain in SHUFFLED_OUTLINE_SLOT_SCHEDULE
)

KEYSETS = {
    "resolved_config": {
        "schema", "preflight_sha256", "repository", "frozen_semantic_input",
        "atlas_assets", "descriptor_constants", "case_and_shuffle_contract",
        "environment", "model_independence", "data_access", "source_sha256",
        "resolved_config_sha256",
    },
    "repository_state": {"branch", "upstream", "head", "upstream_head", "worktree_clean"},
    "repository": {
        "branch", "source_parent_commit", "execution_commit", "origin_commit",
        "worktree_clean", "preflight_path", "preflight_git_blob_sha256",
        "preflight_checkout_sha256",
    },
    "source_receipt": {"relative_path", "git_blob_sha256", "checkout_sha256"},
    "frozen_semantic_input": {
        "source_commit", "output_relative_path", "result_json_sha256",
        "result_payload_sha256", "resolved_config_payload_sha256",
        "inventory_file_count", "inventory_total_bytes", "inventory_sha256",
        "support_index_sha256", "prepared_render_context_sha256",
        "prepared_candidate_annotation_context_sha256", "decoded_template_array_sha256",
        "scalar_conversion_array_sha256", "decoded_annotation_array_sha256",
    },
    "atlas_assets": {
        "template_path", "template_source_sha256", "template_decoded_receipt",
        "scalar_conversion_receipt", "annotation_path", "annotation_source_sha256",
        "annotation_decoded_receipt", "support_index_sha256", "global_tissue_voxel_count",
        "quantile_probabilities", "quantile_values",
    },
    "descriptor_constants": {
        "algorithm", "numeric_dtype", "scalar_padding", "candidate_scaling",
        "tie_tolerance", "context_radius_um", "primary_domain_minima", "mind",
        "hog", "normalized_gradient_like", "common_lattice",
    },
    "mind_constants": {
        "search_displacement_um", "gaussian_patch_sigma_um",
        "gaussian_truncate_sigma", "offset_order",
    },
    "hog_constants": {"cell_width_um", "orientation_bins", "block_cells"},
    "ngf_constants": {
        "gaussian_sigma_um", "gaussian_radius_um", "polarity_invariant",
    },
    "common_lattice_constants": {
        "intensity", "support", "canvas_center_xy_over_wh",
    },
    "pooled_strata_counts": {"challenging_appearance", "damaged"},
    "pooled_strata_membership": {"challenging_appearance", "damaged"},
    "config_affine_transform": {"name", "scale", "offset"},
    "model_independence": {
        "learned_checkpoint_dependencies", "previous_model_dependencies",
        "pretrained_feature_dependencies", "legacy_descriptor_dependencies",
        "initialization",
    },
    "data_access": {
        "allen_template_and_annotation", "synthetic_development",
        "deepslice_ground_truth", "real_lab_histology", "calibration_animals",
        "qualification_animals", "final_test_animals", "full_benchmark",
    },
    "case_and_shuffle_contract": {
        "base_count", "candidate_count", "output_shape_h_w", "outline_order",
        "orientation_counts", "pooled_strata_counts", "pooled_strata_membership",
        "shuffled_offset",
        "candidate_permutation", "chunk_sizes", "affine_case_indices",
        "affine_transforms", "streaming_contract",
    },
    "environment": {"python", "platform", "numpy", "scipy", "torch"},
    "array_receipt": {"dtype", "shape", "array_sha256"},
    "mask_receipt": {
        "dtype", "shape", "bitorder", "bit_count", "byte_count",
        "relative_path", "payload_sha256", "array_sha256",
    },
    "inline_mask_receipt": {
        "dtype", "shape", "bitorder", "bit_count", "byte_count",
        "packed_payload_sha256", "storage",
    },
    "case_mask_record": {
        "case_index", "semantic_case_payload_sha256", "paired_view_group_id",
        "pixel_pitch_um", "mask_receipts", "pixel_counts", "passed",
    },
    "prelaunch_failure": {
        "schema", "status", "failure_code", "execution_contract",
        "frozen_semantic_bindings", "thresholds", "case_mask_records", "failures",
        "score_blind_evidence", "data_access", "model_independence",
        "failure_payload_sha256",
    },
    "prelaunch_failure_item": {
        "case_index", "domain", "observed_pixel_count", "minimum_required_pixels",
    },
    "failure_execution_contract": {
        "execution_commit", "origin_commit", "branch", "worktree_clean",
        "preflight_sha256", "resolved_config", "resolved_config_sha256",
        "environment", "source_sha256",
    },
    "failure_thresholds": {"core", "context"},
    "mask_counts": {"map_safe", "visible", "core", "context", "boundary_ring"},
    "mask_receipt_map": {"map_safe", "visible", "core", "context", "boundary_ring"},
    "score_blind_evidence": {
        "all_64_masks_built", "frozen_replay_passed", "candidate_scalar_render_count",
        "descriptor_call_count", "score_landscape_count", "success_output_created",
    },
    "primary": {
        "schema", "case_index", "semantic_case_payload_sha256", "provenance",
        "frozen_replay", "target", "candidate_bank", "candidate_scalar_receipts",
        "score_domains", "outline_results", "payload_sha256",
    },
    "primary_provenance": {
        "animal_id", "specimen_id", "experiment_id", "atlas", "annotation_source",
        "scalar_source", "reporting_strata", "mask_only_Dice",
    },
    "reporting_strata": {
        "orientation_family", "appearance_family", "damage_event_types",
        "damage_event_count", "damage_union_fraction", "parent_brain_pixel_count",
        "visible_pixel_count", "challenging_appearance_member", "damaged_member",
        "outline_modes",
    },
    "frozen_replay": {
        "semantic_source_relative_path", "semantic_source_file_sha256",
        "semantic_case_payload_sha256", "finite_parent_receipt",
        "finite_parent_receipt_sha256", "case_rejection_records",
        "case_rejection_records_sha256", "outline_descendant_receipts",
        "outline_descendant_receipts_sha256", "candidate_bank_receipt_sha256",
        "replay_passed",
    },
    "target": {
        "parent_plane_realization_id", "paired_view_group_id", "truth_candidate_id",
        "truth_geometry_sha256", "pixel_pitch_um", "output_shape_h_w",
        "target_labels_receipt", "fixed_valid_mask_receipt",
    },
    "target_dewarp": {
        "direction", "scalar_padding", "model_input_image_receipt",
        "fixed_to_source_map_receipt", "dewarped_float32_receipt",
        "dewarped_float64_receipt",
    },
    "candidate_bank": {
        "finite_candidate_bank_id", "finite_candidate_receipt_sha256", "candidate_set_id",
        "ordered_candidate_ids", "ordered_candidate_ids_sha256", "truth_candidate_id",
        "truth_candidate_index", "truth_parent_geometry", "receipt",
    },
    "candidate_scalar_record": {
        "candidate_index", "candidate_id", "scalar", "annotation", "brain_mask",
        "crosscheck_passed",
        "payload_sha256",
    },
    "candidate_scalar_value": {
        "rendered_float32", "scaled_float64", "render_then_scale",
        "global_conversion",
    },
    "score_domain": {
        "domain", "mask_receipt", "pixel_count", "minimum_required_pixels",
    },
    "slot": {
        "status", "reason_code", "case_index", "bank_case_index", "target_case_index",
        "outline_mode", "descriptor", "domain", "domain_mask_receipt_sha256",
        "domain_pixel_count", "eligible_pixel_count", "eligible_block_count",
        "scores", "ranking", "metrics", "entered_gate", "payload_sha256",
    },
    "ranking": {
        "truth_index", "truth_candidate_id", "truth_score", "top1", "true_rank",
        "reciprocal_rank", "truth_versus_decoy_win_fraction", "truth_score_margin",
        "tied_maximum_indices", "tied_maximum_candidate_ids", "selected_index",
        "selected_candidate_id",
    },
    "pose_errors": {
        "rp2_angle_error_deg", "sign_aligned_offset_error_um",
        "corresponding_point_rms_um", "corresponding_point_p95_um",
    },
    "mind_metrics": {
        "target_vbar", "candidate_vbar", "supported_means",
        "candidate_exterior_fractions", "selected_pose_errors",
    },
    "hog_metrics": {"cell_pixels", "eligible_block_count", "block_weights_receipt"},
    "ngf_metrics": {
        "gaussian_radius_px", "effective_domain_count", "target_eta", "candidate_eta",
    },
    "primary_outline": {
        "outline_mode", "synthetic_realization_id", "synthetic_receipt",
        "target_dewarp", "score_slots", "payload_sha256",
    },
    "mask_only_dice": {
        "source_relative_path", "source_file_sha256", "source_case_payload_sha256",
        "source_target_receipt_sha256", "candidate_bank_id",
        "ordered_candidate_ids_sha256", "truth_candidate_id",
        "source_vector_json_pointer", "source_vector_sha256", "scorer_source_sha256",
        "runner_source_sha256", "values", "recomputed_ranking", "entered_gate",
        "payload_sha256",
    },
    "shuffled": {
        "schema", "bank_case_index", "target_case_index", "bank_identity",
        "target_identity", "common_lattice_resampling", "score_domains",
        "outline_results", "payload_sha256",
    },
    "bank_identity": {
        "bank_case_index", "source_primary_payload_sha256", "finite_candidate_bank_id",
        "finite_candidate_receipt_sha256", "ordered_candidate_ids",
        "ordered_candidate_ids_sha256", "truth_candidate_id", "source_pixel_pitch_um",
        "candidate_scalar_receipts_sha256",
    },
    "target_identity": {
        "target_case_index", "target_primary_payload_sha256", "paired_view_group_id",
        "target_outline_payload_sha256", "target_pixel_pitch_um",
        "score_domain_receipts_sha256",
    },
    "resampled_candidate": {
        "candidate_index", "candidate_id", "scalar_float64_receipt",
        "support_bool_receipt", "constant_null_scalar_float64_receipt",
        "constant_supported_mean", "payload_sha256",
    },
    "common_lattice": {
        "mapping", "output_shape_h_w", "coordinate_order", "source_pixel_pitch_um",
        "target_pixel_pitch_um", "coordinate_map_receipt", "intensity_resampler",
        "support_resampler", "null_construction_order", "resampled_candidates",
        "payload_sha256",
    },
    "shuffled_outline": {
        "outline_mode", "target_primary_outline_payload_sha256", "score_slots",
        "payload_sha256",
    },
    "control": {
        "schema", "case_index", "checks", "evidence_receipt_sha256", "payload_sha256",
    },
    "case_checks": {
        "source_replay_metadata_geometry", "candidate_scalar_annotation_mask",
        "dewarp_direction_and_masks", "rp2_and_xy_wh", "scorer_signature_exclusion",
        "target_domain_invariance", "accurate_absent_core_identity",
        "shuffled_binding", "mask_only_verification", "landscape_controls",
        "affine_and_polarity",
    },
    "landscape_control": {
        "source_slot_payload_sha256", "source_status", "case_index", "bank_case_index",
        "target_case_index", "outline_mode", "descriptor", "domain", "permutation",
        "chunks", "status", "reason_code", "passed", "payload_sha256",
    },
    "permutation_control": {
        "mapping", "permutation", "nonidentity_bijection", "original_score_vector_sha256",
        "permuted_score_vector_sha256", "inverse_reindexed_score_vector_sha256",
        "original_ranking_sha256", "recomputed_ranking_sha256", "passed",
    },
    "chunk_control": {
        "chunk_sizes", "score_vector_sha256", "byte_identical",
        "ranking_payload_sha256", "passed",
    },
    "basic_control": {"status", "reason_code", "passed", "evidence_sha256"},
    "affine_transform": {
        "name", "scale", "offset", "scalar_padding", "ranking_payload_sha256",
        "top1", "true_rank", "tied_maximum_candidate_ids", "passed",
    },
    "affine_slot": {
        "case_index", "outline_mode", "descriptor", "domain",
        "source_slot_payload_sha256", "transforms", "status", "reason_code",
        "passed", "payload_sha256",
    },
    "global_controls": {
        "schema", "frozen_inventory_audit", "source_and_signature_audit",
        "affine_and_polarity_controls", "evidence_receipt_sha256", "payload_sha256",
    },
    "frozen_inventory_audit": {
        "expected_file_count", "observed_file_count", "expected_total_bytes",
        "observed_total_bytes", "expected_inventory_sha256",
        "observed_inventory_sha256", "passed",
    },
    "source_signature_audit": {
        "execution_commit", "origin_commit", "worktree_clean", "source_records",
        "scorer_signature_records", "model_dependency_records", "passed",
    },
    "scorer_signature_record": {
        "function", "signature", "source_sha256", "forbidden_tokens",
        "forbidden_matches", "passed",
    },
    "model_dependency_record": {"dependency", "declared_values", "passed"},
    "affine_summary": {
        "required_case_indices", "case_control_payload_sha256", "applicable_slot_count",
        "authenticated_not_applicable_count", "passed",
    },
    "inventory_item": {"path", "size_bytes", "sha256"},
    "slot_summary": {
        "scope", "descriptor", "domain", "outline_mode", "entered_gate",
        "eligible_base_count", "insufficient_base_count", "top1_success_count",
        "top1_rate", "wilson_95", "mean_reciprocal_rank", "median_true_rank",
        "median_truth_versus_decoy_win_fraction", "median_truth_score_margin",
    },
    "stratum_summary": {
        "stratum_type", "stratum_value", "endpoint", "base_indices", "base_count",
        "top1_success_count", "top1_rate", "mean_reciprocal_rank",
        "median_true_rank", "median_truth_score_margin",
    },
    "paired_summary": {
        "comparison", "base_count", "top1_rate_difference", "median_rank_difference",
        "median_truth_margin_difference",
    },
    "result_metrics": {
        "native_slot_summaries", "shuffled_slot_summaries",
        "reporting_stratum_summaries", "pooled_safeguard_summaries",
        "paired_outline_comparisons",
    },
    "atomic_gate": {
        "gate_id", "source_metric_pointer", "operator", "threshold", "observed",
        "passed", "evidence_sha256",
    },
    "gates": {
        "global_controls_payload_sha256", "atomic_checks", "passed", "decision",
    },
    "result": {
        "schema", "interpretation", "resolved_config_sha256", "pre_result_inventory",
        "pre_result_inventory_sha256", "primary_case_payload_sha256",
        "shuffled_case_payload_sha256", "control_payload_sha256", "metrics", "gates",
        "data_access", "model_independence", "result_payload_sha256",
    },
}

DATA_ACCESS = {
    "allen_template_and_annotation": True,
    "synthetic_development": True,
    "deepslice_ground_truth": False,
    "real_lab_histology": False,
    "calibration_animals": False,
    "qualification_animals": False,
    "final_test_animals": False,
    "full_benchmark": False,
}
MODEL_INDEPENDENCE = {
    "learned_checkpoint_dependencies": [],
    "previous_model_dependencies": [],
    "pretrained_feature_dependencies": [],
    "legacy_descriptor_dependencies": [],
    "initialization": "deterministic frozen case streams only; no learned initialization",
}

MASK_COMPACT_RECORDS = (
    (0,47730,15655,11321,25480,9825),(1,46504,19496,15608,29091,9595),
    (2,47480,17189,8776,27716,10527),(3,47762,15045,12454,23141,8096),
    (4,46780,9792,7542,17657,7865),(5,48705,14979,11104,25250,10271),
    (6,47008,13999,9847,23318,9349),(7,47528,22830,18871,33537,10707),
    (8,48705,12287,8602,22856,10569),(9,48259,11857,7688,21080,9223),
    (10,46835,11021,6220,18798,7777),(11,47858,20852,14219,35264,14412),
    (12,48249,7209,4380,13899,6690),(13,48195,21827,17391,32012,10185),
    (14,48705,23068,18715,32129,9061),(15,46536,16149,10984,24735,8586),
    (16,48705,25130,20119,36208,11078),(17,48705,13318,10159,21537,8219),
    (18,47510,12310,7815,20013,7703),(19,47878,17142,13435,23869,6735),
    (20,47772,3187,1840,9812,6625),(21,48705,20728,16928,30152,9424),
    (22,47719,14676,10872,22230,7554),(23,48241,2598,884,8716,6118),
    (24,47548,13283,10505,21592,8309),(25,48705,17230,10513,29113,11883),
    (26,48247,12749,5741,22106,9357),(27,48236,7045,5349,13661,6616),
    (28,47850,3838,2308,7591,3753),(29,48705,9915,7272,16246,6331),
    (30,47784,9322,7306,15335,6013),(31,48705,17094,13788,25830,8736),
    (32,48705,16153,11953,27635,11482),(33,47251,19859,15436,31519,11662),
    (34,46809,14624,7969,24400,9776),(35,47425,9753,7319,15781,6028),
    (36,48705,10185,6662,18190,8005),(37,47496,1391,236,5417,4026),
    (38,48705,13355,10621,20724,7369),(39,48005,16223,10432,27564,11341),
    (40,46985,14380,11778,21770,7390),(41,48086,19540,15874,30253,10713),
    (42,48705,12154,8435,18347,6193),(43,46993,6030,2986,12902,6872),
    (44,48202,11650,9136,18481,6831),(45,47640,22592,17388,32446,9854),
    (46,47976,21560,16991,33141,11586),(47,48705,18110,12347,29170,11060),
    (48,47632,19162,13107,27512,8350),(49,48705,9546,6603,17312,7766),
    (50,47830,3816,1836,11327,7511),(51,48705,6058,2702,13318,7260),
    (52,47625,22690,18133,32922,10234),(53,47003,23787,18571,34657,10870),
    (54,47797,18650,15276,26967,8317),(55,47706,18785,14727,29686,10901),
    (56,48089,2235,548,7507,5272),(57,48018,9638,6248,16938,7300),
    (58,48705,21088,14349,32646,11558),(59,47256,21921,18047,32909,10988),
    (60,47232,4081,2718,8993,4912),(61,46916,6442,3795,15168,8726),
    (62,48705,10156,4407,17970,7814),(63,47745,6027,3173,12996,6969),
)


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
        default=_json_scalar,
    ).encode("utf-8")


def canonical_payload_sha256(value: object, excluded_key: str | None = None) -> str:
    if excluded_key is not None:
        if not isinstance(value, Mapping):
            raise TypeError("a self-hashed payload must be a mapping")
        value = {key: item for key, item in value.items() if key != excluded_key}
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _self_hash(record: dict[str, object], field: str) -> dict[str, object]:
    if field in record:
        raise ValueError(f"self-hash field {field} was already populated")
    record[field] = canonical_payload_sha256(record)
    return record


def validate_payload_keys(value: object, schema_name: str) -> dict[str, object]:
    if schema_name not in KEYSETS or not isinstance(value, dict) or set(value) != KEYSETS[schema_name]:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{schema_name} keys differ from the frozen schema: {actual}")
    return value


def _read_legacy_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not one JSON object")
    return value


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _read_strict_json(path: Path) -> dict[str, object]:
    """Read a newly produced JSON object in its exact canonical byte form."""
    payload = path.read_bytes()
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not canonical UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not one JSON object")
    if payload != _canonical_bytes(value):
        raise ValueError(f"{path} bytes are not exact canonical compact JSON")
    return value


def _read_json(path: Path) -> dict[str, object]:
    """Read a newly produced JSON artifact using the frozen byte contract."""
    return _read_strict_json(path)


def _atomic_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: object) -> str:
    return _atomic_bytes(path, _canonical_bytes(value))


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=not binary
    )
    return completed.stdout.strip() if not binary else completed.stdout


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    return hashlib.sha256(_git("show", f"{commit}:{relative_path}", binary=True)).hexdigest()


def _paths_overlap(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def guard_output_roots(
    ordinary_output: Path | None,
    failed_output: Path | None,
) -> tuple[Path | None, Path | None]:
    """Reject output roots that could overwrite or contain an authenticated input."""
    ordinary = None if ordinary_output is None else Path(ordinary_output).resolve()
    failed = None if failed_output is None else Path(failed_output).resolve()
    if ordinary is not None and failed is not None and _paths_overlap(ordinary, failed):
        raise ValueError("ordinary and failed output roots must not overlap")
    protected = {
        FROZEN_SEMANTIC_OUTPUT.resolve(),
        (ROOT / ATLAS_TEMPLATE_URI).resolve().parent,
        (ROOT / ATLAS_ANNOTATION_URI).resolve().parent,
        *((ROOT / relative).resolve().parent for relative in SOURCE_RELATIVE_PATHS),
    }
    for output in (ordinary, failed):
        if output is None:
            continue
        if any(_paths_overlap(output, item) for item in protected):
            raise ValueError(
                "output root overlaps frozen semantic, source, test, publication, or atlas input"
            )
    return ordinary, failed


def repository_state() -> dict[str, object]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("image-information execution requires a clean tracked and untracked worktree")
    branch = str(_git("rev-parse", "--abbrev-ref", "HEAD"))
    upstream = str(_git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"))
    head = str(_git("rev-parse", "HEAD"))
    upstream_head = str(_git("rev-parse", "@{upstream}"))
    if branch != EXPECTED_BRANCH or upstream != f"origin/{EXPECTED_BRANCH}" or head != upstream_head:
        raise RuntimeError("execution requires codex/joint-registration exactly at its origin upstream")
    for ancestor in (PREFLIGHT_COMMIT, SCORER_COMMIT, FROZEN_SEMANTIC_SOURCE_COMMIT):
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, head], cwd=ROOT
        ).returncode != 0:
            raise RuntimeError(f"required frozen source commit {ancestor} is not an ancestor of HEAD")
    return {
        "branch": branch,
        "upstream": upstream,
        "head": head,
        "upstream_head": upstream_head,
        "worktree_clean": True,
    }


def _source_hash_receipts(repository: dict[str, object]) -> list[dict[str, str]]:
    head = str(repository["head"])
    receipts = []
    for path in SOURCE_RELATIVE_PATHS:
        authority = (
            FROZEN_SEMANTIC_SOURCE_COMMIT if path in FROZEN_DEPENDENCY_PATHS
            else PREFLIGHT_COMMIT if path == PREFLIGHT_RELATIVE_PATH
            else SCORER_COMMIT if path in SCORER_RELATIVE_PATHS
            else head
        )
        authority_hash = _git_blob_sha256(authority, path)
        head_hash = _git_blob_sha256(head, path)
        if head_hash != authority_hash:
            raise RuntimeError(f"{path} differs from its frozen authority commit {authority}")
        receipt = {
            "relative_path": path,
            "git_blob_sha256": head_hash,
            "checkout_sha256": _file_sha256(ROOT / path),
        }
        receipts.append(validate_payload_keys(receipt, "source_receipt"))
    return receipts


def _inventory(root: Path, excluded: set[str] | None = None) -> list[dict[str, object]]:
    excluded = excluded or set()
    records = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        )
    return records


def _authenticate_frozen_semantic_output(
    source_records: list[dict[str, str]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    result_path = FROZEN_SEMANTIC_OUTPUT / "result.json"
    if _file_sha256(result_path) != FROZEN_SEMANTIC_RESULT_FILE_SHA256:
        raise ValueError("frozen semantic result file hash changed")
    inventory = _inventory(FROZEN_SEMANTIC_OUTPUT)
    if (
        len(inventory) != FROZEN_SEMANTIC_FILE_COUNT
        or sum(int(item["size_bytes"]) for item in inventory) != FROZEN_SEMANTIC_TOTAL_BYTES
        or canonical_payload_sha256(inventory) != FROZEN_SEMANTIC_INVENTORY_SHA256
    ):
        raise ValueError("frozen semantic output inventory changed")
    result = verify_semantic_written_result(FROZEN_SEMANTIC_OUTPUT)
    config = result.get("resolved_config", {})
    if (
        config.get("resolved_config_sha256") != FROZEN_SEMANTIC_CONFIG_PAYLOAD_SHA256
        or config.get("source_commit") != FROZEN_SEMANTIC_SOURCE_COMMIT
        or result.get("support_index_sha256") != SUPPORT_INDEX_SHA256
        or result.get("prepared_render_context_sha256") != PREPARED_RENDER_CONTEXT_SHA256
        or result.get("prepared_candidate_annotation_context_sha256")
        != PREPARED_CANDIDATE_CONTEXT_SHA256
    ):
        raise ValueError("frozen semantic result/config/context binding changed")
    if source_records is not None:
        current = {item["relative_path"]: item for item in source_records}
        for path in FROZEN_DEPENDENCY_PATHS:
            if (
                current[path]["git_blob_sha256"] != config["source_sha256"][path]
                or current[path]["checkout_sha256"]
                != config["checkout_source_sha256"][path]
            ):
                raise ValueError(
                    f"frozen dependency {path} no longer matches the authenticated semantic receipts"
                )
    primary = [
        _read_legacy_json(
            FROZEN_SEMANTIC_OUTPUT / "primary" / f"case-{index:03d}.json"
        )
        for index in range(CASE_COUNT)
    ]
    return result, primary


def derive_frozen_pooled_membership(
    frozen_primary: list[dict[str, object]],
) -> dict[str, list[int]]:
    """Derive the two predeclared pools from authenticated semantic metadata only."""
    if (
        len(frozen_primary) != CASE_COUNT
        or [item.get("case_index") for item in frozen_primary]
        != list(range(CASE_COUNT))
    ):
        raise ValueError("pooled membership requires exact semantic cases 0 through 63")
    challenging, damaged = [], []
    for case_index, record in enumerate(frozen_primary):
        strata = record.get("reporting_strata")
        if not isinstance(strata, dict):
            raise ValueError("semantic case lacks authenticated reporting strata")
        family = strata.get("appearance_family")
        damage_count = strata.get("damage_event_count")
        if not isinstance(family, str) or type(damage_count) is not int or damage_count < 0:
            raise ValueError("semantic appearance/damage strata changed type")
        if family in {"label-conditioned", "template-label-mixture"}:
            challenging.append(case_index)
        if damage_count >= 1:
            damaged.append(case_index)
    observed = {
        "challenging_appearance": challenging,
        "damaged": damaged,
    }
    expected = {
        name: list(indices) for name, indices in FROZEN_POOLED_MEMBERSHIP.items()
    }
    if observed != expected:
        raise ValueError("authenticated pooled membership changed from exact 22/27 bases")
    return validate_payload_keys(observed, "pooled_strata_membership")


def _inline_mask_receipt(mask: np.ndarray) -> dict[str, object]:
    value = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    packed = np.packbits(value.reshape(-1, order="C"), bitorder="little").tobytes()
    header = {"dtype": "|b1", "shape": list(value.shape), "bitorder": "little"}
    record = {
        "dtype": "|b1",
        "shape": list(value.shape),
        "bitorder": "little",
        "bit_count": int(value.size),
        "byte_count": len(packed),
        "packed_payload_sha256": hashlib.sha256(_canonical_bytes(header) + packed).hexdigest(),
        "storage": "not_persisted",
    }
    return validate_payload_keys(record, "inline_mask_receipt")


def _persisted_mask_receipt(case_index: int, name: str, mask: np.ndarray) -> dict[str, object]:
    value = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    packed = np.packbits(value.reshape(-1, order="C"), bitorder="little").tobytes()
    relative = f"masks/case-{int(case_index):03d}-{name.replace('_', '-')}.bin"
    record = {
        "dtype": "|b1",
        "shape": list(value.shape),
        "bitorder": "little",
        "bit_count": int(value.size),
        "byte_count": len(packed),
        "relative_path": relative,
        "payload_sha256": hashlib.sha256(packed).hexdigest(),
        "array_sha256": _array_receipt(value)["array_sha256"],
    }
    return validate_payload_keys(record, "mask_receipt")


def _write_mask(output: Path, receipt: dict[str, object], mask: np.ndarray) -> None:
    value = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    packed = np.packbits(value.reshape(-1, order="C"), bitorder="little").tobytes()
    if (
        receipt.get("dtype") != "|b1"
        or receipt.get("shape") != list(value.shape)
        or receipt.get("bitorder") != "little"
        or receipt.get("bit_count") != value.size
        or receipt.get("byte_count") != len(packed)
        or receipt.get("payload_sha256") != hashlib.sha256(packed).hexdigest()
        or receipt.get("array_sha256") != _array_receipt(value)["array_sha256"]
    ):
        raise ValueError("mask receipt does not match its Boolean array")
    if _atomic_bytes(output / str(receipt["relative_path"]), packed) != receipt["payload_sha256"]:
        raise ValueError("packed mask changed during atomic write")


def _semantic_case_file(case_index: int) -> Path:
    return FROZEN_SEMANTIC_OUTPUT / "primary" / f"case-{int(case_index):03d}.json"


def _verified_descendant(
    parent: dict[str, object],
    support_index: dict[str, object],
    case_index: int,
    synthetic_seed: str,
    mode: str,
    expected: dict[str, object],
) -> dict[str, object]:
    descendant = make_arbitrary_plane_synthetic_realization(
        parent,
        support_index,
        root_seed=int(synthetic_seed, 16),
        sample_index=case_index,
        synthetic_stratum="ordinary",
        outline_mode=mode,
        finite_parent_generator_source_commit=FROZEN_SEMANTIC_SOURCE_COMMIT,
    )
    if (
        expected["mode"] != mode
        or descendant["synthetic_realization_id"] != expected["synthetic_realization_id"]
        or synthetic_realization_receipt(descendant) != expected["synthetic_receipt"]
    ):
        raise ValueError(f"case {case_index} {mode} descendant did not replay exactly")
    return descendant


def replay_frozen_case(
    case_index: int,
    frozen_record: dict[str, object],
    render_context: Mapping[str, object],
    candidate_context: Mapping[str, object],
    support_index: dict[str, object],
) -> dict[str, object]:
    """Regenerate one accepted parent, bank and its three exact descendants only."""
    case_index = int(case_index)
    if int(frozen_record.get("case_index", -1)) != case_index:
        raise ValueError("semantic sidecar case index changed")
    attempt = int(frozen_record["accepted_case_attempt_index"])
    seeds = case_seed_lineage(case_index, attempt)
    if seeds != frozen_record["accepted_case_field_stream_seed_uint64"]:
        raise ValueError("accepted semantic seed lineage changed")
    parent = make_finite_arbitrary_plane_render_from_context(
        render_context,
        "development",
        int(seeds["finite-parent"], 16),
        OUTPUT_SHAPE,
        sample_index=case_index,
        margin_um=MARGIN_UM,
        animal_id=None,
        specimen_id=None,
        experiment_id=None,
        max_rejection_attempts=1,
        minimum_brain_pixels=64,
        generator_source_commit=FROZEN_SEMANTIC_SOURCE_COMMIT,
    )
    if finite_render_receipt(parent) != frozen_record["finite_parent_receipt"]:
        raise ValueError(f"case {case_index} finite parent did not replay exactly")
    bank = make_arbitrary_plane_finite_candidate_bank_from_context(
        parent,
        candidate_context,
        support_index,
        finite_parent_generator_source_commit=FROZEN_SEMANTIC_SOURCE_COMMIT,
    )
    bank_receipt = finite_candidate_bank_receipt(bank)
    if (
        bank_receipt != frozen_record["candidate_bank_receipt"]
        or bank["finite_candidate_bank_id"] != frozen_record["candidate_bank_id"]
        or bank["finite_candidate_receipt_sha256"]
        != frozen_record["candidate_bank_receipt_sha256"]
        or list(bank["ordered_candidate_ids"]) != frozen_record["ordered_candidate_ids"]
    ):
        raise ValueError(f"case {case_index} candidate bank did not replay exactly")
    descendant_receipts, descendant_ids = [], []
    first_target = None
    first_masks = None
    for mode, expected in zip(OUTLINE_MODES, frozen_record["outline_descendants"], strict=True):
        descendant = _verified_descendant(
            parent, support_index, case_index, seeds["synthetic"], mode, expected
        )
        if descendant["paired_view_group_id"] != frozen_record["paired_view_group_id"]:
            raise ValueError("outline descendant changed paired-view identity")
        target = build_oracle_target(descendant)
        arrays = descendant["arrays"]
        current_masks = target_score_masks(
            arrays["fixed_to_source_map"],
            arrays["source_map_domain_mask"],
            arrays["fixed_map_domain_mask"],
            arrays["source_valid_correspondence_mask"],
            arrays["fixed_valid_correspondence_mask"],
            float(target["pixel_pitch_um"]),
        )
        if first_target is None:
            first_target = {
                "labels": np.array(target["labels"], copy=True),
                "fixed_valid_mask": np.array(target["fixed_valid_mask"], copy=True),
                "pixel_pitch_um": float(target["pixel_pitch_um"]),
            }
            first_masks = {name: np.array(current_masks[name], copy=True) for name in MASK_NAMES}
        elif (
            target["pixel_pitch_um"] != first_target["pixel_pitch_um"]
            or not np.array_equal(target["labels"], first_target["labels"])
            or not np.array_equal(target["fixed_valid_mask"], first_target["fixed_valid_mask"])
            or any(
                not np.array_equal(current_masks[name], first_masks[name])
                for name in MASK_NAMES
            )
        ):
            raise ValueError("paired descendants changed semantic truth or target-only masks")
        descendant_receipts.append(synthetic_realization_receipt(descendant))
        descendant_ids.append(descendant["synthetic_realization_id"])
        del descendant
    if first_target is None or first_masks is None:
        raise RuntimeError("frozen case did not produce its three outline descendants")
    first = first_target
    if (
        _array_receipt(first["labels"]) != frozen_record["target"]["labels_receipt"]
        or _array_receipt(first["fixed_valid_mask"])
        != frozen_record["target"]["mask_receipt"]
        or float(first["pixel_pitch_um"]) != float(frozen_record["target"]["pixel_pitch_um"])
    ):
        raise ValueError("replayed semantic target does not match its frozen receipt")
    return {
        "case_index": case_index,
        "semantic_record": frozen_record,
        "parent": parent,
        "bank": bank,
        "descendant_receipts": descendant_receipts,
        "descendant_ids": descendant_ids,
        "synthetic_seed": seeds["synthetic"],
        "target": first,
        "masks": first_masks,
    }


def build_score_blind_masks(
    replay_case: Callable[[int], Mapping[str, object]],
    case_count: int = CASE_COUNT,
) -> list[dict[str, object]]:
    """Build only replay- and target-derived masks; no image scorer is reachable here."""
    records = []
    compact = []
    for case_index in range(int(case_count)):
        replayed = replay_case(case_index)
        semantic = replayed["semantic_record"]
        masks = replayed["masks"]
        counts = {name: int(np.asarray(masks[name], dtype=bool).sum()) for name in MASK_NAMES}
        passed = all(counts[name] >= threshold for name, threshold in PRIMARY_MINIMUM_PIXELS.items())
        record = {
            "case_index": case_index,
            "semantic_case_payload_sha256": semantic["case_payload_sha256"],
            "paired_view_group_id": semantic["paired_view_group_id"],
            "pixel_pitch_um": float(replayed["target"]["pixel_pitch_um"]),
            "mask_receipts": validate_payload_keys(
                {name: _inline_mask_receipt(masks[name]) for name in MASK_NAMES},
                "mask_receipt_map",
            ),
            "pixel_counts": validate_payload_keys(counts, "mask_counts"),
            "passed": bool(passed),
        }
        records.append(validate_payload_keys(record, "case_mask_record"))
        compact.append([case_index, *(counts[name] for name in MASK_NAMES)])
    if int(case_count) == CASE_COUNT:
        named = [
            {"case_index": row[0], **dict(zip(MASK_NAMES, row[1:], strict=True))}
            for row in compact
        ]
        if (
            tuple(tuple(value) for value in compact) != MASK_COMPACT_RECORDS
            or canonical_payload_sha256(compact) != MASK_COMPACT_SHA256
            or canonical_payload_sha256(named) != MASK_NAMED_SHA256
        ):
            raise ValueError("score-blind mask counts differ from the frozen replay")
    return records


def expected_output_files() -> set[str]:
    files = {"resolved_config.json", "global_controls.json", "result.json"}
    files.update(f"primary/case-{index:03d}.json" for index in range(CASE_COUNT))
    files.update(f"shuffled/case-{index:03d}.json" for index in range(CASE_COUNT))
    files.update(f"controls/case-{index:03d}.json" for index in range(CASE_COUNT))
    files.update(
        f"masks/case-{index:03d}-{name.replace('_', '-')}.bin"
        for index in range(CASE_COUNT)
        for name in MASK_NAMES
    )
    if len(files) != 515:
        raise RuntimeError("the frozen output inventory does not contain exactly 515 paths")
    return files


def _validate_prelaunch_case_masks(
    case_mask_records: object,
) -> list[dict[str, object]]:
    if (
        not isinstance(case_mask_records, list)
        or len(case_mask_records) != CASE_COUNT
        or [item.get("case_index") if isinstance(item, dict) else None for item in case_mask_records]
        != list(range(CASE_COUNT))
    ):
        raise ValueError("prelaunch receipt requires all 64 cases in exact order 0 through 63")
    for case_index, item in enumerate(case_mask_records):
        validate_payload_keys(item, "case_mask_record")
        _require_sha256(
            item["semantic_case_payload_sha256"], "prelaunch semantic case payload"
        )
        if (
            type(item["case_index"]) is not int
            or item["case_index"] != case_index
            or not isinstance(item["paired_view_group_id"], str)
            or type(item["pixel_pitch_um"]) is not float
            or not math.isfinite(item["pixel_pitch_um"])
            or item["pixel_pitch_um"] <= 0.0
            or not isinstance(item["mask_receipts"], dict)
            or set(item["mask_receipts"]) != set(MASK_NAMES)
            or not isinstance(item["pixel_counts"], dict)
            or set(item["pixel_counts"]) != set(MASK_NAMES)
            or type(item["passed"]) is not bool
        ):
            raise ValueError("prelaunch case-mask identity/type/order changed")
        for name in MASK_NAMES:
            receipt = validate_payload_keys(
                item["mask_receipts"][name], "inline_mask_receipt"
            )
            if (
                receipt["dtype"] != "|b1"
                or receipt["shape"] != list(OUTPUT_SHAPE)
                or receipt["bitorder"] != "little"
                or receipt["bit_count"] != math.prod(OUTPUT_SHAPE)
                or receipt["byte_count"] != math.prod(OUTPUT_SHAPE) // 8
                or receipt["storage"] != "not_persisted"
            ):
                raise ValueError("prelaunch inline mask encoding changed")
            _require_sha256(
                receipt["packed_payload_sha256"], "prelaunch packed mask payload"
            )
            count = item["pixel_counts"][name]
            if type(count) is not int or not 0 <= count <= math.prod(OUTPUT_SHAPE):
                raise ValueError("prelaunch mask count is not a bounded integer")
        expected_passed = all(
            item["pixel_counts"][name] >= threshold
            for name, threshold in PRIMARY_MINIMUM_PIXELS.items()
        )
        if item["passed"] is not expected_passed:
            raise ValueError("prelaunch case pass flag differs from its mask counts")
    return case_mask_records


def _prelaunch_failures(
    case_mask_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    failures = []
    for case in case_mask_records:
        for domain, threshold in PRIMARY_MINIMUM_PIXELS.items():
            observed = case["pixel_counts"][domain]
            if observed < threshold:
                failures.append(
                    validate_payload_keys(
                        {
                            "case_index": case["case_index"],
                            "domain": domain,
                            "observed_pixel_count": observed,
                            "minimum_required_pixels": threshold,
                        },
                        "prelaunch_failure_item",
                    )
                )
    failures.sort(key=lambda item: (item["case_index"], item["domain"]))
    return failures


def write_prelaunch_failure(
    failed_output: Path,
    success_output: Path,
    resolved_config: dict[str, object],
    case_mask_records: list[dict[str, object]],
) -> dict[str, object]:
    success_output, failed_output = guard_output_roots(success_output, failed_output)
    if failed_output.exists() or success_output.exists():
        raise FileExistsError("ordinary and failed image-information outputs must both be fresh")
    _validate_prelaunch_case_masks(case_mask_records)
    failures = _prelaunch_failures(case_mask_records)
    if not failures:
        raise ValueError("a prelaunch failure receipt requires at least one insufficient primary domain")
    repository = resolved_config["repository"]
    execution_contract = {
        "execution_commit": repository["execution_commit"],
        "origin_commit": repository["origin_commit"],
        "branch": repository["branch"],
        "worktree_clean": repository["worktree_clean"],
        "preflight_sha256": resolved_config["preflight_sha256"],
        "resolved_config": copy.deepcopy(resolved_config),
        "resolved_config_sha256": resolved_config["resolved_config_sha256"],
        "environment": copy.deepcopy(resolved_config["environment"]),
        "source_sha256": copy.deepcopy(resolved_config["source_sha256"]),
    }
    validate_payload_keys(execution_contract, "failure_execution_contract")
    thresholds = validate_payload_keys(copy.deepcopy(PRIMARY_MINIMUM_PIXELS), "failure_thresholds")
    score_blind = {
        "all_64_masks_built": True,
        "frozen_replay_passed": True,
        "candidate_scalar_render_count": 0,
        "descriptor_call_count": 0,
        "score_landscape_count": 0,
        "success_output_created": False,
    }
    validate_payload_keys(score_blind, "score_blind_evidence")
    record = {
        "schema": "anatomy-tracker.arbitrary-plane-image-information-prelaunch-failure/v1",
        "status": "failed_before_scoring",
        "failure_code": "INSUFFICIENT_PRIMARY_DOMAIN",
        "execution_contract": execution_contract,
        "frozen_semantic_bindings": copy.deepcopy(resolved_config["frozen_semantic_input"]),
        "thresholds": thresholds,
        "case_mask_records": copy.deepcopy(case_mask_records),
        "failures": failures,
        "score_blind_evidence": score_blind,
        "data_access": copy.deepcopy(DATA_ACCESS),
        "model_independence": copy.deepcopy(MODEL_INDEPENDENCE),
    }
    _self_hash(record, "failure_payload_sha256")
    validate_payload_keys(record, "prelaunch_failure")
    failed_output.mkdir(parents=True, exist_ok=False)
    _atomic_json(failed_output / "prelaunch_failure.json", record)
    actual = [path.relative_to(failed_output).as_posix() for path in failed_output.rglob("*") if path.is_file()]
    if (
        actual != ["prelaunch_failure.json"]
        or _read_strict_json(failed_output / actual[0]) != record
    ):
        raise ValueError("prelaunch failure one-file tree did not verify")
    return record


def _score_vector_sha256(scores: list[float] | np.ndarray) -> str:
    return canonical_payload_sha256(
        {"scores": np.asarray(scores, dtype=np.float64).tolist()}
    )


def _unit_score_vector(result: dict[str, object]) -> np.ndarray:
    values = np.asarray(result["scores"], dtype=np.float64)
    if (
        values.shape != (CANDIDATE_COUNT,)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("every frozen score landscape must be forty finite values in [0,1]")
    return values


def _domain_mask_receipt_sha256(score_domain_record: dict[str, object]) -> str:
    validate_payload_keys(score_domain_record, "score_domain")
    validate_payload_keys(score_domain_record["mask_receipt"], "mask_receipt")
    return canonical_payload_sha256(score_domain_record["mask_receipt"])


def _float64_vector_bytes(scores: list[float] | np.ndarray) -> bytes:
    values = np.ascontiguousarray(np.asarray(scores, dtype=np.dtype("<f8")))
    if values.shape != (CANDIDATE_COUNT,):
        raise ValueError("score-vector byte comparison requires exactly forty values")
    return values.tobytes(order="C")


def _secondary_eligibility(
    descriptor: str,
    domain_name: str,
    domain_mask: np.ndarray,
    pixel_pitch_um: float,
) -> tuple[int | None, int | None]:
    domain = np.asarray(domain_mask, dtype=bool)
    if descriptor == "HOG":
        q = max(4, int(np.floor(400.0 / float(pixel_pitch_um) + 0.5)))
        if domain_name == "boundary_ring":
            count = int(np.count_nonzero(hog_boundary_ring_weights(domain, q)))
        else:
            count = int(np.count_nonzero(hog_complete_block_mask(domain, q)))
        return None, count
    if descriptor == "normalized-gradient-like":
        return int(np.count_nonzero(ngf_evaluation_domain(domain, pixel_pitch_um))), None
    return int(np.count_nonzero(domain)), None


def _validate_slot_indices(
    case_index: int | None,
    bank_case_index: int | None,
    target_case_index: int | None,
) -> tuple[int | None, int | None, int | None]:
    native = case_index is not None and bank_case_index is None and target_case_index is None
    shuffled = case_index is None and bank_case_index is not None and target_case_index is not None
    if native:
        case_index = int(case_index)
        if not 0 <= case_index < CASE_COUNT:
            raise ValueError("native case_index is outside the frozen 64 cases")
        return case_index, None, None
    if shuffled:
        bank_case_index, target_case_index = int(bank_case_index), int(target_case_index)
        if (
            not 0 <= bank_case_index < CASE_COUNT
            or target_case_index != (bank_case_index + 17) % CASE_COUNT
        ):
            raise ValueError("shuffled indices do not follow target=(bank+17)%64")
        return None, bank_case_index, target_case_index
    raise ValueError("slot indices must be exactly native (i,null,null) or shuffled (null,i,j)")


def _call_scorer(
    descriptor: str,
    domain_name: str,
    target_image: np.ndarray,
    candidate_images: np.ndarray,
    domain_mask: np.ndarray,
    pixel_pitch_um: float,
    *,
    candidate_support: np.ndarray | None,
    target_visible: np.ndarray | None,
    padding_value: float,
    chunk_size: int,
) -> dict[str, object] | None:
    if descriptor in {"MIND", "constant-within-support-MIND-null"}:
        return score_mind_candidates(
            target_image,
            candidate_images,
            domain_mask,
            pixel_pitch_um,
            padding_value=padding_value,
            chunk_size=chunk_size,
        )
    if descriptor == "support-penalized-MIND":
        if candidate_support is None or target_visible is None:
            raise ValueError("support-penalized MIND requires explicit support and visible masks")
        return score_support_penalized_mind_candidates(
            target_image,
            candidate_images,
            domain_mask,
            target_visible,
            candidate_support,
            pixel_pitch_um,
            padding_value=padding_value,
            chunk_size=chunk_size,
        )
    if descriptor == "HOG":
        return score_hog_candidates(
            target_image,
            candidate_images,
            domain_mask,
            pixel_pitch_um,
            boundary_ring=domain_name == "boundary_ring",
            padding_value=padding_value,
            chunk_size=chunk_size,
        )
    if descriptor == "normalized-gradient-like":
        return score_ngf_candidates(
            target_image,
            candidate_images,
            domain_mask,
            pixel_pitch_um,
            padding_value=padding_value,
            chunk_size=chunk_size,
        )
    raise ValueError(f"unknown frozen descriptor {descriptor}")


def rank_landscape(
    scores: list[float] | np.ndarray,
    ordered_candidate_ids: list[str],
    truth_candidate_id: str,
    *,
    bank: dict[str, object] | None = None,
    visible_mask: np.ndarray | None = None,
    include_pose_errors: bool = False,
) -> tuple[dict[str, object], dict[str, object] | None]:
    ranking = rank_candidate_scores(
        np.asarray(scores, dtype=np.float64),
        list(ordered_candidate_ids),
        str(truth_candidate_id),
    )
    validate_payload_keys(ranking, "ranking")
    if not include_pose_errors or ranking["selected_candidate_id"] is None:
        return ranking, None
    if bank is None or visible_mask is None:
        raise ValueError("native MIND pose errors require the exact bank and visible mask")
    candidates = list(bank["candidates"])
    truth = next(item for item in candidates if item["candidate_id"] == truth_candidate_id)
    selected = next(
        item for item in candidates if item["candidate_id"] == ranking["selected_candidate_id"]
    )
    plane = rp2_plane_error(
        truth["physical_pose"]["normal_rp2_sign_aligned_ap_dv_ml"],
        truth["physical_pose"]["signed_offset_um"],
        selected["physical_pose"]["normal_rp2_sign_aligned_ap_dv_ml"],
        selected["physical_pose"]["signed_offset_um"],
    )
    points = _finite_point_error_with_evidence(
        _effective_ouv(bank, truth), _effective_ouv(bank, selected), visible_mask
    )
    errors = {
        "rp2_angle_error_deg": float(plane["normal_geodesic_angle_deg"]),
        "sign_aligned_offset_error_um": float(plane["sign_aligned_offset_error_um"]),
        "corresponding_point_rms_um": float(points["corresponding_point_rms_um"]),
        "corresponding_point_p95_um": float(points["corresponding_point_p95_um"]),
    }
    return ranking, validate_payload_keys(errors, "pose_errors")


def _insufficient_slot(
    *,
    case_index: int | None,
    bank_case_index: int | None,
    target_case_index: int | None,
    outline_mode: str,
    descriptor: str,
    domain: str,
    domain_mask_receipt_sha256: str,
    domain_pixel_count: int,
    eligible_pixel_count: int | None,
    eligible_block_count: int | None,
) -> dict[str, object]:
    case_index, bank_case_index, target_case_index = _validate_slot_indices(
        case_index, bank_case_index, target_case_index
    )
    reason = "NO_ELIGIBLE_BLOCKS" if eligible_block_count == 0 else "NO_ELIGIBLE_PIXELS"
    record = {
        "status": "insufficient_domain",
        "reason_code": reason,
        "case_index": None if case_index is None else int(case_index),
        "bank_case_index": None if bank_case_index is None else int(bank_case_index),
        "target_case_index": None if target_case_index is None else int(target_case_index),
        "outline_mode": outline_mode,
        "descriptor": descriptor,
        "domain": domain,
        "domain_mask_receipt_sha256": domain_mask_receipt_sha256,
        "domain_pixel_count": int(domain_pixel_count),
        "eligible_pixel_count": eligible_pixel_count,
        "eligible_block_count": eligible_block_count,
        "scores": None,
        "ranking": None,
        "metrics": None,
        "entered_gate": False,
    }
    _self_hash(record, "payload_sha256")
    return validate_payload_keys(record, "slot")


def score_landscape(
    *,
    case_index: int | None,
    bank_case_index: int | None,
    target_case_index: int | None,
    outline_mode: str,
    descriptor: str,
    domain: str,
    domain_mask: np.ndarray,
    domain_mask_receipt_sha256: str,
    target_image: np.ndarray,
    candidate_images: np.ndarray,
    pixel_pitch_um: float,
    ordered_candidate_ids: list[str],
    truth_candidate_id: str,
    candidate_support: np.ndarray | None = None,
    target_visible: np.ndarray | None = None,
    supported_means: list[float] | None = None,
    bank: dict[str, object] | None = None,
    include_pose_errors: bool = False,
    entered_gate: bool = False,
    padding_value: float = 0.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, object]:
    """Evaluate one declared 40-candidate landscape and build its strict slot."""
    case_index, bank_case_index, target_case_index = _validate_slot_indices(
        case_index, bank_case_index, target_case_index
    )
    domain_array = np.asarray(domain_mask, dtype=bool)
    domain_count = int(domain_array.sum())
    eligible_pixels, eligible_blocks = _secondary_eligibility(
        descriptor, domain, domain_array, pixel_pitch_um
    )
    if eligible_pixels == 0 or eligible_blocks == 0:
        return _insufficient_slot(
            case_index=case_index,
            bank_case_index=bank_case_index,
            target_case_index=target_case_index,
            outline_mode=outline_mode,
            descriptor=descriptor,
            domain=domain,
            domain_mask_receipt_sha256=domain_mask_receipt_sha256,
            domain_pixel_count=domain_count,
            eligible_pixel_count=eligible_pixels,
            eligible_block_count=eligible_blocks,
        )
    result = _call_scorer(
        descriptor,
        domain,
        target_image,
        candidate_images,
        domain_array,
        pixel_pitch_um,
        candidate_support=candidate_support,
        target_visible=target_visible,
        padding_value=padding_value,
        chunk_size=chunk_size,
    )
    if result is None:
        raise ValueError("secondary eligibility and scorer null result disagree")
    scores = _unit_score_vector(result)
    ranking, pose_errors = rank_landscape(
        scores,
        ordered_candidate_ids,
        truth_candidate_id,
        bank=bank,
        visible_mask=target_visible,
        include_pose_errors=include_pose_errors,
    )
    if descriptor in {"MIND", "constant-within-support-MIND-null", "support-penalized-MIND"}:
        metrics = {
            "target_vbar": float(result["target_vbar"]),
            "candidate_vbar": np.asarray(result["candidate_vbar"], dtype=np.float64).tolist(),
            "supported_means": None if supported_means is None else [float(value) for value in supported_means],
            "candidate_exterior_fractions": (
                None
                if "candidate_exterior_fractions" not in result
                else np.asarray(result["candidate_exterior_fractions"], dtype=np.float64).tolist()
            ),
            "selected_pose_errors": pose_errors,
        }
        validate_payload_keys(metrics, "mind_metrics")
    elif descriptor == "HOG":
        metrics = {
            "cell_pixels": int(result["cell_pixels"]),
            "eligible_block_count": int(result["eligible_block_count"]),
            "block_weights_receipt": _array_receipt(result["block_weights"]),
        }
        validate_payload_keys(metrics, "hog_metrics")
    else:
        metrics = {
            "gaussian_radius_px": int(result["gaussian_radius_px"]),
            "effective_domain_count": int(result["effective_domain_count"]),
            "target_eta": float(result["target_eta"]),
            "candidate_eta": np.asarray(result["candidate_eta"], dtype=np.float64).tolist(),
        }
        validate_payload_keys(metrics, "ngf_metrics")
    record = {
        "status": "ok",
        "reason_code": None,
        "case_index": None if case_index is None else int(case_index),
        "bank_case_index": None if bank_case_index is None else int(bank_case_index),
        "target_case_index": None if target_case_index is None else int(target_case_index),
        "outline_mode": outline_mode,
        "descriptor": descriptor,
        "domain": domain,
        "domain_mask_receipt_sha256": domain_mask_receipt_sha256,
        "domain_pixel_count": domain_count,
        "eligible_pixel_count": eligible_pixels,
        "eligible_block_count": eligible_blocks,
        "scores": scores.tolist(),
        "ranking": ranking,
        "metrics": metrics,
        "entered_gate": bool(entered_gate),
    }
    _self_hash(record, "payload_sha256")
    return validate_payload_keys(record, "slot")


def _landscape_control(
    slot: dict[str, object],
    *,
    target_image: np.ndarray,
    candidate_images: np.ndarray,
    domain_mask: np.ndarray,
    pixel_pitch_um: float,
    ordered_candidate_ids: list[str],
    truth_candidate_id: str,
    candidate_support: np.ndarray | None,
    target_visible: np.ndarray | None,
    padding_value: float,
) -> dict[str, object]:
    if slot["status"] == "insufficient_domain":
        na = validate_payload_keys(
            {
                "status": "authenticated_not_applicable",
                "reason_code": "SOURCE_LANDSCAPE_INSUFFICIENT_DOMAIN",
                "passed": None,
                "evidence_sha256": canonical_payload_sha256(
                    {"source_slot_payload_sha256": slot["payload_sha256"]}
                ),
            },
            "basic_control",
        )
        record = {
            "source_slot_payload_sha256": slot["payload_sha256"],
            "source_status": slot["status"],
            "case_index": slot["case_index"],
            "bank_case_index": slot["bank_case_index"],
            "target_case_index": slot["target_case_index"],
            "outline_mode": slot["outline_mode"],
            "descriptor": slot["descriptor"],
            "domain": slot["domain"],
            "permutation": na,
            "chunks": copy.deepcopy(na),
            "status": "authenticated_not_applicable",
            "reason_code": "SOURCE_LANDSCAPE_INSUFFICIENT_DOMAIN",
            "passed": None,
        }
        _self_hash(record, "payload_sha256")
        return validate_payload_keys(record, "landscape_control")

    descriptor = str(slot["descriptor"])
    domain_name = str(slot["domain"])
    original_scores = np.asarray(slot["scores"], dtype=np.float64)
    chunk_hashes = {}
    chunk_passed = True
    for chunk in CHUNK_CONTROL_SIZES:
        result = _call_scorer(
            descriptor,
            domain_name,
            target_image,
            candidate_images,
            domain_mask,
            pixel_pitch_um,
            candidate_support=candidate_support,
            target_visible=target_visible,
            padding_value=padding_value,
            chunk_size=chunk,
        )
        if result is None:
            raise ValueError("status-ok landscape became insufficient in a chunk control")
        values = _unit_score_vector(result)
        chunk_hashes[str(chunk)] = _score_vector_sha256(values)
        chunk_passed &= _float64_vector_bytes(values) == _float64_vector_bytes(original_scores)
    chunk_control = {
        "chunk_sizes": list(CHUNK_CONTROL_SIZES),
        "score_vector_sha256": chunk_hashes,
        "byte_identical": bool(chunk_passed),
        "ranking_payload_sha256": canonical_payload_sha256(slot["ranking"]),
        "passed": bool(chunk_passed),
    }
    validate_payload_keys(chunk_control, "chunk_control")

    permutation = np.asarray(PERMUTATION, dtype=np.int64)
    permuted_support = None if candidate_support is None else candidate_support[permutation]
    permuted_result = _call_scorer(
        descriptor,
        domain_name,
        target_image,
        candidate_images[permutation],
        domain_mask,
        pixel_pitch_um,
        candidate_support=permuted_support,
        target_visible=target_visible,
        padding_value=padding_value,
        chunk_size=DEFAULT_CHUNK_SIZE,
    )
    if permuted_result is None:
        raise ValueError("status-ok landscape became insufficient under permutation")
    permuted_scores = _unit_score_vector(permuted_result)
    inverse = np.asarray(INVERSE_PERMUTATION, dtype=np.int64)
    reindexed = permuted_scores[inverse]
    reindexed_ranking = rank_candidate_scores(
        reindexed, ordered_candidate_ids, truth_candidate_id
    )
    permuted_ids = [ordered_candidate_ids[index] for index in PERMUTATION]
    permuted_ranking = rank_candidate_scores(
        permuted_scores, permuted_ids, truth_candidate_id
    )
    invariant_ranking_keys = (
        "truth_candidate_id", "truth_score", "top1", "true_rank", "reciprocal_rank",
        "truth_versus_decoy_win_fraction", "truth_score_margin",
        "tied_maximum_candidate_ids", "selected_candidate_id",
    )
    permutation_passed = (
        tuple(sorted(PERMUTATION)) == tuple(range(CANDIDATE_COUNT))
        and PERMUTATION != tuple(range(CANDIDATE_COUNT))
        and _float64_vector_bytes(permuted_scores)
        == _float64_vector_bytes(original_scores[permutation])
        and _float64_vector_bytes(reindexed) == _float64_vector_bytes(original_scores)
        and reindexed_ranking == slot["ranking"]
        and all(
            permuted_ranking[key] == slot["ranking"][key]
            for key in invariant_ranking_keys
        )
    )
    permutation_control = {
        "mapping": "new[k]=old[(7*k+3)%40]; inverse_new_index=23*(old_index-3)%40",
        "permutation": list(PERMUTATION),
        "nonidentity_bijection": True,
        "original_score_vector_sha256": _score_vector_sha256(original_scores),
        "permuted_score_vector_sha256": _score_vector_sha256(permuted_scores),
        "inverse_reindexed_score_vector_sha256": _score_vector_sha256(reindexed),
        "original_ranking_sha256": canonical_payload_sha256(slot["ranking"]),
        "recomputed_ranking_sha256": canonical_payload_sha256(
            {"permuted_id_order": permuted_ranking, "inverse_canonical_order": reindexed_ranking}
        ),
        "passed": bool(permutation_passed),
    }
    validate_payload_keys(permutation_control, "permutation_control")
    passed = bool(permutation_passed and chunk_passed)
    record = {
        "source_slot_payload_sha256": slot["payload_sha256"],
        "source_status": slot["status"],
        "case_index": slot["case_index"],
        "bank_case_index": slot["bank_case_index"],
        "target_case_index": slot["target_case_index"],
        "outline_mode": slot["outline_mode"],
        "descriptor": descriptor,
        "domain": domain_name,
        "permutation": permutation_control,
        "chunks": chunk_control,
        "status": "passed" if passed else "failed",
        "reason_code": None,
        "passed": passed,
    }
    _self_hash(record, "payload_sha256")
    return validate_payload_keys(record, "landscape_control")


def _candidate_scalar_payload(
    replayed: dict[str, object],
    render_context: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[dict[str, object]]]:
    rendered = render_candidate_bank_scalars(
        render_context, replayed["bank"], replayed["parent"]
    )
    raw = np.asarray(rendered["scalar_float32"])
    annotations = np.asarray(rendered["annotation"])
    supports = np.asarray(rendered["brain_mask"])
    scaled = np.ascontiguousarray(
        np.stack([scale_candidate_raster(item) for item in raw]), dtype=np.float64
    )
    null_images, means = [], []
    records = []
    for index, candidate_id in enumerate(rendered["candidate_ids"]):
        flattened, supported_mean = constant_within_support_null(
            scaled[index], supports[index]
        )
        null_images.append(flattened)
        means.append(float(supported_mean))
        scalar = {
            "rendered_float32": _array_receipt(raw[index]),
            "scaled_float64": _array_receipt(scaled[index]),
            "render_then_scale": True,
            "global_conversion": "clip((rendered_float32-6.0)/281.0,0.0,1.0) after 2-D rendering",
        }
        validate_payload_keys(scalar, "candidate_scalar_value")
        record = {
            "candidate_index": index,
            "candidate_id": candidate_id,
            "scalar": scalar,
            "annotation": _array_receipt(annotations[index]),
            "brain_mask": _array_receipt(supports[index]),
            "crosscheck_passed": True,
        }
        _self_hash(record, "payload_sha256")
        records.append(validate_payload_keys(record, "candidate_scalar_record"))
    if (
        raw.shape != (CANDIDATE_COUNT, *OUTPUT_SHAPE)
        or raw.dtype != np.float32
        or scaled.shape != raw.shape
        or supports.shape != raw.shape
        or supports.dtype != np.bool_
        or list(rendered["candidate_ids"]) != replayed["bank"]["ordered_candidate_ids"]
    ):
        raise ValueError("candidate scalar adapter changed the frozen bank shape/order/dtype")
    return (
        scaled,
        np.ascontiguousarray(np.stack(null_images), dtype=np.float64),
        np.ascontiguousarray(supports, dtype=bool),
        means,
        records,
    )


def _mask_only_dice_receipt(
    replayed: dict[str, object],
    source_records: list[dict[str, str]],
) -> dict[str, object]:
    semantic = replayed["semantic_record"]
    bank = replayed["bank"]
    target = replayed["target"]
    valid = np.asarray(target["fixed_valid_mask"], dtype=bool)
    target_tissue = valid & (np.asarray(target["labels"]) != 0)
    recomputed = []
    for candidate in bank["candidates"]:
        candidate_tissue = valid & (np.asarray(candidate["rendered_annotation"]) != 0)
        value = (
            2.0 * np.count_nonzero(target_tissue & candidate_tissue) + 1.0e-12
        ) / (
            np.count_nonzero(target_tissue)
            + np.count_nonzero(candidate_tissue)
            + 1.0e-12
        )
        recomputed.append(float(np.clip(value, 0.0, 1.0)))
    values = np.asarray(semantic["scores"]["mask_only_Dice"], dtype=np.float64)
    if _float64_vector_bytes(values) != _float64_vector_bytes(recomputed):
        raise ValueError("copied frozen mask-only Dice does not replay byte-exactly")
    candidate_ids = list(bank["ordered_candidate_ids"])
    truth_id = str(semantic["truth_candidate_id"])
    ranking = rank_candidate_scores(values, candidate_ids, truth_id)
    source_by_path = {item["relative_path"]: item for item in source_records}
    record = {
        "source_relative_path": _semantic_case_file(int(semantic["case_index"])).relative_to(ROOT).as_posix(),
        "source_file_sha256": _file_sha256(_semantic_case_file(int(semantic["case_index"]))),
        "source_case_payload_sha256": semantic["case_payload_sha256"],
        "source_target_receipt_sha256": semantic["target"]["target_receipt_sha256"],
        "candidate_bank_id": bank["finite_candidate_bank_id"],
        "ordered_candidate_ids_sha256": canonical_payload_sha256(candidate_ids),
        "truth_candidate_id": truth_id,
        "source_vector_json_pointer": "/scores/mask_only_Dice",
        "source_vector_sha256": canonical_payload_sha256({"mask_only_Dice": values.tolist()}),
        "scorer_source_sha256": "4297e78a8a21785b570ad652af992d12363033ae3cc7dff324c549037e7aaf58",
        "runner_source_sha256": source_by_path[
            "training/run_arbitrary_plane_semantic_oracle.py"
        ]["checkout_sha256"],
        "values": values.tolist(),
        "recomputed_ranking": ranking,
        "entered_gate": False,
    }
    _self_hash(record, "payload_sha256")
    return validate_payload_keys(record, "mask_only_dice")


def _score_domain_records(
    case_index: int,
    masks: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    records, by_name = [], {}
    for name in MASK_NAMES:
        receipt = _persisted_mask_receipt(case_index, name, masks[name])
        record = {
            "domain": name,
            "mask_receipt": receipt,
            "pixel_count": int(np.asarray(masks[name], dtype=bool).sum()),
            "minimum_required_pixels": PRIMARY_MINIMUM_PIXELS.get(name),
        }
        validate_payload_keys(record, "score_domain")
        records.append(record)
        by_name[name] = record
    return records, by_name


def _target_dewarp_record(descendant: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    arrays = descendant["arrays"]
    dewarped32 = dewarp_target_float32(
        arrays["model_input_image"], arrays["fixed_to_source_map"]
    )
    dewarped64 = np.array(dewarped32, dtype=np.float64, copy=True, order="C")
    if not np.array_equal(
        dewarped64,
        dewarp_target_for_scoring(
            arrays["model_input_image"], arrays["fixed_to_source_map"]
        ),
    ):
        raise ValueError("target float32-then-float64 dewarp contract changed")
    record = {
        "direction": "source-to-fixed via frozen fixed_to_source_map",
        "scalar_padding": 0.0,
        "model_input_image_receipt": _array_receipt(arrays["model_input_image"]),
        "fixed_to_source_map_receipt": _array_receipt(arrays["fixed_to_source_map"]),
        "dewarped_float32_receipt": _array_receipt(dewarped32),
        "dewarped_float64_receipt": _array_receipt(dewarped64),
    }
    return dewarped64, validate_payload_keys(record, "target_dewarp")


def _entered_gate(
    descriptor: str,
    domain: str,
    outline_mode: str,
    *,
    shuffled: bool,
) -> bool:
    if shuffled:
        return descriptor == "MIND" and domain == "context"
    if descriptor == "MIND" and domain == "context":
        return True
    return (
        domain == "core"
        and outline_mode == ABSENT_OUTLINE
        and descriptor in {"MIND", "constant-within-support-MIND-null"}
    )


def _affine_slot_control(
    slot: dict[str, object],
    *,
    target_image: np.ndarray,
    candidate_images: np.ndarray,
    domain_mask: np.ndarray,
    pixel_pitch_um: float,
    ordered_candidate_ids: list[str],
    truth_candidate_id: str,
    candidate_support: np.ndarray | None,
    target_visible: np.ndarray | None,
    original_candidate_images: np.ndarray | None = None,
    constant_null_support: np.ndarray | None = None,
) -> dict[str, object]:
    if slot["status"] == "insufficient_domain":
        record = {
            "case_index": slot["case_index"],
            "outline_mode": slot["outline_mode"],
            "descriptor": slot["descriptor"],
            "domain": slot["domain"],
            "source_slot_payload_sha256": slot["payload_sha256"],
            "transforms": [],
            "status": "authenticated_not_applicable",
            "reason_code": "SOURCE_LANDSCAPE_INSUFFICIENT_DOMAIN",
            "passed": None,
        }
        _self_hash(record, "payload_sha256")
        return validate_payload_keys(record, "affine_slot")
    transforms = []
    source_ranking = slot["ranking"]
    for name, scale, offset in AFFINE_TRANSFORMS:
        transformed_candidates = np.ascontiguousarray(
            scale * candidate_images + offset, dtype=np.float64
        )
        if slot["descriptor"] == "constant-within-support-MIND-null":
            if original_candidate_images is None or constant_null_support is None:
                raise ValueError("affine constant-null control requires native images and support")
            transformed_native = np.ascontiguousarray(
                scale * original_candidate_images + offset, dtype=np.float64
            )
            transformed_candidates = np.ascontiguousarray(
                np.stack(
                    [
                        constant_within_support_null(image, support)[0]
                        for image, support in zip(
                            transformed_native, constant_null_support, strict=True
                        )
                    ]
                ),
                dtype=np.float64,
            )
        transformed = _call_scorer(
            str(slot["descriptor"]),
            str(slot["domain"]),
            np.ascontiguousarray(scale * target_image + offset, dtype=np.float64),
            transformed_candidates,
            domain_mask,
            pixel_pitch_um,
            candidate_support=candidate_support,
            target_visible=target_visible,
            padding_value=float(scale * 0.0 + offset),
            chunk_size=DEFAULT_CHUNK_SIZE,
        )
        if transformed is None:
            raise ValueError("affine transform changed a status-ok landscape to insufficient")
        ranking = rank_candidate_scores(
            _unit_score_vector(transformed),
            ordered_candidate_ids,
            truth_candidate_id,
        )
        invariant = (
            ranking["top1"] == source_ranking["top1"]
            and ranking["true_rank"] == source_ranking["true_rank"]
            and ranking["tied_maximum_candidate_ids"]
            == source_ranking["tied_maximum_candidate_ids"]
            and ranking["selected_candidate_id"] == source_ranking["selected_candidate_id"]
        )
        transforms.append(
            validate_payload_keys(
                {
                    "name": name,
                    "scale": scale,
                    "offset": offset,
                    "scalar_padding": float(offset),
                    "ranking_payload_sha256": canonical_payload_sha256(ranking),
                    "top1": ranking["top1"],
                    "true_rank": ranking["true_rank"],
                    "tied_maximum_candidate_ids": ranking["tied_maximum_candidate_ids"],
                    "passed": bool(invariant),
                },
                "affine_transform",
            )
        )
    passed = all(item["passed"] for item in transforms)
    record = {
        "case_index": slot["case_index"],
        "outline_mode": slot["outline_mode"],
        "descriptor": slot["descriptor"],
        "domain": slot["domain"],
        "source_slot_payload_sha256": slot["payload_sha256"],
        "transforms": transforms,
        "status": "passed" if passed else "failed",
        "reason_code": None,
        "passed": bool(passed),
    }
    _self_hash(record, "payload_sha256")
    return validate_payload_keys(record, "affine_slot")


def _build_primary_case(
    replayed: dict[str, object],
    render_context: Mapping[str, object],
    support_index: dict[str, object],
    source_records: list[dict[str, str]],
) -> tuple[dict[str, object], dict[str, object]]:
    case_index = int(replayed["case_index"])
    semantic, parent, bank = (
        replayed["semantic_record"], replayed["parent"], replayed["bank"]
    )
    masks = replayed["masks"]
    domains, domain_by_name = _score_domain_records(case_index, masks)
    scaled, null_images, supports, supported_means, scalar_records = _candidate_scalar_payload(
        replayed, render_context
    )
    candidate_ids = list(bank["ordered_candidate_ids"])
    truth_id = str(semantic["truth_candidate_id"])
    truth_index = candidate_ids.index(truth_id)
    target = {
        "parent_plane_realization_id": parent["plane_realization_id"],
        "paired_view_group_id": semantic["paired_view_group_id"],
        "truth_candidate_id": truth_id,
        "truth_geometry_sha256": parent["finite_plane_geometry_sha256"],
        "pixel_pitch_um": float(replayed["target"]["pixel_pitch_um"]),
        "output_shape_h_w": list(OUTPUT_SHAPE),
        "target_labels_receipt": _array_receipt(replayed["target"]["labels"]),
        "fixed_valid_mask_receipt": _array_receipt(replayed["target"]["fixed_valid_mask"]),
    }
    validate_payload_keys(target, "target")
    candidate_bank = {
        "finite_candidate_bank_id": bank["finite_candidate_bank_id"],
        "finite_candidate_receipt_sha256": bank["finite_candidate_receipt_sha256"],
        "candidate_set_id": bank["candidate_set_id"],
        "ordered_candidate_ids": candidate_ids,
        "ordered_candidate_ids_sha256": canonical_payload_sha256(candidate_ids),
        "truth_candidate_id": truth_id,
        "truth_candidate_index": truth_index,
        "truth_parent_geometry": copy.deepcopy(bank["truth_parent_geometry"]),
        "receipt": finite_candidate_bank_receipt(bank),
    }
    validate_payload_keys(candidate_bank, "candidate_bank")
    outline_results, target_images, landscape_controls, affine_controls = [], {}, [], []
    reporting_source = None
    for mode, expected in zip(
        OUTLINE_MODES, semantic["outline_descendants"], strict=True
    ):
        descendant = _verified_descendant(
            parent,
            support_index,
            case_index,
            replayed["synthetic_seed"],
            mode,
            expected,
        )
        if reporting_source is None:
            reporting_source = {
                "appearance_family": descendant["g2"]["parameters"]["source_family"],
                "damage_event_types": [
                    event["type"] for event in descendant["g3"]["parameters"]["events"]
                ],
                "damage_event_count": int(descendant["g3"]["parameters"]["event_count"]),
                "damage_union_fraction": float(
                    descendant["g3"]["parameters"]["gates"]["union_damage_fraction"]
                ),
            }
        target_image, dewarp = _target_dewarp_record(descendant)
        target_images[mode] = target_image
        score_slots = []
        for descriptor, domain_name in NATIVE_OUTLINE_SLOT_SCHEDULE:
            candidates = null_images if descriptor == "constant-within-support-MIND-null" else scaled
            candidate_support = supports if descriptor == "support-penalized-MIND" else None
            slot = score_landscape(
                case_index=case_index,
                bank_case_index=None,
                target_case_index=None,
                outline_mode=mode,
                descriptor=descriptor,
                domain=domain_name,
                domain_mask=masks[domain_name],
                domain_mask_receipt_sha256=_domain_mask_receipt_sha256(
                    domain_by_name[domain_name]
                ),
                target_image=target_image,
                candidate_images=candidates,
                pixel_pitch_um=target["pixel_pitch_um"],
                ordered_candidate_ids=candidate_ids,
                truth_candidate_id=truth_id,
                candidate_support=candidate_support,
                target_visible=masks["visible"],
                supported_means=(
                    supported_means
                    if descriptor == "constant-within-support-MIND-null"
                    else None
                ),
                bank=bank,
                include_pose_errors=descriptor == "MIND",
                entered_gate=_entered_gate(descriptor, domain_name, mode, shuffled=False),
            )
            score_slots.append(slot)
            landscape_controls.append(
                _landscape_control(
                    slot,
                    target_image=target_image,
                    candidate_images=candidates,
                    domain_mask=masks[domain_name],
                    pixel_pitch_um=target["pixel_pitch_um"],
                    ordered_candidate_ids=candidate_ids,
                    truth_candidate_id=truth_id,
                    candidate_support=candidate_support,
                    target_visible=masks["visible"],
                    padding_value=0.0,
                )
            )
            if case_index in AFFINE_CASES:
                affine_controls.append(
                    _affine_slot_control(
                        slot,
                        target_image=target_image,
                        candidate_images=candidates,
                        domain_mask=masks[domain_name],
                        pixel_pitch_um=target["pixel_pitch_um"],
                        ordered_candidate_ids=candidate_ids,
                        truth_candidate_id=truth_id,
                        candidate_support=candidate_support,
                        target_visible=masks["visible"],
                        original_candidate_images=(
                            scaled
                            if descriptor == "constant-within-support-MIND-null"
                            else None
                        ),
                        constant_null_support=(
                            supports
                            if descriptor == "constant-within-support-MIND-null"
                            else None
                        ),
                    )
                )
        if [(item["descriptor"], item["domain"]) for item in score_slots] != list(
            NATIVE_OUTLINE_SLOT_SCHEDULE
        ):
            raise ValueError("native outline slots changed frozen order")
        outline = {
            "outline_mode": mode,
            "synthetic_realization_id": descendant["synthetic_realization_id"],
            "synthetic_receipt": synthetic_realization_receipt(descendant),
            "target_dewarp": dewarp,
            "score_slots": score_slots,
        }
        _self_hash(outline, "payload_sha256")
        outline_results.append(validate_payload_keys(outline, "primary_outline"))
        del descendant
    if reporting_source is None:
        raise RuntimeError("primary case did not regenerate an outline descendant")
    reporting = {
        "orientation_family": semantic["reporting_strata"]["orientation_family"],
        "appearance_family": reporting_source["appearance_family"],
        "damage_event_types": reporting_source["damage_event_types"],
        "damage_event_count": reporting_source["damage_event_count"],
        "damage_union_fraction": reporting_source["damage_union_fraction"],
        "parent_brain_pixel_count": int(parent["acceptance_contract"]["brain_pixel_count"]),
        "visible_pixel_count": int(np.asarray(masks["visible"], dtype=bool).sum()),
        "challenging_appearance_member": reporting_source["appearance_family"]
        in {"label-conditioned", "template-label-mixture"},
        "damaged_member": reporting_source["damage_event_count"] >= 1,
        "outline_modes": list(OUTLINE_MODES),
    }
    validate_payload_keys(reporting, "reporting_strata")
    provenance = {
        "animal_id": parent["provenance"]["animal_id"],
        "specimen_id": parent["provenance"]["specimen_id"],
        "experiment_id": parent["provenance"]["experiment_id"],
        "atlas": copy.deepcopy(parent["provenance"]["atlas"]),
        "annotation_source": copy.deepcopy(parent["provenance"]["annotation_source"]),
        "scalar_source": _plain(render_context["asset_receipt"]["scalar_source"]),
        "reporting_strata": reporting,
        "mask_only_Dice": _mask_only_dice_receipt(replayed, source_records),
    }
    validate_payload_keys(provenance, "primary_provenance")
    parent_receipt = finite_render_receipt(parent)
    descendants_receipts = copy.deepcopy(replayed["descendant_receipts"])
    frozen_replay = {
        "semantic_source_relative_path": _semantic_case_file(case_index).relative_to(ROOT).as_posix(),
        "semantic_source_file_sha256": _file_sha256(_semantic_case_file(case_index)),
        "semantic_case_payload_sha256": semantic["case_payload_sha256"],
        "finite_parent_receipt": parent_receipt,
        "finite_parent_receipt_sha256": canonical_payload_sha256(parent_receipt),
        "case_rejection_records": copy.deepcopy(semantic["case_rejection_attempts"]),
        "case_rejection_records_sha256": semantic["case_rejection_attempts_sha256"],
        "outline_descendant_receipts": descendants_receipts,
        "outline_descendant_receipts_sha256": canonical_payload_sha256(descendants_receipts),
        "candidate_bank_receipt_sha256": bank["finite_candidate_receipt_sha256"],
        "replay_passed": True,
    }
    validate_payload_keys(frozen_replay, "frozen_replay")
    record = {
        "schema": "anatomy-tracker.arbitrary-plane-image-information-primary/v1",
        "case_index": case_index,
        "semantic_case_payload_sha256": semantic["case_payload_sha256"],
        "provenance": provenance,
        "frozen_replay": frozen_replay,
        "target": target,
        "candidate_bank": candidate_bank,
        "candidate_scalar_receipts": scalar_records,
        "score_domains": domains,
        "outline_results": outline_results,
    }
    _self_hash(record, "payload_sha256")
    validate_payload_keys(record, "primary")
    runtime = {
        "case_index": case_index,
        "record": record,
        "replayed": replayed,
        "scaled": scaled,
        "null_images": null_images,
        "supports": supports,
        "supported_means": supported_means,
        "target_images": target_images,
        "domain_by_name": domain_by_name,
        "landscape_controls": landscape_controls,
        "affine_controls": affine_controls,
    }
    return record, runtime


def _build_shuffled_case(
    bank_runtime: dict[str, object],
    target_runtime: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    bank_case = int(bank_runtime["case_index"])
    target_case = int(target_runtime["case_index"])
    if target_case != (bank_case + 17) % CASE_COUNT:
        raise ValueError("shuffled runtime pair does not follow the frozen +17 mapping")
    bank_record = bank_runtime["record"]
    target_record = target_runtime["record"]
    source_pitch = float(bank_record["target"]["pixel_pitch_um"])
    target_pitch = float(target_record["target"]["pixel_pitch_um"])
    coordinates = common_lattice_map_yx(OUTPUT_SHAPE, source_pitch, target_pitch)
    scaled = np.ascontiguousarray(
        np.stack(
            [
                resample_common_lattice_intensity(item, coordinates)
                for item in bank_runtime["scaled"]
            ]
        ),
        dtype=np.float64,
    )
    supports = np.ascontiguousarray(
        np.stack(
            [
                resample_common_lattice_support(item, coordinates)
                for item in bank_runtime["supports"]
            ]
        ),
        dtype=bool,
    )
    null_images = np.ascontiguousarray(
        np.stack(
            [
                resample_common_lattice_intensity(item, coordinates)
                for item in bank_runtime["null_images"]
            ]
        ),
        dtype=np.float64,
    )
    candidate_ids = list(bank_record["candidate_bank"]["ordered_candidate_ids"])
    resampled_records = []
    for index, candidate_id in enumerate(candidate_ids):
        item = {
            "candidate_index": index,
            "candidate_id": candidate_id,
            "scalar_float64_receipt": _array_receipt(scaled[index]),
            "support_bool_receipt": _array_receipt(supports[index]),
            "constant_null_scalar_float64_receipt": _array_receipt(null_images[index]),
            "constant_supported_mean": float(bank_runtime["supported_means"][index]),
        }
        _self_hash(item, "payload_sha256")
        resampled_records.append(validate_payload_keys(item, "resampled_candidate"))
    common = {
        "mapping": "source x=W/2+(x-W/2)*target_pitch/source_pitch and likewise y",
        "output_shape_h_w": list(OUTPUT_SHAPE),
        "coordinate_order": "yx float64 C-order",
        "source_pixel_pitch_um": source_pitch,
        "target_pixel_pitch_um": target_pitch,
        "coordinate_map_receipt": _array_receipt(coordinates),
        "intensity_resampler": "scipy.ndimage.map_coordinates order=1 mode=constant cval=0 prefilter=false",
        "support_resampler": "np.rint ties-to-even then explicit-bounds nearest-zero",
        "null_construction_order": "flatten native bank support first, then common-lattice resample",
        "resampled_candidates": resampled_records,
    }
    _self_hash(common, "payload_sha256")
    validate_payload_keys(common, "common_lattice")
    bank_identity = {
        "bank_case_index": bank_case,
        "source_primary_payload_sha256": bank_record["payload_sha256"],
        "finite_candidate_bank_id": bank_record["candidate_bank"]["finite_candidate_bank_id"],
        "finite_candidate_receipt_sha256": bank_record["candidate_bank"][
            "finite_candidate_receipt_sha256"
        ],
        "ordered_candidate_ids": candidate_ids,
        "ordered_candidate_ids_sha256": bank_record["candidate_bank"][
            "ordered_candidate_ids_sha256"
        ],
        "truth_candidate_id": bank_record["candidate_bank"]["truth_candidate_id"],
        "source_pixel_pitch_um": source_pitch,
        "candidate_scalar_receipts_sha256": canonical_payload_sha256(
            bank_record["candidate_scalar_receipts"]
        ),
    }
    validate_payload_keys(bank_identity, "bank_identity")
    target_outline_hashes = [item["payload_sha256"] for item in target_record["outline_results"]]
    target_identity = {
        "target_case_index": target_case,
        "target_primary_payload_sha256": target_record["payload_sha256"],
        "paired_view_group_id": target_record["target"]["paired_view_group_id"],
        "target_outline_payload_sha256": target_outline_hashes,
        "target_pixel_pitch_um": target_pitch,
        "score_domain_receipts_sha256": canonical_payload_sha256(
            target_record["score_domains"]
        ),
    }
    validate_payload_keys(target_identity, "target_identity")
    domain_by_name = target_runtime["domain_by_name"]
    target_masks = target_runtime["replayed"]["masks"]
    shuffled_outlines, controls = [], []
    primary_outline_by_mode = {
        item["outline_mode"]: item for item in target_record["outline_results"]
    }
    for mode in OUTLINE_MODES:
        slots = []
        target_image = target_runtime["target_images"][mode]
        for descriptor, domain_name in SHUFFLED_OUTLINE_SLOT_SCHEDULE:
            candidates = null_images if descriptor == "constant-within-support-MIND-null" else scaled
            slot = score_landscape(
                case_index=None,
                bank_case_index=bank_case,
                target_case_index=target_case,
                outline_mode=mode,
                descriptor=descriptor,
                domain=domain_name,
                domain_mask=target_masks[domain_name],
                domain_mask_receipt_sha256=_domain_mask_receipt_sha256(
                    domain_by_name[domain_name]
                ),
                target_image=target_image,
                candidate_images=candidates,
                pixel_pitch_um=target_pitch,
                ordered_candidate_ids=candidate_ids,
                truth_candidate_id=bank_identity["truth_candidate_id"],
                candidate_support=None,
                target_visible=target_masks["visible"],
                supported_means=(
                    bank_runtime["supported_means"]
                    if descriptor == "constant-within-support-MIND-null"
                    else None
                ),
                include_pose_errors=False,
                entered_gate=_entered_gate(descriptor, domain_name, mode, shuffled=True),
            )
            slots.append(slot)
            controls.append(
                _landscape_control(
                    slot,
                    target_image=target_image,
                    candidate_images=candidates,
                    domain_mask=target_masks[domain_name],
                    pixel_pitch_um=target_pitch,
                    ordered_candidate_ids=candidate_ids,
                    truth_candidate_id=bank_identity["truth_candidate_id"],
                    candidate_support=None,
                    target_visible=target_masks["visible"],
                    padding_value=0.0,
                )
            )
        if [(item["descriptor"], item["domain"]) for item in slots] != list(
            SHUFFLED_OUTLINE_SLOT_SCHEDULE
        ):
            raise ValueError("shuffled outline slots changed frozen order")
        outline = {
            "outline_mode": mode,
            "target_primary_outline_payload_sha256": primary_outline_by_mode[mode][
                "payload_sha256"
            ],
            "score_slots": slots,
        }
        _self_hash(outline, "payload_sha256")
        shuffled_outlines.append(validate_payload_keys(outline, "shuffled_outline"))
    record = {
        "schema": "anatomy-tracker.arbitrary-plane-image-information-shuffled/v1",
        "bank_case_index": bank_case,
        "target_case_index": target_case,
        "bank_identity": bank_identity,
        "target_identity": target_identity,
        "common_lattice_resampling": common,
        "score_domains": copy.deepcopy(target_record["score_domains"]),
        "outline_results": shuffled_outlines,
    }
    _self_hash(record, "payload_sha256")
    return validate_payload_keys(record, "shuffled"), controls


def _basic_control(evidence: object, passed: bool) -> dict[str, object]:
    return validate_payload_keys(
        {
            "status": "passed" if passed else "failed",
            "reason_code": None if passed else "EXACT_CONTROL_MISMATCH",
            "passed": bool(passed),
            "evidence_sha256": canonical_payload_sha256(evidence),
        },
        "basic_control",
    )


def _geometry_control_evidence(
    replayed: dict[str, object], support_index: dict[str, object]
) -> tuple[dict[str, object], bool]:
    bank = replayed["bank"]
    rp2, xy = [], []
    passed = True
    atlas_shape = tuple(support_index["annotation_shape"])
    for candidate in bank["candidates"]:
        pose = candidate["physical_pose"]
        positive = transport_finite_candidate_pose(
            bank["truth_parent_geometry"],
            support_index,
            pose["normal_rp2_sign_aligned_ap_dv_ml"],
            pose["signed_offset_um"],
            pose["roll_delta_rad_from_parallel_transport"],
        )
        antipodal = transport_finite_candidate_pose(
            bank["truth_parent_geometry"],
            support_index,
            -np.asarray(pose["normal_rp2_sign_aligned_ap_dv_ml"]),
            -float(pose["signed_offset_um"]),
            pose["roll_delta_rad_from_parallel_transport"],
        )
        equal = positive == antipodal
        passed &= equal
        rp2.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_pose_sha256": candidate["pose_sha256"],
                "positive_geometry_sha256": positive["candidate_geometry_sha256"],
                "antipodal_geometry_sha256": antipodal["candidate_geometry_sha256"],
                "equal": bool(equal),
            }
        )
        geometry = (
            bank["truth_parent_geometry"]
            if candidate["candidate_class"] == "truth"
            else candidate["geometry"]
        )
        arrays = effective_renderer_sampling_arrays(geometry, atlas_shape)
        points = arrays["coordinate_raster_allen_index_float32"]
        ouv = arrays["allen_index_ouv_ap_dv_ml_float32"]
        height, width = geometry["output_shape_h_w"]
        s = np.arange(width, dtype=np.float32) / np.float32(width)
        t = np.arange(height, dtype=np.float32) / np.float32(height)
        tt, ss = np.meshgrid(t, s, indexing="ij")
        expected = (
            ouv[:3] + ss[..., None] * ouv[3:6] + tt[..., None] * ouv[6:9]
        ).astype(np.float32, copy=False)
        residual = float(np.max(np.abs(points - expected)))
        endpoint_gap = float(
            np.linalg.norm(points[-1, -1] - (ouv[:3] + ouv[3:6] + ouv[6:9]))
        )
        valid = (
            geometry["sampling_contract"] == "quicknii-raster-index-x-over-W-y-over-H-v1"
            and residual <= 5.0e-4
            and endpoint_gap > 1.0e-6
        )
        passed &= valid
        xy.append(
            {
                "candidate_id": candidate["candidate_id"],
                "actual_coordinate_grid_receipt": _array_receipt(points),
                "reconstructed_coordinate_grid_receipt": _array_receipt(expected),
                "maximum_absolute_residual_allen_index": residual,
                "inclusive_endpoint_gap_allen_index": endpoint_gap,
                "passed": bool(valid),
            }
        )
    return {"rp2": rp2, "xy_over_wh": xy}, bool(passed)


def _scorer_signature_evidence() -> tuple[list[dict[str, object]], bool]:
    forbidden = (
        "candidate_id",
        "truth_candidate_id",
        "physical_pose",
        "normal_rp2",
        "signed_offset",
        "candidate_class",
        "rendered_annotation",
        "coordinate_raster",
    )
    records, passed = [], True
    expected = {
        "score_mind_candidates": {
            "target_image_float64", "candidate_images_float64", "domain_mask",
            "pixel_pitch_um", "padding_value", "chunk_size",
        },
        "score_hog_candidates": {
            "target", "candidates", "domain", "pixel_pitch_um", "boundary_ring",
            "padding_value", "chunk_size",
        },
        "score_ngf_candidates": {
            "target", "candidates", "domain", "pixel_pitch_um", "padding_value",
            "chunk_size",
        },
        "score_support_penalized_mind_candidates": {
            "target_image_float64", "candidate_images_float64", "domain_mask",
            "target_visible_mask", "candidate_support_masks", "pixel_pitch_um",
            "padding_value", "chunk_size",
        },
    }
    for function in (
        score_mind_candidates,
        score_support_penalized_mind_candidates,
        score_hog_candidates,
        score_ngf_candidates,
    ):
        source = inspect.getsource(function)
        signature = list(inspect.signature(function).parameters)
        matches = [token for token in forbidden if token in source]
        valid = set(signature) == expected[function.__name__] and not matches
        passed &= valid
        records.append(
            validate_payload_keys({
                "function": function.__name__,
                "signature": signature,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "forbidden_tokens": list(forbidden),
                "forbidden_matches": matches,
                "passed": bool(valid),
            }, "scorer_signature_record")
        )
    return records, bool(passed)


def _recompute_dewarp_and_domain_evidence(
    bank_runtime: dict[str, object], support_index: dict[str, object]
) -> tuple[dict[str, object], bool]:
    replayed = bank_runtime["replayed"]
    primary = bank_runtime["record"]
    semantic = replayed["semantic_record"]
    saved_outlines = {item["outline_mode"]: item for item in primary["outline_results"]}
    saved_domains = {item["domain"]: item for item in primary["score_domains"]}
    evidence = []
    passed = True
    for mode, expected in zip(OUTLINE_MODES, semantic["outline_descendants"], strict=True):
        descendant = _verified_descendant(
            replayed["parent"],
            support_index,
            int(replayed["case_index"]),
            replayed["synthetic_seed"],
            mode,
            expected,
        )
        arrays = descendant["arrays"]
        dewarped32 = dewarp_target_float32(
            arrays["model_input_image"], arrays["fixed_to_source_map"]
        )
        dewarped64 = np.array(dewarped32, dtype=np.float64, copy=True, order="C")
        masks = target_score_masks(
            arrays["fixed_to_source_map"],
            arrays["source_map_domain_mask"],
            arrays["fixed_map_domain_mask"],
            arrays["source_valid_correspondence_mask"],
            arrays["fixed_valid_correspondence_mask"],
            float(primary["target"]["pixel_pitch_um"]),
        )
        saved_dewarp = saved_outlines[mode]["target_dewarp"]
        domain_matches = {
            name: (
                _array_receipt(masks[name])
                == {
                    "dtype": "|b1",
                    "shape": saved_domains[name]["mask_receipt"]["shape"],
                    "array_sha256": saved_domains[name]["mask_receipt"]["array_sha256"],
                }
                and int(masks[name].sum()) == saved_domains[name]["pixel_count"]
            )
            for name in MASK_NAMES
        }
        valid = (
            saved_dewarp["direction"] == "source-to-fixed via frozen fixed_to_source_map"
            and saved_dewarp["scalar_padding"] == 0.0
            and saved_dewarp["model_input_image_receipt"]
            == _array_receipt(arrays["model_input_image"])
            and saved_dewarp["fixed_to_source_map_receipt"]
            == _array_receipt(arrays["fixed_to_source_map"])
            and saved_dewarp["dewarped_float32_receipt"] == _array_receipt(dewarped32)
            and saved_dewarp["dewarped_float64_receipt"] == _array_receipt(dewarped64)
            and all(domain_matches.values())
        )
        passed &= valid
        evidence.append(
            {
                "outline_mode": mode,
                "model_input_image_receipt": _array_receipt(arrays["model_input_image"]),
                "fixed_to_source_map_receipt": _array_receipt(arrays["fixed_to_source_map"]),
                "dewarped_float32_receipt": _array_receipt(dewarped32),
                "dewarped_float64_receipt": _array_receipt(dewarped64),
                "domain_array_receipts": {
                    name: _array_receipt(masks[name]) for name in MASK_NAMES
                },
                "domain_counts": {name: int(masks[name].sum()) for name in MASK_NAMES},
                "passed": bool(valid),
            }
        )
        del descendant
    return {"outlines": evidence}, bool(passed)


def _target_domain_invariance_evidence(
    bank_runtime: dict[str, object]
) -> tuple[dict[str, object], bool]:
    primary = bank_runtime["record"]
    domains = primary["score_domains"]
    domain_hashes = {
        item["domain"]: canonical_payload_sha256(item["mask_receipt"]) for item in domains
    }
    candidate_ids = primary["candidate_bank"]["ordered_candidate_ids"]
    permutation = np.asarray(PERMUTATION, dtype=np.int64)
    original_support = np.asarray(bank_runtime["supports"], dtype=bool)
    permuted_support = original_support[permutation]
    signature = list(inspect.signature(target_score_masks).parameters)
    forbidden_arguments = [
        name
        for name in signature
        if name in {"candidate_support", "candidate_order", "candidate_images"}
    ]
    domain_by_name = {item["domain"]: item for item in domains}
    outline_domain_hashes = {}
    slots_passed = True
    for outline in primary["outline_results"]:
        per_domain = {}
        for slot in outline["score_slots"]:
            expected = domain_by_name[slot["domain"]]
            expected_hash = _domain_mask_receipt_sha256(expected)
            per_domain.setdefault(slot["domain"], expected_hash)
            slots_passed &= (
                slot["domain_mask_receipt_sha256"] == expected_hash
                and slot["domain_pixel_count"] == expected["pixel_count"]
            )
        outline_domain_hashes[outline["outline_mode"]] = per_domain
    support_receipts = [
        _array_receipt(item) for item in np.asarray(bank_runtime["supports"], dtype=bool)
    ]
    saved_support_receipts = [
        item["brain_mask"] for item in primary["candidate_scalar_receipts"]
    ]
    permuted_support_receipts = [support_receipts[index] for index in PERMUTATION]
    scalar_ids = [
        item["candidate_id"] for item in primary["candidate_scalar_receipts"]
    ]
    permuted_ids = [candidate_ids[index] for index in PERMUTATION]
    passed = (
        len(outline_domain_hashes) == 3
        and all(
            all(value.get(name) == domain_hashes[name] for name in value)
            for value in outline_domain_hashes.values()
        )
        and slots_passed
        and not forbidden_arguments
        and tuple(sorted(PERMUTATION)) == tuple(range(CANDIDATE_COUNT))
        and PERMUTATION != tuple(range(CANDIDATE_COUNT))
        and scalar_ids == candidate_ids
        and support_receipts == saved_support_receipts
        and permuted_ids == [scalar_ids[index] for index in PERMUTATION]
        and permuted_support_receipts
        == [_array_receipt(permuted_support[index]) for index in range(CANDIDATE_COUNT)]
    )
    evidence = {
        "target_score_masks_signature": signature,
        "forbidden_candidate_arguments": forbidden_arguments,
        "original_candidate_order_sha256": canonical_payload_sha256(candidate_ids),
        "permuted_candidate_order_sha256": canonical_payload_sha256(
            permuted_ids
        ),
        "original_support_receipt": _array_receipt(original_support),
        "permuted_support_receipt": _array_receipt(permuted_support),
        "permuted_individual_support_receipts": permuted_support_receipts,
        "outline_domain_hashes": outline_domain_hashes,
        "domain_pixel_counts": {
            item["domain"]: item["pixel_count"] for item in domains
        },
    }
    return evidence, bool(passed)


def _shuffled_binding_evidence(
    bank_runtime: dict[str, object],
    shuffled: dict[str, object],
    target_runtime: dict[str, object],
) -> tuple[dict[str, object], bool]:
    bank_case = int(bank_runtime["case_index"])
    target_case = (bank_case + 17) % CASE_COUNT
    common = shuffled["common_lattice_resampling"]
    coordinates = common_lattice_map_yx(
        OUTPUT_SHAPE,
        float(bank_runtime["record"]["target"]["pixel_pitch_um"]),
        float(shuffled["target_identity"]["target_pixel_pitch_um"]),
    )
    records = []
    passed = common["coordinate_map_receipt"] == _array_receipt(coordinates)
    for index, expected in enumerate(common["resampled_candidates"]):
        scalar = resample_common_lattice_intensity(
            bank_runtime["scaled"][index], coordinates
        )
        support = resample_common_lattice_support(
            bank_runtime["supports"][index], coordinates
        )
        null = resample_common_lattice_intensity(
            bank_runtime["null_images"][index], coordinates
        )
        current = {
            "candidate_index": index,
            "candidate_id": bank_runtime["record"]["candidate_bank"][
                "ordered_candidate_ids"
            ][index],
            "scalar_float64_receipt": _array_receipt(scalar),
            "support_bool_receipt": _array_receipt(support),
            "constant_null_scalar_float64_receipt": _array_receipt(null),
            "constant_supported_mean": float(bank_runtime["supported_means"][index]),
        }
        _self_hash(current, "payload_sha256")
        records.append(current)
        passed &= current == expected
    bank_identity = shuffled["bank_identity"]
    target_identity = shuffled["target_identity"]
    target_primary = target_runtime["record"]
    target_domains = {
        item["domain"]: item for item in target_primary["score_domains"]
    }
    slot_domains_passed = True
    for outline in shuffled["outline_results"]:
        for slot in outline["score_slots"]:
            expected_domain = target_domains[slot["domain"]]
            slot_domains_passed &= (
                slot["domain_mask_receipt_sha256"]
                == _domain_mask_receipt_sha256(expected_domain)
                and slot["domain_pixel_count"] == expected_domain["pixel_count"]
            )
    passed &= (
        shuffled["bank_case_index"] == bank_case
        and shuffled["target_case_index"] == target_case
        and bank_identity["source_primary_payload_sha256"]
        == bank_runtime["record"]["payload_sha256"]
        and bank_identity["finite_candidate_bank_id"]
        == bank_runtime["record"]["candidate_bank"]["finite_candidate_bank_id"]
        and bank_identity["finite_candidate_receipt_sha256"]
        == bank_runtime["record"]["candidate_bank"]["finite_candidate_receipt_sha256"]
        and bank_identity["ordered_candidate_ids"]
        == bank_runtime["record"]["candidate_bank"]["ordered_candidate_ids"]
        and bank_identity["ordered_candidate_ids_sha256"]
        == bank_runtime["record"]["candidate_bank"]["ordered_candidate_ids_sha256"]
        and bank_identity["truth_candidate_id"]
        == bank_runtime["record"]["candidate_bank"]["truth_candidate_id"]
        and bank_identity["source_pixel_pitch_um"]
        == bank_runtime["record"]["target"]["pixel_pitch_um"]
        and bank_identity["candidate_scalar_receipts_sha256"]
        == canonical_payload_sha256(bank_runtime["record"]["candidate_scalar_receipts"])
        and common["source_pixel_pitch_um"]
        == bank_runtime["record"]["target"]["pixel_pitch_um"]
        and common["target_pixel_pitch_um"]
        == target_identity["target_pixel_pitch_um"]
        and common["resampled_candidates"] == records
        and target_identity["target_case_index"] == target_case
        and target_identity["target_primary_payload_sha256"]
        == target_primary["payload_sha256"]
        and target_identity["paired_view_group_id"]
        == target_primary["target"]["paired_view_group_id"]
        and target_identity["target_pixel_pitch_um"]
        == target_primary["target"]["pixel_pitch_um"]
        and target_identity["target_outline_payload_sha256"]
        == [item["payload_sha256"] for item in target_primary["outline_results"]]
        and target_identity["score_domain_receipts_sha256"]
        == canonical_payload_sha256(target_primary["score_domains"])
        and shuffled["score_domains"] == target_primary["score_domains"]
        and slot_domains_passed
    )
    evidence = {
        "bank_case_index": bank_case,
        "target_case_index": target_case,
        "bank_identity": bank_identity,
        "target_identity": target_identity,
        "coordinate_map_receipt": _array_receipt(coordinates),
        "resampled_candidates": records,
    }
    return evidence, bool(passed)


def _find_slot(
    record: dict[str, object], outline_mode: str, descriptor: str, domain: str
) -> dict[str, object]:
    outline = next(
        item for item in record["outline_results"] if item["outline_mode"] == outline_mode
    )
    return next(
        item
        for item in outline["score_slots"]
        if item["descriptor"] == descriptor and item["domain"] == domain
    )


def _build_case_control(
    bank_runtime: dict[str, object],
    target_runtime: dict[str, object],
    shuffled_record: dict[str, object],
    shuffled_landscape_controls: list[dict[str, object]],
    support_index: dict[str, object],
) -> dict[str, object]:
    case_index = int(bank_runtime["case_index"])
    primary = bank_runtime["record"]
    replayed = bank_runtime["replayed"]
    geometry_evidence, geometry_passed = _geometry_control_evidence(
        replayed, support_index
    )
    signature_evidence, signature_passed = _scorer_signature_evidence()
    dewarp_evidence, dewarp_passed = _recompute_dewarp_and_domain_evidence(
        bank_runtime, support_index
    )
    invariance_evidence, invariance_passed = _target_domain_invariance_evidence(
        bank_runtime
    )
    shuffled_evidence, shuffled_passed = _shuffled_binding_evidence(
        bank_runtime, shuffled_record, target_runtime
    )
    accurate = _find_slot(primary, ACCURATE_OUTLINE, "MIND", "core")
    absent = _find_slot(primary, ABSENT_OUTLINE, "MIND", "core")
    core_equal = _float64_vector_bytes(accurate["scores"]) == _float64_vector_bytes(
        absent["scores"]
    )
    landscape = list(bank_runtime["landscape_controls"]) + list(
        shuffled_landscape_controls
    )
    if len(landscape) != 48:
        raise ValueError("each case must bind exactly 36 native and 12 shuffled landscapes")
    affine = list(bank_runtime["affine_controls"])
    if len(affine) != (36 if case_index in AFFINE_CASES else 0):
        raise ValueError("affine controls changed their exact four-case/36-slot schedule")
    scalar_passed = all(
        item["crosscheck_passed"] is True
        for item in primary["candidate_scalar_receipts"]
    ) and len(primary["candidate_scalar_receipts"]) == CANDIDATE_COUNT
    domains = primary["score_domains"]
    mask_only = primary["provenance"]["mask_only_Dice"]
    mask_only_passed = (
        mask_only["source_case_payload_sha256"]
        == primary["semantic_case_payload_sha256"]
        and mask_only["candidate_bank_id"]
        == primary["candidate_bank"]["finite_candidate_bank_id"]
        and len(mask_only["values"]) == CANDIDATE_COUNT
    )
    checks = {
        "source_replay_metadata_geometry": _basic_control(
            {
                "frozen_replay": primary["frozen_replay"],
                "provenance": primary["provenance"],
                "candidate_bank": primary["candidate_bank"],
            },
            primary["frozen_replay"]["replay_passed"] is True,
        ),
        "candidate_scalar_annotation_mask": _basic_control(
            primary["candidate_scalar_receipts"], scalar_passed
        ),
        "dewarp_direction_and_masks": _basic_control(
            dewarp_evidence, dewarp_passed
        ),
        "rp2_and_xy_wh": _basic_control(geometry_evidence, geometry_passed),
        "scorer_signature_exclusion": _basic_control(
            signature_evidence, signature_passed
        ),
        "target_domain_invariance": _basic_control(
            invariance_evidence, invariance_passed
        ),
        "accurate_absent_core_identity": _basic_control(
            {
                "accurate_score_vector_sha256": _score_vector_sha256(accurate["scores"]),
                "absent_score_vector_sha256": _score_vector_sha256(absent["scores"]),
            },
            core_equal,
        ),
        "shuffled_binding": _basic_control(
            shuffled_evidence,
            shuffled_passed,
        ),
        "mask_only_verification": _basic_control(mask_only, mask_only_passed),
        "landscape_controls": landscape,
        "affine_and_polarity": affine,
    }
    validate_payload_keys(checks, "case_checks")
    record = {
        "schema": "anatomy-tracker.arbitrary-plane-image-information-controls/v1",
        "case_index": case_index,
        "checks": checks,
        "evidence_receipt_sha256": canonical_payload_sha256(checks),
    }
    _self_hash(record, "payload_sha256")
    return validate_payload_keys(record, "control")


def _wilson_95(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [float(center - half), float(center + half)]


def _summarize_slots(
    scope: str,
    descriptor: str,
    domain: str,
    outline_mode: str | None,
    slots: list[dict[str, object]],
    entered_gate: bool,
) -> dict[str, object]:
    eligible = [item for item in slots if item["status"] == "ok"]
    insufficient = [item for item in slots if item["status"] == "insufficient_domain"]
    if len(eligible) + len(insufficient) != CASE_COUNT:
        raise ValueError("a slot summary must contain exactly one record per base")
    rankings = [item["ranking"] for item in eligible]
    successes = sum(bool(item["top1"]) for item in rankings)
    record = {
        "scope": scope,
        "descriptor": descriptor,
        "domain": domain,
        "outline_mode": outline_mode,
        "entered_gate": bool(entered_gate),
        "eligible_base_count": len(eligible),
        "insufficient_base_count": len(insufficient),
        "top1_success_count": successes,
        "top1_rate": None if not eligible else float(successes / len(eligible)),
        "wilson_95": _wilson_95(successes, len(eligible)),
        "mean_reciprocal_rank": (
            None
            if not eligible
            else float(np.mean([item["reciprocal_rank"] for item in rankings]))
        ),
        "median_true_rank": (
            None if not eligible else float(np.median([item["true_rank"] for item in rankings]))
        ),
        "median_truth_versus_decoy_win_fraction": (
            None
            if not eligible
            else float(
                np.median(
                    [item["truth_versus_decoy_win_fraction"] for item in rankings]
                )
            )
        ),
        "median_truth_score_margin": (
            None
            if not eligible
            else float(np.median([item["truth_score_margin"] for item in rankings]))
        ),
    }
    return validate_payload_keys(record, "slot_summary")


def _stratum_summary(
    primary: list[dict[str, object]],
    indices: list[int],
    stratum_type: str,
    stratum_value: object,
    descriptor: str,
    domain: str,
    outline: str,
) -> dict[str, object]:
    rankings = [
        _find_slot(primary[index], outline, descriptor, domain)["ranking"]
        for index in indices
    ]
    successes = sum(bool(item["top1"]) for item in rankings)
    record = {
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "endpoint": f"native/{descriptor}/{domain}/{outline}",
        "base_indices": indices,
        "base_count": len(indices),
        "top1_success_count": successes,
        "top1_rate": float(successes / len(indices)),
        "mean_reciprocal_rank": float(
            np.mean([item["reciprocal_rank"] for item in rankings])
        ),
        "median_true_rank": float(np.median([item["true_rank"] for item in rankings])),
        "median_truth_score_margin": float(
            np.median([item["truth_score_margin"] for item in rankings])
        ),
    }
    return validate_payload_keys(record, "stratum_summary")


def aggregate_metrics(
    primary: list[dict[str, object]],
    shuffled: list[dict[str, object]],
    pooled_membership: dict[str, list[int]],
) -> dict[str, object]:
    if (
        len(primary) != CASE_COUNT
        or len(shuffled) != CASE_COUNT
        or [item["case_index"] for item in primary] != list(range(CASE_COUNT))
        or [item["bank_case_index"] for item in shuffled] != list(range(CASE_COUNT))
    ):
        raise ValueError("metrics require exact ordered 64-case primary and shuffled records")
    validate_payload_keys(pooled_membership, "pooled_strata_membership")
    for name, expected_count in (("challenging_appearance", 22), ("damaged", 27)):
        indices = pooled_membership[name]
        if (
            not isinstance(indices, list)
            or len(indices) != expected_count
            or indices != sorted(set(indices))
            or any(type(index) is not int or index not in range(CASE_COUNT) for index in indices)
        ):
            raise ValueError(f"frozen pooled stratum {name} changed exact membership")
    expected_membership = {
        name: list(indices) for name, indices in FROZEN_POOLED_MEMBERSHIP.items()
    }
    observed_membership = {
        "challenging_appearance": [
            index
            for index, item in enumerate(primary)
            if item["provenance"]["reporting_strata"]["appearance_family"]
            in {"label-conditioned", "template-label-mixture"}
            and item["provenance"]["reporting_strata"][
                "challenging_appearance_member"
            ]
            is True
        ],
        "damaged": [
            index
            for index, item in enumerate(primary)
            if item["provenance"]["reporting_strata"]["damage_event_count"] >= 1
            and item["provenance"]["reporting_strata"]["damaged_member"] is True
        ],
    }
    if pooled_membership != expected_membership or observed_membership != pooled_membership:
        raise ValueError("primary reporting strata differ from pre-score frozen membership")
    native_summaries = []
    for outline in OUTLINE_MODES:
        for descriptor, domain in NATIVE_OUTLINE_SLOT_SCHEDULE:
            slots = [_find_slot(item, outline, descriptor, domain) for item in primary]
            native_summaries.append(
                _summarize_slots(
                    "native",
                    descriptor,
                    domain,
                    outline,
                    slots,
                    _entered_gate(descriptor, domain, outline, shuffled=False),
                )
            )
    mask_slots = [
        {
            "status": "ok",
            "ranking": item["provenance"]["mask_only_Dice"]["recomputed_ranking"],
        }
        for item in primary
    ]
    native_summaries.append(
        _summarize_slots(
            "native",
            "mask_only_Dice",
            "frozen_semantic_fixed_valid_mask",
            None,
            mask_slots,
            False,
        )
    )
    shuffled_summaries = []
    for outline in OUTLINE_MODES:
        for descriptor, domain in SHUFFLED_OUTLINE_SLOT_SCHEDULE:
            slots = [_find_slot(item, outline, descriptor, domain) for item in shuffled]
            shuffled_summaries.append(
                _summarize_slots(
                    "shuffled",
                    descriptor,
                    domain,
                    outline,
                    slots,
                    _entered_gate(descriptor, domain, outline, shuffled=True),
                )
            )
    reporting = []
    orientation_values = ("near_AP", "near_DV", "near_ML", "general_oblique")
    for value in orientation_values:
        indices = [
            index
            for index, item in enumerate(primary)
            if item["provenance"]["reporting_strata"]["orientation_family"] == value
        ]
        reporting.append(
            _stratum_summary(
                primary, indices, "orientation_family", value, "MIND", "context", ABSENT_OUTLINE
            )
        )
    observed_orientation_counts = {
        value: sum(
            item["provenance"]["reporting_strata"]["orientation_family"] == value
            for item in primary
        )
        for value in orientation_values
    }
    if observed_orientation_counts != ORIENTATION_COUNTS:
        raise ValueError("frozen orientation membership changed from 12/12/12/28")
    appearances = sorted(
        {item["provenance"]["reporting_strata"]["appearance_family"] for item in primary}
    )
    for value in appearances:
        indices = [
            index
            for index, item in enumerate(primary)
            if item["provenance"]["reporting_strata"]["appearance_family"] == value
        ]
        reporting.append(
            _stratum_summary(
                primary, indices, "appearance_family", value, "MIND", "core", ABSENT_OUTLINE
            )
        )
    damage_counts = sorted(
        {item["provenance"]["reporting_strata"]["damage_event_count"] for item in primary}
    )
    for value in damage_counts:
        indices = [
            index
            for index, item in enumerate(primary)
            if item["provenance"]["reporting_strata"]["damage_event_count"] == value
        ]
        reporting.append(
            _stratum_summary(
                primary, indices, "damage_event_count", value, "MIND", "core", ABSENT_OUTLINE
            )
        )
    damage_types = sorted(
        {
            value
            for item in primary
            for value in item["provenance"]["reporting_strata"]["damage_event_types"]
        }
    )
    for value in damage_types:
        indices = [
            index
            for index, item in enumerate(primary)
            if value in item["provenance"]["reporting_strata"]["damage_event_types"]
        ]
        reporting.append(
            _stratum_summary(
                primary, indices, "damage_event_type", value, "MIND", "core", ABSENT_OUTLINE
            )
        )
    pooled = []
    for stratum, field in (
        ("challenging_appearance", "challenging_appearance_member"),
        ("damaged", "damaged_member"),
    ):
        observed_indices = [
            index
            for index, item in enumerate(primary)
            if item["provenance"]["reporting_strata"][field] is True
        ]
        indices = pooled_membership[stratum]
        if observed_indices != indices:
            raise ValueError(f"frozen pooled stratum {stratum} changed membership")
        for descriptor in ("MIND", "constant-within-support-MIND-null"):
            pooled.append(
                _stratum_summary(
                    primary,
                    indices,
                    stratum,
                    "member",
                    descriptor,
                    "core",
                    ABSENT_OUTLINE,
                )
            )
    paired = []
    absent = [
        _find_slot(item, ABSENT_OUTLINE, "MIND", "context")["ranking"]
        for item in primary
    ]
    for outline in (ACCURATE_OUTLINE, IMPERFECT_OUTLINE):
        comparison = [
            _find_slot(item, outline, "MIND", "context")["ranking"] for item in primary
        ]
        record = {
            "comparison": f"{outline}-minus-{ABSENT_OUTLINE}",
            "base_count": CASE_COUNT,
            "top1_rate_difference": float(
                np.mean([item["top1"] for item in comparison])
                - np.mean([item["top1"] for item in absent])
            ),
            "median_rank_difference": float(
                np.median(
                    [
                        current["true_rank"] - reference["true_rank"]
                        for current, reference in zip(comparison, absent, strict=True)
                    ]
                )
            ),
            "median_truth_margin_difference": float(
                np.median(
                    [
                        current["truth_score_margin"] - reference["truth_score_margin"]
                        for current, reference in zip(comparison, absent, strict=True)
                    ]
                )
            ),
        }
        paired.append(validate_payload_keys(record, "paired_summary"))
    metrics = {
        "native_slot_summaries": native_summaries,
        "shuffled_slot_summaries": shuffled_summaries,
        "reporting_stratum_summaries": reporting,
        "pooled_safeguard_summaries": pooled,
        "paired_outline_comparisons": paired,
    }
    return validate_payload_keys(metrics, "result_metrics")


def _metric_slot(
    metrics: dict[str, object],
    collection: str,
    descriptor: str,
    domain: str,
    outline: str,
) -> dict[str, object]:
    return next(
        item
        for item in metrics[collection]
        if item["descriptor"] == descriptor
        and item["domain"] == domain
        and item["outline_mode"] == outline
    )


def _atomic_gate(
    gate_id: str,
    pointer: str,
    operator: str,
    threshold: int | float | bool,
    observed: int | float | bool,
    passed: bool,
) -> dict[str, object]:
    record = {
        "gate_id": gate_id,
        "source_metric_pointer": pointer,
        "operator": operator,
        "threshold": threshold,
        "observed": observed,
        "passed": bool(passed),
    }
    record["evidence_sha256"] = canonical_payload_sha256(record)
    return validate_payload_keys(record, "atomic_gate")


def evaluate_gates(
    metrics: dict[str, object],
    global_controls: dict[str, object],
) -> dict[str, object]:
    """Evaluate only the predeclared gate slots from an aggregate metrics object."""
    _validate_metrics_contract(metrics)
    validate_payload_keys(global_controls, "global_controls")
    inventory_passed = global_controls["frozen_inventory_audit"]["passed"] is True
    source_passed = global_controls["source_and_signature_audit"]["passed"] is True
    controls_passed = global_controls["affine_and_polarity_controls"]["passed"] is True
    checks = [
        _atomic_gate("frozen-inventory-integrity", "global_controls.json#/frozen_inventory_audit/passed", "is", True, inventory_passed, inventory_passed),
        _atomic_gate("source-signature-integrity", "global_controls.json#/source_and_signature_audit/passed", "is", True, source_passed, source_passed),
        _atomic_gate("all-case-controls-integrity", "global_controls.json#/affine_and_polarity_controls/passed", "is", True, controls_passed, controls_passed),
    ]
    def pointer(collection: str, item: dict[str, object], field: str) -> str:
        return f"result.json#/metrics/{collection}/{metrics[collection].index(item)}/{field}"
    absent_context = _metric_slot(
        metrics, "native_slot_summaries", "MIND", "context", ABSENT_OUTLINE
    )
    checks.extend(
        [
            _atomic_gate("absent-context-top1-count", pointer("native_slot_summaries", absent_context, "top1_success_count"), ">=", 39, absent_context["top1_success_count"], absent_context["top1_success_count"] >= 39),
            _atomic_gate("absent-context-wilson-lower", pointer("native_slot_summaries", absent_context, "wilson_95") + "/0", ">=", 0.45, absent_context["wilson_95"][0], absent_context["wilson_95"][0] >= 0.45),
            _atomic_gate("absent-context-mean-rr", pointer("native_slot_summaries", absent_context, "mean_reciprocal_rank"), ">=", 0.70, absent_context["mean_reciprocal_rank"], absent_context["mean_reciprocal_rank"] >= 0.70),
            _atomic_gate("absent-context-median-rank", pointer("native_slot_summaries", absent_context, "median_true_rank"), "<=", 1.0, absent_context["median_true_rank"], absent_context["median_true_rank"] <= 1.0),
            _atomic_gate("absent-context-median-win-fraction", pointer("native_slot_summaries", absent_context, "median_truth_versus_decoy_win_fraction"), ">=", 0.90, absent_context["median_truth_versus_decoy_win_fraction"], absent_context["median_truth_versus_decoy_win_fraction"] >= 0.90),
        ]
    )
    for orientation, minimum in (("near_AP", 5), ("near_DV", 5), ("near_ML", 5), ("general_oblique", 12)):
        summary = next(
            item
            for item in metrics["reporting_stratum_summaries"]
            if item["stratum_type"] == "orientation_family" and item["stratum_value"] == orientation
        )
        checks.append(_atomic_gate(f"orientation-{orientation}-top1-count", pointer("reporting_stratum_summaries", summary, "top1_success_count"), ">=", minimum, summary["top1_success_count"], summary["top1_success_count"] >= minimum))
    absent_core = _metric_slot(metrics, "native_slot_summaries", "MIND", "core", ABSENT_OUTLINE)
    absent_null = _metric_slot(metrics, "native_slot_summaries", "constant-within-support-MIND-null", "core", ABSENT_OUTLINE)
    checks.extend(
        [
            _atomic_gate("absent-core-top1-count", pointer("native_slot_summaries", absent_core, "top1_success_count"), ">=", 32, absent_core["top1_success_count"], absent_core["top1_success_count"] >= 32),
            _atomic_gate("absent-core-mean-rr", pointer("native_slot_summaries", absent_core, "mean_reciprocal_rank"), ">=", 0.65, absent_core["mean_reciprocal_rank"], absent_core["mean_reciprocal_rank"] >= 0.65),
            _atomic_gate("absent-core-minus-null-success-count", "result.json#/metrics/native_slot_summaries", "success-count-difference>=", 7, absent_core["top1_success_count"] - absent_null["top1_success_count"], absent_core["top1_success_count"] - absent_null["top1_success_count"] >= 7),
        ]
    )
    for stratum, success_minimum, rr_minimum in (("challenging_appearance", 7, 0.45), ("damaged", 9, 0.45)):
        mind = next(item for item in metrics["pooled_safeguard_summaries"] if item["stratum_type"] == stratum and "/MIND/" in item["endpoint"])
        null = next(item for item in metrics["pooled_safeguard_summaries"] if item["stratum_type"] == stratum and "/constant-within-support-MIND-null/" in item["endpoint"])
        checks.extend(
            [
                _atomic_gate(f"{stratum}-top1-count", pointer("pooled_safeguard_summaries", mind, "top1_success_count"), ">=", success_minimum, mind["top1_success_count"], mind["top1_success_count"] >= success_minimum),
                _atomic_gate(f"{stratum}-mean-rr", pointer("pooled_safeguard_summaries", mind, "mean_reciprocal_rank"), ">=", rr_minimum, mind["mean_reciprocal_rank"], mind["mean_reciprocal_rank"] >= rr_minimum),
                _atomic_gate(f"{stratum}-minus-null-success-count", "result.json#/metrics/pooled_safeguard_summaries", "success-count-difference>=", 2, mind["top1_success_count"] - null["top1_success_count"], mind["top1_success_count"] - null["top1_success_count"] >= 2),
            ]
        )
    absent_successes = absent_context["top1_success_count"]
    for outline in (ACCURATE_OUTLINE, IMPERFECT_OUTLINE):
        summary = _metric_slot(metrics, "native_slot_summaries", "MIND", "context", outline)
        checks.extend(
            [
                _atomic_gate(f"brush-{outline}-top1-count", pointer("native_slot_summaries", summary, "top1_success_count"), ">=", 32, summary["top1_success_count"], summary["top1_success_count"] >= 32),
                _atomic_gate(f"brush-{outline}-deficit", "result.json#/metrics/native_slot_summaries", "success-count-deficit<=", 6, absent_successes - summary["top1_success_count"], absent_successes - summary["top1_success_count"] <= 6),
            ]
        )
    for outline in OUTLINE_MODES:
        summary = _metric_slot(metrics, "shuffled_slot_summaries", "MIND", "context", outline)
        checks.extend(
            [
                _atomic_gate(f"shuffled-{outline}-top1-count", pointer("shuffled_slot_summaries", summary, "top1_success_count"), "<=", 6, summary["top1_success_count"], summary["top1_success_count"] <= 6),
                _atomic_gate(f"shuffled-{outline}-mean-rr", pointer("shuffled_slot_summaries", summary, "mean_reciprocal_rank"), "<=", 0.15, summary["mean_reciprocal_rank"], summary["mean_reciprocal_rank"] <= 0.15),
            ]
        )
    passed = all(item["passed"] for item in checks)
    record = {
        "global_controls_payload_sha256": global_controls["payload_sha256"],
        "atomic_checks": checks,
        "passed": bool(passed),
        "decision": "PASS" if passed else "FAIL",
    }
    return validate_payload_keys(record, "gates")


def _resolved_config(
    repository_state_record: dict[str, object],
    source_records: list[dict[str, str]],
    semantic_result: dict[str, object],
    render_context: Mapping[str, object],
    support_index: dict[str, object],
    pooled_membership: dict[str, list[int]],
) -> dict[str, object]:
    pooled_membership = copy.deepcopy(pooled_membership)
    validate_payload_keys(pooled_membership, "pooled_strata_membership")
    if pooled_membership != {
        name: list(indices) for name, indices in FROZEN_POOLED_MEMBERSHIP.items()
    }:
        raise ValueError("resolved config pooled membership changed from 22/27 bases")
    source_by_path = {item["relative_path"]: item for item in source_records}
    preflight = source_by_path[PREFLIGHT_RELATIVE_PATH]
    repository = {
        "branch": repository_state_record["branch"],
        "source_parent_commit": SOURCE_PARENT_COMMIT,
        "execution_commit": repository_state_record["head"],
        "origin_commit": repository_state_record["upstream_head"],
        "worktree_clean": repository_state_record["worktree_clean"],
        "preflight_path": PREFLIGHT_RELATIVE_PATH,
        "preflight_git_blob_sha256": preflight["git_blob_sha256"],
        "preflight_checkout_sha256": preflight["checkout_sha256"],
    }
    validate_payload_keys(repository, "repository")
    frozen = {
        "source_commit": FROZEN_SEMANTIC_SOURCE_COMMIT,
        "output_relative_path": FROZEN_SEMANTIC_OUTPUT.relative_to(ROOT).as_posix(),
        "result_json_sha256": FROZEN_SEMANTIC_RESULT_FILE_SHA256,
        "result_payload_sha256": semantic_result["result_payload_sha256"],
        "resolved_config_payload_sha256": FROZEN_SEMANTIC_CONFIG_PAYLOAD_SHA256,
        "inventory_file_count": FROZEN_SEMANTIC_FILE_COUNT,
        "inventory_total_bytes": FROZEN_SEMANTIC_TOTAL_BYTES,
        "inventory_sha256": FROZEN_SEMANTIC_INVENTORY_SHA256,
        "support_index_sha256": SUPPORT_INDEX_SHA256,
        "prepared_render_context_sha256": PREPARED_RENDER_CONTEXT_SHA256,
        "prepared_candidate_annotation_context_sha256": PREPARED_CANDIDATE_CONTEXT_SHA256,
        "decoded_template_array_sha256": DECODED_TEMPLATE_ARRAY_SHA256,
        "scalar_conversion_array_sha256": SCALAR_CONVERSION_ARRAY_SHA256,
        "decoded_annotation_array_sha256": DECODED_ANNOTATION_ARRAY_SHA256,
    }
    validate_payload_keys(frozen, "frozen_semantic_input")
    asset = render_context["asset_receipt"]
    atlas_assets = {
        "template_path": ATLAS_TEMPLATE_URI,
        "template_source_sha256": ATLAS_TEMPLATE_SHA256,
        "template_decoded_receipt": _plain(asset["template_decoded"]),
        "scalar_conversion_receipt": _plain(asset["scalar_conversion"]),
        "annotation_path": ATLAS_ANNOTATION_URI,
        "annotation_source_sha256": ATLAS_ANNOTATION_SHA256,
        "annotation_decoded_receipt": _plain(asset["annotation_decoded"]),
        "support_index_sha256": support_index["support_index_sha256"],
        "global_tissue_voxel_count": 32_387_385,
        "quantile_probabilities": [0.005, 0.995],
        "quantile_values": [6.0, 287.0],
    }
    validate_payload_keys(atlas_assets, "atlas_assets")
    descriptor_constants = {
        "algorithm": IMAGE_INFORMATION_ALGORITHM,
        "numeric_dtype": "float64",
        "scalar_padding": 0.0,
        "candidate_scaling": "render float32 2-D first; clip((J-6)/281,0,1) in float64",
        "tie_tolerance": TIE_TOLERANCE,
        "context_radius_um": 1000.0,
        "primary_domain_minima": copy.deepcopy(PRIMARY_MINIMUM_PIXELS),
        "mind": {
            "search_displacement_um": 100.0,
            "gaussian_patch_sigma_um": 50.0,
            "gaussian_truncate_sigma": 3.0,
            "offset_order": "axial y-/y+/x-/x+ then diagonal --/-+/+-/++",
        },
        "hog": {"cell_width_um": 400.0, "orientation_bins": 9, "block_cells": [2, 2]},
        "normalized_gradient_like": {
            "gaussian_sigma_um": 100.0,
            "gaussian_radius_um": 300.0,
            "polarity_invariant": True,
        },
        "common_lattice": {
            "intensity": "bilinear-zero float64",
            "support": "rint-ties-to-even nearest-zero",
            "canvas_center_xy_over_wh": [0.5, 0.5],
        },
    }
    validate_payload_keys(descriptor_constants["mind"], "mind_constants")
    validate_payload_keys(descriptor_constants["hog"], "hog_constants")
    validate_payload_keys(
        descriptor_constants["normalized_gradient_like"], "ngf_constants"
    )
    validate_payload_keys(
        descriptor_constants["common_lattice"], "common_lattice_constants"
    )
    validate_payload_keys(descriptor_constants, "descriptor_constants")
    case_contract = {
        "base_count": CASE_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "output_shape_h_w": list(OUTPUT_SHAPE),
        "outline_order": list(OUTLINE_MODES),
        "orientation_counts": copy.deepcopy(ORIENTATION_COUNTS),
        "pooled_strata_counts": {"challenging_appearance": 22, "damaged": 27},
        "pooled_strata_membership": pooled_membership,
        "shuffled_offset": 17,
        "candidate_permutation": list(PERMUTATION),
        "chunk_sizes": list(CHUNK_CONTROL_SIZES),
        "affine_case_indices": list(AFFINE_CASES),
        "affine_transforms": [
            {"name": name, "scale": scale, "offset": offset}
            for name, scale, offset in AFFINE_TRANSFORMS
        ],
        "streaming_contract": (
            "one pending bank plus one current bank; descendants one at a time; "
            "only three minimal dewarped target rasters retained for the +17 handoff"
        ),
    }
    validate_payload_keys(case_contract["pooled_strata_counts"], "pooled_strata_counts")
    validate_payload_keys(
        case_contract["pooled_strata_membership"], "pooled_strata_membership"
    )
    for transform in case_contract["affine_transforms"]:
        validate_payload_keys(transform, "config_affine_transform")
    validate_payload_keys(case_contract, "case_and_shuffle_contract")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "torch": version("torch"),
    }
    validate_payload_keys(environment, "environment")
    record = {
        "schema": RUNNER_SCHEMA,
        "preflight_sha256": preflight["git_blob_sha256"],
        "repository": repository,
        "frozen_semantic_input": frozen,
        "atlas_assets": atlas_assets,
        "descriptor_constants": descriptor_constants,
        "case_and_shuffle_contract": case_contract,
        "environment": environment,
        "model_independence": copy.deepcopy(MODEL_INDEPENDENCE),
        "data_access": copy.deepcopy(DATA_ACCESS),
        "source_sha256": copy.deepcopy(source_records),
    }
    validate_payload_keys(record["model_independence"], "model_independence")
    validate_payload_keys(record["data_access"], "data_access")
    _self_hash(record, "resolved_config_sha256")
    return validate_payload_keys(record, "resolved_config")


def _case_control_passed(record: dict[str, object]) -> bool:
    checks = record["checks"]
    basic_names = (
        "source_replay_metadata_geometry",
        "candidate_scalar_annotation_mask",
        "dewarp_direction_and_masks",
        "rp2_and_xy_wh",
        "scorer_signature_exclusion",
        "target_domain_invariance",
        "accurate_absent_core_identity",
        "shuffled_binding",
        "mask_only_verification",
    )
    if any(checks[name]["passed"] is not True for name in basic_names):
        return False
    for name in ("landscape_controls", "affine_and_polarity"):
        for item in checks[name]:
            if not (
                item["passed"] is True
                or (
                    item["status"] == "authenticated_not_applicable"
                    and item["passed"] is None
                )
            ):
                return False
    return True


def _ast_dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _build_global_controls(
    repository: dict[str, object],
    source_records: list[dict[str, str]],
    semantic_inventory: list[dict[str, object]],
    case_controls: list[dict[str, object]],
) -> dict[str, object]:
    fresh_repository = repository_state()
    fresh_sources = _source_hash_receipts(fresh_repository)
    _authenticate_frozen_semantic_output(fresh_sources)
    fresh_semantic_inventory = _inventory(FROZEN_SEMANTIC_OUTPUT)
    observed_count = len(fresh_semantic_inventory)
    observed_bytes = sum(int(item["size_bytes"]) for item in fresh_semantic_inventory)
    observed_hash = canonical_payload_sha256(fresh_semantic_inventory)
    inventory = {
        "expected_file_count": FROZEN_SEMANTIC_FILE_COUNT,
        "observed_file_count": observed_count,
        "expected_total_bytes": FROZEN_SEMANTIC_TOTAL_BYTES,
        "observed_total_bytes": observed_bytes,
        "expected_inventory_sha256": FROZEN_SEMANTIC_INVENTORY_SHA256,
        "observed_inventory_sha256": observed_hash,
        "passed": bool(
            observed_count == FROZEN_SEMANTIC_FILE_COUNT
            and observed_bytes == FROZEN_SEMANTIC_TOTAL_BYTES
            and observed_hash == FROZEN_SEMANTIC_INVENTORY_SHA256
        ),
    }
    validate_payload_keys(inventory, "frozen_inventory_audit")
    signatures, signatures_passed = _scorer_signature_evidence()
    scanned_paths = (RUNNER_RELATIVE_PATH, *SCORER_RELATIVE_PATHS[:3])
    source_text = {
        path: (ROOT / path).read_text(encoding="utf-8") for path in scanned_paths
    }
    dependency_tokens = {
        "learned_checkpoint_dependencies": {"torch.load", "load_state_dict", "checkpoint_path"},
        "previous_model_dependencies": {"model.predict", "previous_model", "old_model"},
        "pretrained_feature_dependencies": {"from_pretrained", "torchvision", "timm", "transformers"},
        "legacy_descriptor_dependencies": {"pseudolabel", "pseudo_label", "dense_registration_mind"},
    }
    model_records = []
    for dependency, tokens in dependency_tokens.items():
        matches = []
        for path, text in source_text.items():
            tree = ast.parse(text, filename=path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names]
                    if isinstance(node, ast.ImportFrom) and node.module:
                        names.append(node.module)
                    for name in names:
                        if any(name == token or name.startswith(token + ".") for token in tokens):
                            matches.append({"relative_path": path, "token": name})
                elif isinstance(node, ast.Call):
                    name = _ast_dotted_name(node.func)
                    if name in tokens or name.rsplit(".", 1)[-1] in tokens:
                        matches.append({"relative_path": path, "token": name})
                elif isinstance(node, ast.Name) and node.id in tokens:
                    matches.append({"relative_path": path, "token": node.id})
        item = {
            "dependency": dependency,
            "declared_values": matches,
            "passed": matches == [],
        }
        model_records.append(validate_payload_keys(item, "model_dependency_record"))
    source_audit = {
        "execution_commit": fresh_repository["head"],
        "origin_commit": fresh_repository["upstream_head"],
        "worktree_clean": fresh_repository["worktree_clean"],
        "source_records": copy.deepcopy(fresh_sources),
        "scorer_signature_records": signatures,
        "model_dependency_records": model_records,
        "passed": bool(
            fresh_repository == repository
            and fresh_sources == source_records
            and semantic_inventory == fresh_semantic_inventory
            and fresh_repository["head"] == fresh_repository["upstream_head"]
            and fresh_repository["worktree_clean"] is True
            and signatures_passed
            and all(item["passed"] for item in model_records)
            and len(source_records) == len(SOURCE_RELATIVE_PATHS)
        ),
    }
    validate_payload_keys(source_audit, "source_signature_audit")
    if (
        len(case_controls) != CASE_COUNT
        or [item["case_index"] for item in case_controls] != list(range(CASE_COUNT))
    ):
        raise ValueError("global controls require 64 ordered case-control sidecars")
    affine_items = [
        item
        for case in case_controls
        for item in case["checks"]["affine_and_polarity"]
    ]
    affine = {
        "required_case_indices": list(AFFINE_CASES),
        "case_control_payload_sha256": [item["payload_sha256"] for item in case_controls],
        "applicable_slot_count": sum(item["status"] != "authenticated_not_applicable" for item in affine_items),
        "authenticated_not_applicable_count": sum(item["status"] == "authenticated_not_applicable" for item in affine_items),
        "passed": bool(all(_case_control_passed(item) for item in case_controls)),
    }
    validate_payload_keys(affine, "affine_summary")
    evidence = {
        "frozen_inventory_audit": inventory,
        "source_and_signature_audit": source_audit,
        "affine_and_polarity_controls": affine,
    }
    record = {
        "schema": "anatomy-tracker.arbitrary-plane-image-information-global-controls/v1",
        **evidence,
        "evidence_receipt_sha256": canonical_payload_sha256(evidence),
    }
    _self_hash(record, "payload_sha256")
    return validate_payload_keys(record, "global_controls")


def _target_runtime_view(runtime: dict[str, object]) -> dict[str, object]:
    return {
        "case_index": runtime["case_index"],
        "record": runtime["record"],
        "target_images": {
            mode: np.array(runtime["target_images"][mode], copy=True, order="C")
            for mode in OUTLINE_MODES
        },
        "domain_by_name": copy.deepcopy(runtime["domain_by_name"]),
        "replayed": {
            "masks": {
                name: np.array(runtime["replayed"]["masks"][name], copy=True, order="C")
                for name in MASK_NAMES
            }
        },
    }


def _compare_or_write_json(
    relative_path: str,
    record: dict[str, object],
    *,
    output: Path | None,
    expected_output: Path | None,
) -> None:
    if (output is None) == (expected_output is None):
        raise ValueError("stream must either write or compare, never both/neither")
    if output is not None:
        _atomic_json(output / relative_path, record)
    elif _read_strict_json(expected_output / relative_path) != record:
        raise ValueError(f"independent replay differs from saved {relative_path}")


def _compare_or_write_masks(
    case_index: int,
    primary: dict[str, object],
    masks: dict[str, np.ndarray],
    *,
    output: Path | None,
    expected_output: Path | None,
) -> None:
    domains = {item["domain"]: item for item in primary["score_domains"]}
    for name in MASK_NAMES:
        receipt = domains[name]["mask_receipt"]
        if receipt != _persisted_mask_receipt(case_index, name, masks[name]):
            raise ValueError("primary mask receipt changed before persistence")
        if output is not None:
            _write_mask(output, receipt, masks[name])
        else:
            path = expected_output / receipt["relative_path"]
            if (
                not path.is_file()
                or _file_sha256(path) != receipt["payload_sha256"]
                or path.read_bytes()
                != np.packbits(
                    np.asarray(masks[name], dtype=bool).reshape(-1, order="C"),
                    bitorder="little",
                ).tobytes()
            ):
                raise ValueError("saved packed mask does not replay exactly")


def _stream_cases(
    frozen_primary: list[dict[str, object]],
    render_context: Mapping[str, object],
    candidate_context: Mapping[str, object],
    support_index: dict[str, object],
    source_records: list[dict[str, str]],
    mask_audit: list[dict[str, object]],
    *,
    output: Path | None = None,
    expected_output: Path | None = None,
) -> list[dict[str, object]]:
    case_controls: dict[int, dict[str, object]] = {}
    pending = None
    first_target = None
    for case_index in shuffled_case_cycle():
        replayed = replay_frozen_case(
            case_index,
            frozen_primary[case_index],
            render_context,
            candidate_context,
            support_index,
        )
        observed_mask_record = build_score_blind_masks(
            lambda _: replayed, case_count=1
        )[0]
        expected_mask_record = copy.deepcopy(mask_audit[case_index])
        observed_mask_record["case_index"] = case_index
        if observed_mask_record != expected_mask_record:
            raise ValueError("scoring replay masks differ from the score-blind prelaunch audit")
        primary, current = _build_primary_case(
            replayed, render_context, support_index, source_records
        )
        _compare_or_write_masks(
            case_index,
            primary,
            replayed["masks"],
            output=output,
            expected_output=expected_output,
        )
        _compare_or_write_json(
            f"primary/case-{case_index:03d}.json",
            primary,
            output=output,
            expected_output=expected_output,
        )
        if case_index == 0:
            first_target = _target_runtime_view(current)
        if pending is not None:
            shuffled, shuffled_controls = _build_shuffled_case(pending, current)
            bank_index = int(pending["case_index"])
            control = _build_case_control(
                pending, current, shuffled, shuffled_controls, support_index
            )
            _compare_or_write_json(
                f"shuffled/case-{bank_index:03d}.json",
                shuffled,
                output=output,
                expected_output=expected_output,
            )
            _compare_or_write_json(
                f"controls/case-{bank_index:03d}.json",
                control,
                output=output,
                expected_output=expected_output,
            )
            case_controls[bank_index] = control
            if output is not None:
                print(
                    json.dumps(
                        {
                            "event": "case-complete",
                            "case_index": bank_index,
                            "primary_payload_sha256": pending["record"]["payload_sha256"],
                            "shuffled_payload_sha256": shuffled["payload_sha256"],
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
            del pending
        pending = current
    if pending is None or first_target is None:
        raise RuntimeError("frozen shuffled cycle did not retain its closing bank/target")
    shuffled, shuffled_controls = _build_shuffled_case(pending, first_target)
    bank_index = int(pending["case_index"])
    control = _build_case_control(
        pending, first_target, shuffled, shuffled_controls, support_index
    )
    _compare_or_write_json(
        f"shuffled/case-{bank_index:03d}.json",
        shuffled,
        output=output,
        expected_output=expected_output,
    )
    _compare_or_write_json(
        f"controls/case-{bank_index:03d}.json",
        control,
        output=output,
        expected_output=expected_output,
    )
    case_controls[bank_index] = control
    if output is not None:
        print(
            json.dumps(
                {
                    "event": "case-complete",
                    "case_index": bank_index,
                    "primary_payload_sha256": pending["record"]["payload_sha256"],
                    "shuffled_payload_sha256": shuffled["payload_sha256"],
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    return [case_controls[index] for index in range(CASE_COUNT)]


def _validate_contexts(
    support_index: dict[str, object],
    render_context: Mapping[str, object],
    candidate_context: Mapping[str, object],
) -> None:
    asset = render_context["asset_receipt"]
    if (
        support_index["support_index_sha256"] != SUPPORT_INDEX_SHA256
        or render_context["prepared_context_sha256"] != PREPARED_RENDER_CONTEXT_SHA256
        or candidate_context["prepared_context_sha256"]
        != PREPARED_CANDIDATE_CONTEXT_SHA256
        or asset["template_decoded"]["array_sha256"] != DECODED_TEMPLATE_ARRAY_SHA256
        or asset["scalar_conversion"]["array_sha256"]
        != SCALAR_CONVERSION_ARRAY_SHA256
        or asset["annotation_decoded"]["array_sha256"]
        != DECODED_ANNOTATION_ARRAY_SHA256
    ):
        raise ValueError("Allen decoded arrays or prepared contexts changed from the preflight")


def validate_resolved_config(
    record: dict[str, object], expected: dict[str, object] | None = None
) -> dict[str, object]:
    validate_payload_keys(record, "resolved_config")
    if record["schema"] != RUNNER_SCHEMA:
        raise ValueError("resolved config schema changed")
    if record["resolved_config_sha256"] != canonical_payload_sha256(
        record, "resolved_config_sha256"
    ):
        raise ValueError("resolved config self-hash changed")
    validate_payload_keys(record["repository"], "repository")
    validate_payload_keys(record["frozen_semantic_input"], "frozen_semantic_input")
    validate_payload_keys(record["atlas_assets"], "atlas_assets")
    validate_payload_keys(record["descriptor_constants"], "descriptor_constants")
    validate_payload_keys(record["descriptor_constants"]["mind"], "mind_constants")
    validate_payload_keys(record["descriptor_constants"]["hog"], "hog_constants")
    validate_payload_keys(
        record["descriptor_constants"]["normalized_gradient_like"], "ngf_constants"
    )
    validate_payload_keys(
        record["descriptor_constants"]["common_lattice"],
        "common_lattice_constants",
    )
    validate_payload_keys(record["case_and_shuffle_contract"], "case_and_shuffle_contract")
    validate_payload_keys(
        record["case_and_shuffle_contract"]["pooled_strata_counts"],
        "pooled_strata_counts",
    )
    membership = validate_payload_keys(
        record["case_and_shuffle_contract"]["pooled_strata_membership"],
        "pooled_strata_membership",
    )
    if membership != {
        name: list(indices) for name, indices in FROZEN_POOLED_MEMBERSHIP.items()
    }:
        raise ValueError("resolved config pooled membership changed exact base indices")
    for transform in record["case_and_shuffle_contract"]["affine_transforms"]:
        validate_payload_keys(transform, "config_affine_transform")
    validate_payload_keys(record["environment"], "environment")
    validate_payload_keys(record["model_independence"], "model_independence")
    validate_payload_keys(record["data_access"], "data_access")
    sources = record["source_sha256"]
    if (
        not isinstance(sources, list)
        or len(sources) != len(SOURCE_RELATIVE_PATHS)
        or [item.get("relative_path") for item in sources] != list(SOURCE_RELATIVE_PATHS)
    ):
        raise ValueError("resolved config source list changed length or order")
    for item in sources:
        validate_payload_keys(item, "source_receipt")
        for field in ("git_blob_sha256", "checkout_sha256"):
            if not isinstance(item[field], str) or re.fullmatch(r"[0-9a-f]{64}", item[field]) is None:
                raise ValueError("source receipt contains a malformed SHA-256")
    if (
        record["repository"]["branch"] != EXPECTED_BRANCH
        or record["repository"]["source_parent_commit"] != SOURCE_PARENT_COMMIT
        or record["repository"]["execution_commit"]
        != record["repository"]["origin_commit"]
        or record["repository"]["worktree_clean"] is not True
        or record["frozen_semantic_input"]["source_commit"]
        != FROZEN_SEMANTIC_SOURCE_COMMIT
        or record["frozen_semantic_input"]["inventory_file_count"]
        != FROZEN_SEMANTIC_FILE_COUNT
        or record["case_and_shuffle_contract"]["base_count"] != CASE_COUNT
        or record["case_and_shuffle_contract"]["candidate_count"] != CANDIDATE_COUNT
        or record["case_and_shuffle_contract"]["outline_order"] != list(OUTLINE_MODES)
        or record["case_and_shuffle_contract"]["candidate_permutation"]
        != list(PERMUTATION)
        or record["case_and_shuffle_contract"]["chunk_sizes"]
        != list(CHUNK_CONTROL_SIZES)
        or record["model_independence"] != MODEL_INDEPENDENCE
        or record["data_access"] != DATA_ACCESS
    ):
        raise ValueError("resolved config changed a frozen scientific or authority constant")
    if expected is not None and record != expected:
        raise ValueError("saved resolved config differs from independently rebuilt authority")
    return record


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} is not one lowercase SHA-256")
    return value


def _reject_nonfinite(value: object, name: str = "payload") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a nonfinite float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{name}/{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{name}/{key}")


def _validate_self_hash(
    record: dict[str, object], schema_name: str, hash_field: str = "payload_sha256"
) -> None:
    validate_payload_keys(record, schema_name)
    _require_sha256(record[hash_field], f"{schema_name} {hash_field}")
    if record[hash_field] != canonical_payload_sha256(record, hash_field):
        raise ValueError(f"{schema_name} self-hash changed")
    _reject_nonfinite(record, schema_name)


def _validate_mask_receipt(
    receipt: dict[str, object], case_index: int, name: str
) -> None:
    validate_payload_keys(receipt, "mask_receipt")
    if (
        receipt["dtype"] != "|b1"
        or receipt["shape"] != list(OUTPUT_SHAPE)
        or receipt["bitorder"] != "little"
        or type(receipt["bit_count"]) is not int
        or receipt["bit_count"] != math.prod(OUTPUT_SHAPE)
        or type(receipt["byte_count"]) is not int
        or receipt["byte_count"] != math.prod(OUTPUT_SHAPE) // 8
        or receipt["relative_path"]
        != f"masks/case-{case_index:03d}-{name.replace('_', '-')}.bin"
    ):
        raise ValueError("packed mask receipt changed shape, encoding, count, or path")
    _require_sha256(receipt["payload_sha256"], "packed mask payload")
    _require_sha256(receipt["array_sha256"], "Boolean mask array")


def _validate_slot_record(
    slot: dict[str, object],
    *,
    indices: tuple[int | None, int | None, int | None],
    outline_mode: str,
    descriptor: str,
    domain: str,
    domain_record: dict[str, object],
    ordered_candidate_ids: list[str],
    truth_candidate_id: str,
) -> None:
    _validate_self_hash(slot, "slot")
    if (
        (slot["case_index"], slot["bank_case_index"], slot["target_case_index"])
        != indices
        or slot["outline_mode"] != outline_mode
        or slot["descriptor"] != descriptor
        or slot["domain"] != domain
        or slot["domain_mask_receipt_sha256"]
        != _domain_mask_receipt_sha256(domain_record)
        or type(slot["domain_pixel_count"]) is not int
        or slot["domain_pixel_count"] != domain_record["pixel_count"]
        or type(slot["entered_gate"]) is not bool
        or slot["entered_gate"]
        != _entered_gate(descriptor, domain, outline_mode, shuffled=indices[0] is None)
    ):
        raise ValueError("score slot identity/domain/gate binding changed")
    _validate_slot_indices(*indices)
    if slot["status"] == "insufficient_domain":
        if (
            slot["reason_code"] not in {"NO_ELIGIBLE_PIXELS", "NO_ELIGIBLE_BLOCKS"}
            or any(slot[key] is not None for key in ("scores", "ranking", "metrics"))
            or slot["entered_gate"] is not False
            or not (slot["eligible_pixel_count"] == 0 or slot["eligible_block_count"] == 0)
        ):
            raise ValueError("insufficient-domain slot changed its exact null contract")
        return
    if slot["status"] != "ok" or slot["reason_code"] is not None:
        raise ValueError("score slot status is neither exact ok nor insufficient")
    scores = np.asarray(slot["scores"], dtype=np.float64)
    if (
        not isinstance(slot["scores"], list)
        or scores.shape != (CANDIDATE_COUNT,)
        or not np.isfinite(scores).all()
        or np.any((scores < 0.0) | (scores > 1.0))
        or not all(type(value) is float for value in slot["scores"])
    ):
        raise ValueError("status-ok score vector is not exactly forty finite JSON floats in [0,1]")
    validate_payload_keys(slot["ranking"], "ranking")
    if slot["ranking"] != rank_candidate_scores(
        scores, ordered_candidate_ids, truth_candidate_id
    ):
        raise ValueError("saved ranking does not replay from raw scores")
    metrics = slot["metrics"]
    if descriptor in {
        "MIND",
        "constant-within-support-MIND-null",
        "support-penalized-MIND",
    }:
        validate_payload_keys(metrics, "mind_metrics")
        if (
            type(metrics["target_vbar"]) is not float
            or not isinstance(metrics["candidate_vbar"], list)
            or len(metrics["candidate_vbar"]) != CANDIDATE_COUNT
            or (
                descriptor == "constant-within-support-MIND-null"
                and (
                    not isinstance(metrics["supported_means"], list)
                    or len(metrics["supported_means"]) != CANDIDATE_COUNT
                )
            )
            or (
                descriptor != "constant-within-support-MIND-null"
                and metrics["supported_means"] is not None
            )
            or (
                descriptor == "support-penalized-MIND"
                and (
                    not isinstance(metrics["candidate_exterior_fractions"], list)
                    or len(metrics["candidate_exterior_fractions"]) != CANDIDATE_COUNT
                )
            )
            or (
                descriptor != "support-penalized-MIND"
                and metrics["candidate_exterior_fractions"] is not None
            )
        ):
            raise ValueError("MIND metric normalization arrays changed nullability or length")
        should_have_pose = (
            descriptor == "MIND"
            and indices[0] is not None
            and slot["ranking"]["selected_candidate_id"] is not None
        )
        if should_have_pose:
            validate_payload_keys(metrics["selected_pose_errors"], "pose_errors")
        elif metrics["selected_pose_errors"] is not None:
            raise ValueError("pose errors are permitted only for unique native descriptor MIND")
    elif descriptor == "HOG":
        validate_payload_keys(metrics, "hog_metrics")
        validate_payload_keys(metrics["block_weights_receipt"], "array_receipt")
    elif descriptor == "normalized-gradient-like":
        validate_payload_keys(metrics, "ngf_metrics")
        if not isinstance(metrics["candidate_eta"], list) or len(metrics["candidate_eta"]) != CANDIDATE_COUNT:
            raise ValueError("NGF candidate eta vector must have length forty")
    else:
        raise ValueError("status-ok slot uses an undeclared descriptor")


def _validate_array_receipt(record: dict[str, object]) -> None:
    validate_payload_keys(record, "array_receipt")
    if (
        not isinstance(record["dtype"], str)
        or not isinstance(record["shape"], list)
        or not all(type(value) is int and value >= 0 for value in record["shape"])
    ):
        raise ValueError("dense array receipt dtype/shape changed type")
    _require_sha256(record["array_sha256"], "dense array")


def _validate_primary_record(record: dict[str, object], case_index: int) -> None:
    _validate_self_hash(record, "primary")
    if (
        record["schema"] != "anatomy-tracker.arbitrary-plane-image-information-primary/v1"
        or type(record["case_index"]) is not int
        or record["case_index"] != case_index
    ):
        raise ValueError("primary case schema/index changed")
    _require_sha256(record["semantic_case_payload_sha256"], "semantic case payload")
    validate_payload_keys(record["provenance"], "primary_provenance")
    provenance = record["provenance"]
    if any(provenance[key] is not None for key in ("animal_id", "specimen_id", "experiment_id")):
        raise ValueError("synthetic pilot subject identifiers must be explicit nulls")
    validate_payload_keys(provenance["reporting_strata"], "reporting_strata")
    mask_only = provenance["mask_only_Dice"]
    _validate_self_hash(mask_only, "mask_only_dice")
    if (
        mask_only["entered_gate"] is not False
        or not isinstance(mask_only["values"], list)
        or len(mask_only["values"]) != CANDIDATE_COUNT
    ):
        raise ValueError("copied mask-only Dice changed length/gate role")
    validate_payload_keys(mask_only["recomputed_ranking"], "ranking")
    validate_payload_keys(record["frozen_replay"], "frozen_replay")
    frozen = record["frozen_replay"]
    if (
        frozen["semantic_case_payload_sha256"] != record["semantic_case_payload_sha256"]
        or frozen["replay_passed"] is not True
        or frozen["case_rejection_records_sha256"]
        != canonical_payload_sha256(frozen["case_rejection_records"])
        or frozen["outline_descendant_receipts_sha256"]
        != canonical_payload_sha256(frozen["outline_descendant_receipts"])
        or len(frozen["outline_descendant_receipts"]) != 3
    ):
        raise ValueError("primary frozen replay receipts do not self-bind")
    target = validate_payload_keys(record["target"], "target")
    if (
        target["output_shape_h_w"] != list(OUTPUT_SHAPE)
        or type(target["pixel_pitch_um"]) is not float
        or not math.isfinite(target["pixel_pitch_um"])
        or target["pixel_pitch_um"] <= 0.0
    ):
        raise ValueError("primary target shape/pitch changed")
    _validate_array_receipt(target["target_labels_receipt"])
    _validate_array_receipt(target["fixed_valid_mask_receipt"])
    bank = validate_payload_keys(record["candidate_bank"], "candidate_bank")
    candidate_ids = bank["ordered_candidate_ids"]
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) != CANDIDATE_COUNT
        or len(set(candidate_ids)) != CANDIDATE_COUNT
        or bank["ordered_candidate_ids_sha256"] != canonical_payload_sha256(candidate_ids)
        or type(bank["truth_candidate_index"]) is not int
        or candidate_ids[bank["truth_candidate_index"]] != bank["truth_candidate_id"]
    ):
        raise ValueError("primary candidate-bank order/truth binding changed")
    scalar_records = record["candidate_scalar_receipts"]
    if not isinstance(scalar_records, list) or len(scalar_records) != CANDIDATE_COUNT:
        raise ValueError("primary must save forty scalar candidate receipts")
    for index, item in enumerate(scalar_records):
        _validate_self_hash(item, "candidate_scalar_record")
        validate_payload_keys(item["scalar"], "candidate_scalar_value")
        if (
            item["candidate_index"] != index
            or item["candidate_id"] != candidate_ids[index]
            or item["crosscheck_passed"] is not True
        ):
            raise ValueError("candidate scalar receipt changed order/crosscheck")
        for receipt in (
            item["scalar"]["rendered_float32"],
            item["scalar"]["scaled_float64"],
            item["annotation"],
            item["brain_mask"],
        ):
            _validate_array_receipt(receipt)
    domains = record["score_domains"]
    if not isinstance(domains, list) or [item.get("domain") for item in domains] != list(MASK_NAMES):
        raise ValueError("primary score domains changed exact five-domain order")
    domain_by_name = {}
    for item in domains:
        validate_payload_keys(item, "score_domain")
        name = item["domain"]
        _validate_mask_receipt(item["mask_receipt"], case_index, name)
        if type(item["pixel_count"]) is not int or item["pixel_count"] < 0:
            raise ValueError("domain pixel count is not a nonnegative integer")
        if item["minimum_required_pixels"] != PRIMARY_MINIMUM_PIXELS.get(name):
            raise ValueError("domain minimum changed")
        domain_by_name[name] = item
    outlines = record["outline_results"]
    if not isinstance(outlines, list) or [item.get("outline_mode") for item in outlines] != list(OUTLINE_MODES):
        raise ValueError("primary outlines changed exact order")
    for outline in outlines:
        _validate_self_hash(outline, "primary_outline")
        validate_payload_keys(outline["target_dewarp"], "target_dewarp")
        for receipt in (
            outline["target_dewarp"]["model_input_image_receipt"],
            outline["target_dewarp"]["fixed_to_source_map_receipt"],
            outline["target_dewarp"]["dewarped_float32_receipt"],
            outline["target_dewarp"]["dewarped_float64_receipt"],
        ):
            _validate_array_receipt(receipt)
        slots = outline["score_slots"]
        if (
            not isinstance(slots, list)
            or [(item.get("descriptor"), item.get("domain")) for item in slots]
            != list(NATIVE_OUTLINE_SLOT_SCHEDULE)
        ):
            raise ValueError("primary outline changed exact 12-slot order")
        for slot, (descriptor, domain) in zip(slots, NATIVE_OUTLINE_SLOT_SCHEDULE, strict=True):
            _validate_slot_record(
                slot,
                indices=(case_index, None, None),
                outline_mode=outline["outline_mode"],
                descriptor=descriptor,
                domain=domain,
                domain_record=domain_by_name[domain],
                ordered_candidate_ids=candidate_ids,
                truth_candidate_id=bank["truth_candidate_id"],
            )


def _validate_shuffled_record(record: dict[str, object], bank_case_index: int) -> None:
    _validate_self_hash(record, "shuffled")
    target_case_index = (bank_case_index + 17) % CASE_COUNT
    if (
        record["schema"] != "anatomy-tracker.arbitrary-plane-image-information-shuffled/v1"
        or record["bank_case_index"] != bank_case_index
        or record["target_case_index"] != target_case_index
    ):
        raise ValueError("shuffled schema/index/+17 binding changed")
    bank = validate_payload_keys(record["bank_identity"], "bank_identity")
    target = validate_payload_keys(record["target_identity"], "target_identity")
    candidate_ids = bank["ordered_candidate_ids"]
    if (
        bank["bank_case_index"] != bank_case_index
        or len(candidate_ids) != CANDIDATE_COUNT
        or len(set(candidate_ids)) != CANDIDATE_COUNT
        or bank["ordered_candidate_ids_sha256"] != canonical_payload_sha256(candidate_ids)
        or candidate_ids.count(bank["truth_candidate_id"]) != 1
        or target["target_case_index"] != target_case_index
        or not isinstance(target["target_outline_payload_sha256"], list)
        or len(target["target_outline_payload_sha256"]) != 3
    ):
        raise ValueError("shuffled bank/target identity changed")
    common = record["common_lattice_resampling"]
    _validate_self_hash(common, "common_lattice")
    _validate_array_receipt(common["coordinate_map_receipt"])
    resampled = common["resampled_candidates"]
    if not isinstance(resampled, list) or len(resampled) != CANDIDATE_COUNT:
        raise ValueError("shuffled common lattice lacks forty candidates")
    for index, item in enumerate(resampled):
        _validate_self_hash(item, "resampled_candidate")
        if item["candidate_index"] != index or item["candidate_id"] != candidate_ids[index]:
            raise ValueError("shuffled resampled candidate order changed")
        for name in (
            "scalar_float64_receipt",
            "support_bool_receipt",
            "constant_null_scalar_float64_receipt",
        ):
            _validate_array_receipt(item[name])
    domains = record["score_domains"]
    if not isinstance(domains, list) or [item.get("domain") for item in domains] != list(MASK_NAMES):
        raise ValueError("shuffled target domains changed order")
    domain_by_name = {}
    for item in domains:
        validate_payload_keys(item, "score_domain")
        _validate_mask_receipt(item["mask_receipt"], target_case_index, item["domain"])
        domain_by_name[item["domain"]] = item
    outlines = record["outline_results"]
    if not isinstance(outlines, list) or [item.get("outline_mode") for item in outlines] != list(OUTLINE_MODES):
        raise ValueError("shuffled outlines changed order")
    for outline in outlines:
        _validate_self_hash(outline, "shuffled_outline")
        slots = outline["score_slots"]
        if [(item.get("descriptor"), item.get("domain")) for item in slots] != list(
            SHUFFLED_OUTLINE_SLOT_SCHEDULE
        ):
            raise ValueError("shuffled outline changed exact four-slot order")
        for slot, (descriptor, domain) in zip(slots, SHUFFLED_OUTLINE_SLOT_SCHEDULE, strict=True):
            _validate_slot_record(
                slot,
                indices=(None, bank_case_index, target_case_index),
                outline_mode=outline["outline_mode"],
                descriptor=descriptor,
                domain=domain,
                domain_record=domain_by_name[domain],
                ordered_candidate_ids=candidate_ids,
                truth_candidate_id=bank["truth_candidate_id"],
            )


def _validate_basic_control_state(
    record: dict[str, object],
    name: str,
    *,
    allow_not_applicable: bool = False,
) -> None:
    validate_payload_keys(record, "basic_control")
    _require_sha256(record["evidence_sha256"], f"{name} evidence")
    passed = record["passed"]
    if passed is True:
        expected = ("passed", None)
    elif passed is False:
        expected = ("failed", "EXACT_CONTROL_MISMATCH")
    elif passed is None and allow_not_applicable:
        expected = (
            "authenticated_not_applicable",
            "SOURCE_LANDSCAPE_INSUFFICIENT_DOMAIN",
        )
    else:
        raise ValueError(f"{name} has an invalid control pass state")
    if (record["status"], record["reason_code"]) != expected:
        raise ValueError(f"{name} status/reason does not match its pass state")


def _validate_permutation_control(record: dict[str, object]) -> None:
    validate_payload_keys(record, "permutation_control")
    if (
        record["mapping"]
        != "new[k]=old[(7*k+3)%40]; inverse_new_index=23*(old_index-3)%40"
        or record["permutation"] != list(PERMUTATION)
        or record["nonidentity_bijection"] is not True
        or type(record["passed"]) is not bool
    ):
        raise ValueError("permutation control changed its exact schedule or pass type")
    for field in (
        "original_score_vector_sha256",
        "permuted_score_vector_sha256",
        "inverse_reindexed_score_vector_sha256",
        "original_ranking_sha256",
        "recomputed_ranking_sha256",
    ):
        _require_sha256(record[field], f"permutation {field}")


def _validate_chunk_control(record: dict[str, object]) -> None:
    validate_payload_keys(record, "chunk_control")
    hashes = record["score_vector_sha256"]
    if (
        record["chunk_sizes"] != list(CHUNK_CONTROL_SIZES)
        or not isinstance(hashes, dict)
        or set(hashes) != {str(item) for item in CHUNK_CONTROL_SIZES}
        or type(record["byte_identical"]) is not bool
        or type(record["passed"]) is not bool
        or record["passed"] is not record["byte_identical"]
    ):
        raise ValueError("chunk control changed its exact schedule or pass relation")
    for size, digest in hashes.items():
        _require_sha256(digest, f"chunk {size} score vector")
    _require_sha256(record["ranking_payload_sha256"], "chunk ranking")


def _validate_case_control(
    record: dict[str, object],
    case_index: int,
    primary: dict[str, object],
    shuffled: dict[str, object],
) -> None:
    _validate_self_hash(record, "control")
    if (
        record["schema"] != "anatomy-tracker.arbitrary-plane-image-information-controls/v1"
        or record["case_index"] != case_index
        or record["evidence_receipt_sha256"] != canonical_payload_sha256(record["checks"])
    ):
        raise ValueError("case-control schema/index/evidence hash changed")
    checks = validate_payload_keys(record["checks"], "case_checks")
    basic_names = (
        "source_replay_metadata_geometry",
        "candidate_scalar_annotation_mask",
        "dewarp_direction_and_masks",
        "rp2_and_xy_wh",
        "scorer_signature_exclusion",
        "target_domain_invariance",
        "accurate_absent_core_identity",
        "shuffled_binding",
        "mask_only_verification",
    )
    for name in basic_names:
        _validate_basic_control_state(checks[name], name)
    expected_slots = [
        item
        for outline in primary["outline_results"]
        for item in outline["score_slots"]
    ] + [
        item
        for outline in shuffled["outline_results"]
        for item in outline["score_slots"]
    ]
    landscape = checks["landscape_controls"]
    if len(landscape) != 48 or len(expected_slots) != 48:
        raise ValueError("case controls must contain exactly 48 landscape controls")
    for item, slot in zip(landscape, expected_slots, strict=True):
        _validate_self_hash(item, "landscape_control")
        if (
            item["source_slot_payload_sha256"] != slot["payload_sha256"]
            or item["source_status"] != slot["status"]
            or any(item[key] != slot[key] for key in (
                "case_index", "bank_case_index", "target_case_index",
                "outline_mode", "descriptor", "domain",
            ))
        ):
            raise ValueError("landscape control changed source-slot identity/order")
        if slot["status"] == "ok":
            _validate_permutation_control(item["permutation"])
            _validate_chunk_control(item["chunks"])
            expected_passed = bool(
                item["permutation"]["passed"] is True
                and item["chunks"]["passed"] is True
            )
            if (
                type(item["passed"]) is not bool
                or item["passed"] is not expected_passed
                or item["status"] != ("passed" if expected_passed else "failed")
                or item["reason_code"] is not None
            ):
                raise ValueError("landscape status does not match permutation/chunk evidence")
        else:
            for key in ("permutation", "chunks"):
                _validate_basic_control_state(
                    item[key], f"landscape {key}", allow_not_applicable=True
                )
            if (
                item["status"] != "authenticated_not_applicable"
                or item["reason_code"] != "SOURCE_LANDSCAPE_INSUFFICIENT_DOMAIN"
                or item["passed"] is not None
            ):
                raise ValueError("insufficient landscape control changed N/A contract")
    affine = checks["affine_and_polarity"]
    if len(affine) != (36 if case_index in AFFINE_CASES else 0):
        raise ValueError("case affine-control count changed")
    expected_affine_slots = expected_slots[:36] if case_index in AFFINE_CASES else []
    for item, slot in zip(affine, expected_affine_slots, strict=True):
        _validate_self_hash(item, "affine_slot")
        if item["source_slot_payload_sha256"] != slot["payload_sha256"]:
            raise ValueError("affine control changed its source slot")
        if slot["status"] == "ok":
            if len(item["transforms"]) != len(AFFINE_TRANSFORMS):
                raise ValueError("applicable affine control changed transform count")
            for transform, expected in zip(item["transforms"], AFFINE_TRANSFORMS, strict=True):
                validate_payload_keys(transform, "affine_transform")
                if (
                    (transform["name"], transform["scale"], transform["offset"])
                    != expected
                    or transform["scalar_padding"] != expected[2]
                    or type(transform["passed"]) is not bool
                ):
                    raise ValueError("affine transform identity/pass type changed")
                _require_sha256(
                    transform["ranking_payload_sha256"], "affine ranking payload"
                )
            expected_passed = all(
                transform["passed"] is True for transform in item["transforms"]
            )
            if (
                type(item["passed"]) is not bool
                or item["passed"] is not expected_passed
                or item["status"] != ("passed" if expected_passed else "failed")
                or item["reason_code"] is not None
            ):
                raise ValueError("affine slot status does not match transform evidence")
        elif (
            item["status"] != "authenticated_not_applicable"
            or item["reason_code"] != "SOURCE_LANDSCAPE_INSUFFICIENT_DOMAIN"
            or item["passed"] is not None
            or item["transforms"] != []
        ):
            raise ValueError("insufficient affine control changed N/A contract")


def _validate_global_controls(
    record: dict[str, object], controls: list[dict[str, object]]
) -> None:
    _validate_self_hash(record, "global_controls")
    evidence = {
        "frozen_inventory_audit": record["frozen_inventory_audit"],
        "source_and_signature_audit": record["source_and_signature_audit"],
        "affine_and_polarity_controls": record["affine_and_polarity_controls"],
    }
    if record["evidence_receipt_sha256"] != canonical_payload_sha256(evidence):
        raise ValueError("global-control evidence hash changed")
    validate_payload_keys(record["frozen_inventory_audit"], "frozen_inventory_audit")
    source = validate_payload_keys(
        record["source_and_signature_audit"], "source_signature_audit"
    )
    affine = validate_payload_keys(
        record["affine_and_polarity_controls"], "affine_summary"
    )
    inventory = record["frozen_inventory_audit"]
    expected_inventory_passed = bool(
        inventory["observed_file_count"] == inventory["expected_file_count"]
        and inventory["observed_total_bytes"] == inventory["expected_total_bytes"]
        and inventory["observed_inventory_sha256"]
        == inventory["expected_inventory_sha256"]
    )
    if (
        type(inventory["passed"]) is not bool
        or inventory["passed"] is not expected_inventory_passed
        or inventory["expected_file_count"] != FROZEN_SEMANTIC_FILE_COUNT
        or inventory["expected_total_bytes"] != FROZEN_SEMANTIC_TOTAL_BYTES
        or inventory["expected_inventory_sha256"]
        != FROZEN_SEMANTIC_INVENTORY_SHA256
    ):
        raise ValueError("frozen inventory pass flag does not match its saved evidence")
    _require_sha256(inventory["observed_inventory_sha256"], "observed inventory")
    for item in source["source_records"]:
        validate_payload_keys(item, "source_receipt")
        _require_sha256(item["git_blob_sha256"], "source Git blob")
        _require_sha256(item["checkout_sha256"], "source checkout")
    signature_passed = True
    for item in source["scorer_signature_records"]:
        validate_payload_keys(item, "scorer_signature_record")
        _require_sha256(item["source_sha256"], "scorer signature source")
        expected = isinstance(item["forbidden_matches"], list) and not item[
            "forbidden_matches"
        ]
        if type(item["passed"]) is not bool or item["passed"] is not expected:
            raise ValueError("scorer signature pass flag differs from its exclusions")
        signature_passed &= expected
    model_passed = True
    for item in source["model_dependency_records"]:
        validate_payload_keys(item, "model_dependency_record")
        expected = isinstance(item["declared_values"], list) and not item[
            "declared_values"
        ]
        if type(item["passed"]) is not bool or item["passed"] is not expected:
            raise ValueError("model-dependency pass flag differs from its evidence")
        model_passed &= expected
    expected_source_passed = bool(
        source["execution_commit"] == source["origin_commit"]
        and source["worktree_clean"] is True
        and len(source["source_records"]) == len(SOURCE_RELATIVE_PATHS)
        and signature_passed
        and model_passed
    )
    if type(source["passed"]) is not bool or source["passed"] is not expected_source_passed:
        raise ValueError("source/signature pass flag differs from its saved evidence")
    affine_items = [
        item
        for control in controls
        for item in control["checks"]["affine_and_polarity"]
    ]
    expected_affine_passed = bool(all(_case_control_passed(item) for item in controls))
    if (
        affine["required_case_indices"] != list(AFFINE_CASES)
        or affine["case_control_payload_sha256"]
        != [item["payload_sha256"] for item in controls]
        or affine["applicable_slot_count"]
        != sum(item["status"] != "authenticated_not_applicable" for item in affine_items)
        or affine["authenticated_not_applicable_count"]
        != sum(item["status"] == "authenticated_not_applicable" for item in affine_items)
        or type(affine["passed"]) is not bool
        or affine["passed"] is not expected_affine_passed
    ):
        raise ValueError("global case-control summary differs from its saved sidecars")


EXPECTED_ATOMIC_GATE_IDS = (
    "frozen-inventory-integrity", "source-signature-integrity",
    "all-case-controls-integrity", "absent-context-top1-count",
    "absent-context-wilson-lower", "absent-context-mean-rr",
    "absent-context-median-rank", "absent-context-median-win-fraction",
    "orientation-near_AP-top1-count", "orientation-near_DV-top1-count",
    "orientation-near_ML-top1-count", "orientation-general_oblique-top1-count",
    "absent-core-top1-count", "absent-core-mean-rr",
    "absent-core-minus-null-success-count", "challenging_appearance-top1-count",
    "challenging_appearance-mean-rr", "challenging_appearance-minus-null-success-count",
    "damaged-top1-count", "damaged-mean-rr", "damaged-minus-null-success-count",
    f"brush-{ACCURATE_OUTLINE}-top1-count", f"brush-{ACCURATE_OUTLINE}-deficit",
    f"brush-{IMPERFECT_OUTLINE}-top1-count", f"brush-{IMPERFECT_OUTLINE}-deficit",
    f"shuffled-{ACCURATE_OUTLINE}-top1-count", f"shuffled-{ACCURATE_OUTLINE}-mean-rr",
    f"shuffled-{IMPERFECT_OUTLINE}-top1-count", f"shuffled-{IMPERFECT_OUTLINE}-mean-rr",
    f"shuffled-{ABSENT_OUTLINE}-top1-count", f"shuffled-{ABSENT_OUTLINE}-mean-rr",
)


def _validate_metrics_contract(
    metrics: dict[str, object],
    pooled_membership: dict[str, list[int]] | None = None,
) -> None:
    validate_payload_keys(metrics, "result_metrics")
    native = metrics["native_slot_summaries"]
    shuffled = metrics["shuffled_slot_summaries"]
    if (
        len(native) != 37
        or [
            (item["descriptor"], item["domain"], item["outline_mode"])
            for item in native[:36]
        ]
        != [(descriptor, domain, outline) for descriptor, domain, outline in NATIVE_SLOT_SCHEDULE]
        or (
            native[36]["descriptor"], native[36]["domain"], native[36]["outline_mode"]
        )
        != ("mask_only_Dice", "frozen_semantic_fixed_valid_mask", None)
        or len(shuffled) != 12
        or [
            (item["descriptor"], item["domain"], item["outline_mode"])
            for item in shuffled
        ]
        != [(descriptor, domain, outline) for descriptor, domain, outline in SHUFFLED_SLOT_SCHEDULE]
    ):
        raise ValueError("aggregate slot summaries changed exact 37/12 order")
    for item in (*native, *shuffled):
        validate_payload_keys(item, "slot_summary")
    reporting = metrics["reporting_stratum_summaries"]
    expected_orientation = (
        ("near_AP", list(range(0, 12))),
        ("near_DV", list(range(12, 24))),
        ("near_ML", list(range(24, 36))),
        ("general_oblique", list(range(36, 64))),
    )
    if len(reporting) < 4:
        raise ValueError("aggregate metrics lack four orientation summaries")
    for item, (name, indices) in zip(reporting[:4], expected_orientation, strict=True):
        validate_payload_keys(item, "stratum_summary")
        if (
            item["stratum_type"] != "orientation_family"
            or item["stratum_value"] != name
            or item["base_indices"] != indices
            or item["base_count"] != len(indices)
        ):
            raise ValueError("orientation summaries changed exact 12/12/12/28 membership")
    if any(
        item["stratum_type"] == "orientation_family" for item in reporting[4:]
    ):
        raise ValueError("orientation summaries contain a duplicate family")
    for item in reporting[4:]:
        validate_payload_keys(item, "stratum_summary")
    pooled = metrics["pooled_safeguard_summaries"]
    if (
        len(pooled) != 4
        or [
            (item["stratum_type"], item["stratum_value"], item["base_count"])
            for item in pooled
        ]
        != [
            ("challenging_appearance", "member", 22),
            ("challenging_appearance", "member", 22),
            ("damaged", "member", 27),
            ("damaged", "member", 27),
        ]
    ):
        raise ValueError("pooled safeguards changed exact order/count")
    for item in pooled:
        validate_payload_keys(item, "stratum_summary")
    if (
        pooled[0]["base_indices"] != pooled[1]["base_indices"]
        or pooled[2]["base_indices"] != pooled[3]["base_indices"]
    ):
        raise ValueError("paired pooled safeguards changed base membership")
    if pooled_membership is not None:
        validate_payload_keys(pooled_membership, "pooled_strata_membership")
        if (
            pooled[0]["base_indices"] != pooled_membership["challenging_appearance"]
            or pooled[2]["base_indices"] != pooled_membership["damaged"]
        ):
            raise ValueError("pooled summaries differ from pre-score frozen membership")
    if len(metrics["paired_outline_comparisons"]) != 2:
        raise ValueError("paired outline summaries must contain exactly two comparisons")
    for item in metrics["paired_outline_comparisons"]:
        validate_payload_keys(item, "paired_summary")
    _reject_nonfinite(metrics, "metrics")


def _resolve_pointer(
    pointer: str,
    metrics: dict[str, object],
    global_controls: dict[str, object],
) -> object:
    if pointer.startswith("result.json#/metrics"):
        value: object = metrics
        suffix = pointer[len("result.json#/metrics") :]
    elif pointer.startswith("global_controls.json#"):
        value = global_controls
        suffix = pointer[len("global_controls.json#") :]
    else:
        raise ValueError("atomic gate source is not a resolvable saved-document JSON pointer")
    for token in (item for item in suffix.split("/") if item):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _validate_gates(
    gates: dict[str, object],
    metrics: dict[str, object],
    global_controls: dict[str, object],
) -> None:
    validate_payload_keys(gates, "gates")
    if (
        gates["global_controls_payload_sha256"] != global_controls["payload_sha256"]
        or [item.get("gate_id") for item in gates["atomic_checks"]]
        != list(EXPECTED_ATOMIC_GATE_IDS)
    ):
        raise ValueError("gate list/hash changed exact predeclared order")
    for item in gates["atomic_checks"]:
        validate_payload_keys(item, "atomic_gate")
        _resolve_pointer(item["source_metric_pointer"], metrics, global_controls)
        evidence = {key: value for key, value in item.items() if key != "evidence_sha256"}
        if item["evidence_sha256"] != canonical_payload_sha256(evidence):
            raise ValueError("atomic gate evidence hash changed")
    if (
        gates["passed"] is not all(
            item["passed"] is True for item in gates["atomic_checks"]
        )
        or gates["decision"] != ("PASS" if gates["passed"] else "FAIL")
    ):
        raise ValueError("aggregate gate decision does not match atomic checks")
    if gates != evaluate_gates(metrics, global_controls):
        raise ValueError("saved atomic gates do not exactly replay from their sources")


def _load_case_records(output: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    primary = [
        _read_strict_json(output / "primary" / f"case-{index:03d}.json")
        for index in range(CASE_COUNT)
    ]
    shuffled = [
        _read_strict_json(output / "shuffled" / f"case-{index:03d}.json")
        for index in range(CASE_COUNT)
    ]
    controls = [
        _read_strict_json(output / "controls" / f"case-{index:03d}.json")
        for index in range(CASE_COUNT)
    ]
    return primary, shuffled, controls


def _assemble_result(
    config: dict[str, object],
    inventory: list[dict[str, object]],
    primary: list[dict[str, object]],
    shuffled: list[dict[str, object]],
    controls: list[dict[str, object]],
    global_controls: dict[str, object],
) -> dict[str, object]:
    metrics = aggregate_metrics(
        primary,
        shuffled,
        config["case_and_shuffle_contract"]["pooled_strata_membership"],
    )
    gates = evaluate_gates(metrics, global_controls)
    record = {
        "schema": RUNNER_SCHEMA,
        "interpretation": INTERPRETATION,
        "resolved_config_sha256": config["resolved_config_sha256"],
        "pre_result_inventory": copy.deepcopy(inventory),
        "pre_result_inventory_sha256": canonical_payload_sha256(inventory),
        "primary_case_payload_sha256": [item["payload_sha256"] for item in primary],
        "shuffled_case_payload_sha256": [item["payload_sha256"] for item in shuffled],
        "control_payload_sha256": [item["payload_sha256"] for item in controls],
        "metrics": metrics,
        "gates": gates,
        "data_access": copy.deepcopy(DATA_ACCESS),
        "model_independence": copy.deepcopy(MODEL_INDEPENDENCE),
    }
    _self_hash(record, "result_payload_sha256")
    return validate_payload_keys(record, "result")


def _validate_prelaunch_failure_record(
    record: dict[str, object],
    expected_config: dict[str, object],
    expected_masks: list[dict[str, object]],
) -> None:
    _validate_self_hash(
        record, "prelaunch_failure", "failure_payload_sha256"
    )
    if (
        record["schema"]
        != "anatomy-tracker.arbitrary-plane-image-information-prelaunch-failure/v1"
        or record["status"] != "failed_before_scoring"
        or record["failure_code"] != "INSUFFICIENT_PRIMARY_DOMAIN"
        or record["data_access"] != DATA_ACCESS
        or record["model_independence"] != MODEL_INDEPENDENCE
    ):
        raise ValueError("prelaunch failure schema/status/scope changed")
    validate_payload_keys(record["data_access"], "data_access")
    validate_payload_keys(record["model_independence"], "model_independence")
    execution = validate_payload_keys(
        record["execution_contract"], "failure_execution_contract"
    )
    saved_config = validate_resolved_config(
        execution["resolved_config"], expected_config
    )
    repository = saved_config["repository"]
    if (
        execution["execution_commit"] != repository["execution_commit"]
        or execution["origin_commit"] != repository["origin_commit"]
        or execution["branch"] != repository["branch"]
        or execution["worktree_clean"] is not repository["worktree_clean"]
        or execution["preflight_sha256"] != saved_config["preflight_sha256"]
        or execution["resolved_config_sha256"]
        != saved_config["resolved_config_sha256"]
        or execution["environment"] != saved_config["environment"]
        or execution["source_sha256"] != saved_config["source_sha256"]
        or record["frozen_semantic_bindings"]
        != saved_config["frozen_semantic_input"]
        or record["thresholds"] != PRIMARY_MINIMUM_PIXELS
    ):
        raise ValueError("prelaunch execution/frozen authority bindings changed")
    validate_payload_keys(record["frozen_semantic_bindings"], "frozen_semantic_input")
    validate_payload_keys(record["thresholds"], "failure_thresholds")
    saved_masks = _validate_prelaunch_case_masks(record["case_mask_records"])
    if saved_masks != expected_masks:
        raise ValueError("prelaunch case-mask evidence does not replay exactly")
    expected_failures = _prelaunch_failures(expected_masks)
    if not expected_failures or record["failures"] != expected_failures:
        raise ValueError("prelaunch insufficiency list does not replay from mask counts")
    for item in record["failures"]:
        validate_payload_keys(item, "prelaunch_failure_item")
        if (
            type(item["case_index"]) is not int
            or item["case_index"] not in range(CASE_COUNT)
            or item["domain"] not in PRIMARY_MINIMUM_PIXELS
            or type(item["observed_pixel_count"]) is not int
            or item["minimum_required_pixels"]
            != PRIMARY_MINIMUM_PIXELS[item["domain"]]
            or item["observed_pixel_count"] >= item["minimum_required_pixels"]
        ):
            raise ValueError("prelaunch failure item changed type or threshold meaning")
    score_blind = validate_payload_keys(
        record["score_blind_evidence"], "score_blind_evidence"
    )
    if score_blind != {
        "all_64_masks_built": True,
        "frozen_replay_passed": True,
        "candidate_scalar_render_count": 0,
        "descriptor_call_count": 0,
        "score_landscape_count": 0,
        "success_output_created": False,
    }:
        raise ValueError("prelaunch receipt no longer proves a score-blind failure")


def verify_prelaunch_failure(
    failed_output: Path | None = None,
    success_output: Path | None = None,
) -> dict[str, object]:
    """Reauthenticate and replay the exact one-file score-blind failure tree."""
    repository = repository_state()
    short_commit = str(repository["head"])[:7]
    failed_output = Path(
        failed_output
        or os.environ.get(
            "ANATOMY_TRACKER_IMAGE_INFORMATION_FAILED_OUTPUT",
            ROOT
            / "build"
            / f"arbitrary_plane_image_information_prelaunch_failed_{short_commit}",
        )
    ).resolve()
    success_output = Path(
        success_output
        or os.environ.get(
            "ANATOMY_TRACKER_IMAGE_INFORMATION_OUTPUT",
            ROOT / "build" / f"arbitrary_plane_image_information_raw_{short_commit}",
        )
    ).resolve()
    success_output, failed_output = guard_output_roots(success_output, failed_output)
    if success_output.exists():
        raise ValueError("a verified prelaunch failure cannot coexist with success output")
    actual = sorted(
        path.relative_to(failed_output).as_posix()
        for path in failed_output.rglob("*")
        if path.is_file()
    )
    if actual != ["prelaunch_failure.json"]:
        raise ValueError("prelaunch failure tree must contain exactly one declared JSON file")
    record = _read_strict_json(failed_output / "prelaunch_failure.json")
    source_records = _source_hash_receipts(repository)
    semantic_result, frozen_primary = _authenticate_frozen_semantic_output(
        source_records
    )
    pooled_membership = derive_frozen_pooled_membership(frozen_primary)
    support, render_context, candidate_context = load_allen_contexts()
    _validate_contexts(support, render_context, candidate_context)
    expected_config = _resolved_config(
        repository,
        source_records,
        semantic_result,
        render_context,
        support,
        pooled_membership,
    )
    replay = lambda index: replay_frozen_case(
        index,
        frozen_primary[index],
        render_context,
        candidate_context,
        support,
    )
    expected_masks = build_score_blind_masks(replay)
    _validate_prelaunch_failure_record(record, expected_config, expected_masks)
    final_repository = repository_state()
    final_sources = _source_hash_receipts(final_repository)
    _authenticate_frozen_semantic_output(final_sources)
    if final_repository != repository or final_sources != source_records:
        raise RuntimeError("source authority changed during prelaunch-failure replay")
    return record


def run_image_information(
    output: Path | None = None,
    failed_output: Path | None = None,
) -> dict[str, object]:
    """Execute the frozen pilot; the committed clean-source gate runs before any output."""
    repository = repository_state()
    short_commit = str(repository["head"])[:7]
    output = Path(
        output
        or os.environ.get(
            "ANATOMY_TRACKER_IMAGE_INFORMATION_OUTPUT",
            ROOT / "build" / f"arbitrary_plane_image_information_raw_{short_commit}",
        )
    ).resolve()
    failed_output = Path(
        failed_output
        or os.environ.get(
            "ANATOMY_TRACKER_IMAGE_INFORMATION_FAILED_OUTPUT",
            ROOT / "build" / f"arbitrary_plane_image_information_prelaunch_failed_{short_commit}",
        )
    ).resolve()
    output, failed_output = guard_output_roots(output, failed_output)
    if output.exists() or failed_output.exists():
        raise FileExistsError("ordinary and failed output paths must both be fresh")
    source_records = _source_hash_receipts(repository)
    semantic_result, frozen_primary = _authenticate_frozen_semantic_output(source_records)
    pooled_membership = derive_frozen_pooled_membership(frozen_primary)
    support, render_context, candidate_context = load_allen_contexts()
    _validate_contexts(support, render_context, candidate_context)
    config = _resolved_config(
        repository,
        source_records,
        semantic_result,
        render_context,
        support,
        pooled_membership,
    )
    replay = lambda index: replay_frozen_case(
        index,
        frozen_primary[index],
        render_context,
        candidate_context,
        support,
    )
    mask_audit = build_score_blind_masks(replay)
    post_replay_repository = repository_state()
    post_replay_sources = _source_hash_receipts(post_replay_repository)
    _authenticate_frozen_semantic_output(post_replay_sources)
    if post_replay_repository != repository or post_replay_sources != source_records:
        raise RuntimeError("source authority changed during the score-blind 64-case replay")
    failures = [item for item in mask_audit if item["passed"] is False]
    if failures:
        record = write_prelaunch_failure(failed_output, output, config, mask_audit)
        verified = verify_prelaunch_failure(failed_output, output)
        if verified != record:
            raise ValueError("mandatory prelaunch-failure verification changed its receipt")
        return record
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "resolved_config.json", config)
    print(
        json.dumps(
            {
                "event": "score-blind-prelaunch-pass",
                "case_count": CASE_COUNT,
                "resolved_config_sha256": config["resolved_config_sha256"],
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    case_controls = _stream_cases(
        frozen_primary,
        render_context,
        candidate_context,
        support,
        source_records,
        mask_audit,
        output=output,
    )
    semantic_inventory = _inventory(FROZEN_SEMANTIC_OUTPUT)
    global_controls = _build_global_controls(
        repository, source_records, semantic_inventory, case_controls
    )
    _atomic_json(output / "global_controls.json", global_controls)
    primary, shuffled, controls = _load_case_records(output)
    actual_before_result = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    expected_before_result = expected_output_files() - {"result.json"}
    if actual_before_result != expected_before_result:
        raise ValueError("pre-result output tree differs from the exact 514-file contract")
    inventory = _inventory(output)
    if len(inventory) != 514:
        raise ValueError("pre-result inventory must contain exactly 514 files")
    for item in inventory:
        validate_payload_keys(item, "inventory_item")
    result = _assemble_result(
        config, inventory, primary, shuffled, controls, global_controls
    )
    _atomic_json(output / "result.json", result)
    verified = verify_written_result(output)
    if verified != result:
        raise ValueError("mandatory in-run verification did not reproduce result.json")
    print(
        json.dumps(
            {
                "event": "run-complete",
                "passed": result["gates"]["passed"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return result


def verify_written_result(output: Path | None = None) -> dict[str, object]:
    """Independently replay the complete saved pilot from raw authenticated inputs."""
    repository = repository_state()
    short_commit = str(repository["head"])[:7]
    output = Path(
        output
        or os.environ.get(
            "ANATOMY_TRACKER_IMAGE_INFORMATION_OUTPUT",
            ROOT / "build" / f"arbitrary_plane_image_information_raw_{short_commit}",
        )
    ).resolve()
    output, _ = guard_output_roots(output, None)
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    expected_files = expected_output_files()
    if actual_files != expected_files:
        raise ValueError(
            "output tree differs from exact 515-file contract; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    source_records = _source_hash_receipts(repository)
    semantic_result, frozen_primary = _authenticate_frozen_semantic_output(source_records)
    pooled_membership = derive_frozen_pooled_membership(frozen_primary)
    support, render_context, candidate_context = load_allen_contexts()
    _validate_contexts(support, render_context, candidate_context)
    expected_config = _resolved_config(
        repository,
        source_records,
        semantic_result,
        render_context,
        support,
        pooled_membership,
    )
    config = validate_resolved_config(
        _read_strict_json(output / "resolved_config.json"), expected_config
    )
    result = _read_strict_json(output / "result.json")
    _validate_self_hash(result, "result", "result_payload_sha256")
    if (
        result["schema"] != RUNNER_SCHEMA
        or result["interpretation"] != INTERPRETATION
        or result["resolved_config_sha256"] != config["resolved_config_sha256"]
        or result["data_access"] != DATA_ACCESS
        or result["model_independence"] != MODEL_INDEPENDENCE
    ):
        raise ValueError("result schema/config/scope binding changed")
    validate_payload_keys(result["data_access"], "data_access")
    validate_payload_keys(result["model_independence"], "model_independence")
    inventory = _inventory(output, {"result.json"})
    if (
        len(inventory) != 514
        or result["pre_result_inventory"] != inventory
        or result["pre_result_inventory_sha256"] != canonical_payload_sha256(inventory)
    ):
        raise ValueError("result pre-result inventory does not match exact saved bytes")
    for item in inventory:
        validate_payload_keys(item, "inventory_item")
    primary, shuffled, controls = _load_case_records(output)
    for case_index in range(CASE_COUNT):
        _validate_primary_record(primary[case_index], case_index)
        _validate_shuffled_record(shuffled[case_index], case_index)
        target_index = (case_index + 17) % CASE_COUNT
        if (
            shuffled[case_index]["bank_identity"]["source_primary_payload_sha256"]
            != primary[case_index]["payload_sha256"]
            or shuffled[case_index]["target_identity"]["target_primary_payload_sha256"]
            != primary[target_index]["payload_sha256"]
            or shuffled[case_index]["target_identity"]["target_pixel_pitch_um"]
            != primary[target_index]["target"]["pixel_pitch_um"]
            or shuffled[case_index]["score_domains"]
            != primary[target_index]["score_domains"]
        ):
            raise ValueError("shuffled sidecar does not cross-bind bank i and target i+17")
        _validate_case_control(
            controls[case_index], case_index, primary[case_index], shuffled[case_index]
        )
    if (
        result["primary_case_payload_sha256"]
        != [item["payload_sha256"] for item in primary]
        or result["shuffled_case_payload_sha256"]
        != [item["payload_sha256"] for item in shuffled]
        or result["control_payload_sha256"]
        != [item["payload_sha256"] for item in controls]
        or any(
            not isinstance(result[field], list) or len(result[field]) != CASE_COUNT
            for field in (
                "primary_case_payload_sha256",
                "shuffled_case_payload_sha256",
                "control_payload_sha256",
            )
        )
    ):
        raise ValueError("result does not contain three ordered 64-item sidecar hash lists")
    global_controls = _read_strict_json(output / "global_controls.json")
    _validate_global_controls(global_controls, controls)
    _validate_metrics_contract(
        result["metrics"],
        config["case_and_shuffle_contract"]["pooled_strata_membership"],
    )
    _validate_gates(result["gates"], result["metrics"], global_controls)
    replayed_metrics = aggregate_metrics(
        primary,
        shuffled,
        config["case_and_shuffle_contract"]["pooled_strata_membership"],
    )
    replayed_gates = evaluate_gates(replayed_metrics, global_controls)
    if replayed_metrics != result["metrics"] or replayed_gates != result["gates"]:
        raise ValueError("result metrics/gates do not replay from raw sidecars")
    replay = lambda index: replay_frozen_case(
        index,
        frozen_primary[index],
        render_context,
        candidate_context,
        support,
    )
    mask_audit = build_score_blind_masks(replay)
    replayed_controls = _stream_cases(
        frozen_primary,
        render_context,
        candidate_context,
        support,
        source_records,
        mask_audit,
        expected_output=output,
    )
    replayed_global = _build_global_controls(
        repository,
        source_records,
        _inventory(FROZEN_SEMANTIC_OUTPUT),
        replayed_controls,
    )
    if replayed_global != global_controls:
        raise ValueError("global controls do not replay from final source/case evidence")
    expected_result = _assemble_result(
        config, inventory, primary, shuffled, controls, global_controls
    )
    if expected_result != result:
        raise ValueError("result does not independently replay byte-for-byte")
    return result


if __name__ == "__main__":
    run_image_information()
