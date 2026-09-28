import json
import math
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from training import arbitrary_plane_development_evaluation_v6 as evaluation_v6
from training import arbitrary_plane_inference_v6 as inference_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components


def _state(ap):
    return full_frame_state_from_components(
        torch.tensor([ap, 0.0, 0.0], dtype=torch.float64),
        torch.eye(3, dtype=torch.float64),
        torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float64),
    )


@pytest.fixture
def i_roots():
    root = Path("I:/AnatomyTracker/tmp") / f"v6-evaluator-test-{uuid.uuid4().hex}"
    cache = root / "cache"
    run = root / "run"
    output = root / "output"
    cache.mkdir(parents=True)
    run.mkdir()
    yield cache, run, output
    shutil.rmtree(root)


def _row(index, mode):
    height = width = 5
    available = mode != "smart-brush-absent"
    channels = np.zeros((height, width, 3), dtype=np.float32)
    if available:
        brush_mask = np.zeros((height, width), dtype=bool)
        brush_mask[1:4, 1:4] = True
        channels[:, :, 0] = 0.5 * brush_mask
        channels[:, :, 1] = inference_v6._outline_from_mask(brush_mask)
    else:
        channels[:, :, 0] = 0.5
    channels[:, :, 2] = float(available)
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    identity = np.stack((y, x), axis=-1).astype(np.float64)
    digest = f"{index + 1:064x}"
    lineage = {
        "animal_id": f"dev-animal-{index}",
        "specimen_id": f"dev-specimen-{index}",
        "experiment_id": f"dev-experiment-{index}",
        "section_id": f"dev-section-{index}",
        "synthetic_animal_id": f"dev-synthetic-animal-{index}",
        "split": "development",
    }
    return {
        "training_row_id": f"{index + 11:064x}",
        "receipt_sha256": digest,
        "synthetic_realization_id": f"{index + 21:064x}",
        "lineage": lineage,
        "selected_mode": mode,
        "array_receipts": {
            "model_input_channels_float32": evaluation_v6.finite_rows_v6._array_receipt(channels),
        },
        "canonical_effective_quicknii_ouv_float64": [0.0] * 9,
        "upstream_reference": {
            "support_supervision_contract": {
                "point_pose_supervision_weight": 1.0,
                "dense_deformation_supervision_weight": 1.0,
            }
        },
        "finite_psf_contract": {
            "finite_psf_sha256": f"{index + 31:064x}",
            "nominal_cut_thickness_um": 80.0,
            "axial_offsets_um": np.linspace(-40.0, 40.0, 9).tolist(),
            "axial_weights": (np.asarray([1, 2, 2, 2, 2, 2, 2, 2, 1]) / 16.0).tolist(),
        },
        "arrays": {
            "model_input_channels_float32": channels,
            "truth_section_pullback_stationary_velocity_yx_px_float64": np.zeros(
                (height, width, 2), dtype=np.float64
            ),
            "truth_section_pullback_map_yx_px_float64": identity,
            "truth_section_deformation_valid_mask": np.ones((height, width), dtype=bool),
            "target_valid_correspondence_mask": np.ones((height, width), dtype=bool),
            "target_correspondence_abstention_mask": np.zeros((height, width), dtype=bool),
            "source_tissue_ground_truth_mask": np.ones((height, width), dtype=bool),
        },
    }


def _setup(monkeypatch, cache, run):
    digest = "a" * 64
    rows = [_row(0, "smart-brush-absent"), _row(1, "smart-brush-accurate")]
    records = [
        {
            "training_row_id": row["training_row_id"],
            "training_row_receipt_sha256": row["receipt_sha256"],
            "lineage": row["lineage"],
            "selected_mode": row["selected_mode"],
        }
        for row in rows
    ]
    manifest = {
        "receipt_sha256": digest,
        "status": evaluation_v6.finite_rows_v6.FROZEN_CACHE_STATUS,
        "row_count": len(rows),
        "rows": records,
        "generation_lineage": {"split": "development"},
        "generator_binding": {
            "receipt_sha256": "b" * 64,
            "generation_lineage_sha256": "c" * 64,
        },
        "freeze_audit": {"all_rows_authenticated": True},
    }
    states = torch.stack((_state(0.0), _state(10.0))).numpy()
    catalogue = {
        "catalogue_id": "tiny-complete-catalogue",
        "receipt_sha256": "d" * 64,
        "arrays": {
            "cell_states_float64": states,
            "cell_id_int64": np.arange(2, dtype=np.int64),
            "normal_offset_table_um_float64": np.asarray([[-1.0, 1.0]]),
        },
        "counts": {"cell_count": 2, "roll_count": 1},
        "coverage_audit": {"max_observed_rp2_angular_covering_radius_rad": 1.0},
        "support_geometry": {
            "support_origin_ap_dv_ml_um": [0.0, 0.0, 0.0],
            "origin_ap_dv_ml_um": [0.0, 0.0, 0.0],
            "voxel_size_ap_dv_ml_um": [1.0, 1.0, 1.0],
            "raster_physical_span_y_x_um": [5.0, 5.0],
        },
    }
    training_identity = {
        key: f"train-{key.replace('_id', '')}" for key in evaluation_v6.FIVE_IDS
    }
    run_manifest = {
        "git_commit": "e" * 40,
        "source_sha256": {"training/model.py": "f" * 64},
        "initialization": "fresh_random_only",
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
        "training_data": {
            "training_data_manifest_receipt_sha256": "1" * 64,
            "ordered_row_identities": [training_identity],
        },
    }
    loaded = {
        "run_manifest_receipt_sha256": "2" * 64,
        "run_state_receipt_sha256": "3" * 64,
        "checkpoint_receipt_sha256": "4" * 64,
        "checkpoint_model_state_sha256": "5" * 64,
        "context": {
            "run_directory": str(run.resolve()),
            "manifest": run_manifest,
            "catalogue": catalogue,
            "atlas_volume": np.zeros((2, 5, 5, 5), dtype=np.float32),
        },
    }

    def load_manifest(path, *, expected_manifest_receipt_sha256):
        assert Path(path).resolve() == cache.resolve()
        assert expected_manifest_receipt_sha256 == digest
        return manifest

    def load_rows(path, indices=None, *, expected_manifest_receipt_sha256):
        load_manifest(path, expected_manifest_receipt_sha256=expected_manifest_receipt_sha256)
        indices = list(range(len(rows))) if indices is None else list(indices)
        return {"rows": [rows[index] for index in indices]}

    calls = []

    def load_inference(path, **kwargs):
        assert Path(path).resolve() == run.resolve()
        assert kwargs["expected_run_manifest_receipt_sha256"] == "2" * 64
        assert kwargs["expected_inference_source_sha256"] == inference_v6._source_receipts()
        return loaded

    def run_inference(received, **kwargs):
        assert received is loaded
        calls.append(kwargs["case_ids"]["animal_id"])
        row = rows[int(kwargs["case_ids"]["animal_id"].rsplit("-", 1)[1])]
        assert np.array_equal(kwargs["model_input_channels_float32"], row["arrays"]["model_input_channels_float32"])
        assert kwargs["expected_model_input_receipt"] == row["array_receipts"]["model_input_channels_float32"]
        assert kwargs["prepared_source_receipt_sha256"] == row["receipt_sha256"]
        assert "brush_mask" not in kwargs and "truth" not in kwargs
        input_receipt = evaluation_v6._expected_input_receipt(row, catalogue)
        height = width = 5
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        identity = torch.stack((y, x)).to(torch.float32)
        pose = {
            "retrieval_teacher_forced_mask": torch.tensor([False]),
            "conditional_within_topk_cell_log_probability": torch.log(
                torch.tensor([[0.75, 0.25]])
            ),
            "retrieval_topk_retained_probability": torch.tensor([1.0]),
            "final_cell_state": torch.from_numpy(states)[None],
            "final_cell_canonical_plane_covariance": torch.eye(3)[None, None].expand(1, 2, 3, 3).clone(),
        }
        refined = {
            "pose": pose,
            "final_stationary_velocity_yx_px": torch.zeros(1, 2, 2, height, width),
            "final_pullback_map_yx_px": identity[None, None].expand(1, 2, 2, height, width).clone(),
            "final_forward_jacobian_determinant": torch.ones(1, 2, 1, height, width),
            "final_forward_then_inverse_valid_mask": torch.ones(1, 2, 1, height, width, dtype=torch.bool),
            "final_inverse_then_forward_valid_mask": torch.ones(1, 2, 1, height, width, dtype=torch.bool),
            "final_forward_then_inverse_error_yx": torch.zeros(1, 2, 2, height, width),
            "final_inverse_then_forward_error_yx": torch.zeros(1, 2, 2, height, width),
            "final_deformed_canonical_render": torch.ones(1, 2, 2, height, width),
        }
        result = {
            "schema_version": inference_v6.INFERENCE_V6_SCHEMA,
            "evaluation_stage": "proposal" if kwargs["proposal_only"] else "joint",
            "probabilities_calibrated": False,
            "probability_status": "raw_uncalibrated",
            "input_receipt": input_receipt,
            "run_binding": {
                "run_manifest_receipt_sha256": "2" * 64,
                "run_state_receipt_sha256": "3" * 64,
                "checkpoint_receipt_sha256": "4" * 64,
                "checkpoint_model_state_sha256": "5" * 64,
                "catalogue_receipt_sha256": "d" * 64,
            },
            "trusted_inference_source_sha256": inference_v6._source_receipts(),
            "posterior": {
                "raw_full_catalogue_proposal_log_probability": torch.log(
                    torch.tensor([[0.6, 0.4]])
                ),
                "honest_hybrid_posterior": {
                    "hybrid_cell_log_probability": torch.log(torch.tensor([[0.6, 0.4]])),
                    "hybrid_topk_catalogue_index": torch.tensor([[0, 1]]),
                },
            },
            "k_poses": {"catalogue_index": torch.tensor([[0, 1]]), "cell_id": torch.tensor([[0, 1]]), "pose": pose},
            "recurrent_output": refined,
            "deformation": {},
            "abstention": {
                "ready_mask": torch.tensor([True]),
                "abstained_mask": torch.tensor([False]),
                "reason": ("ready",),
            },
        }
        if kwargs["proposal_only"]:
            result["posterior"]["honest_hybrid_posterior"] = None
            result["recurrent_output"] = None
            result["deformation"] = None
        return result

    monkeypatch.setattr(evaluation_v6.finite_rows_v6, "load_frozen_row_cache_manifest_v6", load_manifest)
    monkeypatch.setattr(evaluation_v6.finite_rows_v6, "load_frozen_training_rows_v6", load_rows)
    monkeypatch.setattr(evaluation_v6.inference_v6, "load_arbitrary_plane_inference_v6", load_inference)
    monkeypatch.setattr(evaluation_v6.inference_v6, "run_arbitrary_plane_prepared_inference_v6", run_inference)
    monkeypatch.setattr(
        evaluation_v6,
        "_truth_state",
        lambda row, *args: torch.from_numpy(states[int(row["lineage"]["animal_id"].rsplit("-", 1)[1])]),
    )
    monkeypatch.setattr(
        evaluation_v6,
        "_truth_catalogue_index",
        lambda truth, context: int(float(truth[0]) > 0.0),
    )
    return digest, rows, calls


def test_all_row_truth_free_evaluation_and_independent_replay(monkeypatch, i_roots):
    cache, run, output = i_roots
    cache_receipt, rows, calls = _setup(monkeypatch, cache, run)
    source = inference_v6._source_receipts()
    bundle = evaluation_v6.run_arbitrary_plane_development_evaluation_v6(
        cache,
        run,
        output,
        expected_cache_manifest_receipt_sha256=cache_receipt,
        expected_run_manifest_receipt_sha256="2" * 64,
        expected_inference_source_sha256=source,
        row_chunk_size=1,
    )
    assert calls == ["dev-animal-0", "dev-animal-1"]
    assert evaluation_v6.verify_arbitrary_plane_development_evaluation_v6(
        output,
        expected_bundle_receipt_sha256=bundle["receipt_sha256"],
        expected_cache_manifest_receipt_sha256=cache_receipt,
        expected_run_manifest_receipt_sha256="2" * 64,
        expected_inference_source_sha256=source,
    )
    report = json.loads((output / "development_evaluation_report.json").read_text("ascii"))
    assert report["row_accounting"]["reported_row_count"] == len(rows)
    assert report["row_accounting"]["no_rows_dropped"]
    assert report["animal_macro_metrics"]["statistical_unit"] == "animal"
    assert report["animal_macro_metrics"]["animal_count"] == 2
    assert report["mode_stratified_metrics"]["smart-brush-absent"]["row_count"] == 1
    assert report["mode_stratified_metrics"]["smart-brush-accurate"]["row_count"] == 1
    assert report["raw_calibration_diagnostics"]["post_hoc_fitting_performed"] is False
    assert report["probability_status"] == "raw_uncalibrated_non_release"
    assert report["release_qualifying"] is False
    assert report["identity_overlap_with_training"] == {key: [] for key in evaluation_v6.FIVE_IDS}
    assert len(list((output / "raw_predictions").glob("*.pt"))) == len(rows)
    raw = torch.load(next((output / "raw_predictions").glob("*.pt")), weights_only=True)
    assert raw["raw_inference_result"]["posterior"]["raw_full_catalogue_proposal_log_probability"].shape == (1, 2)
    assert raw["identifiers"] == {key: rows[0]["lineage"][key] for key in evaluation_v6.FIVE_IDS}


def test_raw_prediction_tamper_and_untrusted_paths_are_rejected(monkeypatch, i_roots):
    cache, run, output = i_roots
    cache_receipt, _, _ = _setup(monkeypatch, cache, run)
    source = inference_v6._source_receipts()
    bundle = evaluation_v6.run_arbitrary_plane_development_evaluation_v6(
        cache,
        run,
        output,
        expected_cache_manifest_receipt_sha256=cache_receipt,
        expected_run_manifest_receipt_sha256="2" * 64,
        expected_inference_source_sha256=source,
    )
    raw_path = next((output / "raw_predictions").glob("*.pt"))
    with raw_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="file receipt"):
        evaluation_v6.verify_arbitrary_plane_development_evaluation_v6(
            output,
            expected_bundle_receipt_sha256=bundle["receipt_sha256"],
            expected_cache_manifest_receipt_sha256=cache_receipt,
            expected_run_manifest_receipt_sha256="2" * 64,
            expected_inference_source_sha256=source,
        )
    with pytest.raises(ValueError, match="restricted to I"):
        evaluation_v6.run_arbitrary_plane_development_evaluation_v6(
            "C:/Windows",
            run,
            output / "elsewhere",
            expected_cache_manifest_receipt_sha256=cache_receipt,
            expected_run_manifest_receipt_sha256="2" * 64,
            expected_inference_source_sha256=source,
        )
    with pytest.raises(ValueError, match="trusted"):
        evaluation_v6.run_arbitrary_plane_development_evaluation_v6(
            cache,
            run,
            output / "elsewhere",
            expected_cache_manifest_receipt_sha256="bad",
            expected_run_manifest_receipt_sha256="2" * 64,
            expected_inference_source_sha256=source,
        )


def test_cached_boundary_is_preserved_without_inferred_mask():
    row = _row(0, "smart-brush-accurate")
    row["arrays"]["model_input_channels_float32"][2, 2, 0] = 0.0
    row["array_receipts"]["model_input_channels_float32"] = evaluation_v6.finite_rows_v6._array_receipt(
        row["arrays"]["model_input_channels_float32"]
    )
    catalogue = {"support_geometry": {"raster_physical_span_y_x_um": [5.0, 5.0]}}
    arguments = evaluation_v6._input_arguments(row, catalogue)
    assert np.array_equal(arguments["model_input_channels_float32"], row["arrays"]["model_input_channels_float32"])
    assert "brush_mask" not in arguments


def test_proposal_evaluation_omits_refinement_and_reports_animal_macro(monkeypatch, i_roots):
    cache, run, output = i_roots
    cache_receipt, _, calls = _setup(monkeypatch, cache, run)
    evaluation_v6.run_arbitrary_plane_development_evaluation_v6(
        cache, run, output,
        expected_cache_manifest_receipt_sha256=cache_receipt,
        expected_run_manifest_receipt_sha256="2" * 64,
        expected_inference_source_sha256=inference_v6._source_receipts(),
        evaluation_stage="proposal",
    )
    assert len(calls) == 2
    report = json.loads((output / "development_evaluation_report.json").read_text("ascii"))
    macro = report["animal_macro_metrics"]["macro_across_animals"]
    assert macro["retrieval.proposal_top1_recall"]["mean"] == 0.5
    assert macro["retrieval.proposal_top8_recall"]["mean"] == 1.0
    assert macro["retrieval.proposal_truth_nll_nats"]["mean"] == pytest.approx(-0.5 * (math.log(0.6) + math.log(0.4)))
    assert macro["deformation.deformation_failure"]["eligible_count"] == 0
    assert report["row_accounting"]["abstained_row_count"] == 0


def test_fixed_calibration_diagnostics_do_not_fit():
    rows = []
    for animal, confidence, correct, mahalanobis in (
        ("a", 0.9, True, 1.0),
        ("b", 0.8, False, 9.0),
    ):
        rows.append(
            {
                "animal_id": animal,
                "metrics": {
                    "uncertainty": {
                        "raw_top1_confidence": confidence,
                        "raw_top1_correct": correct,
                        "categorical_truth_nll_nats": -math.log(0.5),
                        "categorical_brier_score": 0.5,
                        "local_plane_gaussian": {
                            "available": True,
                            "mahalanobis_squared_df3": mahalanobis,
                            "coverage": {
                                key: mahalanobis <= threshold
                                for key, threshold in evaluation_v6.CHI2_DF3_QUANTILES.items()
                            },
                        },
                    }
                },
            }
        )
    rows.append(rows[0])  # Extra sections must not give this animal more weight.
    diagnostics = evaluation_v6._calibration_diagnostics(rows)
    assert diagnostics["post_hoc_fitting_performed"] is False
    assert diagnostics["coverage_claimed"] is False
    assert diagnostics["local_plane_gaussian"]["empirical_coverage"]["90"] == 0.5
    assert diagnostics["top1_expected_calibration_error"] == pytest.approx(0.45)
