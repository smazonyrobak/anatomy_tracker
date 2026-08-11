from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.evaluate_sealed_registered_holdout import (
    RELEASE_GATE_THRESHOLDS,
    SEALED_SPLIT,
    brain_masked_plane_distance,
    evaluation_domains,
    ordered_experiment_groups,
    paired_animal_bootstrap,
    require_complete_sealed_images,
    release_quality_gate,
    sealed_release_report,
    validate_sealed_boundary,
)
from training.train_atlas_pose_v7 import FINAL_GATE_THRESHOLDS as TRAINING_FINAL_GATE_THRESHOLDS


def record(section_id, experiment_id, specimen_id, section_number, split=SEALED_SPLIT):
    return {
        "section_image_id": section_id,
        "experiment_id": experiment_id,
        "specimen_id": specimen_id,
        "section_number": section_number,
        "split": split,
        "quicknii_ouv": [0.0, 312.0, 320.0, 456.0, 0.0, 0.0, 0.0, 0.0, -320.0],
        "relative_path": f"images/{split}/{experiment_id}/{section_id}.jpg",
        "in_training_ap_domain": True,
    }


def test_sealed_records_are_grouped_by_experiment_and_cutting_index():
    records = [
        record(30, 2, 20, 8),
        record(12, 1, 10, 4),
        record(11, 1, 10, 4),
        record(10, 1, 10, 2),
    ]
    groups = ordered_experiment_groups(validate_sealed_boundary(records))
    assert list(groups) == [1, 2]
    assert [(row["section_number"], row["section_image_id"]) for row in groups[1]] == [
        (2, 10),
        (4, 11),
        (4, 12),
    ]


def test_brain_masked_ouv_plane_distance_uses_corresponding_pixel_locations():
    ground_truth = np.asarray([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    translated = ground_truth.copy()
    translated[0] += 3.0
    mask = np.asarray([[True, False], [False, True]])
    assert brain_masked_plane_distance(ground_truth, translated, mask) == pytest.approx(3.0)

    sheared = ground_truth.copy()
    sheared[3] += 4.0
    left_column = np.asarray([[True, False], [True, False]])
    assert brain_masked_plane_distance(ground_truth, sheared, left_column) == pytest.approx(1.0)


def test_sealed_specimen_or_experiment_cannot_cross_the_training_boundary():
    sealed = record(1, 100, 200, 1)
    with pytest.raises(ValueError, match="sealed specimen or experiment"):
        validate_sealed_boundary([sealed, record(2, 101, 200, 2, "train")])
    with pytest.raises(ValueError, match="sealed specimen or experiment"):
        validate_sealed_boundary([sealed, record(2, 100, 201, 2, "validation")])


def test_inference_gate_refuses_an_incomplete_sealed_image_set(tmp_path: Path):
    rows = [record(1, 100, 200, 1)]
    with pytest.raises(RuntimeError, match="SEALED INFERENCE REFUSED"):
        require_complete_sealed_images(tmp_path, rows, expected_sections=1, expected_experiments=1)


def test_animal_bootstrap_weights_animals_not_their_section_counts():
    rows = pd.DataFrame(
        [
            {"method": method, "specimen_id": animal, "section_image_id": section, "error": error}
            for method, animal, section, error in (
                ("candidate", 1, 1, 0.0),
                ("reference", 1, 1, 2.0),
                ("candidate", 2, 2, 10.0),
                ("reference", 2, 2, 0.0),
                ("candidate", 2, 3, 10.0),
                ("reference", 2, 3, 0.0),
                ("candidate", 2, 4, 10.0),
                ("reference", 2, 4, 0.0),
            )
        ]
    )
    result = paired_animal_bootstrap(rows, "candidate", "reference", "error", iterations=100, seed=3)
    assert result["animal_count"] == 2
    assert result["paired_section_count"] == 4
    assert result["delta_candidate_minus_reference"] == pytest.approx(4.0)


def test_primary_sealed_metrics_exclude_out_of_training_domain_sections():
    table = pd.DataFrame(
        {
            "section_image_id": [1, 2, 3],
            "in_training_ap_domain": [True, False, True],
        }
    )
    primary, excluded = evaluation_domains(table)
    assert primary["section_image_id"].tolist() == [1, 3]
    assert excluded["section_image_id"].tolist() == [2]


def test_sealed_release_report_requires_quality_and_deepslice_superiority():
    assert RELEASE_GATE_THRESHOLDS == TRAINING_FINAL_GATE_THRESHOLDS
    rows = pd.DataFrame(
        [
            {
                "method": "atlas_pose",
                "specimen_id": specimen,
                "section_image_id": specimen,
                "product": str(5 + 3 * (specimen - 1)),
                "in_training_ap_domain": True,
                "gt_ap_um": -1000.0 - 500.0 * specimen,
                "pred_ap_um": -980.0 - 500.0 * specimen,
                "gt_lr_deg": 1.0,
                "pred_lr_deg": 1.2,
                "gt_dv_deg": -2.0,
                "pred_dv_deg": -1.7,
            }
            for specimen in (1, 2)
        ]
    )
    quality = release_quality_gate(rows)
    assert quality["all_gates_passed"] is True
    assert quality["thresholds"] == RELEASE_GATE_THRESHOLDS
    comparisons = [
        {
            "candidate": "atlas_pose",
            "reference": "deepslice_mens_ai_ci",
            "metric": f"absolute_error_{axis}",
            "delta_candidate_minus_reference": -1.0,
            "probability_candidate_lower_error": 0.99,
        }
        for axis in ("ap_um", "lr_deg", "dv_deg")
    ]
    release = sealed_release_report(
        rows,
        comparisons,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        {"trainer.py": "d" * 64},
        {"atlas": {"annotation": "e" * 64}},
        {"sections_sha256": "f" * 64},
        "1" * 64,
        "2" * 64,
        "now",
    )
    assert release["release_approved"] is True
    assert release["promotion_ready"] is True
    assert release["release_report_version"] == 2
    assert len(release["release_integrity_sha256"]) == 64

    comparisons[0]["probability_candidate_lower_error"] = 0.80
    rejected = sealed_release_report(
        rows,
        comparisons,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        {"trainer.py": "d" * 64},
        {"atlas": {"annotation": "e" * 64}},
        {"sections_sha256": "f" * 64},
        "1" * 64,
        "2" * 64,
        "now",
    )
    assert rejected["release_approved"] is False
    assert rejected["promotion_ready"] is False


def test_sealed_evaluator_is_not_imported_by_training_or_model_selection_code():
    root = Path(__file__).parents[1]
    forbidden = "evaluate_sealed_registered_holdout"
    for path in (root / "training").glob("*.py"):
        if path.name == f"{forbidden}.py":
            continue
        assert forbidden not in path.read_text(encoding="utf-8")
