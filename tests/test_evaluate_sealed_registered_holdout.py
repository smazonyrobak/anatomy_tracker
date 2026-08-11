from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.evaluate_sealed_registered_holdout import (
    SEALED_SPLIT,
    brain_masked_plane_distance,
    ordered_experiment_groups,
    paired_animal_bootstrap,
    require_complete_sealed_images,
    validate_sealed_boundary,
)


def record(section_id, experiment_id, specimen_id, section_number, split=SEALED_SPLIT):
    return {
        "section_image_id": section_id,
        "experiment_id": experiment_id,
        "specimen_id": specimen_id,
        "section_number": section_number,
        "split": split,
        "quicknii_ouv": [0.0, 312.0, 320.0, 456.0, 0.0, 0.0, 0.0, 0.0, -320.0],
        "relative_path": f"images/{split}/{experiment_id}/{section_id}.jpg",
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


def test_sealed_evaluator_is_not_imported_by_training_or_model_selection_code():
    root = Path(__file__).parents[1]
    forbidden = "evaluate_sealed_registered_holdout"
    for path in (root / "training").glob("*.py"):
        if path.name == f"{forbidden}.py":
            continue
        assert forbidden not in path.read_text(encoding="utf-8")
