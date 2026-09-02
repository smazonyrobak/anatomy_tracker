import copy

import numpy as np
import pytest
import torch

import training.arbitrary_plane_catalogue_v3 as catalogue_v3
import training.arbitrary_plane_training_bank_v3 as bank_v3


def _fixture():
    support = np.ones((12, 11, 10), dtype=bool)
    catalogue = catalogue_v3.make_arbitrary_plane_catalogue_v3(
        support,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        normal_count=12,
        offset_count=3,
        roll_count=4,
        raster_shape_h_w=(8, 8),
        raster_physical_span_y_x_um=(8.0, 8.0),
    )
    tensors = catalogue["tensors"]
    cells = catalogue["counts"]["cell_count"]
    truth_indices = torch.tensor((5, 77))
    batch = {
        "data_role": "development-training",
        "catalogue_id": catalogue["catalogue_id"],
        "catalogue_receipt_sha256": catalogue["receipt_sha256"],
        "full_catalogue_cell_count": cells,
        "catalogue_scope": bank_v3.COMPLETE_CATALOGUE_SCOPE,
        "row_identity": [
            {
                "training_row_id": f"row-{index}",
                "training_row_receipt_sha256": f"row-receipt-{index}",
                "synthetic_realization_id": f"realization-{index}",
                "animal_id": f"animal-{index}",
                "specimen_id": f"specimen-{index}",
                "experiment_id": f"experiment-{index}",
                "synthetic_animal_id": f"synthetic-animal-{index}",
                "section_id": f"section-{index}",
                "split": "development",
            }
            for index in range(2)
        ],
        "cell_id": tensors["cell_id"],
        "cell_states": tensors["cell_states"].float().expand(2, -1, -1),
        "cell_log_mass": tensors["cell_log_mass"].float().expand(2, -1),
        "representation_log_weight": tensors[
            "representation_log_weight"
        ].float().expand(2, -1, -1),
        "representation_to_canonical_raster_affine": tensors[
            "representation_to_canonical_raster_affine"
        ].float().expand(2, -1, -1, -1, -1),
        "truth_state": tensors["cell_states"][0, truth_indices].float(),
        "truth_catalogue_cell_index": truth_indices,
        "truth_catalogue_cell_source_index": truth_indices.clone(),
        "truth_catalogue_cell_id": truth_indices.clone(),
    }
    assert cells == 144
    return batch, catalogue


def test_training_bank_is_deterministic_unique_truth_first_and_model_ready():
    batch, catalogue = _fixture()
    first = bank_v3.make_training_candidate_batch_v3(
        batch, catalogue, bank_size=24, root_seed="bank-seed"
    )
    second = bank_v3.make_training_candidate_batch_v3(
        copy.deepcopy(batch), catalogue, bank_size=24, root_seed="bank-seed"
    )
    assert torch.equal(
        first["selected_full_catalogue_indices"],
        second["selected_full_catalogue_indices"],
    )
    assert torch.equal(
        first["selected_full_catalogue_indices"][:, 0],
        batch["truth_catalogue_cell_index"],
    )
    assert all(len(set(row.tolist())) == 24 for row in first["selected_full_catalogue_indices"])
    assert first["cell_states"].shape == (2, 24, 12)
    assert first["representation_log_weight"].shape == (2, 24, 2)
    assert torch.equal(first["cell_id"], torch.arange(24))
    assert torch.equal(first["truth_catalogue_cell_index"], torch.zeros(2, dtype=torch.long))
    assert torch.equal(
        first["truth_catalogue_cell_source_index"],
        batch["truth_catalogue_cell_index"],
    )
    assert torch.allclose(first["cell_log_mass"].logsumexp(1), torch.zeros(2))
    assert all(receipt["learned_dependencies"] == [] for receipt in first["training_candidate_bank_receipts"])
    assert all(not receipt["inference_scope"] for receipt in first["training_candidate_bank_receipts"])
    for receipt, identity in zip(
        first["training_candidate_bank_receipts"], first["row_identity"]
    ):
        bank_v3.verify_training_candidate_bank_receipt_v3(
            receipt,
            expected_catalogue_id=catalogue["catalogue_id"],
            expected_catalogue_receipt_sha256=catalogue["receipt_sha256"],
            expected_training_row_id=identity["training_row_id"],
        )
    tampered = copy.deepcopy(first["training_candidate_bank_receipts"][0])
    tampered["bank_size"] += 1
    with pytest.raises(ValueError, match="receipt"):
        bank_v3.verify_training_candidate_bank_receipt_v3(
            tampered,
            expected_catalogue_id=catalogue["catalogue_id"],
            expected_catalogue_receipt_sha256=catalogue["receipt_sha256"],
            expected_training_row_id="row-0",
        )


def test_training_bank_changes_global_draw_by_row_seed_and_rejects_nontraining_use():
    batch, catalogue = _fixture()
    first = bank_v3.make_training_candidate_batch_v3(
        batch, catalogue, bank_size=24, root_seed=1
    )
    second = bank_v3.make_training_candidate_batch_v3(
        batch, catalogue, bank_size=24, root_seed=2
    )
    assert not torch.equal(
        first["selected_full_catalogue_indices"][:, -4:],
        second["selected_full_catalogue_indices"][:, -4:],
    )
    forbidden = dict(batch)
    forbidden["data_role"] = "final-test"
    with pytest.raises(ValueError, match="development-only"):
        bank_v3.make_training_candidate_batch_v3(
            forbidden, catalogue, bank_size=24, root_seed=1
        )
