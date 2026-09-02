import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from training import arbitrary_plane_inference_v6 as inference_v6


class _Runtime:
    def expand(self, batch_size):
        assert batch_size == 1
        return "complete-catalogue-batch"


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.call = None

    def forward(self, *args, **kwargs):
        self.call = (args, kwargs, self.training, torch.is_grad_enabled())
        honest = {"hybrid_topk_catalogue_index": torch.tensor([[4, 8]])}
        return {
            "cascade": {
                "raw_full_catalogue_proposal_log_probability": torch.tensor([[0.0]]),
                "honest_hybrid_posterior": honest,
                "honest_refinement_abstention_reason": ("ready",),
            },
            "refinement_ready_mask": torch.tensor([True]),
            "refinement_abstained_mask": torch.tensor([False]),
            "refinement_selected_catalogue_index": torch.tensor([[4, 8]]),
            "refinement_selected_cell_id": torch.tensor([[14, 18]]),
            "refined_output": {
                "pose": {"final_state": torch.ones(1, 2, 12)},
                "deformation_active_sequence": torch.tensor([[[False, True]]]),
                "final_deformed_canonical_render": torch.ones(1, 2, 1, 4, 4),
            },
        }


def _loaded(monkeypatch):
    model = _Model()
    digest = "a" * 64
    checkpoint = {
        "manifest": {
            "initialization": "fresh_random_only",
            "prior_model_weight_dependencies": [],
            "prior_feature_dependencies": [],
            "prior_pseudolabel_dependencies": [],
        },
        "initialization_receipt": {
            "algorithm": "fresh-pytorch-random-initialization"
        },
        "learned_dependencies": {
            "model_weights": [],
            "features": [],
            "pseudolabels": [],
        },
        "probabilities_calibrated": False,
        "uncertainty_status": "raw_uncalibrated",
        "receipt_sha256": digest,
        "model_state_sha256": digest,
        "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }
    context = {
        "manifest": {
            "initialization": "fresh_random_only",
            "receipt_sha256": digest,
            "training_config": {
                "retrieval_shape_h_w": [2, 2],
                "proposal_top_m": 8,
                "top_k": 2,
                "refinement_steps": 3,
            },
        },
        "run_state": {"receipt_sha256": digest},
        "catalogue": {
            "receipt_sha256": digest,
            "support_geometry": {
                "raster_shape_h_w": [4, 4],
                "raster_physical_span_y_x_um": [40.0, 80.0],
                "origin_ap_dv_ml_um": [0.0, 0.0, 0.0],
                "voxel_size_ap_dv_ml_um": [25.0, 25.0, 25.0],
            },
        },
        "catalogue_runtime": _Runtime(),
        "atlas_volume": np.ones((2, 2, 2), dtype=np.float32),
    }
    monkeypatch.setattr(inference_v6.trainer_v6, "verify_staged_checkpoint_v6", lambda value: True)
    loaded = {
        "schema_version": inference_v6.INFERENCE_V6_SCHEMA,
        "context": context,
        "checkpoint": checkpoint,
        "model": model,
        "run_manifest_receipt_sha256": digest,
        "run_state_receipt_sha256": digest,
        "checkpoint_receipt_sha256": digest,
        "checkpoint_model_state_sha256": digest,
        "trusted_inference_source_sha256": inference_v6._source_receipts(),
    }
    loaded["authenticated_context_seals"] = inference_v6._context_seals(context)
    return loaded


def _schedule(thickness=80.0):
    return (
        np.linspace(-thickness / 2, thickness / 2, 9),
        np.asarray([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=np.float64) / 16,
    )


def test_truth_free_imperfect_mask_inference_and_receipts(monkeypatch):
    loaded = _loaded(monkeypatch)
    image = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    outline = np.zeros((4, 4), dtype=np.uint8)
    outline[1:3, 1:3] = 1
    offsets, weights = _schedule()
    result = inference_v6.run_arbitrary_plane_inference_v6(
        loaded,
        image,
        input_mode="imperfect-mask",
        outline=outline,
        outline_available=True,
        physical_fov_y_x_um=(40.0, 80.0),
        pixel_size_y_x_um=(10.0, 20.0),
        nominal_cut_thickness_um=80.0,
        axial_offsets_um=offsets,
        axial_weights=weights,
        case_ids={"animal_id": "animal-1", "section_id": "section-4"},
    )
    args, kwargs, training, grad_enabled = loaded["model"].call
    assert not training and not grad_enabled
    assert not ({"training_truth_catalogue_index", "dense_deformation_supervision_weight"} & set(kwargs))
    assert args[4] == "complete-catalogue-batch"
    assert torch.count_nonzero(args[0][0, 0][args[1][0, 0] == 0]) == 0
    assert result["probability_status"] == "raw_uncalibrated"
    assert result["posterior"]["honest_hybrid_posterior"]["hybrid_topk_catalogue_index"].tolist() == [[4, 8]]
    assert result["k_poses"]["pose"]["final_state"].shape == (1, 2, 12)
    assert result["deformation"]["deformation_active_sequence"].shape == (1, 1, 2)
    assert result["input_receipt"]["case_ids"] == {
        "animal_id": "animal-1",
        "specimen_id": None,
        "experiment_id": None,
        "section_id": "section-4",
        "synthetic_animal_id": None,
    }
    assert len(result["input_receipt"]["receipt_sha256"]) == 64


def test_raw_mode_is_unmodified_and_assisted_contract_is_explicit(monkeypatch):
    loaded = _loaded(monkeypatch)
    image = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    offsets, weights = _schedule()
    inference_v6.run_arbitrary_plane_inference_v6(
        loaded,
        image,
        input_mode="raw",
        outline=None,
        outline_available=False,
        physical_fov_y_x_um=(40.0, 80.0),
        pixel_size_y_x_um=(10.0, 20.0),
        nominal_cut_thickness_um=80.0,
        axial_offsets_um=offsets,
        axial_weights=weights,
    )
    assert np.array_equal(loaded["model"].call[0][0][0, 0].numpy(), image)
    with pytest.raises(ValueError, match="raw mode"):
        inference_v6._prepare_input(image, "raw", np.ones_like(image), True)
    with pytest.raises(ValueError, match="require"):
        inference_v6._prepare_input(image, "black-exterior", None, False)
    with pytest.raises(ValueError, match="normalized"):
        inference_v6._prepare_input(np.full((4, 4), 255, dtype=np.uint8), "raw", None, False)


def test_schedule_geometry_model_and_source_tampering_are_rejected(monkeypatch):
    loaded = _loaded(monkeypatch)
    image = np.ones((4, 4), dtype=np.float32)
    offsets, weights = _schedule()
    weights[0] += 0.01
    with pytest.raises(ValueError, match="S=9"):
        inference_v6.run_arbitrary_plane_inference_v6(
            loaded,
            image,
            input_mode="raw",
            outline=None,
            outline_available=False,
            physical_fov_y_x_um=(40.0, 80.0),
            pixel_size_y_x_um=(10.0, 20.0),
            nominal_cut_thickness_um=80.0,
            axial_offsets_um=offsets,
            axial_weights=weights,
        )
    loaded = _loaded(monkeypatch)
    loaded["model"].weight.data.add_(1)
    offsets, weights = _schedule()
    with pytest.raises(ValueError, match="model state changed"):
        inference_v6.run_arbitrary_plane_inference_v6(
            loaded,
            image,
            input_mode="raw",
            outline=None,
            outline_available=False,
            physical_fov_y_x_um=(40.0, 80.0),
            pixel_size_y_x_um=(10.0, 20.0),
            nominal_cut_thickness_um=80.0,
            axial_offsets_um=offsets,
            axial_weights=weights,
        )


def test_module_has_no_legacy_model_checkpoint_or_automatic_segmentation_imports():
    path = Path(inference_v6.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("v3" in name or "v4" in name for name in imports)
    assert not any("segmentation" in name or "candidate_bank" in name for name in imports)


def test_loader_requires_trusted_anchor_and_i_only_io():
    source = inference_v6._source_receipts()
    with pytest.raises(ValueError, match="trusted"):
        inference_v6.load_arbitrary_plane_inference_v6(
            "I:/AnatomyTracker",
            expected_run_manifest_receipt_sha256="bad",
            expected_inference_source_sha256=source,
        )
    with pytest.raises(ValueError, match="receipt map"):
        inference_v6.load_arbitrary_plane_inference_v6(
            "I:/AnatomyTracker",
            expected_run_manifest_receipt_sha256="a" * 64,
            expected_inference_source_sha256=None,
        )
    with pytest.raises(ValueError, match="restricted to I"):
        inference_v6.load_arbitrary_plane_inference_v6(
            "C:/Windows",
            expected_run_manifest_receipt_sha256="a" * 64,
            expected_inference_source_sha256=source,
        )


def test_loader_rejects_self_consistent_but_externally_untrusted_source(monkeypatch):
    trusted = inference_v6._source_receipts()
    modified = dict(trusted)
    modified["training/arbitrary_plane_inference_v6.py"] = "b" * 64
    monkeypatch.setattr(inference_v6, "_source_receipts", lambda: modified)
    with pytest.raises(ValueError, match="trusted external"):
        inference_v6.load_arbitrary_plane_inference_v6(
            "I:/AnatomyTracker",
            expected_run_manifest_receipt_sha256="a" * 64,
            expected_inference_source_sha256=trusted,
        )
