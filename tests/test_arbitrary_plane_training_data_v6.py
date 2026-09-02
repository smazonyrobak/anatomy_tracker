import ast
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_training_data_v6 as data_v6
from training.arbitrary_plane_catalogue_runtime_v6 import (
    make_complete_catalogue_runtime_v6,
    verify_bound_complete_catalogue_batch_v6,
)
from training.arbitrary_plane_catalogue_v3 import (
    make_arbitrary_plane_catalogue_v3,
)
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
    full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import frame_to_quicknii_ouv
from training.arbitrary_plane_recurrent_model import (
    compose_antipodal_plane_frame_residual,
)


ATLAS_SHAPE = (3, 3, 3)
ORIGIN = (0.0, 0.0, 0.0)
SPACING = (25.0, 25.0, 25.0)
RASTER_SHAPE = (8, 8)


@pytest.fixture(scope="module")
def catalogue():
    return make_arbitrary_plane_catalogue_v3(
        np.ones(ATLAS_SHAPE, dtype=bool),
        ORIGIN,
        SPACING,
        normal_count=384,
        offset_count=16,
        roll_count=16,
        raster_shape_h_w=RASTER_SHAPE,
        raster_physical_span_y_x_um=(200.0, 200.0),
    )


@pytest.fixture(scope="module")
def runtime(catalogue):
    return make_complete_catalogue_runtime_v6(
        catalogue,
        expected_catalogue_receipt_sha256=catalogue["receipt_sha256"],
        device="cpu",
        dtype=torch.float32,
    )


def _outline(mask):
    eroded = mask.copy()
    eroded[1:] &= mask[:-1]
    eroded[:-1] &= mask[1:]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    eroded[[0, -1]] = False
    eroded[:, [0, -1]] = False
    return mask & ~eroded


def _quicknii_from_physical_state(state):
    center, frame, basis = full_frame_state_to_components(state.double())
    center_index = (center - torch.tensor(ORIGIN, dtype=torch.float64)) / 25.0 - 0.5
    return frame_to_quicknii_ouv(
        center_index, frame, basis / 25.0, ATLAS_SHAPE
    ).reshape(3, 3)


def _row(catalogue, index, mode, *, thickness, split="train"):
    height, width = RASTER_SHAPE
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    acquired = (0.1 + 0.8 * (x + width * y) / (height * width - 1)).astype(
        np.float32
    )
    if mode == "smart-brush-absent":
        mask = np.zeros(RASTER_SHAPE, dtype=bool)
        image = acquired.copy()
        available = 0.0
        black_exterior = None
    else:
        mask = np.zeros(RASTER_SHAPE, dtype=bool)
        mask[1:7, 1:7] = True
        if mode == "smart-brush-imperfect":
            mask[1, 1:3] = False
            mask[6, 6] = False
        image = np.where(mask, acquired, 0.0).astype(np.float32)
        available = 1.0
        black_exterior = True
    channels = np.stack(
        (
            image,
            _outline(mask).astype(np.float32),
            np.full(RASTER_SHAPE, available, dtype=np.float32),
        ),
        axis=-1,
    )
    valid = np.ones(RASTER_SHAPE, dtype=bool)
    valid[0, 0] = False
    abstention = np.zeros(RASTER_SHAPE, dtype=bool)
    abstention[0, 1] = True
    arrays = {
        "model_input_channels_float32": channels,
        "source_label_ground_truth_canvas_int64": np.ones(
            RASTER_SHAPE, dtype=np.int64
        ),
        "source_tissue_ground_truth_mask": np.ones(RASTER_SHAPE, dtype=bool),
        "target_ccf_coordinates_ap_dv_ml_um_float64": np.zeros(
            (*RASTER_SHAPE, 3), dtype=np.float64
        ),
        "target_valid_correspondence_mask": valid,
        "target_correspondence_weight_float32": np.full(
            RASTER_SHAPE, 0.75, dtype=np.float32
        ),
        "target_correspondence_abstention_mask": abstention,
        "truth_section_pullback_map_yx_px_float64": np.stack((y, x), axis=-1).astype(
            np.float64
        ),
        "truth_section_pullback_stationary_velocity_yx_px_float64": np.full(
            (*RASTER_SHAPE, 2), index * 1.0e-6, dtype=np.float64
        ),
        "truth_section_deformation_valid_mask": np.ones(
            RASTER_SHAPE, dtype=bool
        ),
    }
    schedule = psf_v4.make_finite_psf_schedule_v4(
        "finite_boxcar",
        thickness,
        thickness_selection_sha256=f"{index % 15 + 1:x}" * 64,
    )
    provenance_sha = f"{(index + 3) % 15 + 1:x}" * 64
    slab_receipt = f"{(index + 5) % 15 + 1:x}" * 64
    synthetic_id = acquisition_v2._payload_sha256(
        {"synthetic-realization": index, "mode": mode}
    )
    upstream = {
        "slab_observation_id": f"{(index + 7) % 15 + 1:x}" * 64,
        "centre_plane_targets_receipt_sha256": f"{(index + 9) % 15 + 1:x}" * 64,
        "slab_observation_v4_receipt_sha256": slab_receipt,
        "finite_psf_sha256": schedule["finite_psf_sha256"],
        "finite_psf_capability_sha256": schedule[
            "finite_psf_capability_sha256"
        ],
        "selected_synthetic_provenance_sha256": provenance_sha,
        "selected_synthetic_lineage_sha256": f"{(index + 11) % 15 + 1:x}" * 64,
        "finite_parent_provenance_sha256": f"{(index + 12) % 15 + 1:x}" * 64,
        "finite_slab_adapter_receipt_sha256": f"{(index + 13) % 15 + 1:x}" * 64,
        "selected_input_mask_receipt": acquisition_v2._array_receipt(mask),
        "selected_black_exterior_exact": black_exterior,
        "automatic_segmentation_dependency": False,
        "support_supervision_contract": {
            "point_pose_supervision_weight": 1.0,
            "dense_deformation_supervision_weight": 1.0,
        },
        "prior_model_weight_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
    }
    row = {
        "schema_version": psf_v4.TRAINING_ROW_V4_SCHEMA,
        "source_observation_receipt_sha256": f"{(index + 2) % 15 + 1:x}" * 64,
        "lineage": {
            "animal_id": f"animal-{index}",
            "specimen_id": f"specimen-{index}",
            "experiment_id": f"experiment-{index}",
            "synthetic_animal_id": f"synthetic-animal-{index}",
            "section_id": f"section-{index}",
            "split": split,
        },
        "upstream_reference": upstream,
        "numeric_rng_provenance": {"sample_index": index},
        "rng_sources": {},
        "selected_mode": mode,
        "selected_descendant_id": synthetic_id,
        "deformation_pose_gauge_reference": {},
        "reflection_state": "none",
        "reflection_representation_index": 0,
        "reflection_representation_affine_xy_float64": np.eye(3).tolist(),
        "canonical_effective_quicknii_ouv_float64": _quicknii_from_physical_state(
            catalogue["tensors"]["cell_states"][0, index]
        ).tolist(),
        "observed_effective_quicknii_ouv_float64": _quicknii_from_physical_state(
            catalogue["tensors"]["cell_states"][0, index]
        ).tolist(),
        "proper_physical_pose_unchanged": _quicknii_from_physical_state(
            catalogue["tensors"]["cell_states"][0, index]
        ).tolist(),
        "prior_model_dependencies": [],
        "prior_feature_dependencies": [],
        "prior_pseudolabel_dependencies": [],
        "reflection_transform_id": acquisition_v2._payload_sha256(
            {"reflection": index}
        ),
        "reflection_realization_id": acquisition_v2._payload_sha256(
            {"reflection-realization": index}
        ),
        "paired_view_group_id": acquisition_v2._payload_sha256(
            {"paired-view": index}
        ),
        "synthetic_realization_id": synthetic_id,
        "paired_mode_reflected_receipts": {},
        "arrays": arrays,
        "array_receipts": {
            name: acquisition_v2._array_receipt(value)
            for name, value in arrays.items()
        },
        "finite_psf_contract": {**schedule, "slab_observation_v4_receipt_sha256": slab_receipt},
    }
    row["training_row_id"] = acquisition_v2._payload_sha256(
        {
            "domain": psf_v4.TRAINING_ROW_V4_SCHEMA,
            "synthetic_realization_id": synthetic_id,
            "array_receipts": row["array_receipts"],
            "finite_psf_sha256": schedule["finite_psf_sha256"],
            "slab_observation_v4_receipt_sha256": slab_receipt,
        }
    )
    row["receipt_sha256"] = acquisition_v2._payload_sha256(
        psf_v4.training_row_receipt_v4(row)
    )
    return row, mask, acquired


def _frozen_payload(rows, manifest_receipt="e" * 64):
    payload = {
        "schema_version": data_v6.FROZEN_ROWS_V6_SCHEMA,
        "training_data_manifest_receipt_sha256": manifest_receipt,
        "cache_manifest_receipt_sha256": manifest_receipt,
        "generator_binding_receipt_sha256": "b" * 64,
        "generation_lineage_sha256": "c" * 64,
        "row_indices": list(range(len(rows))),
        "training_row_ids": [row["training_row_id"] for row in rows],
        "training_row_receipts_sha256": [row["receipt_sha256"] for row in rows],
        "rows": rows,
    }
    payload["selection_receipt_sha256"] = acquisition_v2._payload_sha256(
        data_v6._frozen_rows_receipt(payload)
    )
    return payload


def test_complete_batch_preserves_modes_identities_receipts_psf_and_dense_truth(
    catalogue, runtime
):
    indices = (137, 40_015, 92_111)
    modes = (
        "smart-brush-absent",
        "smart-brush-accurate",
        "smart-brush-imperfect",
    )
    made = [
        _row(catalogue, index, mode, thickness=thickness)
        for index, mode, thickness in zip(indices, modes, (25.0, 50.0, 100.0))
    ]
    rows = [item[0] for item in made]
    batch = data_v6.model_ready_rows_v6(
        _frozen_payload(rows),
        catalogue,
        runtime,
        torch.ones(2, *ATLAS_SHAPE),
        origin_ap_dv_ml_um=ORIGIN,
        voxel_size_ap_dv_ml_um=SPACING,
        finite_psf_capability=psf_v4.finite_psf_model_capability_v4(),
        expected_training_data_manifest_receipt_sha256="e" * 64,
    )

    assert batch["input_mode"] == ["raw", "black-exterior", "imperfect-mask"]
    assert batch["outline_available"].tolist() == [0.0, 1.0, 1.0]
    assert np.array_equal(batch["image"][0, 0].numpy(), made[0][2])
    for row_index in (1, 2):
        assert not np.any(batch["image"][row_index, 0].numpy()[~made[row_index][1]])
    assert torch.equal(batch["truth_catalogue_index"], torch.tensor(indices))
    assert torch.equal(batch["truth_catalogue_cell_id"], torch.tensor(indices))
    assert torch.allclose(
        batch["truth_catalogue_recomposed_state_float64"],
        batch["truth_catalogue_aligned_state_float64"],
        atol=1.0e-10,
        rtol=0.0,
    )
    assert batch["axial_offsets_um"].shape == (3, 9)
    assert batch["axial_weights"].shape == (3, 9)
    assert batch["truth_stationary_velocity_yx_px"].shape == (3, 2, 8, 8)
    assert batch["deformation_weight"].shape == (3, 1, 8, 8)
    assert all(item["exact_minimum_tie_count"] == 1 for item in batch["catalogue_truth_mapping_audit"])
    for row, provenance, receipts in zip(rows, batch["provenance"], batch["row_receipts"]):
        assert all(provenance[name] == row["lineage"][name] for name in data_v6.LINEAGE_KEYS_V6)
        assert provenance["training_row_id"] == row["training_row_id"]
        assert provenance["training_row_receipt_sha256"] == row["receipt_sha256"]
        assert provenance["provenance_sha256"] == row["upstream_reference"]["selected_synthetic_provenance_sha256"]
        assert receipts["finite_psf_sha256"] == row["finite_psf_contract"]["finite_psf_sha256"]
    bound = verify_bound_complete_catalogue_batch_v6(
        batch["catalogue_batch"], expected_runtime=runtime
    )
    assert bound["cell_states"].shape == (3, 98_304, 12)


def test_model_batch_rejects_bare_rows_and_wrong_run_manifest(catalogue, runtime):
    row = _row(catalogue, 137, "smart-brush-absent", thickness=25.0)[0]
    kwargs = {
        "origin_ap_dv_ml_um": ORIGIN,
        "voxel_size_ap_dv_ml_um": SPACING,
        "finite_psf_capability": psf_v4.finite_psf_model_capability_v4(),
        "expected_training_data_manifest_receipt_sha256": "e" * 64,
    }
    with pytest.raises(ValueError, match="authenticated frozen-row selection"):
        data_v6.model_ready_rows_v6(
            [row], catalogue, runtime, torch.ones(2, *ATLAS_SHAPE), **kwargs
        )
    with pytest.raises(ValueError, match="run-bound data manifest"):
        data_v6.model_ready_rows_v6(
            _frozen_payload([row], manifest_receipt="d" * 64),
            catalogue,
            runtime,
            torch.ones(2, *ATLAS_SHAPE),
            **kwargs,
        )


def test_model_batch_rejects_held_out_development_rows(catalogue, runtime):
    row = _row(
        catalogue,
        137,
        "smart-brush-absent",
        thickness=25.0,
        split="development",
    )[0]
    with pytest.raises(ValueError, match="exact train-split"):
        data_v6.model_ready_rows_v6(
            _frozen_payload([row]),
            catalogue,
            runtime,
            torch.ones(2, *ATLAS_SHAPE),
            origin_ap_dv_ml_um=ORIGIN,
            voxel_size_ap_dv_ml_um=SPACING,
            finite_psf_capability=psf_v4.finite_psf_model_capability_v4(),
            expected_training_data_manifest_receipt_sha256="e" * 64,
        )


def test_arbitrary_orientations_and_antipodal_lifts_replay_the_full_residual(
    catalogue,
):
    indices = torch.tensor((571, 44_019, 97_123))
    base = catalogue["tensors"]["cell_states"][0, indices].double()
    origin = torch.tensor(
        catalogue["support_geometry"]["support_origin_ap_dv_ml_um"],
        dtype=torch.float64,
    )
    residual = torch.tensor(
        (
            (0.003, -0.002, 0.10, 0.005, 0.20, -0.30, 0.01, -0.02, 0.015),
            (-0.002, 0.004, -0.20, -0.006, -0.40, 0.10, -0.02, 0.01, -0.010),
            (0.001, 0.002, 0.15, 0.004, 0.30, 0.25, 0.02, 0.01, 0.005),
        ),
        dtype=torch.float64,
    )
    truth = compose_antipodal_plane_frame_residual(base, residual, origin)
    center, frame, basis = full_frame_state_to_components(truth)
    frame_sign = torch.tensor((-1.0, 1.0, -1.0), dtype=torch.float64)
    basis_sign = torch.tensor((-1.0, 1.0), dtype=torch.float64)
    truth[1] = full_frame_state_from_components(
        center[1],
        frame[1] * frame_sign[None],
        basis_sign[:, None] * basis[1] * basis_sign[None, :],
    )
    targets = data_v6.catalogue_truth_targets_v6(truth, catalogue)

    assert torch.equal(targets["truth_catalogue_index"], indices)
    assert torch.allclose(
        targets["truth_catalogue_residual_float64"], residual, atol=1.0e-10, rtol=0.0
    )
    assert targets["catalogue_truth_mapping_audit"][1]["antipodal_truth_normal_sign"] == -1
    for audit in targets["catalogue_truth_mapping_audit"]:
        assert audit["recomposition"]["center_max_abs_error_um"] <= 1.0e-7
        assert audit["recomposition"]["frame_max_abs_error"] <= 1.0e-10
        assert audit["recomposition"]["basis_max_abs_error_um"] <= 1.0e-7


def test_frozen_cache_adapter_requires_trusted_frozen_manifest_and_preserves_rows(
    monkeypatch, catalogue
):
    row = _row(catalogue, 19, "smart-brush-absent", thickness=25.0)[0]
    manifest_receipt = "a" * 64
    expected = _frozen_payload([row], manifest_receipt=manifest_receipt)
    called = []

    def load(path, indices=None, *, expected_manifest_receipt_sha256):
        called.append((path, indices, expected_manifest_receipt_sha256))
        return expected

    monkeypatch.setattr(data_v6.finite_rows_v6, "load_frozen_training_rows_v6", load)
    payload = data_v6.load_frozen_training_rows_v6(
        r"I:\AnatomyTracker\pytest-v6-data-adapter",
        expected_manifest_receipt_sha256=manifest_receipt,
    )
    assert called == [
        (r"I:\AnatomyTracker\pytest-v6-data-adapter", None, manifest_receipt)
    ]
    assert payload["rows"][0] is row
    assert payload["training_data_manifest_receipt_sha256"] == manifest_receipt
    assert payload["cache_manifest_receipt_sha256"] == manifest_receipt
    assert len(payload["selection_receipt_sha256"]) == 64
    assert payload["training_row_ids"] == [row["training_row_id"]]
    assert payload["training_row_receipts_sha256"] == [row["receipt_sha256"]]


def test_tampered_mode_provenance_and_prior_sources_are_rejected(catalogue, runtime):
    row = _row(catalogue, 41, "smart-brush-accurate", thickness=25.0)[0]
    cases = []
    changed = copy.deepcopy(row)
    changed["lineage"]["animal_id"] = ""
    cases.append(changed)
    changed = copy.deepcopy(row)
    changed["upstream_reference"]["selected_black_exterior_exact"] = False
    cases.append(changed)
    changed = copy.deepcopy(row)
    changed["upstream_reference"]["candidate_bank_receipt_sha256"] = "d" * 64
    cases.append(changed)
    changed = copy.deepcopy(row)
    changed["upstream_reference"]["training_bank_receipt_sha256"] = "d" * 64
    cases.append(changed)
    changed = copy.deepcopy(row)
    changed["upstream_reference"]["prior-model-weight-dependencies"] = ["d" * 64]
    cases.append(changed)
    for changed in cases:
        changed["receipt_sha256"] = acquisition_v2._payload_sha256(
            psf_v4.training_row_receipt_v4(changed)
        )
        with pytest.raises(ValueError):
            data_v6.model_ready_rows_v6(
                _frozen_payload([changed]),
                catalogue,
                runtime,
                torch.ones(2, *ATLAS_SHAPE),
                origin_ap_dv_ml_um=ORIGIN,
                voxel_size_ap_dv_ml_um=SPACING,
                finite_psf_capability=psf_v4.finite_psf_model_capability_v4(),
                expected_training_data_manifest_receipt_sha256="e" * 64,
            )


def test_missing_mapping_geometry_assigns_no_label(catalogue):
    changed = dict(catalogue)
    changed["arrays"] = dict(catalogue["arrays"])
    changed["array_receipts"] = dict(catalogue["array_receipts"])
    changed["arrays"].pop("normal_offset_table_um_float64")
    changed["array_receipts"].pop("normal_offset_table_um_float64")
    with pytest.raises(ValueError, match="no truth label was assigned"):
        data_v6.catalogue_truth_targets_v6(
            catalogue["tensors"]["cell_states"][0, 0], changed
        )


def test_frozen_loader_rejects_a_c_drive_source_without_writing():
    with pytest.raises(ValueError, match="only on I"):
        data_v6.load_frozen_training_rows_v6(
            r"C:\forbidden-v6-row-cache",
            expected_manifest_receipt_sha256="a" * 64,
        )


def test_data_adapter_has_no_forbidden_import_path():
    tree = ast.parse(Path(data_v6.__file__).read_text(encoding="utf-8"))
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert not any("candidate_bank" in name or "legacy" in name for name in imports)


def test_data_adapter_import_does_not_load_prior_retrieval_or_training_modules():
    code = """
import sys
import training.arbitrary_plane_training_data_v6

forbidden = [
    name for name in sys.modules
    if any(fragment in name for fragment in (
        "candidate_bank", "training_bank", "staged_training",
        "arbitrary_plane_inference_v3", "arbitrary_plane_joint_model",
        "arbitrary_plane_legacy_chain_v3", "arbitrary_plane_psf_v4",
        "arbitrary_plane_row_cache_v4", "arbitrary_plane_batch_v3",
    ))
]
assert not forbidden, forbidden
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
