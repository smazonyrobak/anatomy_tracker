import numpy as np
import pytest
import torch

import training.arbitrary_plane_batch_v3 as batch_v3
import training.arbitrary_plane_catalogue_v3 as catalogue_v3
from training.arbitrary_plane_geometry import frame_to_quicknii_ouv


def test_quicknii_pose_converts_to_physical_micrometre_state():
    centre = torch.tensor((8.0, 6.0, 7.0), dtype=torch.float64)
    frame = torch.eye(3, dtype=torch.float64)
    basis = torch.diag(torch.tensor((4.0, 3.0), dtype=torch.float64))
    quicknii = frame_to_quicknii_ouv(centre, frame, basis, (20, 18, 22)).reshape(3, 3)
    state = batch_v3.physical_state_from_quicknii_ouv_v3(
        quicknii, (20, 18, 22), (100.0, 200.0, 300.0), (25.0, 25.0, 25.0)
    )
    assert torch.allclose(
        state[:3], torch.tensor((312.5, 362.5, 487.5), dtype=torch.float64), atol=1e-10
    )
    assert torch.allclose(
        state[3:9],
        torch.tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=torch.float64),
    )
    assert torch.allclose(
        torch.exp(state[9:11]), torch.tensor((100.0, 75.0), dtype=torch.float64)
    )


def test_row_adapter_uses_correspondence_truth_not_smart_brush_as_loss_gate():
    height, width = 4, 5
    quicknii = np.array(((0.0, 20.0, 18.0), (0.0, -4.0, 0.0), (0.0, 0.0, -3.0)))
    valid = np.ones((height, width), dtype=bool)
    valid[0, 0] = False
    abstention = np.zeros_like(valid)
    abstention[0, 1] = True
    row = {
        "canonical_effective_quicknii_ouv_float64": quicknii,
        "training_row_id": "row",
        "synthetic_realization_id": "realization",
        "selected_mode": "absent",
        "reflection_state": "none",
        "lineage": {"animal_id": "a", "specimen_id": "s", "experiment_id": "e"},
        "arrays": {
            "model_input_channels_float32": np.zeros((height, width, 3), np.float32),
            "truth_section_pullback_stationary_velocity_yx_px_float64": np.zeros(
                (height, width, 2), np.float64
            ),
            "truth_section_pullback_map_yx_px_float64": np.zeros(
                (height, width, 2), np.float64
            ),
            "truth_section_deformation_valid_mask": np.ones_like(valid),
            "target_valid_correspondence_mask": valid,
            "target_correspondence_abstention_mask": abstention,
            "target_correspondence_weight_float32": np.ones_like(valid, np.float32),
        },
    }
    converted = batch_v3.training_row_to_tensors_v3(
        row,
        atlas_shape_ap_dv_ml=(20, 18, 22),
        origin_ap_dv_ml_um=(0.0, 0.0, 0.0),
        voxel_size_ap_dv_ml_um=(25.0, 25.0, 25.0),
    )
    weight = converted["tensors"]["deformation_weight"]
    assert weight.shape == (1, 1, height, width)
    assert weight[0, 0, 0, 0] == 0.0
    assert weight[0, 0, 0, 1] == 0.0
    assert weight.sum() == height * width - 2
    assert converted["tensors"]["pose_supervision_weight"].item() == 1.0
    assert (
        converted["tensors"]["dense_deformation_supervision_weight"].item()
        == 1.0
    )

    row["upstream_reference"] = {
        "support_supervision_contract": {
            "point_pose_supervision_weight": 0.0,
            "dense_deformation_supervision_weight": 0.0,
        }
    }
    censored = batch_v3.training_row_to_tensors_v3(
        row,
        atlas_shape_ap_dv_ml=(20, 18, 22),
        origin_ap_dv_ml_um=(0.0, 0.0, 0.0),
        voxel_size_ap_dv_ml_um=(25.0, 25.0, 25.0),
    )
    assert censored["tensors"]["pose_supervision_weight"].item() == 0.0
    assert (
        censored["tensors"]["dense_deformation_supervision_weight"].item()
        == 0.0
    )
    assert torch.equal(censored["tensors"]["deformation_weight"], weight)


def test_exact_catalogue_state_assigns_to_its_own_cell_and_topk_position():
    z, y, x = np.indices((12, 10, 11))
    support = ((z - 5.5) / 5.0) ** 2 + ((y - 4.5) / 4.0) ** 2 + ((x - 5.0) / 4.5) ** 2 <= 1.0
    catalogue = catalogue_v3.make_arbitrary_plane_catalogue_v3(
        support,
        (0.0, 0.0, 0.0),
        (25.0, 25.0, 25.0),
        normal_count=12,
        offset_count=5,
        roll_count=6,
        raster_shape_h_w=(32, 40),
        raster_physical_span_y_x_um=(800.0, 1000.0),
    )
    target = torch.tensor([73, 214])
    truth = catalogue["tensors"]["cell_states"][0, target]
    assert torch.equal(batch_v3.nearest_catalogue_cell_v3(truth, catalogue), target)
    topk = torch.tensor([[5, 73, 9], [214, 7, 8]])
    assert torch.equal(
        batch_v3.truth_index_within_topk_v3(topk, target), torch.tensor([1, 0])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_nearest_catalogue_cell_aligns_catalogue_geometry_to_cuda_truth():
    z, y, x = np.indices((12, 10, 11))
    support = ((z - 5.5) / 5.0) ** 2 + ((y - 4.5) / 4.0) ** 2 + ((x - 5.0) / 4.5) ** 2 <= 1.0
    catalogue = catalogue_v3.make_arbitrary_plane_catalogue_v3(
        support,
        (0.0, 0.0, 0.0),
        (25.0, 25.0, 25.0),
        normal_count=12,
        offset_count=5,
        roll_count=6,
        raster_shape_h_w=(32, 40),
        raster_physical_span_y_x_um=(800.0, 1000.0),
    )
    catalogue["tensors"] = {
        name: value.cuda() for name, value in catalogue["tensors"].items()
    }
    target = torch.tensor([73, 214], device="cuda")
    truth = catalogue["tensors"]["cell_states"][0, target]
    observed = batch_v3.nearest_catalogue_cell_v3(truth, catalogue)
    assert observed.is_cuda
    assert torch.equal(observed, target)
