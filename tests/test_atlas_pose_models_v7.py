from io import BytesIO

import numpy as np
import pytest
import torch
import onnxruntime as ort

from training.acquire_allen_s2p import quicknii_to_tracker_pose
from training.atlas_pose_models_v7 import (
    AP_BIN_COUNT,
    AP_MAX_UM,
    AP_MIN_UM,
    AP_STEP_UM,
    BACKBONES,
    TILT_BIN_COUNT,
    TILT_MAX_DEG,
    TILT_MIN_DEG,
    TILT_STEP_DEG,
    AtlasPoseV7,
    AtlasPoseV7Export,
    BinnedPoseHead,
    DirectPoseHead,
    OUVPoseHead,
    SpatialPyramidPoseFeatures,
    ap_bin_centers,
    ap_bin_edges,
    atlas_pose_v7_loss,
    binned_pose_loss,
    decode_binned_prediction,
    direct_pose_loss,
    encode_binned_target,
    pose_to_quicknii_ouv,
    quicknii_ouv_to_pose,
    tilt_bin_centers,
    tilt_bin_edges,
)


def test_physical_bin_centers_edges_and_label_encoding_are_exact():
    ap_centers = ap_bin_centers(dtype=torch.float64)
    ap_edges = ap_bin_edges(dtype=torch.float64)
    tilt_centers = tilt_bin_centers(dtype=torch.float64)
    tilt_edges = tilt_bin_edges(dtype=torch.float64)

    assert len(ap_centers) == AP_BIN_COUNT == 201
    assert ap_centers[0] == AP_MIN_UM
    assert ap_centers[-1] == AP_MAX_UM
    assert torch.all(ap_centers.diff() == AP_STEP_UM)
    assert ap_edges[0] == AP_MIN_UM - AP_STEP_UM / 2.0
    assert ap_edges[-1] == AP_MAX_UM + AP_STEP_UM / 2.0
    assert torch.all(ap_edges.diff() == AP_STEP_UM)

    assert len(tilt_centers) == TILT_BIN_COUNT == 71
    assert tilt_centers[0] == TILT_MIN_DEG
    assert tilt_centers[-1] == TILT_MAX_DEG
    assert torch.all(tilt_centers.diff() == TILT_STEP_DEG)
    assert tilt_edges[0] == TILT_MIN_DEG - TILT_STEP_DEG / 2.0
    assert tilt_edges[-1] == TILT_MAX_DEG + TILT_STEP_DEG / 2.0

    target = torch.tensor([-4500.0, -4478.125, 0.0, 500.0], dtype=torch.float64)
    index, residual = encode_binned_target(target, ap_centers, AP_STEP_UM)
    assert index.tolist() == [0, 1, 180, 200]
    assert torch.allclose(residual, torch.tensor([0.0, -0.25, 0.0, 0.0], dtype=torch.float64))


def test_expected_value_decoder_and_residual_are_physical():
    centers = ap_bin_centers()
    logits = torch.full((2, AP_BIN_COUNT), -100.0)
    residuals = torch.zeros_like(logits)
    logits[0, 0] = 100.0
    logits[1, 180] = 100.0
    residuals[1, 180] = torch.atanh(torch.tensor(0.5))
    decoded = decode_binned_prediction(logits, residuals, centers, AP_STEP_UM)
    assert torch.allclose(decoded, torch.tensor([-4500.0, 6.25]))


def test_binned_heads_losses_and_auxiliary_head_have_gradients():
    model = AtlasPoseV7(pretrained=False)
    image = torch.rand(2, 3, 64, 64)
    target = torch.tensor([[-1200.0, 8.0, -4.0], [-2400.0, -12.0, 9.0]])
    orientation_target = torch.tensor([0.0, 1.0])
    outputs = model.training_outputs(image)

    assert outputs["pose"].shape == (2, 3)
    assert outputs["ap_logits"].shape == (2, AP_BIN_COUNT)
    assert outputs["lr_logits"].shape == outputs["dv_logits"].shape == (2, TILT_BIN_COUNT)
    assert outputs["pooled_features"].shape == (2, 512)
    assert outputs["anatomy_logits"].shape == (2, 9, 64, 64)
    assert outputs["orientation_inverted_logit"].shape == (2,)

    loss = atlas_pose_v7_loss(outputs, target, orientation_target) + outputs["anatomy_logits"].square().mean()
    loss.backward()
    assert next(model.encoder.backbone.parameters()).grad is not None
    assert model.pose_head.ap.weight.grad is not None
    assert model.orientation_head.weight.grad is not None
    assert model.anatomy_head.decoder[0].weight.grad is not None


def test_spatial_pyramid_distinguishes_layouts_with_identical_global_means():
    torch.manual_seed(7)
    pool = SpatialPyramidPoseFeatures(8).eval()
    feature_map = torch.zeros(1, 8, 7, 7)
    feature_map[:, :, :3, :3] = 1.0
    moved = torch.flip(feature_map, dims=(-2, -1))
    assert torch.allclose(feature_map.mean(dim=(-2, -1)), moved.mean(dim=(-2, -1)))
    assert not torch.allclose(pool(feature_map), pool(moved))


def test_all_controlled_backbones_retain_a_spatial_feature_map():
    assert BACKBONES == {
        "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "maxvit_tiny": "maxvit_tiny_rw_224.sw_in1k",
        "xception": "legacy_xception.tf_in1k",
    }
    for architecture in BACKBONES:
        model = AtlasPoseV7(architecture=architecture, pretrained=False)
        feature_map = model.encoder(torch.zeros(1, 3, 299, 299))
        assert feature_map.ndim == 4
        assert min(feature_map.shape[-2:]) >= 7


def test_ouv_pose_round_trip_matches_existing_tracker_convention():
    pose = torch.tensor(
        [[500.0, -35.0, 35.0], [0.0, 0.0, 0.0], [-1375.0, 12.0, -7.0], [-4500.0, 35.0, -35.0]],
        dtype=torch.float64,
    )
    ouv = pose_to_quicknii_ouv(pose)
    recovered = quicknii_ouv_to_pose(ouv)
    assert torch.allclose(recovered, pose, atol=1e-9, rtol=0.0)
    for expected, plane in zip(pose.numpy(), ouv.numpy()):
        assert np.allclose(quicknii_to_tracker_pose(plane), expected, atol=1e-9, rtol=0.0)


def test_ouv_ablation_head_returns_physical_pose_and_nine_coordinates():
    head = OUVPoseHead(16)
    features = torch.zeros(3, 16)
    outputs = head.components(features)
    assert outputs["pose"].shape == (3, 3)
    assert outputs["ouv"].shape == (3, 9)
    assert torch.allclose(outputs["pose"], torch.zeros(3, 3), atol=1e-5)


def test_direct_regression_baseline_decodes_physical_pose_and_trains():
    head = DirectPoseHead(16)
    features = torch.randn(3, 16, requires_grad=True)
    outputs = head.components(features)
    target = torch.tensor([[-4500.0, -35.0, 35.0], [-2000.0, 0.0, 0.0], [500.0, 35.0, -35.0]])
    assert outputs["pose"].shape == (3, 3)
    loss = direct_pose_loss(outputs["normalized_pose"], target)
    loss.backward()
    assert head.normalized_pose.weight.grad is not None
    assert features.grad is not None


def test_orientation_output_converts_image_frame_tilts_to_physical_tilts():
    image_frame_pose = torch.tensor([[-1200.0, 8.0, -3.0], [-1800.0, 8.0, -3.0]])
    physical = AtlasPoseV7.physical_pose(image_frame_pose, torch.tensor([-2.0, 2.0]))
    assert torch.equal(physical, torch.tensor([[-1200.0, 8.0, -3.0], [-1800.0, -8.0, 3.0]]))


def test_cpu_inference_exports_exact_runtime_contract_to_onnx():
    pytest.importorskip("onnx")
    model = AtlasPoseV7Export(AtlasPoseV7(pretrained=False)).eval()
    image = torch.zeros(1, 3, 299, 299)
    pose, orientation = model(image)
    assert pose.shape == (1, 3)
    assert orientation.shape == (1,)
    assert torch.isfinite(pose).all()
    assert torch.isfinite(orientation).all()

    stream = BytesIO()
    torch.onnx.export(
        model,
        image,
        stream,
        input_names=["images"],
        output_names=["pose_ap_um_lr_deg_dv_deg", "orientation_inverted_logit"],
        dynamic_axes={
            "images": {0: "batch"},
            "pose_ap_um_lr_deg_dv_deg": {0: "batch"},
            "orientation_inverted_logit": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(stream.getvalue(), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "images"
    assert session.get_inputs()[0].shape == ["batch", 3, 299, 299]
    assert [output.name for output in session.get_outputs()] == [
        "pose_ap_um_lr_deg_dv_deg",
        "orientation_inverted_logit",
    ]
    assert session.get_outputs()[0].shape == ["batch", 3]
    assert session.get_outputs()[1].shape == ["batch"]
    onnx_pose, onnx_orientation = session.run(None, {"images": image.numpy()})
    assert onnx_pose.shape == (1, 3)
    assert onnx_orientation.shape == (1,)
