import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

from training.train_atlas_pose_v7 import (
    ABLATIONS_20K,
    SCHEDULE,
    animal_bootstrap_comparison,
    cosine_learning_rate,
    ema_state,
    ensure_fixed_manifest,
    ensure_paired_manifest,
    export_onnx,
    file_sha256,
    manifest_sha256,
    paired_invariance,
    promote_export,
    registered_report,
    renderer_variant,
    registered_style,
    stratified_pose_metrics,
    training_objective,
    update_ema,
    validation_selection_metric,
)
from training.synthetic_atlas import APPEARANCE_MANIFEST_KEYS


class ToyPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 3, padding=1)
        self.features = nn.Linear(8, 16)
        self.pose_head = nn.Linear(16, 3)
        self.orientation_head = nn.Linear(16, 1)
        self.anatomy_head = nn.Conv2d(8, 9, 1)
        self.register_buffer("center", torch.tensor((-2000.0, 0.0, 0.0)))
        self.register_buffer("scale", torch.tensor((2500.0, 20.0, 20.0)))

    def training_outputs(self, image):
        feature_map = torch.relu(self.encoder(image))
        features = self.features(feature_map.mean((-2, -1)))
        normalized_pose = self.pose_head(features)
        image_frame_pose = normalized_pose * self.scale + self.center
        orientation = self.orientation_head(features).squeeze(1)
        sign = torch.where(orientation > 0.0, -torch.ones_like(orientation), torch.ones_like(orientation))
        pose = torch.cat((image_frame_pose[:, :1], image_frame_pose[:, 1:] * sign[:, None]), 1)
        return {
            "pose": pose,
            "image_frame_pose": image_frame_pose,
            "normalized_pose": normalized_pose,
            "orientation_inverted_logit": orientation,
            "pooled_features": features,
            "anatomy_logits": self.anatomy_head(feature_map),
        }

    def forward_with_orientation(self, image):
        output = self.training_outputs(image)
        return output["pose"], output["orientation_inverted_logit"]


def validation_rows(split="validation"):
    return [
        {
            "split": split,
            "specimen_id": specimen,
            "experiment_id": specimen,
            "section_image_id": specimen * 10 + section,
            "product": str(5 + specimen % 2 * 3),
            "target_ap": -1000.0 - section * 100.0,
            "target_lr": float(specimen),
            "target_dv": -float(section),
            "prediction_ap": -980.0 - section * 100.0,
            "prediction_lr": float(specimen) + 0.5,
            "prediction_dv": -float(section) - 0.25,
        }
        for specimen in (1, 2)
        for section in (0, 1)
    ]


def test_schedule_encodes_controlled_scientific_stages():
    assert SCHEDULE == {
        "ablation_20k": 20_000,
        "surviving_heads_100k": 100_000,
        "backbones_100k": 100_000,
        "final_unique_views": 1_000_000,
    }
    assert {config["head"] for config in ABLATIONS_20K.values()} == {"direct", "binned", "ouv"}
    assert {config["renderer"] for config in ABLATIONS_20K.values()} == {"minimal", "v7"}
    assert any(config["consistency"] == 0.0 for config in ABLATIONS_20K.values())
    assert any(config["anatomy"] == 0.0 for config in ABLATIONS_20K.values())


def test_fixed_latents_and_paired_views_are_reproducible_without_image_cache(tmp_path):
    first, path = ensure_fixed_manifest(tmp_path, "train", 32, 17)
    second, repeated_path = ensure_fixed_manifest(tmp_path, "train", 32, 17)
    paired, paired_path = ensure_paired_manifest(tmp_path, first, "train", 23)
    assert path == repeated_path
    assert manifest_sha256(first) == manifest_sha256(second)
    assert paired_path.is_file()
    for key in first.keys() - set(APPEARANCE_MANIFEST_KEYS):
        assert np.array_equal(first[key], paired[key])
    minimal = renderer_variant(first, "minimal")
    assert not minimal["warp"].any()
    assert not minimal["flaw_mask"].any()
    assert not minimal["sensor_enabled"].any()
    assert not minimal["occlusion_type"].any()
    assert not list(tmp_path.rglob("*.npy"))


def test_paired_objective_updates_pose_orientation_consistency_and_anatomy():
    torch.manual_seed(3)
    model = ToyPoseModel()
    images = torch.rand(4, 2, 3, 16, 16)
    pose = torch.tensor([[-1200.0, 4.0, -2.0], [-1800.0, -5.0, 7.0], [-2400.0, 8.0, 1.0], [-3000.0, -3.0, -6.0]])
    orientation = torch.tensor([0.0, 1.0, 0.0, 1.0])
    anatomy = torch.randint(0, 9, (4, 16, 16))
    loss, components = training_objective(model, images, pose, orientation, anatomy, 0.15, 0.20)
    loss.backward()
    assert set(components) == {"total", "pose", "feature_consistency", "prediction_consistency", "anatomy"}
    assert components["feature_consistency"] > 0.0
    assert components["prediction_consistency"] > 0.0
    assert model.pose_head.weight.grad is not None
    assert model.orientation_head.weight.grad is not None
    assert model.anatomy_head.weight.grad is not None
    assert model.encoder.weight.grad is not None


def test_metrics_cover_cohorts_bands_bias_calibration_and_registered_groups():
    target = np.asarray([
        [-4450.0, 2.0, -1.0], [-3900.0, 8.0, 4.0], [-2400.0, 18.0, -3.0], [-1200.0, 28.0, -32.0]
    ])
    prediction = target + np.asarray((25.0, 1.0, -0.5))
    report = stratified_pose_metrics(target, prediction, np.arange(4))
    assert set(report["artifact_severity"]) == {"clean", "mild", "moderate", "severe"}
    assert len(report["ap_500um_bands"]) == 4
    assert len(report["tilt_bands"]) == 4
    assert report["overall"]["bias"] == pytest.approx([25.0, 1.0, -0.5])
    assert report["overall"]["calibration_slope"] == pytest.approx([1.0, 1.0, 1.0])

    rows = validation_rows()
    registered = registered_report(rows)
    assert set(registered["per_specimen"]) == {"1", "2"}
    assert set(registered["per_product"]) == {"5", "8"}
    assert validation_selection_metric(rows) > 0.0
    for forbidden in ("test", "sealed_deepslice_s2p"):
        with pytest.raises(RuntimeError, match="validation"):
            validation_selection_metric(validation_rows(forbidden))


def test_paired_invariance_and_animal_bootstrap_resample_animals():
    target = np.zeros((6, 3))
    first = np.ones((6, 3))
    second = np.full((6, 3), 2.0)
    invariant = paired_invariance(target, first, second)
    assert invariant["mean_absolute_prediction_shift"] == [1.0, 1.0, 1.0]
    specimens = np.repeat((10, 20, 30), 2)
    comparison = animal_bootstrap_comparison(target, first, second, specimens, iterations=1000, seed=7)
    assert comparison["animal_count"] == 3
    assert comparison["candidate_minus_reference"] < 0.0
    assert comparison["probability_candidate_better"] == 1.0
    assert set(comparison["components"]) == {"ap_um", "lr_deg", "dv_deg"}
    assert comparison["components"]["ap_um"]["candidate_minus_reference"] == -1.0


def test_registered_style_is_deterministic_local_and_geometry_preserving():
    image = np.tile(np.arange(192, dtype=np.uint8), (128, 1))
    first = registered_style(image, np.random.default_rng(18))
    second = registered_style(image, np.random.default_rng(18))
    assert first.shape == image.shape
    assert first.dtype == image.dtype
    assert np.array_equal(first, second)
    design = np.column_stack((image.ravel(), np.ones(image.size)))
    residual = first.ravel() - design @ np.linalg.lstsq(design, first.ravel(), rcond=None)[0]
    assert residual.std() > 1.0


def test_ema_and_warmup_cosine_are_deterministic():
    model = nn.Linear(2, 1)
    ema = ema_state(model)
    with torch.no_grad():
        model.weight.add_(2.0)
    old = ema["weight"].clone()
    update_ema(ema, model, 0.75)
    assert torch.allclose(ema["weight"], old + 0.5)
    rates = [cosine_learning_rate(step, 10, 2, 1e-3) for step in range(10)]
    assert rates[:2] == pytest.approx([5e-4, 1e-3])
    assert rates[-1] == pytest.approx(0.0, abs=1e-12)


def test_export_is_verified_and_promotion_is_explicit(tmp_path):
    model = ToyPoseModel().eval()
    export_folder = tmp_path / "workspace" / "run" / "export"
    metadata = export_onnx(
        model,
        export_folder,
        {"selection_split": "validation", "manifest_sha256": {"toy": "abc"}},
        torch.zeros(2, 3, 299, 299),
    )
    model_path = export_folder / "atlas_pose.onnx"
    assert metadata["sha256"] == file_sha256(model_path)
    assert metadata["preprocessing_version"] == "smart-mask-scale-invariant-v1"
    assert len(metadata["preprocessing_source_sha256"]) == 64
    assert (export_folder / "atlas_pose.json").is_file()
    assert json.loads((export_folder / "provenance.json").read_text())["excluded_from_selection"] == [
        "registered_test", "sealed_deepslice_s2p"
    ]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [output.name for output in session.get_outputs()] == [
        "pose_ap_um_lr_deg_dv_deg", "orientation_inverted_logit"
    ]
    destination = tmp_path / "promoted"
    assert not destination.exists()
    promote_export(export_folder, destination)
    assert {path.name for path in destination.iterdir()} == {"atlas_pose.onnx", "atlas_pose.json"}
