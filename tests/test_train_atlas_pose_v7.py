import csv
import inspect
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn
import training.train_atlas_pose_v7 as trainer

from training.train_atlas_pose_v7 import (
    ABLATIONS_20K,
    COMPARISON_SEEDS,
    FINAL_GATE_THRESHOLDS,
    SCHEDULE,
    _canonical_json_sha256,
    atlas_data_hashes,
    bootstrap_seed_group_comparison,
    cosine_learning_rate,
    ema_state,
    ensure_fixed_manifest,
    ensure_paired_manifest,
    export_onnx,
    final_acceptance_summary,
    file_sha256,
    held_out_reports,
    manifest_sha256,
    module_state_sha256,
    paired_invariance,
    promote_export,
    registered_data_hashes,
    registered_report,
    registered_domain_reports,
    representative_onnx_batch,
    seed_animal_component_errors,
    select_model_family,
    renderer_variant,
    registered_style,
    stratified_pose_metrics,
    training_objective,
    update_ema,
    validation_selection_summary,
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

    def training_outputs(self, image, include_anatomy=True):
        feature_map = torch.relu(self.encoder(image))
        features = self.features(feature_map.mean((-2, -1)))
        normalized_pose = self.pose_head(features)
        image_frame_pose = normalized_pose * self.scale + self.center
        orientation = self.orientation_head(features).squeeze(1)
        sign = torch.where(orientation > 0.0, -torch.ones_like(orientation), torch.ones_like(orientation))
        pose = torch.cat((image_frame_pose[:, :1], image_frame_pose[:, 1:] * sign[:, None]), 1)
        output = {
            "pose": pose,
            "image_frame_pose": image_frame_pose,
            "normalized_pose": normalized_pose,
            "orientation_inverted_logit": orientation,
            "pooled_features": features,
        }
        if include_anatomy:
            output["anatomy_logits"] = self.anatomy_head(feature_map)
        return output

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
        "head_screen_20k": 20_000,
        "ablation_20k": 20_000,
        "surviving_heads_100k": 100_000,
        "backbones_100k": 100_000,
        "final_unique_views": 1_000_000,
    }
    assert COMPARISON_SEEDS == (73191, 41777, 90217)
    assert "full" not in ABLATIONS_20K
    assert set(ABLATIONS_20K) == {"renderer_minimal", "no_consistency", "no_anatomy"}
    assert ABLATIONS_20K["renderer_minimal"]["renderer"] == "minimal"
    assert any(config.get("consistency") == 0.0 for config in ABLATIONS_20K.values())
    assert any(config.get("anatomy") == 0.0 for config in ABLATIONS_20K.values())


def test_schedule_ablations_use_explicit_selected_backbone_control(tmp_path, monkeypatch):
    calls = []

    def comparison(name, samples, overrides):
        calls.append((name, samples, dict(overrides)))
        return [name]

    summary = {
        "component_mae": {"ap_um": 80.0, "lr_deg": 1.0, "dv_deg": 2.0},
        "composite_score": 1.0,
        "worst_gate_ratio": 1.2,
    }
    monkeypatch.setattr(trainer, "WORKSPACE", tmp_path)
    monkeypatch.setattr(trainer, "run_comparison_group", comparison)
    monkeypatch.setattr(trainer, "seed_group_selection_summary", lambda _results: summary)
    monkeypatch.setattr(
        trainer,
        "bootstrap_seed_group_comparison",
        lambda *_args, **_kwargs: {"probability_candidate_better": 0.5},
    )
    monkeypatch.setattr(
        trainer,
        "select_model_family",
        lambda groups, label, *_args: {
            "winner": "binned" if label == "pose head" else "maxvit_tiny",
            "runner_up": next(iter(groups)),
            "summaries": {name: summary for name in groups},
        },
    )
    monkeypatch.setattr(
        trainer,
        "run_experiment",
        lambda *_args, **_kwargs: {"held_out_reports": {"test": {}}},
    )
    result = trainer.run_schedule()
    assert {call[0] for call in calls if call[0].startswith("20k_head_")} == {
        "20k_head_direct", "20k_head_binned", "20k_head_ouv"
    }
    backbone_calls = [call[0] for call in calls if call[0].startswith("100k_backbone_")]
    assert set(backbone_calls) == {
        "100k_backbone_maxvit_tiny_binned",
        "100k_backbone_xception_binned",
    }
    assert not any("convnext_tiny" in name for name in backbone_calls)
    ablation_calls = [call for call in calls if call[0].startswith("20k_ablation")]
    assert len(ablation_calls) == 4
    assert all(call[2]["architecture"] == "maxvit_tiny" for call in ablation_calls)
    assert all(call[2]["head"] == "binned" for call in ablation_calls)
    assert result["ablation_control"] == summary


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


def test_paired_manifest_cache_is_keyed_by_the_base_latents(tmp_path):
    base, _ = ensure_fixed_manifest(tmp_path, "train", 8, 19)
    minimal = renderer_variant(base, "minimal")
    _, full_path = ensure_paired_manifest(tmp_path, base, "train", 23)
    paired_minimal, minimal_path = ensure_paired_manifest(tmp_path, minimal, "train", 23)

    assert full_path != minimal_path
    for key in minimal.keys() - set(APPEARANCE_MANIFEST_KEYS):
        assert np.array_equal(minimal[key], paired_minimal[key])


def test_fixed_synthetic_validation_cache_is_lossless_and_renders_only_once(tmp_path):
    class Renderer:
        atlas_folder = tmp_path
        device = torch.device("cpu")

        def __init__(self):
            self.calls = 0

        def batch(self, manifest, start, count):
            self.calls += 1
            pixels = torch.from_numpy(manifest["pixels"][start : start + count])
            image = pixels[:, None, None, None].expand(-1, 3, 5, 5).clone()
            target = torch.from_numpy(
                np.column_stack(
                    (
                        manifest["ap_um"][start : start + count],
                        np.zeros((count, 2), dtype=np.float32),
                    )
                )
            )
            return image, torch.zeros_like(target), target

    manifest = {
        "ap_um": np.asarray([-100.0, -200.0, -300.0, -400.0], dtype=np.float32),
        "cohort": np.arange(4, dtype=np.uint8),
        "pixels": np.asarray([0.0012345, 0.1723456, 0.5012345, 0.9987654], dtype=np.float32),
    }
    paired = {
        **manifest,
        "pixels": np.asarray([0.9981234, 0.5123456, 0.0612345, 0.004321], dtype=np.float32),
    }
    renderer = Renderer()
    trainer._SYNTHETIC_VALIDATION_CACHE.clear()
    first = trainer._synthetic_validation_images(renderer, manifest, paired, batch_size=2)
    second = trainer._synthetic_validation_images(renderer, manifest, paired, batch_size=2)
    assert renderer.calls == 4
    assert all(left.data_ptr() == right.data_ptr() for left, right in zip(first, second))
    assert torch.equal(first[0], torch.from_numpy(manifest["pixels"])[:, None, None].expand(-1, 5, 5))
    assert torch.equal(first[1], torch.from_numpy(paired["pixels"])[:, None, None].expand(-1, 5, 5))
    assert torch.equal(first[2][:, 0], torch.from_numpy(manifest["ap_um"]))


def test_paired_objective_updates_pose_orientation_consistency_and_anatomy():
    torch.manual_seed(3)
    model = ToyPoseModel()
    images = torch.rand(4, 2, 3, 16, 16)
    pose = torch.tensor([[-1200.0, 4.0, -2.0], [-1800.0, -5.0, 7.0], [-2400.0, 8.0, 1.0], [-3000.0, -3.0, -6.0]])
    orientation = torch.tensor([0.0, 1.0, 0.0, 1.0])
    anatomy = torch.randint(0, 9, (4, 16, 16))
    loss, components = training_objective(model, images, pose, orientation, anatomy, 0.15, 0.20)
    loss.backward()
    assert set(components) == {
        "total",
        "pose",
        "image_frame_pose",
        "soft_physical_tilt",
        "weighted_soft_physical_tilt",
        "representation_auxiliary",
        "weighted_representation_auxiliary",
        "orientation",
        "weighted_orientation",
        "feature_consistency",
        "prediction_consistency",
        "anatomy",
    }
    assert components["feature_consistency"] > 0.0
    assert components["prediction_consistency"] > 0.0
    assert model.pose_head.weight.grad is not None
    assert model.orientation_head.weight.grad is not None
    assert model.anatomy_head.weight.grad is not None
    assert model.encoder.weight.grad is not None


def test_no_anatomy_objective_does_not_execute_or_update_anatomy_decoder():
    model = ToyPoseModel()
    image = torch.rand(2, 3, 16, 16)
    pose = torch.zeros(2, 3)
    loss, components = training_objective(
        model,
        image,
        pose,
        torch.zeros(2),
        None,
        consistency_weight=0.0,
        anatomy_weight=0.0,
    )
    loss.backward()
    assert components["anatomy"] == 0.0
    assert model.anatomy_head.weight.grad is None


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
    assert registered["worst_ap_bin"] in registered["ap_500um_bands"]
    assert validation_selection_summary(rows)["composite_score"] > 0.0
    for forbidden in ("test", "sealed_deepslice_s2p"):
        with pytest.raises(RuntimeError, match="validation"):
            validation_selection_summary(validation_rows(forbidden))

    for row in rows:
        row["in_training_ap_domain"] = True
    excluded = {
        **rows[0],
        "section_image_id": 999,
        "target_ap": -5000.0,
        "prediction_ap": -4900.0,
        "in_training_ap_domain": False,
    }
    domains = registered_domain_reports([*rows, excluded])
    assert domains["primary_in_training_ap_domain"]["overall"]["count"] == len(rows)
    assert domains["out_of_domain"]["overall"]["count"] == 1


def test_validation_selection_equal_weights_each_specimen_ap_bin_and_enforces_axis_gates():
    rows = []
    for section in range(100):
        rows.append(
            {
                **validation_rows()[0],
                "section_image_id": section,
                "target_ap": -750.0,
                "prediction_ap": -750.0,
                "prediction_lr": 0.0,
                "target_lr": 0.0,
                "prediction_dv": 0.0,
                "target_dv": 0.0,
            }
        )
    rows.extend(
        (
            {
                **validation_rows()[0],
                "section_image_id": 101,
                "target_ap": -1750.0,
                "prediction_ap": -1630.0,
                "prediction_lr": 0.0,
                "target_lr": 0.0,
                "prediction_dv": 0.0,
                "target_dv": 0.0,
            },
            {
                **validation_rows()[2],
                "section_image_id": 201,
                "target_ap": -750.0,
                "prediction_ap": -690.0,
                "prediction_lr": 0.0,
                "target_lr": 0.0,
                "prediction_dv": 0.0,
                "target_dv": 0.0,
            },
        )
    )
    summary = validation_selection_summary(rows)
    assert summary["component_mae"]["ap_um"] == pytest.approx(60.0)
    assert summary["all_mean_gates_passed"] is True
    final_gate = final_acceptance_summary(rows, "validation")
    assert final_gate["all_gates_passed"] is False
    assert final_gate["passed"]["worst_ap_band_mae_um"] is False
    assert final_gate["thresholds"] == FINAL_GATE_THRESHOLDS

    for row in rows:
        row["prediction_lr"] = row["target_lr"] + 0.91
    failed = validation_selection_summary(rows)
    assert failed["component_passed"]["ap_um"] is True
    assert failed["component_passed"]["lr_deg"] is False
    assert failed["all_mean_gates_passed"] is False


def test_paired_invariance_reports_prediction_shift():
    target = np.zeros((6, 3))
    first = np.ones((6, 3))
    second = np.full((6, 3), 2.0)
    invariant = paired_invariance(target, first, second)
    assert invariant["mean_absolute_prediction_shift"] == [1.0, 1.0, 1.0]


def test_model_family_selection_uses_paired_seed_and_animal_uncertainty(tmp_path):
    def results(name, ap_errors):
        output = []
        for seed, ap_error in zip(COMPARISON_SEEDS, ap_errors):
            folder = tmp_path / f"{name}_{seed}"
            folder.mkdir()
            rows = validation_rows()
            for row in rows:
                row["prediction_ap"] = row["target_ap"] + ap_error
                row["prediction_lr"] = row["target_lr"]
                row["prediction_dv"] = row["target_dv"]
            with (folder / "validation_registered.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output.append(
                {
                    "selection_split": "validation",
                    "best_checkpoint": str(folder / "best.pt"),
                    "config": {"training_seed": seed},
                }
            )
        return output

    candidate = results("candidate", (60.0, 120.0, 180.0))
    reference = results("reference", (120.0, 180.0, 240.0))
    seeds, animals, errors = seed_animal_component_errors(candidate)
    assert seeds.tolist() == sorted(COMPARISON_SEEDS)
    assert animals.tolist() == [1, 2]
    assert errors.shape == (3, 2, 3)
    comparison = bootstrap_seed_group_comparison(
        candidate,
        reference,
        tmp_path / "comparison.json",
        iterations=1000,
        seed=7,
    )
    assert comparison["unit"] == "paired training_seed x specimen_id"
    assert comparison["seed_count"] == 3
    assert comparison["animal_count"] == 2
    assert comparison["components"]["ap_um"]["candidate_minus_reference"] == pytest.approx(-60.0)
    assert comparison["probability_candidate_better"] == 1.0

    tied = results("tied", (60.0, 120.0, 180.0))
    decision = select_model_family(
        {"preferred": tied, "point_best": candidate},
        "test family",
        ("preferred", "point_best"),
        tmp_path / "family",
    )
    assert decision["decision"] == "prespecified_tie_priority"
    assert decision["winner"] == "preferred"


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
    update_ema(ema, model, 0.75, 100)
    assert torch.allclose(ema["weight"], old + 0.5)
    ema = ema_state(nn.Linear(2, 1))
    source = nn.Linear(2, 1)
    source.load_state_dict({name: value.clone() for name, value in ema.items()})
    with torch.no_grad():
        source.weight.add_(2.0)
    update_ema(ema, source, 0.999, 1)
    assert torch.allclose(ema["weight"], source.weight - 2.0 * (2.0 / 11.0))
    rates = [cosine_learning_rate(step, 10, 2, 1e-3) for step in range(10)]
    assert rates[:2] == pytest.approx([5e-4, 1e-3])
    assert rates[-1] == pytest.approx(0.0, abs=1e-12)


def test_foreach_ema_matches_the_scalar_update_formula_exactly():
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    model.register_buffer("updates", torch.tensor(7, dtype=torch.int64))
    ema = ema_state(model)
    expected = {name: value.clone() for name, value in ema.items()}
    with torch.no_grad():
        for value in model.parameters():
            value.add_(torch.linspace(0.1, 0.9, value.numel()).reshape_as(value))
        model.updates.add_(1)
    decay = min(0.999, (1.0 + 17) / (10.0 + 17))
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            expected[name].mul_(decay).add_(value, alpha=1.0 - decay)
        else:
            expected[name].copy_(value)
    update_ema(ema, model, 0.999, 17)
    assert all(torch.equal(ema[name], expected[name]) for name in ema)


def test_export_is_verified_and_promotion_is_explicit(tmp_path):
    model = ToyPoseModel().eval()
    export_folder = tmp_path / "workspace" / "run" / "export"
    metadata = export_onnx(
        model,
        export_folder,
        {
            "selection_split": "validation",
            "manifest_sha256": {"toy": "a" * 64},
            "registered_data": {"sha256": {"downloads.jsonl": "b" * 64}},
            "atlas_data_sha256": {"average_template_25.nrrd": "c" * 64},
        },
        representative_onnx_batch()[:2],
    )
    model_path = export_folder / "atlas_pose.onnx"
    assert metadata["sha256"] == file_sha256(model_path)
    assert metadata["preprocessing_version"] == "smart-mask-scale-invariant-v1"
    assert metadata["preprocessing_contract_sha256"] == trainer.atlas_pose_preprocessing_contract_sha256()
    assert metadata["verification_sample_count"] == 2
    assert "CPUExecutionProvider" in metadata["verification_by_provider"]
    for provider in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        if provider in ort.get_available_providers():
            assert provider in metadata["verification_by_provider"]
    assert len(metadata["verification_input_sha256"]) == 64
    assert (export_folder / "atlas_pose.json").is_file()
    provenance = json.loads((export_folder / "provenance.json").read_text())
    assert provenance["registered_data_sha256"]["downloads.jsonl"] == "b" * 64
    assert provenance["excluded_from_selection"] == [
        "registered_test", "sealed_deepslice_s2p"
    ]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [output.name for output in session.get_outputs()] == [
        "pose_ap_um_lr_deg_dv_deg", "orientation_inverted_logit"
    ]
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    metrics = sealed / "SEALED_metrics.json"
    sealed_source = {
        "sections_sha256": "1" * 64,
        "datasets_sha256": "2" * 64,
        "provenance_sha256": "3" * 64,
        "downloads_sha256": "4" * 64,
        "registered_image_quality_manifest_sha256": "5" * 64,
    }
    metrics.write_text(
        json.dumps({"source": sealed_source, "evaluator_sha256": "6" * 64}),
        encoding="utf-8",
    )
    metadata_path = export_folder / "atlas_pose.json"
    release_payload = {
        "release_report_version": 2,
        "sealed": True,
        "benchmark_role": "final_release_gate",
        "release_approved": True,
        "promotion_ready": True,
        "quality_gate": {"all_gates_passed": True, "passed": {"mean_ap_um": True}},
        "deepslice_component_passed": {"ap_um": True, "lr_deg": True, "dv_deg": True},
        "model_sha256": file_sha256(model_path),
        "metadata_sha256": file_sha256(metadata_path),
        "preprocessing_contract_sha256": metadata["preprocessing_contract_sha256"],
        "training_source_sha256": metadata["source_sha256"],
        "training_data_sha256": {
            "synthetic_manifests": metadata["manifest_sha256"],
            "registered_data": metadata["registered_data"]["sha256"],
            "atlas_data": metadata["atlas_data_sha256"],
        },
        "sealed_data_sha256": sealed_source,
        "sealed_metrics_sha256": file_sha256(metrics),
        "evaluator_sha256": "6" * 64,
    }
    release_payload["release_integrity_sha256"] = _canonical_json_sha256(release_payload)
    release = sealed / "RELEASE_REPORT.json"
    release.write_text(json.dumps(release_payload), encoding="utf-8")
    destination = tmp_path / "promoted"
    assert not destination.exists()
    pins = promote_export(export_folder, release, destination)
    assert pins == {
        "APPROVED_ATLAS_POSE_MODEL_SHA256": file_sha256(model_path),
        "APPROVED_ATLAS_POSE_METADATA_SHA256": file_sha256(metadata_path),
        "APPROVED_ATLAS_POSE_EVIDENCE_SHA256": file_sha256(release),
    }
    assert {path.name for path in destination.iterdir()} == {
        "atlas_pose.onnx",
        "atlas_pose.json",
        "provenance.json",
        "RELEASE_REPORT.json",
        "SEALED_metrics.json",
    }


def test_registered_provenance_binds_download_manifest_and_ordinary_holdout_excludes_sealed(tmp_path):
    for name in ("datasets.jsonl", "sections.jsonl", "provenance.json", "downloads.jsonl"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    hashes = registered_data_hashes(tmp_path)
    assert set(hashes) == {"datasets.jsonl", "sections.jsonl", "provenance.json", "downloads.jsonl"}
    old = hashes["downloads.jsonl"]
    (tmp_path / "downloads.jsonl").write_text("changed", encoding="utf-8")
    assert registered_data_hashes(tmp_path)["downloads.jsonl"] != old
    assert "sealed_deepslice_s2p" not in inspect.getsource(held_out_reports)


def test_atlas_and_initial_model_state_provenance_are_content_bound(tmp_path):
    names = ("average_template_25.nrrd", "annotation_25.nrrd", "query.csv", "atlas_labels.pkl")
    for index, name in enumerate(names):
        (tmp_path / name).write_bytes(bytes((index, index + 1)))
    hashes = atlas_data_hashes(tmp_path)
    assert set(hashes) == set(names)
    (tmp_path / "query.csv").write_bytes(b"changed")
    assert atlas_data_hashes(tmp_path)["query.csv"] != hashes["query.csv"]

    module = nn.Linear(3, 2)
    digest = module_state_sha256(module)
    assert digest == module_state_sha256(module)
    with torch.no_grad():
        module.weight[0, 0].add_(1.0)
    assert module_state_sha256(module) != digest


def test_unknown_renderer_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown AtlasPose renderer"):
        renderer_variant({"cohort": np.zeros(1)}, "typo")
