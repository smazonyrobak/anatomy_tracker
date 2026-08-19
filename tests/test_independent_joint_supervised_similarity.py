from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import training.independent_joint_data as independent_data
import training.run_independent_pose_identifiability as diagnostic
from source.dense_registration_preprocessing import MODEL_SHAPE
from training.independent_joint_model import IndependentJointModel, StructuralPyramid
from training.independent_joint_variants import (
    IndependentJointSimilarityCanonicalizedModel,
    IndependentJointSpatialMomentSimilarityCanonicalizedModel,
    SupervisedSimilarityCanonicalizer,
    VariantInitializerExport,
)


ROOT = Path(__file__).parents[1]
REFERENCE_CONFIG = ROOT / "training/configs/independent_pose_identifiability_300_r4322.json"
REFERENCE_CONFIG_SHA256 = "efcd541ed9824ca286ff065a0cd7693091cac13bbb353459a8d7f289b2aada6b"
BASE_STN_CONFIG = (
    ROOT / "training/configs/independent_pose_identifiability_supervised_similarity_300_r4322.json"
)
MOMENT_STN_CONFIG = (
    ROOT / "training/configs/independent_pose_identifiability_spatial_moment_supervised_similarity_300_r4322.json"
)


def _small(model_class):
    torch.manual_seed(29)
    return model_class(
        pyramid_channels=(8, 8, 8, 8),
        pose_context_features=24,
        pair_features=16,
        hidden_channels=16,
        integration_steps=3,
    )


def _source(batch: int):
    generator = torch.Generator().manual_seed(71 + batch)
    image = torch.rand(batch, 1, 32, 40, generator=generator)
    mask = torch.zeros(batch, 1, 32, 40, dtype=torch.bool)
    available = torch.zeros(batch, 1, 1, 1)
    return image, mask, available


def test_identity_initialized_canonicalizer_is_exact_noop_and_shared_between_variants():
    base = _small(IndependentJointSimilarityCanonicalizedModel).eval()
    moment = _small(IndependentJointSpatialMomentSimilarityCanonicalizedModel).eval()
    planes = StructuralPyramid._input(*_source(3))
    with torch.no_grad():
        warped, parameters = base.source_view_canonicalizer(planes)

    assert torch.equal(parameters["source_view_rotation_deg"], torch.zeros(3))
    assert torch.equal(parameters["source_view_log_scale"], torch.zeros(3))
    assert torch.equal(parameters["source_view_scale"], torch.ones(3))
    assert torch.allclose(warped, planes, atol=5e-6, rtol=0.0)
    assert all(
        torch.equal(value, moment.source_view_canonicalizer.state_dict()[name])
        for name, value in base.source_view_canonicalizer.state_dict().items()
    )


def test_forward_source_view_homography_is_the_canonicalization_sampling_direction():
    height, width = MODEL_SHAPE
    rotation = torch.tensor([27.0])
    scale = torch.tensor([0.85])
    log_scale = torch.log(scale)
    theta = SupervisedSimilarityCanonicalizer.sampling_theta(
        rotation, log_scale, height, width
    )
    grid = F.affine_grid(theta, (1, 1, height, width), align_corners=True)
    sampled_pixels = torch.stack(
        (
            (grid[..., 0] + 1.0) * (width - 1.0) / 2.0,
            (grid[..., 1] + 1.0) * (height - 1.0) / 2.0,
        ),
        dim=1,
    )
    identity = independent_data._identity_map(1, torch.device("cpu"))
    homogeneous = torch.cat((identity, torch.ones_like(identity[:, :1])), dim=1)
    source_view_h = independent_data._source_view_homography(rotation, scale)
    expected = torch.einsum("bij,bjhw->bihw", source_view_h[:, :2], homogeneous)
    assert torch.allclose(sampled_pixels, expected, atol=2e-4, rtol=0.0)

    marker = torch.zeros(1, 1, height, width)
    marker[:, :, 72:112, 91:151] = 0.65
    marker[:, :, 112:151, 91:111] = 1.0
    marker[:, :, 83:95, 139:181] = 0.35
    false_mask = torch.zeros_like(marker, dtype=torch.bool)
    identity_map = independent_data._identity_map(1, torch.device("cpu"))
    pair = {
        "moving": marker,
        "moving_tissue_mask": false_mask,
        "moving_damage_mask": false_mask,
        "moving_visible_mask": torch.ones_like(false_mask),
        "moving_brush_mask": false_mask,
        "moving_labels": torch.zeros_like(marker, dtype=torch.long),
        "moving_to_fixed": identity_map,
        "fixed_to_moving": identity_map,
        "fixed_visible_mask": torch.ones_like(false_mask),
        "similarity_h": torch.eye(3)[None],
    }
    manifest = {
        "source_view_rotation_deg": rotation.numpy(),
        "source_view_scale": scale.numpy(),
        "outline_plan": {
            "mode": [2],
            "sample_receipt_sha256": ["marker"],
            "plan_sha256": "marker-plan",
        },
    }
    viewed = independent_data._apply_source_view(pair, manifest, marker, false_mask)
    viewed_planes = StructuralPyramid._input(
        viewed["source_image"], viewed["source_mask"], viewed["mask_available"]
    )
    canonicalizer = SupervisedSimilarityCanonicalizer()
    restored = canonicalizer.warp_with_parameters(viewed_planes, rotation, log_scale)
    wrong = canonicalizer.warp_with_parameters(viewed_planes, -rotation, -log_scale)
    correct_error = (restored[:, :1] - marker).abs().mean()
    wrong_error = (wrong[:, :1] - marker).abs().mean()
    assert correct_error < 0.002
    assert correct_error < wrong_error * 0.2


def test_nuisance_loss_has_no_anatomical_pose_target_and_all_trainable_gradients_are_finite():
    model = _small(IndependentJointSpatialMomentSimilarityCanonicalizedModel).train()
    parameters = diagnostic._pose_parameter_group(model)
    output = model.initialize(*_source(4))
    output["pose"].retain_grad()
    nuisance_truth = torch.tensor(
        [[-30.0, 0.8], [-10.0, 0.95], [10.0, 1.05], [30.0, 1.2]]
    )
    nuisance = diagnostic.source_view_supervision_loss(
        output, nuisance_truth, model
    )
    nuisance["loss"].backward()

    assert output["pose"].grad is None
    assert all(
        value.grad is None for value in model.pose_head.parameters()
    )
    canonicalizer_parameters = list(model.source_view_canonicalizer.parameters())
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in canonicalizer_parameters
    )
    assert torch.count_nonzero(
        model.source_view_canonicalizer.parameters_head.weight.grad
    ) > 0

    model.zero_grad(set_to_none=True)
    output = model.initialize(*_source(4))
    pose_truth = torch.tensor(
        [[-4175.0, -13.25, -18.25], [-3100.0, -13.25, 18.25],
         [-2175.0, 13.25, -18.25], [-1100.0, 13.25, 18.25]]
    )
    pose = diagnostic.categorical_residual_loss(output, pose_truth, model)
    (pose["categorical"] + 0.5 * pose["sub_bin_residual"]).backward()
    assert all(value.grad is None for value in canonicalizer_parameters)
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in model.pyramid.slice_stem.parameters()
    )

    model.zero_grad(set_to_none=True)
    output = model.initialize(*_source(4))
    pose = diagnostic.categorical_residual_loss(output, pose_truth, model)
    nuisance = diagnostic.source_view_supervision_loss(output, nuisance_truth, model)
    (pose["categorical"] + 0.5 * pose["sub_bin_residual"] + nuisance["loss"]).backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in parameters)


def test_parameter_delta_is_tiny_and_default_model_and_initializer_abi_are_unchanged():
    default = IndependentJointModel().eval()
    base = IndependentJointSimilarityCanonicalizedModel().eval()
    moment = IndependentJointSpatialMomentSimilarityCanonicalizedModel().eval()
    assert sum(value.numel() for value in default.parameters()) == 1_369_070
    assert sum(value.numel() for value in base.parameters()) == 1_373_904
    assert sum(value.numel() for value in moment.parameters()) == 1_378_172
    assert sum(value.numel() for value in base.source_view_canonicalizer.parameters()) == 4_834
    assert sum(value.numel() for value in base.parameters()) < 1.004 * sum(
        value.numel() for value in default.parameters()
    )
    default_output = default.initialize(*_source(2))
    assert "source_view_rotation_deg" not in default_output
    assert len(VariantInitializerExport(base)(*_source(2))) == 10
    assert default.learned_weight_dependencies == base.learned_weight_dependencies == ()
    assert default.initialization == base.initialization == "random"


def test_frozen_stn_configs_are_an_exact_matched_pair_and_exclude_protected_targets():
    assert hashlib.sha256(REFERENCE_CONFIG.read_bytes()).hexdigest() == REFERENCE_CONFIG_SHA256
    reference = diagnostic.inspect_pose_identifiability_config(REFERENCE_CONFIG)
    base = diagnostic.load_pose_identifiability_config(BASE_STN_CONFIG)
    moment = diagnostic.load_pose_identifiability_config(MOMENT_STN_CONFIG)
    for name in (
        "schema_version", "frozen", "purpose", "role", "product5_access",
        "calibration_access", "final_test_access", "learned_checkpoint_dependencies",
        "seed", "device", "paths", "data", "training", "evaluation", "gates",
    ):
        assert base[name] == moment[name]
    for name in (
        "schema_version", "frozen", "purpose", "role", "product5_access",
        "calibration_access", "final_test_access", "learned_checkpoint_dependencies",
        "seed", "device", "paths", "data", "evaluation",
    ):
        assert base[name] == reference[name]
    assert {
        name: base["training"][name]
        for name in reference["training"]
        if name != "loss_weights"
    } == {
        name: value for name, value in reference["training"].items()
        if name != "loss_weights"
    }
    assert {
        name: base["gates"][name] for name in reference["gates"]
    } == reference["gates"]
    assert base["training"]["loss_weights"] == {
        "categorical": 1.0,
        "sub_bin_residual": 0.5,
        "source_view_supervision": 1.0,
    }
    assert base["training"]["source_view_supervision_contract"] == {
        "canonicalizer_initialization_seed": 12731,
        "targets": ["source_view_rotation_deg", "source_view_scale"],
        "loss": "smooth-l1-normalized-rotation-and-log-scale",
        "pose_gradient_to_canonicalizer": "blocked-at-sampling-parameters",
        "gradient_clipping": "separate-pose-and-canonicalizer-groups-at-5.0",
        "anatomical_pose_target_access": False,
    }
    assert {
        name: base["gates"][name]
        for name in (
            "seen_source_view_rotation_mae_deg_maximum",
            "seen_source_view_scale_mae_maximum",
            "held_source_view_rotation_mae_deg_maximum",
            "held_source_view_scale_mae_maximum",
        )
    } == {
        "seen_source_view_rotation_mae_deg_maximum": 2.0,
        "seen_source_view_scale_mae_maximum": 0.03,
        "held_source_view_rotation_mae_deg_maximum": 3.0,
        "held_source_view_scale_mae_maximum": 0.05,
    }
    assert base["learned_checkpoint_dependencies"] == []
    assert not base["product5_access"]
    assert base["model"]["kwargs"] == moment["model"]["kwargs"]
    assert base["model"]["class"].endswith(
        ".IndependentJointSimilarityCanonicalizedModel"
    )
    assert moment["model"]["class"].endswith(
        ".IndependentJointSpatialMomentSimilarityCanonicalizedModel"
    )
    base_contract = diagnostic._model_contract(base)
    moment_contract = diagnostic._model_contract(moment)
    assert base_contract["source_view_supervision"]
    assert moment_contract["source_view_supervision"]


def test_separate_clipping_keeps_matched_canonicalizer_step_bit_exact():
    source = _source(4)
    pose_truth = torch.tensor(
        [[-4175.0, -13.25, -18.25], [-3100.0, -13.25, 18.25],
         [-2175.0, 13.25, -18.25], [-1100.0, 13.25, 18.25]]
    )
    nuisance_truth = torch.tensor(
        [[-30.0, 0.8], [-10.0, 0.95], [10.0, 1.05], [30.0, 1.2]]
    )
    states = []
    clipping_records = []
    for model_class in (
        IndependentJointSimilarityCanonicalizedModel,
        IndependentJointSpatialMomentSimilarityCanonicalizedModel,
    ):
        model = _small(model_class).train()
        parameters = diagnostic._pose_parameter_group(model)
        optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)
        output = model.initialize(*source)
        pose = diagnostic.categorical_residual_loss(output, pose_truth, model)
        nuisance = diagnostic.source_view_supervision_loss(
            output, nuisance_truth, model
        )
        (pose["categorical"] + 0.5 * pose["sub_bin_residual"] + nuisance["loss"]).backward()
        clipping_records.append(
            diagnostic._clip_training_gradients(model, parameters, 5.0)
        )
        optimizer.step()
        states.append(model.source_view_canonicalizer.state_dict())

    assert clipping_records[0]["canonicalizer_preclip_norm"] == pytest.approx(
        clipping_records[1]["canonicalizer_preclip_norm"], abs=0.0
    )
    assert all(
        torch.equal(value, states[1][name]) for name, value in states[0].items()
    )


def test_failed_transform_attribution_gate_is_classified_before_pose_failure():
    config = diagnostic.load_pose_identifiability_config(BASE_STN_CONFIG)
    truth_sd = torch.tensor([1400.0, 12.5, 17.5])
    evaluation = {
        "seen": {
            "bin_accuracy": torch.tensor([0.1, 0.1, 0.1]),
            "residual_improvement_over_zero": torch.tensor([0.2, 0.2, 0.2]),
            "source_view_canonicalization": {
                "rotation_mae_deg": torch.tensor(2.01),
                "scale_mae": torch.tensor(0.02),
            },
        },
        "held": {
            "mae": torch.tensor([1000.0, 10.0, 10.0]),
            "prediction_sd": truth_sd,
            "truth_sd": truth_sd,
            "physical_improvement_over_constant_prior": -0.1,
            "residual_improvement_over_zero": torch.tensor([0.2, 0.2, 0.2]),
            "source_view_canonicalization": {
                "rotation_mae_deg": torch.tensor(2.0),
                "scale_mae": torch.tensor(0.04),
            },
        },
        "nonfinite_output_count": 0,
    }
    gradients = [
        {"update": update, "clipped": False} for update in range(1, 301)
    ]
    qualification = diagnostic.qualification_status(
        evaluation, gradients, 0, config
    )
    assert qualification["decision"] == "stop"
    assert qualification["classification"] == (
        "source-view-canonicalizer-not-identified-on-seen-transforms"
    )
    assert not qualification["checks"]["seen_source_view_rotation"]["passed"]

    evaluation["seen"]["source_view_canonicalization"]["rotation_mae_deg"] = torch.tensor(2.0)
    evaluation["held"]["source_view_canonicalization"]["scale_mae"] = torch.tensor(0.051)
    qualification = diagnostic.qualification_status(evaluation, gradients, 0, config)
    assert qualification["classification"] == (
        "source-view-canonicalizer-identified-on-seen-but-held-transform-"
        "generalization-insufficient"
    )


@pytest.mark.parametrize(
    "model_class",
    [
        IndependentJointSimilarityCanonicalizedModel,
        IndependentJointSpatialMomentSimilarityCanonicalizedModel,
    ],
)
def test_stn_initializer_onnx_checker_and_cpu_dml_runtime_dynamic_batch(model_class):
    onnx = __import__("onnx")
    ort = __import__("onnxruntime")
    model = _small(model_class).eval()
    with torch.no_grad():
        model.source_view_canonicalizer.parameters_head.weight.zero_()
        model.source_view_canonicalizer.parameters_head.bias.copy_(
            torch.tensor(
                [
                    math.atanh(27.0 / 45.0),
                    math.atanh(math.log(0.85) / math.log(1.5)),
                ]
            )
        )
        predicted = model.source_view_canonicalizer.predict_parameters(
            StructuralPyramid._input(*_source(2))
        )
    assert torch.allclose(
        predicted["source_view_rotation_deg"], torch.full((2,), 27.0), atol=1e-5
    )
    assert torch.allclose(
        predicted["source_view_scale"], torch.full((2,), 0.85), atol=1e-6
    )
    wrapper = VariantInitializerExport(model)
    traced_inputs = _source(2)
    output_names = [
        "pose", "pose_context", "ap_logits", "lr_logits", "dv_logits",
        "pose_cholesky", "source_feature_0", "source_feature_1",
        "source_feature_2", "source_feature_3",
    ]
    buffer = io.BytesIO()
    torch.onnx.export(
        wrapper,
        traced_inputs,
        buffer,
        input_names=["source_image", "source_mask", "mask_available"],
        output_names=output_names,
        dynamic_axes={
            **{name: {0: "source_batch"} for name in (
                "source_image", "source_mask", "mask_available",
            )},
            **{name: {0: "source_batch"} for name in output_names},
        },
        opset_version=20,
        dynamo=False,
    )
    graph = onnx.load_from_string(buffer.getvalue())
    onnx.checker.check_model(graph)
    operator_list = [node.op_type for node in graph.graph.node]
    operators = set(operator_list)
    assert {"AffineGrid", "GridSample"}.issubset(operators)
    assert operator_list.count("AffineGrid") == 1
    assert operator_list.count("GridSample") == 1

    dynamic_inputs = _source(3)
    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = False
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    inputs = {
        "source_image": dynamic_inputs[0].numpy(),
        "source_mask": dynamic_inputs[1].numpy(),
        "mask_available": dynamic_inputs[2].numpy(),
    }
    cpu_session = ort.InferenceSession(
        buffer.getvalue(), sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    actual = cpu_session.run(
        None,
        inputs,
    )
    with torch.no_grad():
        expected = wrapper(*dynamic_inputs)
    assert actual[0].shape == (3, 3)
    assert all(
        torch.allclose(left, torch.from_numpy(right), atol=7e-4, rtol=2e-4)
        for left, right in zip(expected, actual)
    )
    if "DmlExecutionProvider" in ort.get_available_providers():
        dml_session = ort.InferenceSession(
            buffer.getvalue(), sess_options=session_options,
            providers=["DmlExecutionProvider"],
        )
        dml = dml_session.run(None, inputs)
        assert all(
            torch.allclose(
                torch.from_numpy(left), torch.from_numpy(right),
                atol=7e-4, rtol=2e-4,
            )
            for left, right in zip(actual, dml)
        )


class _FakeGenerator:
    def __init__(self, atlas_folder, device):
        self.device = torch.device(device)
        self.contract = {"contract_sha256": "a" * 64}
        self.annotation = torch.ones(2, 2, 2, dtype=torch.int16)


class _FakeSynthetic:
    def __init__(self, generator):
        self.contract = {"contract_sha256": "d" * 64}


def _fake_supervised_panels(config, synthetic, generator):
    truth = torch.from_numpy(diagnostic.latent_pose_table(config))
    transforms = diagnostic.nuisance_transform_tables(config)
    base = torch.linspace(0.0, 1.0, 24 * 32 * 40).reshape(24, 1, 32, 40)
    panels = {"seen": [], "held": []}
    for kind in panels:
        for index in range(2):
            nuisance = torch.from_numpy(transforms[kind][index]).float()
            panels[kind].append(
                {
                    "source_image": torch.roll(
                        base, index + int(kind == "held"), dims=-1
                    ),
                    "source_mask": torch.zeros(24, 1, 32, 40, dtype=torch.bool),
                    "mask_available": torch.zeros(24, 1, 1, 1),
                    "true_pose": truth.clone(),
                    "truth_source_view_parameters": nuisance,
                    "manifest_sha256": f"{kind}-{index}",
                    "generator_manifest_sha256": "g" * 64,
                    "outline_plan_sha256": "o" * 64,
                    "data_contract_sha256": synthetic.contract["contract_sha256"],
                }
            )
    return {"contract_sha256": "p" * 64}, panels, torch.ones(24, 7, 9, dtype=torch.bool)


def test_supervised_runner_one_update_pause_resume_records_nuisance_contract(
    tmp_path, monkeypatch
):
    config = copy.deepcopy(diagnostic.load_pose_identifiability_config(BASE_STN_CONFIG))
    config["device"] = "cpu"
    config["name"] = "supervised-stn-resume-smoke"
    config["training"]["amp"] = False
    config["training"]["max_updates"] = 3
    config["training"]["gradient_clip_warmup_updates"] = 0
    config["training"]["resume_state_every_updates"] = 1
    config["model"]["kwargs"] = {
        "pyramid_channels": (8, 8, 8, 8),
        "pose_context_features": 24,
        "pair_features": 16,
        "hidden_channels": 16,
        "integration_steps": 3,
    }
    config["model"]["expected_parameter_count"] = sum(
        value.numel()
        for value in _small(IndependentJointSimilarityCanonicalizedModel).parameters()
    )
    monkeypatch.setattr(
        diagnostic,
        "load_pose_identifiability_config",
        lambda path: copy.deepcopy(config),
    )
    monkeypatch.setattr(diagnostic, "SyntheticRegistrationGenerator", _FakeGenerator)
    monkeypatch.setattr(
        diagnostic.independent_data, "IndependentSyntheticData", _FakeSynthetic
    )
    monkeypatch.setattr(diagnostic, "_prepare_fixed_panels", _fake_supervised_panels)
    monkeypatch.setattr(
        diagnostic,
        "_resolve_paths",
        lambda loaded: (tmp_path / "atlas", tmp_path / "run"),
    )

    first = diagnostic.run_pose_identifiability(BASE_STN_CONFIG, max_updates_this_call=1)
    second = diagnostic.run_pose_identifiability(BASE_STN_CONFIG, max_updates_this_call=1)
    assert first["status"] == second["status"] == "paused"
    assert first["updates"] == 1
    assert second["updates"] == 2
    receipt = json.loads(second["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["source_view_supervision"]["enabled"] is True
    assert receipt["source_view_supervision"]["anatomical_pose_target_access"] is False
    assert receipt["progress"] == {
        "optimizer_updates": 2,
        "sample_presentations": 48,
    }
    assert len(receipt["gradient_records"]) == 2
    for record in receipt["gradient_records"]:
        assert all(
            name in record
            for name in (
                "pose_preclip_norm",
                "canonicalizer_preclip_norm",
                "source_view_supervision_loss",
                "source_view_rotation_mae_deg",
                "source_view_scale_mae",
            )
        )
        assert np.isfinite(
            [
                record["pose_preclip_norm"],
                record["canonicalizer_preclip_norm"],
                record["source_view_supervision_loss"],
                record["source_view_rotation_mae_deg"],
                record["source_view_scale_mae"],
            ]
        ).all()
