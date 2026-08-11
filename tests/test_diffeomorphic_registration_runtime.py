import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

import diffeomorphic_registration_runtime as registration_runtime
from diffeomorphic_registration_runtime import (
    DiffeomorphicRegistrationRejected,
    MODEL_PIXEL_SPACING_UM,
    MODEL_SHAPE,
    _correspondence_diagnostics,
    _correspondence_failures,
    _gray_unit,
    _mind_descriptor,
    _verified_model_manifest,
    run_diffeomorphic_registration,
)
from nonlinear_registration import (
    COORDINATE_CONVENTION,
    MODEL_CONTRACT_VERSION,
    MODEL_INPUT_NAMES,
    MODEL_OUTPUT_NAMES,
    MODEL_SPATIAL_CONTRACT,
    RUNTIME_GATE_CONTRACT,
)
from training.diffeomorphic_registration_model import mind_descriptor, preprocess_registration_tensor
from training.train_diffeomorphic_registration import write_model_manifest


class ValueInfo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(
        self,
        *,
        rejection_logit=-10.0,
        fold=False,
        one_cell_fold=False,
        inverse_spike=False,
        tissue_translation=False,
    ):
        self.rejection_logit = rejection_logit
        self.fold = fold
        self.one_cell_fold = one_cell_fold
        self.inverse_spike = inverse_spike
        self.tissue_translation = tissue_translation
        self.feeds = None

    def get_inputs(self):
        return [ValueInfo(name, ["batch", 1, *MODEL_SHAPE]) for name in (
            "fixed", "moving", "fixed_mask", "moving_mask"
        )]

    def get_outputs(self):
        return [
            ValueInfo("atlas_to_affine", ["batch", 2, *MODEL_SHAPE]),
            ValueInfo("affine_to_atlas", ["batch", 2, *MODEL_SHAPE]),
            ValueInfo("velocity", ["batch", 2, *MODEL_SHAPE]),
            ValueInfo("rejection_logit", ["batch"]),
        ]

    def get_providers(self):
        return ["FakeExecutionProvider"]

    def run(self, output_names, feeds):
        assert output_names == ["atlas_to_affine", "affine_to_atlas", "velocity", "rejection_logit"]
        self.feeds = feeds
        yy, xx = np.mgrid[: MODEL_SHAPE[0], : MODEL_SHAPE[1]].astype(np.float32)
        identity = np.stack((xx, yy))[None]
        forward = identity.copy()
        inverse = identity.copy()
        if self.fold:
            forward[:, 0] = forward[:, 0, :, ::-1]
        if self.one_cell_fold:
            forward[:, 0, 30, 40] = forward[:, 0, 30, 39] - 0.5
        if self.inverse_spike:
            inverse[:, 0, 30, 40] += 3.0
        if self.tissue_translation:
            trusted = feeds["fixed_mask"][:, 0] > 0.5
            forward[:, 0][trusted] += 2.5
            inverse[:, 0][trusted] -= 2.5
        return forward, inverse, np.zeros_like(identity), np.asarray([self.rejection_logit], np.float32)


def inputs(shape):
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    image = xx + 2.0 * yy
    mask = np.ones(shape, dtype=bool)
    return image, image.copy(), mask, mask.copy()


def write_manifest(model_path, **changes):
    commitment = {
        "source": {"sections_sha256": "3" * 64},
        "evaluation_manifest_sha256": "2" * 64,
    }
    evidence_path = model_path.with_suffix(".prelocked.json")
    evidence_path.write_text(json.dumps({
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "synthetic_gate": {"passed": True},
        "onnx_gate": {"passed": True},
        "locked_real_histology_commitment": commitment,
    }), encoding="utf-8")
    payload = {
        "format_version": MODEL_CONTRACT_VERSION,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_shape": list(MODEL_SHAPE),
        "pixel_spacing_um": MODEL_PIXEL_SPACING_UM,
        "spatial_contract": MODEL_SPATIAL_CONTRACT,
        "coordinate_convention": COORDINATE_CONVENTION,
        "input_names": list(MODEL_INPUT_NAMES),
        "output_names": list(MODEL_OUTPUT_NAMES),
        "runtime_gates": RUNTIME_GATE_CONTRACT,
        "prelocked_evidence_file": evidence_path.name,
        "prelocked_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "locked_real_histology_commitment": commitment,
        "onnx_gate_passed": True,
        "real_histology_gate_passed": True,
        "real_histology_gate_report_sha256": "1" * 64,
        "real_histology_evaluation_manifest_sha256": "2" * 64,
        "real_histology_benchmark_role": "locked_promotion_gate",
        "promotion_ready": True,
    }
    payload.update(changes)
    model_path.with_suffix(".manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_centered_padding_unpads_identity_maps_and_coordinates_exactly():
    shape = (42, 70)
    session = FakeSession()
    warp, diagnostics = run_diffeomorphic_registration(
        *inputs(shape), session=session, pixel_spacing_um=MODEL_PIXEL_SPACING_UM
    )
    top = (MODEL_SHAPE[0] - shape[0]) // 2
    left = (MODEL_SHAPE[1] - shape[1]) // 2

    assert diagnostics["model_offset_yx"] == (top, left)
    assert session.feeds["fixed"].shape == (1, 1, *MODEL_SHAPE)
    assert session.feeds["fixed_mask"][0, 0, top, left] == 1.0
    assert session.feeds["fixed_mask"][0, 0, top - 1, left] == 0.0
    assert np.array_equal(warp.atlas_to_affine_xy, warp.affine_to_atlas_xy)
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    assert np.array_equal(warp.atlas_to_affine_xy, np.stack((xx, yy), axis=-1))
    assert diagnostics["inverse_p95_px"] == pytest.approx(0.0)
    assert diagnostics["displacement_max_px"] == pytest.approx(0.0)


def test_center_crop_restores_native_coordinate_offsets():
    shape = (MODEL_SHAPE[0] + 20, MODEL_SHAPE[1] + 16)
    fixed, moving, fixed_mask, moving_mask = inputs(shape)
    fixed_mask[:] = moving_mask[:] = False
    fixed_mask[20:-20, 20:-20] = True
    moving_mask[:] = fixed_mask
    warp, diagnostics = run_diffeomorphic_registration(
        fixed, moving, fixed_mask, moving_mask, session=FakeSession(),
        pixel_spacing_um=MODEL_PIXEL_SPACING_UM,
    )

    assert diagnostics["source_offset_yx"] == (10, 8)
    assert diagnostics["model_offset_yx"] == (0, 0)
    points = np.asarray([[8.0, 10.0], [120.0, 90.0], [shape[1] - 9.0, shape[0] - 11.0]])
    assert np.allclose(warp.map_atlas_to_affine(points), points)


def test_model_rejection_is_explicit_and_returns_no_warp():
    with pytest.raises(DiffeomorphicRegistrationRejected) as error:
        run_diffeomorphic_registration(
            *inputs((40, 64)), session=FakeSession(rejection_logit=10.0),
            pixel_spacing_um=MODEL_PIXEL_SPACING_UM,
        )

    assert "model rejection probability" in str(error.value)
    assert error.value.diagnostics["rejection_probability"] > 0.99


def test_folded_map_is_rejected_before_caller_can_install_it():
    with pytest.raises(DiffeomorphicRegistrationRejected) as error:
        run_diffeomorphic_registration(
            *inputs(MODEL_SHAPE), session=FakeSession(fold=True),
            pixel_spacing_um=MODEL_PIXEL_SPACING_UM,
        )

    assert "Jacobian" in str(error.value)
    assert error.value.diagnostics["fold_count"] > 0


def test_one_cell_fold_is_rejected_before_caller_can_install_it():
    with pytest.raises(DiffeomorphicRegistrationRejected) as error:
        run_diffeomorphic_registration(
            *inputs(MODEL_SHAPE), session=FakeSession(one_cell_fold=True),
            pixel_spacing_um=MODEL_PIXEL_SPACING_UM,
        )

    assert error.value.diagnostics["minimum_forward_jacobian"] == pytest.approx(-0.5)


def test_local_inverse_failure_is_rejected_even_below_the_p95_tail():
    with pytest.raises(DiffeomorphicRegistrationRejected, match="maximum") as error:
        run_diffeomorphic_registration(
            *inputs(MODEL_SHAPE),
            session=FakeSession(inverse_spike=True),
            pixel_spacing_um=MODEL_PIXEL_SPACING_UM,
        )

    assert error.value.diagnostics["inverse_p95_px"] == pytest.approx(0.0)
    assert error.value.diagnostics["inverse_max_px"] > 2.0


def test_final_tissue_map_affine_is_rejected_even_when_velocity_claims_zero():
    shape = MODEL_SHAPE
    fixed, moving, fixed_mask, moving_mask = inputs(shape)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    fixed_mask[:] = ((xx - shape[1] / 2) / 90.0) ** 2 + ((yy - shape[0] / 2) / 65.0) ** 2 < 1.0
    moving_mask[:] = fixed_mask
    with pytest.raises(DiffeomorphicRegistrationRejected) as error:
        run_diffeomorphic_registration(
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            session=FakeSession(tissue_translation=True),
            pixel_spacing_um=MODEL_PIXEL_SPACING_UM,
        )

    assert "residual global affine" in str(error.value)
    assert error.value.diagnostics["residual_affine_max_px"] > 2.0


def test_numpy_and_torch_preprocessing_are_identical():
    rng = np.random.default_rng(17)
    image = rng.normal(size=(37, 53)).astype(np.float32)
    mask = np.zeros(image.shape, bool)
    mask[3:-4, 5:-6] = True
    numpy_result = _gray_unit(image, mask)
    torch_result = preprocess_registration_tensor(
        torch.from_numpy(image)[None, None], torch.from_numpy(mask.astype(np.float32))[None, None]
    )[0, 0].numpy()

    assert np.allclose(numpy_result, torch_result, atol=2e-6)


def test_runtime_mind_descriptor_matches_the_training_implementation():
    rng = np.random.default_rng(19)
    image = rng.random((37, 53), dtype=np.float32)
    expected = mind_descriptor(torch.from_numpy(image)[None, None])[0].numpy()
    assert np.allclose(_mind_descriptor(image), expected, atol=2e-6)


def test_safe_but_anatomically_worse_warp_is_rejected():
    rng = np.random.default_rng(23)
    shape = (80, 96)
    fixed = rng.random(shape, dtype=np.float32)
    moving = fixed.copy()
    mask = np.ones(shape, dtype=bool)
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    shift = 0.75 * np.sin(12.0 * np.pi * yy / (shape[0] - 1.0))
    forward = np.stack((xx + shift, yy), axis=-1)
    diagnostics = _correspondence_diagnostics(fixed, moving, mask, mask, forward)
    failures = _correspondence_failures(diagnostics)
    assert diagnostics["mind_improvement"] < 0.0
    assert "nonlinear warp does not improve MIND correspondence" in failures


def test_onnx_names_and_shapes_are_enforced():
    session = FakeSession()
    session.get_inputs = lambda: [ValueInfo("wrong", ["batch", 1, *MODEL_SHAPE])]
    with pytest.raises(RuntimeError, match="training contract"):
        run_diffeomorphic_registration(
            *inputs((40, 64)), session=session, pixel_spacing_um=MODEL_PIXEL_SPACING_UM
        )


def test_pixel_spacing_is_explicit_and_forbids_rescaling():
    with pytest.raises(ValueError, match="explicit 25 um"):
        run_diffeomorphic_registration(*inputs((40, 64)), session=FakeSession())
    with pytest.raises(ValueError, match="explicit 25 um"):
        run_diffeomorphic_registration(
            *inputs((40, 64)), session=FakeSession(), pixel_spacing_um=50.0
        )


def test_fractional_masks_use_the_same_hard_threshold_as_training():
    fixed, moving, _, _ = inputs((40, 64))
    fractional = np.zeros((40, 64), np.float32)
    fractional[5:35, 7:57] = 0.51
    fractional[0, 0] = 0.49
    session = FakeSession()
    run_diffeomorphic_registration(
        fixed, moving, fractional, fractional, session=session,
        pixel_spacing_um=MODEL_PIXEL_SPACING_UM,
    )
    top = (MODEL_SHAPE[0] - 40) // 2
    left = (MODEL_SHAPE[1] - 64) // 2
    fed = session.feeds["fixed_mask"][0, 0, top : top + 40, left : left + 64]
    assert np.array_equal(fed, fractional > 0.5)


def test_center_crop_rejects_tissue_outside_one_to_one_model_field():
    shape = (MODEL_SHAPE[0] + 20, MODEL_SHAPE[1] + 16)
    fixed, moving, fixed_mask, moving_mask = inputs(shape)
    with pytest.raises(DiffeomorphicRegistrationRejected, match="field of view"):
        run_diffeomorphic_registration(
            fixed, moving, fixed_mask, moving_mask, session=FakeSession(),
            pixel_spacing_um=MODEL_PIXEL_SPACING_UM,
        )


def test_model_manifest_verifies_sha_contract_evidence_and_source_pin(tmp_path, monkeypatch):
    model_path = tmp_path / "diffeomorphic.onnx"
    model_path.write_bytes(b"validated model")
    report = {
        "synthetic_gate": {"passed": True},
        "onnx_gate": {"passed": True},
        "real_histology_ground_truth_gate": {
            "passed": True,
            "report_sha256": "1" * 64,
            "evaluation_manifest_sha256": "2" * 64,
            "source": {"sections_sha256": "3" * 64},
            "benchmark_role": "locked_promotion_gate",
        },
        "locked_real_histology_commitment": {
            "source": {"sections_sha256": "3" * 64},
            "evaluation_manifest_sha256": "2" * 64,
        },
        "promotion_ready": True,
    }
    _, written_manifest_sha = write_model_manifest(model_path, report)
    with pytest.raises(RuntimeError, match="source-approved"):
        _verified_model_manifest(model_path)
    approved = {
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "manifest_sha256": written_manifest_sha,
        "real_histology_gate_report_sha256": "1" * 64,
        "real_histology_evaluation_manifest_sha256": "2" * 64,
    }
    monkeypatch.setattr(registration_runtime, "APPROVED_NONLINEAR_RELEASE", approved)
    model_sha, manifest_sha, manifest = _verified_model_manifest(model_path)
    assert model_sha == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert manifest_sha == written_manifest_sha
    assert manifest["spatial_contract"] == MODEL_SPATIAL_CONTRACT
    assert manifest["real_histology_gate_report_sha256"] == "1" * 64
    assert manifest["real_histology_evaluation_manifest_sha256"] == "2" * 64
    assert manifest["real_histology_benchmark_role"] == "locked_promotion_gate"

    write_manifest(model_path, promotion_ready=False)
    with pytest.raises(RuntimeError, match="promotion_ready"):
        _verified_model_manifest(model_path)
    write_manifest(model_path, model_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="model_sha256"):
        _verified_model_manifest(model_path)

    write_manifest(model_path, real_histology_gate_report_sha256="not-a-hash")
    with pytest.raises(RuntimeError, match="real_histology_gate_report_sha256"):
        _verified_model_manifest(model_path)
    write_manifest(model_path, real_histology_evaluation_manifest_sha256=None)
    with pytest.raises(RuntimeError, match="real_histology_evaluation_manifest_sha256"):
        _verified_model_manifest(model_path)
    write_manifest(model_path, real_histology_benchmark_role="validation")
    with pytest.raises(RuntimeError, match="real_histology_benchmark_role"):
        _verified_model_manifest(model_path)

    write_manifest(model_path)
    monkeypatch.setattr(
        registration_runtime,
        "APPROVED_NONLINEAR_RELEASE",
        {**approved, "real_histology_gate_report_sha256": "f" * 64},
    )
    with pytest.raises(RuntimeError, match="source-approved release"):
        _verified_model_manifest(model_path)
