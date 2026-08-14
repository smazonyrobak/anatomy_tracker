import hashlib
import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

import source.dense_registration_runtime as runtime
from source.dense_registration_runtime import (
    DENSE_REGISTRATION_V2_RELEASE_PROTOCOL,
    DENSE_REGISTRATION_V2_RELEASE_PROTOCOL_SHA256,
    INPUT_CONTRACT,
    INPUT_NAMES,
    MODEL_SHAPE,
    NATIVE_SHAPE,
    OUTPUT_CONTRACT,
    OUTPUT_NAMES,
    PINNED_ONNXRUNTIME_DIRECTML_VERSION,
    PREPROCESSING_CONTRACT_V2,
    native_absolute_map,
    preprocess_dense_registration_inputs,
    run_dense_registration,
    verify_dense_registration_bundle,
    verify_dense_registration_evaluated_bundle,
    verify_dense_registration_v2_bundle,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_map(dx=0.0, dy=0.0):
    yy, xx = np.mgrid[: MODEL_SHAPE[0], : MODEL_SHAPE[1]].astype(np.float32)
    return np.stack((xx + dx, yy + dy), axis=0)[None]


def _write_v2_bundle(tmp_path, model_bytes=b"released v2 ONNX"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / "dense_registration.onnx"
    metadata_path = tmp_path / "dense_registration.metadata.json"
    model_path.write_bytes(model_bytes)
    metadata = {
        "format_version": 2,
        "benchmark_id": DENSE_REGISTRATION_V2_RELEASE_PROTOCOL["benchmark_id"],
        "release_protocol": DENSE_REGISTRATION_V2_RELEASE_PROTOCOL,
        "release_protocol_sha256": DENSE_REGISTRATION_V2_RELEASE_PROTOCOL_SHA256,
        "scope": DENSE_REGISTRATION_V2_RELEASE_PROTOCOL["scope"],
        "candidate": {},
        "candidate_payload_sha256": "1" * 64,
        "candidate_file_sha256": "2" * 64,
        "candidate_checkpoint_file_sha256": "3" * 64,
        "checkpoint": {},
        "onnx_model_file_sha256": _sha256(model_path),
        "model_shape": list(MODEL_SHAPE),
        "model_config": {},
        "preprocessing_contract": PREPROCESSING_CONTRACT_V2,
        "mask_contract_payload_sha256": runtime.MASK_CONTRACT_SHA256,
        "appearance_contract_sha256": "4" * 64,
        "query_sha256": "5" * 64,
        "generator_contract_payload_sha256": "6" * 64,
        "generator_contract": {},
        "input_contract": INPUT_CONTRACT,
        "output_contract": OUTPUT_CONTRACT,
        "sealed_test": {},
        "production_provider": runtime.PRODUCTION_PROVIDER,
        "onnxruntime_directml_version": PINNED_ONNXRUNTIME_DIRECTML_VERSION,
        "onnxruntime_parity": {},
        "environment": {},
        "source_file_sha256": {},
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return model_path, metadata_path, metadata


def _write_evaluated_bundle(tmp_path, model_bytes=b"evaluated v2 ONNX"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / "dense_registration.onnx"
    metadata_path = tmp_path / "dense_registration.metadata.json"
    model_path.write_bytes(model_bytes)
    checkpoint = {"checkpoint_file_sha256": "3" * 64}
    qualification = {
        "status": "rejected",
        "receipt_file_sha256": "7" * 64,
        "receipt": {
            "status": "rejected",
            "checkpoint": checkpoint,
            "release_protocol_sha256": DENSE_REGISTRATION_V2_RELEASE_PROTOCOL_SHA256,
        },
        "per_seed": {
            str(seed): {
                "foreground_correspondence": 0.958,
                "macro_region_dice": 0.917,
                "endpoint_p95_px": 1.23,
                "release_gate_passed": False,
            }
            for seed in DENSE_REGISTRATION_V2_RELEASE_PROTOCOL["cohorts"]["qualification"]["seeds"]
        },
    }
    metadata = {
        "format_version": 2,
        "bundle_kind": "evaluated-v2",
        "release_approved": False,
        "benchmark_id": DENSE_REGISTRATION_V2_RELEASE_PROTOCOL["benchmark_id"],
        "release_protocol_sha256": DENSE_REGISTRATION_V2_RELEASE_PROTOCOL_SHA256,
        "scope": DENSE_REGISTRATION_V2_RELEASE_PROTOCOL["scope"],
        "checkpoint": checkpoint,
        "candidate_checkpoint_file_sha256": checkpoint["checkpoint_file_sha256"],
        "onnx_model_file_sha256": _sha256(model_path),
        "model_shape": list(MODEL_SHAPE),
        "model_config": {},
        "preprocessing_contract": PREPROCESSING_CONTRACT_V2,
        "mask_contract_payload_sha256": runtime.MASK_CONTRACT_SHA256,
        "input_contract": INPUT_CONTRACT,
        "output_contract": OUTPUT_CONTRACT,
        "qualification": qualification,
        "sealed_test": {"status": "not_run"},
        "production_provider": runtime.PRODUCTION_PROVIDER,
        "onnxruntime_directml_version": PINNED_ONNXRUNTIME_DIRECTML_VERSION,
        "onnxruntime_parity": {
            "passed": True,
            "production_provider": runtime.PRODUCTION_PROVIDER,
            "diagnostic_provider": "CPUExecutionProvider",
            "onnxruntime_directml_version": PINNED_ONNXRUNTIME_DIRECTML_VERSION,
            "provider_aggregates": {
                runtime.PRODUCTION_PROVIDER: {"passed": True},
                "CPUExecutionProvider": {"passed": True},
            },
        },
        "environment": {},
        "source_file_sha256": {},
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return model_path, metadata_path, metadata


def _images():
    yy, xx = np.mgrid[: NATIVE_SHAPE[0], : NATIVE_SHAPE[1]]
    atlas = (xx + yy).astype(np.float32)
    moving = np.clip(atlas / 3.0, 0, 255).astype(np.uint8)
    return atlas, np.ones(NATIVE_SHAPE, dtype=bool), moving


class _TensorInfo:
    def __init__(self, name, shape=None, tensor_type="tensor(float)"):
        self.name = name
        self.shape = shape or [None, 2, *MODEL_SHAPE]
        self.type = tensor_type


class _FakeSession:
    def __init__(self, providers, owner):
        self.providers = providers
        self.owner = owner
        owner.session = self

    def get_inputs(self):
        return [
            _TensorInfo(name, self.owner.input_shape, self.owner.tensor_type)
            for name in self.owner.input_names
        ]

    def get_outputs(self):
        return [
            _TensorInfo(name, self.owner.output_shape, self.owner.tensor_type)
            for name in self.owner.output_names
        ]

    def get_providers(self):
        return self.providers if self.owner.session_providers is None else self.owner.session_providers

    def run(self, names, feed):
        self.owner.names, self.owner.feed = names, feed
        return self.owner.outputs


class _FakeOrt:
    __version__ = PINNED_ONNXRUNTIME_DIRECTML_VERSION

    def __init__(
        self,
        *,
        available=None,
        session_providers=None,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        input_shape=None,
        output_shape=None,
        tensor_type="tensor(float)",
        outputs=None,
    ):
        self.available = available or [runtime.PRODUCTION_PROVIDER, "CPUExecutionProvider"]
        self.session_providers = session_providers
        self.input_names = input_names
        self.output_names = output_names
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.tensor_type = tensor_type
        self.outputs = outputs or [_model_map(2.0), _model_map(-2.0)]
        self.session = self.names = self.feed = None

    def get_available_providers(self):
        return self.available

    def InferenceSession(self, _path, providers):
        return _FakeSession(providers, self)


def _run(model, metadata, ort):
    atlas, mask, moving = _images()
    return run_dense_registration(
        model,
        atlas,
        mask,
        moving,
        mask,
        expected_model_sha256=_sha256(model),
        expected_metadata_sha256=_sha256(metadata),
        metadata_path=metadata,
        ort_module=ort,
    )


def test_model_map_crop_and_codomain_conversion_for_identity_and_translation():
    native_identity = native_absolute_map(_model_map())
    yy, xx = np.mgrid[: NATIVE_SHAPE[0], : NATIVE_SHAPE[1]].astype(np.float32)
    assert np.array_equal(native_identity, np.stack((xx, yy), axis=-1))
    translated = native_absolute_map(_model_map(7.25, -3.5))
    assert np.allclose(translated[..., 0], xx + 7.25)
    assert np.allclose(translated[..., 1], yy - 3.5)


def test_preprocessing_is_v2_only_and_masks_slice_appearance():
    yy, xx = np.mgrid[: NATIVE_SHAPE[0], : NATIVE_SHAPE[1]]
    atlas = (2.0 * xx + yy).astype(np.float32)
    moving = np.clip((xx + 3.0 * yy + 20.0) / 6.0, 0, 255).astype(np.uint8)
    atlas_mask = (xx > 20) & (xx < 430) & (yy > 15) & (yy < 300)
    moving_mask = (xx > 35) & (xx < 410) & (yy > 25) & (yy < 290)
    fixed, current = preprocess_dense_registration_inputs(
        atlas, atlas_mask, moving, moving_mask
    )
    assert fixed.shape == current.shape == (1, 2, *MODEL_SHAPE)
    assert not current[0, 0, :20, 4:460].any()
    assert np.array_equal(current[0, 1, :, 4:460], moving_mask)
    with pytest.raises(ValueError, match="Unsupported"):
        preprocess_dense_registration_inputs(
            atlas,
            atlas_mask,
            moving,
            moving_mask,
            preprocessing_contract="legacy-v1",
        )


def test_bundle_verifier_is_unpinned_for_build_but_checks_embedded_model_hash(tmp_path):
    model, metadata_path, _ = _write_v2_bundle(tmp_path)
    verified = verify_dense_registration_v2_bundle(model, metadata_path)
    assert verified["model_file_sha256"] == _sha256(model)
    assert verified["metadata_file_sha256"] == _sha256(metadata_path)
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        verify_dense_registration_v2_bundle(model, metadata_path)


def test_evaluated_bundle_has_a_separate_exact_schema_and_generic_dispatch(tmp_path):
    model, metadata_path, _ = _write_evaluated_bundle(tmp_path)
    evaluated = verify_dense_registration_evaluated_bundle(model, metadata_path)
    dispatched = verify_dense_registration_bundle(model, metadata_path)

    assert evaluated["bundle_kind"] == "evaluated-v2"
    assert evaluated["release_approved"] is False
    assert dispatched == evaluated
    with pytest.raises(RuntimeError, match="metadata schema"):
        verify_dense_registration_v2_bundle(model, metadata_path)


@pytest.mark.parametrize(
    "case",
    [
        "extra-key",
        "release-approved",
        "qualification-passed",
        "sealed-test-run",
        "cpu-parity-failed",
        "dml-parity-failed",
    ],
)
def test_evaluated_bundle_rejects_non_evaluated_or_unverified_evidence(tmp_path, case):
    model, metadata_path, metadata = _write_evaluated_bundle(tmp_path)
    if case == "extra-key":
        metadata["unexpected"] = True
    elif case == "release-approved":
        metadata["release_approved"] = True
    elif case == "qualification-passed":
        metadata["qualification"]["status"] = "passed"
    elif case == "sealed-test-run":
        metadata["sealed_test"] = {"status": "passed"}
    elif case == "cpu-parity-failed":
        metadata["onnxruntime_parity"]["provider_aggregates"]["CPUExecutionProvider"][
            "passed"
        ] = False
    else:
        metadata["onnxruntime_parity"]["provider_aggregates"][runtime.PRODUCTION_PROVIDER][
            "passed"
        ] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError):
        verify_dense_registration_evaluated_bundle(model, metadata_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", 1),
        ("model_shape", [1, 2]),
        ("preprocessing_contract", "wrong"),
        ("input_contract", "wrong"),
        ("output_contract", "wrong"),
        ("production_provider", "CPUExecutionProvider"),
        ("onnxruntime_directml_version", "1.24.3"),
    ],
)
def test_bundle_verifier_rejects_wrong_v2_runtime_contract(tmp_path, field, value):
    model, metadata_path, metadata = _write_v2_bundle(tmp_path)
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError):
        verify_dense_registration_v2_bundle(model, metadata_path)


def test_bundle_verifier_rejects_wrong_schema_and_installed_directml(tmp_path, monkeypatch):
    model, metadata_path, metadata = _write_v2_bundle(tmp_path)
    metadata.pop("input_contract")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema"):
        verify_dense_registration_v2_bundle(model, metadata_path)

    model, metadata_path, _ = _write_v2_bundle(tmp_path)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda _name: "1.24.3")
    with pytest.raises(RuntimeError, match="DirectML version"):
        verify_dense_registration_v2_bundle(model, metadata_path)


def test_production_inference_requires_pins_and_rejects_tampered_files(tmp_path):
    model, metadata_path, metadata = _write_v2_bundle(tmp_path)
    atlas, mask, moving = _images()
    model_pin, metadata_pin = _sha256(model), _sha256(metadata_path)
    with pytest.raises(ValueError, match="external SHA-256 pins"):
        run_dense_registration(
            model,
            atlas,
            mask,
            moving,
            mask,
            expected_model_sha256=None,
            expected_metadata_sha256=None,
            metadata_path=metadata_path,
            ort_module=_FakeOrt(),
        )
    with pytest.raises(RuntimeError, match="model differs"):
        run_dense_registration(
            model,
            atlas,
            mask,
            moving,
            mask,
            expected_model_sha256="0" * 64,
            expected_metadata_sha256=metadata_pin,
            metadata_path=metadata_path,
            ort_module=_FakeOrt(),
        )

    original_model = model.read_bytes()
    model.write_bytes(original_model + b"tampered")
    with pytest.raises(RuntimeError, match="model differs"):
        run_dense_registration(
            model,
            atlas,
            mask,
            moving,
            mask,
            expected_model_sha256=model_pin,
            expected_metadata_sha256=metadata_pin,
            metadata_path=metadata_path,
            ort_module=_FakeOrt(),
        )
    model.write_bytes(original_model)

    metadata["sealed_test"] = {"tampered": True}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata differs"):
        run_dense_registration(
            model,
            atlas,
            mask,
            moving,
            mask,
            expected_model_sha256=model_pin,
            expected_metadata_sha256=metadata_pin,
            metadata_path=metadata_path,
            ort_module=_FakeOrt(),
        )


def test_runtime_uses_directml_and_preserves_map_directions(tmp_path):
    model, metadata_path, _ = _write_v2_bundle(tmp_path)
    ort = _FakeOrt()
    result = _run(model, metadata_path, ort)
    yy, xx = np.mgrid[: NATIVE_SHAPE[0], : NATIVE_SHAPE[1]]
    assert ort.session.providers == [runtime.PRODUCTION_PROVIDER]
    assert ort.names == list(OUTPUT_NAMES)
    assert set(ort.feed) == set(INPUT_NAMES)
    assert np.allclose(result["atlas_to_affine_xy"][..., 0], xx + 2.0)
    assert np.allclose(result["affine_to_atlas_xy"][..., 0], xx - 2.0)
    assert result["metadata"]["provider"] == runtime.PRODUCTION_PROVIDER
    assert json.loads(result["registration_metadata_json"])["method"] == "dense-registration-onnx-v2"


def test_runtime_records_evaluated_bundle_status_without_claiming_release(tmp_path):
    model, metadata_path, _ = _write_evaluated_bundle(tmp_path)
    result = _run(model, metadata_path, _FakeOrt())

    assert result["metadata"]["method"] == "dense-registration-onnx-evaluated-v2"
    assert result["metadata"]["bundle_kind"] == "evaluated-v2"
    assert result["metadata"]["release_approved"] is False
    assert result["metadata"]["qualification_status"] == "rejected"
    assert json.loads(result["registration_metadata_json"])["release_approved"] is False


def test_cpu_diagnostic_is_never_accepted_as_production(tmp_path):
    model, metadata_path, _ = _write_v2_bundle(tmp_path)
    with pytest.raises(RuntimeError, match="DmlExecutionProvider is unavailable"):
        _run(
            model,
            metadata_path,
            _FakeOrt(available=["CPUExecutionProvider"]),
        )
    with pytest.raises(RuntimeError, match="primary provider"):
        _run(
            model,
            metadata_path,
            _FakeOrt(session_providers=["CPUExecutionProvider"]),
        )


@pytest.mark.parametrize(
    "ort",
    [
        _FakeOrt(input_names=("wrong", INPUT_NAMES[1])),
        _FakeOrt(output_names=(OUTPUT_NAMES[0], "wrong")),
        _FakeOrt(input_shape=[None, 1, *MODEL_SHAPE]),
        _FakeOrt(output_shape=[2, 2, *MODEL_SHAPE]),
        _FakeOrt(tensor_type="tensor(double)"),
    ],
)
def test_runtime_rejects_wrong_onnx_io_contract(tmp_path, ort):
    model, metadata_path, _ = _write_v2_bundle(tmp_path)
    with pytest.raises(RuntimeError, match="input/output names|tensor shapes"):
        _run(model, metadata_path, ort)


def test_runtime_rejects_wrong_loaded_onnxruntime_version(tmp_path):
    model, metadata_path, _ = _write_v2_bundle(tmp_path)
    ort = _FakeOrt()
    ort.__version__ = "1.24.3"
    with pytest.raises(RuntimeError, match="runtime version"):
        _run(model, metadata_path, ort)


class _IdentityMapModel(torch.nn.Module):
    def forward(self, fixed, moving):
        height, width = fixed.shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(height, dtype=fixed.dtype, device=fixed.device),
            torch.arange(width, dtype=fixed.dtype, device=fixed.device),
            indexing="ij",
        )
        mapping = torch.stack((x, y))[None].expand(fixed.shape[0], -1, -1, -1)
        mapping = mapping + moving[:, :1] * 1e-12
        return mapping, mapping


def test_real_torch_onnx_bundle_and_directml_runtime(tmp_path):
    model_path = tmp_path / "dense_registration.onnx"
    fixed = torch.zeros(1, 2, *MODEL_SHAPE)
    torch.onnx.export(
        _IdentityMapModel(),
        (fixed, fixed),
        model_path,
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={name: {0: "batch"} for name in (*INPUT_NAMES, *OUTPUT_NAMES)},
        opset_version=17,
        dynamo=False,
    )
    _, metadata_path, _ = _write_v2_bundle(tmp_path, model_path.read_bytes())
    verify_dense_registration_v2_bundle(model_path, metadata_path)
    result = _run(model_path, metadata_path, ort)
    yy, xx = np.mgrid[: NATIVE_SHAPE[0], : NATIVE_SHAPE[1]]
    assert np.allclose(result["atlas_to_affine_xy"][..., 0], xx)
    assert np.allclose(result["atlas_to_affine_xy"][..., 1], yy)
