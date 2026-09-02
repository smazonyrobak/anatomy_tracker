import copy

import numpy as np
import pytest
import torch

from tests.arbitrary_plane_production_v3_fixtures import catalogue
from training import arbitrary_plane_catalogue_v3 as catalogue_v3
from training.arbitrary_plane_catalogue_runtime_v6 import (
    make_complete_catalogue_runtime_v6,
    verify_bound_complete_catalogue_batch_v6,
    verify_complete_catalogue_runtime_v6,
)


def _runtime(dtype=torch.float32):
    artifact = catalogue()
    return make_complete_catalogue_runtime_v6(
        artifact,
        expected_catalogue_receipt_sha256=artifact["receipt_sha256"],
        device="cpu",
        dtype=dtype,
    )


def test_verified_runtime_expands_canonical_device_tensors_without_copying_cells():
    runtime = _runtime(dtype=torch.float64)
    batch = runtime.expand(3)
    assert verify_complete_catalogue_runtime_v6(runtime)
    tensors = verify_bound_complete_catalogue_batch_v6(
        batch, expected_runtime=runtime
    )
    assert runtime.binding["catalogue_receipt_sha256"]
    assert len(runtime.binding["catalogue_receipt_sha256"]) == 64
    assert tensors["cell_id"].tolist() == list(range(runtime.cell_count))
    assert tensors["cell_states"].shape == (3, runtime.cell_count, 12)
    assert tensors["cell_states"].stride(0) == 0
    assert tensors["support_origin_ap_dv_ml_um"].shape == (3,)
    assert tuple(tensors["support_origin_ap_dv_ml_um"].tolist()) == tuple(
        runtime.binding["support_origin_ap_dv_ml_um"]
    )
    assert torch.allclose(
        torch.logsumexp(tensors["cell_log_mass"], dim=1),
        torch.zeros(3, dtype=tensors["cell_log_mass"].dtype),
    )
    assert tensors["representation_log_weight"].shape == (
        3,
        runtime.cell_count,
        runtime.representation_count,
    )
    assert tensors["representation_to_canonical_raster_affine"].shape == (
        3,
        runtime.cell_count,
        runtime.representation_count,
        2,
        3,
    )


def test_reordered_catalogue_cannot_claim_the_original_trusted_receipt():
    artifact = catalogue()
    tampered = copy.deepcopy(artifact)
    order = torch.arange(artifact["counts"]["cell_count"] - 1, -1, -1)
    tampered["arrays"]["cell_states_float64"] = tampered["arrays"][
        "cell_states_float64"
    ][order.numpy()].copy()
    tampered["tensors"]["cell_states"] = tampered["tensors"]["cell_states"][
        :, order
    ].clone()
    tampered["array_receipts"] = {
        name: catalogue_v3._array_receipt(value)
        for name, value in tampered["arrays"].items()
    }
    tampered["receipt_sha256"] = catalogue_v3._hash(
        catalogue_v3.catalogue_receipt_v3(tampered)
    )
    with pytest.raises(ValueError, match="trusted expected receipt"):
        make_complete_catalogue_runtime_v6(
            tampered,
            expected_catalogue_receipt_sha256=artifact["receipt_sha256"],
            device="cpu",
            dtype=torch.float32,
        )


def test_arbitrary_tensor_mapping_cannot_impersonate_a_bound_batch():
    runtime = _runtime()
    genuine = runtime.expand(1)
    resolved = verify_bound_complete_catalogue_batch_v6(
        genuine, expected_runtime=runtime
    )
    forged = {name: value.clone() for name, value in resolved.items()}
    forged["cell_states"] = forged["cell_states"].flip(1)
    with pytest.raises(ValueError, match="not issued by the expected runtime"):
        verify_bound_complete_catalogue_batch_v6(
            forged, expected_runtime=runtime
        )


def test_opaque_batch_does_not_expose_replaceable_or_mutable_catalogue_tensors():
    runtime = _runtime()
    batch = runtime.expand(1)
    with pytest.raises(TypeError):
        batch["cell_states"]
    with pytest.raises(AttributeError, match="immutable"):
        batch._runtime = _runtime()
    with pytest.raises(AttributeError):
        batch.cell_states = torch.zeros(1)
    tensors = verify_bound_complete_catalogue_batch_v6(
        batch, expected_runtime=runtime
    )
    assert tensors["cell_states"].shape == (1, runtime.cell_count, 12)


def test_runtime_version_guard_detects_in_place_tensor_mutation():
    runtime = _runtime()
    batch = runtime.expand(1)
    runtime._tensors["cell_states"].add_(1.0)
    with pytest.raises(ValueError, match="runtime is not intact"):
        verify_bound_complete_catalogue_batch_v6(
            batch, expected_runtime=runtime
        )
    with pytest.raises(TypeError):
        runtime._tensors["cell_states"] = torch.zeros_like(
            runtime._tensors["cell_states"]
        )


@pytest.mark.parametrize("name", ["cell_states", "support_origin_ap_dv_ml_um"])
def test_runtime_content_seal_detects_data_bypass_mutation(name):
    runtime = _runtime()
    batch = runtime.expand(1)
    runtime._tensors[name].data.add_(123.0)
    with pytest.raises(ValueError, match="runtime is not intact"):
        verify_bound_complete_catalogue_batch_v6(
            batch, expected_runtime=runtime
        )


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_runtime_rejects_re_receipted_noninteger_cell_ids(dtype):
    artifact = catalogue()
    tampered = copy.deepcopy(artifact)
    changed = tampered["arrays"]["cell_id_int64"].astype(dtype)
    tampered["arrays"]["cell_id_int64"] = changed
    tampered["tensors"]["cell_id"] = torch.from_numpy(changed)
    tampered["array_receipts"] = {
        name: catalogue_v3._array_receipt(value)
        for name, value in tampered["arrays"].items()
    }
    tampered["receipt_sha256"] = catalogue_v3._hash(
        catalogue_v3.catalogue_receipt_v3(tampered)
    )
    with pytest.raises(ValueError, match="ID dtypes"):
        make_complete_catalogue_runtime_v6(
            tampered,
            expected_catalogue_receipt_sha256=tampered["receipt_sha256"],
            device="cpu",
            dtype=torch.float32,
        )


@pytest.mark.parametrize("receipt", ["", "g" * 64, "A" * 64, "0" * 63])
def test_runtime_requires_a_canonical_sha256_receipt(receipt):
    artifact = catalogue()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        make_complete_catalogue_runtime_v6(
            artifact,
            expected_catalogue_receipt_sha256=receipt,
            device="cpu",
            dtype=torch.float32,
        )
