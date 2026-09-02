"""Exact catalogue-v3 binding checks without inference or training imports."""

from __future__ import annotations

import torch

import training.arbitrary_plane_catalogue_v3 as catalogue_v3


def verify_catalogue_binding_v3(catalogue):
    arrays = catalogue.get("arrays", {})
    receipts = catalogue.get("array_receipts", {})
    tensors = catalogue.get("tensors", {})
    tensor_to_array = {
        "cell_id": "cell_id_int64",
        "cell_states": "cell_states_float64",
        "cell_log_mass": "cell_log_mass_float64",
        "representation_log_weight": "representation_log_weight_float64",
        "representation_to_canonical_raster_affine": "representation_to_canonical_raster_affine_float64",
    }
    valid = (
        catalogue.get("schema_version") == catalogue_v3.CATALOGUE_V3_SCHEMA
        and set(arrays) == set(receipts)
        and receipts
        and all(
            catalogue_v3._array_receipt(value) == receipts[name]
            for name, value in arrays.items()
        )
        and catalogue.get("receipt_sha256")
        == catalogue_v3._hash(catalogue_v3.catalogue_receipt_v3(catalogue))
        and set(tensors) == set(tensor_to_array)
        and all(
            torch.equal(
                torch.as_tensor(tensors[name]),
                torch.as_tensor(arrays[array_name])[
                    None if name != "cell_id" else slice(None)
                ],
            )
            for name, array_name in tensor_to_array.items()
        )
    )
    if not valid:
        raise ValueError("catalogue arrays or immutable receipt are invalid")
    cell_id = torch.as_tensor(arrays["cell_id_int64"])
    if not torch.equal(cell_id, torch.arange(cell_id.numel())):
        raise ValueError("catalogue cell IDs must be complete, unique, and canonical")
    return True


__all__ = ["verify_catalogue_binding_v3"]
