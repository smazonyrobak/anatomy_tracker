"""Deterministic truth-centred candidate banks for tractable cold-start training."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components


TRAINING_BANK_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-training-bank/v3"
TRAINING_BANK_V3_ALGORITHM = (
    "truth-plus-joint-angle-offset-roll-hard-negatives-and-uniform-global/v3"
)
COMPLETE_CATALOGUE_SCOPE = "complete catalogue posterior/inference scope"
TRAINING_CANDIDATE_BANK_SCOPE = (
    "truth-centred sampled training bank; not a complete posterior or inference path"
)


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return _json(value.item())
    return value


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            _json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _tensor_receipt(value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def _seed(root_seed, row_identity, catalogue_id):
    digest = hashlib.blake2b(digest_size=8, person=b"APBANKV3")
    for value in (str(root_seed), _hash(row_identity), str(catalogue_id)):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "big")


def verify_training_candidate_bank_receipt_v3(
    receipt,
    *,
    expected_catalogue_id,
    expected_catalogue_receipt_sha256,
    expected_training_row_id,
    expected_training_row_receipt_sha256=None,
    expected_training_row_identity_sha256=None,
):
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    selected = payload.get("selected_full_catalogue_indices", [])
    if (
        receipt.get("receipt_sha256") != _hash(payload)
        or payload.get("schema_version") != TRAINING_BANK_V3_SCHEMA
        or payload.get("algorithm") != TRAINING_BANK_V3_ALGORITHM
        or payload.get("catalogue_id") != expected_catalogue_id
        or payload.get("catalogue_receipt_sha256")
        != expected_catalogue_receipt_sha256
        or payload.get("training_row_id") != expected_training_row_id
        or not isinstance(payload.get("training_row_receipt_sha256"), str)
        or not payload["training_row_receipt_sha256"]
        or not isinstance(payload.get("training_row_identity_sha256"), str)
        or len(payload["training_row_identity_sha256"]) != 64
        or bool(set(payload["training_row_identity_sha256"].lower()) - set("0123456789abcdef"))
        or (
            expected_training_row_receipt_sha256 is not None
            and payload["training_row_receipt_sha256"]
            != expected_training_row_receipt_sha256
        )
        or (
            expected_training_row_identity_sha256 is not None
            and payload["training_row_identity_sha256"]
            != expected_training_row_identity_sha256
        )
        or payload.get("learned_dependencies") != []
        or payload.get("inference_scope") is not False
        or payload.get("local_truth_catalogue_index") != 0
        or len(selected) != payload.get("bank_size")
        or len(set(selected)) != len(selected)
        or not selected
        or selected[0] != payload.get("full_truth_catalogue_index")
        or set(payload.get("selected_training_tensor_receipts", {}))
        != {
            "cell_states",
            "cell_log_mass",
            "representation_log_weight",
            "representation_to_canonical_raster_affine",
        }
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < payload.get("full_catalogue_cell_count", 0)
            for value in selected
        )
    ):
        raise ValueError("training candidate-bank receipt is invalid")


def verify_training_catalogue_batch_v3(
    batch,
    *,
    expected_catalogue_id,
    expected_catalogue_receipt_sha256,
    expected_full_catalogue_cell_count,
):
    """Verify complete or truth-sampled catalogue tensors before optimization."""
    row_count = len(batch.get("row_identity", ()))
    local_count = int(torch.as_tensor(batch.get("cell_id", ())).numel())
    if (
        batch.get("catalogue_id") != expected_catalogue_id
        or batch.get("catalogue_receipt_sha256")
        != expected_catalogue_receipt_sha256
        or batch.get("full_catalogue_cell_count")
        != expected_full_catalogue_cell_count
        or row_count < 1
        or local_count < 1
        or not torch.equal(
            torch.as_tensor(batch["cell_id"]).detach().cpu(),
            torch.arange(local_count),
        )
        or tuple(batch["cell_states"].shape[:2]) != (row_count, local_count)
        or tuple(batch["cell_log_mass"].shape[:2]) != (row_count, local_count)
        or tuple(batch["representation_log_weight"].shape[:2])
        != (row_count, local_count)
        or tuple(batch["representation_to_canonical_raster_affine"].shape[:2])
        != (row_count, local_count)
    ):
        raise ValueError("training catalogue batch binding or tensor shape is invalid")

    scope = batch.get("catalogue_scope")
    truth_index = torch.as_tensor(batch["truth_catalogue_cell_index"]).detach().cpu()
    truth_source = torch.as_tensor(
        batch["truth_catalogue_cell_source_index"]
    ).detach().cpu()
    truth_id = torch.as_tensor(batch["truth_catalogue_cell_id"]).detach().cpu()
    if truth_index.shape != (row_count,) or truth_source.shape != (row_count,) or truth_id.shape != (row_count,):
        raise ValueError("training catalogue truth mappings are malformed")

    if scope == COMPLETE_CATALOGUE_SCOPE:
        if (
            local_count != expected_full_catalogue_cell_count
            or batch.get("training_candidate_bank_receipts", []) != []
            or "selected_full_catalogue_indices" in batch
            or not torch.equal(truth_index, truth_source)
            or not torch.equal(truth_index, truth_id)
        ):
            raise ValueError("complete-catalogue training batch is not complete")
        return True

    if scope != TRAINING_CANDIDATE_BANK_SCOPE:
        raise ValueError("training catalogue scope is missing or unknown")
    receipts = batch.get("training_candidate_bank_receipts", [])
    selected = batch.get("selected_full_catalogue_indices")
    if (
        batch.get("training_candidate_bank_scope") != TRAINING_CANDIDATE_BANK_SCOPE
        or len(receipts) != row_count
        or selected is None
        or tuple(selected.shape) != (row_count, local_count)
        or not torch.equal(truth_index, torch.zeros(row_count, dtype=torch.long))
        or not torch.equal(truth_id, torch.zeros(row_count, dtype=torch.long))
        or not torch.equal(selected[:, 0].detach().cpu(), truth_source)
    ):
        raise ValueError("sampled training-bank mapping is invalid")
    for row_index, (receipt, identity) in enumerate(
        zip(receipts, batch["row_identity"])
    ):
        verify_training_candidate_bank_receipt_v3(
            receipt,
            expected_catalogue_id=expected_catalogue_id,
            expected_catalogue_receipt_sha256=expected_catalogue_receipt_sha256,
            expected_training_row_id=identity["training_row_id"],
            expected_training_row_receipt_sha256=identity[
                "training_row_receipt_sha256"
            ],
            expected_training_row_identity_sha256=_hash(identity),
        )
        observed_receipts = {
            name: _tensor_receipt(batch[name][row_index])
            for name in (
                "cell_states",
                "cell_log_mass",
                "representation_log_weight",
                "representation_to_canonical_raster_affine",
            )
        }
        if (
            selected[row_index].detach().cpu().tolist()
            != receipt["selected_full_catalogue_indices"]
            or receipt["full_catalogue_cell_count"]
            != expected_full_catalogue_cell_count
            or receipt["bank_size"] != local_count
            or receipt["full_truth_catalogue_index"] != int(truth_source[row_index])
            or receipt["local_truth_catalogue_index"] != int(truth_id[row_index])
            or receipt["selected_training_tensor_receipts"] != observed_receipts
        ):
            raise ValueError("sampled training-bank tensors differ from their receipt")
    return True


def _stable_unique_ranked(rankings, truth_index, structured_count):
    selected = [int(truth_index)]
    cursors = [0] * len(rankings)
    while len(selected) < structured_count + 1:
        changed = False
        for ranking_index, ranking in enumerate(rankings):
            while cursors[ranking_index] < len(ranking):
                candidate = int(ranking[cursors[ranking_index]])
                cursors[ranking_index] += 1
                if candidate not in selected:
                    selected.append(candidate)
                    changed = True
                    break
            if len(selected) >= structured_count + 1:
                break
        if not changed:
            break
    return selected


def make_training_candidate_batch_v3(
    full_batch,
    catalogue,
    *,
    bank_size,
    root_seed,
):
    """Select a deterministic supervision-only bank; full inference stays unchanged."""
    if full_batch.get("data_role") != "development-training":
        raise ValueError("training candidate banks are development-only")
    cell_count = int(catalogue["counts"]["cell_count"])
    verify_training_catalogue_batch_v3(
        full_batch,
        expected_catalogue_id=catalogue["catalogue_id"],
        expected_catalogue_receipt_sha256=catalogue["receipt_sha256"],
        expected_full_catalogue_cell_count=cell_count,
    )
    if (
        not isinstance(bank_size, int)
        or isinstance(bank_size, bool)
        or not 5 <= bank_size <= cell_count
        or full_batch["cell_states"].shape[1] != cell_count
        or full_batch["cell_id"].shape != (cell_count,)
        or not torch.equal(
            full_batch["cell_id"].detach().cpu(), torch.arange(cell_count)
        )
    ):
        raise ValueError("bank size and complete contiguous catalogue are invalid")
    truth_full = full_batch["truth_catalogue_cell_index"].detach().cpu().numpy()
    if truth_full.shape != (len(full_batch["row_identity"]),):
        raise ValueError("one full-catalogue truth index is required per row")

    normals = np.asarray(
        catalogue["arrays"]["cell_normal_ap_dv_ml_float64"], dtype=np.float64
    )
    offsets = np.asarray(
        catalogue["arrays"]["cell_signed_offset_um_float64"], dtype=np.float64
    )
    states = torch.as_tensor(
        catalogue["arrays"]["cell_states_float64"], dtype=torch.float64
    )
    candidate_u = states[:, 3:6].numpy()
    truth_state = full_batch["truth_state"].detach().cpu().to(torch.float64)
    truth_center, truth_frame, _ = full_frame_state_to_components(truth_state)
    truth_center = truth_center.numpy()
    truth_normal = truth_frame[:, :, 2].numpy()
    truth_u = truth_frame[:, :, 0].numpy()
    support_origin = np.asarray(
        catalogue["support_geometry"]["support_origin_ap_dv_ml_um"],
        dtype=np.float64,
    )
    normal_scale = max(
        float(
            catalogue["coverage_audit"][
                "max_observed_rp2_angular_covering_radius_rad"
            ]
        ),
        1.0e-3,
    )
    offset_table = np.asarray(
        catalogue["arrays"]["normal_offset_table_um_float64"], dtype=np.float64
    )
    offset_scale = max(float(np.median(np.abs(np.diff(offset_table, axis=1)))), 1.0)
    roll_scale = math.pi / int(catalogue["counts"]["roll_count"])
    global_count = max(1, (bank_size - 1) // 5)
    structured_count = bank_size - 1 - global_count

    selected_rows = []
    receipt_payloads = []
    for row, truth_index, center, normal, u in zip(
        full_batch["row_identity"], truth_full, truth_center, truth_normal, truth_u
    ):
        dot = normals @ normal
        sign = np.where(dot < 0.0, -1.0, 1.0)
        angle = np.arccos(np.clip(np.abs(dot), 0.0, 1.0)) / normal_scale
        truth_offset = float((center - support_origin) @ normal)
        offset = np.abs(truth_offset * sign - offsets) / offset_scale
        aligned_u = candidate_u * sign[:, None]
        roll = np.arccos(np.clip(aligned_u @ u, -1.0, 1.0)) / roll_scale
        joint = np.square(angle) + np.square(offset) + np.square(roll)
        rankings = tuple(
            np.argsort(score, kind="stable")
            for score in (
                joint,
                angle + 0.05 * offset + 0.05 * roll,
                offset + 0.05 * angle + 0.05 * roll,
                roll + 0.05 * angle + 0.05 * offset,
            )
        )
        selected = _stable_unique_ranked(
            rankings, int(truth_index), structured_count
        )
        remaining = np.setdiff1d(
            np.arange(cell_count, dtype=np.int64),
            np.asarray(selected, dtype=np.int64),
            assume_unique=False,
        )
        rng = np.random.Generator(
            np.random.PCG64DXSM(_seed(root_seed, row, catalogue["catalogue_id"]))
        )
        fill_count = bank_size - len(selected)
        if fill_count:
            selected.extend(
                int(value)
                for value in rng.choice(remaining, size=fill_count, replace=False)
            )
        selected_array = np.asarray(selected, dtype=np.int64)
        selected_rows.append(selected_array)
        payload = {
            "schema_version": TRAINING_BANK_V3_SCHEMA,
            "algorithm": TRAINING_BANK_V3_ALGORITHM,
            "catalogue_id": catalogue["catalogue_id"],
            "catalogue_receipt_sha256": catalogue["receipt_sha256"],
            "training_row_id": row["training_row_id"],
            "training_row_receipt_sha256": row[
                "training_row_receipt_sha256"
            ],
            "training_row_identity_sha256": _hash(row),
            "root_seed": str(root_seed),
            "full_truth_catalogue_index": int(truth_index),
            "full_catalogue_cell_count": cell_count,
            "local_truth_catalogue_index": 0,
            "bank_size": bank_size,
            "structured_count": structured_count,
            "global_uniform_count": bank_size - 1 - structured_count,
            "selected_full_catalogue_indices": selected_array.tolist(),
            "learned_dependencies": [],
            "inference_scope": False,
        }
        receipt_payloads.append(payload)

    selected = torch.as_tensor(
        np.stack(selected_rows), device=full_batch["cell_states"].device
    )

    def gather(value):
        index = selected.reshape(
            *selected.shape, *([1] * (value.ndim - 2))
        ).expand(*selected.shape, *value.shape[2:])
        return torch.gather(value, 1, index)

    output = dict(full_batch)
    output["cell_states"] = gather(full_batch["cell_states"])
    selected_log_mass = gather(full_batch["cell_log_mass"])
    output["cell_log_mass"] = selected_log_mass - torch.logsumexp(
        selected_log_mass, dim=1, keepdim=True
    )
    output["representation_log_weight"] = gather(
        full_batch["representation_log_weight"]
    )
    output["representation_to_canonical_raster_affine"] = gather(
        full_batch["representation_to_canonical_raster_affine"]
    )
    output["cell_id"] = torch.arange(bank_size, device=selected.device)
    output["truth_catalogue_cell_index"] = torch.zeros(
        selected.shape[0], device=selected.device, dtype=torch.long
    )
    output["truth_catalogue_cell_source_index"] = torch.as_tensor(
        truth_full, device=selected.device, dtype=torch.long
    )
    output["truth_catalogue_cell_id"] = output["truth_catalogue_cell_index"].clone()
    receipts = []
    for row_index, payload in enumerate(receipt_payloads):
        payload["selected_training_tensor_receipts"] = {
            name: _tensor_receipt(output[name][row_index])
            for name in (
                "cell_states",
                "cell_log_mass",
                "representation_log_weight",
                "representation_to_canonical_raster_affine",
            )
        }
        receipts.append({**payload, "receipt_sha256": _hash(payload)})
    output["catalogue_scope"] = TRAINING_CANDIDATE_BANK_SCOPE
    output["training_candidate_bank_scope"] = TRAINING_CANDIDATE_BANK_SCOPE
    output["training_candidate_bank_receipts"] = receipts
    output["selected_full_catalogue_indices"] = selected
    return output
