import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pytest
import torch

import test_arbitrary_plane_row_cache_v4 as cache_fixture
import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_finite_row_binding_v6 as binding_v6
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_row_cache_v4 as cache_v4


@pytest.fixture
def frozen_cache():
    parent = Path("I:/AnatomyTracker/test_tmp/finite_row_binding_v6")
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / uuid.uuid4().hex
    generator_binding = cache_fixture._binding()
    rows = [
        cache_fixture._row(0, 25.0),
        cache_fixture._row(1, 100.0),
    ]
    cache_v4.initialize_training_row_cache_v4(
        root, generator_binding=generator_binding
    )
    cache_v4.append_training_rows_v4(root, rows)
    manifest = cache_v4.freeze_training_row_cache_v4(root)
    yield root, manifest, rows
    shutil.rmtree(root, ignore_errors=True)


def _rewrite_manifest(root, mutate):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["receipt_sha256"] = cache_v4._hash_json(
        cache_v4._manifest_payload(manifest)
    )
    cache_v4._atomic_json(path, manifest)
    return manifest


def _rehash_row(row):
    row["receipt_sha256"] = acquisition_v2._payload_sha256(
        psf_v4.training_row_receipt_v4(row)
    )
    return row


def test_pure_loader_authenticates_frozen_manifest_selected_rows_and_psf_tensors(
    frozen_cache,
):
    root, expected_manifest, expected_rows = frozen_cache
    manifest = binding_v6.load_frozen_row_cache_manifest_v6(
        root,
        expected_manifest_receipt_sha256=expected_manifest["receipt_sha256"],
    )
    payload = binding_v6.load_frozen_training_rows_v6(
        root,
        [1, 0],
        expected_manifest_receipt_sha256=expected_manifest["receipt_sha256"],
    )
    assert manifest == expected_manifest
    assert payload["training_data_manifest_receipt_sha256"] == manifest[
        "receipt_sha256"
    ]
    assert payload["cache_manifest_receipt_sha256"] == manifest["receipt_sha256"]
    assert payload["row_indices"] == [1, 0]
    assert payload["training_row_ids"] == [
        expected_rows[1]["training_row_id"],
        expected_rows[0]["training_row_id"],
    ]
    assert payload["training_row_receipts_sha256"] == [
        expected_rows[1]["receipt_sha256"],
        expected_rows[0]["receipt_sha256"],
    ]
    assert payload["selection_receipt_sha256"] == (
        binding_v6.frozen_row_selection_receipt_v6(payload)
    )
    assert payload["selection_receipt_sha256"] != payload[
        "training_data_manifest_receipt_sha256"
    ]
    other_selection = binding_v6.load_frozen_training_rows_v6(
        root,
        [0],
        expected_manifest_receipt_sha256=expected_manifest["receipt_sha256"],
    )
    assert other_selection["training_data_manifest_receipt_sha256"] == payload[
        "training_data_manifest_receipt_sha256"
    ]
    assert other_selection["selection_receipt_sha256"] != payload[
        "selection_receipt_sha256"
    ]
    assert "receipt_sha256" not in payload
    assert all(
        array.flags.c_contiguous
        for row in payload["rows"]
        for array in row["arrays"].values()
    )
    tensors = binding_v6.finite_psf_tensors_from_training_row_v6(
        payload["rows"][0],
        finite_psf_capability=manifest["finite_psf_capability"],
        dtype=torch.float64,
    )
    assert tensors["axial_offsets_um"].shape == (1, 9)
    assert tensors["axial_weights"].shape == (1, 9)
    assert tensors["axial_offsets_um"].dtype == torch.float64
    assert torch.equal(
        tensors["axial_offsets_um"],
        torch.tensor(
            payload["rows"][0]["finite_psf_contract"]["axial_offsets_um"],
            dtype=torch.float64,
        )[None],
    )
    assert torch.equal(
        tensors["axial_weights"].sum(dim=1), torch.ones(1, dtype=torch.float64)
    )


def test_centre_plane_ablation_produces_one_sample_psf_tensors():
    row = cache_fixture._row(0, 0.0, "centre_plane_ablation")
    tensors = binding_v6.finite_psf_tensors_from_training_row_v6(
        row,
        finite_psf_capability=psf_v4.finite_psf_model_capability_v4(),
    )
    assert torch.equal(tensors["axial_offsets_um"], torch.zeros((1, 1)))
    assert torch.equal(tensors["axial_weights"], torch.ones((1, 1)))


def test_import_is_independent_of_frozen_writers_generators_and_training_banks():
    code = """
import sys
import training.arbitrary_plane_finite_row_binding_v6

forbidden = [
    name
    for name in sys.modules
    if name.startswith("training.")
    and any(
        fragment in name
        for fragment in (
            "arbitrary_plane_psf_v4",
            "arbitrary_plane_row_cache_v4",
            "arbitrary_plane_training_row_v3",
            "arbitrary_plane_observation_v3",
            "arbitrary_plane_legacy_chain_v3",
            "generator",
            "candidate_bank",
            "training_bank",
            "staged_training",
        )
    )
]
assert not forbidden, forbidden
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    tree = ast.parse(Path(binding_v6.__file__).read_text(encoding="utf-8"))
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(name.startswith("training") for name in imports)


def test_loader_requires_i_drive_trusted_receipt_and_frozen_status(frozen_cache):
    root, manifest, _ = frozen_cache
    with pytest.raises(ValueError, match="only on I"):
        binding_v6.load_frozen_training_rows_v6(
            r"C:\forbidden-finite-row-cache",
            expected_manifest_receipt_sha256=manifest["receipt_sha256"],
        )
    with pytest.raises(ValueError, match="failed authentication"):
        binding_v6.load_frozen_row_cache_manifest_v6(
            root,
            expected_manifest_receipt_sha256="0" * 64,
        )
    changed = _rewrite_manifest(
        root,
        lambda value: value.update({"status": "OPEN", "freeze_audit": None}),
    )
    with pytest.raises(ValueError, match="failed authentication"):
        binding_v6.load_frozen_row_cache_manifest_v6(
            root,
            expected_manifest_receipt_sha256=changed["receipt_sha256"],
        )


@pytest.mark.parametrize("target", ["metadata", "arrays"])
def test_selected_row_file_hash_tamper_is_rejected(frozen_cache, target):
    root, manifest, _ = frozen_cache
    record = manifest["rows"][0]
    path = root / record[f"{target}_relative_path"]
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="file hash differs"):
        binding_v6.load_frozen_training_rows_v6(
            root,
            [0],
            expected_manifest_receipt_sha256=manifest["receipt_sha256"],
        )


def test_re_receipted_freeze_audit_and_row_record_tamper_are_rejected(frozen_cache):
    root, _, _ = frozen_cache
    changed = _rewrite_manifest(
        root,
        lambda value: value["freeze_audit"].update(
            {"ordered_finite_psf_sha256": "0" * 64}
        ),
    )
    with pytest.raises(ValueError, match="cache audit changed"):
        binding_v6.load_frozen_row_cache_manifest_v6(
            root,
            expected_manifest_receipt_sha256=changed["receipt_sha256"],
        )

    cache_v4._atomic_json(root / "manifest.json", frozen_cache[1])
    changed = _rewrite_manifest(
        root,
        lambda value: value["rows"][0].update(
            {"selected_mode": "smart-brush-accurate"}
        ),
    )
    with pytest.raises(ValueError, match="ordered manifest record"):
        binding_v6.load_frozen_training_rows_v6(
            root,
            [0],
            expected_manifest_receipt_sha256=changed["receipt_sha256"],
        )


def test_complete_array_receipts_and_training_row_receipt_are_rechecked(frozen_cache):
    _, manifest, rows = frozen_cache
    changed_array = copy.deepcopy(rows[0])
    changed_array["arrays"]["model_input_channels_float32"][0, 0, 0] += 0.25
    with pytest.raises(ValueError, match="receipt, arrays"):
        binding_v6.verify_finite_training_row_v6(
            changed_array,
            finite_psf_capability=manifest["finite_psf_capability"],
        )
    changed_receipt = copy.deepcopy(rows[0])
    changed_receipt["selected_mode"] = "smart-brush-accurate"
    with pytest.raises(ValueError, match="receipt, arrays"):
        binding_v6.verify_finite_training_row_v6(
            changed_receipt,
            finite_psf_capability=manifest["finite_psf_capability"],
        )


def test_re_receipted_psf_capability_schedule_and_source_tamper_are_rejected(
    frozen_cache,
):
    _, manifest, rows = frozen_cache
    capability = copy.deepcopy(manifest["finite_psf_capability"])
    capability["unknown_thickness_policy"] = "accept"
    capability["receipt_sha256"] = acquisition_v2._payload_sha256(
        {key: value for key, value in capability.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="capability"):
        binding_v6.verify_finite_training_row_v6(
            rows[0], finite_psf_capability=capability
        )

    schedule = copy.deepcopy(rows[0])
    contract = schedule["finite_psf_contract"]
    contract["axial_offsets_um"][0] -= 1.0
    contract["axial_offsets_um"][-1] += 1.0
    contract["finite_psf_sha256"] = acquisition_v2._payload_sha256(
        psf_v4._finite_psf_payload(contract)
    )
    schedule["upstream_reference"]["finite_psf_sha256"] = contract[
        "finite_psf_sha256"
    ]
    schedule["training_row_id"] = acquisition_v2._payload_sha256(
        {
            "domain": psf_v4.TRAINING_ROW_V4_SCHEMA,
            "synthetic_realization_id": schedule["synthetic_realization_id"],
            "array_receipts": schedule["array_receipts"],
            "finite_psf_sha256": contract["finite_psf_sha256"],
            "slab_observation_v4_receipt_sha256": contract[
                "slab_observation_v4_receipt_sha256"
            ],
        }
    )
    _rehash_row(schedule)
    with pytest.raises(ValueError, match="thickness or axial schedule"):
        binding_v6.verify_finite_training_row_v6(
            schedule,
            finite_psf_capability=manifest["finite_psf_capability"],
        )

    source = copy.deepcopy(rows[0])
    source["upstream_reference"]["implementation_source_sha256"][
        "fixture.py"
    ] = "f" * 64
    _rehash_row(source)
    with pytest.raises(ValueError, match="implementation differs"):
        binding_v6.verify_finite_training_row_v6(
            source,
            finite_psf_capability=manifest["finite_psf_capability"],
            cache_manifest=manifest,
        )

    slab = copy.deepcopy(rows[0])
    slab["upstream_reference"]["slab_observation_v4_receipt_sha256"] = "e" * 64
    _rehash_row(slab)
    with pytest.raises(ValueError, match="source binding"):
        binding_v6.verify_finite_training_row_v6(
            slab,
            finite_psf_capability=manifest["finite_psf_capability"],
        )
