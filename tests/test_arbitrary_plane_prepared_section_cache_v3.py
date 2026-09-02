import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import test_arbitrary_plane_observation_v3 as observation_fixture
import training.arbitrary_plane_legacy_chain_v3 as legacy_chain_v3
import training.arbitrary_plane_observation_v3 as observation_v3
import training.arbitrary_plane_prepared_section_cache_v3 as cache_v3
import training.arbitrary_plane_training_data_v3 as training_data_v3


@pytest.fixture(autouse=True)
def authenticated_parent(monkeypatch):
    monkeypatch.setattr(
        observation_v3,
        "_verify_section_processing_render_with_mapper_v2",
        observation_fixture._fake_verify_parent,
    )
    monkeypatch.setattr(
        observation_v3,
        "section_processing_render_receipt_v2",
        observation_fixture._parent_receipt,
    )
    monkeypatch.setattr(
        observation_v3.section_processing,
        "_accepted_field",
        lambda plan: lambda points, return_gradient=False: np.zeros_like(points),
    )


@pytest.fixture
def i_cache_directory():
    base = Path("I:/AnatomyTracker/joint-model/.pytest-prepared-section-cache-v3")
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="case-", dir=base))
    try:
        yield path
    finally:
        if path.exists() and path.parent == base:
            shutil.rmtree(path)


def _spec():
    return {
        "support_root_seed": "0x535550504f525401",
        "split": "train",
        "split_index": 1,
        "animal_index": 7,
        "animal_id": "animal-007",
        "section_index": 3,
        "plane_stratum": "general_oblique",
        "specimen_id": "specimen-007-A",
        "experiment_id": "experiment-2026-007",
        "section_root_seed": "0x53454354494f4e01",
        "section_id": "section-003",
        "section_deformation_mode": "identity",
        "window_root_seed": "0x57494e444f570304",
        "observation_root_seed": observation_fixture.ROOT_SEED,
        "observation_index": 2,
        "modality": "brightfield-nissl-like",
        "realization_index": 0,
    }


def _prepared():
    inputs = observation_fixture._inputs()
    inputs["subject_slab_render"]["receipt_sha256"] = "d" * 64
    inputs["section_processing_plan"]["receipt_sha256"] = "e" * 64
    inputs["processed_render"]["receipt_sha256"] = "f" * 64
    precursor = inputs["precursor"]
    precursor.update(
        {
            "centre_plane_render_id": "centre-plane-render-id",
            "slab_recipe_id": "slab-recipe-id",
            "receipt_sha256": "a" * 64,
            "geometry": {
                "geometry_contract_v3": {"schema_version": "fixture-geometry/v3"},
                "global_reference_grid_id": "global-reference-grid-id",
            },
        }
    )
    subject_plan = MappingProxyType(
        {
            "subject_deformation_plan_id": "b" * 64,
            "synthetic_animal_id": "synthetic-animal-007",
        }
    )
    parent_authentication = observation_v3.authenticate_observation_parent_v3(
        inputs["processed_render"],
        inputs["subject_slab_render"],
        inputs["section_processing_plan"],
        inputs["prepared_context"],
        precursor,
        subject_plan=subject_plan,
    )
    spec = _spec()
    prepared = {
        "schema_version": training_data_v3.PREPARED_TRAINING_SECTION_V3_SCHEMA,
        "algorithm": training_data_v3.PREPARED_TRAINING_SECTION_V3_ALGORITHM,
        "implementation_source_sha256": training_data_v3._prepared_source_hashes(),
        "preparation_spec": training_data_v3._preparation_spec(spec),
        "point_batch_size": training_data_v3.DEFAULT_V3_POINT_BATCH_SIZE,
        "legacy_chain_adapter_v3": legacy_chain_v3.adapter_receipt_v3(precursor),
        "subject_plan": subject_plan,
        "support_resolution": {
            "receipt_sha256": "c" * 64,
            "byte_exact_storage_probe": np.asfortranarray(
                np.array(
                    [[0.0, -0.0, np.nan], [1.25, -2.5, 3.75]], dtype=">f4"
                )
            ),
        },
        "precursor": precursor,
        "subject_slab": inputs["subject_slab_render"],
        "section_plan": inputs["section_processing_plan"],
        "section_render": inputs["processed_render"],
        "parent_authentication_v3": parent_authentication,
    }
    prepared["prepared_training_section_id"] = (
        training_data_v3.acquisition._payload_sha256(
            {
                "domain": training_data_v3.PREPARED_TRAINING_SECTION_V3_SCHEMA,
                "preparation_spec": prepared["preparation_spec"],
                "legacy_chain_adapter_v3": prepared["legacy_chain_adapter_v3"],
                "subject_deformation_plan_id": subject_plan[
                    "subject_deformation_plan_id"
                ],
                "parent_authentication_receipt_sha256": parent_authentication[
                    "receipt_sha256"
                ],
            }
        )
    )
    prepared["receipt_sha256"] = training_data_v3.acquisition._payload_sha256(
        training_data_v3.prepared_training_section_receipt_v3(prepared)
    )
    return prepared, inputs["prepared_context"], spec


def _assert_arrays_equal(left, right):
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(
            right
        ).tobytes()
        return
    if isinstance(left, Mapping):
        assert type(left) is type(right)
        assert set(left) == set(right)
        for key in left:
            _assert_arrays_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_arrays_equal(first, second)
    else:
        assert left == right


def _rewrite_manifest(path, mutate):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["receipt_sha256"] = cache_v3._hash_json(
        cache_v3._manifest_payload(manifest)
    )
    path.write_text(cache_v3._canonical_json(manifest) + "\n", encoding="utf-8")


def test_roundtrip_verifies_and_yields_one_byte_identical_descendant(
    i_cache_directory,
):
    prepared, context, spec = _prepared()
    baseline = training_data_v3.make_training_bundle_from_prepared_section_v3(
        prepared, context, spec
    )
    manifest = cache_v3.save_prepared_training_section_v3(
        i_cache_directory, prepared, context
    )
    loaded = cache_v3.load_prepared_training_section_v3(
        i_cache_directory,
        prepared["receipt_sha256"],
        context,
        expected_manifest_receipt_sha256=manifest["receipt_sha256"],
    )
    assert training_data_v3.verify_prepared_training_section_v3(loaded, context)
    _assert_arrays_equal(prepared, loaded)
    replay = training_data_v3.make_training_bundle_from_prepared_section_v3(
        loaded, context, spec
    )
    assert baseline["receipt_sha256"] == replay["receipt_sha256"]
    assert baseline["training_row"]["receipt_sha256"] == replay["training_row"][
        "receipt_sha256"
    ]
    for stage in ("observation", "training_row"):
        for name, array in baseline[stage]["arrays"].items():
            assert np.array_equal(array, replay[stage]["arrays"][name], equal_nan=True)
    with pytest.raises(FileExistsError, match="overwrite refused"):
        cache_v3.save_prepared_training_section_v3(
            i_cache_directory, prepared, context
        )


@pytest.mark.parametrize("damage", ["array", "metadata", "truncation"])
def test_array_metadata_and_truncation_tamper_are_rejected(
    i_cache_directory, damage
):
    prepared, context, _ = _prepared()
    manifest = cache_v3.save_prepared_training_section_v3(
        i_cache_directory, prepared, context
    )
    entry = (
        i_cache_directory
        / cache_v3.PREPARED_SECTION_NAMESPACE
        / prepared["receipt_sha256"]
    )
    if damage == "metadata":
        with (entry / "metadata.json").open("ab") as handle:
            handle.write(b"tamper")
    elif damage == "truncation":
        arrays_path = entry / "arrays.npz"
        with arrays_path.open("r+b") as handle:
            handle.truncate(max(arrays_path.stat().st_size // 2, 1))
    else:
        arrays_path = entry / "arrays.npz"
        with np.load(arrays_path, allow_pickle=False) as stored:
            arrays = {name: np.array(stored[name], copy=True) for name in stored.files}
        first = sorted(arrays)[0]
        arrays[first].view(np.uint8).reshape(-1)[0] ^= np.uint8(1)
        with arrays_path.open("wb") as handle:
            np.savez(handle, **arrays)
        _rewrite_manifest(
            entry / "manifest.json",
            lambda value: value.update(
                {"arrays_file_sha256": cache_v3._file_sha256(arrays_path)}
            ),
        )
    with pytest.raises(ValueError, match="file hash|array receipt"):
        cache_v3.load_prepared_training_section_v3(
            i_cache_directory,
            prepared["receipt_sha256"],
            context,
            expected_manifest_receipt_sha256=(
                None if damage == "array" else manifest["receipt_sha256"]
            ),
        )


@pytest.mark.parametrize("damage", ["source", "schema", "receipt", "escape"])
def test_manifest_source_schema_receipt_and_path_escape_are_rejected(
    i_cache_directory, damage
):
    prepared, context, _ = _prepared()
    cache_v3.save_prepared_training_section_v3(i_cache_directory, prepared, context)
    entry = (
        i_cache_directory
        / cache_v3.PREPARED_SECTION_NAMESPACE
        / prepared["receipt_sha256"]
    )

    def mutate(manifest):
        if damage == "source":
            manifest["implementation_source_sha256"][
                "arbitrary_plane_prepared_section_cache_v3.py"
            ] = "0" * 64
        elif damage == "schema":
            manifest["prepared_schema_version"] = "wrong-schema"
        elif damage == "receipt":
            manifest["prepared_receipt_sha256"] = "0" * 64
        else:
            manifest["metadata_relative_path"] = "../metadata.json"

    _rewrite_manifest(entry / "manifest.json", mutate)
    with pytest.raises(ValueError, match="source/schema/receipt|escapes"):
        cache_v3.load_prepared_training_section_v3(
            i_cache_directory, prepared["receipt_sha256"], context
        )


def test_i_only_and_streaming_audit(i_cache_directory, monkeypatch):
    prepared, context, _ = _prepared()
    cache_v3.save_prepared_training_section_v3(i_cache_directory, prepared, context)
    second, _, _ = _prepared()
    second["point_batch_size"] += 1
    second["receipt_sha256"] = training_data_v3.acquisition._payload_sha256(
        training_data_v3.prepared_training_section_receipt_v3(second)
    )
    cache_v3.save_prepared_training_section_v3(i_cache_directory, second, context)
    calls = []
    original = cache_v3._load_prepared_section_directory

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result[0]["receipt_sha256"])
        return result

    monkeypatch.setattr(cache_v3, "_load_prepared_section_directory", counted)
    audit = cache_v3.audit_prepared_training_section_cache_v3(
        i_cache_directory, context
    )
    assert calls == sorted(
        [prepared["receipt_sha256"], second["receipt_sha256"]]
    )
    assert audit["prepared_section_count"] == 2
    assert audit["all_prepared_sections_authenticated"]
    with pytest.raises(ValueError, match="only on I"):
        cache_v3.save_prepared_training_section_v3(
            "C:/forbidden-prepared-section-cache-v3", prepared, context
        )
    with pytest.raises(ValueError, match="content address"):
        cache_v3.load_prepared_training_section_v3(
            i_cache_directory, "../escape", context
        )
