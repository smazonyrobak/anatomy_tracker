import copy

import numpy as np
import pytest

import arbitrary_plane_production_v3_fixtures as fixture
import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
import training.arbitrary_plane_training_row_v3 as training_row_v3


def _initialize(path):
    return row_cache_v3.initialize_training_row_cache_v3(
        path,
        generator_binding=fixture.generator_binding(),
        generation_config={"row_count": 2, "modes": ["raw", "black", "imperfect"]},
        seed_record={"root_seed": "0xabc", "subject_seed": "0xdef"},
    )


def test_resumable_cache_freezes_exact_rows_and_lineage(tmp_path):
    cache = tmp_path / "rows"
    initial = _initialize(cache)
    assert initial["status"] == row_cache_v3.OPEN_CACHE_STATUS
    first = row_cache_v3.append_training_rows_v3(cache, [fixture.row(0)])
    assert first["row_count"] == 1
    resumed = row_cache_v3.append_training_rows_v3(
        cache, [fixture.row(0), fixture.row(1)]
    )
    assert resumed["row_count"] == 2
    frozen = row_cache_v3.freeze_training_row_cache_v3(cache)
    loaded = row_cache_v3.load_training_rows_v3(
        cache, [1, 0], expected_manifest_receipt_sha256=frozen["receipt_sha256"]
    )
    assert [row["training_row_id"] for row in loaded] == ["row-1", "row-0"]
    assert loaded[0]["lineage"]["animal_id"] == "animal-1"
    assert loaded[0]["lineage"]["specimen_id"] == "specimen-1"
    assert loaded[0]["lineage"]["experiment_id"] == "experiment-1"
    assert np.array_equal(
        loaded[0]["arrays"]["model_input_channels_float32"],
        fixture.row(1)["arrays"]["model_input_channels_float32"],
    )
    audit = row_cache_v3.audit_training_row_cache_v3(cache)
    assert audit["all_rows_authenticated"]
    assert audit["row_count"] == 2
    with pytest.raises(ValueError, match="frozen"):
        row_cache_v3.append_training_rows_v3(cache, [fixture.row(2)])


def test_cache_rejects_learned_dependencies_and_non_development_rows(tmp_path):
    cache = tmp_path / "rows"
    _initialize(cache)
    contaminated = copy.deepcopy(fixture.row(0))
    contaminated["prior_feature_dependencies"] = ["legacy-embedding.npy"]
    with pytest.raises(ValueError, match="learned-dependent"):
        row_cache_v3.append_training_rows_v3(cache, [contaminated])
    with pytest.raises(ValueError, match="non-development"):
        row_cache_v3.append_training_rows_v3(
            cache, [fixture.row(0, split="final-test")]
        )
    stale_gauge = copy.deepcopy(fixture.row(0))
    stale_gauge["deformation_pose_gauge_reference"]["algorithm"] = "legacy-gauge"
    stale_gauge["receipt_sha256"] = acquisition_v2._payload_sha256(
        training_row_v3.training_row_receipt_v3(stale_gauge)
    )
    with pytest.raises(ValueError, match="gauge reference is invalid or stale"):
        row_cache_v3.append_training_rows_v3(cache, [stale_gauge])


def test_cache_detects_asset_tampering_and_rejects_non_i_drive(tmp_path):
    cache = tmp_path / "rows"
    _initialize(cache)
    manifest = row_cache_v3.append_training_rows_v3(cache, [fixture.row(0)])
    arrays_path = cache / manifest["rows"][0]["arrays_relative_path"]
    with arrays_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="file hash"):
        row_cache_v3.load_training_rows_v3(cache)
    with pytest.raises(ValueError, match="only on I"):
        row_cache_v3.initialize_training_row_cache_v3(
            "C:\\forbidden-row-cache-v3",
            generator_binding=fixture.generator_binding(),
            generation_config={"rows": 1},
            seed_record={"seed": 1},
        )


def test_full_audit_streams_records_without_materializing_the_public_row_list(
    tmp_path, monkeypatch
):
    cache = tmp_path / "rows"
    _initialize(cache)
    row_cache_v3.append_training_rows_v3(
        cache, [fixture.row(index) for index in range(12)]
    )
    calls = []
    original = row_cache_v3._load_record

    def counted_load_record(root, record, geometry_gauge_contract):
        calls.append(record["row_index"])
        return original(root, record, geometry_gauge_contract)

    def forbidden_all_row_loader(*args, **kwargs):
        raise AssertionError("full audit must not materialize load_training_rows_v3")

    monkeypatch.setattr(row_cache_v3, "_load_record", counted_load_record)
    monkeypatch.setattr(
        row_cache_v3, "load_training_rows_v3", forbidden_all_row_loader
    )
    audit = row_cache_v3.audit_training_row_cache_v3(cache)
    assert calls == list(range(12))
    assert audit["row_count"] == 12
    assert audit["all_rows_authenticated"]
