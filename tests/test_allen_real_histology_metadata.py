import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from training.allen_real_histology_metadata import (
    OFFICIAL_DOCUMENTS,
    _canonical_bytes,
    _official_url,
    _snapshot_root,
    acquire_metadata_snapshot,
    split_for_animal,
    verify_metadata_snapshot,
)


def _response(url: str, payload=None, content: bytes | None = None):
    content = content if content is not None else _canonical_bytes(payload)
    response = Mock(content=content, url=url, headers={"Content-Type": "application/json"})
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _dataset(experiment_id: int, specimen_id: int, animal_id: int, section_id: int) -> dict:
    return {
        "id": experiment_id,
        "specimen_id": specimen_id,
        "specimen": {"id": specimen_id, "donor_id": animal_id, "donor": {"id": animal_id}},
        "products": [{"id": 5, "name": "Mouse Connectivity Projection"}],
        "failed": False,
        "reference_space_id": 9,
        "plane_of_section_id": 1,
        "section_thickness": 100.0,
        "red_channel": "background fluorescence",
        "green_channel": "rAAV",
        "blue_channel": "background fluorescence",
        "alignment3d": {f"tvr_{index:02d}": float(index) for index in range(12)},
        "equalization": {
            "red_lower": 0,
            "red_upper": 700,
            "green_lower": 0,
            "green_upper": 800,
            "blue_lower": 0,
            "blue_upper": 4095,
        },
        "section_images": [
            {
                "id": section_id,
                "section_number": 4,
                "failed": False,
                "width": 3200,
                "height": 2400,
                "resolution": 0.35,
                "alignment2d": {f"tsv_{index:02d}": float(index) for index in range(6)},
            }
        ],
    }


def _fake_get():
    datasets = {
        11: _dataset(11, 101, 7, 1101),
        12: _dataset(12, 102, 7, 1201),
    }

    def get(url, params=None, timeout=60):
        del timeout
        if url.endswith("/data/query.json"):
            payload = {
                "success": True,
                "total_rows": 2,
                "msg": [{"id": 11}, {"id": 12}],
            }
            return _response(f"{url}?q=official-index", payload)
        if "/SectionDataSet/" in url:
            experiment_id = int(url.rsplit("/", 1)[1].split(".")[0])
            payload = {"success": True, "total_rows": 1, "msg": [datasets[experiment_id]]}
            return _response(f"{url}?include=official-metadata", payload)
        if url in OFFICIAL_DOCUMENTS.values():
            return _response(url, content=f"official:{url}".encode("utf-8"))
        raise AssertionError((url, params))

    return get


def test_metadata_snapshot_is_donor_split_immutable_and_contains_no_images(tmp_path: Path):
    output = tmp_path / "allen"
    manifest = acquire_metadata_snapshot(
        output,
        2,
        page_size=2,
        get=_fake_get(),
        retrieved_at_utc="2026-09-02T00:00:00+00:00",
    )
    report = verify_metadata_snapshot(output)
    experiments = [json.loads(line) for line in (output / "experiments.jsonl").read_text().splitlines()]
    sections = [json.loads(line) for line in (output / "sections.jsonl").read_text().splitlines()]

    assert report == {
        "schema_version": "allen-real-histology-metadata-v1",
        "animals": 1,
        "experiments": 2,
        "sections": 2,
        "development_splits": [split_for_animal(7)],
        "images_downloaded": 0,
    }
    assert {row["animal_id"] for row in experiments} == {7}
    assert {row["animal_partition_key"] for row in experiments} == {"allen-donor:7"}
    assert {row["specimen_id"] for row in experiments} == {101, 102}
    assert {row["split"] for row in experiments} == {split_for_animal(7)}
    assert {row["section_id"] for row in sections} == {1101, 1201}
    assert all(row["image_status"] == "not_downloaded" and row["image_sha256"] is None for row in sections)
    assert all(row["training_role"] == "real_histology_appearance_only" for row in experiments + sections)
    assert manifest["partition_policy"]["unit"] == "Allen Donor.id"
    assert manifest["partition_policy"]["final_test"] == "not defined or accessed by this development snapshot"
    assert manifest["data_role"]["pretrained_models_features_pseudolabels"] == "none"
    assert all(not path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"} for path in output.rglob("*"))

    acquire_metadata_snapshot(
        output,
        2,
        page_size=2,
        get=_fake_get(),
        retrieved_at_utc="2026-09-02T00:00:00+00:00",
    )


def test_verifier_rejects_record_tampering(tmp_path: Path):
    output = tmp_path / "allen"
    acquire_metadata_snapshot(
        output,
        2,
        page_size=2,
        get=_fake_get(),
        retrieved_at_utc="2026-09-02T00:00:00+00:00",
    )
    (output / "sections.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_metadata_snapshot(output)


def test_donor_partition_is_stable_and_not_specimen_based():
    assert split_for_animal(7) == split_for_animal(7)
    assert len({split_for_animal(animal_id) for animal_id in range(128)}) == 2


def test_nonofficial_or_nonhttps_sources_are_rejected():
    with pytest.raises(ValueError, match="official HTTPS host"):
        _official_url("https://example.org/metadata.json")
    with pytest.raises(ValueError, match="official HTTPS host"):
        _official_url("http://api.brain-map.org/api/v2/data/query.json")


def test_snapshot_root_rejects_c_and_relative_paths_resolving_on_c(monkeypatch):
    with pytest.raises(ValueError, match="must resolve to I:"):
        _snapshot_root("C:/AnatomyTracker/blocked")
    monkeypatch.chdir(Path.home())
    assert Path.cwd().drive.upper() == "C:"
    with pytest.raises(ValueError, match="must resolve to I:"):
        _snapshot_root("relative-allen-snapshot")


def test_snapshot_root_accepts_and_canonicalizes_i():
    root = _snapshot_root("I:/AnatomyTracker/data/../data/allen-metadata")
    assert root == Path("I:/AnatomyTracker/data/allen-metadata")
    assert root.drive.upper() == "I:"
