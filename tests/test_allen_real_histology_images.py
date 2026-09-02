import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlparse

import pytest
from PIL import Image

from training.acquire_allen_real_histology_images import (
    acquire_image_snapshot,
    deterministic_section_selection,
)
from training.allen_real_histology_metadata import (
    OFFICIAL_DOCUMENTS,
    _canonical_bytes,
    acquire_metadata_snapshot,
)
from training.verify_allen_real_histology_images import verify_image_snapshot


def _response(url: str, payload=None, content: bytes | None = None, content_type="application/json"):
    content = content if content is not None else _canonical_bytes(payload)
    response = Mock(content=content, url=url, headers={"Content-Type": content_type})
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _dataset(experiment_id: int, specimen_id: int, animal_id: int) -> dict:
    sections = []
    for offset in range(2):
        section_id = experiment_id * 100 + offset
        sections.append(
            {
                "id": section_id,
                "section_number": offset + 1,
                "failed": False,
                "width": 3200,
                "height": 2400,
                "resolution": 0.35,
                "alignment2d": {f"tsv_{index:02d}": float(index) for index in range(6)},
            }
        )
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
        "section_images": sections,
    }


def _metadata_get():
    datasets = {
        11: _dataset(11, 101, 1),
        12: _dataset(12, 102, 2),
        13: _dataset(13, 103, 3),
    }

    def get(url, params=None, timeout=60):
        del timeout
        if url.endswith("/data/query.json"):
            payload = {
                "success": True,
                "total_rows": 3,
                "msg": [{"id": 11}, {"id": 12}, {"id": 13}],
            }
            return _response(f"{url}?q=official-index", payload)
        if "/SectionDataSet/" in url:
            experiment_id = int(url.rsplit("/", 1)[1].split(".")[0])
            payload = {"success": True, "total_rows": 1, "msg": [datasets[experiment_id]]}
            return _response(f"{url}?include=official-metadata", payload)
        if url in OFFICIAL_DOCUMENTS.values():
            return _response(url, content=f"official:{url}".encode("utf-8"), content_type="text/html")
        raise AssertionError((url, params))

    return get


def _jpeg(section_id: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (12, 10), (section_id % 251, 20, 30)).save(stream, format="JPEG")
    return stream.getvalue()


def _image_get(calls: list[int]):
    def get(url, timeout=180):
        del timeout
        section_id = int(urlparse(url).path.rsplit("/", 1)[1])
        calls.append(section_id)
        return _response(url, content=_jpeg(section_id), content_type="image/jpeg")

    return get


def _metadata_snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "metadata"
    acquire_metadata_snapshot(
        root,
        3,
        page_size=3,
        get=_metadata_get(),
        retrieved_at_utc="2026-09-02T00:00:00+00:00",
    )
    return root


def test_image_snapshot_is_donor_round_robin_hash_bound_and_independently_verified(tmp_path: Path):
    metadata = _metadata_snapshot(tmp_path)
    sections = [json.loads(line) for line in (metadata / "sections.jsonl").read_text().splitlines()]
    selected = deterministic_section_selection(
        sections,
        {"development_train": 2, "development_validation": 1},
    )
    assert {row["animal_id"] for row in selected if row["split"] == "development_train"} == {1, 2}
    assert {row["animal_id"] for row in selected if row["split"] == "development_validation"} == {3}

    calls = []
    output = tmp_path / "images"
    manifest = acquire_image_snapshot(
        metadata,
        output,
        {"development_train": 2, "development_validation": 1},
        get=_image_get(calls),
        retrieved_at_utc="2026-09-02T01:00:00+00:00",
    )
    report = verify_image_snapshot(output, metadata)
    rows = [json.loads(line) for line in (output / "images.jsonl").read_text().splitlines()]

    assert calls == [row["section_id"] for row in selected]
    assert report["images"] == 3
    assert report["animals"] == 3
    assert report["splits"] == ["development_train", "development_validation"]
    assert manifest["counts"]["pre_exclusion"] == {
        "section_records_total": 6,
        "eligible_for_appearance_training": 6,
        "excluded_by_metadata": 0,
    }
    assert manifest["counts"]["post_selection"] == {
        "selected_for_download": 3,
        "eligible_not_selected": 3,
    }
    assert manifest["counts"]["post_download"] == {
        "downloaded_and_hash_bound": 3,
        "transport_or_decode_exclusions": 0,
    }
    assert manifest["terms"]["license_spdx"] is None
    assert "outside Git" in manifest["terms"]["redistribution_caveat"]
    assert all(row["learned_source_dependency"] == "none" for row in rows)
    for row in rows:
        content = (output / row["relative_path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == row["sha256"]
        assert row["source_experiment_response_sha256"]


def test_independent_verifier_rejects_raw_image_tampering(tmp_path: Path):
    metadata = _metadata_snapshot(tmp_path)
    output = tmp_path / "images"
    acquire_image_snapshot(
        metadata,
        output,
        {"development_train": 1, "development_validation": 0},
        get=_image_get([]),
        retrieved_at_utc="2026-09-02T01:00:00+00:00",
    )
    row = json.loads((output / "images.jsonl").read_text().splitlines()[0])
    (output / row["relative_path"]).write_bytes(b"not the Allen image")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_image_snapshot(output, metadata)


def test_independent_verifier_recomputes_deterministic_selection(tmp_path: Path):
    metadata = _metadata_snapshot(tmp_path)
    output = tmp_path / "images"
    acquire_image_snapshot(
        metadata,
        output,
        {"development_train": 2, "development_validation": 0},
        get=_image_get([]),
        retrieved_at_utc="2026-09-02T01:00:00+00:00",
    )
    rows = [json.loads(line) for line in (output / "images.jsonl").read_text().splitlines()]
    rows.reverse()
    (output / "images.jsonl").write_bytes(b"".join(_canonical_bytes(row) for row in rows))
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for row in receipt["files"]:
        if row["relative_path"] == "images.jsonl":
            content = (output / "images.jsonl").read_bytes()
            row["bytes"] = len(content)
            row["sha256"] = hashlib.sha256(content).hexdigest()
    receipt_path.write_bytes(_canonical_bytes(receipt))
    with pytest.raises(ValueError, match="deterministic selection"):
        verify_image_snapshot(output, metadata)


def test_independent_verifier_rehashes_raw_metadata_responses(tmp_path: Path):
    metadata = _metadata_snapshot(tmp_path)
    output = tmp_path / "images"
    acquire_image_snapshot(
        metadata,
        output,
        {"development_train": 1, "development_validation": 0},
        get=_image_get([]),
        retrieved_at_utc="2026-09-02T01:00:00+00:00",
    )
    raw_experiment = next((metadata / "raw" / "api" / "experiments").glob("*.json"))
    raw_experiment.write_bytes(b"tampered official response")
    with pytest.raises(ValueError, match="raw source metadata hash mismatch"):
        verify_image_snapshot(output, metadata)


def test_image_output_inside_git_is_rejected_before_download(tmp_path: Path):
    metadata = _metadata_snapshot(tmp_path)
    git_root = tmp_path / "git-worktree"
    (git_root / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="outside a Git worktree"):
        acquire_image_snapshot(
            metadata,
            git_root / "data" / "forbidden-raw-images",
            {"development_train": 1},
            get=_image_get([]),
        )


def test_completed_snapshot_rejects_changed_quota_before_download_or_write(tmp_path: Path):
    metadata = _metadata_snapshot(tmp_path)
    output = tmp_path / "images"
    acquire_image_snapshot(
        metadata,
        output,
        {"development_train": 1, "development_validation": 0},
        get=_image_get([]),
        retrieved_at_utc="2026-09-02T01:00:00+00:00",
    )
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    calls = []
    with pytest.raises(FileExistsError, match="different quotas"):
        acquire_image_snapshot(
            metadata,
            output,
            {"development_train": 2, "development_validation": 0},
            get=_image_get(calls),
            retrieved_at_utc="2026-09-02T01:00:00+00:00",
        )
    assert calls == []
    assert before == {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }


def test_nonempty_unsealed_output_is_rejected_before_download(tmp_path: Path):
    metadata = _metadata_snapshot(tmp_path)
    output = tmp_path / "images"
    output.mkdir()
    unrelated = output / "unrelated.bin"
    unrelated.write_bytes(b"not part of this snapshot")
    calls = []
    with pytest.raises(FileExistsError, match="existing incomplete snapshot"):
        acquire_image_snapshot(
            metadata,
            output,
            {"development_train": 1},
            get=_image_get(calls),
        )
    assert calls == []
    assert unrelated.read_bytes() == b"not part of this snapshot"


@pytest.mark.parametrize("field", ["animal_id_namespace", "section_id_namespace"])
def test_independent_verifier_rejects_namespace_tampering(tmp_path: Path, field: str):
    metadata = _metadata_snapshot(tmp_path)
    output = tmp_path / "images"
    acquire_image_snapshot(
        metadata,
        output,
        {"development_train": 1, "development_validation": 0},
        get=_image_get([]),
        retrieved_at_utc="2026-09-02T01:00:00+00:00",
    )
    image_manifest_path = output / "images.jsonl"
    rows = [json.loads(line) for line in image_manifest_path.read_text().splitlines()]
    rows[0][field] = "tampered-namespace"
    image_manifest_path.write_bytes(b"".join(_canonical_bytes(row) for row in rows))
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for row in receipt["files"]:
        if row["relative_path"] == "images.jsonl":
            content = image_manifest_path.read_bytes()
            row["bytes"] = len(content)
            row["sha256"] = hashlib.sha256(content).hexdigest()
    receipt_path.write_bytes(_canonical_bytes(receipt))
    with pytest.raises(ValueError, match="lineage differs"):
        verify_image_snapshot(output, metadata)


def test_concurrent_publishers_cannot_overwrite_the_winning_snapshot(tmp_path: Path):
    metadata = _metadata_snapshot(tmp_path)
    output = tmp_path / "images"
    barrier = threading.Barrier(2)

    def get(url, timeout=180):
        del timeout
        barrier.wait(timeout=5)
        section_id = int(urlparse(url).path.rsplit("/", 1)[1])
        return _response(url, content=_jpeg(section_id), content_type="image/jpeg")

    def acquire():
        return acquire_image_snapshot(
            metadata,
            output,
            {"development_train": 1, "development_validation": 0},
            get=get,
            retrieved_at_utc="2026-09-02T01:00:00+00:00",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        manifests = list(pool.map(lambda _: acquire(), range(2)))
    assert manifests[0] == manifests[1]
    assert verify_image_snapshot(output, metadata)["images"] == 1
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_independent_verifier_rejects_false_selection_unit(tmp_path: Path):
    metadata = _metadata_snapshot(tmp_path)
    output = tmp_path / "images"
    acquire_image_snapshot(
        metadata,
        output,
        {"development_train": 1, "development_validation": 0},
        get=_image_get([]),
        retrieved_at_utc="2026-09-02T01:00:00+00:00",
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection"]["unit"] = "section"
    manifest_path.write_bytes(_canonical_bytes(manifest))
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for row in receipt["files"]:
        if row["relative_path"] == "manifest.json":
            content = manifest_path.read_bytes()
            row["bytes"] = len(content)
            row["sha256"] = hashlib.sha256(content).hexdigest()
    receipt_path.write_bytes(_canonical_bytes(receipt))
    with pytest.raises(ValueError, match="selection contract"):
        verify_image_snapshot(output, metadata)
