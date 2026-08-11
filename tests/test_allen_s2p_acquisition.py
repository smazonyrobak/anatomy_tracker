import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from training.acquire_allen_s2p import (
    DEEPSLICE_S2P_EXPERIMENT_IDS,
    audit_coordinates,
    download_sections,
    image_to_reference,
    query_training_datasets,
    quicknii_to_tracker_pose,
    section_manifest_records,
    section_quicknii_ouv,
    select_pilot_datasets,
    split_for_specimen,
)


class Response:
    def __init__(self, payload=None, content=b""):
        self.payload = payload
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def iter_content(self, _):
        yield self.content


def zero_tilt_dataset():
    return {
        "id": 1,
        "section_thickness": 25.0,
        "alignment3d": {
            "tvr_00": 0.0,
            "tvr_01": 0.0,
            "tvr_02": 1.0,
            "tvr_03": 0.0,
            "tvr_04": 1.0,
            "tvr_05": 0.0,
            "tvr_06": 1.0,
            "tvr_07": 0.0,
            "tvr_08": 0.0,
            "tvr_09": 0.0,
            "tvr_10": 0.0,
            "tvr_11": 0.0,
        },
        "section_images": [],
    }


def zero_tilt_section():
    return {
        "id": 2,
        "section_number": 216,
        "width": 11400,
        "height": 8000,
        "alignment2d": {
            "tsv_00": 1.0,
            "tsv_01": 0.0,
            "tsv_02": 0.0,
            "tsv_03": 1.0,
            "tsv_04": 0.0,
            "tsv_05": 0.0,
        },
    }


def test_queries_only_requested_products_and_alignment_filters():
    queries = []

    def get(url, params=None, timeout=None):
        del url, timeout
        query = params["q"]
        queries.append(query)
        product = 5 if "products[id$eq5]" in query else 8
        return Response(
            {
                "success": True,
                "total_rows": 1,
                "msg": [{"id": product * 10, "specimen_id": product * 100}],
            }
        )

    records = query_training_datasets(get, page_size=10)
    assert [(row["id"], row["product_ids"]) for row in records] == [(50, [5]), (80, [8])]
    assert len(queries) == 2
    assert all("[failed$eqfalse]" in query for query in queries)
    assert all("[reference_space_id$eq9]" in query for query in queries)
    assert all("[plane_of_section_id$eq1]" in query for query in queries)
    assert not any("products[id$eq" in query and "products[id$eq5]" not in query and "products[id$eq8]" not in query for query in queries)


def test_specimen_splits_do_not_leak_and_published_experiments_are_fixed():
    assert DEEPSLICE_S2P_EXPERIMENT_IDS == (
        287533790,
        287808449,
        306881784,
        509463587,
        510582887,
        516491813,
        589399902,
        601885751,
        603717226,
        638978767,
    )
    sealed = {41}
    assert split_for_specimen(41, sealed) == "sealed_deepslice_s2p"
    assert split_for_specimen(41, sealed) == split_for_specimen(41, sealed)
    assert split_for_specimen(42, sealed) == split_for_specimen(42, sealed)
    assert split_for_specimen(42, sealed) != "sealed_deepslice_s2p"


def test_pilot_cap_samples_every_unsealed_split_by_specimen():
    specimens = {split: [] for split in ("train", "validation", "test")}
    specimen_id = 1
    while min(map(len, specimens.values())) < 8:
        specimens[split_for_specimen(specimen_id, set())].append(specimen_id)
        specimen_id += 1
    index = {
        value: {"id": value, "specimen_id": value}
        for split in specimens.values()
        for value in split
    }
    selected = select_pilot_datasets(index, set(), 20)
    split_counts = {
        split: sum(split_for_specimen(index[dataset_id]["specimen_id"], set()) == split for dataset_id in selected)
        for split in specimens
    }
    assert len(selected) == 20
    assert split_counts == {"train": 18, "validation": 1, "test": 1}


def test_official_image_to_reference_allen2quicknii_and_tracker_pose_math():
    dataset = zero_tilt_dataset()
    section = zero_tilt_section()
    assert np.allclose(image_to_reference(dataset, section, 100.0, 200.0), [5400.0, 200.0, 100.0])
    ouv = section_quicknii_ouv(dataset, section)
    assert np.allclose(ouv, [0.0, 312.0, 320.0, 456.0, 0.0, 0.0, 0.0, 0.0, -320.0])
    assert np.allclose(quicknii_to_tracker_pose(ouv), [0.0, 0.0, 0.0])

    lr_deg, dv_deg, ap_um = 12.0, -7.0, -1375.0
    ml_slope, dv_slope = np.tan(np.deg2rad([lr_deg, dv_deg]))
    origin_ml, origin_dv = 100.0, 50.0
    center_index = 216.0 - ap_um / 25.0
    origin_ap = center_index - ml_slope * (227.5 - origin_ml) - dv_slope * (159.5 - origin_dv)
    tilted_ouv = np.asarray(
        [
            origin_ml,
            528.0 - origin_ap,
            320.0 - origin_dv,
            100.0,
            -ml_slope * 100.0,
            0.0,
            0.0,
            -dv_slope * 100.0,
            -100.0,
        ]
    )
    assert np.allclose(quicknii_to_tracker_pose(tilted_ouv), [ap_um, lr_deg, dv_deg])


def test_training_is_ap_filtered_but_the_published_benchmark_stays_complete():
    dataset = zero_tilt_dataset()
    section = zero_tilt_section()
    section["section_number"] = 0
    section["failed"] = False
    dataset.update(specimen_id=3, equalization=None, section_images=[section])
    assert section_manifest_records(dataset, "train") == []
    sealed = section_manifest_records(dataset, "sealed_deepslice_s2p")
    assert len(sealed) == 1
    assert sealed[0]["in_training_ap_domain"] is False


def test_coordinate_audit_compares_local_transform_to_api():
    dataset = zero_tilt_dataset()
    section = zero_tilt_section()
    dataset["section_images"] = [section]
    record = {
        "section_image_id": 2,
        "experiment_id": 1,
        "width": section["width"],
        "height": section["height"],
    }

    def get(url, params=None, timeout=None):
        del url, timeout
        point = image_to_reference(dataset, section, params["x"], params["y"])
        return Response(
            {"success": True, "msg": {"image_to_reference": dict(zip("xyz", point.tolist()))}}
        )

    result = audit_coordinates([record], {1: dataset}, get, count=1)
    assert result["count"] == 1
    assert result["max_absolute_error_um"] == 0.0


def test_jpeg_download_is_hashed_atomic_and_resumable(tmp_path: Path):
    stream = io.BytesIO()
    Image.new("RGB", (12, 10), (10, 20, 30)).save(stream, format="JPEG")
    jpeg = stream.getvalue()
    calls = []

    def get(url, stream=None, timeout=None):
        calls.append((url, stream, timeout))
        return Response(content=jpeg)

    record = {
        "section_image_id": 7,
        "relative_path": "images/train/3/7.jpg",
        "download_url": "https://example.test/7.jpg",
    }
    first = download_sections([record], tmp_path, get, workers=1)
    second = download_sections([record], tmp_path, get, workers=1)
    digest = hashlib.sha256(jpeg).hexdigest()
    assert first == second == [{"section_image_id": 7, "sha256": digest}]
    assert len(calls) == 1
    assert (tmp_path / record["relative_path"]).read_bytes() == jpeg
    assert not list(tmp_path.rglob("*.part"))


def test_jpeg_download_retries_a_transient_gateway_failure(tmp_path: Path, monkeypatch):
    stream = io.BytesIO()
    Image.new("RGB", (12, 10), (10, 20, 30)).save(stream, format="JPEG")
    responses = [requests.Response(), Response(content=stream.getvalue())]
    responses[0].status_code = 502
    monkeypatch.setattr("training.acquire_allen_s2p.time.sleep", lambda _: None)

    def get(*_, **__):
        return responses.pop(0)

    record = {"section_image_id": 7, "relative_path": "7.jpg", "download_url": "unused"}
    download_sections([record], tmp_path, get, workers=1)
    assert not responses


def test_jpeg_download_retries_an_invalid_success_payload(tmp_path: Path, monkeypatch):
    stream = io.BytesIO()
    Image.new("RGB", (12, 10), (10, 20, 30)).save(stream, format="JPEG")
    responses = [Response(content=b"temporary upstream error"), Response(content=stream.getvalue())]
    monkeypatch.setattr("training.acquire_allen_s2p.time.sleep", lambda _: None)

    def get(*_, **__):
        return responses.pop(0)

    record = {"section_image_id": 7, "relative_path": "7.jpg", "download_url": "unused"}
    download_sections([record], tmp_path, get, workers=1)
    assert not responses
    assert not list(tmp_path.rglob("*.part"))


def test_download_manifest_rejects_a_different_section_set(tmp_path: Path):
    (tmp_path / "downloads.jsonl").write_text('{"section_image_id":1,"sha256":"x"}\n')
    with pytest.raises(FileExistsError, match="does not match"):
        download_sections(
            [{"section_image_id": 2, "relative_path": "2.jpg", "download_url": "unused"}],
            tmp_path,
            workers=1,
        )
