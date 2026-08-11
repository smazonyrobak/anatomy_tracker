from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image


API_ROOT = "https://api.brain-map.org"
PRODUCT_IDS = (5, 8)
DEEPSLICE_S2P_EXPERIMENT_IDS = (
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
SPLIT_SALT = "atlaspose-allen-s2p-specimen-v1"
AP_MIN_UM = -4500.0
AP_MAX_UM = 500.0
VOXEL_UM = 25.0
BREGMA_AP_INDEX = 216.0
QUICKNII_SHAPE = np.asarray((456.0, 528.0, 320.0))
ATLAS_CENTER_ML_DV = np.asarray((227.5, 159.5))
ALIGNMENT2D_KEYS = tuple(f"tsv_{index:02d}" for index in range(6))
ALIGNMENT3D_KEYS = tuple(f"tvr_{index:02d}" for index in range(12))


def _json_response(response):
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Allen API query failed: {payload}")
    return payload


def _get_json(get, url: str, params: dict | None = None):
    return _json_response(get(url, params=params, timeout=60))


def query_training_datasets(get=requests.get, page_size: int = 200) -> list[dict]:
    datasets: dict[int, dict] = {}
    endpoint = f"{API_ROOT}/api/v2/data/query.json"
    for product_id in PRODUCT_IDS:
        start = 0
        while True:
            query = (
                "model::SectionDataSet,rma::criteria,"
                "[failed$eqfalse],[reference_space_id$eq9],[plane_of_section_id$eq1],"
                f"products[id$eq{product_id}],"
                f"rma::options[num_rows$eq{page_size}][start_row$eq{start}][order$eq'id']"
            )
            payload = _get_json(get, endpoint, {"q": query})
            for record in payload["msg"]:
                dataset_id = int(record["id"])
                current = datasets.setdefault(dataset_id, {**record, "product_ids": []})
                current["product_ids"] = sorted(set(current["product_ids"]) | {product_id})
            start += len(payload["msg"])
            if start >= int(payload["total_rows"]) or not payload["msg"]:
                break
    return [datasets[dataset_id] for dataset_id in sorted(datasets)]


def fetch_dataset(dataset_id: int, get=requests.get) -> dict:
    url = f"{API_ROOT}/api/v2/data/SectionDataSet/{dataset_id}.json"
    payload = _get_json(
        get,
        url,
        {"include": "products,alignment3d,equalization,section_images(alignment2d)"},
    )
    if len(payload["msg"]) != 1:
        raise RuntimeError(f"Allen returned {len(payload['msg'])} records for experiment {dataset_id}")
    dataset = payload["msg"][0]
    if (
        bool(dataset["failed"])
        or int(dataset["reference_space_id"]) != 9
        or int(dataset["plane_of_section_id"]) != 1
    ):
        raise RuntimeError(f"Experiment {dataset_id} violates the S2P acquisition filters")
    dataset["product_ids"] = sorted(int(product["id"]) for product in dataset.get("products", []))
    dataset["source_url"] = url
    return dataset


def split_for_specimen(specimen_id: int, sealed_specimens: set[int]) -> str:
    specimen_id = int(specimen_id)
    if specimen_id in sealed_specimens:
        return "sealed_deepslice_s2p"
    digest = hashlib.sha256(f"{SPLIT_SALT}:{specimen_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < 0.90:
        return "train"
    return "validation" if fraction < 0.95 else "test"


def select_pilot_datasets(index: dict[int, dict], sealed_specimens: set[int], dataset_cap: int | None) -> list[int]:
    candidates = {
        dataset_id: record
        for dataset_id, record in index.items()
        if int(record["specimen_id"]) not in sealed_specimens
    }
    if dataset_cap is None or dataset_cap >= len(candidates):
        return sorted(candidates)
    if dataset_cap < 1:
        return []

    grouped = {split: {} for split in ("train", "validation", "test")}
    for dataset_id, record in candidates.items():
        specimen_id = int(record["specimen_id"])
        split = split_for_specimen(specimen_id, sealed_specimens)
        grouped[split].setdefault(specimen_id, []).append(dataset_id)

    def interleaved(split: str) -> list[int]:
        specimens = sorted(
            grouped[split],
            key=lambda specimen_id: hashlib.sha256(f"pilot:{specimen_id}".encode()).digest(),
        )
        experiments = [sorted(grouped[split][specimen_id]) for specimen_id in specimens]
        return [
            experiment[position]
            for position in range(max(map(len, experiments), default=0))
            for experiment in experiments
            if position < len(experiment)
        ]

    validation_count = max(1, round(dataset_cap * 0.05)) if dataset_cap >= 3 else 0
    test_count = max(1, round(dataset_cap * 0.05)) if dataset_cap >= 3 else 0
    quotas = {
        "train": dataset_cap - validation_count - test_count,
        "validation": validation_count,
        "test": test_count,
    }
    ordered = {split: interleaved(split) for split in grouped}
    selected = [dataset_id for split in quotas for dataset_id in ordered[split][: quotas[split]]]
    if len(selected) < dataset_cap:
        selected_set = set(selected)
        remainder = sorted(
            (dataset_id for dataset_id in candidates if dataset_id not in selected_set),
            key=lambda dataset_id: hashlib.sha256(f"pilot-dataset:{dataset_id}".encode()).digest(),
        )
        selected.extend(remainder[: dataset_cap - len(selected)])
    return sorted(selected)


def image_to_reference(dataset: dict, section: dict, x: float, y: float) -> np.ndarray:
    alignment2d = section["alignment2d"]
    alignment3d = dataset["alignment3d"]
    volume = np.asarray(
        (
            alignment2d["tsv_00"] * x + alignment2d["tsv_01"] * y + alignment2d["tsv_04"],
            alignment2d["tsv_02"] * x + alignment2d["tsv_03"] * y + alignment2d["tsv_05"],
            section["section_number"] * dataset["section_thickness"],
        ),
        dtype=np.float64,
    )
    matrix = np.asarray(
        [[alignment3d[f"tvr_{row * 3 + column:02d}"] for column in range(3)] for row in range(3)],
        dtype=np.float64,
    )
    translation = np.asarray([alignment3d[f"tvr_{index:02d}"] for index in range(9, 12)])
    return matrix @ volume + translation


def section_quicknii_ouv(dataset: dict, section: dict) -> np.ndarray:
    origin, upper_right, lower_left = (
        image_to_reference(dataset, section, x, y) / VOXEL_UM
        for x, y in ((0.0, 0.0), (float(section["width"]), 0.0), (0.0, float(section["height"])))
    )
    quicknii_origin = np.asarray(
        (origin[2], QUICKNII_SHAPE[1] - origin[0], QUICKNII_SHAPE[2] - origin[1])
    )
    quicknii_u = np.asarray(
        (upper_right[2] - origin[2], origin[0] - upper_right[0], origin[1] - upper_right[1])
    )
    quicknii_v = np.asarray(
        (lower_left[2] - origin[2], origin[0] - lower_left[0], origin[1] - lower_left[1])
    )
    return np.concatenate((quicknii_origin, quicknii_u, quicknii_v))


def quicknii_to_tracker_pose(ouv: np.ndarray) -> np.ndarray:
    ouv = np.asarray(ouv, dtype=np.float64)
    origin = ouv[:3]
    normal = np.cross(ouv[3:6], ouv[6:9])
    if normal[1] < 0.0:
        normal = -normal
    if abs(normal[1]) < 1e-9:
        raise ValueError("Alignment is not a coronal plane")
    ap_per_ml = -normal[0] / normal[1]
    ap_per_dv = -normal[2] / normal[1]
    origin_ml = QUICKNII_SHAPE[0] - origin[0]
    origin_ap = QUICKNII_SHAPE[1] - origin[1]
    origin_dv = QUICKNII_SHAPE[2] - origin[2]
    ap_index = (
        origin_ap
        + ap_per_ml * (ATLAS_CENTER_ML_DV[0] - origin_ml)
        + ap_per_dv * (ATLAS_CENTER_ML_DV[1] - origin_dv)
    )
    return np.asarray(
        (
            (BREGMA_AP_INDEX - ap_index) * VOXEL_UM,
            np.degrees(np.arctan(ap_per_ml)),
            np.degrees(np.arctan(ap_per_dv)),
        )
    )


def equalization_range(dataset: dict) -> list[int] | None:
    equalization = dataset.get("equalization")
    if not equalization:
        return None
    return [
        int(equalization[f"{channel}_{bound}"])
        for channel in ("red", "green", "blue")
        for bound in ("lower", "upper")
    ]


def image_download_url(section_id: int, ranges: list[int] | None, downsample: int = 5) -> str:
    if ranges is None:
        return f"{API_ROOT}/api/v2/image_download/{section_id}?downsample={downsample}"
    values = ",".join(map(str, ranges))
    return f"{API_ROOT}/api/v2/image_download/{section_id}?range={values}&downsample={downsample}"


def dataset_manifest_record(dataset: dict, split: str) -> dict:
    return {
        "experiment_id": int(dataset["id"]),
        "specimen_id": int(dataset["specimen_id"]),
        "product_ids": sorted(map(int, dataset["product_ids"])),
        "split": split,
        "failed": False,
        "reference_space_id": 9,
        "plane_of_section_id": 1,
        "section_thickness_um": float(dataset["section_thickness"]),
        "alignment3d_tvr": [float(dataset["alignment3d"][key]) for key in ALIGNMENT3D_KEYS],
        "equalization_range_rgb": equalization_range(dataset),
        "source_url": dataset["source_url"],
    }


def section_manifest_records(dataset: dict, split: str) -> list[dict]:
    ranges = equalization_range(dataset)
    records = []
    for section in sorted(dataset["section_images"], key=lambda item: (item["section_number"], item["id"])):
        if section.get("failed") or not section.get("alignment2d"):
            continue
        ouv = section_quicknii_ouv(dataset, section)
        pose = quicknii_to_tracker_pose(ouv)
        in_training_domain = bool(AP_MIN_UM <= pose[0] <= AP_MAX_UM)
        if split != "sealed_deepslice_s2p" and not in_training_domain:
            continue
        section_id = int(section["id"])
        relative_path = f"images/{split}/{int(dataset['id'])}/{section_id}.jpg"
        records.append(
            {
                "section_image_id": section_id,
                "experiment_id": int(dataset["id"]),
                "specimen_id": int(dataset["specimen_id"]),
                "split": split,
                "section_number": int(section["section_number"]),
                "width": int(section["width"]),
                "height": int(section["height"]),
                "alignment2d_tsv": [float(section["alignment2d"][key]) for key in ALIGNMENT2D_KEYS],
                "quicknii_ouv": ouv.tolist(),
                "ap_um": float(pose[0]),
                "tilt_lr_deg": float(pose[1]),
                "tilt_dv_deg": float(pose[2]),
                "in_training_ap_domain": in_training_domain,
                "download_url": image_download_url(section_id, ranges),
                "relative_path": relative_path,
            }
        )
    return records


def _jsonl_bytes(records: list[dict]) -> bytes:
    return "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records).encode()


def _immutable_write(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Immutable dataset artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _valid_jpeg(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, SyntaxError):
        return False


def _download_response(get, url: str, attempts: int = 8):
    retryable = {429, 500, 502, 503, 504}
    for attempt in range(attempts):
        try:
            response = get(url, stream=True, timeout=180)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            status = getattr(error.response, "status_code", None)
            if attempt + 1 == attempts or (status is not None and status not in retryable):
                raise
            time.sleep(min(30.0, 2.0**attempt))


def _download_section(record: dict, output: Path, expected_sha256: str | None, get) -> dict:
    destination = output / record["relative_path"]
    if _valid_jpeg(destination):
        digest = _sha256_file(destination)
        if expected_sha256 is None or digest == expected_sha256:
            return {"section_image_id": record["section_image_id"], "sha256": digest}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    for payload_attempt in range(8):
        response = _download_response(get, record["download_url"])
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                if block:
                    stream.write(block)
        if _valid_jpeg(temporary):
            break
        temporary.unlink(missing_ok=True)
        if payload_attempt == 7:
            raise RuntimeError(f"Allen repeatedly returned an invalid JPEG for section {record['section_image_id']}")
        time.sleep(min(30.0, 2.0**payload_attempt))
    digest = _sha256_file(temporary)
    if expected_sha256 is not None and digest != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Allen image changed for section {record['section_image_id']}")
    os.replace(temporary, destination)
    return {"section_image_id": record["section_image_id"], "sha256": digest}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def download_sections(records: list[dict], output: Path, get=requests.get, workers: int = 8) -> list[dict]:
    manifest_path = output / "downloads.jsonl"
    expected = {}
    if manifest_path.exists():
        expected = {int(row["section_image_id"]): row["sha256"] for row in _read_jsonl(manifest_path)}
        if set(expected) != {int(row["section_image_id"]) for row in records}:
            raise FileExistsError("Immutable download manifest does not match sections.jsonl")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        downloads = list(
            pool.map(
                lambda record: _download_section(
                    record,
                    output,
                    expected.get(int(record["section_image_id"])),
                    get,
                ),
                records,
            )
        )
    downloads.sort(key=lambda row: row["section_image_id"])
    _immutable_write(manifest_path, _jsonl_bytes(downloads))
    return downloads


def audit_coordinates(
    section_records: list[dict],
    datasets: dict[int, dict],
    get=requests.get,
    count: int = 20,
    seed: int = 94731,
) -> dict:
    if not section_records or count <= 0:
        return {"count": 0, "max_absolute_error_um": 0.0, "samples": []}
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(section_records), min(count, len(section_records)), replace=False)
    sections_by_id = {
        int(section["id"]): section
        for dataset in datasets.values()
        for section in dataset["section_images"]
    }
    samples = []
    for index in indices:
        record = section_records[int(index)]
        section = sections_by_id[record["section_image_id"]]
        dataset = datasets[record["experiment_id"]]
        x = float(rng.uniform(0.0, section["width"]))
        y = float(rng.uniform(0.0, section["height"]))
        local = image_to_reference(dataset, section, x, y)
        url = f"{API_ROOT}/api/v2/image_to_reference/{record['section_image_id']}.json"
        payload = _get_json(get, url, {"x": x, "y": y})
        remote_record = payload["msg"]["image_to_reference"]
        remote = np.asarray([remote_record[axis] for axis in "xyz"], dtype=np.float64)
        absolute_error = np.abs(local - remote)
        samples.append(
            {
                "section_image_id": record["section_image_id"],
                "x": x,
                "y": y,
                "local_reference_pir_um": local.tolist(),
                "api_reference_pir_um": remote.tolist(),
                "max_absolute_error_um": float(absolute_error.max()),
                "source_url": url,
            }
        )
    maximum = max(sample["max_absolute_error_um"] for sample in samples)
    if maximum > 1e-3:
        raise RuntimeError(f"Local Allen coordinate transform audit failed ({maximum:.6g} um)")
    return {"count": len(samples), "max_absolute_error_um": maximum, "samples": samples}


def acquire(
    output: Path,
    metadata_only: bool = False,
    dataset_cap: int | None = None,
    image_cap: int | None = None,
    workers: int = 8,
    audit_count: int = 20,
    get=requests.get,
) -> dict:
    output = Path(output)
    training_index = query_training_datasets(get)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        benchmark_details = list(pool.map(lambda dataset_id: fetch_dataset(dataset_id, get), DEEPSLICE_S2P_EXPERIMENT_IDS))
    benchmark_by_id = {int(dataset["id"]): dataset for dataset in benchmark_details}
    sealed_specimens = {int(dataset["specimen_id"]) for dataset in benchmark_details}

    index = {int(record["id"]): record for record in training_index}
    for dataset in benchmark_details:
        index[int(dataset["id"])] = dataset
    selected_ids = [
        dataset_id
        for dataset_id, record in sorted(index.items())
        if int(record["specimen_id"]) in sealed_specimens
    ]
    trainable_ids = select_pilot_datasets(index, sealed_specimens, dataset_cap)
    selected_ids.extend(trainable_ids)

    details = dict(benchmark_by_id)
    missing_ids = [dataset_id for dataset_id in selected_ids if dataset_id not in details]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = pool.map(lambda dataset_id: fetch_dataset(dataset_id, get), missing_ids)
        details.update((int(dataset["id"]), dataset) for dataset in fetched)

    dataset_records = []
    section_records = []
    for dataset_id in sorted(details):
        dataset = details[dataset_id]
        split = split_for_specimen(int(dataset["specimen_id"]), sealed_specimens)
        if not dataset.get("alignment3d"):
            continue
        dataset_records.append(dataset_manifest_record(dataset, split))
        section_records.extend(section_manifest_records(dataset, split))
    section_records.sort(key=lambda row: (row["experiment_id"], row["section_number"], row["section_image_id"]))
    if image_cap is not None:
        section_records = section_records[:image_cap]

    dataset_content = _jsonl_bytes(dataset_records)
    section_content = _jsonl_bytes(section_records)
    _immutable_write(output / "datasets.jsonl", dataset_content)
    _immutable_write(output / "sections.jsonl", section_content)
    audit = audit_coordinates(section_records, details, get, audit_count)
    audit_content = json.dumps(audit, indent=2, sort_keys=True).encode()
    _immutable_write(output / "coordinate_audit.json", audit_content)

    snapshot_date = datetime.now(timezone.utc).date().isoformat()
    existing_provenance = output / "provenance.json"
    if existing_provenance.exists():
        snapshot_date = json.loads(existing_provenance.read_text(encoding="utf-8"))["snapshot_date_utc"]
    split_counts = {}
    for split in ("train", "validation", "test", "sealed_deepslice_s2p"):
        split_datasets = [row for row in dataset_records if row["split"] == split]
        split_sections = [row for row in section_records if row["split"] == split]
        split_counts[split] = {
            "specimens": len({row["specimen_id"] for row in split_datasets}),
            "experiments": len(split_datasets),
            "sections": len(split_sections),
            "sections_in_training_ap_domain": sum(row["in_training_ap_domain"] for row in split_sections),
        }
    provenance = {
        "schema_version": 1,
        "snapshot_date_utc": snapshot_date,
        "api_root": API_ROOT,
        "trainable_query": {
            "products": list(PRODUCT_IDS),
            "failed": False,
            "reference_space_id": 9,
            "plane_of_section_id": 1,
        },
        "sealed_deepslice_s2p_experiment_ids": list(DEEPSLICE_S2P_EXPERIMENT_IDS),
        "sealed_benchmark_source": {
            "article_doi": "10.1038/s41467-023-41645-4",
            "figshare_doi": "10.25949/22802411",
            "figure_1_source_data_url": "https://ndownloader.figshare.com/files/40568876",
        },
        "coordinate_sources": {
            "section_image_formula": "https://api.brain-map.org/doc/SectionImage.html#image_to_reference-instance_method",
            "allen2quicknii": "https://github.com/Neural-Systems-at-UIO/allen2quicknii/blob/master/allen2quicknii.py",
        },
        "pose_convention": "AP um from bregma (+ anterior); L-R and D-V tilt in degrees",
        "training_ap_domain_um": [AP_MIN_UM, AP_MAX_UM],
        "download": {
            "format": "JPEG",
            "downsample": 5,
            "equalization": "published dataset RGB ranges when available; Allen defaults otherwise",
        },
        "split": {"unit": "specimen_id", "salt": SPLIT_SALT, "fractions": [0.90, 0.05, 0.05]},
        "datasets_sha256": _sha256_bytes(dataset_content),
        "sections_sha256": _sha256_bytes(section_content),
        "coordinate_audit_sha256": _sha256_bytes(audit_content),
        "dataset_count": len(dataset_records),
        "section_count": len(section_records),
        "split_counts": split_counts,
    }
    _immutable_write(output / "provenance.json", json.dumps(provenance, indent=2, sort_keys=True).encode())
    downloads = [] if metadata_only else download_sections(section_records, output, get, workers)
    return {**provenance, "downloaded_section_count": len(downloads), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire registered Allen S2P sections for AtlasPose training")
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dataset-cap", type=int)
    parser.add_argument("--image-cap", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--audit-count", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(acquire(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
