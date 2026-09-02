from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests


SCHEMA_VERSION = "allen-real-histology-metadata-v1"
API_ROOT = "https://api.brain-map.org"
PRODUCT_ID = 5
REFERENCE_SPACE_ID = 9
PLANE_OF_SECTION_ID = 1
DOWNSAMPLE = 5
SPLIT_SALT = "anatomy-tracker-allen-donor-development-v1"
OFFICIAL_DOCUMENTS = {
    "connectivity_api": "https://brain-map.org/support/documentation/api-allen-brain-connectivity-atlas",
    "image_download": "https://brain-map.org/support/tutorials/downloading-an-image",
    "terms_of_use": "https://alleninstitute.org/legal/terms-of-use",
    "citation_policy": "https://alleninstitute.org/legal/citation-policy",
}
ALLOWED_SOURCE_HOSTS = frozenset(
    {"api.brain-map.org", "brain-map.org", "www.brain-map.org", "alleninstitute.org", "www.alleninstitute.org"}
)
ALIGNMENT2D_KEYS = tuple(f"tsv_{index:02d}" for index in range(6))
ALIGNMENT3D_KEYS = tuple(f"tvr_{index:02d}" for index in range(12))


def _canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Immutable Allen metadata artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _official_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SOURCE_HOSTS:
        raise ValueError(f"Allen metadata source is not an approved official HTTPS host: {url}")
    return url


def _snapshot_response(
    get,
    url: str,
    output: Path,
    relative_path: str,
    params: dict | None = None,
) -> tuple[object, dict]:
    _official_url(url)
    response = get(url, params=params, timeout=60)
    response.raise_for_status()
    final_url = _official_url(response.url)
    content = bytes(response.content)
    _write_once(output, content)
    receipt = {
        "requested_url": url,
        "response_url": final_url,
        "sha256": _sha256(content),
        "bytes": len(content),
        "content_type": response.headers.get("Content-Type"),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "relative_path": relative_path,
    }
    return response, receipt


def split_for_animal(animal_id: int) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{int(animal_id)}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "development_train" if fraction < 0.9 else "development_validation"


def _index_query(start: int, count: int) -> str:
    return (
        "model::SectionDataSet,rma::criteria,"
        "[failed$eqfalse],[reference_space_id$eq9],[plane_of_section_id$eq1],"
        "products[id$eq5],"
        f"rma::options[num_rows$eq{count}][start_row$eq{start}][order$eq'id']"
    )


def _equalization_range(dataset: dict) -> list[int] | None:
    equalization = dataset.get("equalization")
    if not equalization:
        return None
    return [
        int(equalization[f"{channel}_{bound}"])
        for channel in ("red", "green", "blue")
        for bound in ("lower", "upper")
    ]


def _image_url(section_id: int, ranges: list[int] | None) -> str:
    query = {"downsample": DOWNSAMPLE}
    if ranges is not None:
        query = {"range": ",".join(map(str, ranges)), **query}
    return f"{API_ROOT}/api/v2/image_download/{int(section_id)}?{urlencode(query)}"


def _experiment_record(dataset: dict, source_receipt: dict) -> dict:
    specimen = dataset.get("specimen") or {}
    donor = specimen.get("donor") or {}
    animal_id = donor.get("id", specimen.get("donor_id"))
    split = "ineligible_unresolved_animal" if animal_id is None else split_for_animal(int(animal_id))
    product_ids = sorted(int(product["id"]) for product in dataset.get("products", []))
    return {
        "schema_version": SCHEMA_VERSION,
        "animal_id_namespace": "Allen Donor.id",
        "animal_id": None if animal_id is None else int(animal_id),
        "animal_partition_key": None if animal_id is None else f"allen-donor:{int(animal_id)}",
        "specimen_id": int(dataset["specimen_id"]),
        "experiment_id": int(dataset["id"]),
        "split": split,
        "product_ids": product_ids,
        "failed": bool(dataset["failed"]),
        "reference_space_id": int(dataset["reference_space_id"]),
        "plane_of_section_id": int(dataset["plane_of_section_id"]),
        "section_thickness_um": float(dataset["section_thickness"]),
        "alignment3d_tvr": (
            [float(dataset["alignment3d"][key]) for key in ALIGNMENT3D_KEYS]
            if dataset.get("alignment3d")
            else None
        ),
        "red_channel": dataset.get("red_channel"),
        "green_channel": dataset.get("green_channel"),
        "blue_channel": dataset.get("blue_channel"),
        "equalization_range_rgb": _equalization_range(dataset),
        "source_api_url": source_receipt["response_url"],
        "source_response_sha256": source_receipt["sha256"],
        "source_experiment_page_url": (
            f"https://connectivity.brain-map.org/projection/experiment/{int(dataset['id'])}"
        ),
        "upstream_release_version": None,
        "upstream_version_status": "not exposed by the Allen API; frozen by access time and response hash",
        "training_role": "real_histology_appearance_only",
        "geometry_role": "registered_canonical_coronal_metadata_not_arbitrary_plane_coverage",
        "learned_source_dependency": "none",
    }


def _section_records(dataset: dict, experiment: dict, source_receipt: dict) -> list[dict]:
    ranges = _equalization_range(dataset)
    rows = []
    for section in sorted(dataset.get("section_images", []), key=lambda row: (row["section_number"], row["id"])):
        reasons = []
        if experiment["animal_id"] is None:
            reasons.append("unresolved_animal_id")
        if section.get("failed"):
            reasons.append("failed_section")
        if not section.get("alignment2d"):
            reasons.append("missing_alignment2d")
        alignment = section.get("alignment2d")
        section_id = int(section["id"])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "animal_id_namespace": experiment["animal_id_namespace"],
                "animal_id": experiment["animal_id"],
                "animal_partition_key": experiment["animal_partition_key"],
                "specimen_id": experiment["specimen_id"],
                "experiment_id": experiment["experiment_id"],
                "section_id_namespace": "Allen SectionImage.id",
                "section_id": section_id,
                "section_number": int(section["section_number"]),
                "split": experiment["split"],
                "eligible_for_appearance_training": not reasons,
                "exclusion_reasons": reasons,
                "width_full_resolution_px": int(section["width"]),
                "height_full_resolution_px": int(section["height"]),
                "resolution_um_per_px": float(section["resolution"]),
                "alignment2d_tsv": (
                    [float(alignment[key]) for key in ALIGNMENT2D_KEYS] if alignment else None
                ),
                "source_experiment_response_sha256": source_receipt["sha256"],
                "source_section_api_url": f"{API_ROOT}/api/v2/data/SectionImage/{section_id}.json",
                "image_download_url": _image_url(section_id, ranges),
                "image_status": "not_downloaded",
                "image_sha256": None,
                "training_role": "real_histology_appearance_only",
                "geometry_role": "registered_canonical_coronal_metadata_not_arbitrary_plane_coverage",
                "learned_source_dependency": "none",
            }
        )
    return rows


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(_canonical_bytes(row) for row in rows)


def acquire_metadata_snapshot(
    output: str | Path,
    max_experiments: int,
    *,
    page_size: int = 64,
    get=requests.get,
    retrieved_at_utc: str | None = None,
) -> dict:
    output = Path(output)
    if max_experiments < 1 or page_size < 1:
        raise ValueError("max_experiments and page_size must be positive")
    retrieved_at_utc = retrieved_at_utc or datetime.now(timezone.utc).isoformat()

    raw_receipts = []
    experiment_ids = []
    start = 0
    total_rows = None
    while len(experiment_ids) < max_experiments and (total_rows is None or start < total_rows):
        response, receipt = _snapshot_response(
            get,
            f"{API_ROOT}/api/v2/data/query.json",
            output / "raw" / "api" / f"index_{start:06d}.json",
            f"raw/api/index_{start:06d}.json",
            {"q": _index_query(start, page_size)},
        )
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"Allen index query failed: {payload}")
        raw_receipts.append(receipt)
        total_rows = int(payload["total_rows"])
        page = [int(row["id"]) for row in payload["msg"]]
        experiment_ids.extend(page[: max_experiments - len(experiment_ids)])
        if not page:
            break
        start += len(page)

    experiments = []
    sections = []
    for experiment_id in experiment_ids:
        response, receipt = _snapshot_response(
            get,
            f"{API_ROOT}/api/v2/data/SectionDataSet/{experiment_id}.json",
            output / "raw" / "api" / "experiments" / f"{experiment_id}.json",
            f"raw/api/experiments/{experiment_id}.json",
            {"include": "specimen(donor),products,alignment3d,equalization,section_images(alignment2d)"},
        )
        payload = response.json()
        if not payload.get("success") or len(payload.get("msg", [])) != 1:
            raise RuntimeError(f"Allen experiment query failed for {experiment_id}: {payload}")
        dataset = payload["msg"][0]
        if (
            bool(dataset["failed"])
            or int(dataset["reference_space_id"]) != REFERENCE_SPACE_ID
            or int(dataset["plane_of_section_id"]) != PLANE_OF_SECTION_ID
            or PRODUCT_ID not in {int(product["id"]) for product in dataset.get("products", [])}
        ):
            raise RuntimeError(f"Allen experiment {experiment_id} violates the Product-5 source contract")
        raw_receipts.append(receipt)
        experiment = _experiment_record(dataset, receipt)
        experiments.append(experiment)
        sections.extend(_section_records(dataset, experiment, receipt))

    official_source_receipts = {}
    for name, url in OFFICIAL_DOCUMENTS.items():
        _, receipt = _snapshot_response(
            get,
            url,
            output / "raw" / "official_documents" / f"{name}.html",
            f"raw/official_documents/{name}.html",
        )
        official_source_receipts[name] = receipt
        raw_receipts.append(receipt)

    experiments.sort(key=lambda row: row["experiment_id"])
    sections.sort(key=lambda row: (row["experiment_id"], row["section_number"], row["section_id"]))
    _write_once(output / "experiments.jsonl", _jsonl_bytes(experiments))
    _write_once(output / "sections.jsonl", _jsonl_bytes(sections))
    source_code_sha256 = _sha256(Path(__file__).read_bytes())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "retrieved_at_utc": retrieved_at_utc,
        "source": {
            "provider": "Allen Institute",
            "api_contract": "Allen Brain Atlas RMA API v2",
            "product_id": PRODUCT_ID,
            "product_name": "Mouse Connectivity Projection",
            "reference_space_id": REFERENCE_SPACE_ID,
            "plane_of_section_id": PLANE_OF_SECTION_ID,
            "upstream_release_version": None,
            "upstream_version_status": "not exposed by the API; this snapshot is identified by time and exact response hashes",
            "official_document_receipts": official_source_receipts,
            "terms_summary": "Allen Institute Content is subject to its Terms of Use and Citation Policy; no SPDX dataset license is asserted here",
            "redistribution_policy": "manifest metadata only; image redistribution requires a separate terms review",
        },
        "data_role": {
            "real_sections": "appearance/domain supervision from mostly canonical coronal sections",
            "arbitrary_plane_geometry": "must come from the CCF atlas and provenance-bound synthetic arbitrary-plane generator",
            "alignment_metadata": "retained for provenance and bounded diagnostics, not evidence of arbitrary-plane coverage",
            "pretrained_models_features_pseudolabels": "none",
            "images": "not downloaded",
        },
        "partition_policy": {
            "unit": "Allen Donor.id",
            "partition_key": "allen-donor:{Donor.id}",
            "development_train_fraction": 0.9,
            "development_validation_fraction": 0.1,
            "split_salt": SPLIT_SALT,
            "section_or_experiment_splitting": "forbidden",
            "final_test": "not defined or accessed by this development snapshot",
        },
        "counts": {
            "animals": len({row["animal_id"] for row in experiments if row["animal_id"] is not None}),
            "specimens": len({row["specimen_id"] for row in experiments}),
            "experiments": len(experiments),
            "sections_total": len(sections),
            "sections_eligible_for_appearance_training": sum(
                row["eligible_for_appearance_training"] for row in sections
            ),
            "sections_excluded": sum(not row["eligible_for_appearance_training"] for row in sections),
        },
        "code": {
            "source_relative_path": "training/allen_real_histology_metadata.py",
            "source_sha256": source_code_sha256,
        },
        "raw_source_receipts": sorted(raw_receipts, key=lambda row: row["relative_path"]),
    }
    _write_once(output / "manifest.json", _canonical_bytes(manifest))
    file_receipts = []
    for path in sorted(path for path in output.rglob("*") if path.is_file() and path.name != "receipt.json"):
        content = path.read_bytes()
        file_receipts.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": _sha256(content),
                "bytes": len(content),
            }
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "metadata_only_no_images",
        "files": file_receipts,
    }
    _write_once(output / "receipt.json", _canonical_bytes(receipt))
    verify_metadata_snapshot(output)
    return manifest


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_metadata_snapshot(output: str | Path) -> dict:
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION or receipt["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Allen metadata schema version mismatch")
    expected_paths = {row["relative_path"] for row in receipt["files"]} | {"receipt.json"}
    actual_paths = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("Allen metadata snapshot artifact set mismatch")
    for row in receipt["files"]:
        content = (output / row["relative_path"]).read_bytes()
        if len(content) != row["bytes"] or _sha256(content) != row["sha256"]:
            raise ValueError(f"Allen metadata artifact hash mismatch: {row['relative_path']}")

    experiments = _read_jsonl(output / "experiments.jsonl")
    sections = _read_jsonl(output / "sections.jsonl")
    experiment_by_id = {row["experiment_id"]: row for row in experiments}
    if len(experiment_by_id) != len(experiments) or len({row["section_id"] for row in sections}) != len(sections):
        raise ValueError("Allen experiment or section IDs are not unique")
    animal_splits = {}
    for row in experiments:
        if row["animal_id"] is not None:
            animal_splits.setdefault(row["animal_id"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in animal_splits.values()):
        raise ValueError("Allen animal crosses development splits")
    for row in sections:
        parent = experiment_by_id.get(row["experiment_id"])
        if parent is None or any(
            row[field] != parent[field]
            for field in ("animal_id", "animal_partition_key", "specimen_id", "split")
        ):
            raise ValueError("Allen section lineage differs from its experiment")
        if row["image_status"] != "not_downloaded" or row["image_sha256"] is not None:
            raise ValueError("Metadata-only Allen snapshot unexpectedly claims image bytes")
        _official_url(row["source_section_api_url"])
        _official_url(row["image_download_url"])
    if manifest["counts"]["experiments"] != len(experiments) or manifest["counts"]["sections_total"] != len(sections):
        raise ValueError("Allen metadata manifest counts mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "animals": len(animal_splits),
        "experiments": len(experiments),
        "sections": len(sections),
        "development_splits": sorted({row["split"] for row in experiments}),
        "images_downloaded": 0,
    }
