from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


REGISTERED_IMAGE_QUALITY_MANIFEST = "registered_image_quality.json"
REGISTERED_IMAGE_QUALITY_VERSION = "gray-p0.01-p99.99-span8-std1-v1"
REGISTERED_IMAGE_QUALITY_SPLITS = (
    "train",
    "validation",
    "test",
    "sealed_deepslice_s2p",
)
REGISTERED_IMAGE_MIN_ROBUST_SPAN = 8.0
REGISTERED_IMAGE_MIN_STANDARD_DEVIATION = 1.0


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def registered_image_quality_metrics(image: np.ndarray) -> dict[str, float]:
    pixels = np.squeeze(np.asarray(image))
    gray = (
        pixels[..., :3].astype(np.float32).mean(axis=2)
        if pixels.ndim == 3
        else pixels.astype(np.float32)
    )
    low, high = np.percentile(gray, [0.01, 99.99])
    return {
        "robust_span": float(high - low),
        "standard_deviation": float(gray.std()),
    }


def registered_image_rejection_reason(metrics: dict[str, float]) -> str | None:
    if (
        metrics["robust_span"] < REGISTERED_IMAGE_MIN_ROBUST_SPAN
        or metrics["standard_deviation"] < REGISTERED_IMAGE_MIN_STANDARD_DEVIATION
    ):
        return "uninformative_intensity"
    return None


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_registered_image_quality_manifest(root: str | Path, workers: int = 16) -> dict:
    root = Path(root)
    sections_path = root / "sections.jsonl"
    downloads_path = root / "downloads.jsonl"
    provenance_path = root / "provenance.json"
    sections = _read_jsonl(sections_path)
    downloads = {
        int(record["section_image_id"]): record["sha256"]
        for record in _read_jsonl(downloads_path)
    }
    assessed = [record for record in sections if record["split"] in REGISTERED_IMAGE_QUALITY_SPLITS]
    def assess(record: dict) -> dict | None:
        section_id = int(record["section_image_id"])
        with Image.open(root / record["relative_path"]) as image:
            metrics = registered_image_quality_metrics(np.asarray(image))
        reason = registered_image_rejection_reason(metrics)
        if reason is None:
            return None
        return {
            "section_image_id": section_id,
            "image_sha256": downloads[section_id],
            "reason": reason,
            "metrics": metrics,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rejected = [record for record in pool.map(assess, assessed) if record is not None]
    rejected_ids = {record["section_image_id"] for record in rejected}
    approved_records = sorted(
        (int(record["section_image_id"]), downloads[int(record["section_image_id"])])
        for record in assessed
        if int(record["section_image_id"]) not in rejected_ids
    )
    payload = {
        "schema_version": 1,
        "quality_contract_version": REGISTERED_IMAGE_QUALITY_VERSION,
        "assessed_splits": list(REGISTERED_IMAGE_QUALITY_SPLITS),
        "thresholds": {
            "minimum_robust_span": REGISTERED_IMAGE_MIN_ROBUST_SPAN,
            "minimum_standard_deviation": REGISTERED_IMAGE_MIN_STANDARD_DEVIATION,
        },
        "source": {
            "sections_sha256": _file_sha256(sections_path),
            "downloads_sha256": _file_sha256(downloads_path),
            "provenance_sha256": _file_sha256(provenance_path),
        },
        "assessed_record_count": len(assessed),
        "approved_record_count": len(approved_records),
        "approved_records_sha256": _json_sha256(approved_records),
        "rejected_records": sorted(rejected, key=lambda record: record["section_image_id"]),
    }
    payload["manifest_sha256"] = _json_sha256(payload)
    destination = root / REGISTERED_IMAGE_QUALITY_MANIFEST
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)
    return payload


def load_registered_image_quality_manifest(
    root: str | Path,
) -> tuple[dict, frozenset[int], dict[int, dict]]:
    root = Path(root)
    path = root / REGISTERED_IMAGE_QUALITY_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"Registered-image quality manifest is unavailable: {path}. "
            "Build it after the immutable image download is complete."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checksum = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if checksum != _json_sha256(payload):
        raise ValueError("Registered-image quality manifest checksum failed")
    if (
        manifest.get("quality_contract_version") != REGISTERED_IMAGE_QUALITY_VERSION
        or manifest.get("assessed_splits") != list(REGISTERED_IMAGE_QUALITY_SPLITS)
        or manifest.get("thresholds")
        != {
            "minimum_robust_span": REGISTERED_IMAGE_MIN_ROBUST_SPAN,
            "minimum_standard_deviation": REGISTERED_IMAGE_MIN_STANDARD_DEVIATION,
        }
    ):
        raise ValueError("Registered-image quality manifest uses a different quality contract")
    expected_source = {
        "sections_sha256": _file_sha256(root / "sections.jsonl"),
        "downloads_sha256": _file_sha256(root / "downloads.jsonl"),
        "provenance_sha256": _file_sha256(root / "provenance.json"),
    }
    if manifest.get("source") != expected_source:
        raise ValueError("Registered-image quality manifest does not match the immutable source")

    sections = _read_jsonl(root / "sections.jsonl")
    downloads = {
        int(record["section_image_id"]): record["sha256"]
        for record in _read_jsonl(root / "downloads.jsonl")
    }
    assessed = [record for record in sections if record["split"] in REGISTERED_IMAGE_QUALITY_SPLITS]
    assessed_ids = {int(record["section_image_id"]) for record in assessed}
    rejected = {int(record["section_image_id"]): record for record in manifest["rejected_records"]}
    if len(rejected) != len(manifest["rejected_records"]) or not set(rejected).issubset(assessed_ids):
        raise ValueError("Registered-image quality manifest has invalid rejected section IDs")
    for section_id, record in rejected.items():
        if (
            record.get("image_sha256") != downloads.get(section_id)
            or registered_image_rejection_reason(record.get("metrics", {})) != record.get("reason")
        ):
            raise ValueError("Registered-image quality rejection differs from its source or thresholds")
    approved = assessed_ids - set(rejected)
    approved_records = sorted((section_id, downloads[section_id]) for section_id in approved)
    if (
        manifest.get("assessed_record_count") != len(assessed)
        or manifest.get("approved_record_count") != len(approved)
        or manifest.get("approved_records_sha256") != _json_sha256(approved_records)
    ):
        raise ValueError("Registered-image approved-record contract is inconsistent")
    return manifest, frozenset(approved), rejected
