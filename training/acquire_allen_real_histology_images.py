from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

from training.allen_real_histology_metadata import (
    _canonical_bytes,
    _official_url,
    _snapshot_root,
    verify_metadata_snapshot,
)


SCHEMA_VERSION = "allen-real-histology-images-v1"
SELECTION_SALT = "anatomy-tracker-allen-real-image-selection-v1"
DEVELOPMENT_SPLITS = ("development_train", "development_validation")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(_canonical_bytes(row) for row in rows)


def _write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Immutable Allen image artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _outside_git_on_i(output: str | Path) -> Path:
    root = _snapshot_root(output)
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
        raise ValueError(f"Allen raw image bytes must be outside a Git worktree: {root}")
    return root


def _rank(*parts: object) -> bytes:
    return hashlib.sha256(":".join((SELECTION_SALT, *map(str, parts))).encode("utf-8")).digest()


def deterministic_section_selection(sections: list[dict], quotas: dict[str, int]) -> list[dict]:
    if set(quotas) - set(DEVELOPMENT_SPLITS) or any(int(value) < 0 for value in quotas.values()):
        raise ValueError("Allen image quotas may contain only nonnegative development train/validation counts")
    selected = []
    for split in DEVELOPMENT_SPLITS:
        quota = int(quotas.get(split, 0))
        eligible = [
            row
            for row in sections
            if row["split"] == split
            and row["eligible_for_appearance_training"]
            and row["animal_id"] is not None
        ]
        by_animal = {}
        for row in eligible:
            by_animal.setdefault(int(row["animal_id"]), []).append(row)
        animals = sorted(by_animal, key=lambda animal_id: _rank(split, "animal", animal_id))
        for animal_id in animals:
            by_animal[animal_id].sort(
                key=lambda row: _rank(split, animal_id, row["experiment_id"], row["section_id"])
            )
        split_selected = []
        round_index = 0
        while len(split_selected) < quota:
            added = False
            for animal_id in animals:
                if len(split_selected) == quota:
                    break
                rows = by_animal[animal_id]
                if round_index < len(rows):
                    split_selected.append(rows[round_index])
                    added = True
            if not added:
                break
            round_index += 1
        if len(split_selected) != quota:
            raise ValueError(
                f"Requested {quota} Allen images from {split}, but only {len(eligible)} eligible rows exist"
            )
        selected.extend(split_selected)
    return selected


def _image_relative_path(row: dict) -> str:
    return (
        f"images/{row['split']}/animal_{int(row['animal_id'])}/"
        f"experiment_{int(row['experiment_id'])}/section_{int(row['section_id'])}.jpg"
    )


def _download_record(get, row: dict, output: Path, rank: int) -> dict:
    url = _official_url(row["image_download_url"])
    response = get(url, timeout=180)
    response.raise_for_status()
    response_url = _official_url(response.url)
    content = bytes(response.content)
    with Image.open(io.BytesIO(content)) as image:
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "JPEG":
            raise ValueError(f"Allen SectionImage {row['section_id']} did not return JPEG bytes")
        width, height = image.size
        mode = image.mode
    relative_path = _image_relative_path(row)
    _write_once(output / relative_path, content)
    source_row_sha256 = _sha256(_canonical_bytes(row))
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_rank": rank,
        "animal_id_namespace": row["animal_id_namespace"],
        "animal_id": int(row["animal_id"]),
        "animal_partition_key": row["animal_partition_key"],
        "specimen_id": int(row["specimen_id"]),
        "experiment_id": int(row["experiment_id"]),
        "section_id_namespace": row["section_id_namespace"],
        "section_id": int(row["section_id"]),
        "section_number": int(row["section_number"]),
        "split": row["split"],
        "source_section_record_sha256": source_row_sha256,
        "source_experiment_response_sha256": row["source_experiment_response_sha256"],
        "requested_url": url,
        "response_url": response_url,
        "content_type": response.headers.get("Content-Type"),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "relative_path": relative_path,
        "sha256": _sha256(content),
        "bytes": len(content),
        "image_format": "JPEG",
        "image_mode": mode,
        "width_px": int(width),
        "height_px": int(height),
        "training_role": "real_histology_appearance_only",
        "geometry_role": "registered_canonical_coronal_metadata_not_arbitrary_plane_coverage",
        "learned_source_dependency": "none",
    }


def _counts(sections: list[dict], selected: list[dict], downloaded: list[dict]) -> dict:
    result = {
        "pre_exclusion": {
            "section_records_total": len(sections),
            "eligible_for_appearance_training": sum(
                row["eligible_for_appearance_training"] for row in sections
            ),
            "excluded_by_metadata": sum(
                not row["eligible_for_appearance_training"] for row in sections
            ),
        },
        "post_selection": {
            "selected_for_download": len(selected),
            "eligible_not_selected": sum(
                row["eligible_for_appearance_training"] for row in sections
            )
            - len(selected),
        },
        "post_download": {
            "downloaded_and_hash_bound": len(downloaded),
            "transport_or_decode_exclusions": 0,
        },
        "by_split": {},
    }
    for split in DEVELOPMENT_SPLITS:
        split_rows = [row for row in sections if row["split"] == split]
        split_selected = [row for row in selected if row["split"] == split]
        split_downloaded = [row for row in downloaded if row["split"] == split]
        result["by_split"][split] = {
            "animals_available": len(
                {
                    row["animal_id"]
                    for row in split_rows
                    if row["animal_id"] is not None and row["eligible_for_appearance_training"]
                }
            ),
            "sections_total": len(split_rows),
            "sections_eligible": sum(row["eligible_for_appearance_training"] for row in split_rows),
            "animals_selected": len({row["animal_id"] for row in split_selected}),
            "sections_selected": len(split_selected),
            "sections_downloaded": len(split_downloaded),
        }
    return result


def acquire_image_snapshot(
    metadata_snapshot: str | Path,
    output: str | Path,
    quotas: dict[str, int],
    *,
    get=requests.get,
    retrieved_at_utc: str | None = None,
) -> dict:
    metadata_snapshot = _snapshot_root(metadata_snapshot)
    output = _outside_git_on_i(output)
    verify_metadata_snapshot(metadata_snapshot)
    retrieved_at_utc = retrieved_at_utc or datetime.now(timezone.utc).isoformat()

    metadata_files = {
        name: _file_sha256(metadata_snapshot / name)
        for name in ("manifest.json", "receipt.json", "experiments.jsonl", "sections.jsonl")
    }
    metadata_manifest = json.loads(
        (metadata_snapshot / "manifest.json").read_text(encoding="utf-8")
    )
    experiments = _read_jsonl(metadata_snapshot / "experiments.jsonl")
    sections = _read_jsonl(metadata_snapshot / "sections.jsonl")
    experiment_by_id = {row["experiment_id"]: row for row in experiments}
    normalized_quotas = {split: int(quotas.get(split, 0)) for split in DEVELOPMENT_SPLITS}
    selected = deterministic_section_selection(sections, normalized_quotas)
    downloaded = [
        _download_record(get, row, output, rank)
        for rank, row in enumerate(selected)
    ]
    for record in downloaded:
        experiment = experiment_by_id[record["experiment_id"]]
        record["source_experiment_record_sha256"] = _sha256(_canonical_bytes(experiment))

    _write_once(output / "images.jsonl", _jsonl_bytes(downloaded))
    counts = _counts(sections, selected, downloaded)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "retrieved_at_utc": retrieved_at_utc,
        "artifact_role": "bounded_allen_product5_real_image_development_source",
        "source_metadata": {
            "root": metadata_snapshot.as_posix(),
            "file_sha256": metadata_files,
            "schema_version": metadata_manifest["schema_version"],
            "product_id": metadata_manifest["source"]["product_id"],
            "reference_space_id": metadata_manifest["source"]["reference_space_id"],
            "official_document_receipts": metadata_manifest["source"][
                "official_document_receipts"
            ],
        },
        "selection": {
            "algorithm": "sha256 donor-round-robin then section rank v1",
            "salt": SELECTION_SALT,
            "quotas": normalized_quotas,
            "unit": "Allen Donor.id",
            "inherited_split_only": True,
            "final_test": "not defined or accessed",
        },
        "data_role": {
            "real_images": "appearance/domain supervision only",
            "arbitrary_plane_geometry": "CCF and provenance-bound synthetic generator only",
            "pretrained_models_features_pseudolabels": "none",
            "public_external_or_final_data": "not accessed",
        },
        "terms": {
            "terms_of_use_url": "https://alleninstitute.org/legal/terms-of-use",
            "citation_policy_url": "https://alleninstitute.org/legal/citation-policy",
            "license_spdx": None,
            "use_caveat": "Allen Institute Content remains subject to its Terms of Use and Citation Policy",
            "redistribution_caveat": "raw image bytes remain outside Git and are not approved for redistribution by this manifest",
        },
        "counts": counts,
        "code": {
            "source_relative_path": "training/acquire_allen_real_histology_images.py",
            "source_sha256": _file_sha256(Path(__file__)),
        },
    }
    _write_once(output / "manifest.json", _canonical_bytes(manifest))
    files = []
    for path in sorted(path for path in output.rglob("*") if path.is_file() and path.name != "receipt.json"):
        files.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "immutable_raw_allen_product5_bytes_and_manifests",
        "files": files,
    }
    _write_once(output / "receipt.json", _canonical_bytes(receipt))

    from training.verify_allen_real_histology_images import verify_image_snapshot

    verify_image_snapshot(output, metadata_snapshot)
    return manifest
