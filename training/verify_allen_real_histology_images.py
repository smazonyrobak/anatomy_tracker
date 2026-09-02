from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image


SCHEMA_VERSION = "allen-real-histology-images-v1"
SELECTION_SALT = "anatomy-tracker-allen-real-image-selection-v1"
DEVELOPMENT_SPLITS = ("development_train", "development_validation")


def _root_on_i(path: str | Path) -> Path:
    root = Path(path).resolve()
    if root.drive.upper() != "I:":
        raise ValueError(f"Allen image verification roots must resolve to I:, got {root}")
    return root


def _outside_git(root: Path) -> None:
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
        raise ValueError(f"Allen raw image bytes must remain outside a Git worktree: {root}")


def _canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rank(*parts: object) -> bytes:
    return hashlib.sha256(":".join((SELECTION_SALT, *map(str, parts))).encode("utf-8")).digest()


def _expected_selection(sections: list[dict], quotas: dict[str, int]) -> list[dict]:
    selected = []
    for split in DEVELOPMENT_SPLITS:
        quota = int(quotas[split])
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
            raise ValueError("Image manifest requests more rows than the source metadata contains")
        selected.extend(split_selected)
    return selected


def _expected_counts(sections: list[dict], selected: list[dict], images: list[dict]) -> dict:
    counts = {
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
            "downloaded_and_hash_bound": len(images),
            "transport_or_decode_exclusions": 0,
        },
        "by_split": {},
    }
    for split in DEVELOPMENT_SPLITS:
        split_rows = [row for row in sections if row["split"] == split]
        split_selected = [row for row in selected if row["split"] == split]
        split_images = [row for row in images if row["split"] == split]
        counts["by_split"][split] = {
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
            "sections_downloaded": len(split_images),
        }
    return counts


def _official_api_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "api.brain-map.org":
        raise ValueError(f"Allen image source is not the official HTTPS API: {url}")


def verify_image_snapshot(output: str | Path, metadata_snapshot: str | Path) -> dict:
    output = _root_on_i(output)
    metadata_snapshot = _root_on_i(metadata_snapshot)
    _outside_git(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION or receipt["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Allen image schema version mismatch")
    if (
        manifest["artifact_role"]
        != "bounded_allen_product5_real_image_development_source"
        or receipt["artifact_role"]
        != "immutable_raw_allen_product5_bytes_and_manifests"
    ):
        raise ValueError("Allen image artifact role mismatch")

    expected_paths = {row["relative_path"] for row in receipt["files"]} | {"receipt.json"}
    actual_paths = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("Allen image snapshot artifact set mismatch")
    for row in receipt["files"]:
        path = output / row["relative_path"]
        if path.stat().st_size != row["bytes"] or _file_sha256(path) != row["sha256"]:
            raise ValueError(f"Allen image artifact hash mismatch: {row['relative_path']}")

    source_binding = manifest["source_metadata"]
    if Path(source_binding["root"]).resolve() != metadata_snapshot:
        raise ValueError("Allen image snapshot was verified against a different metadata root")
    for name, digest in source_binding["file_sha256"].items():
        if _file_sha256(metadata_snapshot / name) != digest:
            raise ValueError(f"Allen source metadata hash mismatch: {name}")
    metadata_receipt = json.loads(
        (metadata_snapshot / "receipt.json").read_text(encoding="utf-8")
    )
    expected_metadata_paths = {
        row["relative_path"] for row in metadata_receipt["files"]
    } | {"receipt.json"}
    actual_metadata_paths = {
        path.relative_to(metadata_snapshot).as_posix()
        for path in metadata_snapshot.rglob("*")
        if path.is_file()
    }
    if actual_metadata_paths != expected_metadata_paths:
        raise ValueError("Allen source metadata artifact set mismatch")
    for row in metadata_receipt["files"]:
        path = metadata_snapshot / row["relative_path"]
        if path.stat().st_size != row["bytes"] or _file_sha256(path) != row["sha256"]:
            raise ValueError(f"Allen raw source metadata hash mismatch: {row['relative_path']}")
    metadata_manifest = json.loads(
        (metadata_snapshot / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_binding["schema_version"] != metadata_manifest["schema_version"]
        or source_binding["product_id"] != 5
        or source_binding["reference_space_id"] != 9
        or source_binding["official_document_receipts"]
        != metadata_manifest["source"]["official_document_receipts"]
    ):
        raise ValueError("Allen image source metadata semantics mismatch")

    experiments = _read_jsonl(metadata_snapshot / "experiments.jsonl")
    sections = _read_jsonl(metadata_snapshot / "sections.jsonl")
    images = _read_jsonl(output / "images.jsonl")
    experiment_by_id = {row["experiment_id"]: row for row in experiments}
    section_by_id = {row["section_id"]: row for row in sections}
    if len(section_by_id) != len(sections) or len({row["section_id"] for row in images}) != len(images):
        raise ValueError("Allen source or image section IDs are not unique")
    quotas = manifest["selection"]["quotas"]
    if (
        set(quotas) != set(DEVELOPMENT_SPLITS)
        or manifest["selection"]["salt"] != SELECTION_SALT
        or manifest["selection"]["algorithm"]
        != "sha256 donor-round-robin then section rank v1"
        or manifest["selection"]["inherited_split_only"] is not True
    ):
        raise ValueError("Allen image selection contract mismatch")
    selected = _expected_selection(sections, quotas)
    if [row["section_id"] for row in selected] != [row["section_id"] for row in images]:
        raise ValueError("Allen image rows differ from independent deterministic selection")

    raw_source_receipts = {
        row["relative_path"]: row["sha256"]
        for row in metadata_manifest["raw_source_receipts"]
    }
    for rank, image_row in enumerate(images):
        source = section_by_id[image_row["section_id"]]
        experiment = experiment_by_id[image_row["experiment_id"]]
        raw_experiment_path = f"raw/api/experiments/{int(image_row['experiment_id'])}.json"
        inherited = (
            "animal_id",
            "animal_partition_key",
            "specimen_id",
            "experiment_id",
            "section_id",
            "section_number",
            "split",
        )
        if image_row["selection_rank"] != rank or any(image_row[key] != source[key] for key in inherited):
            raise ValueError("Allen downloaded image lineage differs from source metadata")
        if image_row["split"] not in DEVELOPMENT_SPLITS:
            raise ValueError("Allen image snapshot contains a non-development split")
        if image_row["source_section_record_sha256"] != _sha256(_canonical_bytes(source)):
            raise ValueError("Allen source section record binding mismatch")
        if image_row["source_experiment_record_sha256"] != _sha256(_canonical_bytes(experiment)):
            raise ValueError("Allen source experiment record binding mismatch")
        if (
            image_row["source_experiment_response_sha256"]
            != source["source_experiment_response_sha256"]
            or raw_source_receipts.get(raw_experiment_path)
            != source["source_experiment_response_sha256"]
        ):
            raise ValueError("Allen raw API response binding mismatch")
        if image_row["requested_url"] != source["image_download_url"]:
            raise ValueError("Allen image URL differs from source section metadata")
        _official_api_url(image_row["requested_url"])
        _official_api_url(image_row["response_url"])
        expected_relative = (
            f"images/{source['split']}/animal_{int(source['animal_id'])}/"
            f"experiment_{int(source['experiment_id'])}/section_{int(source['section_id'])}.jpg"
        )
        if image_row["relative_path"] != expected_relative:
            raise ValueError("Allen image path does not match its source lineage")
        content = (output / expected_relative).read_bytes()
        if len(content) != image_row["bytes"] or _sha256(content) != image_row["sha256"]:
            raise ValueError("Allen image byte binding mismatch")
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            if (
                image.format != image_row["image_format"]
                or image.mode != image_row["image_mode"]
                or list(image.size) != [image_row["width_px"], image_row["height_px"]]
            ):
                raise ValueError("Allen decoded image metadata mismatch")
        if (
            image_row["training_role"] != "real_histology_appearance_only"
            or image_row["learned_source_dependency"] != "none"
            or image_row["geometry_role"]
            != "registered_canonical_coronal_metadata_not_arbitrary_plane_coverage"
            or image_row["image_format"] != "JPEG"
        ):
            raise ValueError("Allen image data role mismatch")

    if manifest["counts"] != _expected_counts(sections, selected, images):
        raise ValueError("Allen image pre/post exclusion counts mismatch")
    if (
        manifest["selection"]["final_test"] != "not defined or accessed"
        or manifest["data_role"]["pretrained_models_features_pseudolabels"] != "none"
        or manifest["data_role"]["public_external_or_final_data"] != "not accessed"
        or manifest["terms"]["license_spdx"] is not None
        or manifest["terms"]["terms_of_use_url"]
        != "https://alleninstitute.org/legal/terms-of-use"
        or manifest["terms"]["citation_policy_url"]
        != "https://alleninstitute.org/legal/citation-policy"
        or "outside Git" not in manifest["terms"]["redistribution_caveat"]
    ):
        raise ValueError("Allen image development/terms contract mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "animals": len({row["animal_id"] for row in images}),
        "experiments": len({row["experiment_id"] for row in images}),
        "images": len(images),
        "bytes": sum(row["bytes"] for row in images),
        "splits": sorted({row["split"] for row in images}),
        "source_metadata_manifest_sha256": source_binding["file_sha256"]["manifest.json"],
    }
