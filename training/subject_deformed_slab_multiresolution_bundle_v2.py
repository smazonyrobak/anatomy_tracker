"""Atomic raw bundle and independent replay verification for the fixed-case assessment."""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import nrrd
import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_subject_slab_v2 as subject_slab
import training.arbitrary_plane_synthetic_generator_v2 as slab
import training.subject_deformed_slab_multiresolution_assessment_v2 as assessment
from training.arbitrary_plane_support import build_annotation_support_index


BUNDLE_SCHEMA = "anatomy-tracker.subject-slab-fixed-case-multiresolution-bundle/v2"
TEMPLATE_SHA256 = "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
ANNOTATION_SHA256 = "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
PYNRRD_VERSION = "1.1.3"
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "subject_deformed_slab_multiresolution_bundle_v2.py",
    "subject_deformed_slab_multiresolution_assessment_v2.py",
    "run_subject_deformed_slab_multiresolution_v2.py",
    "verify_subject_deformed_slab_multiresolution_bundle_v2.py",
    "subject_deformed_slab_qualification_v2.py",
    "slab_refinement_gate_status_v2.py",
    "arbitrary_plane_acquisition_v2.py",
    "arbitrary_plane_subject_deformation_v2.py",
    "arbitrary_plane_subject_slab_v2.py",
    "arbitrary_plane_synthetic_generator_v2.py",
    "arbitrary_plane_support.py",
)


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_allen_inputs(atlas_folder: str | Path) -> dict[str, object]:
    if nrrd.__version__ != PYNRRD_VERSION:
        raise ValueError("pynrrd runtime does not match the pinned decoder")
    folder = Path(atlas_folder).resolve()
    template = folder / "average_template_25.nrrd"
    annotation = folder / "annotation_25.nrrd"
    result = {
        "decoder": f"pynrrd {PYNRRD_VERSION}",
        "index_order": "F",
        "template": {
            "path": str(template),
            "sha256": _file_sha256(template),
            "byte_count": template.stat().st_size,
        },
        "annotation": {
            "path": str(annotation),
            "sha256": _file_sha256(annotation),
            "byte_count": annotation.stat().st_size,
        },
    }
    if (
        result["template"]["sha256"] != TEMPLATE_SHA256
        or result["annotation"]["sha256"] != ANNOTATION_SHA256
    ):
        raise ValueError("Allen inputs or decoder do not match the pinned contract")
    return result


def load_pinned_allen_context(
    atlas_folder: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    inputs = pinned_allen_inputs(atlas_folder)
    template = nrrd.read(inputs["template"]["path"], index_order="F")[0]
    annotation = nrrd.read(inputs["annotation"]["path"], index_order="F")[0]
    support = build_annotation_support_index(
        annotation,
        atlas_id="Allen CCFv3",
        atlas_version="2017 25um",
        source_uri="data/Allen Brain Atlas 25um/annotation_25.nrrd",
        source_sha256=ANNOTATION_SHA256,
        source_entity_type="atlas-annotation",
        voxel_size_um=(25.0, 25.0, 25.0),
        origin_um=(0.0, 0.0, 0.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    context = acquisition.prepare_arbitrary_plane_acquisition_context_v2(
        template,
        annotation,
        support,
        scalar_source_uri="data/Allen Brain Atlas 25um/average_template_25.nrrd",
        scalar_source_sha256=TEMPLATE_SHA256,
        scalar_source_entity_type="atlas-template",
        template_decoder=f"pynrrd {PYNRRD_VERSION}",
        template_index_order="F",
        annotation_decoder=f"pynrrd {PYNRRD_VERSION}",
        annotation_index_order="F",
    )
    return context, inputs


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        acquisition._canonical_json(value) + "\n", encoding="utf-8", newline="\n"
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_arrays(value: object, arrays: dict[str, np.ndarray]) -> object:
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):04d}"
        arrays[key] = np.ascontiguousarray(value)
        return {"__ndarray__": key}
    if isinstance(value, Mapping):
        return {str(key): _split_arrays(item, arrays) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_split_arrays(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_split_arrays(item, arrays) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _restore_arrays(value: object, arrays: Mapping[str, np.ndarray]) -> object:
    if isinstance(value, dict) and set(value) == {"__ndarray__"}:
        return np.array(arrays[value["__ndarray__"]], copy=True, order="C")
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_restore_arrays(item, arrays) for item in value["__tuple__"])
    if isinstance(value, dict):
        return {key: _restore_arrays(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_arrays(item, arrays) for item in value]
    return value


def _write_raw_artifact(staging: Path, stem: str, value: Mapping[str, object]) -> dict[str, str]:
    arrays = {}
    metadata = _split_arrays(value, arrays)
    metadata_path = staging / f"{stem}.metadata.json"
    arrays_path = staging / f"{stem}.arrays.npz"
    _write_json(metadata_path, metadata)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    with arrays_path.open("wb") as stream:
        np.savez(stream, **arrays)
    return {
        "metadata": metadata_path.relative_to(staging).as_posix(),
        "arrays": arrays_path.relative_to(staging).as_posix(),
    }


def _read_raw_artifact(root: Path, reference: Mapping[str, str]) -> dict[str, object]:
    metadata = _read_json(root / reference["metadata"])
    with np.load(root / reference["arrays"], allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    return _restore_arrays(metadata, arrays)


def _inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "byte_count": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "bundle-manifest.json"
    ]


def write_staged_bundle_v2(
    output: str | Path,
    *,
    repository: Mapping[str, object],
    allen_inputs: Mapping[str, object],
    failed_report: Mapping[str, object],
    plan: Mapping[str, object],
    subject_plan: Mapping[str, object],
    rendered: Mapping[str, object],
    result: Mapping[str, object],
) -> Path:
    """Write a complete unpublished sibling tree; no existing path is overwritten."""
    output = Path(output).resolve()
    staging = output.parent / f".{output.name}.partial"
    if output.exists() or staging.exists():
        raise FileExistsError("frozen output or its partial sibling already exists")
    staging.mkdir(parents=True)
    _write_json(staging / "legacy-failed-report.json", failed_report)
    _write_json(staging / "plan.json", plan)
    _write_json(staging / "assessment.json", result)
    raw = {
        "subject_deformation_plan": _write_raw_artifact(
            staging, "raw/subject-deformation-plan", subject_plan
        ),
        "precursors": {},
        "renders": {arm: {} for arm in assessment.ARM_NAMES},
    }
    for step in assessment.AXIAL_STEPS_UM_MAX:
        key = f"{step:g}"
        raw["precursors"][key] = _write_raw_artifact(
            staging, f"raw/precursors/{key}", rendered["precursors"][key]
        )
        for arm in assessment.ARM_NAMES:
            raw["renders"][arm][key] = _write_raw_artifact(
                staging,
                f"raw/renders/{arm}/{key}",
                rendered["renders"][arm][key],
            )
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "role": "immutable threshold-free fixed-case numerical assessment",
        "qualification_eligible": False,
        "acceptance_thresholds": None,
        "repository": acquisition._json_value(repository),
        "allen_inputs": acquisition._json_value(allen_inputs),
        "documents": {
            "failed_report": "legacy-failed-report.json",
            "plan": "plan.json",
            "assessment": "assessment.json",
        },
        "raw_artifacts": raw,
        "file_inventory": _inventory(staging),
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_sha256_canonicalization": (
            acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        ),
    }
    manifest["bundle_receipt_sha256"] = acquisition._payload_sha256(manifest)
    _write_json(staging / "bundle-manifest.json", manifest)
    return staging


def publish_staged_bundle_v2(staging: str | Path, output: str | Path) -> None:
    staging = Path(staging).resolve()
    output = Path(output).resolve()
    if output.exists() or staging != output.parent / f".{output.name}.partial":
        raise FileExistsError("output exists or staging is not its exact sibling")
    os.replace(staging, output)


def _verify_persisted_array_receipts(
    rendered: Mapping[str, object],
) -> None:
    for step in assessment.AXIAL_STEPS_UM_MAX:
        key = f"{step:g}"
        precursor = rendered["precursors"][key]
        precursor_arrays = slab._slab_arrays(precursor["raster"])
        if precursor["raster"]["array_receipts"] != {
            name: acquisition._array_receipt(array)
            for name, array in precursor_arrays.items()
        }:
            raise ValueError("precursor persisted array receipt does not match")
        for arm in assessment.ARM_NAMES:
            artifact = rendered["renders"][arm][key]
            coordinate = artifact["coordinate_map"]
            samples = artifact["sample_arrays"]
            reduced = subject_slab._reduced_arrays(artifact["raster"])
            if (
                coordinate["array_receipts"]
                != {
                    name: acquisition._array_receipt(array)
                    for name, array in coordinate["arrays"].items()
                }
                or artifact["sample_array_receipts"]
                != {
                    name: acquisition._array_receipt(array)
                    for name, array in samples.items()
                }
                or artifact["raster_array_receipts"]
                != {
                    name: acquisition._array_receipt(array)
                    for name, array in reduced.items()
                }
            ):
                raise ValueError("subject render persisted array receipt does not match")


def verify_frozen_bundle_v2(
    root: str | Path,
    prepared_context: Mapping[str, object],
    allen_inputs: Mapping[str, object],
    *,
    repository: Mapping[str, object],
    batch_size: int | None = None,
) -> dict[str, object]:
    """Verify staged bytes, receipts, source binding, and measurements without rerendering."""
    root = Path(root).resolve()
    manifest = _read_json(root / "bundle-manifest.json")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "bundle-manifest.json"
    }
    inventory = manifest.get("file_inventory", [])
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA
        or manifest.get("qualification_eligible") is not False
        or manifest.get("acceptance_thresholds") is not None
        or manifest.get("repository") != acquisition._json_value(repository)
        or manifest.get("allen_inputs") != acquisition._json_value(allen_inputs)
        or manifest.get("implementation_source_sha256") != _source_hashes()
        or manifest.get("implementation_source_sha256_canonicalization")
        != acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        or not _receipt_matches(manifest, "bundle_receipt_sha256")
        or actual_files != {item["path"] for item in inventory}
        or any(
            (root / item["path"]).stat().st_size != item["byte_count"]
            or _file_sha256(root / item["path"]) != item["sha256"]
            for item in inventory
        )
    ):
        raise ValueError("frozen multiresolution bundle inventory or provenance is invalid")
    documents = manifest["documents"]
    failed_report = _read_json(root / documents["failed_report"])
    plan = _read_json(root / documents["plan"])
    result = _read_json(root / documents["assessment"])
    raw = manifest["raw_artifacts"]
    subject_plan = _read_raw_artifact(root, raw["subject_deformation_plan"])
    rendered = {
        "precursors": {
            key: _read_raw_artifact(root, reference)
            for key, reference in raw["precursors"].items()
        },
        "renders": {
            arm: {
                key: _read_raw_artifact(root, reference)
                for key, reference in levels.items()
            }
            for arm, levels in raw["renders"].items()
        },
    }
    assessment._verify_fixed_case_multiresolution_plan_structure_v2(
        plan, failed_report, prepared_context
    )
    if not assessment._subject_plan_matches(plan, subject_plan):
        raise ValueError("frozen nonidentity subject plan does not match the selected animal")
    _verify_persisted_array_receipts(rendered)
    assessment.verify_fixed_case_multiresolution_assessment_v2(
        result, plan, rendered
    )
    return {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_receipt_sha256": manifest["bundle_receipt_sha256"],
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "assessment_receipt_sha256": result["assessment_receipt_sha256"],
        "raw_render_count": sum(len(levels) for levels in raw["renders"].values()),
        "qualification_eligible": False,
        "acceptance_thresholds": None,
    }


def _receipt_matches(payload: Mapping[str, object], receipt_name: str) -> bool:
    return payload.get(receipt_name) == acquisition._payload_sha256(
        {key: value for key, value in payload.items() if key != receipt_name}
    )
