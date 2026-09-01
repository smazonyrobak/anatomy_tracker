"""Independent read-only verifier for a frozen multiresolution raw bundle."""

import hashlib
import json
import os
import stat
import subprocess
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import nrrd
import numpy as np

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_subject_deformation_v2 as deformation
import training.arbitrary_plane_subject_slab_v2 as subject_slab
import training.arbitrary_plane_synthetic_generator_v2 as slab
import training.slab_refinement_gate_status_v2 as gate_status
import training.subject_deformed_slab_multiresolution_assessment_v2 as assessment
from training.arbitrary_plane_support import build_annotation_support_index


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ENVIRONMENT = "ANATOMY_TRACKER_SUBJECT_SLAB_MULTIRESOLUTION_OUTPUT"
ATLAS_FOLDER = Path(
    os.environ.get(
        "ANATOMY_TRACKER_ATLAS_FOLDER", str(ROOT / "data" / "Allen Brain Atlas 25um")
    )
).resolve()
BUNDLE_SCHEMA = "anatomy-tracker.subject-slab-fixed-case-multiresolution-bundle/v2"
TEMPLATE_SHA256 = "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
ANNOTATION_SHA256 = "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
PYNRRD_VERSION = "1.1.3"
SOURCE_SHA256_CANONICALIZATION = "CRLF and CR normalized to LF before SHA-256"
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
_LEVELS = tuple(f"{step:g}" for step in assessment.AXIAL_STEPS_UM_MAX)
EXPECTED_COMMIT_ENVIRONMENT = "ANATOMY_TRACKER_EXPECTED_SOURCE_COMMIT"
EXPECTED_BRANCH = "codex/arbitrary-plane-joint-model"
UPSTREAM_REF = f"origin/{EXPECTED_BRANCH}"


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_state_v2() -> dict[str, object]:
    expected_commit = os.environ.get(EXPECTED_COMMIT_ENVIRONMENT, "").strip()
    repository_root = Path(_git_output("rev-parse", "--show-toplevel")).resolve()
    head = _git_output("rev-parse", "HEAD")
    branch = _git_output("branch", "--show-current")
    upstream_head = _git_output("rev-parse", UPSTREAM_REF)
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    if (
        not expected_commit
        or repository_root != ROOT
        or head != expected_commit
        or branch != EXPECTED_BRANCH
        or upstream_head != head
        or status
    ):
        raise RuntimeError(
            "multiresolution verification requires the explicit expected pushed "
            "commit and a clean tracked, staged, and untracked worktree"
        )
    return {
        "branch": branch,
        "head": head,
        "expected_commit": expected_commit,
        "upstream_ref": UPSTREAM_REF,
        "upstream_head": upstream_head,
        "clean_tracked_staged_untracked": True,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_receipt(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array)
    dtype = values.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(values.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"dtype": dtype.str, "shape": list(values.shape)}).encode(
            "utf-8"
        )
    )
    digest.update(normalized.tobytes(order="C"))
    return {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "array_sha256": digest.hexdigest(),
    }


def _normalized_text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        name: _normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _duplicate_free_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_canonical_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_free_object,
            parse_constant=_invalid_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path.name}")
    if raw != (_canonical_json(value) + "\n").encode("utf-8"):
        raise ValueError(f"JSON is not canonical: {path.name}")
    return value


def _expected_raw_references() -> dict[str, object]:
    def reference(stem: str) -> dict[str, str]:
        return {
            "metadata": f"{stem}.metadata.json",
            "arrays": f"{stem}.arrays.npz",
        }

    return {
        "subject_deformation_plan": reference("raw/subject-deformation-plan"),
        "precursors": {
            level: reference(f"raw/precursors/{level}") for level in _LEVELS
        },
        "renders": {
            arm: {
                level: reference(f"raw/renders/{arm}/{level}")
                for level in _LEVELS
            }
            for arm in assessment.ARM_NAMES
        },
    }


def _expected_files() -> set[str]:
    files = {
        "bundle-manifest.json",
        "legacy-failed-report.json",
        "plan.json",
        "assessment.json",
    }
    raw = _expected_raw_references()
    references = [raw["subject_deformation_plan"]]
    references.extend(raw["precursors"].values())
    for levels in raw["renders"].values():
        references.extend(levels.values())
    for reference in references:
        files.update(reference.values())
    return files


def _expected_directories() -> set[str]:
    return {
        ".",
        "raw",
        "raw/precursors",
        "raw/renders",
        *(f"raw/renders/{arm}" for arm in assessment.ARM_NAMES),
    }


def _resolve_frozen_member(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("frozen member path is not a nonempty string")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or Path(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in relative
    ):
        raise ValueError("frozen member path is absolute or traverses the root")
    path = (root / Path(*pure.parts)).resolve()
    if path == root or root not in path.parents:
        raise ValueError("frozen member resolves outside the frozen root")
    return path


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _verify_exact_tree(root: Path) -> None:
    actual_files = set()
    actual_directories = {"."}

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = (prefix / entry.name).as_posix()
                if _is_link_or_reparse(path):
                    raise ValueError("frozen bundle contains a link or reparse point")
                if entry.is_file(follow_symlinks=False):
                    actual_files.add(relative)
                elif entry.is_dir(follow_symlinks=False):
                    actual_directories.add(relative)
                    visit(path, prefix / entry.name)
                else:
                    raise ValueError(
                        "frozen bundle contains a non-file/non-directory member"
                    )

    visit(root, PurePosixPath())
    if actual_files != _expected_files() or actual_directories != _expected_directories():
        raise ValueError("frozen bundle tree has missing or extra members")


def _inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "byte_count": _resolve_frozen_member(root, relative).stat().st_size,
            "sha256": _file_sha256(_resolve_frozen_member(root, relative)),
        }
        for relative in sorted(_expected_files() - {"bundle-manifest.json"})
    ]


def _collect_array_placeholders(value: object, names: list[str]) -> None:
    if isinstance(value, dict):
        if "__ndarray__" in value or "__tuple__" in value:
            if set(value) == {"__ndarray__"}:
                name = value["__ndarray__"]
                if not isinstance(name, str) or name in names:
                    raise ValueError("raw metadata has an invalid or duplicate array placeholder")
                names.append(name)
                return
            if set(value) == {"__tuple__"} and isinstance(value["__tuple__"], list):
                for item in value["__tuple__"]:
                    _collect_array_placeholders(item, names)
                return
            raise ValueError("raw metadata has a malformed reserved placeholder")
        for item in value.values():
            _collect_array_placeholders(item, names)
    elif isinstance(value, list):
        for item in value:
            _collect_array_placeholders(item, names)


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


def _read_raw_artifact(root: Path, reference: Mapping[str, str]) -> dict[str, object]:
    if set(reference) != {"metadata", "arrays"}:
        raise ValueError("raw artifact reference has missing or extra fields")
    metadata_path = _resolve_frozen_member(root, reference["metadata"])
    arrays_path = _resolve_frozen_member(root, reference["arrays"])
    metadata = _read_canonical_json(metadata_path)
    placeholders = []
    _collect_array_placeholders(metadata, placeholders)
    expected_names = [f"array_{index:04d}" for index in range(len(placeholders))]
    if set(placeholders) != set(expected_names):
        raise ValueError("raw metadata array placeholders are not exact and contiguous")
    with zipfile.ZipFile(arrays_path, "r") as archive:
        members = archive.infolist()
        expected_members = [f"{name}.npy" for name in expected_names]
        if (
            [member.filename for member in members] != expected_members
            or len({member.filename for member in members}) != len(members)
            or any(member.is_dir() or member.compress_type != zipfile.ZIP_STORED for member in members)
        ):
            raise ValueError("raw NPZ has missing, extra, duplicate, reordered, or compressed members")
    with np.load(arrays_path, allow_pickle=False) as archive:
        if archive.files != expected_names:
            raise ValueError("raw NPZ member names do not match metadata placeholders")
        arrays = {}
        for name in expected_names:
            array = np.asarray(archive[name])
            if array.dtype.hasobject or not array.flags.c_contiguous:
                raise ValueError("raw NPZ array is object-valued or non-contiguous")
            arrays[name] = np.array(array, copy=True, order="C")
    restored = _restore_arrays(metadata, arrays)
    if not isinstance(restored, dict):
        raise ValueError("raw artifact root is not a mapping")
    return restored


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


def _verify_persisted_array_receipts(rendered: Mapping[str, object]) -> None:
    for level in _LEVELS:
        precursor = rendered["precursors"][level]
        precursor_arrays = slab._slab_arrays(precursor["raster"])
        if precursor["raster"]["array_receipts"] != {
            name: _array_receipt(array)
            for name, array in precursor_arrays.items()
        }:
            raise ValueError("precursor persisted array receipt does not match")
        for arm in assessment.ARM_NAMES:
            artifact = rendered["renders"][arm][level]
            coordinate = artifact["coordinate_map"]
            samples = artifact["sample_arrays"]
            reduced = subject_slab._reduced_arrays(artifact["raster"])
            if (
                coordinate["array_receipts"]
                != {
                    name: _array_receipt(array)
                    for name, array in coordinate["arrays"].items()
                }
                or artifact["sample_array_receipts"]
                != {
                    name: _array_receipt(array)
                    for name, array in samples.items()
                }
                or artifact["raster_array_receipts"]
                != {
                    name: _array_receipt(array)
                    for name, array in reduced.items()
                }
            ):
                raise ValueError("subject render persisted array receipt does not match")


def verify_frozen_bundle_independently_v2(
    root: str | Path,
    prepared_context: Mapping[str, object],
    allen_inputs: Mapping[str, object],
    *,
    batch_size: int | None = None,
) -> dict[str, object]:
    """Verify frozen bytes and perform one essential scientific replay."""
    requested_root = Path(root).absolute()
    if requested_root.is_symlink() or (
        requested_root.exists() and _is_link_or_reparse(requested_root)
    ):
        raise ValueError("frozen bundle root is a link or reparse point")
    if not requested_root.is_dir():
        raise FileNotFoundError("frozen multiresolution output directory does not exist")
    root = requested_root.resolve()
    repository = repository_state_v2()
    _verify_exact_tree(root)
    manifest = _read_canonical_json(root / "bundle-manifest.json")
    expected_raw = _expected_raw_references()
    expected_documents = {
        "failed_report": "legacy-failed-report.json",
        "plan": "plan.json",
        "assessment": "assessment.json",
    }
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "bundle_receipt_sha256"
    }
    if (
        set(manifest)
        != {
            "schema_version",
            "role",
            "qualification_eligible",
            "acceptance_thresholds",
            "repository",
            "allen_inputs",
            "documents",
            "raw_artifacts",
            "file_inventory",
            "implementation_source_sha256",
            "implementation_source_sha256_canonicalization",
            "bundle_receipt_sha256",
        }
        or manifest.get("schema_version") != BUNDLE_SCHEMA
        or manifest.get("role") != "immutable threshold-free fixed-case numerical assessment"
        or manifest.get("qualification_eligible") is not False
        or manifest.get("acceptance_thresholds") is not None
        or manifest.get("repository") != repository
        or manifest.get("allen_inputs") != _json_value(allen_inputs)
        or manifest.get("documents") != expected_documents
        or manifest.get("raw_artifacts") != expected_raw
        or manifest.get("file_inventory") != _inventory(root)
        or manifest.get("implementation_source_sha256") != _source_hashes()
        or manifest.get("implementation_source_sha256_canonicalization")
        != SOURCE_SHA256_CANONICALIZATION
        or manifest.get("bundle_receipt_sha256")
        != _payload_sha256(manifest_payload)
    ):
        raise ValueError("frozen multiresolution manifest or inventory is invalid")

    failed_report = _read_canonical_json(
        _resolve_frozen_member(root, expected_documents["failed_report"])
    )
    plan = _read_canonical_json(_resolve_frozen_member(root, expected_documents["plan"]))
    result = _read_canonical_json(
        _resolve_frozen_member(root, expected_documents["assessment"])
    )
    subject_plan = _read_raw_artifact(root, expected_raw["subject_deformation_plan"])
    rendered = {
        "precursors": {
            level: _read_raw_artifact(root, expected_raw["precursors"][level])
            for level in _LEVELS
        },
        "renders": {
            arm: {
                level: _read_raw_artifact(root, expected_raw["renders"][arm][level])
                for level in _LEVELS
            }
            for arm in assessment.ARM_NAMES
        },
    }
    _verify_persisted_array_receipts(rendered)

    assessment.verify_fixed_case_multiresolution_plan_v2(
        plan, failed_report, prepared_context, batch_size=batch_size
    )
    support = acquisition._context_support(prepared_context)
    lower = np.asarray(support["origin_um"], dtype=np.float64)
    upper = lower + np.asarray(support["annotation_shape"], dtype=np.float64) * np.asarray(
        support["voxel_size_um"], dtype=np.float64
    )
    subject_to_ccf_mapper = deformation._verified_subject_to_ccf_mapper_v2(
        subject_plan,
        expected_ccf_context_sha256=prepared_context["v2_context_sha256"],
        expected_full_ccf_lower_um=lower,
        expected_full_ccf_upper_um=upper,
    )
    if not assessment._subject_plan_matches(plan, subject_plan):
        raise ValueError("frozen nonidentity subject plan does not match the selected animal")
    for level in _LEVELS:
        precursor = rendered["precursors"][level]
        slab.verify_v2_generic_global_reference_slab_render(
            precursor, prepared_context
        )
        for arm, arm_plan in (
            ("same_nonidentity_subject_deformation", subject_plan),
            ("identity_control", None),
        ):
            arm_mapper = subject_to_ccf_mapper if arm_plan is subject_plan else None
            subject_slab._verify_subject_slab_render_with_mapper_v2(
                rendered["renders"][arm][level],
                prepared_context,
                precursor,
                subject_plan=arm_plan,
                batch_size=batch_size,
                subject_to_ccf_mapper=arm_mapper,
            )
    assessment.verify_fixed_case_multiresolution_assessment_v2(
        result, plan, rendered
    )
    rejected = gate_status.legacy_gate_contract()
    if (
        rejected.get("decision") != "reject_legacy_universal_gate"
        or rejected.get("qualification_eligible") is not False
        or result.get("acceptance_thresholds") is not None
        or result.get("scientific_decision")
        != "not_evaluated_pending_threshold_predeclaration"
    ):
        raise ValueError("frozen assessment or scientific gate status is invalid")
    return {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_receipt_sha256": manifest["bundle_receipt_sha256"],
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "assessment_receipt_sha256": result["assessment_receipt_sha256"],
        "raw_render_count": 8,
        "qualification_eligible": False,
        "acceptance_thresholds": None,
        "legacy_gate_decision": rejected["decision"],
    }


def main() -> None:
    value = os.environ.get(OUTPUT_ENVIRONMENT)
    if not value or not Path(value).is_absolute():
        raise ValueError(f"{OUTPUT_ENVIRONMENT} must name the frozen absolute directory")
    requested_output = Path(value).absolute()
    if requested_output.is_symlink() or (
        requested_output.exists() and _is_link_or_reparse(requested_output)
    ):
        raise ValueError("frozen multiresolution output root is a link or reparse point")
    output = requested_output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("frozen multiresolution output must be outside the repository")
    if not output.is_dir():
        raise FileNotFoundError("frozen multiresolution output directory does not exist")
    context, allen_inputs = load_pinned_allen_context(ATLAS_FOLDER)
    verified = verify_frozen_bundle_independently_v2(
        output, context, allen_inputs
    )
    print(
        json.dumps(
            {"event": "subject-slab-fixed-case-multiresolution-verified", **verified},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
