"""Generate the fixed substantive finite-S9 data caches for fresh v6 training."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import uuid
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

import training.arbitrary_plane_finite_composite_v4 as finite_composite_v4
import training.arbitrary_plane_finite_joint_curriculum_v5 as finite_joint_v5
import training.arbitrary_plane_finite_pose_curriculum_v4 as finite_pose_v4
import training.arbitrary_plane_finite_row_binding_v6 as finite_row_binding_v6
from training.arbitrary_plane_rendered_generator import prepare_finite_render_context
from training.arbitrary_plane_support import (
    build_annotation_support_index,
    verify_annotation_support_index,
)


SUBSTANTIVE_FINITE_DATA_CONFIG_V6_SCHEMA = (
    "anatomy-tracker.substantive-finite-data-config/v6"
)
SUBSTANTIVE_FINITE_DATA_SUMMARY_V6_SCHEMA = (
    "anatomy-tracker.substantive-finite-data-summary/v6"
)
ANATOMY_ROOT = Path(r"I:\AnatomyTracker")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ANATOMY_ROOT / "runs" / "arbitrary_plane_finite_v6_substantive_data_001"
TEMP_ROOT = ANATOMY_ROOT / "tmp" / "arbitrary_plane_finite_v6_substantive_data_001"
TRAINING_CACHE = OUTPUT_ROOT / "training_cache"
INTERNAL_DEVELOPMENT_CACHE = OUTPUT_ROOT / "internal_development_cache"
SUMMARY_PATH = OUTPUT_ROOT / "substantive_finite_data_summary_v6.json"
TEMPLATE_PATH = ANATOMY_ROOT / "data" / "Allen Brain Atlas 25um" / "average_template_25.nrrd"
ANNOTATION_PATH = ANATOMY_ROOT / "data" / "Allen Brain Atlas 25um" / "annotation_25.nrrd"
TEMPLATE_SHA256 = "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
ANNOTATION_SHA256 = "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
SUPPORT_INDEX_SHA256 = "f0e89d9e2abdacdbe3eeffb55c3bcda077d38ce3419004d2fbf88aa239f4d4cc"
PREPARED_CONTEXT_SHA256 = "1e1d3db8a0fe8b532633fc4c4a23b2e13b0dbf855447982ce27915b09f51c68d"
FINITE_PSF_CAPABILITY_SHA256 = (
    "bcd6441a685e902fb5b59e85bb7003ef3261207d906a0b9390d4a219c3ae3d3e"
)
FIVE_IDS = (
    "animal_id",
    "specimen_id",
    "experiment_id",
    "synthetic_animal_id",
    "section_id",
)
TRAINABLE_MODES = (
    "smart-brush-accurate",
    "smart-brush-imperfect",
    "smart-brush-absent",
)
MODE_SEMANTICS = {
    "smart-brush-accurate": "exact-black smart-brush exterior",
    "smart-brush-imperfect": "imperfect-mask input",
    "smart-brush-absent": "raw acquired background without an outline",
}
PARTITIONS = {
    "training": {
        "data_role": "training",
        "split": "train",
        "cache_directory": str(TRAINING_CACHE),
        "generation_run_id": "finite-v6-substantive-train-001",
        "identity_prefix": "finite-v6-substantive-train",
        "pose_root_seed_uint64": "0x2026090300002001",
        "joint_root_seed_uint64": "0x2026090300002002",
        "pose_row_count": 3_072,
        "joint_row_count": 2_048,
        "row_count": 5_120,
        "animal_count": 320,
        "mode_counts": {
            "smart-brush-accurate": 1_707,
            "smart-brush-imperfect": 1_707,
            "smart-brush-absent": 1_706,
        },
    },
    "internal_development": {
        "data_role": "internal-development",
        "split": "development",
        "cache_directory": str(INTERNAL_DEVELOPMENT_CACHE),
        "generation_run_id": "finite-v6-substantive-internal-development-001",
        "identity_prefix": "finite-v6-substantive-internal-development",
        "pose_root_seed_uint64": "0x2026090300002003",
        "joint_root_seed_uint64": "0x2026090300002004",
        "pose_row_count": 384,
        "joint_row_count": 256,
        "row_count": 640,
        "animal_count": 40,
        "mode_counts": {
            "smart-brush-accurate": 214,
            "smart-brush-imperfect": 213,
            "smart-brush-absent": 213,
        },
    },
}


def _plain(value):
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def _canonical_json(value) -> str:
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _with_receipt(payload: Mapping[str, object]) -> dict[str, object]:
    value = _plain(payload)
    return {**value, "receipt_sha256": _hash_json(value)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _i_path(path, *, must_exist: bool = False) -> Path:
    target = Path(path).resolve()
    if os.path.splitdrive(str(target))[0].upper() != "I:":
        raise ValueError("substantive finite v6 data paths must remain on I:")
    if must_exist and not target.exists():
        raise FileNotFoundError(target)
    return target


def substantive_finite_data_configuration_v6() -> dict[str, object]:
    return _with_receipt(
        {
            "schema_version": SUBSTANTIVE_FINITE_DATA_CONFIG_V6_SCHEMA,
            "scientific_scope": (
                "internal synthetic development only; no public benchmark, "
                "external-validation animal, or final-test animal"
            ),
            "output_root": str(OUTPUT_ROOT),
            "temp_root": str(TEMP_ROOT),
            "summary_path": str(SUMMARY_PATH),
            "partitions": copy.deepcopy(PARTITIONS),
            "output_shape_h_w": [96, 96],
            "sections_per_animal": 16,
            "cache_chunk_size": 16,
            "full_replay_chunk_size": 128,
            "render_mode": "finite_boxcar",
            "axial_sample_count": 9,
            "nominal_cut_thickness_policy": {
                "kind": "authenticated-per-row-closed-interval",
                "minimum_um": 25.0,
                "maximum_um": 100.0,
            },
            "finite_psf_capability_sha256": FINITE_PSF_CAPABILITY_SHA256,
            "trainable_modes": list(TRAINABLE_MODES),
            "mode_semantics": MODE_SEMANTICS,
            "atlas_sources": {
                "template": {
                    "path": str(TEMPLATE_PATH),
                    "byte_count": 32_998_960,
                    "sha256": TEMPLATE_SHA256,
                },
                "annotation": {
                    "path": str(ANNOTATION_PATH),
                    "byte_count": 4_035_363,
                    "sha256": ANNOTATION_SHA256,
                },
                "decoder": {"distribution": "pynrrd", "version": "1.1.3", "index_order": "F"},
            },
            "support_index_sha256": SUPPORT_INDEX_SHA256,
            "prepared_context_sha256": PREPARED_CONTEXT_SHA256,
            "initialization": "data-generation-only-no-model-construction",
            "prior_model_weight_dependencies": [],
            "prior_feature_dependencies": [],
            "prior_pseudolabel_dependencies": [],
            "public_benchmark_accessed": False,
            "external_validation_accessed": False,
            "final_test_accessed": False,
        }
    )


def _require_i_environment(config: Mapping[str, object]) -> None:
    for name in ("output_root", "temp_root", "summary_path"):
        _i_path(config[name])
    for partition in config["partitions"].values():
        _i_path(partition["cache_directory"])
    expected_temp = _i_path(config["temp_root"])
    for name in ("TEMP", "TMP"):
        value = os.environ.get(name)
        if not value or Path(value).resolve() != expected_temp:
            raise RuntimeError(f"set {name} exactly to {expected_temp} before launch")


def _clean_repository_source() -> dict[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("substantive finite v6 data generation requires a clean committed worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if len(commit) != 40 or set(commit) - set("0123456789abcdef"):
        raise RuntimeError("repository HEAD is not a canonical Git commit")
    relative = Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    current = Path(__file__).read_bytes()
    if committed != current:
        raise RuntimeError("launcher source bytes differ from the committed HEAD blob")
    return {
        "git_commit": commit,
        "launcher_relative_path": relative,
        "launcher_sha256": hashlib.sha256(current).hexdigest(),
    }


def _prepare_finite_context(config: Mapping[str, object]):
    import nrrd

    if importlib.metadata.version("pynrrd") != "1.1.3":
        raise RuntimeError("pinned pynrrd version differs from 1.1.3")
    template = _i_path(config["atlas_sources"]["template"]["path"], must_exist=True)
    annotation = _i_path(config["atlas_sources"]["annotation"]["path"], must_exist=True)
    for path, record in (
        (template, config["atlas_sources"]["template"]),
        (annotation, config["atlas_sources"]["annotation"]),
    ):
        if (
            not path.is_file()
            or path.stat().st_size != record["byte_count"]
            or _file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError("pinned Allen raw source bytes differ")
    template_array = nrrd.read(str(template), index_order="F")[0]
    annotation_array = nrrd.read(str(annotation), index_order="F")[0]
    support_mask = annotation_array != 0
    if (
        template_array.shape != (528, 320, 456)
        or annotation_array.shape != template_array.shape
        or template_array.dtype != np.uint16
        or annotation_array.dtype != np.uint32
        or tuple(float(value) for value in np.quantile(template_array[support_mask], (0.01, 0.99)))
        != (9.0, 273.0)
    ):
        raise RuntimeError("decoded Allen arrays or in-support quantiles differ")
    support = build_annotation_support_index(
        annotation_array,
        atlas_id="Allen CCFv3",
        atlas_version="2017 25um",
        source_uri=str(annotation),
        source_sha256=ANNOTATION_SHA256,
        source_entity_type="atlas-annotation",
        voxel_size_um=(25.0, 25.0, 25.0),
        origin_um=(0.0, 0.0, 0.0),
        coordinate_axes=("AP", "DV", "ML"),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    verify_annotation_support_index(support)
    if support["support_index_sha256"] != config["support_index_sha256"]:
        raise RuntimeError("Allen support index differs from the frozen v6 support")
    prepared = prepare_finite_render_context(
        template_array,
        annotation_array,
        support,
        scalar_source_uri=str(template),
        scalar_source_sha256=TEMPLATE_SHA256,
        scalar_source_entity_type="atlas-template",
        scalar_dtype="float32",
        template_decoder="pynrrd==1.1.3 nrrd.read(index_order=F)",
        template_index_order="F",
        annotation_decoder="pynrrd==1.1.3 nrrd.read(index_order=F)",
        annotation_index_order="F",
    )
    if prepared["prepared_context_sha256"] != config["prepared_context_sha256"]:
        raise RuntimeError("prepared Allen finite-render context receipt differs")
    return prepared


def _make_partition_binding(
    prepared_context,
    partition: Mapping[str, object],
    config: Mapping[str, object],
    source_commit: str,
):
    common = {
        "start_index": 0,
        "output_shape_h_w": tuple(config["output_shape_h_w"]),
        "sections_per_animal": config["sections_per_animal"],
        "split": partition["split"],
        "stratum": "reference",
        "margin_um": (0.0, 0.0),
        "finite_parent_generator_source_commit": source_commit,
        "finite_slab_generator_source_commit": source_commit,
    }
    pose = finite_pose_v4.finite_pose_curriculum_generation_config_v4(
        prepared_context,
        root_seed=partition["pose_root_seed_uint64"],
        row_count=partition["pose_row_count"],
        identity_prefix=f"{partition['identity_prefix']}-pose",
        **common,
    )
    joint = finite_joint_v5.finite_joint_curriculum_generation_config_v5(
        prepared_context,
        root_seed=partition["joint_root_seed_uint64"],
        row_count=partition["joint_row_count"],
        identity_prefix=f"{partition['identity_prefix']}-joint",
        amplitude_band_cycle=("mild", "moderate"),
        render_mode="finite_boxcar",
        nominal_cut_thickness_um=None,
        **common,
    )
    composite = finite_composite_v4.make_finite_composite_generation_config_v4(
        pose, joint
    )
    if (
        composite["row_count"] != partition["row_count"]
        or composite["output_shape_h_w"] != config["output_shape_h_w"]
        or composite["finite_psf_run_contract"]["axial_sample_count"] != 9
    ):
        raise RuntimeError("substantive composite generation declaration changed")
    return finite_composite_v4.make_finite_composite_generator_binding_v4(
        composite,
        generation_run_id=partition["generation_run_id"],
        source_commit=source_commit,
    )


def _full_replay_partition(
    cache_directory: Path,
    partition: Mapping[str, object],
    config: Mapping[str, object],
    manifest_receipt_sha256: str,
):
    manifest = finite_row_binding_v6.load_frozen_row_cache_manifest_v6(
        cache_directory,
        expected_manifest_receipt_sha256=manifest_receipt_sha256,
    )
    row_count = partition["row_count"]
    if (
        manifest["receipt_sha256"] != manifest_receipt_sha256
        or manifest["status"] != "FROZEN"
        or manifest["row_count"] != row_count
        or manifest["generation_lineage"]["split"] != partition["split"]
        or manifest["finite_psf_run_contract"]["render_mode"] != "finite_boxcar"
        or manifest["finite_psf_run_contract"]["axial_sample_count"] != 9
        or manifest["finite_psf_capability_sha256"]
        != config["finite_psf_capability_sha256"]
    ):
        raise RuntimeError("frozen substantive partition manifest changed")
    modes = Counter()
    identities = {name: set() for name in FIVE_IDS}
    sections_per_animal = Counter()
    row_ids = []
    row_receipts = []
    selection_receipts = []
    chunk_size = config["full_replay_chunk_size"]
    for start in range(0, row_count, chunk_size):
        indices = list(range(start, min(start + chunk_size, row_count)))
        payload = finite_row_binding_v6.load_frozen_training_rows_v6(
            cache_directory,
            indices,
            expected_manifest_receipt_sha256=manifest_receipt_sha256,
        )
        if (
            payload["row_indices"] != indices
            or payload["training_data_manifest_receipt_sha256"]
            != manifest_receipt_sha256
            or payload["cache_manifest_receipt_sha256"]
            != manifest_receipt_sha256
            or payload["generator_binding_receipt_sha256"]
            != manifest["generator_binding"]["receipt_sha256"]
            or payload["generation_lineage_sha256"]
            != manifest["generator_binding"]["generation_lineage_sha256"]
            or payload["selection_receipt_sha256"]
            != finite_row_binding_v6.frozen_row_selection_receipt_v6(payload)
            or len(payload["rows"]) != len(indices)
            or payload["training_row_ids"]
            != [row["training_row_id"] for row in payload["rows"]]
            or payload["training_row_receipts_sha256"]
            != [row["receipt_sha256"] for row in payload["rows"]]
        ):
            raise RuntimeError("v6 full-row replay returned a different selection")
        selection_receipts.append(payload["selection_receipt_sha256"])
        for row in payload["rows"]:
            lineage = row["lineage"]
            if (
                lineage.get("split") != partition["split"]
                or any(not isinstance(lineage.get(name), str) or not lineage[name] for name in FIVE_IDS)
                or row["selected_mode"] not in TRAINABLE_MODES
                or row["finite_psf_contract"]["render_mode"] != "finite_boxcar"
                or row["finite_psf_contract"]["axial_sample_count"] != 9
            ):
                raise RuntimeError("replayed substantive row semantics changed")
            modes[row["selected_mode"]] += 1
            for name in FIVE_IDS:
                identities[name].add(lineage[name])
            sections_per_animal[lineage["animal_id"]] += 1
            row_ids.append(row["training_row_id"])
            row_receipts.append(row["receipt_sha256"])
    identity_counts = {name: len(values) for name, values in identities.items()}
    expected_identity_counts = {
        "animal_id": partition["animal_count"],
        "specimen_id": partition["animal_count"],
        "experiment_id": partition["animal_count"],
        "synthetic_animal_id": partition["animal_count"],
        "section_id": row_count,
    }
    if (
        dict(modes) != partition["mode_counts"]
        or identity_counts != expected_identity_counts
        or len(row_ids) != row_count
        or len(set(row_ids)) != row_count
        or len(set(row_receipts)) != row_count
        or set(sections_per_animal.values()) != {config["sections_per_animal"]}
    ):
        raise RuntimeError("substantive row count, modes, or exact identities changed")
    return (
        {
            "data_role": partition["data_role"],
            "cache_directory": str(cache_directory),
            "manifest_receipt_sha256": manifest_receipt_sha256,
            "generator_binding_receipt_sha256": manifest["generator_binding"][
                "receipt_sha256"
            ],
            "generation_lineage_sha256": manifest["generator_binding"][
                "generation_lineage_sha256"
            ],
            "row_count": row_count,
            "animal_count": partition["animal_count"],
            "identity_counts": identity_counts,
            "mode_counts": dict(sorted(modes.items())),
            "render_mode": "finite_boxcar",
            "axial_sample_count": 9,
            "full_row_replay": True,
            "full_replay_chunk_size": chunk_size,
            "selection_receipts_sha256": _hash_json(selection_receipts),
            "ordered_training_row_ids_sha256": _hash_json(row_ids),
            "ordered_training_row_receipts_sha256": _hash_json(row_receipts),
        },
        identities,
    )


def _generate_and_replay_partition(
    prepared_context,
    partition: Mapping[str, object],
    config: Mapping[str, object],
    source_commit: str,
):
    cache_directory = _i_path(partition["cache_directory"])
    binding = _make_partition_binding(
        prepared_context, partition, config, source_commit
    )
    manifest, audit = finite_composite_v4.resume_finite_composite_cache_v4(
        cache_directory,
        prepared_context,
        binding,
        chunk_size=config["cache_chunk_size"],
    )
    if (
        manifest["status"] != "FROZEN"
        or manifest["row_count"] != partition["row_count"]
        or manifest["generator_binding"] != binding
        or manifest["generation_lineage"]["source_commit"] != source_commit
        or audit["row_count"] != partition["row_count"]
        or audit["manifest_receipt_sha256"] != manifest["receipt_sha256"]
        or audit["generator_binding_receipt_sha256"]
        != binding["receipt_sha256"]
        or audit["all_rows_authenticated"] is not True
        or audit["learned_dependencies"] != []
        or audit["temporary_file_count"] != 0
        or audit["render_mode"] != "finite_boxcar"
        or audit["axial_sample_count"] != 9
        or any(
            manifest[name] != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise RuntimeError("substantive finite cache generation or freeze audit failed")
    return _full_replay_partition(
        cache_directory,
        partition,
        config,
        manifest["receipt_sha256"],
    )


def _write_or_verify_summary(path: Path, summary: Mapping[str, object]) -> None:
    target = _i_path(path)
    content = (_canonical_json(summary) + "\n").encode("utf-8")
    if target.exists():
        if target.read_bytes() != content:
            raise RuntimeError("existing substantive finite v6 summary differs")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.writing-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def generate_substantive_finite_data_v6() -> dict[str, object]:
    """Resume, freeze, replay, and cross-audit both fixed synthetic partitions."""
    config = substantive_finite_data_configuration_v6()
    source = _clean_repository_source()
    _require_i_environment(config)
    _i_path(config["temp_root"]).mkdir(parents=True, exist_ok=True)
    prepared = _prepare_finite_context(config)
    partition_records = {}
    identity_sets = {}
    for name in ("training", "internal_development"):
        record, identities = _generate_and_replay_partition(
            prepared,
            config["partitions"][name],
            config,
            source["git_commit"],
        )
        partition_records[name] = record
        identity_sets[name] = identities
    overlap = {
        name: sorted(identity_sets["training"][name] & identity_sets["internal_development"][name])
        for name in FIVE_IDS
    }
    if any(overlap.values()):
        raise RuntimeError("training and internal-development exact IDs overlap")
    if _clean_repository_source() != source:
        raise RuntimeError("repository source changed during substantive data generation")
    summary = _with_receipt(
        {
            "schema_version": SUBSTANTIVE_FINITE_DATA_SUMMARY_V6_SCHEMA,
            "configuration": config,
            "source": source,
            "atlas_binding": {
                "template_raw_sha256": TEMPLATE_SHA256,
                "annotation_raw_sha256": ANNOTATION_SHA256,
                "support_index_sha256": SUPPORT_INDEX_SHA256,
                "prepared_context_sha256": PREPARED_CONTEXT_SHA256,
            },
            "partitions": partition_records,
            "cross_partition_identity_overlap": overlap,
            "all_partition_ids_disjoint": True,
            "all_rows_replayed_by_finite_row_binding_v6": True,
            "render_mode": "finite_boxcar",
            "axial_sample_count": 9,
            "input_mode_semantics": MODE_SEMANTICS,
            "initialization": "data-generation-only-no-model-construction",
            "prior_model_weight_dependencies": [],
            "prior_feature_dependencies": [],
            "prior_pseudolabel_dependencies": [],
            "public_benchmark_accessed": False,
            "external_validation_accessed": False,
            "final_test_accessed": False,
            "release_qualifying": False,
        }
    )
    _write_or_verify_summary(Path(config["summary_path"]), summary)
    return summary


def main() -> None:
    summary = generate_substantive_finite_data_v6()
    print(_canonical_json(summary))


if __name__ == "__main__":
    main()


__all__ = [
    "INTERNAL_DEVELOPMENT_CACHE",
    "OUTPUT_ROOT",
    "PARTITIONS",
    "SUBSTANTIVE_FINITE_DATA_CONFIG_V6_SCHEMA",
    "SUBSTANTIVE_FINITE_DATA_SUMMARY_V6_SCHEMA",
    "SUMMARY_PATH",
    "TEMP_ROOT",
    "TRAINING_CACHE",
    "generate_substantive_finite_data_v6",
    "substantive_finite_data_configuration_v6",
]
