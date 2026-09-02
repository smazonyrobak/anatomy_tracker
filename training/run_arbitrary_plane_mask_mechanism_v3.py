"""Frozen-input, model-free runner for the paired smart-brush mechanism replay.

The study is intentionally not executable until a committed freeze receipt is
supplied to :func:`run_mask_mechanism_v3`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from training.arbitrary_plane_finite_candidates import (
    finite_candidate_bank_receipt,
    make_arbitrary_plane_finite_candidate_bank_from_context,
)
from training.arbitrary_plane_image_candidate_scalar import render_candidate_bank_scalars
from training.arbitrary_plane_image_information import (
    common_lattice_map_yx,
    dewarp_target_float32,
    rank_candidate_scores,
    resample_common_lattice_intensity,
    scale_candidate_raster,
    score_mind_candidates,
)
from training.arbitrary_plane_mask_mechanism_v3 import (
    MASK_MECHANISM_SCHEMA,
    MASK_VARIANTS,
    SCIENTIFIC_AMBIGUITY,
    array_receipt,
    paired_smart_brush_inputs,
    payload_sha256,
)
from training.arbitrary_plane_rendered_generator import (
    finite_render_receipt,
    make_finite_arbitrary_plane_render_from_context,
)
from training.arbitrary_plane_synthetic_generator import (
    ACCURATE_OUTLINE,
    IMPERFECT_OUTLINE,
    make_arbitrary_plane_synthetic_realization,
    synthetic_realization_receipt,
)
from training.run_arbitrary_plane_image_information import _validate_contexts
from training.run_arbitrary_plane_semantic_oracle import _array_receipt, load_allen_contexts


ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT = Path(r"I:\AnatomyTracker\runs\arbitrary_plane_image_information_raw_ed13b2a")
RAW_RESULT_FILE_SHA256 = "c9bcc6b6b30358c237f357caf8d84b170b33a226b6be31769d206e77e7273cc3"
RAW_CONFIG_FILE_SHA256 = "fd27fbdf6d1ec1ba901bb4873da27e870d9024a9a35d626d24d74b616f8f028d"
RAW_RESULT_PAYLOAD_SHA256 = "c11b65f2497b15d33857a7990a3728e9608d634d924fb7542ed315433f48d6fc"
RAW_CONFIG_PAYLOAD_SHA256 = "923bbc2eb1c3517988e9c13b33a98329374cff4dc67d02a67aaa3f24624d88db"
RAW_INVENTORY_SHA256 = "c5532bba21aaf1cda23663d6ae40724e5b8c10183588cd79e0167279c63604c9"
RAW_FILE_COUNT = 514
RAW_TOTAL_BYTES = 66_832_147
CASE_COUNT = 64
CANDIDATE_COUNT = 40
SHUFFLED_OFFSET = 17
OUTPUT_SHAPE = (192, 256)
FROZEN_GENERATOR_COMMIT = "27c4ba644c7b75ab6f676d944e917811df961b05"
FREEZE_SCHEMA = "anatomy-tracker.arbitrary-plane-mask-mechanism-freeze/v3"
RESULT_SCHEMA = "anatomy-tracker.arbitrary-plane-mask-mechanism-result/v3"
FREEZE_RECEIPT_RELATIVE_PATH = "publication/arbitrary_plane_mask_mechanism_v3_preflight.yaml"

SOURCE_PATHS = (
    "training/arbitrary_plane_mask_mechanism_v3.py",
    "training/run_arbitrary_plane_mask_mechanism_v3.py",
    "tests/test_arbitrary_plane_mask_mechanism_v3.py",
    "training/arbitrary_plane_synthetic_observation.py",
    "training/arbitrary_plane_synthetic_generator.py",
    "training/arbitrary_plane_rendered_generator.py",
    "training/arbitrary_plane_finite_candidates.py",
    "training/arbitrary_plane_image_candidate_scalar.py",
    "training/arbitrary_plane_image_information.py",
    "training/run_arbitrary_plane_image_information.py",
    "training/run_arbitrary_plane_semantic_oracle.py",
    "publication/arbitrary_plane_image_information_result.yaml",
)
MODEL_INDEPENDENCE = {
    "learned_checkpoint_dependencies": [],
    "previous_model_dependencies": [],
    "pretrained_feature_dependencies": [],
    "legacy_descriptor_dependencies": [],
    "initialization": "deterministic frozen case streams only; no learned initialization",
}
DATA_ACCESS = {
    "allen_template_and_annotation": True,
    "synthetic_development": True,
    "deepslice_ground_truth": False,
    "real_lab_histology": False,
    "calibration_animals": False,
    "qualification_animals": False,
    "final_test_animals": False,
    "full_benchmark": False,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token} in {path}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _check_self_hash(record: Mapping[str, object], field: str) -> None:
    payload = dict(record)
    observed = payload.pop(field, None)
    if observed != payload_sha256(payload):
        raise ValueError(f"{field} does not authenticate its JSON payload")


def _inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, object]]:
    excluded = exclude or set()
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


def verify_inventory_bound_tree(
    root: Path,
    *,
    expected_result_file_sha256: str | None = None,
) -> dict[str, object]:
    """Authenticate every byte in a result tree against its embedded inventory."""
    root = Path(root).resolve()
    result_path = root / "result.json"
    if expected_result_file_sha256 and _file_sha256(result_path) != expected_result_file_sha256:
        raise ValueError("result.json file hash differs from its frozen authority")
    result = _read_json(result_path)
    if "result_payload_sha256" in result:
        _check_self_hash(result, "result_payload_sha256")
    expected = result.get("pre_result_inventory")
    if not isinstance(expected, list) or expected != sorted(expected, key=lambda item: item["path"]):
        raise ValueError("embedded pre-result inventory is absent or not path-sorted")
    if result.get("pre_result_inventory_sha256") != payload_sha256(expected):
        raise ValueError("embedded pre-result inventory hash differs")
    for item in expected:
        if set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("inventory item schema differs")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("inventory path escapes its result tree")
    actual = _inventory(root, exclude={"result.json"})
    if actual != expected:
        raise ValueError("result-tree paths, byte counts, or SHA-256 values differ")
    return result


def authenticate_frozen_image_information(
    raw_root: Path = RAW_INPUT,
) -> tuple[dict[str, object], dict[str, object]]:
    """Authenticate the exact migrated 64-case image-information source tree."""
    raw_root = Path(raw_root).resolve()
    if raw_root != RAW_INPUT.resolve():
        raise ValueError("mask replay accepts only the declared immutable I: raw source")
    result = verify_inventory_bound_tree(
        raw_root, expected_result_file_sha256=RAW_RESULT_FILE_SHA256
    )
    if (
        result.get("result_payload_sha256") != RAW_RESULT_PAYLOAD_SHA256
        or result.get("pre_result_inventory_sha256") != RAW_INVENTORY_SHA256
        or len(result["pre_result_inventory"]) != RAW_FILE_COUNT
        or sum(int(item["size_bytes"]) for item in result["pre_result_inventory"])
        != RAW_TOTAL_BYTES
    ):
        raise ValueError("frozen result identity or inventory totals differ")
    config_path = raw_root / "resolved_config.json"
    if _file_sha256(config_path) != RAW_CONFIG_FILE_SHA256:
        raise ValueError("frozen resolved_config.json file hash differs")
    config = _read_json(config_path)
    _check_self_hash(config, "resolved_config_sha256")
    if (
        config["resolved_config_sha256"] != RAW_CONFIG_PAYLOAD_SHA256
        or result["resolved_config_sha256"] != RAW_CONFIG_PAYLOAD_SHA256
        or config["case_and_shuffle_contract"]["base_count"] != CASE_COUNT
        or config["case_and_shuffle_contract"]["candidate_count"] != CANDIDATE_COUNT
        or config["case_and_shuffle_contract"]["shuffled_offset"] != SHUFFLED_OFFSET
        or config["data_access"] != DATA_ACCESS
        or config["model_independence"] != MODEL_INDEPENDENCE
        or result["data_access"] != DATA_ACCESS
        or result["model_independence"] != MODEL_INDEPENDENCE
    ):
        raise ValueError("frozen configuration, scope, or model independence differs")

    for case_index in range(CASE_COUNT):
        primary = _read_json(raw_root / "primary" / f"case-{case_index:03d}.json")
        shuffled = _read_json(raw_root / "shuffled" / f"case-{case_index:03d}.json")
        control = _read_json(raw_root / "controls" / f"case-{case_index:03d}.json")
        for record in (primary, shuffled, control):
            _check_self_hash(record, "payload_sha256")
        if (
            primary["case_index"] != case_index
            or primary["payload_sha256"] != result["primary_case_payload_sha256"][case_index]
            or shuffled["payload_sha256"] != result["shuffled_case_payload_sha256"][case_index]
            or control["payload_sha256"] != result["control_payload_sha256"][case_index]
            or len(primary["candidate_bank"]["ordered_candidate_ids"]) != CANDIDATE_COUNT
            or primary["frozen_replay"]["replay_passed"] is not True
            or shuffled["bank_case_index"] != case_index
            or shuffled["target_case_index"] != (case_index + SHUFFLED_OFFSET) % CASE_COUNT
            or any(primary["provenance"].get(key) is not None for key in ("animal_id", "specimen_id", "experiment_id"))
        ):
            raise ValueError(f"frozen case {case_index} identity or development scope differs")
    return result, config


def source_sha256() -> dict[str, str]:
    return {relative: _file_sha256(ROOT / relative) for relative in SOURCE_PATHS}


def verify_freeze_receipt(
    receipt: Mapping[str, object],
    *,
    check_repository: bool = True,
    expected_sources: Mapping[str, str] | None = None,
) -> None:
    """Require a pre-execution, source-bound design freeze."""
    required = {
        "schema",
        "status",
        "freeze_commit",
        "source_sha256",
        "raw_input",
        "case_count",
        "candidate_count",
        "mask_variants",
        "descriptor",
        "domain",
        "shuffled_offset",
        "model_independence",
        "data_access",
        "scientific_ambiguity",
    }
    sources = dict(expected_sources or source_sha256())
    if (
        set(receipt) != required
        or receipt["schema"] != FREEZE_SCHEMA
        or receipt["status"] != "frozen-before-execution"
        or re.fullmatch(r"[0-9a-f]{40}", str(receipt["freeze_commit"])) is None
        or receipt["source_sha256"] != sources
        or receipt["raw_input"]
        != {
            "directory": RAW_INPUT.as_posix(),
            "result_file_sha256": RAW_RESULT_FILE_SHA256,
            "result_payload_sha256": RAW_RESULT_PAYLOAD_SHA256,
            "inventory_sha256": RAW_INVENTORY_SHA256,
        }
        or receipt["case_count"] != CASE_COUNT
        or receipt["candidate_count"] != CANDIDATE_COUNT
        or receipt["mask_variants"] != list(MASK_VARIANTS)
        or receipt["descriptor"] != "MIND"
        or receipt["domain"] != "context"
        or receipt["shuffled_offset"] != SHUFFLED_OFFSET
        or receipt["model_independence"] != MODEL_INDEPENDENCE
        or receipt["data_access"] != DATA_ACCESS
        or receipt["scientific_ambiguity"] != SCIENTIFIC_AMBIGUITY
    ):
        raise ValueError("mask-mechanism freeze receipt is absent, incomplete, or changed")
    if check_repository:
        freeze_commit = str(receipt["freeze_commit"])
        receipt_path = ROOT / FREEZE_RECEIPT_RELATIVE_PATH
        if not receipt_path.is_file() or _read_json(receipt_path) != dict(receipt):
            raise ValueError("the supplied freeze receipt is not the committed publication receipt")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", FREEZE_RECEIPT_RELATIVE_PATH],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", freeze_commit, "HEAD"],
            cwd=ROOT,
            check=False,
        )
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                *SOURCE_PATHS,
                FREEZE_RECEIPT_RELATIVE_PATH,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if tracked.returncode != 0 or ancestor.returncode != 0 or dirty:
            raise ValueError(
                "freeze receipt is uncommitted, freeze commit is not an ancestor, "
                "or frozen sources are dirty"
            )


def _guard_output(output: Path) -> Path:
    output = Path(output).resolve()
    if output.drive.upper() != "I:":
        raise ValueError("mask-mechanism outputs are restricted to the I: drive")
    if output == RAW_INPUT.resolve() or RAW_INPUT.resolve() in output.parents:
        raise ValueError("output cannot overlap the immutable raw source")
    if output.exists():
        raise FileExistsError("mask-mechanism output must be a fresh directory")
    return output


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _raw_primary(case_index: int) -> dict[str, object]:
    return _read_json(RAW_INPUT / "primary" / f"case-{case_index:03d}.json")


def _raw_shuffled(case_index: int) -> dict[str, object]:
    return _read_json(RAW_INPUT / "shuffled" / f"case-{case_index:03d}.json")


def _outline(raw_case: Mapping[str, object], mode: str) -> dict[str, object]:
    return next(item for item in raw_case["outline_results"] if item["outline_mode"] == mode)


def _mind_context_scores(raw_case: Mapping[str, object], mode: str) -> list[float]:
    outline = _outline(raw_case, mode)
    return next(
        slot["scores"]
        for slot in outline["score_slots"]
        if slot["descriptor"] == "MIND" and slot["domain"] == "context"
    )


def _replay_parent_bank(
    raw_case: Mapping[str, object],
    render_context: Mapping[str, object],
    candidate_context: Mapping[str, object],
    support_index: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    expected_parent = raw_case["frozen_replay"]["finite_parent_receipt"]
    config = expected_parent["generator"]["resolved_config"]
    parent = make_finite_arbitrary_plane_render_from_context(
        render_context,
        config["split"],
        int(config["root_seed"], 16),
        tuple(config["output_shape_h_w"]),
        sample_index=int(config["sample_index"]),
        stratum=config["stratum"],
        boundary_stress_fraction=float(config["boundary_stress_fraction"]),
        margin_um=tuple(config["margin_u_v_um"]),
        animal_id=config["animal_id"],
        specimen_id=config["specimen_id"],
        experiment_id=config["experiment_id"],
        max_rejection_attempts=int(config["max_rejection_attempts"]),
        minimum_brain_pixels=int(config["minimum_brain_pixels"]),
        generator_source_commit=FROZEN_GENERATOR_COMMIT,
    )
    if finite_render_receipt(parent) != expected_parent:
        raise ValueError(f"case {raw_case['case_index']} parent did not replay exactly")
    bank = make_arbitrary_plane_finite_candidate_bank_from_context(
        parent,
        candidate_context,
        support_index,
        finite_parent_generator_source_commit=FROZEN_GENERATOR_COMMIT,
    )
    expected_bank = raw_case["candidate_bank"]
    if (
        finite_candidate_bank_receipt(bank) != expected_bank["receipt"]
        or bank["finite_candidate_bank_id"] != expected_bank["finite_candidate_bank_id"]
        or list(bank["ordered_candidate_ids"]) != expected_bank["ordered_candidate_ids"]
    ):
        raise ValueError(f"case {raw_case['case_index']} candidate bank did not replay exactly")
    return parent, bank


def _replay_descendant(
    parent: dict[str, object],
    support_index: dict[str, object],
    raw_case: Mapping[str, object],
    mode: str,
) -> dict[str, object]:
    expected = _outline(raw_case, mode)["synthetic_receipt"]
    config = expected["generator"]["resolved_config"]
    descendant = make_arbitrary_plane_synthetic_realization(
        parent,
        support_index,
        root_seed=config["root_seed"],
        sample_index=int(config["sample_index"]),
        synthetic_stratum=config["synthetic_stratum"],
        outline_mode=mode,
        finite_parent_generator_source_commit=FROZEN_GENERATOR_COMMIT,
    )
    if synthetic_realization_receipt(descendant) != expected:
        raise ValueError(f"case {raw_case['case_index']} {mode} did not replay exactly")
    return descendant


def _unpack_context_mask(raw_case: Mapping[str, object]) -> np.ndarray:
    receipt = next(
        item["mask_receipt"]
        for item in raw_case["score_domains"]
        if item["domain"] == "context"
    )
    packed = (RAW_INPUT / receipt["relative_path"]).read_bytes()
    if hashlib.sha256(packed).hexdigest() != receipt["payload_sha256"]:
        raise ValueError("packed context mask hash differs")
    mask = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8), bitorder=receipt["bitorder"]
    )[: int(receipt["bit_count"])].reshape(receipt["shape"])
    mask = np.ascontiguousarray(mask, dtype=bool)
    if _array_receipt(mask)["array_sha256"] != receipt["array_sha256"]:
        raise ValueError("unpacked context mask receipt differs")
    return mask


def _target_runtime(
    raw_case: Mapping[str, object],
    render_context: Mapping[str, object],
    candidate_context: Mapping[str, object],
    support_index: dict[str, object],
) -> dict[str, object]:
    parent, _ = _replay_parent_bank(
        raw_case, render_context, candidate_context, support_index
    )
    accurate = _replay_descendant(parent, support_index, raw_case, ACCURATE_OUTLINE)
    imperfect = _replay_descendant(parent, support_index, raw_case, IMPERFECT_OUTLINE)
    arrays = accurate["arrays"]
    imperfect_parameters = imperfect["outline"]["parameters"]
    accepted = int(imperfect_parameters["accepted_attempt_index"])
    seed = imperfect_parameters["rejection_attempts"][accepted]["field_stream_seed_uint64"]
    shape = arrays["damaged_acquired_image"].shape
    variants = paired_smart_brush_inputs(
        arrays["damaged_acquired_image"],
        arrays["observable_footprint_mask"],
        seed,
        morphology_radius_px=min(5, max(1, int(round(0.02 * min(shape))))),
        jitter_amplitude_px=max(0.25, 0.01 * min(shape)),
    )
    frozen_accurate = _outline(raw_case, ACCURATE_OUTLINE)["synthetic_receipt"]["array_receipts"]
    frozen_imperfect = _outline(raw_case, IMPERFECT_OUTLINE)["synthetic_receipt"]["array_receipts"]
    if (
        variants["accurate"]["mask_receipt"] != frozen_accurate["input_outline_mask"]
        or variants["accurate"]["model_input_image_receipt"] != frozen_accurate["model_input_image"]
        or variants["full-imperfect"]["mask_receipt"] != frozen_imperfect["input_outline_mask"]
        or variants["full-imperfect"]["model_input_image_receipt"] != frozen_imperfect["model_input_image"]
        or variants["full-imperfect"]["parameters"]["sampled_morphology_px"]
        != imperfect_parameters["parameters"]["morphology_px"]
    ):
        raise ValueError("paired accurate/full masks do not match frozen descendants bit-for-bit")
    targets = {}
    for name in MASK_VARIANTS:
        dewarped = dewarp_target_float32(
            variants[name]["model_input_image"], arrays["fixed_to_source_map"]
        )
        targets[name] = np.array(dewarped, dtype=np.float64, copy=True, order="C")
    return {
        "case_index": int(raw_case["case_index"]),
        "pixel_pitch_um": float(raw_case["target"]["pixel_pitch_um"]),
        "context_mask": _unpack_context_mask(raw_case),
        "targets": targets,
        "variant_receipts": {
            name: {
                key: variants[name][key]
                for key in (
                    "mask_receipt",
                    "model_input_image_receipt",
                    "quality_iou",
                    "parameters",
                    "black_exterior_exact",
                )
            }
            for name in MASK_VARIANTS
        },
        "source_binding": {
            "primary_file_sha256": _file_sha256(
                RAW_INPUT / "primary" / f"case-{int(raw_case['case_index']):03d}.json"
            ),
            "primary_payload_sha256": raw_case["payload_sha256"],
            "finite_parent_receipt_sha256": raw_case["frozen_replay"]["finite_parent_receipt_sha256"],
            "candidate_bank_receipt_sha256": raw_case["frozen_replay"]["candidate_bank_receipt_sha256"],
            "paired_view_group_id": raw_case["target"]["paired_view_group_id"],
            "accepted_imperfect_perturbation_seed_uint64": seed,
        },
    }


def _candidate_runtime(
    raw_case: Mapping[str, object],
    render_context: Mapping[str, object],
    candidate_context: Mapping[str, object],
    support_index: dict[str, object],
) -> tuple[np.ndarray, list[str], str]:
    parent, bank = _replay_parent_bank(
        raw_case, render_context, candidate_context, support_index
    )
    rendered = render_candidate_bank_scalars(render_context, bank, parent)
    raw = np.asarray(rendered["scalar_float32"])
    scaled = np.ascontiguousarray(
        np.stack([scale_candidate_raster(item) for item in raw]), dtype=np.float64
    )
    ids = list(rendered["candidate_ids"])
    expected = raw_case["candidate_scalar_receipts"]
    if (
        scaled.shape != (CANDIDATE_COUNT, *OUTPUT_SHAPE)
        or ids != raw_case["candidate_bank"]["ordered_candidate_ids"]
        or any(
            _array_receipt(raw[index]) != expected[index]["scalar"]["rendered_float32"]
            or _array_receipt(scaled[index]) != expected[index]["scalar"]["scaled_float64"]
            for index in range(CANDIDATE_COUNT)
        )
    ):
        raise ValueError(f"case {raw_case['case_index']} scalar candidate bank differs")
    return scaled, ids, str(raw_case["candidate_bank"]["truth_candidate_id"])


def _score(
    target: np.ndarray,
    candidates: np.ndarray,
    context_mask: np.ndarray,
    pixel_pitch_um: float,
    candidate_ids: list[str],
    truth_candidate_id: str,
) -> dict[str, object]:
    result = score_mind_candidates(
        target,
        candidates,
        context_mask,
        pixel_pitch_um,
        padding_value=0.0,
        chunk_size=8,
    )
    scores = np.asarray(result["scores"], dtype=np.float64)
    return {
        "scores": scores.tolist(),
        "scores_sha256": payload_sha256({"scores": scores.tolist()}),
        "ranking": rank_candidate_scores(scores, candidate_ids, truth_candidate_id),
        "target_vbar": float(result["target_vbar"]),
        "candidate_vbar": np.asarray(result["candidate_vbar"], dtype=np.float64).tolist(),
    }


def _finalize_record(record: dict[str, object]) -> dict[str, object]:
    record["payload_sha256"] = payload_sha256(record)
    return record


def aggregate_metrics(
    primary: list[Mapping[str, object]], shuffled: list[Mapping[str, object]]
) -> dict[str, object]:
    """Aggregate only paired finite-bank development outcomes; no gate or claim."""
    if len(primary) != CASE_COUNT or len(shuffled) != CASE_COUNT:
        raise ValueError("mechanism summaries require exactly 64 paired cases")

    def summarize(records: list[Mapping[str, object]]) -> dict[str, object]:
        summaries = {}
        for variant in MASK_VARIANTS:
            rankings = [record["variants"][variant]["ranking"] for record in records]
            summaries[variant] = {
                "top1_count": sum(bool(item["top1"]) for item in rankings),
                "top1_rate": sum(bool(item["top1"]) for item in rankings) / CASE_COUNT,
                "mean_reciprocal_rank": float(
                    np.mean([float(item["reciprocal_rank"]) for item in rankings])
                ),
            }
        accurate = [record["variants"]["accurate"]["ranking"]["top1"] for record in records]
        for variant in MASK_VARIANTS[1:]:
            current = [record["variants"][variant]["ranking"]["top1"] for record in records]
            summaries[variant]["paired_top1_vs_accurate"] = {
                "accurate_only": sum(bool(a and not b) for a, b in zip(accurate, current, strict=True)),
                "variant_only": sum(bool(b and not a) for a, b in zip(accurate, current, strict=True)),
                "both": sum(bool(a and b) for a, b in zip(accurate, current, strict=True)),
                "neither": sum(bool(not a and not b) for a, b in zip(accurate, current, strict=True)),
            }
        return summaries

    return {"native": summarize(primary), "shuffled": summarize(shuffled)}


def run_mask_mechanism_v3(
    output: Path,
    freeze_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Run the study only after its separate committed design freeze exists."""
    output = _guard_output(output)
    verify_freeze_receipt(freeze_receipt)
    frozen_result, frozen_config = authenticate_frozen_image_information()
    support, render_context, candidate_context = load_allen_contexts()
    _validate_contexts(support, render_context, candidate_context)
    raw_cases = [_raw_primary(index) for index in range(CASE_COUNT)]
    targets = [
        _target_runtime(case, render_context, candidate_context, support)
        for case in raw_cases
    ]

    output.mkdir(parents=True, exist_ok=False)
    (output / "primary").mkdir()
    (output / "shuffled").mkdir()
    source_hashes = source_sha256()
    config = _finalize_record(
        {
            "schema": MASK_MECHANISM_SCHEMA,
            "status": "development-only-paired-mask-mechanism-replay",
            "freeze_receipt": dict(freeze_receipt),
            "source_sha256": source_hashes,
            "raw_input": {
                "directory": RAW_INPUT.as_posix(),
                "result_file_sha256": RAW_RESULT_FILE_SHA256,
                "result_payload_sha256": frozen_result["result_payload_sha256"],
                "resolved_config_file_sha256": RAW_CONFIG_FILE_SHA256,
                "resolved_config_payload_sha256": frozen_config["resolved_config_sha256"],
                "inventory_sha256": frozen_result["pre_result_inventory_sha256"],
            },
            "case_count": CASE_COUNT,
            "candidate_count": CANDIDATE_COUNT,
            "mask_variants": list(MASK_VARIANTS),
            "descriptor": "MIND",
            "domain": "context",
            "shuffled_offset": SHUFFLED_OFFSET,
            "model_independence": MODEL_INDEPENDENCE,
            "data_access": DATA_ACCESS,
            "scientific_ambiguity": SCIENTIFIC_AMBIGUITY,
        }
    )
    _write_json(output / "resolved_config.json", config)

    primary_records, shuffled_records = [], []
    for case_index, raw_case in enumerate(raw_cases):
        candidates, candidate_ids, truth_id = _candidate_runtime(
            raw_case, render_context, candidate_context, support
        )
        target = targets[case_index]
        native_variants = {}
        for variant in MASK_VARIANTS:
            score = _score(
                target["targets"][variant],
                candidates,
                target["context_mask"],
                target["pixel_pitch_um"],
                candidate_ids,
                truth_id,
            )
            if variant in {"accurate", "full-imperfect"}:
                old_mode = ACCURATE_OUTLINE if variant == "accurate" else IMPERFECT_OUTLINE
                if score["scores"] != _mind_context_scores(raw_case, old_mode):
                    raise ValueError("native accurate/full score crosscheck differs from frozen source")
                score["frozen_score_crosscheck"] = True
            else:
                score["frozen_score_crosscheck"] = None
            native_variants[variant] = score
        primary = _finalize_record(
            {
                "schema": "anatomy-tracker.arbitrary-plane-mask-mechanism-primary/v3",
                "case_index": case_index,
                "source_binding": target["source_binding"],
                "candidate_bank_id": raw_case["candidate_bank"]["finite_candidate_bank_id"],
                "ordered_candidate_ids_sha256": payload_sha256(candidate_ids),
                "truth_candidate_id": truth_id,
                "context_mask_receipt": _array_receipt(target["context_mask"]),
                "variant_receipts": target["variant_receipts"],
                "variants": native_variants,
            }
        )
        _write_json(output / "primary" / f"case-{case_index:03d}.json", primary)
        primary_records.append(primary)

        target_index = (case_index + SHUFFLED_OFFSET) % CASE_COUNT
        shuffled_target = targets[target_index]
        coordinates = common_lattice_map_yx(
            OUTPUT_SHAPE,
            target["pixel_pitch_um"],
            shuffled_target["pixel_pitch_um"],
        )
        resampled = np.ascontiguousarray(
            np.stack(
                [resample_common_lattice_intensity(item, coordinates) for item in candidates]
            ),
            dtype=np.float64,
        )
        frozen_shuffled = _raw_shuffled(case_index)
        shuffled_variants = {}
        for variant in MASK_VARIANTS:
            score = _score(
                shuffled_target["targets"][variant],
                resampled,
                shuffled_target["context_mask"],
                shuffled_target["pixel_pitch_um"],
                candidate_ids,
                truth_id,
            )
            if variant in {"accurate", "full-imperfect"}:
                old_mode = ACCURATE_OUTLINE if variant == "accurate" else IMPERFECT_OUTLINE
                if score["scores"] != _mind_context_scores(frozen_shuffled, old_mode):
                    raise ValueError("shuffled accurate/full control differs from frozen source")
                score["frozen_score_crosscheck"] = True
            else:
                score["frozen_score_crosscheck"] = None
            shuffled_variants[variant] = score
        shuffled = _finalize_record(
            {
                "schema": "anatomy-tracker.arbitrary-plane-mask-mechanism-shuffled/v3",
                "bank_case_index": case_index,
                "target_case_index": target_index,
                "source_bank_primary_payload_sha256": raw_case["payload_sha256"],
                "source_target_primary_payload_sha256": raw_cases[target_index]["payload_sha256"],
                "source_shuffled_file_sha256": _file_sha256(
                    RAW_INPUT / "shuffled" / f"case-{case_index:03d}.json"
                ),
                "source_shuffled_payload_sha256": frozen_shuffled["payload_sha256"],
                "candidate_bank_id": raw_case["candidate_bank"]["finite_candidate_bank_id"],
                "ordered_candidate_ids_sha256": payload_sha256(candidate_ids),
                "truth_candidate_id": truth_id,
                "common_lattice_coordinate_receipt": _array_receipt(coordinates),
                "variants": shuffled_variants,
            }
        )
        _write_json(output / "shuffled" / f"case-{case_index:03d}.json", shuffled)
        shuffled_records.append(shuffled)

    if source_sha256() != source_hashes:
        raise RuntimeError("source files changed during mask-mechanism replay")
    authenticate_frozen_image_information()
    pre_result_inventory = _inventory(output)
    result = _finalize_record(
        {
            "schema": RESULT_SCHEMA,
            "status": "development-mechanism-replay-complete-no-promotion",
            "resolved_config_sha256": config["payload_sha256"],
            "primary_case_payload_sha256": [item["payload_sha256"] for item in primary_records],
            "shuffled_case_payload_sha256": [item["payload_sha256"] for item in shuffled_records],
            "metrics": aggregate_metrics(primary_records, shuffled_records),
            "pre_result_inventory": pre_result_inventory,
            "pre_result_inventory_sha256": payload_sha256(pre_result_inventory),
            "model_independence": MODEL_INDEPENDENCE,
            "data_access": DATA_ACCESS,
            "scientific_ambiguity": SCIENTIFIC_AMBIGUITY,
            "interpretation": (
                "paired finite-bank synthetic development mechanism diagnostic only; "
                "not a benchmark, qualification, calibration, architecture selection, "
                "learned-model result, or final-test result"
            ),
        }
    )
    _write_json(output / "result.json", result)
    verify_inventory_bound_tree(output)
    return result


if __name__ == "__main__":
    raise SystemExit(
        "Development implementation only: call run_mask_mechanism_v3(output, freeze_receipt) "
        "after the separate source-bound freeze commit."
    )
