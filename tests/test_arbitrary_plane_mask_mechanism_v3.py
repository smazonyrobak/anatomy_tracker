import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from training.arbitrary_plane_mask_mechanism_v3 import (
    MASK_VARIANTS,
    SCIENTIFIC_AMBIGUITY,
    paired_smart_brush_inputs,
    paired_smart_brush_masks,
    payload_sha256,
)
from training.arbitrary_plane_synthetic_observation import _imperfect_outline
from training.run_arbitrary_plane_mask_mechanism_v3 import (
    CANDIDATE_COUNT,
    CASE_COUNT,
    DATA_ACCESS,
    FREEZE_SCHEMA,
    MODEL_INDEPENDENCE,
    RAW_INPUT,
    RAW_INVENTORY_SHA256,
    RAW_RESULT_FILE_SHA256,
    RAW_RESULT_PAYLOAD_SHA256,
    SHUFFLED_OFFSET,
    _guard_output,
    aggregate_metrics,
    authenticate_frozen_image_information,
    run_mask_mechanism_v3,
    verify_freeze_receipt,
    verify_inventory_bound_tree,
)


def _footprint() -> np.ndarray:
    y, x = np.ogrid[:64, :80]
    return ((x - 39) / 27) ** 2 + ((y - 31) / 22) ** 2 <= 1


@pytest.mark.parametrize("seed", [0, 17, 2**64 - 1])
def test_full_imperfect_is_bit_exact_legacy_replay(seed):
    footprint = _footprint()
    expected, parameters = _imperfect_outline(
        footprint,
        np.random.Generator(np.random.PCG64(seed)),
        4,
        1.92,
        (0.02, 0.06),
        (0.01, 0.04),
    )
    variants = paired_smart_brush_masks(footprint, seed)
    full = variants["full-imperfect"]
    assert np.array_equal(full["mask"], expected)
    assert full["parameters"]["sampled_morphology_px"] == parameters["morphology_px"]
    assert full["parameters"]["gap_applied"] == parameters["gap_applied"]
    assert full["parameters"]["island_applied"] == parameters["island_applied"]


def test_paired_masks_isolate_stages_and_keep_exact_black_exterior():
    footprint = _footprint()
    image = np.linspace(0.1, 1.0, footprint.size, dtype=np.float32).reshape(footprint.shape)
    variants = paired_smart_brush_inputs(image, footprint, "0x0000000000000011")
    assert tuple(variants) == MASK_VARIANTS
    assert np.array_equal(variants["accurate"]["mask"], footprint)
    sampled = variants["morphology-only"]["parameters"]["sampled_morphology_px"]
    expected_morphology = (
        ndimage.binary_dilation(footprint, iterations=sampled)
        if sampled > 0
        else ndimage.binary_erosion(footprint, iterations=-sampled)
    )
    assert np.array_equal(variants["morphology-only"]["mask"], expected_morphology)
    assert variants["jitter-gap-island-only"]["parameters"]["applied_morphology_px"] == 0
    assert variants["full-imperfect"]["parameters"]["applied_morphology_px"] == sampled
    for variant in variants.values():
        assert variant["black_exterior_exact"] is True
        assert np.all(variant["model_input_image"][~variant["mask"]] == 0)


def test_inventory_authentication_detects_any_byte_change(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"paired-mask-mechanism")
    inventory = [
        {
            "path": "payload.bin",
            "sha256": __import__("hashlib").sha256(payload.read_bytes()).hexdigest(),
            "size_bytes": payload.stat().st_size,
        }
    ]
    result = {
        "pre_result_inventory": inventory,
        "pre_result_inventory_sha256": payload_sha256(inventory),
    }
    result["result_payload_sha256"] = payload_sha256(result)
    (tmp_path / "result.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert verify_inventory_bound_tree(tmp_path) == result
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="result-tree"):
        verify_inventory_bound_tree(tmp_path)


def _freeze_receipt(sources):
    return {
        "schema": FREEZE_SCHEMA,
        "status": "frozen-before-execution",
        "freeze_commit": "1" * 40,
        "source_sha256": sources,
        "raw_input": {
            "directory": RAW_INPUT.as_posix(),
            "result_file_sha256": RAW_RESULT_FILE_SHA256,
            "result_payload_sha256": RAW_RESULT_PAYLOAD_SHA256,
            "inventory_sha256": RAW_INVENTORY_SHA256,
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


def test_execution_requires_exact_source_bound_freeze_receipt():
    sources = {"training/example.py": "a" * 64}
    receipt = _freeze_receipt(sources)
    verify_freeze_receipt(receipt, check_repository=False, expected_sources=sources)
    receipt["mask_variants"] = list(reversed(MASK_VARIANTS))
    with pytest.raises(ValueError, match="freeze receipt"):
        verify_freeze_receipt(receipt, check_repository=False, expected_sources=sources)


def test_full_study_cannot_start_without_freeze_and_creates_no_output(tmp_path):
    output = tmp_path / "unfrozen-study"
    with pytest.raises(ValueError, match="freeze receipt"):
        run_mask_mechanism_v3(output, {})
    assert not output.exists()


def test_output_guard_restricts_writes_to_fresh_i_drive():
    with pytest.raises(ValueError, match="I: drive"):
        _guard_output(Path(r"C:\mask-mechanism-result"))
    with pytest.raises(ValueError, match="immutable raw"):
        _guard_output(RAW_INPUT / "nested")


def test_metrics_are_paired_and_do_not_create_a_gate():
    primary, shuffled = [], []
    for case_index in range(CASE_COUNT):
        variants = {}
        for variant in MASK_VARIANTS:
            top1 = case_index < 60
            if variant == "full-imperfect":
                top1 = case_index < 52
            variants[variant] = {
                "ranking": {"top1": top1, "reciprocal_rank": 1.0 if top1 else 0.5}
            }
        primary.append({"variants": variants})
        shuffled.append(
            {
                "variants": {
                    variant: {"ranking": {"top1": False, "reciprocal_rank": 0.1}}
                    for variant in MASK_VARIANTS
                }
            }
        )
    metrics = aggregate_metrics(primary, shuffled)
    assert metrics["native"]["accurate"]["top1_count"] == 60
    assert metrics["native"]["full-imperfect"]["top1_count"] == 52
    assert metrics["native"]["full-imperfect"]["paired_top1_vs_accurate"] == {
        "accurate_only": 8,
        "variant_only": 0,
        "both": 52,
        "neither": 4,
    }
    assert "passed" not in metrics and "gate" not in metrics


def test_declared_immutable_raw_tree_authenticates_without_scientific_replay():
    result, config = authenticate_frozen_image_information()
    assert result["result_payload_sha256"] == RAW_RESULT_PAYLOAD_SHA256
    assert config["case_and_shuffle_contract"]["base_count"] == CASE_COUNT
