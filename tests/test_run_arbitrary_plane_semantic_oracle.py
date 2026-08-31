import copy
import hashlib
import json

import numpy as np
import pytest

import training.run_arbitrary_plane_semantic_oracle as runner
from training.arbitrary_plane_finite_candidates import make_arbitrary_plane_finite_candidate_bank
from training.arbitrary_plane_rendered_generator import make_finite_arbitrary_plane_render
from training.arbitrary_plane_support import build_annotation_support_index
from training.arbitrary_plane_semantic_oracle import (
    canonical_payload_sha256,
    rank_candidate_ids,
    semantic_gate_summary,
    shuffled_target_index,
)
from training.run_arbitrary_plane_semantic_oracle import (
    CASE_COUNT,
    CASE_ROOT_SEED_HEX,
    CONTROL_SCHEMAS,
    OUTLINE_MODES,
    _atomic_json,
    _control_record,
    _ranking_payload,
    _shuffled_record,
    _target_receipt,
    _write_control_sidecar,
    case_seed_lineage,
    derive_case_seed,
    expected_orientation,
    orientation_accepts,
    repository_state,
    shuffled_case_cycle,
    verify_written_result,
)


def _digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


_FIXTURE_SUPPORT = None


def _fixture_support():
    global _FIXTURE_SUPPORT
    if _FIXTURE_SUPPORT is None:
        annotation = np.zeros((9, 11, 13), dtype=np.uint8)
        annotation[1:8, 2:10, 1:12] = 7
        _FIXTURE_SUPPORT = build_annotation_support_index(
            annotation,
            atlas_id="fixture-ccf",
            atlas_version="fixture-v1",
            source_uri="fixture/annotation.nrrd",
            source_sha256="3" * 64,
            source_entity_type="atlas-annotation",
            voxel_size_um=(100.0, 100.0, 100.0),
            origin_um=(-450.0, -550.0, -650.0),
            coordinate_axis_directions=("posterior", "inferior", "right"),
        )
    return _FIXTURE_SUPPORT


@pytest.fixture(autouse=True)
def _recorded_git_blobs(monkeypatch):
    monkeypatch.setattr(runner, "_git_blob_sha256", lambda _commit, path: _digest(path))
    original_transport = runner.transport_finite_candidate_pose

    def transport(parent, support, normal, offset, roll):
        if "fixture_case_index" not in parent:
            return original_transport(parent, support, normal, offset, roll)
        case_index = int(parent["fixture_case_index"])
        slot = int(round(abs(float(offset))))
        return {
            "candidate_geometry_sha256": _digest(
                f"truth-transport-{case_index}"
                if slot == 0
                else f"candidate-geometry-{case_index}-{slot}"
            )
        }

    monkeypatch.setattr(runner, "transport_finite_candidate_pose", transport)


def _source_hashes():
    return {path: _digest(path) for path in runner.SOURCE_RELATIVE_PATHS}


def _prepared_receipts():
    shape = [528, 320, 456]
    template = {
        "decoder": "pynrrd 1.1.3",
        "index_order": "F",
        "dtype": "<u2",
        "shape": shape,
        "array_sha256": _digest("decoded-template"),
    }
    annotation = {
        "decoder": "pynrrd 1.1.3",
        "index_order": "F",
        "dtype": "<u4",
        "shape": shape,
        "array_sha256": _digest("decoded-annotation"),
    }
    conversion = {
        "operation": "numpy.array(dtype=<f4, copy=True, order=C)",
        "normalization": "none",
        "dtype": "<f4",
        "shape": shape,
        "array_sha256": _digest("converted-template"),
    }
    sampling = {
        "operation": "native-dtype nearest indexing, then sampled H-by-W labels converted to torch.int64",
        "losslessness": "required nonnegative labels <= int64 maximum",
        "full_volume_copy": "none",
        "rendered_output_dtype": "<i8",
    }
    support_sha256 = _fixture_support()["support_index_sha256"]
    render = {
        "schema": "anatomy-tracker.prepared-finite-render-context/v1",
        "support_index_sha256": support_sha256,
        "template_decoded": template,
        "scalar_conversion": conversion,
        "annotation_decoded": annotation,
        "annotation_sampling": sampling,
        "scalar_source": {
            "source_entity_type": "atlas-template",
            "uri": runner.ATLAS_TEMPLATE_URI,
            "source_sha256": runner.ATLAS_TEMPLATE_SHA256,
            "source_sha256_semantics": "raw source bytes",
        },
    }
    candidate = {
        "schema": "anatomy-tracker.prepared-finite-candidate-annotation/v1",
        "support_index_sha256": support_sha256,
        "annotation": {
            "dtype": annotation["dtype"],
            "shape": shape,
            "array_sha256": annotation["array_sha256"],
            "storage": "owned immutable C-order bytes",
        },
    }
    return support_sha256, render, candidate


def _fake_score_for_controls(target_labels, candidate_labels, fixed_valid_mask, pixel_pitch_um):
    values = 1.0 - np.asarray(candidate_labels[:, 0, 0], dtype=np.float64) / 100.0
    return {
        "semantic_score": values,
        "raw_id_agreement": values,
        "mask_dice": values,
        "target_large_region_ids": np.array([7], dtype=np.int64),
        "target_small_region_ids": np.array([], dtype=np.int64),
        "channel_count": 1,
        "smoothing_sigma_px": 75.0 / pixel_pitch_um,
    }


def test_case_seed_derivation_is_the_exact_frozen_little_endian_literal():
    payload = (
        "arbitrary-plane-oracle-cases/v1\0"
        f"{CASE_ROOT_SEED_HEX}\0synthetic\0{37}\0{5}"
    )
    expected = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")
    assert derive_case_seed("synthetic", 37, 5) == expected
    assert derive_case_seed("finite-parent", 37, 5) != expected
    with pytest.raises(ValueError, match="frozen design"):
        derive_case_seed("unknown", 0, 0)


def test_orientation_assignment_and_single_shuffled_cycle_cover_all_frozen_cases():
    assert {name: sum(expected_orientation(index) == name for index in range(64)) for name in (
        "near_AP", "near_DV", "near_ML", "general_oblique"
    )} == {"near_AP": 12, "near_DV": 12, "near_ML": 12, "general_oblique": 28}
    assert orientation_accepts(0, [1.0, 0.0, 0.0])
    assert orientation_accepts(12, [0.0, -1.0, 0.0])
    assert orientation_accepts(24, [0.0, 0.0, 1.0])
    assert orientation_accepts(36, np.ones(3) / np.sqrt(3))
    assert not orientation_accepts(36, [1.0, 0.0, 0.0])
    cycle = shuffled_case_cycle()
    assert len(cycle) == len(set(cycle)) == CASE_COUNT
    assert all(shuffled_target_index(first) == second for first, second in zip(cycle, cycle[1:] + cycle[:1]))


def test_case_builder_forwards_frozen_commit_to_parent_and_candidate_verification(monkeypatch):
    source_commit = "a" * 40
    seen = {}

    def finite_parent(*args, generator_source_commit, **kwargs):
        seen["finite_parent"] = generator_source_commit
        return {"geometry": {"normal_rp2_ap_dv_ml": [1.0, 0.0, 0.0]}}

    def candidate_bank(
        parent,
        candidate_context,
        support_index,
        *,
        finite_parent_generator_source_commit,
    ):
        seen["candidate_bank"] = finite_parent_generator_source_commit
        raise RuntimeError("commit-forwarded")

    monkeypatch.setattr(runner, "make_finite_arbitrary_plane_render_from_context", finite_parent)
    monkeypatch.setattr(
        runner, "make_arbitrary_plane_finite_candidate_bank_from_context", candidate_bank
    )
    monkeypatch.setattr(runner, "orientation_accepts", lambda case_index, normal: True)

    with pytest.raises(RuntimeError, match="commit-forwarded"):
        runner.build_oracle_case(0, {}, {}, {}, source_commit)

    assert seen == {"finite_parent": source_commit, "candidate_bank": source_commit}


def test_atomic_json_is_durable_idempotent_and_refuses_changed_frozen_output(tmp_path):
    path = tmp_path / "nested" / "receipt.json"
    first = _atomic_json(path, {"seed": np.uint64(7), "status": "frozen"})
    second = _atomic_json(path, {"status": "frozen", "seed": 7})
    assert first == second == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="nonidentical"):
        _atomic_json(path, {"seed": 8, "status": "frozen"})


def _small_bank():
    ouv = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 3.0, 0.0]
    candidates = []
    for index in range(3):
        pose = {
            "normal_rp2_sign_aligned_ap_dv_ml": [1.0, 0.0, 0.0],
            "signed_offset_um": float(index),
            "roll_delta_rad_from_parallel_transport": 0.0,
        }
        candidate = {
            "candidate_class": "truth" if index == 0 else "offset_only",
            "candidate_id": _digest(f"candidate-{index}"),
            "physical_pose": pose,
        }
        if index:
            candidate["geometry"] = {"effective_physical_ouv_ap_dv_ml_um": [value + index for value in ouv]}
        candidates.append(candidate)
    return {
        "candidates": candidates,
        "ordered_candidate_ids": [item["candidate_id"] for item in candidates],
        "truth_parent_geometry": {"effective_physical_ouv_ap_dv_ml_um": ouv},
        "finite_candidate_bank_id": _digest("bank"),
        "finite_candidate_receipt_sha256": _digest("bank-receipt"),
    }


def test_target_receipt_ranking_and_shuffled_record_bind_raw_arrays_not_metadata():
    labels = np.zeros((6, 8), dtype=np.int64)
    labels[1:5, 1:4] = 7
    labels[1:5, 4:7] = 19
    valid = labels != 0
    candidates = np.stack((labels, np.roll(labels, 1, 1), np.zeros_like(labels)))
    from training.arbitrary_plane_semantic_oracle import score_semantic_candidates

    scores = score_semantic_candidates(labels, candidates, valid, 100.0)
    target = {"labels": labels, "fixed_valid_mask": valid, "pixel_pitch_um": 100.0}
    receipt = _target_receipt(17, _digest("paired-17"), target, scores)
    assert receipt["labels_receipt"]["shape"] == [6, 8]
    assert receipt["channel_receipt_sha256"] == canonical_payload_sha256(receipt["channel_receipt"])
    assert receipt["target_receipt_sha256"] == canonical_payload_sha256({key: value for key, value in receipt.items() if key != "target_receipt_sha256"})

    bank = _small_bank()
    ranking = _ranking_payload(bank, scores, valid)
    assert ranking["selected_candidate_id"] == bank["ordered_candidate_ids"][0]
    assert ranking["selected_pose_error"]["corresponding_point_rms_um"] == 0.0
    tied = dict(scores)
    tied["semantic_score"] = np.array([1.0, 1.0, 0.0])
    tied_ranking = _ranking_payload(bank, tied, valid)
    assert tied_ranking["selected_pose_error"] is None
    assert tied_ranking["tied_maximum_candidate_ids"] == sorted(bank["ordered_candidate_ids"][:2])

    pending = {
        "case_index": 0,
        "bank": bank,
        "candidate_labels": candidates,
        "primary_scores": scores,
        "primary_record": {"paired_view_group_id": _digest("paired-0")},
    }
    target_record = {
        "case_index": 17,
        "paired_view_group_id": _digest("paired-17"),
        "target": receipt,
    }
    shuffled = _shuffled_record(pending, target_record, target)
    assert shuffled["target"] == receipt
    assert shuffled["ordered_candidate_ids"] == bank["ordered_candidate_ids"]
    assert shuffled["shuffled_payload_sha256"] == canonical_payload_sha256({key: value for key, value in shuffled.items() if key != "shuffled_payload_sha256"})

    changed_labels = {**target, "labels": target["labels"].copy()}
    changed_labels["labels"][valid] = np.where(changed_labels["labels"][valid] == 7, 19, 7)
    with pytest.raises(ValueError, match="arrays or pitch"):
        _shuffled_record(pending, target_record, changed_labels)
    changed_mask = {**target, "fixed_valid_mask": target["fixed_valid_mask"].copy()}
    changed_mask["fixed_valid_mask"][1, 1] = False
    with pytest.raises(ValueError, match="arrays or pitch"):
        _shuffled_record(pending, target_record, changed_mask)
    with pytest.raises(ValueError, match="arrays or pitch"):
        _shuffled_record(pending, target_record, {**target, "pixel_pitch_um": 101.0})


def test_five_control_records_stream_full_name_bound_evidence_and_require_all_cases(tmp_path):
    primary = []
    for index in range(64):
        family = expected_orientation(index)
        normal = (
            [1.0, 0.0, 0.0]
            if family == "near_AP"
            else [0.0, 1.0, 0.0]
            if family == "near_DV"
            else [0.0, 0.0, 1.0]
            if family == "near_ML"
            else (np.ones(3) / np.sqrt(3)).tolist()
        )
        primary.append(_primary_fixture(index, normal, family, 1))
    for name in (
        "exact_replay",
        "candidate_order_permutation_equivariance",
        "rp2_sign_equivalence",
        "truth_metadata_coordinate_channel_exclusion",
        "xy_over_wh_coordinate_contract",
    ):
        references = [
            _write_control_sidecar(tmp_path, _strict_control(name, primary[index]))
            for index in range(64)
        ]
        record = _control_record(name, references)
        assert record["passed"] is True
        assert record["evidence"]["control"] == name
        assert len(record["evidence"]["case_evidence"]) == 64
        assert json.loads((tmp_path / references[0]["relative_path"]).read_text())["control"] == name
        assert record["evidence_receipt_sha256"] == canonical_payload_sha256({"control": name, "passed": True, "evidence": record["evidence"]})
        assert _control_record(name, references[:-1])["passed"] is False


def test_case_controls_producer_emits_full_replayable_evidence(monkeypatch):
    record = _primary_fixture(0, [1.0, 0.0, 0.0], "near_AP", 1)
    bank = record["candidate_bank_receipt"]

    def sampling_arrays(geometry, _atlas_shape):
        ouv = np.asarray(geometry["effective_allen_index_ouv_ap_dv_ml"], dtype=np.float32)
        y, x = np.meshgrid(
            np.arange(192, dtype=np.float32) / np.float32(192),
            np.arange(256, dtype=np.float32) / np.float32(256),
            indexing="ij",
        )
        points = (ouv[:3] + x[..., None] * ouv[3:6] + y[..., None] * ouv[6:9]).astype(
            np.float32
        )
        return {
            "coordinate_raster_allen_index_float32": points,
            "allen_index_ouv_ap_dv_ml_float32": ouv,
        }

    for slot, candidate in enumerate(bank["candidates"]):
        geometry = bank["truth_parent_geometry"] if slot == 0 else candidate["geometry"]
        geometry["array_receipts"] = {
            (
                "effective_coordinate_raster_allen_index_float32"
                if slot == 0
                else "coordinate_raster_allen_index_float32"
            ): runner._array_receipt(sampling_arrays(geometry, None)["coordinate_raster_allen_index_float32"])
        }
    labels = np.zeros((40, 192, 256), dtype=np.int64)
    labels[:, 0, 0] = np.arange(40)
    target = {
        "labels": np.zeros((192, 256), dtype=np.int64),
        "fixed_valid_mask": np.ones((192, 256), dtype=bool),
        "pixel_pitch_um": 25.0,
    }
    primary_scores = _fake_score_for_controls(
        target["labels"], labels, target["fixed_valid_mask"], target["pixel_pitch_um"]
    )
    record["scores"] = runner._score_payload(primary_scores)
    record["candidate_bank_receipt"] = bank
    monkeypatch.setattr(runner, "score_semantic_candidates", _fake_score_for_controls)
    monkeypatch.setattr(runner, "effective_renderer_sampling_arrays", sampling_arrays)
    monkeypatch.setattr(
        runner,
        "transport_finite_candidate_pose",
        lambda _geometry, _support, _normal, offset, _roll: {
            "candidate_geometry_sha256": _digest(
                f"candidate-geometry-0-{int(round(abs(float(offset))))}"
            )
        },
    )
    controls = runner._case_controls(
        record,
        target,
        {"bank": bank, "candidate_labels": labels, "primary_scores": primary_scores},
        {"annotation_shape": [528, 320, 456]},
    )
    assert set(controls) == set(CONTROL_SCHEMAS) - {"exact_replay"}
    for name, evidence in controls.items():
        runner._verify_control_payload(name, evidence, record, _fixture_support())


def test_real_truth_candidate_rp2_transport_evidence_verifies_without_hash_aliasing():
    shape = (33, 31, 35)
    annotation = np.zeros(shape, dtype=np.uint16)
    annotation[2:-2, 2:-2, 2:-2] = 7
    annotation[7:17, 5:15, 4:18] = 19
    annotation[17:29, 15:27, 18:31] = 41
    ap, dv, ml = np.indices(shape)
    template = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.uint16)
    support = build_annotation_support_index(
        annotation,
        atlas_id="rp2-fixture-ccf",
        atlas_version="fixture-v1",
        source_uri="fixture/annotation.nrrd",
        source_sha256="3" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(100.0, 100.0, 100.0),
        origin_um=(-1700.0, -1500.0, -1800.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    parent = make_finite_arbitrary_plane_render(
        template,
        annotation,
        support,
        "development",
        1,
        (48, 64),
        sample_index=3,
        margin_um=(250.0, 250.0),
        scalar_source_uri="fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="fixture",
        annotation_decoder="fixture",
        minimum_brain_pixels=64,
    )
    bank = make_arbitrary_plane_finite_candidate_bank(parent, annotation, support)
    target = {
        "labels": np.asarray(parent["raster"]["annotation"]),
        "fixed_valid_mask": np.asarray(parent["raster"]["brain_mask"]),
        "pixel_pitch_um": parent["geometry"]["reference_aspect_policy"][
            "pixel_pitch_u_um"
        ],
    }
    candidate_labels = np.stack(
        [candidate["rendered_annotation"] for candidate in bank["candidates"]]
    )
    scores = runner.score_semantic_candidates(
        target["labels"], candidate_labels, target["fixed_valid_mask"],
        target["pixel_pitch_um"],
    )
    primary = {
        "case_index": 0,
        "truth_candidate_id": next(
            item["candidate_id"] for item in bank["candidates"]
            if item["candidate_class"] == "truth"
        ),
        "finite_parent_receipt": runner.finite_render_receipt(parent),
        "candidate_bank_receipt": runner.finite_candidate_bank_receipt(bank),
        "ordered_candidate_ids": bank["ordered_candidate_ids"],
        "scores": runner._score_payload(scores),
    }
    evidence = runner._case_controls(
        primary,
        target,
        {"bank": bank, "candidate_labels": candidate_labels, "primary_scores": scores},
        support,
    )["rp2_sign_equivalence"]
    truth_index = next(
        index for index, item in enumerate(bank["candidates"])
        if item["candidate_class"] == "truth"
    )
    truth_evidence = evidence["candidate_receipts"][truth_index]
    assert truth_evidence["positive_geometry_sha256"] != truth_evidence[
        "stored_candidate_geometry_sha256"
    ]
    runner._verify_control_payload(
        "rp2_sign_equivalence", evidence, primary, support
    )
    changed = copy.deepcopy(evidence)
    forged = _digest("coordinated-positive-and-antipodal-forgery")
    changed["candidate_receipts"][truth_index]["positive_geometry_sha256"] = forged
    changed["candidate_receipts"][truth_index]["antipodal_geometry_sha256"] = forged
    with pytest.raises(ValueError, match="authenticated support index"):
        runner._verify_control_payload(
            "rp2_sign_equivalence", changed, primary, support
        )


def test_repository_state_requires_clean_branch_at_its_exact_upstream(monkeypatch):
    values = {
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
        ("rev-parse", "--abbrev-ref", "HEAD"): "codex/joint-registration",
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"): "origin/codex/joint-registration",
        ("rev-parse", "HEAD"): "a" * 40,
        ("rev-parse", "@{upstream}"): "a" * 40,
    }
    monkeypatch.setattr(runner, "_git", lambda *arguments: values[arguments])
    assert repository_state() == {
        "branch": "codex/joint-registration",
        "upstream": "origin/codex/joint-registration",
        "head": "a" * 40,
        "upstream_head": "a" * 40,
        "worktree_clean": True,
    }
    values[("status", "--porcelain=v1", "--untracked-files=all")] = "?? runner.py"
    with pytest.raises(RuntimeError, match="clean worktree"):
        repository_state()
    values[("status", "--porcelain=v1", "--untracked-files=all")] = ""
    values[("rev-parse", "@{upstream}")] = "b" * 40
    with pytest.raises(RuntimeError, match="equals its upstream"):
        repository_state()


def test_source_receipts_separate_git_blobs_from_loaded_checkout_bytes(monkeypatch):
    monkeypatch.setattr(runner, "_git_blob_sha256", lambda _commit, path: _digest(f"git:{path}"))
    monkeypatch.setattr(runner, "_file_sha256", lambda path: _digest(f"checkout:{path.name}"))
    git_blobs, checkout = runner._source_hash_receipts("a" * 40)
    assert set(git_blobs) == set(checkout) == set(runner.SOURCE_RELATIVE_PATHS)
    assert all(git_blobs[path] != checkout[path] for path in runner.SOURCE_RELATIVE_PATHS)


def test_result_write_must_self_verify_before_run_completion(monkeypatch, tmp_path):
    calls = []
    result = {"result_payload_sha256": _digest("result")}
    monkeypatch.setattr(
        runner,
        "_atomic_json",
        lambda path, value: calls.append(("write", path.name, value)) or _digest("file"),
    )
    monkeypatch.setattr(
        runner,
        "_verify_written_result",
        lambda output, support: calls.append(("verify", output, support)) or result,
    )
    support = _fixture_support()
    assert runner._write_and_verify_result(tmp_path, result, support) == result
    assert [item[0] for item in calls] == ["write", "verify"]
    assert calls[1][2] is support
    source = runner.inspect.getsource(runner.run_oracle)
    assert source.index("_write_and_verify_result(output, result, support)") < source.index(
        '"event": "run-complete"'
    )


def test_public_offline_verifier_rebuilds_authenticated_allen_support(monkeypatch, tmp_path):
    support = _fixture_support()
    calls = []
    monkeypatch.setattr(
        runner,
        "_load_authenticated_allen_support_index",
        lambda: calls.append("rebuild") or support,
    )
    monkeypatch.setattr(
        runner,
        "_verify_written_result",
        lambda output, authenticated: calls.append((output, authenticated)) or {"ok": True},
    )
    assert verify_written_result(tmp_path) == {"ok": True}
    assert calls == ["rebuild", (tmp_path, support)]


def test_offline_verifier_accepts_only_allowlisted_synthetic_retry_then_accept():
    record = _primary_fixture(0, [1.0, 0.0, 0.0], "near_AP", 1)
    record["accepted_case_attempt_index"] = 1
    record["accepted_case_field_stream_seed_uint64"] = case_seed_lineage(0, 1)
    record["outline_assignment"]["field_stream_seed_uint64"] = case_seed_lineage(0, 1)["outline"]
    record["outline_assignment_sha256"] = canonical_payload_sha256(record["outline_assignment"])
    rejected_parent = copy.deepcopy(record["finite_parent_receipt"])
    rejected_parent["plane_realization_id"] = _digest("rejected-parent")
    rejection = {
        "attempt_index": 0,
        "field_stream_seed_uint64": case_seed_lineage(0, 0),
        "stage": "synthetic-eligibility",
        "outline_mode": OUTLINE_MODES[1],
        "finite_parent_receipt": rejected_parent,
        "candidate_bank_id": _digest("rejected-bank"),
        "candidate_bank_receipt_sha256": _digest("rejected-bank-receipt"),
        "reason": {
            "exception_type": "ValueError",
            "message": "G2 realization failed all deterministic information-content rejection attempts",
            "structured_candidate_rejection": None,
        },
    }
    record["case_rejection_attempts"] = [rejection]
    record["case_rejection_attempts_sha256"] = canonical_payload_sha256([rejection])
    record["case_payload_sha256"] = canonical_payload_sha256(
        {key: value for key, value in record.items() if key != "case_payload_sha256"}
    )
    runner._verify_primary_record(record, 0)
    rejection["reason"]["message"] = "unexpected implementation failure"
    record["case_rejection_attempts_sha256"] = canonical_payload_sha256([rejection])
    with pytest.raises(ValueError, match="frozen stochastic eligibility"):
        runner._verify_primary_record(record, 0)
    rejection["reason"]["message"] = (
        "G2 realization failed all deterministic information-content rejection attempts"
    )
    rejection["candidate_bank_receipt_sha256"] = "not-a-digest"
    record["case_rejection_attempts_sha256"] = canonical_payload_sha256([rejection])
    with pytest.raises(ValueError, match="frozen stochastic eligibility"):
        runner._verify_primary_record(record, 0)


def test_offline_verifier_requires_exact_hash_bound_nested_rejection_receipts():
    record = _primary_fixture(0, [1.0, 0.0, 0.0], "near_AP", 1)
    record["accepted_case_attempt_index"] = 1
    record["accepted_case_field_stream_seed_uint64"] = case_seed_lineage(0, 1)
    record["outline_assignment"]["field_stream_seed_uint64"] = case_seed_lineage(0, 1)[
        "outline"
    ]
    record["outline_assignment_sha256"] = canonical_payload_sha256(
        record["outline_assignment"]
    )
    attempts = [{"candidate_class": "global_hard_negative", "attempt_index": 4095}]
    structured = {
        "schema": "anatomy-tracker.finite-candidate-case-rejection/v1",
        "reason": "candidate bank is not exactly 40 unique identities",
        "candidate_attempts": attempts,
        "candidate_attempts_sha256": canonical_payload_sha256(attempts),
    }
    rejection = {
        "attempt_index": 0,
        "field_stream_seed_uint64": case_seed_lineage(0, 0),
        "stage": "finite-candidate-bank",
        "finite_parent_receipt": copy.deepcopy(record["finite_parent_receipt"]),
        "reason": {
            "exception_type": "ValueError",
            "message": "finite candidate case rejected: "
            + json.dumps(structured, sort_keys=True, separators=(",", ":")),
            "structured_candidate_rejection": structured,
        },
    }
    record["case_rejection_attempts"] = [rejection]
    record["case_rejection_attempts_sha256"] = canonical_payload_sha256([rejection])
    record["case_payload_sha256"] = canonical_payload_sha256(
        {key: value for key, value in record.items() if key != "case_payload_sha256"}
    )
    runner._verify_primary_record(record, 0)

    changed = copy.deepcopy(record)
    changed["case_rejection_attempts"][0]["reason"]["unexpected"] = True
    changed["case_rejection_attempts_sha256"] = canonical_payload_sha256(
        changed["case_rejection_attempts"]
    )
    with pytest.raises(ValueError, match="exact exception receipt"):
        runner._verify_primary_record(changed, 0)

    changed = copy.deepcopy(record)
    nested = changed["case_rejection_attempts"][0]["reason"][
        "structured_candidate_rejection"
    ]
    nested["candidate_attempts_sha256"] = _digest("wrong-attempts")
    changed["case_rejection_attempts"][0]["reason"]["message"] = (
        "finite candidate case rejected: "
        + json.dumps(nested, sort_keys=True, separators=(",", ":"))
    )
    changed["case_rejection_attempts_sha256"] = canonical_payload_sha256(
        changed["case_rejection_attempts"]
    )
    with pytest.raises(ValueError, match="hash-bound stochastic eligibility"):
        runner._verify_primary_record(changed, 0)

    changed = copy.deepcopy(record)
    changed["case_rejection_attempts"][0]["finite_parent_receipt"][
        "support_index_sha256"
    ] = "not-a-digest"
    changed["case_rejection_attempts_sha256"] = canonical_payload_sha256(
        changed["case_rejection_attempts"]
    )
    with pytest.raises(ValueError, match="support_index_sha256"):
        runner._verify_primary_record(changed, 0)


def test_saved_pose_error_replays_pose_and_exact_point_evidence():
    record = _primary_fixture(0, [1.0, 0.0, 0.0], "near_AP", 2)
    derived = rank_candidate_ids(
        record["scores"]["semantic"],
        record["ordered_candidate_ids"],
        record["truth_candidate_id"],
    )
    runner._verify_saved_ranking(
        record,
        derived,
        "primary",
        record["candidate_bank_receipt"],
        _fixture_mask(0),
    )
    changed = copy.deepcopy(record)
    changed["ranking"]["selected_pose_error"]["normal_geodesic_angle_deg"] = 1.0
    with pytest.raises(ValueError, match="normal or offset error"):
        runner._verify_saved_ranking(
            changed,
            derived,
            "primary",
            changed["candidate_bank_receipt"],
            _fixture_mask(0),
        )
    changed = copy.deepcopy(record)
    changed["ranking"]["selected_pose_error"]["corresponding_point_rms_um"] = 1.5
    with pytest.raises(ValueError, match="exact mask"):
        runner._verify_saved_ranking(
            changed,
            derived,
            "primary",
            changed["candidate_bank_receipt"],
            _fixture_mask(0),
        )
    changed = copy.deepcopy(record)
    pose_error = changed["ranking"]["selected_pose_error"]
    evidence = pose_error["evidence"]
    forged_error = 7.0
    count = evidence["fixed_valid_pixel_count"]
    evidence["corresponding_point_error_receipt"]["array_sha256"] = _digest(
        "coordinated-forged-point-errors"
    )
    evidence["squared_error_sum_um2"] = count * forged_error**2
    evidence["p95_linear_quantile"]["lower_value_um"] = forged_error
    evidence["p95_linear_quantile"]["upper_value_um"] = forged_error
    pose_error["corresponding_point_rms_um"] = forged_error
    pose_error["corresponding_point_p95_um"] = forged_error
    pose_error["evidence_sha256"] = canonical_payload_sha256(evidence)
    with pytest.raises(ValueError, match="exact mask"):
        runner._verify_saved_ranking(
            changed,
            derived,
            "primary",
            changed["candidate_bank_receipt"],
            _fixture_mask(0),
        )


def _scores_for_rank(rank):
    scores = np.linspace(0.39, 0.01, 40)
    scores[0] = 1.0 if rank == 1 else 0.5
    if rank > 1:
        scores[1:rank] = np.linspace(0.9, 0.6, rank - 1)
    return scores.tolist()


def _score_payload_for_rank(rank):
    values = _scores_for_rank(rank)
    return {
        "semantic": values,
        "raw_ID_agreement": values,
        "mask_only_Dice": values,
        "channel_count": 1,
        "smoothing_sigma_px": 3.0,
    }


def _fixture_mask(index):
    size = 192 * 256
    flat = np.zeros(size, dtype=bool)
    flat[(np.arange(40000) + 137 * int(index)) % size] = True
    return flat.reshape(192, 256)


def _target(index, paired_id):
    channel = {"large_ids": [7], "small_ids": [], "channel_count": 1}
    mask = _fixture_mask(index)
    target = {
        "source_case_index": index,
        "paired_view_group_id": paired_id,
        "labels_receipt": {"shape": [192, 256], "array_sha256": _digest(f"labels-{index}")},
        "mask_receipt": runner._array_receipt(mask),
        "fixed_valid_mask_binary": runner._mask_binary_receipt(index, mask),
        "fixed_valid_pixel_count": int(mask.sum()),
        "channel_receipt": channel,
        "channel_receipt_sha256": canonical_payload_sha256(channel),
        "pixel_pitch_um": 25.0,
    }
    target["target_receipt_sha256"] = canonical_payload_sha256(target)
    return target


def _independent_generator():
    return {
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
    }


def _render_generator():
    support_sha256, render, _ = _prepared_receipts()
    source_hashes = _source_hashes()
    resolved = {
        "support_index_sha256": support_sha256,
        "prepared_context_sha256": canonical_payload_sha256(render),
        "annotation_array_sha256": render["annotation_decoded"]["array_sha256"],
        "template_decoded_array_sha256": render["template_decoded"]["array_sha256"],
        "scalar_source_uri": runner.ATLAS_TEMPLATE_URI,
        "scalar_source_sha256": runner.ATLAS_TEMPLATE_SHA256,
        "output_shape_h_w": [192, 256],
        "margin_u_v_um": [250.0, 250.0],
        "max_rejection_attempts": 1,
        "minimum_brain_pixels": 64,
        "animal_id": None,
        "specimen_id": None,
        "experiment_id": None,
        "numpy_version": "fixture",
        "torch_version": "fixture",
    }
    implementation = {
        "loaded_source_sha256": source_hashes[
            "training/arbitrary_plane_rendered_generator.py"
        ],
        "source_commit": "a" * 40,
        "loaded_dependency_source_sha256": {
            name: source_hashes[f"training/{name}"]
            for name in (
                "arbitrary_plane_geometry.py", "arbitrary_plane_manifest.py",
                "arbitrary_plane_support.py",
            )
        },
    }
    implementation["implementation_sha256"] = canonical_payload_sha256(implementation)
    return {
        "implementation": implementation,
        "resolved_config": resolved,
        "resolved_config_sha256": canonical_payload_sha256(resolved),
        **_independent_generator(),
    }


def _candidate_generator():
    support_sha256, render, candidate = _prepared_receipts()
    source_hashes = _source_hashes()
    resolved = {
        "support_index_sha256": support_sha256,
        "annotation_array_sha256": render["annotation_decoded"]["array_sha256"],
        "prepared_annotation_context_sha256": canonical_payload_sha256(candidate),
        "output_shape_h_w": [192, 256],
    }
    implementation = {
        "loaded_source_sha256": {
                "candidate_generator": source_hashes[
                    "training/arbitrary_plane_finite_candidates.py"
                ],
                "finite_renderer": source_hashes[
                    "training/arbitrary_plane_rendered_generator.py"
                ],
                "geometry": source_hashes["training/arbitrary_plane_geometry.py"],
                "manifest": source_hashes["training/arbitrary_plane_manifest.py"],
                "support": source_hashes["training/arbitrary_plane_support.py"],
                "predeclared_protocol": source_hashes[
                    "publication/arbitrary_plane_oracle_pose_ranking_preflight.yaml"
                ],
            },
        "numpy_version": "fixture",
        "torch_version": "fixture",
    }
    return {
        "implementation": implementation,
        "implementation_sha256": canonical_payload_sha256(implementation),
        "resolved_config": resolved,
        "resolved_config_sha256": canonical_payload_sha256(resolved),
        **_independent_generator(),
    }


def _synthetic_generator():
    source_hashes = _source_hashes()
    implementation = {
        "loaded_source_sha256": {
                "generator": source_hashes[
                    "training/arbitrary_plane_synthetic_generator.py"
                ],
                "ops": source_hashes["training/arbitrary_plane_synthetic_ops.py"],
                "observation": source_hashes[
                    "training/arbitrary_plane_synthetic_observation.py"
                ],
                "finite_renderer": source_hashes[
                    "training/arbitrary_plane_rendered_generator.py"
                ],
                "predeclared_config": source_hashes[
                    "publication/arbitrary_plane_synthetic_preflight.yaml"
                ],
            },
        "numpy_version": "fixture",
        "scipy_version": "fixture",
    }
    resolved = {"synthetic_stratum": "ordinary"}
    return {
        "implementation": implementation,
        "implementation_sha256": canonical_payload_sha256(implementation),
        "resolved_config": resolved,
        "resolved_config_sha256": canonical_payload_sha256(resolved),
        **_independent_generator(),
    }


def _physical_error(bank, target, selected_candidate_id):
    truth = next(item for item in bank["candidates"] if item["candidate_class"] == "truth")
    selected = next(
        item for item in bank["candidates"] if item["candidate_id"] == selected_candidate_id
    )
    truth_ouv = runner._effective_ouv(bank, truth)
    selected_ouv = runner._effective_ouv(bank, selected)
    plane = runner.rp2_plane_error(
        truth["physical_pose"]["normal_rp2_sign_aligned_ap_dv_ml"],
        truth["physical_pose"]["signed_offset_um"],
        selected["physical_pose"]["normal_rp2_sign_aligned_ap_dv_ml"],
        selected["physical_pose"]["signed_offset_um"],
    )
    return {
        **plane,
        **runner._finite_point_error_with_evidence(
            truth_ouv,
            selected_ouv,
            _fixture_mask(target["source_case_index"]),
        ),
    }


def _rank_payload(scores, bank, target):
    candidate_ids = bank["ordered_candidate_ids"]
    ranking = rank_candidate_ids(scores["semantic"], candidate_ids, candidate_ids[0])
    ranking["selected_pose_error"] = (
        None
        if ranking["selected_candidate_id"] is None
        else _physical_error(bank, target, ranking["selected_candidate_id"])
    )
    return ranking


def _candidate_bank(index, candidate_ids, bank_id, parent):
    candidates = []
    truth_grid = {
        "dtype": "<f4",
        "shape": [192, 256, 3],
        "array_sha256": _digest(f"grid-{index}-0"),
    }
    for slot, candidate_id in enumerate(candidate_ids):
        grid = {
            "dtype": "<f4",
            "shape": [192, 256, 3],
            "array_sha256": _digest(f"grid-{index}-{slot}"),
        }
        candidate = {
            "candidate_class": "truth" if slot == 0 else "offset_only",
            "candidate_id": candidate_id,
            "pose_sha256": _digest(f"pose-{index}-{slot}"),
            "geometry_storage": "truth_parent_geometry" if slot == 0 else "candidate",
            "candidate_geometry_sha256": (
                parent["finite_plane_geometry_sha256"]
                if slot == 0
                else _digest(f"candidate-geometry-{index}-{slot}")
            ),
            "physical_pose": {
                "normal_rp2_sign_aligned_ap_dv_ml": [1.0, 0.0, 0.0],
                "signed_offset_um": float(slot),
                "roll_delta_rad_from_parallel_transport": 0.0,
            },
        }
        if slot:
            candidate["geometry"] = {
                "array_receipts": {"coordinate_raster_allen_index_float32": grid},
                "output_shape_h_w": [192, 256],
                "effective_allen_index_ouv_ap_dv_ml": [
                    float(slot), 0.0, 0.0, 256.0, 0.0, 0.0, 0.0, 192.0, 0.0
                ],
                "effective_physical_ouv_ap_dv_ml_um": [
                    float(slot), 0.0, 0.0, 256.0, 0.0, 0.0, 0.0, 192.0, 0.0
                ],
                "sampling_contract": "quicknii-raster-index-x-over-W-y-over-H-v1",
            }
        candidates.append(candidate)
    return {
        "finite_candidate_bank_id": bank_id,
        "ordered_candidate_ids": candidate_ids,
        "candidates": candidates,
        "truth_parent_geometry": {
            "fixture_case_index": index,
            "array_receipts": {"effective_coordinate_raster_allen_index_float32": truth_grid},
            "output_shape_h_w": [192, 256],
            "effective_allen_index_ouv_ap_dv_ml": [
                0.0, 0.0, 0.0, 256.0, 0.0, 0.0, 0.0, 192.0, 0.0
            ],
            "effective_physical_ouv_ap_dv_ml_um": [
                0.0, 0.0, 0.0, 256.0, 0.0, 0.0, 0.0, 192.0, 0.0
            ],
            "sampling_contract": "quicknii-raster-index-x-over-W-y-over-H-v1",
        },
        "finite_parent_receipt": copy.deepcopy(parent),
        "support_index_sha256": parent["support_index_sha256"],
        "provenance": copy.deepcopy(parent["provenance"]),
        "generator": _candidate_generator(),
    }


def _primary_fixture(index, normal, family, rank):
    candidate_ids = [_digest(f"candidate-{index}-{slot}") for slot in range(40)]
    paired_id = _digest(f"paired-{index}")
    bank_id = _digest(f"bank-{index}")
    target = _target(index, paired_id)
    parent_id = _digest(f"parent-{index}")
    support_sha256, render, _ = _prepared_receipts()
    provenance = {
        "atlas": {"id": "Allen CCFv3", "version": "2017 25um"},
        "annotation_source": {
            "source_entity_type": "atlas",
            "annotation_uri": runner.ATLAS_ANNOTATION_URI,
            "source_sha256": runner.ATLAS_ANNOTATION_SHA256,
            "source_sha256_semantics": "raw source bytes",
            "annotation_array_sha256": render["annotation_decoded"]["array_sha256"],
        },
        "annotation_decoded": copy.deepcopy(render["annotation_decoded"]),
        "annotation_sampling": copy.deepcopy(render["annotation_sampling"]),
        "scalar_source": {
            **copy.deepcopy(render["scalar_source"]),
            "decoded": copy.deepcopy(render["template_decoded"]),
            "float_conversion": copy.deepcopy(render["scalar_conversion"]),
        },
        "animal_id": None,
        "specimen_id": None,
        "experiment_id": None,
    }
    parent = {
        "schema_version": "anatomy-tracker.finite-arbitrary-plane-render/v1",
        "generator_algorithm": "uniform-rp2-component-union-finite-render/v1",
        "plane_realization_id": parent_id,
        "split": "development",
        "support_index_sha256": support_sha256,
        "provenance": provenance,
        "provenance_sha256": canonical_payload_sha256(provenance),
        "generator": _render_generator(),
        "rejection_attempts": [],
        "rejection_attempts_sha256": canonical_payload_sha256([]),
        "finite_plane_geometry_sha256": _digest(f"finite-geometry-{index}"),
        "rendered_artifacts_sha256": _digest(f"rendered-artifacts-{index}"),
        "geometry": {"normal_rp2_ap_dv_ml": normal},
    }
    bank = _candidate_bank(index, candidate_ids, bank_id, parent)
    bank_sha256 = canonical_payload_sha256(bank)
    outline_ids = [_digest(f"outline-{index}-{mode}") for mode in OUTLINE_MODES]
    descendants = []
    for mode, descendant_id in zip(OUTLINE_MODES, outline_ids, strict=True):
        descendants.append(
            {
                "mode": mode,
                "synthetic_realization_id": descendant_id,
                "synthetic_receipt": {
                    "synthetic_realization_id": descendant_id,
                    "paired_view_group_id": paired_id,
                    "support_index_sha256": support_sha256,
                    "provenance": copy.deepcopy(provenance),
                    "generator": _synthetic_generator(),
                },
                "oracle_target_labels_receipt": copy.deepcopy(target["labels_receipt"]),
                "oracle_target_mask_receipt": copy.deepcopy(target["mask_receipt"]),
            }
        )
    assignment = {
        "field_stream_seed_uint64": case_seed_lineage(index, 0)["outline"],
        "assignment": "three frozen explicit paired-counterfactual mode strings",
        "ordered_modes": list(OUTLINE_MODES),
    }
    scores = _score_payload_for_rank(rank)
    record = {
        "schema": "anatomy-tracker.arbitrary-plane-semantic-oracle-case/v1",
        "case_index": index,
        "case_root_seed": CASE_ROOT_SEED_HEX,
        "accepted_case_attempt_index": 0,
        "accepted_case_field_stream_seed_uint64": case_seed_lineage(index, 0),
        "case_rejection_attempts": [],
        "case_rejection_attempts_sha256": canonical_payload_sha256([]),
        "orientation": family,
        "truth_normal_ap_dv_ml": normal,
        "truth_signed_offset_um": 0.0,
        "parent_plane_realization_id": parent_id,
        "finite_parent_receipt": parent,
        "paired_view_group_id": paired_id,
        "outline_assignment": assignment,
        "outline_assignment_sha256": canonical_payload_sha256(assignment),
        "outline_descendant_ids": outline_ids,
        "outline_descendants": descendants,
        "candidate_bank_id": bank_id,
        "candidate_bank_receipt_sha256": bank_sha256,
        "candidate_bank_receipt": bank,
        "ordered_candidate_ids": candidate_ids,
        "truth_candidate_id": candidate_ids[0],
        "target": target,
        "scores": scores,
        "ranking": _rank_payload(scores, bank, target),
        "provenance": {
            "animal_id": None,
            "specimen_id": None,
            "experiment_id": None,
            "atlas": copy.deepcopy(provenance["atlas"]),
            "annotation_source": copy.deepcopy(provenance["annotation_source"]),
        },
        "data_access": {
            "allen_synthetic_development_only": True,
            "deepslice_ground_truth_accessed": False,
            "real_lab_histology_accessed": False,
            "final_test_animals_accessed": False,
        },
        "reporting_strata": {
            "orientation_family": family,
            "appearance_family": "label-conditioned",
            "damage_event_types": [],
            "damage_event_count": 0,
            "damage_union_fraction": 0.0,
            "parent_brain_pixel_count": 1000,
            "fixed_valid_pixel_count": target["fixed_valid_pixel_count"],
        },
    }
    record["case_payload_sha256"] = canonical_payload_sha256(record)
    return record


def _strict_control(name, primary):
    index = primary["case_index"]
    common = {
        "schema": CONTROL_SCHEMAS[name],
        "control": name,
        "case_index": index,
        "passed": True,
    }
    if name == "exact_replay":
        return {
            **common,
            "case_payload_sha256": primary["case_payload_sha256"],
            "replayed_case_payload_sha256": primary["case_payload_sha256"],
            "target_receipt_sha256": primary["target"]["target_receipt_sha256"],
            "candidate_bank_id": primary["candidate_bank_id"],
            "raw_scores_sha256": canonical_payload_sha256(primary["scores"]),
            "paired_outline_semantic_receipts": [
                [
                    item["oracle_target_labels_receipt"]["array_sha256"],
                    item["oracle_target_mask_receipt"]["array_sha256"],
                ]
                for item in primary["outline_descendants"]
            ],
        }
    if name == "candidate_order_permutation_equivariance":
        permutation = list(range(39, -1, -1))
        permuted_ids = [primary["ordered_candidate_ids"][slot] for slot in permutation]
        permuted_scores = {
            **primary["scores"],
            **{
                key: [primary["scores"][key][slot] for slot in permutation]
                for key in ("semantic", "raw_ID_agreement", "mask_only_Dice")
            },
        }
        return {
            **common,
            "permutation": permutation,
            "original_ordered_candidate_ids": primary["ordered_candidate_ids"],
            "permuted_ordered_candidate_ids": permuted_ids,
            "truth_candidate_id": primary["truth_candidate_id"],
            "original_scores_sha256": canonical_payload_sha256(primary["scores"]),
            "permuted_scores": permuted_scores,
            "permuted_scores_sha256": canonical_payload_sha256(permuted_scores),
            "original_ranking": rank_candidate_ids(
                primary["scores"]["semantic"], primary["ordered_candidate_ids"], primary["truth_candidate_id"]
            ),
            "permuted_ranking": rank_candidate_ids(
                permuted_scores["semantic"], permuted_ids, primary["truth_candidate_id"]
            ),
        }
    candidates = primary["candidate_bank_receipt"]["candidates"]
    if name == "rp2_sign_equivalence":
        return {
            **common,
            "candidate_receipts": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "source_pose_sha256": candidate["pose_sha256"],
                    "geometry_storage": candidate["geometry_storage"],
                    "stored_candidate_geometry_sha256": candidate[
                        "candidate_geometry_sha256"
                    ],
                    "positive_geometry_sha256": (
                        _digest(f"truth-transport-{index}")
                        if candidate["candidate_class"] == "truth"
                        else candidate["candidate_geometry_sha256"]
                    ),
                    "antipodal_geometry_sha256": (
                        _digest(f"truth-transport-{index}")
                        if candidate["candidate_class"] == "truth"
                        else candidate["candidate_geometry_sha256"]
                    ),
                    "equal": True,
                }
                for slot, candidate in enumerate(candidates)
            ],
        }
    if name == "truth_metadata_coordinate_channel_exclusion":
        source = runner.inspect.getsource(runner.score_semantic_candidates)
        return {
            **common,
            "scorer_signature": list(runner.inspect.signature(runner.score_semantic_candidates).parameters),
            "scorer_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "forbidden_source_tokens": ["candidate_id", "truth_index", "coordinate", "geometry", "normal", "offset", "roll"],
            "forbidden_matches": [],
        }
    bank = primary["candidate_bank_receipt"]
    receipts = []
    for slot, candidate in enumerate(candidates):
        geometry = bank["truth_parent_geometry"] if slot == 0 else candidate["geometry"]
        source = next(iter(geometry["array_receipts"].values()))
        height, width = geometry["output_shape_h_w"]
        ouv = np.asarray(geometry["effective_allen_index_ouv_ap_dv_ml"], dtype=np.float32)
        s = np.arange(width, dtype=np.float32) / np.float32(width)
        t = np.arange(height, dtype=np.float32) / np.float32(height)
        tt, ss = np.meshgrid(t, s, indexing="ij")
        reconstructed = (
            ouv[:3] + ss[..., None] * ouv[3:6] + tt[..., None] * ouv[6:9]
        ).astype(np.float32, copy=False)
        receipts.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_coordinate_raster_receipt": source,
                "reconstructed_xy_over_wh_grid_receipt": {
                    **runner._array_receipt(reconstructed),
                },
                "output_shape_h_w": [192, 256],
                "grid_point_count": 192 * 256,
                "maximum_absolute_residual_allen_index": 0.0,
                "inclusive_endpoint_gap_allen_index": 1.0,
                "equal_within_float32_tolerance": True,
            }
        )
    return {**common, "candidate_receipts": receipts}


def _written_fixture(folder):
    normals = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], (np.ones(3) / np.sqrt(3)).tolist())
    family_successes = {"near_AP": 10, "near_DV": 10, "near_ML": 10, "general_oblique": 22}
    local_counts = {name: 0 for name in family_successes}
    primary = []
    for index in range(64):
        family = expected_orientation(index)
        family_index = local_counts[family]
        local_counts[family] += 1
        normal = normals[0 if family == "near_AP" else 1 if family == "near_DV" else 2 if family == "near_ML" else 3]
        record = _primary_fixture(
            index,
            normal,
            family,
            1 if family_index < family_successes[family] else 2,
        )
        _atomic_json(folder / "primary" / f"case-{index:03d}.json", record)
        runner._write_mask_sidecar(folder, record["target"], _fixture_mask(index))
        primary.append(record)
    shuffled = []
    for index, base in enumerate(primary):
        scores = _score_payload_for_rank(1 if index < 6 else 40)
        target = copy.deepcopy(primary[shuffled_target_index(index)]["target"])
        record = {
            "schema": "anatomy-tracker.arbitrary-plane-semantic-oracle-shuffled/v1",
            "case_index": index,
            "paired_view_group_id": base["paired_view_group_id"],
            "candidate_bank_id": base["candidate_bank_id"],
            "candidate_bank_receipt_sha256": base["candidate_bank_receipt_sha256"],
            "ordered_candidate_ids": base["ordered_candidate_ids"],
            "truth_candidate_id": base["truth_candidate_id"],
            "target": target,
            "scores": scores,
            "ranking": _rank_payload(scores, base["candidate_bank_receipt"], target),
            "mapping": "target=(case_index+17)%64; candidate bank and order unchanged",
        }
        record["shuffled_payload_sha256"] = canonical_payload_sha256(record)
        _atomic_json(folder / "shuffled" / f"case-{index:03d}.json", record)
        shuffled.append(record)
    controls = {}
    for name in CONTROL_SCHEMAS:
        references = [
            _write_control_sidecar(folder, _strict_control(name, primary[index]))
            for index in range(64)
        ]
        controls[name] = _control_record(name, references)
    summary = semantic_gate_summary(primary, shuffled, exact_controls=controls)
    source_hashes = _source_hashes()
    support_sha256, render_receipt, candidate_receipt = _prepared_receipts()
    resolved_config = {
        "schema": runner.RUNNER_SCHEMA,
        "source_commit": "a" * 40,
        "repository": {
            "branch": "codex/joint-registration",
            "upstream": "origin/codex/joint-registration",
            "head": "a" * 40,
            "upstream_head": "a" * 40,
            "worktree_clean": True,
        },
        "source_sha256": source_hashes,
        "checkout_source_sha256": copy.deepcopy(source_hashes),
        "source_hash_contract": copy.deepcopy(runner.SOURCE_HASH_CONTRACT),
        "preflight_sha256": source_hashes[
            "publication/arbitrary_plane_oracle_pose_ranking_preflight.yaml"
        ],
        "case_count": 64,
        "output_shape_h_w": [192, 256],
        "margin_u_v_um": [250.0, 250.0],
        "maximum_case_rejection_attempts": runner.MAXIMUM_CASE_ATTEMPTS,
        "case_root_seed": CASE_ROOT_SEED_HEX,
        "case_seed_algorithm": runner.CASE_SEED_ALGORITHM,
        "orientation_counts": runner.ORIENTATION_COUNTS,
        "outline_modes": list(OUTLINE_MODES),
        "memory_contract": {
            "maximum_live_candidate_banks": 3,
            "reason": "one preceding shuffled bank, one current bank, and one transient exact-replay bank; never all 64 banks",
            "candidate_or_synthetic_arrays_saved_to_json": False,
        },
        "atlas_assets": {
            "template": {
                "uri": runner.ATLAS_TEMPLATE_URI,
                "raw_sha256": runner.ATLAS_TEMPLATE_SHA256,
                "decoder": "pynrrd 1.1.3",
                "index_order": "F",
            },
            "annotation": {
                "uri": runner.ATLAS_ANNOTATION_URI,
                "raw_sha256": runner.ATLAS_ANNOTATION_SHA256,
                "decoder": "pynrrd 1.1.3",
                "index_order": "F",
            },
        },
        "shuffled_mapping": "(i+17)%64",
        "animal_id": None,
        "specimen_id": None,
        "experiment_id": None,
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
        "deepslice_ground_truth_accessed": False,
        "real_lab_histology_accessed": False,
        "final_test_animals_accessed": False,
        "environment": {
            "python": "3.11 fixture",
            "numpy": "fixture",
            "scipy": "fixture",
            "torch": "fixture",
            "pynrrd": "1.1.3",
        },
    }
    resolved_config["resolved_config_sha256"] = canonical_payload_sha256(resolved_config)
    _atomic_json(folder / "resolved_config.json", resolved_config)
    result = {
        "schema": runner.RUNNER_SCHEMA,
        "resolved_config": resolved_config,
        "support_index_sha256": support_sha256,
        "prepared_render_context_sha256": canonical_payload_sha256(render_receipt),
        "prepared_candidate_annotation_context_sha256": canonical_payload_sha256(
            candidate_receipt
        ),
        "prepared_render_asset_receipt": render_receipt,
        "prepared_candidate_annotation_receipt": candidate_receipt,
        "primary_case_payload_sha256": [item["case_payload_sha256"] for item in primary],
        "shuffled_case_payload_sha256": [item["shuffled_payload_sha256"] for item in shuffled],
        "exact_controls": controls,
        "semantic_gate": summary,
        "interpretation": runner.INTERPRETATION,
    }
    result["result_payload_sha256"] = canonical_payload_sha256(result)
    _atomic_json(folder / "result.json", result)


def _replace_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _rehash_config_and_result(folder, config, result):
    config["resolved_config_sha256"] = canonical_payload_sha256(
        {key: value for key, value in config.items() if key != "resolved_config_sha256"}
    )
    result["resolved_config"] = copy.deepcopy(config)
    result["result_payload_sha256"] = canonical_payload_sha256(
        {key: value for key, value in result.items() if key != "result_payload_sha256"}
    )
    _replace_json(folder / "resolved_config.json", config)
    _replace_json(folder / "result.json", result)


def test_written_sidecar_verifier_recomputes_hashes_and_gate_without_atlas_access(tmp_path):
    _written_fixture(tmp_path)
    assert runner._verify_written_result(
        tmp_path, _fixture_support()
    )["semantic_gate"]["passed"]
    mask_path = tmp_path / "masks" / "case-000.bin"
    original_mask = mask_path.read_bytes()
    changed_mask = bytearray(original_mask)
    changed_mask[0] ^= 1
    mask_path.write_bytes(changed_mask)
    with pytest.raises(ValueError, match="mask binary"):
        runner._verify_written_result(tmp_path, _fixture_support())
    mask_path.write_bytes(original_mask)
    control_path = tmp_path / "controls" / "exact_replay" / "case-000.json"
    original_control = control_path.read_bytes()
    changed_control = json.loads(original_control)
    changed_control["passed"] = False
    control_path.write_text(json.dumps(changed_control), encoding="utf-8")
    with pytest.raises(ValueError, match="control evidence file hash"):
        runner._verify_written_result(tmp_path, _fixture_support())
    control_path.write_bytes(original_control)
    path = tmp_path / "primary" / "case-000.json"
    changed = json.loads(path.read_text())
    changed["scores"]["semantic"][0] = 0.0
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="primary case payload hash"):
        runner._verify_written_result(tmp_path, _fixture_support())


def test_written_verifier_rejects_rank_inconsistency_and_any_stale_file(tmp_path):
    _written_fixture(tmp_path)
    primary_path = tmp_path / "primary" / "case-000.json"
    result_path = tmp_path / "result.json"
    primary = json.loads(primary_path.read_text())
    result = json.loads(result_path.read_text())
    primary["ranking"]["true_rank"] = 2
    primary["case_payload_sha256"] = canonical_payload_sha256(
        {key: value for key, value in primary.items() if key != "case_payload_sha256"}
    )
    result["primary_case_payload_sha256"][0] = primary["case_payload_sha256"]
    exact_path = tmp_path / "controls" / "exact_replay" / "case-000.json"
    exact = json.loads(exact_path.read_text())
    exact["case_payload_sha256"] = primary["case_payload_sha256"]
    exact["replayed_case_payload_sha256"] = primary["case_payload_sha256"]
    exact_bytes = (json.dumps(exact, sort_keys=True, indent=2) + "\n").encode()
    exact_path.write_bytes(exact_bytes)
    references = result["exact_controls"]["exact_replay"]["evidence"]["case_evidence"]
    references[0]["payload_sha256"] = canonical_payload_sha256(exact)
    references[0]["file_sha256"] = hashlib.sha256(exact_bytes).hexdigest()
    result["exact_controls"]["exact_replay"] = _control_record("exact_replay", references)
    result["result_payload_sha256"] = canonical_payload_sha256(
        {key: value for key, value in result.items() if key != "result_payload_sha256"}
    )
    primary_path.write_text(json.dumps(primary), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="saved primary ranking"):
        runner._verify_written_result(tmp_path, _fixture_support())

    primary["ranking"]["true_rank"] = 1
    primary["case_payload_sha256"] = canonical_payload_sha256(
        {key: value for key, value in primary.items() if key != "case_payload_sha256"}
    )
    result["primary_case_payload_sha256"][0] = primary["case_payload_sha256"]
    result["result_payload_sha256"] = canonical_payload_sha256(
        {key: value for key, value in result.items() if key != "result_payload_sha256"}
    )
    primary_path.write_text(json.dumps(primary), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "stale.tmp").write_text("stale")
    with pytest.raises(ValueError, match="exact frozen file set"):
        runner._verify_written_result(tmp_path, _fixture_support())


def test_written_verifier_rejects_rehashed_config_result_and_context_substitution(tmp_path):
    _written_fixture(tmp_path)
    config_path = tmp_path / "resolved_config.json"
    result_path = tmp_path / "result.json"
    original_config = config_path.read_bytes()
    original_result = result_path.read_bytes()

    config = json.loads(original_config)
    result = json.loads(original_result)
    del config["shuffled_mapping"]
    _rehash_config_and_result(tmp_path, config, result)
    with pytest.raises(ValueError, match="exact production keyset"):
        runner._verify_written_result(tmp_path, _fixture_support())

    config_path.write_bytes(original_config)
    result_path.write_bytes(original_result)
    config = json.loads(original_config)
    result = json.loads(original_result)
    config["source_sha256"][runner.SOURCE_RELATIVE_PATHS[0]] = _digest("substituted-source")
    _rehash_config_and_result(tmp_path, config, result)
    with pytest.raises(ValueError, match="recorded Git blob"):
        runner._verify_written_result(tmp_path, _fixture_support())

    config_path.write_bytes(original_config)
    result_path.write_bytes(original_result)
    result = json.loads(original_result)
    del result["interpretation"]
    result["result_payload_sha256"] = canonical_payload_sha256(
        {key: value for key, value in result.items() if key != "result_payload_sha256"}
    )
    _replace_json(result_path, result)
    with pytest.raises(ValueError, match="exact production schema"):
        runner._verify_written_result(tmp_path, _fixture_support())

    result_path.write_bytes(original_result)
    result = json.loads(original_result)
    replacement = _digest("coordinated-decoded-annotation-substitution")
    result["prepared_render_asset_receipt"]["annotation_decoded"]["array_sha256"] = replacement
    result["prepared_candidate_annotation_receipt"]["annotation"]["array_sha256"] = replacement
    result["prepared_render_context_sha256"] = canonical_payload_sha256(
        result["prepared_render_asset_receipt"]
    )
    result["prepared_candidate_annotation_context_sha256"] = canonical_payload_sha256(
        result["prepared_candidate_annotation_receipt"]
    )
    result["result_payload_sha256"] = canonical_payload_sha256(
        {key: value for key, value in result.items() if key != "result_payload_sha256"}
    )
    _replace_json(result_path, result)
    with pytest.raises(ValueError, match="not bound to the frozen source and atlas contexts"):
        runner._verify_written_result(tmp_path, _fixture_support())


def test_written_verifier_rejects_rehashed_semantically_empty_control(tmp_path):
    _written_fixture(tmp_path)
    path = tmp_path / "controls" / "exact_replay" / "case-000.json"
    item = {"control": "exact_replay", "case_index": 0, "passed": True}
    _replace_json(path, item)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    references = result["exact_controls"]["exact_replay"]["evidence"]["case_evidence"]
    references[0]["payload_sha256"] = canonical_payload_sha256(item)
    references[0]["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["exact_controls"]["exact_replay"] = _control_record("exact_replay", references)
    result["result_payload_sha256"] = canonical_payload_sha256(
        {key: value for key, value in result.items() if key != "result_payload_sha256"}
    )
    _replace_json(result_path, result)
    with pytest.raises(ValueError, match="strict per-control schema"):
        runner._verify_written_result(tmp_path, _fixture_support())
