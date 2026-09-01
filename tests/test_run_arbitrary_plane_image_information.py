import copy
import hashlib
import json

import numpy as np
import pytest

import training.run_arbitrary_plane_image_information as runner


def test_frozen_schedules_permutation_and_top_level_schemas_are_exact():
    assert runner.CASE_COUNT == 64
    assert runner.CANDIDATE_COUNT == 40
    assert runner.OUTPUT_SHAPE == (192, 256)
    assert runner.OUTLINE_MODES == (
        "accurate-outline-black-exterior",
        "imperfect-outline-black-exterior",
        "absent-outline-acquired-background",
    )

    native = [
        (descriptor, domain, outline)
        for outline in runner.OUTLINE_MODES
        for descriptor, domain in (
            ("MIND", "core"),
            ("MIND", "context"),
            ("constant-within-support-MIND-null", "core"),
            ("constant-within-support-MIND-null", "context"),
            ("support-penalized-MIND", "core"),
            ("support-penalized-MIND", "context"),
            ("HOG", "core"),
            ("HOG", "context"),
            ("HOG", "boundary_ring"),
            ("normalized-gradient-like", "core"),
            ("normalized-gradient-like", "context"),
            ("normalized-gradient-like", "boundary_ring"),
        )
    ]
    shuffled = [
        (descriptor, domain, outline)
        for outline in runner.OUTLINE_MODES
        for descriptor, domain in (
            ("MIND", "core"),
            ("MIND", "context"),
            ("constant-within-support-MIND-null", "core"),
            ("constant-within-support-MIND-null", "context"),
        )
    ]
    assert runner.NATIVE_SLOT_SCHEDULE == tuple(native)
    assert runner.SHUFFLED_SLOT_SCHEDULE == tuple(shuffled)
    assert len(native) + 1 == 37
    assert len(shuffled) == 12

    expected_permutation = tuple((7 * index + 3) % 40 for index in range(40))
    assert runner.PERMUTATION == expected_permutation
    assert sorted(runner.PERMUTATION) == list(range(40))
    assert all(
        runner.INVERSE_PERMUTATION[runner.PERMUTATION[index]] == index
        for index in range(40)
    )

    assert runner.KEYSETS["resolved_config"] == {
        "schema", "preflight_sha256", "repository", "frozen_semantic_input",
        "atlas_assets", "descriptor_constants", "case_and_shuffle_contract",
        "environment", "model_independence", "data_access", "source_sha256",
        "resolved_config_sha256",
    }
    assert runner.KEYSETS["primary"] == {
        "schema", "case_index", "semantic_case_payload_sha256", "provenance",
        "frozen_replay", "target", "candidate_bank", "candidate_scalar_receipts",
        "score_domains", "outline_results", "payload_sha256",
    }
    assert runner.KEYSETS["shuffled"] == {
        "schema", "bank_case_index", "target_case_index", "bank_identity",
        "target_identity", "common_lattice_resampling", "score_domains",
        "outline_results", "payload_sha256",
    }
    assert runner.KEYSETS["control"] == {
        "schema", "case_index", "checks", "evidence_receipt_sha256", "payload_sha256",
    }
    assert runner.KEYSETS["global_controls"] == {
        "schema", "frozen_inventory_audit", "source_and_signature_audit",
        "affine_and_polarity_controls", "evidence_receipt_sha256", "payload_sha256",
    }
    assert runner.KEYSETS["result"] == {
        "schema", "interpretation", "resolved_config_sha256", "pre_result_inventory",
        "pre_result_inventory_sha256", "primary_case_payload_sha256",
        "shuffled_case_payload_sha256", "control_payload_sha256", "metrics", "gates",
        "data_access", "model_independence", "result_payload_sha256",
    }
    assert runner.MODEL_INDEPENDENCE == {
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
        "legacy_descriptor_dependencies": [],
        "initialization": "deterministic frozen case streams only; no learned initialization",
    }
    assert runner.DATA_ACCESS == {
        "allen_template_and_annotation": True,
        "synthetic_development": True,
        "deepslice_ground_truth": False,
        "real_lab_histology": False,
        "calibration_animals": False,
        "qualification_animals": False,
        "final_test_animals": False,
        "full_benchmark": False,
    }


def test_canonical_hash_and_exact_key_validator_reject_rehashed_schema_tampering():
    record = {key: None for key in runner.KEYSETS["result"]}
    record["result_payload_sha256"] = runner.canonical_payload_sha256(
        record, excluded_key="result_payload_sha256"
    )
    assert runner.validate_payload_keys(record, "result") is record
    assert record["result_payload_sha256"] == runner.canonical_payload_sha256(
        {key: value for key, value in record.items() if key != "result_payload_sha256"}
    )

    for changed in (
        {key: value for key, value in record.items() if key != "gates"},
        {**record, "post_hoc_metric": 1.0},
    ):
        changed["result_payload_sha256"] = runner.canonical_payload_sha256(
            changed, excluded_key="result_payload_sha256"
        )
        with pytest.raises(ValueError, match="frozen schema"):
            runner.validate_payload_keys(changed, "result")

    with pytest.raises(ValueError, match="frozen schema"):
        runner.validate_payload_keys(record, "unknown")
    with pytest.raises(ValueError, match="frozen schema"):
        runner.validate_payload_keys([], "result")
    with pytest.raises(ValueError):
        runner.canonical_payload_sha256({"nonfinite": np.nan})


def test_json_reader_requires_exact_canonical_bytes_and_rejects_duplicate_keys(tmp_path):
    payload = {"a": 1, "b": {"c": [2, 3]}}
    path = tmp_path / "payload.json"
    path.write_bytes(runner._canonical_bytes(payload))
    assert runner._read_json(path) == payload

    noncanonical = (
        json.dumps(payload, indent=2),
        runner._canonical_bytes(payload).decode("utf-8") + "\n",
        '{"b":{"c":[2,3]},"a":1}',
        '{"a":1,"a":1,"b":{"c":[2,3]}}',
    )
    for index, text in enumerate(noncanonical):
        changed = tmp_path / f"noncanonical-{index}.json"
        changed.write_text(text, encoding="utf-8", newline="")
        with pytest.raises(ValueError, match="canonical|duplicate"):
            runner._read_json(changed)


def test_score_blind_masks_finish_before_any_image_scorer_and_bind_thresholds(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("an image scorer was called during the score-blind mask pass")

    for name in (
        "score_mind_candidates",
        "score_support_penalized_mind_candidates",
        "score_hog_candidates",
        "score_ngf_candidates",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    shape = (32, 32)

    def replay(case_index):
        masks = {name: np.zeros(shape, dtype=bool) for name in runner.MASK_NAMES}
        masks["map_safe"][:] = True
        masks["visible"].flat[:800] = True
        masks["core"].flat[: 128 - case_index] = True
        masks["context"].flat[:512] = True
        masks["boundary_ring"].flat[:224] = True
        return {
            "semantic_record": {
                "case_payload_sha256": f"case-{case_index}",
                "paired_view_group_id": f"pair-{case_index}",
            },
            "target": {"pixel_pitch_um": 25.0},
            "masks": masks,
        }

    records = runner.build_score_blind_masks(replay, case_count=2)
    assert [record["case_index"] for record in records] == [0, 1]
    assert records[0]["pixel_counts"] == {
        "map_safe": 1024,
        "visible": 800,
        "core": 128,
        "context": 512,
        "boundary_ring": 224,
    }
    assert records[0]["passed"] is True
    assert records[1]["passed"] is False
    assert set(records[0]["mask_receipts"]) == set(runner.MASK_NAMES)
    assert all(
        receipt["storage"] == "not_persisted"
        for receipt in records[0]["mask_receipts"].values()
    )
    core = np.zeros(shape, dtype=bool)
    core.flat[:128] = True
    packed = np.packbits(core.reshape(-1, order="C"), bitorder="little").tobytes()
    header = json.dumps(
        {"dtype": "|b1", "shape": list(shape), "bitorder": "little"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert records[0]["mask_receipts"]["core"]["packed_payload_sha256"] == hashlib.sha256(
        header + packed
    ).hexdigest()
    tampered = copy.deepcopy(records[0])
    tampered["score"] = 1.0
    with pytest.raises(ValueError, match="frozen schema"):
        runner.validate_payload_keys(tampered, "case_mask_record")
    with pytest.raises(ValueError, match="frozen replay"):
        runner.build_score_blind_masks(replay)


def test_expected_success_tree_and_conservative_ranking_are_frozen():
    files = runner.expected_output_files()
    assert len(files) == 515
    assert {"resolved_config.json", "global_controls.json", "result.json"} <= files
    assert {f"primary/case-{index:03d}.json" for index in range(64)} <= files
    assert {f"shuffled/case-{index:03d}.json" for index in range(64)} <= files
    assert {f"controls/case-{index:03d}.json" for index in range(64)} <= files
    assert all(not path.endswith(".tmp") for path in files)

    ids = [f"c{index:02d}" for index in range(40)]
    scores = np.linspace(0.8, 0.0, 40)
    scores[1] = scores[0] - 0.5e-12
    ranking, pose_errors = runner.rank_landscape(scores, ids, ids[0])
    assert ranking["tied_maximum_candidate_ids"] == ["c00", "c01"]
    assert ranking["true_rank"] == 2
    assert ranking["top1"] is False
    assert ranking["selected_candidate_id"] is None
    assert pose_errors is None


def _toy_mind_result(scores):
    return {
        "scores": np.asarray(scores, dtype=np.float64),
        "target_vbar": 1.0,
        "candidate_vbar": np.ones(40, dtype=np.float64),
    }


def _toy_slot(monkeypatch, *, case_index, bank_case_index, target_case_index):
    scores = np.linspace(1.0, 0.0, 40, dtype=np.float64)
    monkeypatch.setattr(runner, "_call_scorer", lambda *_args, **_kwargs: _toy_mind_result(scores))
    ids = [f"c{index:02d}" for index in range(40)]
    images = np.arange(40, dtype=np.float64).reshape(40, 1, 1)
    slot = runner.score_landscape(
        case_index=case_index,
        bank_case_index=bank_case_index,
        target_case_index=target_case_index,
        outline_mode=runner.OUTLINE_MODES[2],
        descriptor="MIND",
        domain="core",
        domain_mask=np.ones((1, 1), dtype=bool),
        domain_mask_receipt_sha256="d" * 64,
        target_image=np.ones((1, 1), dtype=np.float64),
        candidate_images=images,
        pixel_pitch_um=25.0,
        ordered_candidate_ids=ids,
        truth_candidate_id=ids[0],
    )
    return slot, scores, ids, images


def test_native_and_shuffled_slots_use_the_nullable_index_triad(monkeypatch):
    native, *_ = _toy_slot(
        monkeypatch, case_index=7, bank_case_index=None, target_case_index=None
    )
    shuffled, *_ = _toy_slot(
        monkeypatch, case_index=None, bank_case_index=7, target_case_index=24
    )
    assert (native["case_index"], native["bank_case_index"], native["target_case_index"]) == (
        7, None, None
    )
    assert (
        shuffled["case_index"],
        shuffled["bank_case_index"],
        shuffled["target_case_index"],
    ) == (None, 7, 24)
    for invalid in ((7, 7, 24), (None, 7, None), (7, None, 24), (None, None, None)):
        with pytest.raises(ValueError, match="slot indices"):
            _toy_slot(
                monkeypatch,
                case_index=invalid[0],
                bank_case_index=invalid[1],
                target_case_index=invalid[2],
            )


def test_landscape_controls_bind_permuted_ids_inverse_ranking_and_signed_zero(monkeypatch):
    slot, scores, ids, images = _toy_slot(
        monkeypatch, case_index=3, bank_case_index=None, target_case_index=None
    )
    domain = np.ones((1, 1), dtype=bool)
    target = np.ones((1, 1), dtype=np.float64)

    def exact_scorer(_descriptor, _domain, _target, candidates, *_args, **_kwargs):
        indices = np.asarray(candidates)[:, 0, 0].astype(int)
        return _toy_mind_result(scores[indices])

    monkeypatch.setattr(runner, "_call_scorer", exact_scorer)
    exact = runner._landscape_control(
        slot,
        target_image=target,
        candidate_images=images,
        domain_mask=domain,
        pixel_pitch_um=25.0,
        ordered_candidate_ids=ids,
        truth_candidate_id=ids[0],
        candidate_support=None,
        target_visible=None,
        padding_value=0.0,
    )
    assert exact["chunks"]["byte_identical"] is True
    assert exact["permutation"]["passed"] is True
    assert exact["passed"] is True

    def signed_zero_scorer(_descriptor, _domain, _target, candidates, *_args, **kwargs):
        indices = np.asarray(candidates)[:, 0, 0].astype(int)
        values = scores[indices].copy()
        if kwargs["chunk_size"] == 1 and np.array_equal(indices, np.arange(40)):
            values[-1] = -0.0
        return _toy_mind_result(values)

    monkeypatch.setattr(runner, "_call_scorer", signed_zero_scorer)
    signed_zero = runner._landscape_control(
        slot,
        target_image=target,
        candidate_images=images,
        domain_mask=domain,
        pixel_pitch_um=25.0,
        ordered_candidate_ids=ids,
        truth_candidate_id=ids[0],
        candidate_support=None,
        target_visible=None,
        padding_value=0.0,
    )
    assert np.array_equal(scores, np.where(np.arange(40) == 39, -0.0, scores))
    assert runner._float64_vector_bytes(scores) != runner._float64_vector_bytes(
        np.where(np.arange(40) == 39, -0.0, scores)
    )
    assert signed_zero["chunks"]["byte_identical"] is False
    assert signed_zero["permutation"]["passed"] is True
    assert signed_zero["passed"] is False


def _prelaunch_mask_record(case_index, *, core_count=128):
    receipt = {
        "dtype": "|b1",
        "shape": list(runner.OUTPUT_SHAPE),
        "bitorder": "little",
        "bit_count": int(np.prod(runner.OUTPUT_SHAPE)),
        "byte_count": int(np.prod(runner.OUTPUT_SHAPE)) // 8,
        "packed_payload_sha256": f"{case_index:064x}",
        "storage": "not_persisted",
    }
    return runner.validate_payload_keys(
        {
            "case_index": case_index,
            "semantic_case_payload_sha256": f"{case_index + 64:064x}",
            "paired_view_group_id": f"pair-{case_index}",
            "pixel_pitch_um": 25.0,
            "mask_receipts": {name: copy.deepcopy(receipt) for name in runner.MASK_NAMES},
            "pixel_counts": {
                "map_safe": 1024,
                "visible": 800,
                "core": core_count,
                "context": 512,
                "boundary_ring": 224,
            },
            "passed": core_count >= 128,
        },
        "case_mask_record",
    )


def _prelaunch_config():
    return {
        "resolved_config_sha256": "c" * 64,
        "frozen_semantic_input": {"result_payload_sha256": "f" * 64},
        "repository": {
            "execution_commit": "e" * 40,
            "origin_commit": "e" * 40,
            "branch": "codex/joint-registration",
            "worktree_clean": True,
        },
        "preflight_sha256": "p" * 64,
        "environment": {"python": "fixture"},
        "source_sha256": [],
    }


def test_prelaunch_failure_is_score_blind_one_file_and_mutually_exclusive(tmp_path):
    records = [_prelaunch_mask_record(index) for index in range(64)]
    records[37] = _prelaunch_mask_record(37, core_count=127)
    config = _prelaunch_config()
    failed = tmp_path / "failed"
    success = tmp_path / "success"
    receipt = runner.write_prelaunch_failure(failed, success, config, records)
    assert not success.exists()
    assert [path.relative_to(failed).as_posix() for path in failed.rglob("*") if path.is_file()] == [
        "prelaunch_failure.json"
    ]
    assert receipt["status"] == "failed_before_scoring"
    assert receipt["score_blind_evidence"] == {
        "all_64_masks_built": True,
        "frozen_replay_passed": True,
        "candidate_scalar_render_count": 0,
        "descriptor_call_count": 0,
        "score_landscape_count": 0,
        "success_output_created": False,
    }
    assert receipt["failures"] == [
        {
            "case_index": 37,
            "domain": "core",
            "observed_pixel_count": 127,
            "minimum_required_pixels": 128,
        }
    ]
    assert receipt["failure_payload_sha256"] == runner.canonical_payload_sha256(
        receipt, excluded_key="failure_payload_sha256"
    )

    with pytest.raises(ValueError, match="all 64 cases"):
        runner.write_prelaunch_failure(tmp_path / "partial", tmp_path / "unused", config, records[:-1])
    assert not (tmp_path / "partial").exists()
    preexisting_success = tmp_path / "preexisting-success"
    preexisting_success.mkdir()
    with pytest.raises(FileExistsError, match="both be fresh"):
        runner.write_prelaunch_failure(
            tmp_path / "must-not-exist", preexisting_success, config, records
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_public_prelaunch_verifier_replays_authority_and_rejects_rehashed_tamper(
    monkeypatch, tmp_path
):
    records = [_prelaunch_mask_record(index) for index in range(runner.CASE_COUNT)]
    records[37] = _prelaunch_mask_record(37, core_count=127)
    config = _toy_resolved_config()
    failed_output = tmp_path / "failed"
    success_output = tmp_path / "success"
    receipt = runner.write_prelaunch_failure(
        failed_output, success_output, config, records
    )
    receipt_path = failed_output / "prelaunch_failure.json"
    repository = {
        "branch": runner.EXPECTED_BRANCH,
        "upstream": f"origin/{runner.EXPECTED_BRANCH}",
        "head": config["repository"]["execution_commit"],
        "upstream_head": config["repository"]["origin_commit"],
        "worktree_clean": True,
    }
    calls = {"repository": 0, "authenticate": 0, "replay": 0}

    def repository_state():
        calls["repository"] += 1
        return copy.deepcopy(repository)

    def authenticate(source_records):
        calls["authenticate"] += 1
        assert source_records == config["source_sha256"]
        challenging = set(runner.FROZEN_POOLED_MEMBERSHIP["challenging_appearance"])
        damaged = set(runner.FROZEN_POOLED_MEMBERSHIP["damaged"])
        return {"result_payload_sha256": "s" * 64}, [
            {
                "case_index": index,
                "reporting_strata": {
                    "appearance_family": (
                        "label-conditioned" if index in challenging else "ordinary"
                    ),
                    "damage_event_count": 1 if index in damaged else 0,
                },
            }
            for index in range(runner.CASE_COUNT)
        ]

    def build_masks(replay):
        calls["replay"] += 1
        assert replay(0) == {"case_index": 0}
        assert replay(runner.CASE_COUNT - 1) == {
            "case_index": runner.CASE_COUNT - 1
        }
        return copy.deepcopy(records)

    monkeypatch.setattr(runner, "repository_state", repository_state)
    monkeypatch.setattr(
        runner, "_source_hash_receipts", lambda _repository: config["source_sha256"]
    )
    monkeypatch.setattr(runner, "_authenticate_frozen_semantic_output", authenticate)
    monkeypatch.setattr(runner, "load_allen_contexts", lambda: ({}, {}, {}))
    monkeypatch.setattr(runner, "_validate_contexts", lambda *_args: None)
    monkeypatch.setattr(runner, "_resolved_config", lambda *_args: config)
    monkeypatch.setattr(
        runner,
        "replay_frozen_case",
        lambda index, *_args: {"case_index": index},
    )
    monkeypatch.setattr(runner, "build_score_blind_masks", build_masks)

    assert runner.verify_prelaunch_failure(failed_output, success_output) == receipt
    assert calls == {"repository": 2, "authenticate": 2, "replay": 1}

    changed = copy.deepcopy(receipt)
    changed["case_mask_records"][37]["pixel_counts"]["core"] = 126
    changed["failures"][0]["observed_pixel_count"] = 126
    changed["failure_payload_sha256"] = runner.canonical_payload_sha256(
        changed, excluded_key="failure_payload_sha256"
    )
    receipt_path.write_bytes(runner._canonical_bytes(changed))
    with pytest.raises(ValueError, match="does not replay exactly"):
        runner.verify_prelaunch_failure(failed_output, success_output)

    changed = copy.deepcopy(receipt)
    changed["case_mask_records"][37]["mask_receipts"]["core"][
        "packed_payload_sha256"
    ] = "f" * 64
    changed["failure_payload_sha256"] = runner.canonical_payload_sha256(
        changed, excluded_key="failure_payload_sha256"
    )
    receipt_path.write_bytes(runner._canonical_bytes(changed))
    with pytest.raises(ValueError, match="does not replay exactly"):
        runner.verify_prelaunch_failure(failed_output, success_output)

    receipt_path.write_bytes(runner._canonical_bytes(receipt))
    extra = failed_output / ".stale.tmp"
    extra.write_bytes(b"partial")
    with pytest.raises(ValueError, match="exactly one"):
        runner.verify_prelaunch_failure(failed_output, success_output)
    extra.unlink()

    receipt_path.write_bytes(runner._canonical_bytes(receipt) + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        runner.verify_prelaunch_failure(failed_output, success_output)


def test_runner_authority_abort_creates_nothing_and_mask_failure_never_enters_scoring(
    monkeypatch, tmp_path
):
    success = tmp_path / "success"
    failed = tmp_path / "failed"
    monkeypatch.setattr(
        runner,
        "repository_state",
        lambda: (_ for _ in ()).throw(RuntimeError("authority-abort")),
    )
    with pytest.raises(RuntimeError, match="authority-abort"):
        runner.run_image_information(success, failed)
    assert not success.exists() and not failed.exists()

    records = [_prelaunch_mask_record(index) for index in range(64)]
    records[12] = _prelaunch_mask_record(12, core_count=0)
    challenging = set(runner.FROZEN_POOLED_MEMBERSHIP["challenging_appearance"])
    damaged = set(runner.FROZEN_POOLED_MEMBERSHIP["damaged"])
    frozen_primary = [
        {
            "case_index": index,
            "reporting_strata": {
                "appearance_family": (
                    "label-conditioned" if index in challenging else "ordinary"
                ),
                "damage_event_count": 1 if index in damaged else 0,
            },
        }
        for index in range(runner.CASE_COUNT)
    ]
    monkeypatch.setattr(runner, "repository_state", lambda: {"head": "e" * 40})
    monkeypatch.setattr(runner, "_source_hash_receipts", lambda _repository: [])
    monkeypatch.setattr(
        runner,
        "_authenticate_frozen_semantic_output",
        lambda _sources: ({}, frozen_primary),
    )
    monkeypatch.setattr(runner, "load_allen_contexts", lambda: ({}, {}, {}))
    monkeypatch.setattr(runner, "_validate_contexts", lambda *_args: None)
    monkeypatch.setattr(runner, "_resolved_config", lambda *_args: _prelaunch_config())
    monkeypatch.setattr(runner, "build_score_blind_masks", lambda _replay: records)
    verification_calls = []
    monkeypatch.setattr(
        runner,
        "verify_prelaunch_failure",
        lambda failed_output, success_output: (
            verification_calls.append((failed_output, success_output))
            or runner._read_strict_json(failed_output / "prelaunch_failure.json")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_stream_cases",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scoring was entered after a score-blind mask failure")
        ),
    )
    receipt = runner.run_image_information(success, failed)
    assert receipt["status"] == "failed_before_scoring"
    assert verification_calls == [(failed, success)]
    assert not success.exists()
    assert [path.name for path in failed.iterdir()] == ["prelaunch_failure.json"]


def test_inventory_includes_hidden_and_temporary_files_for_exact_tree_rejection(tmp_path):
    (tmp_path / "expected.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".stale.tmp").write_text("partial", encoding="utf-8")
    (tmp_path / ".hidden").write_text("extra", encoding="utf-8")
    assert [item["path"] for item in runner._inventory(tmp_path)] == [
        ".hidden",
        ".stale.tmp",
        "expected.json",
    ]


def test_output_root_guard_rejects_equal_nested_and_protected_paths(tmp_path):
    ordinary = tmp_path / "ordinary"
    failed = tmp_path / "failed"
    assert runner.guard_output_roots(ordinary, failed) == (
        ordinary.resolve(),
        failed.resolve(),
    )
    assert runner.guard_output_roots(ordinary, None) == (ordinary.resolve(), None)
    assert runner.guard_output_roots(None, failed) == (None, failed.resolve())

    for first, second in (
        (ordinary, ordinary),
        (ordinary, ordinary / "failed-child"),
        (ordinary / "success-child", ordinary),
    ):
        with pytest.raises(ValueError, match="must not overlap"):
            runner.guard_output_roots(first, second)

    source_path = runner.ROOT / runner.RUNNER_RELATIVE_PATH
    for protected in (
        runner.FROZEN_SEMANTIC_OUTPUT,
        runner.FROZEN_SEMANTIC_OUTPUT / "nested-output",
        runner.FROZEN_SEMANTIC_OUTPUT.parent,
        source_path,
        source_path.parent,
    ):
        with pytest.raises(ValueError, match="overlaps frozen semantic"):
            runner.guard_output_roots(protected, failed)
        with pytest.raises(ValueError, match="overlaps frozen semantic"):
            runner.guard_output_roots(ordinary, protected)


def test_score_vectors_are_finite_unit_interval_and_domain_hash_binds_nested_mask_receipt():
    assert np.array_equal(
        runner._unit_score_vector({"scores": np.linspace(0.0, 1.0, 40)}),
        np.linspace(0.0, 1.0, 40),
    )
    for value in (-np.finfo(np.float64).eps, 1.0 + np.finfo(np.float64).eps, np.nan):
        scores = np.full(40, 0.5)
        scores[7] = value
        with pytest.raises(ValueError, match=r"\[0,1\]"):
            runner._unit_score_vector({"scores": scores})

    masks = {name: np.ones((3, 4), dtype=bool) for name in runner.MASK_NAMES}
    _, domains = runner._score_domain_records(5, masks)
    domain = domains["core"]
    observed = runner._domain_mask_receipt_sha256(domain)
    assert observed == runner.canonical_payload_sha256(domain["mask_receipt"])
    assert observed != runner.canonical_payload_sha256(domain)
    changed = copy.deepcopy(domain)
    changed["mask_receipt"]["relative_path"] = "masks/substituted.bin"
    assert runner._domain_mask_receipt_sha256(changed) != observed


def test_affine_constant_null_transforms_native_image_before_support_flattening(monkeypatch):
    base = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float64)
    original = np.stack([base + index * 1.0e-4 for index in range(40)])
    supports = np.ones(original.shape, dtype=bool)
    supports[:, 1, 2] = False
    flattened_and_means = [
        runner.constant_within_support_null(image, support)
        for image, support in zip(original, supports, strict=True)
    ]
    flattened = np.stack([item[0] for item in flattened_and_means])
    means = [item[1] for item in flattened_and_means]
    scores = np.linspace(1.0, 0.0, 40)
    ids = [f"c{index:02d}" for index in range(40)]
    monkeypatch.setattr(runner, "_call_scorer", lambda *_args, **_kwargs: _toy_mind_result(scores))
    slot = runner.score_landscape(
        case_index=0,
        bank_case_index=None,
        target_case_index=None,
        outline_mode=runner.OUTLINE_MODES[2],
        descriptor="constant-within-support-MIND-null",
        domain="core",
        domain_mask=np.ones(base.shape, dtype=bool),
        domain_mask_receipt_sha256="m" * 64,
        target_image=base,
        candidate_images=flattened,
        pixel_pitch_um=25.0,
        ordered_candidate_ids=ids,
        truth_candidate_id=ids[0],
        supported_means=means,
    )

    seen = []

    def capture(_descriptor, _domain, _target, candidates, *_args, **_kwargs):
        seen.append(np.asarray(candidates).copy())
        return _toy_mind_result(scores)

    monkeypatch.setattr(runner, "_call_scorer", capture)
    control = runner._affine_slot_control(
        slot,
        target_image=base,
        candidate_images=flattened,
        domain_mask=np.ones(base.shape, dtype=bool),
        pixel_pitch_um=25.0,
        ordered_candidate_ids=ids,
        truth_candidate_id=ids[0],
        candidate_support=None,
        target_visible=None,
        original_candidate_images=original,
        constant_null_support=supports,
    )
    expected = np.stack(
        [
            runner.constant_within_support_null(0.7 * image + 0.1, support)[0]
            for image, support in zip(original, supports, strict=True)
        ]
    )
    wrong_order = 0.7 * flattened + 0.1
    assert np.array_equal(seen[0], expected)
    assert not np.array_equal(expected, wrong_order)
    assert control["passed"] is True


def _toy_dewarp_runtime(monkeypatch):
    shape = (33, 35)
    y, x = np.indices(shape, dtype=np.float32)
    arrays = {
        "model_input_image": ((3 * y + 5 * x) % 17).astype(np.float32) / 16.0,
        "fixed_to_source_map": np.stack((x, y)),
        "source_map_domain_mask": np.ones(shape, dtype=bool),
        "fixed_map_domain_mask": np.ones(shape, dtype=bool),
        "source_valid_correspondence_mask": np.ones(shape, dtype=bool),
        "fixed_valid_correspondence_mask": np.ones(shape, dtype=bool),
    }
    arrays["source_map_domain_mask"][0, 0] = False
    descendant = {"arrays": arrays}
    monkeypatch.setattr(runner, "_verified_descendant", lambda *_args, **_kwargs: descendant)
    _, dewarp = runner._target_dewarp_record(descendant)
    masks = runner.target_score_masks(
        arrays["fixed_to_source_map"],
        arrays["source_map_domain_mask"],
        arrays["fixed_map_domain_mask"],
        arrays["source_valid_correspondence_mask"],
        arrays["fixed_valid_correspondence_mask"],
        100.0,
    )
    domains, _ = runner._score_domain_records(0, masks)
    primary = {
        "target": {"pixel_pitch_um": 100.0},
        "outline_results": [
            {"outline_mode": mode, "target_dewarp": copy.deepcopy(dewarp)}
            for mode in runner.OUTLINE_MODES
        ],
        "score_domains": domains,
    }
    return {
        "record": primary,
        "replayed": {
            "semantic_record": {"outline_descendants": [{}, {}, {}]},
            "parent": {},
            "case_index": 0,
            "synthetic_seed": 1,
        },
    }


def test_dewarp_and_four_corner_mask_control_recomputes_evidence_and_rejects_tamper(monkeypatch):
    runtime = _toy_dewarp_runtime(monkeypatch)
    evidence, passed = runner._recompute_dewarp_and_domain_evidence(runtime, {})
    assert passed is True
    assert len(evidence["outlines"]) == 3

    fixed_map = copy.deepcopy(runtime)
    fixed_map["record"]["outline_results"][0]["target_dewarp"][
        "fixed_to_source_map_receipt"
    ]["array_sha256"] = "0" * 64
    assert runner._recompute_dewarp_and_domain_evidence(fixed_map, {})[1] is False

    dewarp = copy.deepcopy(runtime)
    dewarp["record"]["outline_results"][1]["target_dewarp"][
        "dewarped_float64_receipt"
    ]["array_sha256"] = "1" * 64
    assert runner._recompute_dewarp_and_domain_evidence(dewarp, {})[1] is False

    four_corner = copy.deepcopy(runtime)
    map_safe = next(
        item for item in four_corner["record"]["score_domains"] if item["domain"] == "map_safe"
    )
    map_safe["mask_receipt"]["array_sha256"] = "2" * 64
    assert runner._recompute_dewarp_and_domain_evidence(four_corner, {})[1] is False


def test_signature_audit_covers_every_scorer_including_support_penalized_mind():
    records, passed = runner._scorer_signature_evidence()
    assert passed is True
    assert {item["function"] for item in records} == {
        "score_mind_candidates",
        "score_support_penalized_mind_candidates",
        "score_hog_candidates",
        "score_ngf_candidates",
    }
    assert all(item["passed"] is True and item["forbidden_matches"] == [] for item in records)


def _gate_metrics():
    primary, shuffled = _aggregate_fixture()
    metrics = runner.aggregate_metrics(
        primary,
        shuffled,
        {
            name: list(indices)
            for name, indices in runner.FROZEN_POOLED_MEMBERSHIP.items()
        },
    )
    context_successes = dict(zip(runner.OUTLINE_MODES, (33, 33, 39), strict=True))
    for item in metrics["native_slot_summaries"]:
        if item["descriptor"] == "MIND" and item["domain"] == "context":
            item["top1_success_count"] = context_successes[item["outline_mode"]]
        if (
            item["descriptor"] == "MIND"
            and item["domain"] == "context"
            and item["outline_mode"] == runner.OUTLINE_MODES[2]
        ):
            item["wilson_95"] = [0.45, 0.75]
            item["mean_reciprocal_rank"] = 0.70
            item["median_true_rank"] = 1.0
            item["median_truth_versus_decoy_win_fraction"] = 0.90
        if item["domain"] == "core" and item["outline_mode"] == runner.OUTLINE_MODES[2]:
            if item["descriptor"] == "MIND":
                item["top1_success_count"] = 32
                item["mean_reciprocal_rank"] = 0.65
            elif item["descriptor"] == "constant-within-support-MIND-null":
                item["top1_success_count"] = 25
    for item in metrics["shuffled_slot_summaries"]:
        if item["descriptor"] == "MIND" and item["domain"] == "context":
            item["top1_success_count"] = 6
            item["mean_reciprocal_rank"] = 0.15
    for item, count in zip(
        metrics["reporting_stratum_summaries"][:4], (5, 5, 5, 12), strict=True
    ):
        item["top1_success_count"] = count
    for item in metrics["pooled_safeguard_summaries"]:
        if item["stratum_type"] == "challenging_appearance":
            item["top1_success_count"] = 7 if "/MIND/" in item["endpoint"] else 5
        else:
            item["top1_success_count"] = 9 if "/MIND/" in item["endpoint"] else 7
        if "/MIND/" in item["endpoint"]:
            item["mean_reciprocal_rank"] = 0.45
    runner._validate_metrics_contract(metrics)
    return metrics


def _global_controls_fixture():
    return runner.validate_payload_keys(
        {
            "schema": "anatomy-tracker.arbitrary-plane-image-information-global-controls/v1",
            "frozen_inventory_audit": {"passed": True},
            "source_and_signature_audit": {"passed": True},
            "affine_and_polarity_controls": {"passed": True},
            "evidence_receipt_sha256": "e" * 64,
            "payload_sha256": "p" * 64,
        },
        "global_controls",
    )


def test_gate_slots_and_exact_boundary_decisions_are_predeclared_only():
    native_gate_slots = {
        slot
        for slot in runner.NATIVE_SLOT_SCHEDULE
        if runner._entered_gate(*slot, shuffled=False)
    }
    shuffled_gate_slots = {
        slot
        for slot in runner.SHUFFLED_SLOT_SCHEDULE
        if runner._entered_gate(*slot, shuffled=True)
    }
    assert native_gate_slots == {
        ("MIND", "context", outline) for outline in runner.OUTLINE_MODES
    } | {
        ("MIND", "core", runner.OUTLINE_MODES[2]),
        ("constant-within-support-MIND-null", "core", runner.OUTLINE_MODES[2]),
    }
    assert shuffled_gate_slots == {
        ("MIND", "context", outline) for outline in runner.OUTLINE_MODES
    }

    metrics = _gate_metrics()
    controls = _global_controls_fixture()
    gates = runner.evaluate_gates(metrics, controls)
    assert gates["passed"] is True and gates["decision"] == "PASS"
    assert len(gates["atomic_checks"]) == 31
    assert all(item["passed"] is True for item in gates["atomic_checks"])
    assert [item["gate_id"] for item in gates["atomic_checks"][:3]] == [
        "frozen-inventory-integrity",
        "source-signature-integrity",
        "all-case-controls-integrity",
    ]
    assert gates["global_controls_payload_sha256"] == controls["payload_sha256"]

    changed = copy.deepcopy(metrics)
    absent = next(
        item
        for item in changed["native_slot_summaries"]
        if item["descriptor"] == "MIND"
        and item["domain"] == "context"
        and item["outline_mode"] == runner.OUTLINE_MODES[2]
    )
    absent["top1_success_count"] = 38
    failed = runner.evaluate_gates(changed, controls)
    assert failed["passed"] is False and failed["decision"] == "FAIL"
    assert next(
        item for item in failed["atomic_checks"] if item["gate_id"] == "absent-context-top1-count"
    )["passed"] is False
    failed_controls = copy.deepcopy(controls)
    failed_controls["source_and_signature_audit"]["passed"] = False
    assert runner.evaluate_gates(metrics, failed_controls)["passed"] is False


def test_gate_verifier_rejects_rehashed_order_pointer_observation_and_decision_tamper():
    metrics = _gate_metrics()
    controls = _global_controls_fixture()
    gates = runner.evaluate_gates(metrics, controls)
    runner._validate_gates(gates, metrics, controls)

    changed = copy.deepcopy(gates)
    changed["atomic_checks"][0], changed["atomic_checks"][1] = (
        changed["atomic_checks"][1],
        changed["atomic_checks"][0],
    )
    with pytest.raises(ValueError, match="predeclared order"):
        runner._validate_gates(changed, metrics, controls)

    changed = copy.deepcopy(gates)
    changed["global_controls_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="list/hash"):
        runner._validate_gates(changed, metrics, controls)

    changed = copy.deepcopy(gates)
    check = changed["atomic_checks"][3]
    check["source_metric_pointer"] = "result.json#/metrics/not_declared/0"
    check["evidence_sha256"] = runner.canonical_payload_sha256(
        {key: value for key, value in check.items() if key != "evidence_sha256"}
    )
    with pytest.raises((KeyError, ValueError)):
        runner._validate_gates(changed, metrics, controls)

    changed = copy.deepcopy(gates)
    check = changed["atomic_checks"][3]
    check["observed"] = 64
    check["evidence_sha256"] = runner.canonical_payload_sha256(
        {key: value for key, value in check.items() if key != "evidence_sha256"}
    )
    with pytest.raises(ValueError, match="observed|source|decision"):
        runner._validate_gates(changed, metrics, controls)

    changed = copy.deepcopy(gates)
    changed["passed"] = False
    changed["decision"] = "FAIL"
    with pytest.raises(ValueError, match="decision"):
        runner._validate_gates(changed, metrics, controls)


def test_insufficient_landscape_controls_are_the_only_authenticated_not_applicable_case(
):
    ids = [f"c{index:02d}" for index in range(40)]
    slot = runner.score_landscape(
        case_index=1,
        bank_case_index=None,
        target_case_index=None,
        outline_mode=runner.OUTLINE_MODES[0],
        descriptor="MIND",
        domain="core",
        domain_mask=np.zeros((1, 1), dtype=bool),
        domain_mask_receipt_sha256="d" * 64,
        target_image=np.ones((1, 1), dtype=np.float64),
        candidate_images=np.ones((40, 1, 1), dtype=np.float64),
        pixel_pitch_um=25.0,
        ordered_candidate_ids=ids,
        truth_candidate_id=ids[0],
    )
    assert slot["status"] == "insufficient_domain"

    landscape = runner._landscape_control(
        slot,
        target_image=np.ones((1, 1), dtype=np.float64),
        candidate_images=np.ones((40, 1, 1), dtype=np.float64),
        domain_mask=np.zeros((1, 1), dtype=bool),
        pixel_pitch_um=25.0,
        ordered_candidate_ids=ids,
        truth_candidate_id=ids[0],
        candidate_support=None,
        target_visible=None,
        padding_value=0.0,
    )
    affine = runner._affine_slot_control(
        slot,
        target_image=np.ones((1, 1), dtype=np.float64),
        candidate_images=np.ones((40, 1, 1), dtype=np.float64),
        domain_mask=np.zeros((1, 1), dtype=bool),
        pixel_pitch_um=25.0,
        ordered_candidate_ids=ids,
        truth_candidate_id=ids[0],
        candidate_support=None,
        target_visible=None,
    )
    assert (landscape["status"], landscape["reason_code"], landscape["passed"]) == (
        "authenticated_not_applicable",
        "SOURCE_LANDSCAPE_INSUFFICIENT_DOMAIN",
        None,
    )
    assert (affine["status"], affine["reason_code"], affine["passed"]) == (
        "authenticated_not_applicable",
        "SOURCE_LANDSCAPE_INSUFFICIENT_DOMAIN",
        None,
    )
    assert landscape["permutation"] == landscape["chunks"]

    runner._validate_basic_control_state(
        landscape["permutation"], "fixture N/A", allow_not_applicable=True
    )
    changed = copy.deepcopy(landscape["permutation"])
    changed["reason_code"] = "EXACT_CONTROL_MISMATCH"
    with pytest.raises(ValueError, match="status/reason"):
        runner._validate_basic_control_state(
            changed, "fixture N/A", allow_not_applicable=True
        )
    with pytest.raises(ValueError, match="invalid control pass state"):
        runner._validate_basic_control_state(landscape["permutation"], "fixture N/A")


def _control_sidecar_fixture(*, failed_basic=False):
    case_index = 1
    target_index = (case_index + 17) % runner.CASE_COUNT
    primary_outlines, shuffled_outlines, source_slots = [], [], []
    for outline in runner.OUTLINE_MODES:
        native_slots = []
        for slot_index, (descriptor, domain) in enumerate(
            runner.NATIVE_OUTLINE_SLOT_SCHEDULE
        ):
            slot = {
                "payload_sha256": f"{len(source_slots) + 1:064x}",
                "status": "ok",
                "case_index": case_index,
                "bank_case_index": None,
                "target_case_index": None,
                "outline_mode": outline,
                "descriptor": descriptor,
                "domain": domain,
            }
            native_slots.append(slot)
            source_slots.append(slot)
        primary_outlines.append({"outline_mode": outline, "score_slots": native_slots})
    for outline in runner.OUTLINE_MODES:
        shuffled_slots = []
        for descriptor, domain in runner.SHUFFLED_OUTLINE_SLOT_SCHEDULE:
            slot = {
                "payload_sha256": f"{len(source_slots) + 1:064x}",
                "status": "ok",
                "case_index": None,
                "bank_case_index": case_index,
                "target_case_index": target_index,
                "outline_mode": outline,
                "descriptor": descriptor,
                "domain": domain,
            }
            shuffled_slots.append(slot)
            source_slots.append(slot)
        shuffled_outlines.append(
            {"outline_mode": outline, "score_slots": shuffled_slots}
        )

    permutation = {
        "mapping": "new[k]=old[(7*k+3)%40]; inverse_new_index=23*(old_index-3)%40",
        "permutation": list(runner.PERMUTATION),
        "nonidentity_bijection": True,
        "original_score_vector_sha256": "1" * 64,
        "permuted_score_vector_sha256": "2" * 64,
        "inverse_reindexed_score_vector_sha256": "1" * 64,
        "original_ranking_sha256": "3" * 64,
        "recomputed_ranking_sha256": "3" * 64,
        "passed": True,
    }
    chunks = {
        "chunk_sizes": list(runner.CHUNK_CONTROL_SIZES),
        "score_vector_sha256": {str(size): "4" * 64 for size in runner.CHUNK_CONTROL_SIZES},
        "byte_identical": True,
        "ranking_payload_sha256": "3" * 64,
        "passed": True,
    }
    landscape = []
    for slot in source_slots:
        item = {
            "source_slot_payload_sha256": slot["payload_sha256"],
            "source_status": slot["status"],
            "case_index": slot["case_index"],
            "bank_case_index": slot["bank_case_index"],
            "target_case_index": slot["target_case_index"],
            "outline_mode": slot["outline_mode"],
            "descriptor": slot["descriptor"],
            "domain": slot["domain"],
            "permutation": copy.deepcopy(permutation),
            "chunks": copy.deepcopy(chunks),
            "status": "passed",
            "reason_code": None,
            "passed": True,
        }
        runner._self_hash(item, "payload_sha256")
        landscape.append(item)
    basic_names = (
        "source_replay_metadata_geometry",
        "candidate_scalar_annotation_mask",
        "dewarp_direction_and_masks",
        "rp2_and_xy_wh",
        "scorer_signature_exclusion",
        "target_domain_invariance",
        "accurate_absent_core_identity",
        "shuffled_binding",
        "mask_only_verification",
    )
    checks = {
        name: runner._basic_control(
            {"name": name}, not (failed_basic and name == "target_domain_invariance")
        )
        for name in basic_names
    }
    checks["landscape_controls"] = landscape
    checks["affine_and_polarity"] = []
    control = {
        "schema": "anatomy-tracker.arbitrary-plane-image-information-controls/v1",
        "case_index": case_index,
        "checks": checks,
        "evidence_receipt_sha256": runner.canonical_payload_sha256(checks),
    }
    runner._self_hash(control, "payload_sha256")
    return control, {"outline_results": primary_outlines}, {
        "outline_results": shuffled_outlines
    }


def test_complete_deterministic_case_control_failure_validates_and_drives_fail_gate():
    control, primary, shuffled = _control_sidecar_fixture(failed_basic=True)
    runner._validate_case_control(control, 1, primary, shuffled)
    assert runner._case_control_passed(control) is False

    changed = copy.deepcopy(control)
    changed["checks"]["target_domain_invariance"]["reason_code"] = None
    changed["evidence_receipt_sha256"] = runner.canonical_payload_sha256(
        changed["checks"]
    )
    changed["payload_sha256"] = runner.canonical_payload_sha256(
        changed, excluded_key="payload_sha256"
    )
    with pytest.raises(ValueError, match="status/reason"):
        runner._validate_case_control(changed, 1, primary, shuffled)

    controls = []
    for index in range(runner.CASE_COUNT):
        current = copy.deepcopy(control)
        current["case_index"] = index
        current["payload_sha256"] = f"{index + 256:064x}"
        controls.append(current)
    inventory = {
        "expected_file_count": runner.FROZEN_SEMANTIC_FILE_COUNT,
        "observed_file_count": runner.FROZEN_SEMANTIC_FILE_COUNT,
        "expected_total_bytes": runner.FROZEN_SEMANTIC_TOTAL_BYTES,
        "observed_total_bytes": runner.FROZEN_SEMANTIC_TOTAL_BYTES,
        "expected_inventory_sha256": runner.FROZEN_SEMANTIC_INVENTORY_SHA256,
        "observed_inventory_sha256": runner.FROZEN_SEMANTIC_INVENTORY_SHA256,
        "passed": True,
    }
    source_records = [
        {
            "relative_path": path,
            "git_blob_sha256": "5" * 64,
            "checkout_sha256": "5" * 64,
        }
        for path in runner.SOURCE_RELATIVE_PATHS
    ]
    source = {
        "execution_commit": "6" * 40,
        "origin_commit": "6" * 40,
        "worktree_clean": True,
        "source_records": source_records,
        "scorer_signature_records": [],
        "model_dependency_records": [],
        "passed": True,
    }
    affine = {
        "required_case_indices": list(runner.AFFINE_CASES),
        "case_control_payload_sha256": [item["payload_sha256"] for item in controls],
        "applicable_slot_count": 0,
        "authenticated_not_applicable_count": 0,
        "passed": False,
    }
    evidence = {
        "frozen_inventory_audit": inventory,
        "source_and_signature_audit": source,
        "affine_and_polarity_controls": affine,
    }
    global_controls = {
        "schema": "anatomy-tracker.arbitrary-plane-image-information-global-controls/v1",
        **evidence,
        "evidence_receipt_sha256": runner.canonical_payload_sha256(evidence),
    }
    runner._self_hash(global_controls, "payload_sha256")
    runner._validate_global_controls(global_controls, controls)

    gates = runner.evaluate_gates(_gate_metrics(), global_controls)
    assert gates["passed"] is False and gates["decision"] == "FAIL"
    assert gates["atomic_checks"][2]["passed"] is False


def _toy_domain_runtime():
    masks = {name: np.ones((3, 4), dtype=bool) for name in runner.MASK_NAMES}
    domains, by_name = runner._score_domain_records(9, masks)
    ids = [f"c{index:02d}" for index in range(40)]
    supports = np.zeros((40, 3, 4), dtype=bool)
    for index in range(40):
        supports[index].flat[: 1 + index % 11] = True
    outlines = []
    for mode in runner.OUTLINE_MODES:
        slots = []
        for descriptor, domain in runner.NATIVE_OUTLINE_SLOT_SCHEDULE:
            slots.append(
                {
                    "descriptor": descriptor,
                    "domain": domain,
                    "domain_mask_receipt_sha256": runner._domain_mask_receipt_sha256(
                        by_name[domain]
                    ),
                    "domain_pixel_count": by_name[domain]["pixel_count"],
                }
            )
        outlines.append({"outline_mode": mode, "score_slots": slots})
    return {
        "record": {
            "score_domains": domains,
            "candidate_bank": {"ordered_candidate_ids": ids},
            "candidate_scalar_receipts": [
                {
                    "candidate_id": candidate_id,
                    "brain_mask": runner._array_receipt(supports[index]),
                }
                for index, candidate_id in enumerate(ids)
            ],
            "outline_results": outlines,
        },
        "supports": supports,
    }


def test_domain_invariance_rejects_outline_support_order_and_denominator_tamper():
    runtime = _toy_domain_runtime()
    assert runner._target_domain_invariance_evidence(runtime)[1] is True

    outline = copy.deepcopy(runtime)
    outline["record"]["outline_results"][1]["outline_mode"] = runner.OUTLINE_MODES[0]
    assert runner._target_domain_invariance_evidence(outline)[1] is False

    support = copy.deepcopy(runtime)
    support["supports"][0, 0, 0] = ~support["supports"][0, 0, 0]
    assert runner._target_domain_invariance_evidence(support)[1] is False

    order = copy.deepcopy(runtime)
    order["record"]["candidate_bank"]["ordered_candidate_ids"][0:2] = reversed(
        order["record"]["candidate_bank"]["ordered_candidate_ids"][0:2]
    )
    assert runner._target_domain_invariance_evidence(order)[1] is False

    denominator = copy.deepcopy(runtime)
    denominator["record"]["outline_results"][0]["score_slots"][0][
        "domain_pixel_count"
    ] += 1
    assert runner._target_domain_invariance_evidence(denominator)[1] is False

    mask_hash = copy.deepcopy(runtime)
    mask_hash["record"]["outline_results"][0]["score_slots"][0][
        "domain_mask_receipt_sha256"
    ] = "0" * 64
    assert runner._target_domain_invariance_evidence(mask_hash)[1] is False


def _toy_shuffled_binding(monkeypatch):
    monkeypatch.setattr(runner, "OUTPUT_SHAPE", (3, 4))
    ids = [f"c{index:02d}" for index in range(40)]
    base = np.arange(12, dtype=np.float64).reshape(3, 4) / 20.0
    scaled = np.stack([base + index / 1000.0 for index in range(40)])
    supports = np.ones(scaled.shape, dtype=bool)
    supports[:, 0, 0] = False
    flattened = [
        runner.constant_within_support_null(image, support)
        for image, support in zip(scaled, supports, strict=True)
    ]
    null_images = np.stack([item[0] for item in flattened])
    means = [float(item[1]) for item in flattened]
    scalar_receipts = [{"candidate_id": candidate_id} for candidate_id in ids]
    bank_candidate = {
        "finite_candidate_bank_id": "bank-id",
        "finite_candidate_receipt_sha256": "b" * 64,
        "ordered_candidate_ids": ids,
        "ordered_candidate_ids_sha256": runner.canonical_payload_sha256(ids),
        "truth_candidate_id": ids[0],
    }
    bank_record = {
        "payload_sha256": "p" * 64,
        "target": {"pixel_pitch_um": 25.0},
        "candidate_bank": bank_candidate,
        "candidate_scalar_receipts": scalar_receipts,
    }
    masks = {name: np.ones((3, 4), dtype=bool) for name in runner.MASK_NAMES}
    domains, domain_by_name = runner._score_domain_records(17, masks)
    target_outlines = [
        {"outline_mode": mode, "payload_sha256": f"outline-{index}"}
        for index, mode in enumerate(runner.OUTLINE_MODES)
    ]
    target_record = {
        "payload_sha256": "t" * 64,
        "target": {"paired_view_group_id": "target-pair", "pixel_pitch_um": 30.0},
        "outline_results": target_outlines,
        "score_domains": domains,
    }
    coordinates = runner.common_lattice_map_yx((3, 4), 25.0, 30.0)
    resampled = []
    for index, candidate_id in enumerate(ids):
        item = {
            "candidate_index": index,
            "candidate_id": candidate_id,
            "scalar_float64_receipt": runner._array_receipt(
                runner.resample_common_lattice_intensity(scaled[index], coordinates)
            ),
            "support_bool_receipt": runner._array_receipt(
                runner.resample_common_lattice_support(supports[index], coordinates)
            ),
            "constant_null_scalar_float64_receipt": runner._array_receipt(
                runner.resample_common_lattice_intensity(null_images[index], coordinates)
            ),
            "constant_supported_mean": means[index],
        }
        runner._self_hash(item, "payload_sha256")
        resampled.append(item)
    bank_identity = {
        "bank_case_index": 0,
        "source_primary_payload_sha256": bank_record["payload_sha256"],
        "finite_candidate_bank_id": bank_candidate["finite_candidate_bank_id"],
        "finite_candidate_receipt_sha256": bank_candidate["finite_candidate_receipt_sha256"],
        "ordered_candidate_ids": ids,
        "ordered_candidate_ids_sha256": bank_candidate["ordered_candidate_ids_sha256"],
        "truth_candidate_id": ids[0],
        "source_pixel_pitch_um": 25.0,
        "candidate_scalar_receipts_sha256": runner.canonical_payload_sha256(scalar_receipts),
    }
    target_identity = {
        "target_case_index": 17,
        "target_primary_payload_sha256": target_record["payload_sha256"],
        "paired_view_group_id": "target-pair",
        "target_outline_payload_sha256": [item["payload_sha256"] for item in target_outlines],
        "target_pixel_pitch_um": 30.0,
        "score_domain_receipts_sha256": runner.canonical_payload_sha256(domains),
    }
    common = {
        "source_pixel_pitch_um": 25.0,
        "target_pixel_pitch_um": 30.0,
        "coordinate_map_receipt": runner._array_receipt(coordinates),
        "resampled_candidates": resampled,
    }
    shuffled_outlines = []
    for mode in runner.OUTLINE_MODES:
        slots = [
            {
                "descriptor": descriptor,
                "domain": domain,
                "domain_mask_receipt_sha256": runner._domain_mask_receipt_sha256(
                    domain_by_name[domain]
                ),
                "domain_pixel_count": domain_by_name[domain]["pixel_count"],
            }
            for descriptor, domain in runner.SHUFFLED_OUTLINE_SLOT_SCHEDULE
        ]
        shuffled_outlines.append({"outline_mode": mode, "score_slots": slots})
    shuffled = {
        "bank_case_index": 0,
        "target_case_index": 17,
        "bank_identity": bank_identity,
        "target_identity": target_identity,
        "common_lattice_resampling": common,
        "score_domains": copy.deepcopy(domains),
        "outline_results": shuffled_outlines,
    }
    bank_runtime = {
        "case_index": 0,
        "record": bank_record,
        "scaled": scaled,
        "supports": supports,
        "null_images": null_images,
        "supported_means": means,
    }
    return bank_runtime, shuffled, {"record": target_record}


def test_shuffled_binding_rejects_pitch_map_target_bank_and_resampling_tamper(monkeypatch):
    bank, shuffled, target = _toy_shuffled_binding(monkeypatch)
    assert runner._shuffled_binding_evidence(bank, shuffled, target)[1] is True

    mutations = []
    for path, value in (
        (("bank_identity", "finite_candidate_bank_id"), "substituted-bank"),
        (("bank_identity", "ordered_candidate_ids_sha256"), "0" * 64),
        (("common_lattice_resampling", "source_pixel_pitch_um"), 24.0),
        (("target_identity", "target_pixel_pitch_um"), 31.0),
    ):
        changed = copy.deepcopy(shuffled)
        changed[path[0]][path[1]] = value
        mutations.append(changed)
    changed = copy.deepcopy(shuffled)
    changed["common_lattice_resampling"]["coordinate_map_receipt"]["array_sha256"] = "1" * 64
    mutations.append(changed)
    changed = copy.deepcopy(shuffled)
    changed["common_lattice_resampling"]["resampled_candidates"][0][
        "scalar_float64_receipt"
    ]["array_sha256"] = "2" * 64
    mutations.append(changed)
    changed = copy.deepcopy(shuffled)
    changed["outline_results"][0]["score_slots"][0]["domain_mask_receipt_sha256"] = "3" * 64
    mutations.append(changed)
    changed_target = copy.deepcopy(target)
    changed_target["record"]["payload_sha256"] = "4" * 64

    assert all(
        runner._shuffled_binding_evidence(bank, changed, target)[1] is False
        for changed in mutations
    )
    assert runner._shuffled_binding_evidence(bank, shuffled, changed_target)[1] is False


def _toy_resolved_config():
    sources = [
        {
            "relative_path": path,
            "git_blob_sha256": f"{index:064x}",
            "checkout_sha256": f"{index + 1:064x}",
        }
        for index, path in enumerate(runner.SOURCE_RELATIVE_PATHS)
    ]
    return runner._resolved_config(
        {
            "branch": "codex/joint-registration",
            "head": "a" * 40,
            "upstream_head": "a" * 40,
            "worktree_clean": True,
        },
        sources,
        {"result_payload_sha256": "s" * 64},
        {
            "asset_receipt": {
                "template_decoded": {"array_sha256": runner.DECODED_TEMPLATE_ARRAY_SHA256},
                "scalar_conversion": {"array_sha256": runner.SCALAR_CONVERSION_ARRAY_SHA256},
                "annotation_decoded": {"array_sha256": runner.DECODED_ANNOTATION_ARRAY_SHA256},
            }
        },
        {"support_index_sha256": runner.SUPPORT_INDEX_SHA256},
        {
            name: list(indices)
            for name, indices in runner.FROZEN_POOLED_MEMBERSHIP.items()
        },
    )


def _rehash_config(config):
    config["resolved_config_sha256"] = runner.canonical_payload_sha256(
        config, excluded_key="resolved_config_sha256"
    )


def test_resolved_config_validator_rejects_nested_schema_order_hash_and_authority_tamper():
    config = _toy_resolved_config()
    assert runner.validate_resolved_config(config) is config

    changed = copy.deepcopy(config)
    del changed["descriptor_constants"]["mind"]["search_displacement_um"]
    _rehash_config(changed)
    with pytest.raises(ValueError, match="mind_constants"):
        runner.validate_resolved_config(changed)

    changed = copy.deepcopy(config)
    changed["repository"]["unexpected"] = True
    _rehash_config(changed)
    with pytest.raises(ValueError, match="repository"):
        runner.validate_resolved_config(changed)

    changed = copy.deepcopy(config)
    changed["source_sha256"][0], changed["source_sha256"][1] = (
        changed["source_sha256"][1],
        changed["source_sha256"][0],
    )
    _rehash_config(changed)
    with pytest.raises(ValueError, match="source list"):
        runner.validate_resolved_config(changed)

    changed = copy.deepcopy(config)
    changed["resolved_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="self-hash"):
        runner.validate_resolved_config(changed)

    changed = copy.deepcopy(config)
    changed["model_independence"]["previous_model_dependencies"] = ["legacy"]
    _rehash_config(changed)
    with pytest.raises(ValueError, match="authority"):
        runner.validate_resolved_config(changed, expected=config)

    changed = copy.deepcopy(config)
    changed["case_and_shuffle_contract"]["pooled_strata_membership"][
        "challenging_appearance"
    ][0] = 0
    _rehash_config(changed)
    with pytest.raises(ValueError, match="pooled membership"):
        runner.validate_resolved_config(changed)


def test_pooled_membership_is_derived_only_from_exact_authenticated_semantic_cases():
    expected = {
        name: list(indices) for name, indices in runner.FROZEN_POOLED_MEMBERSHIP.items()
    }
    challenging = set(expected["challenging_appearance"])
    damaged = set(expected["damaged"])
    frozen_primary = [
        {
            "case_index": case_index,
            "reporting_strata": {
                "appearance_family": (
                    "label-conditioned" if case_index in challenging else "ordinary"
                ),
                "damage_event_count": 1 if case_index in damaged else 0,
            },
        }
        for case_index in range(runner.CASE_COUNT)
    ]
    assert runner.derive_frozen_pooled_membership(frozen_primary) == expected

    changed = copy.deepcopy(frozen_primary)
    changed[expected["challenging_appearance"][0]]["reporting_strata"][
        "appearance_family"
    ] = "ordinary"
    with pytest.raises(ValueError, match="exact 22/27"):
        runner.derive_frozen_pooled_membership(changed)

    changed = copy.deepcopy(frozen_primary)
    changed[0]["case_index"] = 63
    with pytest.raises(ValueError, match="cases 0 through 63"):
        runner.derive_frozen_pooled_membership(changed)


def _rehash_slot(slot):
    slot["payload_sha256"] = runner.canonical_payload_sha256(
        slot, excluded_key="payload_sha256"
    )


def test_slot_verifier_rejects_rehashed_raw_score_ranking_domain_and_schema_tamper(monkeypatch):
    slot, _, ids, _ = _toy_slot(
        monkeypatch, case_index=7, bank_case_index=None, target_case_index=None
    )
    masks = {name: np.ones((3, 4), dtype=bool) for name in runner.MASK_NAMES}
    _, domains = runner._score_domain_records(7, masks)
    domain = domains["core"]
    slot["descriptor"] = "constant-within-support-MIND-null"
    slot["metrics"]["supported_means"] = [0.0] * 40
    slot["domain_mask_receipt_sha256"] = runner._domain_mask_receipt_sha256(domain)
    slot["domain_pixel_count"] = domain["pixel_count"]
    slot["entered_gate"] = True
    _rehash_slot(slot)
    runner._validate_slot_record(
        slot,
        indices=(7, None, None),
        outline_mode=runner.OUTLINE_MODES[2],
        descriptor="constant-within-support-MIND-null",
        domain="core",
        domain_record=domain,
        ordered_candidate_ids=ids,
        truth_candidate_id=ids[0],
    )

    changed = copy.deepcopy(slot)
    changed["scores"][0], changed["scores"][1] = changed["scores"][1], changed["scores"][0]
    _rehash_slot(changed)
    with pytest.raises(ValueError, match="ranking"):
        runner._validate_slot_record(
            changed,
            indices=(7, None, None),
            outline_mode=runner.OUTLINE_MODES[2],
            descriptor="constant-within-support-MIND-null",
            domain="core",
            domain_record=domain,
            ordered_candidate_ids=ids,
            truth_candidate_id=ids[0],
        )

    changed = copy.deepcopy(slot)
    changed["domain_mask_receipt_sha256"] = "0" * 64
    _rehash_slot(changed)
    with pytest.raises(ValueError, match="identity/domain"):
        runner._validate_slot_record(
            changed,
            indices=(7, None, None),
            outline_mode=runner.OUTLINE_MODES[2],
            descriptor="constant-within-support-MIND-null",
            domain="core",
            domain_record=domain,
            ordered_candidate_ids=ids,
            truth_candidate_id=ids[0],
        )

    changed = copy.deepcopy(slot)
    changed["undeclared"] = True
    _rehash_slot(changed)
    with pytest.raises(ValueError, match="frozen schema"):
        runner._validate_slot_record(
            changed,
            indices=(7, None, None),
            outline_mode=runner.OUTLINE_MODES[2],
            descriptor="constant-within-support-MIND-null",
            domain="core",
            domain_record=domain,
            ordered_candidate_ids=ids,
            truth_candidate_id=ids[0],
        )


def _aggregate_fixture():
    success = {
        "top1": True,
        "true_rank": 1,
        "reciprocal_rank": 1.0,
        "truth_versus_decoy_win_fraction": 1.0,
        "truth_score_margin": 0.5,
    }
    failure = {
        "top1": False,
        "true_rank": 40,
        "reciprocal_rank": 0.025,
        "truth_versus_decoy_win_fraction": 0.0,
        "truth_score_margin": -0.5,
    }
    primary, shuffled = [], []
    for case_index in range(64):
        challenging = case_index in runner.FROZEN_POOLED_MEMBERSHIP[
            "challenging_appearance"
        ]
        damaged = case_index in runner.FROZEN_POOLED_MEMBERSHIP["damaged"]
        orientation = (
            "near_AP"
            if case_index < 12
            else "near_DV"
            if case_index < 24
            else "near_ML"
            if case_index < 36
            else "general_oblique"
        )
        primary_outlines = []
        shuffled_outlines = []
        for mode in runner.OUTLINE_MODES:
            primary_slots = []
            for descriptor, domain in runner.NATIVE_OUTLINE_SLOT_SCHEDULE:
                ranking = (
                    failure
                    if descriptor == "constant-within-support-MIND-null"
                    and domain == "core"
                    and mode == runner.OUTLINE_MODES[2]
                    else success
                )
                primary_slots.append(
                    {
                        "status": "ok",
                        "descriptor": descriptor,
                        "domain": domain,
                        "ranking": copy.deepcopy(ranking),
                    }
                )
            primary_outlines.append({"outline_mode": mode, "score_slots": primary_slots})
            shuffled_outlines.append(
                {
                    "outline_mode": mode,
                    "score_slots": [
                        {
                            "status": "ok",
                            "descriptor": descriptor,
                            "domain": domain,
                            "ranking": copy.deepcopy(failure),
                        }
                        for descriptor, domain in runner.SHUFFLED_OUTLINE_SLOT_SCHEDULE
                    ],
                }
            )
        primary.append(
            {
                "case_index": case_index,
                "provenance": {
                    "reporting_strata": {
                        "orientation_family": orientation,
                        "appearance_family": (
                            "label-conditioned" if challenging else "ordinary"
                        ),
                        "damage_event_count": 1 if damaged else 0,
                        "damage_event_types": ["tear"] if damaged else [],
                        "challenging_appearance_member": challenging,
                        "damaged_member": damaged,
                    },
                    "mask_only_Dice": {"recomputed_ranking": copy.deepcopy(success)},
                },
                "outline_results": primary_outlines,
            }
        )
        shuffled.append(
            {"bank_case_index": case_index, "outline_results": shuffled_outlines}
        )
    return primary, shuffled


def test_aggregate_metrics_binds_exact_orientation_membership_and_rejects_duplicates():
    primary, shuffled = _aggregate_fixture()
    membership = {
        name: list(indices) for name, indices in runner.FROZEN_POOLED_MEMBERSHIP.items()
    }
    metrics = runner.aggregate_metrics(primary, shuffled, membership)
    runner._validate_metrics_contract(metrics)
    assert [
        (item["stratum_value"], item["base_count"])
        for item in metrics["reporting_stratum_summaries"][:4]
    ] == [("near_AP", 12), ("near_DV", 12), ("near_ML", 12), ("general_oblique", 28)]

    changed = copy.deepcopy(metrics)
    changed["reporting_stratum_summaries"][0]["base_indices"][0] = 63
    with pytest.raises(ValueError, match="orientation summaries"):
        runner._validate_metrics_contract(changed)

    changed = copy.deepcopy(metrics)
    changed["reporting_stratum_summaries"].append(
        copy.deepcopy(changed["reporting_stratum_summaries"][0])
    )
    with pytest.raises(ValueError, match="duplicate family"):
        runner._validate_metrics_contract(changed)

    with pytest.raises(TypeError):
        runner.aggregate_metrics(primary, shuffled)

    with pytest.raises(TypeError):
        runner.evaluate_gates(metrics)
