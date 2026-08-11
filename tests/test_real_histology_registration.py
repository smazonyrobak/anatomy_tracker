import hashlib
import json
from pathlib import Path

import nrrd
import numpy as np
import pytest
import torch
from PIL import Image

from source.registered_image_quality import build_registered_image_quality_manifest

import training.evaluate_locked_nonlinear_histology as locked_evaluator
from training.evaluate_locked_nonlinear_histology import run_locked_evaluation
from training.real_histology_registration import (
    MAX_DENSE_EPE_MEDIAN_PX,
    MAX_DENSE_EPE_P95_PX,
    MAX_JACOBIAN_ERROR_P95,
    MAX_NATIVE_MIND_UPPER95,
    MAX_NATIVE_WRONG_DISPLACEMENT_P95_PX,
    MAX_NATIVE_SURFACE_DICE_LOSS,
    MAX_TRE_MEDIAN_PX,
    MAX_TRE_MEDIAN_CCF_UM,
    MAX_TRE_P95_PX,
    MAX_TRE_P95_CCF_UM,
    MIN_DENSE_IMPROVEMENT_PX,
    MIN_DENSE_RELATIVE_IMPROVEMENT,
    MIN_DENSE_SECTION_PASS_RATE,
    MIN_INTERIOR_IMPROVEMENT_PX,
    MIN_NATIVE_ACCEPT_RATE,
    MIN_NATIVE_MIND_IMPROVEMENT_RATE,
    MIN_NATIVE_RETAINED_COVERAGE,
    MIN_NATIVE_WRONG_REJECT_RATE,
    MIN_VALID_ACCEPT_RATE,
    RegisteredHistologySource,
    _animal_bootstrap,
    _animal_summary,
    _dense_row,
    _native_row,
    model_canvas_plane_basis_um,
    native_registration_batch,
    canonical_registered_pair,
    canonical_sha256,
    downloaded_pixel_to_reference_index,
    file_sha256,
    real_histology_gate_failures,
    real_histology_gate_violation,
    select_native_wrong_entries,
    torch_model_sha256,
)
from training.train_diffeomorphic_registration import (
    RegistrationWithRejector,
    make_synthetic_pair,
)
from training.diffeomorphic_registration_model import pixel_identity_grid


def write_jsonl(path: Path, records: list[dict]):
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records), encoding="utf-8")


def source_fixture(tmp_path: Path, *, crossing_specimen: bool = False):
    atlas = tmp_path / "atlas"
    root = tmp_path / "registered"
    atlas.mkdir()
    root.mkdir()
    average = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    annotation = np.ones_like(average, dtype=np.int16)
    nrrd.write(str(atlas / "average_template_25.nrrd"), average)
    nrrd.write(str(atlas / "annotation_25.nrrd"), annotation)
    transform = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    datasets = [
        {
            "experiment_id": index,
            "specimen_id": 10 if crossing_specimen and index < 3 else index * 10,
            "split": split,
            "section_thickness_um": 25.0,
            "alignment3d_tvr": transform,
        }
        for index, split in enumerate(("train", "validation", "test"), 1)
    ]
    scale = 25.0 / 32.0
    offset = -15.5 * scale
    sections, downloads = [], []
    for dataset in datasets:
        section_id = dataset["experiment_id"] * 100
        relative_path = f"images/{dataset['split']}/{section_id}.jpg"
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.arange(48, dtype=np.uint8).reshape(6, 8) * 4 + dataset["experiment_id"]
        Image.fromarray(image).save(
            path, format="JPEG", quality=100, subsampling=0
        )
        sections.append({
            "section_image_id": section_id,
            "experiment_id": dataset["experiment_id"],
            "specimen_id": dataset["specimen_id"],
            "split": dataset["split"],
            "section_number": 2,
            "alignment2d_tsv": [scale, 0.0, 0.0, scale, offset, offset],
            "ap_um": -1000.0 + section_id,
            "tilt_lr_deg": 0.0,
            "tilt_dv_deg": 0.0,
            "relative_path": relative_path,
        })
        downloads.append({"section_image_id": section_id, "sha256": file_sha256(path)})
    write_jsonl(root / "datasets.jsonl", datasets)
    write_jsonl(root / "sections.jsonl", sections)
    write_jsonl(root / "downloads.jsonl", downloads)
    provenance = {
        "datasets_sha256": file_sha256(root / "datasets.jsonl"),
        "sections_sha256": file_sha256(root / "sections.jsonl"),
        "split": {
            "unit": "specimen_id",
            "salt": "atlaspose-allen-s2p-specimen-v1",
            "fractions": [0.9, 0.05, 0.05],
        },
        "download": {"downsample": 5},
    }
    (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    build_registered_image_quality_manifest(root)
    return root, atlas, datasets, sections


def test_registered_source_binds_provenance_split_and_deterministic_selection(tmp_path):
    root, atlas, _, _ = source_fixture(tmp_path)
    source = RegisteredHistologySource(root, atlas)
    manifest = source.evaluation_manifest("validation", 17)
    source.verify_manifest(manifest)
    assert manifest["split"] == "validation"
    assert manifest["sealed_data_used"] is False
    assert manifest["entries"][0]["synthetic_seeds"]
    training = source.training_bank_manifest()
    assert training["split"] == "train" and training["sealed_data_used"] is False

    changed = json.loads(json.dumps(manifest))
    changed["entries"][0]["ap_um"] += 25.0
    changed["manifest_sha256"] = canonical_sha256({
        key: value for key, value in changed.items() if key != "manifest_sha256"
    })
    with pytest.raises(ValueError, match="preregistered deterministic"):
        source.verify_manifest(changed)
    with pytest.raises(ValueError, match="restricted"):
        source.evaluation_manifest("sealed_deepslice_s2p", 17)


def test_registered_source_rejects_animal_leakage_and_changed_image(tmp_path):
    crossing = tmp_path / "crossing"
    crossing.mkdir()
    root, atlas, _, _ = source_fixture(crossing, crossing_specimen=True)
    with pytest.raises(ValueError, match="crosses Allen splits"):
        RegisteredHistologySource(root, atlas)

    clean = tmp_path / "clean"
    clean.mkdir()
    root, atlas, _, sections = source_fixture(clean)
    source = RegisteredHistologySource(root, atlas)
    path = root / sections[1]["relative_path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity"):
        source.section(sections[1]["section_image_id"])


def test_registered_source_uses_the_shared_quality_approval_contract(tmp_path):
    root, atlas, _, sections = source_fixture(tmp_path)
    rejected = sections[1]
    path = root / rejected["relative_path"]
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(
        path, format="JPEG", quality=100, subsampling=0
    )
    downloads = [json.loads(line) for line in (root / "downloads.jsonl").read_text().splitlines()]
    for record in downloads:
        if int(record["section_image_id"]) == int(rejected["section_image_id"]):
            record["sha256"] = file_sha256(path)
    write_jsonl(root / "downloads.jsonl", downloads)
    build_registered_image_quality_manifest(root)

    source = RegisteredHistologySource(root, atlas)
    assert source.records["validation"] == []
    assert set(source.rejected_records) == {int(rejected["section_image_id"])}
    with pytest.raises(ValueError, match="quality contract"):
        source.section(rejected["section_image_id"])


def test_native_registered_resampling_preserves_one_to_one_25um_geometry():
    scale = 25.0 / 32.0
    dataset = {
        "section_thickness_um": 25.0,
        "alignment3d_tvr": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    section = {
        "section_number": 2,
        "alignment2d_tsv": [scale, 0.0, 0.0, scale, -15.5 * scale, -15.5 * scale],
    }
    transform = downloaded_pixel_to_reference_index(dataset, section)
    assert np.allclose(transform, [[0, 0, 2], [0, 1, 0], [1, 0, 0]])
    assert np.allclose(
        model_canvas_plane_basis_um(dataset, section),
        [[0, 0], [0, 25], [25, 0]],
    )
    oblique_dataset = {
        **dataset,
        "alignment3d_tvr": [0.2, 0.1, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    assert np.allclose(
        model_canvas_plane_basis_um(oblique_dataset, section),
        [[5.0, 2.5], [0.0, 25.0], [25.0, 0.0]],
    )

    image = np.arange(48, dtype=np.float32).reshape(6, 8) / 47.0
    average = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    fixed, moving, fixed_mask, moving_mask = canonical_registered_pair(
        image, np.ones_like(image, bool), average, np.ones_like(average, bool), dataset, section
    )
    top, left = (320 - 6) // 2, (464 - 8) // 2
    assert np.allclose(fixed[top : top + 6, left : left + 8], average[2])
    assert np.allclose(moving[top : top + 6, left : left + 8], image)
    assert fixed_mask.sum() == moving_mask.sum() == 48


def test_interior_dense_stratum_has_identical_surface_masks():
    y, x = torch.meshgrid(torch.arange(96), torch.arange(128), indexing="ij")
    mask = (((x - 64) / 50).square() + ((y - 48) / 38).square() < 1)[None, None].float()
    image = torch.rand_like(mask) * mask
    pair = make_synthetic_pair(
        image,
        torch.zeros_like(image, dtype=torch.long),
        mask,
        seed=71,
        stratum="real_histology_interior_label_free",
    )
    assert torch.equal(pair["fixed_mask"], pair["moving_mask"])
    assert pair["target_velocity"].abs().max() > 0.1
    assert pair["target_velocity"][:, :, :10].abs().max() == 0.0


def test_dense_oracle_beats_identity_and_binds_exact_target_maps():
    y, x = torch.meshgrid(torch.arange(96), torch.arange(128), indexing="ij")
    mask = (((x - 64) / 50).square() + ((y - 48) / 38).square() < 1)[None, None].float()
    image = torch.rand_like(mask) * mask
    pair = make_synthetic_pair(
        image, torch.zeros_like(image, dtype=torch.long), mask,
        seed=81, stratum="real_histology_interior_label_free",
    )
    accepted = torch.tensor([-10.0])
    oracle = (
        pair["target_atlas_to_affine"], pair["target_affine_to_atlas"],
        pair["target_velocity"], accepted,
    )
    identity = pixel_identity_grid(1, 96, 128)
    baseline = (identity, identity, torch.zeros_like(identity), accepted)
    identifiers = {"specimen_id": 1, "experiment_id": 2, "section_image_id": 3, "stratum": "interior"}
    oracle_row = _dense_row(oracle, pair, 0, identifiers)
    baseline_row = _dense_row(baseline, pair, 0, identifiers)
    assert oracle_row["epe_p95_px"] == 0.0
    assert oracle_row["epe_p95_improvement_px"] > 0.0
    assert len(oracle_row["target_maps_sha256"]) == 64
    assert baseline_row["epe_improvement_px"] == 0.0


def test_dense_tre_uses_exact_oblique_basis_for_ccf_microns():
    y, x = torch.meshgrid(torch.arange(48), torch.arange(64), indexing="ij")
    mask = (((x - 32) / 25).square() + ((y - 24) / 18).square() < 1)[None, None].float()
    pair = make_synthetic_pair(
        torch.rand_like(mask) * mask,
        torch.zeros_like(mask, dtype=torch.long),
        mask,
        seed=91,
        stratum="identity_extreme_label_free",
    )
    pair["plane_basis_um"] = torch.tensor([[[15.0, 0.0], [0.0, 25.0], [20.0, 0.0]]])
    identity = pixel_identity_grid(1, 48, 64)
    forward, inverse = identity.clone(), identity.clone()
    forward[:, 0] += 1.0
    inverse[:, 0] -= 1.0
    row = _dense_row(
        (forward, inverse, torch.zeros_like(identity), torch.tensor([-10.0])),
        pair,
        0,
        {"specimen_id": 1, "stratum": "physical"},
    )
    assert row["tre_median_px"] == pytest.approx(1.0)
    assert row["tre_median_ccf_um"] == pytest.approx(25.0)
    assert row["tre_p95_ccf_um"] == pytest.approx(25.0)


def test_native_wrong_selection_encodes_ap_tilt_and_independent_plane_constraints():
    entries = [
        {"section_image_id": 1, "specimen_id": 1, "ap_um": 0.0, "tilt_lr_deg": 0.0, "tilt_dv_deg": 0.0},
        {"section_image_id": 2, "specimen_id": 2, "ap_um": -1200.0, "tilt_lr_deg": 0.2, "tilt_dv_deg": 0.1},
        {"section_image_id": 3, "specimen_id": 3, "ap_um": 100.0, "tilt_lr_deg": 8.0, "tilt_dv_deg": 5.0},
        {"section_image_id": 4, "specimen_id": 4, "ap_um": -300.0, "tilt_lr_deg": -2.0, "tilt_dv_deg": 1.0},
        {"section_image_id": 5, "specimen_id": 5, "ap_um": -700.0, "tilt_lr_deg": 0.1, "tilt_dv_deg": 0.1},
    ]
    target = entries[:1]
    assert select_native_wrong_entries(entries, target, "wrong_ap", 7)[0]["section_image_id"] == 5
    assert select_native_wrong_entries(entries, target, "wrong_tilt", 7)[0]["section_image_id"] == 4
    plane = select_native_wrong_entries(entries, target, "wrong_plane", 7)[0]
    assert plane["section_image_id"] != 1 and plane["specimen_id"] != 1

    repeated = select_native_wrong_entries(entries, target * 3, "wrong_plane", 7)
    assert len({entry["section_image_id"] for entry in repeated}) == 3


def test_native_similarity_cannot_improve_by_discarding_support():
    height, width = 32, 40
    identity = pixel_identity_grid(1, height, width)
    forward, inverse = identity.clone(), identity.clone()
    forward[:, 0] += 8.0
    inverse[:, 0] -= 8.0
    image = torch.rand(1, 1, height, width)
    mask = torch.zeros_like(image)
    mask[:, :, 4:-4, 5:-5] = 1.0
    batch = {"fixed": image, "moving": image, "fixed_mask": mask, "moving_mask": mask}
    outputs = (forward, inverse, torch.zeros_like(identity), torch.tensor([-10.0]))
    row = _native_row(outputs, batch, 0, {"specimen_id": 1})
    assert row["mind_before"] < 1e-6
    assert row["mind_after"] > 0.0
    assert row["retained_coverage"] < 1.0


def passing_real_gates():
    return {
        "animal_count": 20,
        "section_count": 80,
        "dense_epe_median_px": MAX_DENSE_EPE_MEDIAN_PX,
        "dense_epe_p95_px": MAX_DENSE_EPE_P95_PX,
        "tre_median_px": MAX_TRE_MEDIAN_PX,
        "tre_p95_px": MAX_TRE_P95_PX,
        "tre_median_ccf_um": MAX_TRE_MEDIAN_CCF_UM,
        "tre_p95_ccf_um": MAX_TRE_P95_CCF_UM,
        "jacobian_error_p95": MAX_JACOBIAN_ERROR_P95,
        "epe_improvement_px": MIN_DENSE_IMPROVEMENT_PX,
        "interior_epe_improvement_px": MIN_INTERIOR_IMPROVEMENT_PX,
        "epe_relative_improvement": MIN_DENSE_RELATIVE_IMPROVEMENT,
        "epe_p95_improvement_px": MIN_DENSE_IMPROVEMENT_PX,
        "interior_epe_p95_improvement_px": MIN_INTERIOR_IMPROVEMENT_PX,
        "epe_p95_relative_improvement": MIN_DENSE_RELATIVE_IMPROVEMENT,
        "dense_accept_rate": MIN_VALID_ACCEPT_RATE,
        "dense_section_pass_rate": MIN_DENSE_SECTION_PASS_RATE,
        "native_accept_rate": MIN_NATIVE_ACCEPT_RATE,
        "native_mind_improvement_rate": MIN_NATIVE_MIND_IMPROVEMENT_RATE,
        "native_wrong_reject_rate": MIN_NATIVE_WRONG_REJECT_RATE,
        "native_wrong_displacement_p95_px": MAX_NATIVE_WRONG_DISPLACEMENT_P95_PX,
        "native_mind_delta": MAX_NATIVE_MIND_UPPER95 - 1e-3,
        "native_surface_dice_delta": -MAX_NATIVE_SURFACE_DICE_LOSS,
        "native_retained_coverage": MIN_NATIVE_RETAINED_COVERAGE,
        "geometry_passed": True,
    }


def test_real_gate_requires_independent_animals_dense_accuracy_and_native_improvement():
    assert real_histology_gate_failures(passing_real_gates()) == []
    assert real_histology_gate_violation(passing_real_gates()) == 0.0
    gates = passing_real_gates()
    gates.update({"animal_count": 5, "dense_epe_p95_px": 3.0, "native_retained_coverage": 0.80})
    failures = real_histology_gate_failures(gates)
    assert any("independent animals" in failure for failure in failures)
    assert any("EPE p95" in failure for failure in failures)
    assert any("support" in failure for failure in failures)
    gates = passing_real_gates()
    gates["native_mind_delta"] = 0.0
    assert any("statistically supported" in failure for failure in real_histology_gate_failures(gates))
    assert real_histology_gate_violation(gates) > 0.0


def test_acceptance_is_a_rate_and_sections_cannot_outvote_an_animal():
    rows = [
        *({"specimen_id": 1, "accepted": 1.0, "score": 0.0} for _ in range(100)),
        {"specimen_id": 2, "accepted": 0.0, "score": 10.0},
    ]
    animals = _animal_summary(rows, ("accepted", "score"))
    interval = _animal_bootstrap(animals, ("accepted", "score"), seed=3)
    assert animals[1]["accepted"] == 1.0 and animals[2]["accepted"] == 0.0
    assert interval["score"]["estimate"] == 5.0


def test_animal_summary_does_not_hide_catastrophic_section_failures():
    rows = [
        {"specimen_id": animal, "error": error}
        for animal in range(20)
        for error in (0.0, 0.0, 0.0, 100.0)
    ]
    animals = _animal_summary(rows, ("error",))
    interval = _animal_bootstrap(animals, ("error",), seed=3)
    assert interval["error"]["lower95"] == pytest.approx(25.0)


def test_model_hash_is_deterministic_and_parameter_sensitive():
    model = torch.nn.Conv2d(1, 2, 3)
    digest = torch_model_sha256(model)
    assert digest == torch_model_sha256(model)
    with torch.no_grad():
        model.weight[0, 0, 0, 0] += 1.0
    assert digest != torch_model_sha256(model)


def test_locked_evaluator_refuses_an_unbound_or_already_claimed_candidate(tmp_path):
    model_path = tmp_path / "candidate.onnx"
    model_path.write_bytes(b"frozen")
    manifest_path = model_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "model_sha256": hashlib.sha256(b"other").hexdigest(),
        "synthetic_gate_passed": True,
        "onnx_gate_passed": True,
        "real_histology_gate_passed": False,
        "promotion_ready": False,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its manifest"):
        run_locked_evaluation(model_path, tmp_path / "unused", tmp_path / "out")

    payload = json.loads(manifest_path.read_text())
    payload.update(model_sha256=file_sha256(model_path), real_histology_gate_passed=True)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="already claims"):
        run_locked_evaluation(model_path, tmp_path / "unused", tmp_path / "out")


def test_locked_evaluator_atomically_consumes_candidate_before_test_and_cannot_retry(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "candidate.onnx"
    model_path.write_bytes(b"frozen-model")
    model_sha256 = file_sha256(model_path)
    commitment = {
        "source": {"source": "fixed-test-source"},
        "evaluation_manifest_sha256": "a" * 64,
    }
    evidence_path = model_path.with_suffix(".prelocked.json")
    evidence_path.write_text(json.dumps({
        "model_sha256": model_sha256,
        "synthetic_gate": {"passed": True},
        "onnx_gate": {"passed": True},
        "locked_real_histology_commitment": commitment,
    }), encoding="utf-8")
    manifest_path = model_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "model_sha256": model_sha256,
        "prelocked_evidence_file": evidence_path.name,
        "prelocked_evidence_sha256": file_sha256(evidence_path),
        "locked_real_histology_commitment": commitment,
        "synthetic_gate_passed": True,
        "onnx_gate_passed": True,
        "real_histology_gate_passed": False,
        "promotion_ready": False,
    }), encoding="utf-8")
    registered_root = tmp_path / "registered"
    ledger = registered_root / ".nonlinear_locked_test_consumption"

    class FakeSource:
        def __init__(self, *_):
            assert (ledger / f"{model_sha256}.claim").is_file()
            assert (ledger / f"{model_sha256}.json").is_file()
            self.contract = {"source": "fixed-test-source"}

        def evaluation_manifest(self, split, seed):
            assert split == "test"
            return {"manifest_sha256": "a" * 64, "split": split, "seed": seed}

    monkeypatch.setattr(locked_evaluator, "RegisteredHistologySource", FakeSource)
    monkeypatch.setattr(locked_evaluator, "OnnxRegistrationModel", lambda _: object())

    def fake_evaluate(*_args, **_kwargs):
        assert (ledger / f"{model_sha256}.json").is_file()
        return {"passed": True, "report_sha256": "b" * 64}

    monkeypatch.setattr(locked_evaluator, "evaluate_real_histology", fake_evaluate)
    release = run_locked_evaluation(
        model_path,
        registered_root,
        tmp_path / "output-1",
        tmp_path / "atlas",
    )
    assert release["test_split_consumed"] is True
    assert len(release["consumption_receipt_sha256"]) == 64

    with pytest.raises(RuntimeError, match="already consumed"):
        run_locked_evaluation(
            model_path,
            registered_root,
            tmp_path / "output-2",
            tmp_path / "atlas",
        )
