import copy
import math

import pytest
import torch

import arbitrary_plane_production_v3_fixtures as fixture
import training.arbitrary_plane_catalogue_capture_audit_v3 as capture_v3
import training.arbitrary_plane_row_cache_v3 as row_cache_v3
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
    full_frame_state_to_components,
)
from training.arbitrary_plane_recurrent_model import (
    compose_antipodal_plane_frame_residual,
)


LARGE_LIMITS = (math.pi, math.pi, 1000.0, math.pi, 1000.0, 1000.0, 4.0, 4.0, 4.0)


def _frozen_cache(path, row_count=2):
    row_cache_v3.initialize_training_row_cache_v3(
        path,
        generator_binding=fixture.generator_binding(),
        generation_config={
            "row_count": row_count,
            "plane_domain": "all brain-intersecting",
        },
        seed_record={"root_seed": "0xabc", "subject_seed": "0xdef"},
    )
    row_cache_v3.append_training_rows_v3(
        path, [fixture.row(index) for index in range(row_count)]
    )
    return row_cache_v3.freeze_training_row_cache_v3(path)


def _audit(cache, frozen, catalogue, update_limits=LARGE_LIMITS):
    return capture_v3.audit_catalogue_capture_v3(
        cache,
        catalogue,
        atlas_shape_ap_dv_ml=(10, 10, 10),
        origin_ap_dv_ml_um=(0.0, 0.0, 0.0),
        voxel_size_ap_dv_ml_um=(1.0, 1.0, 1.0),
        update_limits=update_limits,
        refinement_steps=1,
        expected_cache_manifest_receipt_sha256=frozen["receipt_sha256"],
        expected_catalogue_receipt_sha256=catalogue["receipt_sha256"],
    )


def test_direct_residual_decomposition_exactly_recomposes_full_physical_frame():
    dtype = torch.float64
    base = full_frame_state_from_components(
        torch.tensor((1.0, 2.0, 3.0), dtype=dtype),
        torch.eye(3, dtype=dtype),
        torch.tensor(((8.0, 1.5), (0.0, 6.0)), dtype=dtype),
    )
    expected = torch.tensor(
        (0.13, -0.09, 2.5, 0.17, 1.2, -0.8, 0.11, -0.07, 0.04),
        dtype=dtype,
    )
    truth = compose_antipodal_plane_frame_residual(
        base, expected, (0.5, 0.5, 0.5)
    )
    decomposition = capture_v3.decompose_catalogue_capture_residual_v3(
        base, truth, (0.5, 0.5, 0.5)
    )
    assert torch.allclose(decomposition["residual"], expected, atol=2e-12, rtol=0.0)
    assert decomposition["recomposition"]["physical_landmark_max_error_um"] < 1e-10

    aligned_center, aligned_frame, aligned_basis = full_frame_state_to_components(
        truth
    )
    frame_sign = torch.diag(torch.tensor((-1.0, 1.0, -1.0), dtype=dtype))
    basis_sign = torch.diag(torch.tensor((-1.0, 1.0), dtype=dtype))
    antipodal_truth = full_frame_state_from_components(
        aligned_center,
        aligned_frame @ frame_sign,
        basis_sign @ aligned_basis @ basis_sign,
    )
    antipodal = capture_v3.decompose_catalogue_capture_residual_v3(
        base, antipodal_truth, (0.5, 0.5, 0.5)
    )
    assert antipodal["antipodal_truth_normal_sign"] == -1
    assert torch.allclose(antipodal["residual"], expected, atol=2e-12, rtol=0.0)
    assert antipodal["recomposition"]["truth_landmark_permutation"] == [1, 0, 3, 2, 4]
    assert antipodal["recomposition"][
        "antipodal_representation_equivalence_max_error_um"
    ] < 1e-10


def test_frozen_cache_audit_is_streaming_read_only_and_receipt_bound(
    tmp_path, monkeypatch
):
    cache = tmp_path / "capture-cache"
    frozen = _frozen_cache(cache)
    catalogue = fixture.catalogue()
    manifest_path = cache / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    calls = []
    original = row_cache_v3._load_record

    def counted_load_record(root, record, geometry_gauge_contract):
        calls.append(record["row_index"])
        return original(root, record, geometry_gauge_contract)

    def forbidden_dense_loader(*_args, **_kwargs):
        raise AssertionError("capture audit must not load the full row cache")

    monkeypatch.setattr(row_cache_v3, "_load_record", counted_load_record)
    monkeypatch.setattr(
        row_cache_v3, "load_training_rows_v3", forbidden_dense_loader
    )
    report = _audit(cache, frozen, catalogue)

    assert calls == [0, 1]
    assert manifest_path.read_bytes() == manifest_before
    assert report["row_count"] == 2
    assert [item["training_row_id"] for item in report["rows"]] == [
        "row-0",
        "row-1",
    ]
    assert [item["animal_id"] for item in report["rows"]] == [
        "animal-0",
        "animal-1",
    ]
    assert all(item["selected_mode"] == "smart-brush-accurate" for item in report["rows"])
    assert report["model_capture_contract"][
        "cumulative_component_envelope_description"
    ] == capture_v3.CUMULATIVE_ENVELOPE_DESCRIPTION
    assert set(report["component_summary"]) == set(capture_v3.RESIDUAL_COMPONENTS)
    assert report["maximum_recomposition_error"][
        "physical_landmark_max_error_um"
    ] < 1e-8
    assert report["catalogue_binding"]["receipt_sha256"] == catalogue[
        "receipt_sha256"
    ]
    assert report["row_cache_binding"]["manifest_receipt_sha256"] == frozen[
        "receipt_sha256"
    ]
    assert capture_v3.verify_catalogue_capture_audit_report_v3(report)
    changed = copy.deepcopy(report)
    changed["rows"][0]["animal_id"] = "tampered-animal"
    with pytest.raises(ValueError, match="failed its receipt"):
        capture_v3.verify_catalogue_capture_audit_report_v3(changed)


def test_deliberately_unreachable_offset_and_basis_fail_cumulative_envelope(
    tmp_path,
):
    cache = tmp_path / "unreachable-cache"
    frozen = _frozen_cache(cache, row_count=1)
    catalogue = fixture.catalogue()
    reachable = _audit(cache, frozen, catalogue)
    residual = reachable["rows"][0]["direct_residual"]
    assert abs(residual["support_origin_normal_offset_um"]) > 1e-6
    assert max(
        abs(residual["delta_log_basis_u"]),
        abs(residual["delta_log_basis_v"]),
    ) > 1e-6
    limits = list(LARGE_LIMITS)
    update_count = 2
    limits[2] = abs(residual["support_origin_normal_offset_um"]) / (
        2.0 * update_count
    )
    limits[6] = max(abs(residual["delta_log_basis_u"]), 1e-9) / (
        2.0 * update_count
    )
    limits[7] = max(abs(residual["delta_log_basis_v"]), 1e-9) / (
        2.0 * update_count
    )
    with pytest.raises(ValueError, match="cumulative-component envelope"):
        _audit(cache, frozen, catalogue, update_limits=limits)


def test_catalogue_array_tamper_is_rejected_before_row_streaming(tmp_path):
    cache = tmp_path / "tampered-catalogue-cache"
    frozen = _frozen_cache(cache, row_count=1)
    catalogue = fixture.catalogue()
    changed = copy.deepcopy(catalogue)
    changed["arrays"]["cell_states_float64"][0, 0] += 1.0
    with pytest.raises(ValueError, match="immutable receipt"):
        _audit(cache, frozen, changed)
