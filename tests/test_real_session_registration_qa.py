import hashlib
import io
import json
import zipfile

import numpy as np
import tifffile

from source.nonlinear_registration import SliceAtlasTransform2D
from training import real_session_registration_qa as qa


def _write_archive(path, state, members=None):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("session.json", json.dumps(state))
        for name, payload in (members or {}).items():
            archive.writestr(name, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(fields, *, image_sha256="0" * 64, brush=None):
    return {
        "format": "Proprietary Anatomy Tracker session",
        "version": 1,
        "saved_utc": "2026-08-13T22:22:03Z",
        "atlas_folder": "atlas",
        "atlas_file_hashes": {},
        "sessions": [
            {
                "name": "0.tif",
                "image_member": "slices/0000.tif",
                "image_sha256": image_sha256,
                "brush_mask_member": brush,
                "fields": fields,
            }
        ],
    }


def test_public_manifest_hides_landmarks_until_explicit_reveal(tmp_path, monkeypatch):
    monkeypatch.setitem(qa.SESSION_LANDMARK_COUNTS, 0, 2)
    fields = {
        "atlas_landmarks": [[1, 2], [3, 4]],
        "slice_landmarks": [[5, 6], [7, 8], [9, 10]],
        "rotation_deg": 0,
    }
    archive = tmp_path / "session.attracker"
    digest = _write_archive(archive, _state(fields))

    manifest = qa.load_public_archive_manifest(archive, digest)
    assert "atlas_landmarks" not in manifest["sessions"][0]["fields"]
    assert "slice_landmarks" not in manifest["sessions"][0]["fields"]

    atlas, moving, orphans = qa.reveal_landmarks(archive, 0)
    assert atlas.tolist() == [[1, 2], [3, 4]]
    assert moving.tolist() == [[5, 6], [7, 8]]
    assert orphans == {"atlas_orphan_count": 0, "slice_orphan_count": 1}


def test_reconstruct_case_uses_embedded_raw_image_rotation_and_saved_brush_mask(tmp_path):
    image = np.arange(320 * 456, dtype=np.uint16).reshape(320, 456)
    image_payload = io.BytesIO()
    tifffile.imwrite(image_payload, image)
    image_bytes = image_payload.getvalue()
    mask = np.zeros((320, 456), np.uint8)
    mask[40:280, 50:406] = 1
    mask_payload = io.BytesIO()
    np.savez_compressed(mask_payload, mask=mask)
    fields = {
        "atlas_plane": "coronal",
        "atlas_index": 1,
        "atlas_tilt_ml_deg": 0.0,
        "atlas_tilt_dv_deg": 0.0,
        "rotation_deg": 0.0,
        "flip_horizontal": False,
        "flip_vertical": False,
        "brain_outline_points": [],
        "brain_outline_closed": False,
    }
    record = {
        "name": "0.tif",
        "image_member": "slices/0000.tif",
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "brush_mask_member": "state/0000_brush_mask.npz",
        "fields": fields,
    }
    archive = tmp_path / "session.attracker"
    _write_archive(
        archive,
        _state(fields, image_sha256=record["image_sha256"], brush=record["brush_mask_member"]),
        {record["image_member"]: image_bytes, record["brush_mask_member"]: mask_payload.getvalue()},
    )
    atlas = np.zeros((3, 320, 456), np.uint16)
    atlas[:, 40:280, 50:406] = np.arange(240 * 356, dtype=np.uint16).reshape(240, 356)
    annotation = np.zeros_like(atlas)
    annotation[:, 40:280, 50:406] = 1

    case = qa.reconstruct_case(archive, record, atlas, annotation)

    assert np.array_equal(case["moving_image"], qa.normalize_u8(image))
    assert np.array_equal(case["moving_mask"], mask.astype(bool))
    assert case["canvases"][0].shape == (320, 456)
    assert np.allclose(case["base_h"], np.eye(3), atol=1e-6)


def test_score_landmarks_reports_exact_pixel_and_physical_error():
    shape = (320, 456)
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    identity = np.stack((xx, yy), axis=-1)
    transform = SliceAtlasTransform2D(
        np.eye(3), shape, shape, identity, identity, np.ones(shape, bool)
    )
    case = {
        "slice_transform": np.eye(3),
        "base_h": np.eye(3),
        "atlas_mask": np.ones(shape, bool),
        "atlas_index": 5,
        "tilt_ml_deg": 0.0,
        "tilt_dv_deg": 0.0,
    }
    atlas = np.asarray([[20.0, 30.0], [42.0, 60.0]])
    moving = np.asarray([[20.0, 30.0], [38.0, 57.0]])

    score = qa.score_landmarks(case, transform, atlas, moving, (10, *shape))

    assert score["invalid_target_count"] == 0
    assert score["nonfinite_prediction_count"] == 0
    assert np.allclose(score["nonlinear_error_px"], [0.0, 5.0])
    assert np.allclose(score["nonlinear_error_mm"], [0.0, 0.125])


def test_summary_applies_frozen_real_session_gates_and_reports_slice9_orphan(tmp_path):
    model = tmp_path / "model.onnx"
    metadata = tmp_path / "model.metadata.json"
    model.write_bytes(b"model")
    metadata.write_text("{}", encoding="utf-8")
    results = []
    for index, count in qa.SESSION_LANDMARK_COUNTS.items():
        zeros = np.zeros(count)
        results.append(
            {
                "session_index": index,
                "case": {"name": f"{index}.tif", "atlas_index": 200 + index, "tilt_ml_deg": -4.0, "tilt_dv_deg": 2.0},
                "score": {
                    "landmark_count": count,
                    "invalid_target_count": 0,
                    "nonfinite_prediction_count": 0,
                    "valid_count": count,
                    "nonlinear_error_px": zeros,
                    "affine_error_px": np.ones(count),
                    "nonlinear_error_mm": zeros,
                    "affine_error_mm": np.full(count, 0.025),
                    "signed_error_xy_px": np.zeros((count, 2)),
                },
                "orphans": {"atlas_orphan_count": 0, "slice_orphan_count": int(index == 9)},
                "diagnostics": {"fold_count": 0, "minimum_jacobian": 0.8, "inverse_cycle_p95_px": 0.1, "inverse_cycle_max_px": 0.2},
                "runtime_metadata": {
                    "method": "dense-registration-onnx-v2",
                    "provider": "DmlExecutionProvider",
                    "candidate_checkpoint_file_sha256": "a" * 64,
                    "preprocessing_contract": "v2",
                    "map_contract": "absolute maps",
                },
            }
        )
    manifest = {"archive_path": "frozen.attracker", "archive_sha256": qa.ARCHIVE_SHA256, "saved_utc": "now"}

    report = qa.summarize_results(results, manifest, model, metadata)

    assert report["summary"]["landmark_count"] == 230
    assert report["hard_gates"]["passed"]
    assert report["strong_target"]["passed"]
    assert report["per_slice"][-1]["orphan_landmarks"]["slice_orphan_count"] == 1
    assert "not AP/tilt detection QA" in report["scope"]
