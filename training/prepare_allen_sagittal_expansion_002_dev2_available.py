"""Prepare only source-available frozen sagittal weak-DEV2 images."""

import hashlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


sys.dont_write_bytecode = True
source = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_dev2_images_available_20261008")
metadata = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_20261008")
out = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_expansion_002_dev2_inputs_available_20261008")
acquisition_bytes = (source / "manifest.json").read_bytes()
receipt_bytes = (source / "receipt.json").read_bytes()
acquisition = json.loads(acquisition_bytes)
receipt = json.loads(receipt_bytes)
assert acquisition["version"] == "allen-sagittal-ish-expansion-002-dev2-available-images"
assert acquisition["source_metadata_manifest_sha256"] == "b4f77e00325273f937e4327b52d8b3f7db95458c130d4f88f682a8a08323f7bd"
assert acquisition["availability_manifest_sha256"] == "8cec8d29c4d177073e58b8bc2871e849c1051620a4d00eb88d636017f402066d"
assert len(acquisition["experiments"]) == 8 and len(acquisition["sections"]) == 158
assert [row["section_id"] for row in acquisition["unavailable_source_sections_not_downloaded"]] == [101345593]
assert all(section["image_status"] == "downloaded_and_decoded" for section in acquisition["sections"])
assert not ({section["donor_id"] for section in acquisition["sections"]}
            & set(acquisition["sealed_holdout_donor_ids_not_downloaded"]))
transform_bytes = (metadata / "api" / "reference_space_10_to_9.json").read_bytes()
candidate = json.loads((metadata / "manifest.json").read_bytes())
assert hashlib.sha256(transform_bytes).hexdigest() == candidate["reference_space_10_to_9"]["source_sha256"]
transform = json.loads(transform_bytes)["msg"][0]
assert transform == candidate["reference_space_10_to_9"]["transform"]
ref9_from_ref10 = np.array([[transform[f"t_{j:02d}"] for j in range(i, i + 3)] + [transform[f"t_{9 + i // 3:02d}"]]
                            for i in (0, 3, 6)], dtype=np.float64)
assert np.array_equal(ref9_from_ref10, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 11350]])

side = 256
scale = 16
gain = 4.259364821544654
yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
model_grid = np.stack((xx.ravel(), yy.ravel(), np.ones(side * side)))
experiments = {row["experiment_id"]: row for row in acquisition["experiments"]}
images = np.empty((158, 1, side, side), dtype=np.float32)
provenance = []
cv2.setNumThreads(4)
out.mkdir(parents=True, exist_ok=False)
(out / "reference_space_10_to_9.json").write_bytes(transform_bytes)
for index, section in enumerate(acquisition["sections"]):
    image_path = source / section["image_path"]
    image_bytes = image_path.read_bytes()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    assert image_sha256 == section["image_sha256"] == receipt[section["image_path"]]["sha256"]
    red = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.float32)[..., 0] / 255.0
    height_downloaded, width_downloaded = red.shape
    full_width, full_height = section["image_width_full_px"], section["image_height_full_px"]
    resolution = section["resolution_um_per_full_px"]
    field_um = 1.05 * max(full_width, full_height) * resolution
    step_full = field_um / (side * resolution)
    full_from_model = np.array([[step_full, 0, (full_width - 1) / 2 - field_um / (2 * resolution)],
                                [0, step_full, (full_height - 1) / 2 - field_um / (2 * resolution)],
                                [0, 0, 1]], dtype=np.float64)
    downloaded_from_full = np.array([[1 / scale, 0, .5 / scale - .5],
                                     [0, 1 / scale, .5 / scale - .5], [0, 0, 1]], dtype=np.float64)
    downloaded_from_model = downloaded_from_full @ full_from_model
    sample_grid = (downloaded_from_model @ model_grid)[:2].reshape(2, side, side).astype(np.float32)
    sigma = .5 * np.sqrt(max((step_full / scale) ** 2 - 1, 0))
    border_red = float(np.median(np.concatenate((red[0], red[-1], red[:, 0], red[:, -1]))))
    filtered = cv2.GaussianBlur(red, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE) if sigma > 0 else red
    sampled_red = cv2.remap(filtered, sample_grid[0], sample_grid[1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=border_red)
    images[index, 0] = np.clip(gain * (1 - sampled_red), 0, 1)

    experiment = experiments[section["experiment_id"]]
    a2 = section["alignment2d"]
    volume_from_full = np.array([[a2["tsv_00"], a2["tsv_01"], a2["tsv_04"]],
                                 [a2["tsv_02"], a2["tsv_03"], a2["tsv_05"]],
                                 [0, 0, section["section_number"] * experiment["section_thickness_um"]]], dtype=np.float64)
    a3 = experiment["alignment3d"]
    ref10_from_volume = np.array([[a3[f"tvr_{3 * j + i:02d}"] for i in range(3)] + [a3[f"tvr_{9 + j:02d}"]]
                                  for j in range(3)], dtype=np.float64)
    ref10_from_full = ref10_from_volume[:, :3] @ volume_from_full
    ref10_from_full[:, 2] += ref10_from_volume[:, 3]
    ref9_from_full = ref9_from_ref10[:, :3] @ ref10_from_full
    ref9_from_full[:, 2] += ref9_from_ref10[:, 3]
    model_ccf = ref9_from_full @ full_from_model
    model_ccf[:, 2] += 12.5
    normal = np.cross(model_ccf[:, 0], model_ccf[:, 1])
    normal /= np.linalg.norm(normal)
    native_support = ((sample_grid[0] >= 0) & (sample_grid[0] <= width_downloaded - 1)
                      & (sample_grid[1] >= 0) & (sample_grid[1] <= height_downloaded - 1))
    provenance.append({
        "array_row_index": index, "split": "weak_dev2_source_available", "donor_id": section["donor_id"],
        "specimen_id": section["specimen_id"], "experiment_id": section["experiment_id"],
        "section_id": section["section_id"], "section_number": section["section_number"],
        "image_url": section["image_url"], "response_url": section["response_url"],
        "image_path": section["image_path"], "image_sha256": image_sha256,
        "downloaded_shape_h_w": [height_downloaded, width_downloaded],
        "full_shape_h_w": [full_height, full_width], "resolution_um_per_full_px": resolution,
        "field_um": field_um, "antialias_sigma_downloaded_px": float(sigma), "border_red": border_red,
        "downloaded_support_fraction": float(native_support.mean()), "fixed_gain": gain,
        "model_pixel_to_full_pixel": full_from_model.tolist(),
        "model_pixel_to_downloaded_pixel": downloaded_from_model.tolist(),
        "model_pixel_to_ccf_ref9_ap_dv_ml_um": model_ccf.tolist(),
        "upstream_normal_ap_dv_ml": normal.tolist(),
        "alignment2d": a2, "alignment3d": a3,
    })
    if (index + 1) % 50 == 0:
        print(f"prepared {index + 1}/158", flush=True)

np.save(out / "model_input.npy", images)
(out / "geometry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8")
summary = {
    "version": "allen-sagittal-ish-expansion-002-dev2-available-inputs",
    "source_manifest_sha256": hashlib.sha256(acquisition_bytes).hexdigest(),
    "source_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    "source_metadata_manifest_sha256": acquisition["source_metadata_manifest_sha256"],
    "availability_manifest_sha256": acquisition["availability_manifest_sha256"],
    "source_availability_manifest_sha256": acquisition["availability_manifest_sha256"],
    "official_reference_space_10_to_9_sha256": hashlib.sha256(transform_bytes).hexdigest(),
    "unavailable_source_section_ids_excluded": [101345593],
    "original_section_count": 159,
    "source_available_section_count": 158,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "array": "float32[158,1,256,256] in source-available DEV2 acquisition-manifest section order",
    "roles": {"weak_dev2_source_available": {"donors": 8, "sections": 158}},
    "preprocessing": "one inverted red/255 view per acquired brightfield image, whole-acquisition-centred 1.05x maximal physical dimension square, Gaussian antialias, bilinear resize, acquired border-median padding before inversion, fixed TRAIN-derived gain 4.259364821544654 and clipping on channel 0, no brush, no tissue crop or paired version",
    "target_geometry": "Allen automated Alignment2d and Alignment3d, official reference-space-10-to-9 transform, +12.5 um CCF voxel-centre shift; weak plane references only, not independently verified anatomy or dense warp truth",
    "rights": acquisition["rights"], "terms_url": acquisition["terms_url"], "citation_url": acquisition["citation_url"],
    "output_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                      for name in ("model_input.npy", "geometry.jsonl", "reference_space_10_to_9.json")},
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "model_input_sha256": summary["output_sha256"]["model_input.npy"],
                  "geometry_sha256": summary["output_sha256"]["geometry.jsonl"],
                  "summary_sha256": hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()}, indent=2), flush=True)
