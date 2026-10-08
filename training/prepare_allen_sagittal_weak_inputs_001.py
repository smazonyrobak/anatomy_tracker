"""Prepare the frozen physically sagittal Allen cohort without image selection."""

import hashlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


sys.dont_write_bytecode = True
source = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_train_images_001_20261008")
out = Path(r"I:\AnatomyTracker\data\allen_sagittal_ish_weak_inputs_001_20261008")
acquisition_bytes = (source / "manifest.json").read_bytes()
receipt_bytes = (source / "receipt.json").read_bytes()
split_bytes = (source / "split_manifest.json").read_bytes()
split_receipt_bytes = (source / "split_receipt.json").read_bytes()
assert hashlib.sha256(acquisition_bytes).hexdigest() == "04d3fc9a4bc0f18d7f32acffa04fdba854034161b50fdf029b49e319a2c2b826"
assert hashlib.sha256(receipt_bytes).hexdigest() == "6a6e687cb773eb1f93f87b84a601dd4ba979814ea6db732059acb5d06df374c1"
assert hashlib.sha256(split_bytes).hexdigest() == "d290fbfba30fa748893f6654b3f731a611e58ab6b90d7e73b01e9be6af8b10a6"
assert hashlib.sha256(split_receipt_bytes).hexdigest() == "1d5f67ef52314a76dfd9826363420957b279d168047fb9308725212071441f59"
acquisition = json.loads(acquisition_bytes)
receipt = json.loads(receipt_bytes)
split = json.loads(split_bytes)
assert split["acquisition_manifest_sha256"] == hashlib.sha256(acquisition_bytes).hexdigest()
assert split["acquisition_receipt_sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
assert len(split["sections"]) == 240 and len(split["train_donor_ids"]) == 12 and len(split["weak_dev_donor_ids"]) == 4

side = 256
scale = 16
yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
model_grid = np.stack((xx.ravel(), yy.ravel(), np.ones(side * side)))
experiments = {row["experiment_id"]: row for row in split["experiments"]}
transform = split["reference_space_10_to_9"]["transform"]
ref9_from_ref10 = np.array([[transform[f"t_{j:02d}"] for j in range(i, i + 3)] + [transform[f"t_{9 + i // 3:02d}"]]
                            for i in (0, 3, 6)], dtype=np.float64)
assert np.array_equal(ref9_from_ref10, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 11350]])

out.mkdir(parents=True, exist_ok=False)
images = np.empty((240, 1, side, side), dtype=np.float32)
provenance = []
cv2.setNumThreads(4)
for index, section in enumerate(split["sections"]):
    image_path = source / section["image_path"]
    image_bytes = image_path.read_bytes()
    assert hashlib.sha256(image_bytes).hexdigest() == section["image_sha256"] == receipt[section["image_path"]]["sha256"]
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
    images[index, 0] = 1 - sampled_red

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
        "array_row_index": index, "split": section["split"], "donor_id": section["donor_id"],
        "specimen_id": section["specimen_id"], "experiment_id": section["experiment_id"],
        "section_id": section["section_id"], "section_number": section["section_number"],
        "image_url": section["image_url"], "image_path": section["image_path"],
        "image_sha256": section["image_sha256"], "downloaded_shape_h_w": [height_downloaded, width_downloaded],
        "full_shape_h_w": [full_height, full_width], "resolution_um_per_full_px": resolution,
        "field_um": field_um, "antialias_sigma_downloaded_px": float(sigma), "border_red": border_red,
        "downloaded_support_fraction": float(native_support.mean()),
        "model_pixel_to_full_pixel": full_from_model.tolist(),
        "model_pixel_to_downloaded_pixel": downloaded_from_model.tolist(),
        "model_pixel_to_ccf_ref9_ap_dv_ml_um": model_ccf.tolist(),
        "upstream_normal_ap_dv_ml": normal.tolist(),
        "alignment2d": a2, "alignment3d": a3,
    })
    if (index + 1) % 40 == 0:
        print(f"prepared {index + 1}/240", flush=True)

np.save(out / "model_input.npy", images)
(out / "geometry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8")
summary = {
    "version": "allen-sagittal-ish-weak-inputs-001", "source_manifest_sha256": hashlib.sha256(acquisition_bytes).hexdigest(),
    "source_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
    "split_receipt_sha256": hashlib.sha256(split_receipt_bytes).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "array": "float32[240,1,256,256] in frozen split-manifest section order",
    "roles": {"train": {"donors": 12, "sections": 160}, "weak_dev": {"donors": 4, "sections": 80}},
    "preprocessing": "one inverted red/255 view per acquired brightfield image, whole-acquisition-centred 1.05x maximal physical dimension square, Gaussian antialias, bilinear resize, acquired border-median padding before inversion, no brush, no tissue crop",
    "target_geometry": "Allen automated Alignment2d and Alignment3d, official reference-space-10-to-9 transform, +12.5 um CCF voxel-centre shift; weak plane references only, not independently verified anatomy or dense warp truth",
    "rights": acquisition["rights"], "terms_url": acquisition["terms_url"], "citation_url": acquisition["citation_url"],
    "output_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                      for name in ("model_input.npy", "geometry.jsonl")},
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "model_input_sha256": summary["output_sha256"]["model_input.npy"],
                  "geometry_sha256": summary["output_sha256"]["geometry.jsonl"]}, indent=2), flush=True)
