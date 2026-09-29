"""Fixed raw-red inputs and descriptive statistics from the 58 training donors only."""

import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
sys.dont_write_bytecode = True

import hashlib
import io
import json
import subprocess
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from PIL import Image

COHORT = ROOT / "data/allen_real_development_20260928"
OUTPUT = ROOT / "data/allen_real_training_inputs_20260929"
SIDE, FIELD_UM = 96, 12000.0
API_TO_MODEL_SHIFT_UM = np.full(3, 12.5)
cohort_bindings = {
    "cohort_summary.json": "94ae9603dfed354ecb1c4aa82f57ae79f81657448542878b50b4941f4c87302f",
    "images/receipt.json": "653be9c2357c1461a67e2d997ebd210c622931aceefa3dcd257d44146d4171fa",
    "metadata/receipt.json": "34d5fc1e8b75398b0d8393059327402dbe1a02d3e15235293c3c5054a49244f7",
}
for name, digest in cohort_bindings.items():
    assert hashlib.sha256((COHORT / name).read_bytes()).hexdigest() == digest, name
image_files = {r["relative_path"]: r["sha256"] for r in json.loads((COHORT / "images/receipt.json").read_text())["files"]}
for name in ("images.jsonl", "manifest.json"):
    assert hashlib.sha256((COHORT / "images" / name).read_bytes()).hexdigest() == image_files[name]
image_manifest = json.loads((COHORT / "images/manifest.json").read_text())
for name, digest in image_manifest["source_metadata"]["file_sha256"].items():
    assert hashlib.sha256((COHORT / "metadata" / name).read_bytes()).hexdigest() == digest, name
experiments = {r["experiment_id"]: r for r in map(json.loads, (COHORT / "metadata/experiments.jsonl").read_text().splitlines())}
sections = {r["section_id"]: r for r in map(json.loads, (COHORT / "metadata/sections.jsonl").read_text().splitlines())}
records = [r for r in map(json.loads, (COHORT / "images/images.jsonl").read_text().splitlines()) if r["split"] == "development_train"]
animals = np.asarray([r["animal_id"] for r in records])
assert len(records) == 256 and len(np.unique(animals)) == 58
assert set(animals).isdisjoint({14452, 15219, 15336, 15439, 15447, 15935})
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "preparation_source.py").write_bytes(Path(__file__).read_bytes())
cv2.setNumThreads(4)
images = np.empty((256, 1, SIDE, SIDE), dtype=np.float32)
affine_normal = np.empty((256, 3), dtype=np.float64)
pixel_x, pixel_y = np.meshgrid(np.arange(SIDE), np.arange(SIDE))
pixel_grid = np.stack((pixel_x.ravel(), pixel_y.ravel(), np.ones(SIDE * SIDE)))
provenance = []

for i, record in enumerate(records):
    section = sections[record["section_id"]]
    experiment = experiments[record["experiment_id"]]
    image_bytes = (COHORT / "images" / record["relative_path"]).read_bytes()
    assert hashlib.sha256(image_bytes).hexdigest() == record["sha256"] == image_files[record["relative_path"]]
    red = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.float32)[..., 0] / 255.0
    native_height, native_width = red.shape
    pyramid_scale = 2 ** int(parse_qs(urlparse(record["requested_url"]).query)["downsample"][0])
    resolution = section["resolution_um_per_px"]
    step = FIELD_UM / (SIDE * resolution)
    center_x = (section["width_full_resolution_px"] - 1) / 2
    center_y = (section["height_full_resolution_px"] - 1) / 2
    full_from_model = np.array([[step, 0, center_x - FIELD_UM / (2 * resolution)],
                                [0, step, center_y - FIELD_UM / (2 * resolution)], [0, 0, 1]])
    native_from_full = np.array([[1 / pyramid_scale, 0, 0.5 / pyramid_scale - 0.5],
                                 [0, 1 / pyramid_scale, 0.5 / pyramid_scale - 0.5], [0, 0, 1]])
    native_from_model = native_from_full @ full_from_model
    native_grid = (native_from_model @ pixel_grid)[:2].reshape(2, SIDE, SIDE).astype(np.float32)
    sigma = 0.5 * np.sqrt(max((step / pyramid_scale) ** 2 - 1, 0))
    padding = float(np.median(np.concatenate((red[0], red[-1], red[:, 0], red[:, -1]))))
    filtered = cv2.GaussianBlur(red, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE) if sigma > 0 else red
    images[i, 0] = cv2.remap(filtered, native_grid[0], native_grid[1], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=padding)

    # Upstream affine is retained for provenance/coverage, not invented dense truth.
    tsv = np.asarray(section["alignment2d_tsv"], dtype=np.float64)
    tvr = np.asarray(experiment["alignment3d_tvr"], dtype=np.float64)
    volume_from_full = np.array([[tsv[0], tsv[1], tsv[4]], [tsv[2], tsv[3], tsv[5]],
                                 [0, 0, section["section_number"] * experiment["section_thickness_um"]]])
    pir_from_full = tvr[:9].reshape(3, 3) @ volume_from_full
    pir_from_full[:, 2] += tvr[9:]
    physical_from_model = pir_from_full @ full_from_model
    physical_from_model[:, 2] += API_TO_MODEL_SHIFT_UM
    normal = np.cross(physical_from_model[:, 0], physical_from_model[:, 1])
    affine_normal[i] = normal / np.linalg.norm(normal)
    native_support = ((native_grid[0] >= 0) & (native_grid[0] <= native_width - 1)
                      & (native_grid[1] >= 0) & (native_grid[1] <= native_height - 1))
    provenance.append({
        **record, "array_row_index": i, "actual_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "full_resolution_shape_h_w": [section["height_full_resolution_px"], section["width_full_resolution_px"]],
        "resolution_um_per_full_pixel": resolution, "pyramid_scale": pyramid_scale,
        "model_pixel_to_full_pixel": full_from_model.tolist(),
        "model_pixel_to_downloaded_pixel": native_from_model.tolist(),
        "model_pixel_to_ap_dv_ml_um": physical_from_model.tolist(),
        "alignment2d_tsv": tsv.tolist(), "alignment3d_tvr": tvr.tolist(),
        "section_thickness_um": experiment["section_thickness_um"],
        "antialias_sigma_downloaded_px": float(sigma), "padding_red_value": padding,
        "native_support_fraction": float(native_support.mean()),
        "upstream_affine_center_ap_dv_ml_um": (physical_from_model @ np.array([48., 48., 1.])).tolist(),
        "upstream_affine_normal_ap_dv_ml": affine_normal[i].tolist(),
        "outline_available": False,
    })
    if (i + 1) % 64 == 0:
        print(json.dumps({"training_images_prepared": i + 1, "training_donors_total": 58}), flush=True)

np.save(OUTPUT / "raw_model_input.npy", images)
(OUTPUT / "image_geometry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8")
flat = images[:, 0].reshape(256, -1).astype(np.float64)
std, dynamic = flat.std(-1), np.ptp(flat, axis=-1)
quantiles = np.quantile(flat, [.01, .05, .5, .95, .99], axis=-1).T
signed_normal = affine_normal * np.where(affine_normal[:, :1] < 0, -1., 1.)
ap_angle = np.degrees(np.arctan2(np.linalg.norm(affine_normal[:, 1:], axis=-1), np.abs(affine_normal[:, 0])))
pair_dot = np.clip(np.abs(affine_normal @ affine_normal.T), 0, 1)
pair_angle = np.degrees(np.arccos(pair_dot))
by_animal = {}
for animal in np.unique(animals):
    rows = np.flatnonzero(animals == animal)
    by_animal[str(animal)] = {
        "section_count": len(rows), "section_ids": [records[j]["section_id"] for j in rows],
        "std_min_mean_max": [float(std[rows].min()), float(std[rows].mean()), float(std[rows].max())],
        "dynamic_range_min_mean_max": [float(dynamic[rows].min()), float(dynamic[rows].mean()), float(dynamic[rows].max())],
        "mean_image_q01_q05_q50_q95_q99": quantiles[rows].mean(0).tolist(),
        "native_support_fraction_mean": float(np.mean([provenance[j]["native_support_fraction"] for j in rows])),
        "upstream_affine_ap_axis_angle_deg_min_mean_max": [float(ap_angle[rows].min()), float(ap_angle[rows].mean()), float(ap_angle[rows].max())],
        "upstream_affine_normal_ap_positive_min": signed_normal[rows].min(0).tolist(),
        "upstream_affine_normal_ap_positive_max": signed_normal[rows].max(0).tolist(),
        "upstream_affine_pairwise_antipodal_span_deg": float(pair_angle[np.ix_(rows, rows)].max()),
    }
np.savez(OUTPUT / "input_statistics.npz", animal_id=animals,
         specimen_id=[r["specimen_id"] for r in records], experiment_id=[r["experiment_id"] for r in records],
         section_id=[r["section_id"] for r in records], image_std=std, image_dynamic_range=dynamic,
         image_q01_q05_q50_q95_q99=quantiles, upstream_affine_normal_ap_dv_ml=affine_normal,
         upstream_affine_ap_axis_angle_deg=ap_angle)
summary = {
    "role": "real acquired train-only appearance inputs; upstream affine retained solely as provenance/coverage, no learned/dense labels",
    "source_cohort": str(COHORT), "cohort_bindings": cohort_bindings, "image_manifest_sha256": image_files["manifest.json"],
    "source_metadata_sha256": image_manifest["source_metadata"]["file_sha256"],
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
    "selection": "all and only development_train records in original frozen image-manifest order; no image rejection",
    "split": "development_train", "animal_count": 58, "section_count": 256,
    "excluded_development_donor_ids": [14452, 15219, 15336, 15439, 15447, 15935],
    "preprocessing": "unchanged raw red/255, acquisition-centred12mm96x96, Gaussian antialias+bilinear, acquired-border median padding; no segmentation/crop/normalization tuning",
    "model_pixel_coordinate_contract": "x/96,y/96; full=(downloaded+0.5)*pyramid_scale-0.5; exact transforms and native coverage per row",
    "api_to_model_shift_ap_dv_ml_um": API_TO_MODEL_SHIFT_UM.tolist(), "outline_available": False,
    "array_contract": "float32[256,1,96,96]; IDs and exact array row indices in image_geometry.jsonl; no model features or labels",
    "by_animal": by_animal,
    "equal_donor_mean_image_std": float(np.mean([r["std_min_mean_max"][1] for r in by_animal.values()])),
    "equal_donor_mean_image_quantiles": np.mean([r["mean_image_q01_q05_q50_q95_q99"] for r in by_animal.values()], axis=0).tolist(),
    "image_std_min_max": [float(std.min()), float(std.max())], "image_dynamic_range_min_max": [float(dynamic.min()), float(dynamic.max())],
    "upstream_affine_ap_axis_angle_deg_min_max": [float(ap_angle.min()), float(ap_angle.max())],
    "upstream_affine_pairwise_antipodal_span_deg": float(pair_angle.max()),
    "normal_span_caveat": "upstream per-experiment affine normals, not independent section poses, manual anatomy labels or arbitrary-plane real coverage",
    "appearance_caveat": "full-canvas pixel statistics include acquired/padded background; no inferred tissue/background classification",
    "historical_exposure": "development-only cohort; no joinable complete historical benchmark exclusion list was available",
    "terms": image_manifest["terms"], "numpy_version": np.__version__, "opencv_version": cv2.__version__,
    "output_sha256": {name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() for name in (
        "raw_model_input.npy", "image_geometry.jsonl", "input_statistics.npz",
    )},
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({"output": str(OUTPUT), "sections": 256, "training_donors": 58,
    "equal_donor_mean_image_std": summary["equal_donor_mean_image_std"],
    "equal_donor_mean_image_quantiles": summary["equal_donor_mean_image_quantiles"],
    "affine_normal_ap_angle_range_deg": summary["upstream_affine_ap_axis_angle_deg_min_max"],
    "affine_normal_span_deg": summary["upstream_affine_pairwise_antipodal_span_deg"]}), flush=True)
