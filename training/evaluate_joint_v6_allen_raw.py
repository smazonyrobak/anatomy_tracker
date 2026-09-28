"""Coarse donor-level Allen diagnostic; run only after the checkpoint's writer exits."""

import os
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = str(ROOT / "tmp")
os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache" / "torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache" / "cuda")

import hashlib
import io
import json
import subprocess
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
from PIL import Image

from training import arbitrary_plane_catalogue_runtime_v6 as catalogue_runtime
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

CHECKPOINT = ROOT / "runs/joint_v6_proposal_substantive_001/joint_model_step_10000.pt"
COHORT = ROOT / "data/allen_real_development_20260928"
OUTPUT = ROOT / "runs" / f"{CHECKPOINT.parent.name}_{CHECKPOINT.stem}_allen_raw"
BATCH = 16
DEVICE = "cuda"
SIDE = 96
FIELD_UM = 12000.0
API_TO_MODEL_SHIFT_UM = np.full(3, 12.5)

OUTPUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(8)
experiments = {r["experiment_id"]: r for r in map(json.loads, (COHORT / "metadata/experiments.jsonl").read_text().splitlines())}
sections = {r["section_id"]: r for r in map(json.loads, (COHORT / "metadata/sections.jsonl").read_text().splitlines())}
records = [r for r in map(json.loads, (COHORT / "images/images.jsonl").read_text().splitlines()) if r["split"] == "development_validation"]
images = np.empty((len(records), 1, SIDE, SIDE), dtype=np.float32)
truth_center = np.empty((len(records), 3), dtype=np.float64)
truth_normal = np.empty_like(truth_center)
pixel_x, pixel_y = np.meshgrid(np.arange(SIDE), np.arange(SIDE))
pixel_grid = np.stack((pixel_x.ravel(), pixel_y.ravel(), np.ones(SIDE * SIDE)))
provenance = []

for i, record in enumerate(records):
    section = sections[record["section_id"]]
    experiment = experiments[record["experiment_id"]]
    image_bytes = (COHORT / "images" / record["relative_path"]).read_bytes()
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

    # Allen's documented image_to_reference: image -> section volume -> PIR µm.
    tsv = np.asarray(section["alignment2d_tsv"], dtype=np.float64)
    tvr = np.asarray(experiment["alignment3d_tvr"], dtype=np.float64)
    volume_from_full = np.array([[tsv[0], tsv[1], tsv[4]], [tsv[2], tsv[3], tsv[5]],
                                 [0, 0, section["section_number"] * experiment["section_thickness_um"]]])
    pir_from_full = tvr[:9].reshape(3, 3) @ volume_from_full
    pir_from_full[:, 2] += tvr[9:]
    physical_from_model = pir_from_full @ full_from_model
    physical_from_model[:, 2] += API_TO_MODEL_SHIFT_UM
    truth_center[i] = physical_from_model @ np.array([SIDE / 2, SIDE / 2, 1])
    normal = np.cross(physical_from_model[:, 0], physical_from_model[:, 1])
    truth_normal[i] = normal / np.linalg.norm(normal)
    native_support = ((native_grid[0] >= 0) & (native_grid[0] <= native_width - 1)
                      & (native_grid[1] >= 0) & (native_grid[1] <= native_height - 1))
    provenance.append({
        **record, "actual_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "full_resolution_shape_h_w": [section["height_full_resolution_px"], section["width_full_resolution_px"]],
        "resolution_um_per_full_pixel": resolution, "pyramid_scale": pyramid_scale,
        "model_pixel_to_full_pixel": full_from_model.tolist(),
        "model_pixel_to_downloaded_pixel": native_from_model.tolist(),
        "model_pixel_to_ap_dv_ml_um": physical_from_model.tolist(),
        "alignment2d_tsv": tsv.tolist(), "alignment3d_tvr": tvr.tolist(),
        "section_thickness_um": experiment["section_thickness_um"],
        "antialias_sigma_downloaded_px": float(sigma), "padding_red_value": padding,
        "native_support_fraction": float(native_support.mean()),
        "truth_center_ap_dv_ml_um": truth_center[i].tolist(), "truth_normal_ap_dv_ml": truth_normal[i].tolist(),
    })

np.save(OUTPUT / "raw_model_input.npy", images)
(OUTPUT / "image_geometry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8")
print(f"Prepared {len(records)} unsegmented real sections from {len({r['animal_id'] for r in records})} development donors", flush=True)

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
config = checkpoint["experiment"]
catalogue = torch.load(CHECKPOINT.parent / "catalogue.pt", map_location="cpu", weights_only=False)
runtime = catalogue_runtime.make_complete_catalogue_runtime_v6(
    catalogue, expected_catalogue_receipt_sha256=config["catalogue_receipt_sha256"],
    device=DEVICE, dtype=torch.float32,
)
model = ArbitraryPlaneJointModelV6(runtime, **config["model_kwargs"]).to(DEVICE)
model.load_state_dict(checkpoint["model_state"], strict=True)
model.eval()
raw = np.lib.format.open_memmap(OUTPUT / "raw_cell_log_probability.npy", mode="w+", dtype=np.float32,
                               shape=(len(records), runtime.cell_count))
with torch.no_grad():
    for start in range(0, len(records), BATCH):
        image = torch.from_numpy(images[start:start + BATCH]).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            prediction = model.pose_model.forward_proposal_only(
                image, torch.zeros_like(image), torch.zeros(len(image), device=DEVICE),
                runtime.expand(len(image)), tuple(config["retrieval_shape_h_w"]),
            )
        raw[start:start + len(image)] = prediction["raw_full_catalogue_cell_log_probability"].float().cpu().numpy()
raw.flush()
selected_cell = np.asarray(raw.argmax(axis=1))
cell_states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
center, frame, _ = full_frame_state_to_components(cell_states[selected_cell])
center, normal = center.numpy(), frame[:, :, 2].numpy()
support_origin = np.asarray(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"])
dot = np.sum(normal * truth_normal, axis=1)
sign = np.where(dot < 0, -1, 1)
truth_offset = np.sum((truth_center - support_origin) * truth_normal, axis=1)
predicted_offset = np.sum((center - support_origin) * normal, axis=1)
metrics = {
    "plane_angle_deg": np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1))),
    "normal_offset_error_um": np.abs(predicted_offset - sign * truth_offset),
}
animals = np.array([r["animal_id"] for r in records])
by_animal = {str(animal): {"section_count": int((animals == animal).sum()),
    **{key: float(value[animals == animal].mean()) for key, value in metrics.items()}} for animal in np.unique(animals)}
summary = {
    "scope": "internal real-image domain diagnostic; not untouched final validation or a public benchmark",
    "reference": "Allen upstream affine registration; not blinded expert anatomical or dense deformation truth",
    "checkpoint": str(CHECKPOINT), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    "checkpoint_step": checkpoint["step"], "training_experiment": config,
    "catalogue_receipt_sha256": config["catalogue_receipt_sha256"],
    "cohort": str(COHORT), "cohort_summary_sha256": hashlib.sha256((COHORT / "cohort_summary.json").read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
    "preprocessing": "unsegmented equalized JPEG red/255; acquisition-centred12mm field96x96; Gaussian antialias+bilinear; edge-median padding; no outline",
    "model_pixel_coordinate_contract": "x/96,y/96; full=(downloaded+0.5)*32-0.5; exact per-image maps saved",
    "api_to_model_shift_ap_dv_ml_um": API_TO_MODEL_SHIFT_UM.tolist(),
    "offset_origin_ap_dv_ml_um": support_origin.tolist(),
    "section_count": len(records), "animal_count": len(by_animal), "by_animal": by_animal,
    "animal_macro": {key: float(np.mean([row[key] for row in by_animal.values()])) for key in metrics},
    "probabilities_calibrated": False, "outline_available": False,
    "torch_version": torch.__version__, "opencv_version": cv2.__version__,
}
np.savez(OUTPUT / "plane_predictions.npz", section_id=[r["section_id"] for r in records], animal_id=animals,
         specimen_id=[r["specimen_id"] for r in records], experiment_id=[r["experiment_id"] for r in records],
         selected_cell=selected_cell, predicted_center_ap_dv_ml_um=center, predicted_normal_ap_dv_ml=normal,
         reference_center_ap_dv_ml_um=truth_center, reference_normal_ap_dv_ml=truth_normal,
         reference_signed_offset_um=truth_offset, predicted_signed_offset_um=predicted_offset, **metrics)
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({"output": str(OUTPUT), "animal_macro": summary["animal_macro"], "animal_count": len(by_animal)}), flush=True)
