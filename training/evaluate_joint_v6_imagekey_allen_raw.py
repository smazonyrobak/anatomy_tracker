"""Hash-locked, post-audit raw-Allen image-key diagnostic; prepare only until bound."""

import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
RUN = ROOT / "runs/joint_v6_imagekey_retrieval_001"
CHECKPOINT = RUN / "joint_model_step_04000.pt"
GALLERY = RUN / "gallery_descriptors_step_04000.npy"
AUDIT = ROOT / "runs/joint_v6_imagekey_retrieval_001_independent_audit/audit.json"
CHECKPOINT_SHA256 = "d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5"
GALLERY_SHA256 = "1568da8f1347641bba4ff93679ff39d5d6250693345db5d915322312bbe64666"
AUDIT_SHA256 = "f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b"
if "UNSET" in (CHECKPOINT_SHA256, GALLERY_SHA256, AUDIT_SHA256):
    raise RuntimeError("Prepare-only: set externally approved checkpoint/gallery/audit hashes after EXIT and a passing independent audit; no run outputs have been read")

os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True

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

COHORT = ROOT / "data/allen_real_development_20260928"
OUTPUT = ROOT / "runs/joint_v6_imagekey_retrieval_001_allen_raw"
BATCH, DEVICE, SIDE, FIELD_UM = 16, "cuda", 96, 12000.0
API_TO_MODEL_SHIFT_UM = np.full(3, 12.5)
audit_bytes = AUDIT.read_bytes()
assert hashlib.sha256(audit_bytes).hexdigest() == AUDIT_SHA256
audit = json.loads(audit_bytes)
assert audit["integrity_passed"] and audit["performance_gates_passed"] and audit["advance_to_joint_integration_pilot"]
bindings = {
    CHECKPOINT: CHECKPOINT_SHA256, GALLERY: GALLERY_SHA256,
    RUN / "catalogue.pt": audit["frozen_run_inventory_sha256"]["catalogue.pt"],
}
for path, expected in bindings.items():
    assert audit["frozen_run_inventory_sha256"][path.name] == expected
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, path
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
config = checkpoint["experiment"]
assert checkpoint["step"] == checkpoint["optimizer_steps_applied"] == 4000
assert config["temperature"] == .1 and config["gallery"]["cell_count"] == 98304
repository = Path(__file__).resolve().parents[1]
inference_sources = (
    "training/arbitrary_plane_joint_model_v6.py",
    "training/arbitrary_plane_recurrent_model_v6.py",
    "training/arbitrary_plane_recurrent_model.py",
    "training/arbitrary_plane_coarse_proposal_v6.py",
    "training/arbitrary_plane_catalogue_runtime_v6.py",
    "training/arbitrary_plane_full_frame_primitives.py",
)
for name in inference_sources:
    assert hashlib.sha256((repository / name).read_bytes()).hexdigest() == config["source"]["file_sha256"][name], name
cohort_bindings = {
    "cohort_summary.json": "94ae9603dfed354ecb1c4aa82f57ae79f81657448542878b50b4941f4c87302f",
    "images/receipt.json": "653be9c2357c1461a67e2d997ebd210c622931aceefa3dcd257d44146d4171fa",
    "metadata/receipt.json": "34d5fc1e8b75398b0d8393059327402dbe1a02d3e15235293c3c5054a49244f7",
}
for name, digest in cohort_bindings.items():
    assert hashlib.sha256((COHORT / name).read_bytes()).hexdigest() == digest, name
image_files = {row["relative_path"]: row["sha256"] for row in json.loads((COHORT / "images/receipt.json").read_text())["files"]}
for name in ("images.jsonl", "manifest.json"):
    assert hashlib.sha256((COHORT / "images" / name).read_bytes()).hexdigest() == image_files[name]
image_manifest = json.loads((COHORT / "images/manifest.json").read_text())
for name, digest in image_manifest["source_metadata"]["file_sha256"].items():
    assert hashlib.sha256((COHORT / "metadata" / name).read_bytes()).hexdigest() == digest, name
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "diagnostic_source.py").write_bytes(Path(__file__).read_bytes())
torch.set_num_threads(4)

experiments = {r["experiment_id"]: r for r in map(json.loads, (COHORT / "metadata/experiments.jsonl").read_text().splitlines())}
sections = {r["section_id"]: r for r in map(json.loads, (COHORT / "metadata/sections.jsonl").read_text().splitlines())}
records = [r for r in map(json.loads, (COHORT / "images/images.jsonl").read_text().splitlines()) if r["split"] == "development_validation"]
assert len(records) == 64 and {r["animal_id"] for r in records} == {14452, 15219, 15336, 15439, 15447, 15935}
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


catalogue = torch.load(RUN / "catalogue.pt", map_location="cpu", weights_only=False)
runtime = catalogue_runtime.make_complete_catalogue_runtime_v6(
    catalogue, expected_catalogue_receipt_sha256=config["catalogue_receipt_sha256"], device=DEVICE, dtype=torch.float32,
)
assert runtime.cell_count == 98304 and runtime.representation_count == 2
model = ArbitraryPlaneJointModelV6(runtime, **config["model_kwargs"]).to(DEVICE)
model.load_state_dict(checkpoint["model_state"], strict=True)
model.eval()
del checkpoint
bank_array = np.load(GALLERY, mmap_mode="r", allow_pickle=False)
assert bank_array.shape == (98304, 2, 256) and bank_array.dtype == np.float16
bank = torch.from_numpy(np.array(bank_array)).to(DEVICE)
cell_log_mass = catalogue["tensors"]["cell_log_mass"][0].to(DEVICE, dtype=torch.float32)
representation_log_weight = catalogue["tensors"]["representation_log_weight"][0].to(DEVICE, dtype=torch.float32)
raw = np.lib.format.open_memmap(OUTPUT / "raw_cell_log_probability.npy", mode="w+", dtype=np.float32, shape=(64, 98304))
raw_components = np.lib.format.open_memmap(OUTPUT / "raw_component_log_score.npy", mode="w+", dtype=np.float32, shape=(64, 98304, 2))
query_descriptors, selected, top_representation_probability = [], [], []
with torch.no_grad():
    for start in range(0, 64, BATCH):
        image = torch.from_numpy(images[start:start + BATCH]).to(DEVICE)
        # Match image-key endpoint evaluation: FP32 encoding, no autocast.
        query = model.pose_model.descriptor_from_features(model.pose_model.encode_histology(
            image, torch.zeros_like(image), torch.zeros(len(image), device=DEVICE),
        ))
        score = model.pose_model.image_key_cosine_logits(query, bank, config["temperature"])
        score = score + representation_log_weight[None]
        cell_score = torch.logsumexp(score, dim=-1) + cell_log_mass[None]
        log_probability = cell_score.log_softmax(dim=-1)
        assert bool(torch.isfinite(score).all() and torch.isfinite(log_probability).all())
        top = torch.argsort(log_probability, dim=-1, descending=True, stable=True)[:, :128]
        raw[start:start + len(image)] = log_probability.cpu().numpy()
        raw_components[start:start + len(image)] = (score + cell_log_mass[None, :, None]).cpu().numpy()
        query_descriptors.append(query.cpu().numpy())
        selected.append(top.cpu().numpy())
        top_representation_probability.append(score[torch.arange(len(image), device=DEVICE)[:, None], top].softmax(-1).cpu().numpy())
        print(json.dumps({"encoded_real_sections": start + len(image)}), flush=True)
raw.flush()
raw_components.flush()
top_indices = np.concatenate(selected)
query_descriptors = np.concatenate(query_descriptors)
top_representation_probability = np.concatenate(top_representation_probability)
normalization_error = float(np.abs(np.logaddexp.reduce(np.asarray(raw, dtype=np.float64), axis=1)).max())
assert normalization_error <= 2e-5
cell_states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
center, frame, _ = full_frame_state_to_components(cell_states)
center, normal = center.numpy(), frame[:, :, 2].numpy()
support_origin = np.asarray(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"])
dot = (normal[top_indices] * truth_normal[:, None]).sum(-1)
angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(normal[top_indices], truth_normal[:, None]), axis=-1), np.abs(dot)))
truth_offset = ((truth_center - support_origin) * truth_normal).sum(-1)
cell_offset = ((center - support_origin) * normal).sum(-1)
offsets = np.abs(cell_offset[top_indices] - np.where(dot < 0, -1, 1) * truth_offset[:, None])
metrics = {
    "plane_angle_deg": angles[:, 0], "normal_offset_error_um": offsets[:, 0],
    **{f"physical_plane_capture_at_{k}": ((angles[:, :k] <= 10.) & (offsets[:, :k] <= 500.)).any(-1).astype(float) for k in (32, 128)},
}
animals = np.array([r["animal_id"] for r in records])
by_animal = {str(animal): {"section_count": int((animals == animal).sum()),
    **{key: float(value[animals == animal].mean()) for key, value in metrics.items()}} for animal in np.unique(animals)}
np.savez(OUTPUT / "plane_predictions.npz", section_id=[r["section_id"] for r in records], animal_id=animals,
         specimen_id=[r["specimen_id"] for r in records], experiment_id=[r["experiment_id"] for r in records],
         selected_cell=top_indices[:, 0], top128_cell_index=top_indices, query_descriptor=query_descriptors,
         top128_representation_probability=top_representation_probability,
         predicted_center_ap_dv_ml_um=center[top_indices[:, 0]], predicted_normal_ap_dv_ml=normal[top_indices[:, 0]],
         reference_center_ap_dv_ml_um=truth_center, reference_normal_ap_dv_ml=truth_normal,
         reference_signed_offset_um=truth_offset, predicted_signed_offset_um=cell_offset[top_indices[:, 0]], **metrics)
summary = {
    "scope": "internal real-image coarse domain diagnostic; not untouched final validation, biological anatomical-accuracy proof or public benchmark",
    "reference": "Allen upstream affine registrations, not blinded expert landmarks, dense deformation or calibrated uncertainty",
    "checkpoint": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_step": 4000,
    "gallery": str(GALLERY), "gallery_sha256": GALLERY_SHA256, "independent_audit": str(AUDIT), "independent_audit_sha256": AUDIT_SHA256,
    "training_experiment": config, "catalogue_receipt_sha256": config["catalogue_receipt_sha256"],
    "cohort": str(COHORT), "cohort_bindings": cohort_bindings, "image_manifest_sha256": image_files["manifest.json"],
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    "inference_source_sha256": {name: config["source"]["file_sha256"][name] for name in inference_sources},
    "preprocessing": "identical existing raw-Allen red/255, acquisition-centred12mm96x96, Gaussian antialias+bilinear, acquired-border median padding, no automatic mask/outline",
    "model_pixel_coordinate_contract": "x/96,y/96; full=(downloaded+0.5)*pyramid_scale-0.5; exact source transforms saved",
    "api_to_model_shift_ap_dv_ml_um": API_TO_MODEL_SHIFT_UM.tolist(), "offset_origin_ap_dv_ml_um": support_origin.tolist(),
    "scoring": "full98304x2 stored gallery, FP32 normalized cosine/0.1 plus representation prior and cell mass once; no candidates removed or truth injected",
    "raw_component_score_semantics": "unnormalized joint log scores including both priors; cell posterior is logsumexp over representation then full-cell normalization",
    "gallery_known_psf": config["gallery"], "query_psf": "acquired thickness retained as reference provenance, not inferred or provided to the image encoder",
    "physical_capture": "same topK candidate has antipodal normal<=10deg AND sign-aligned normaloffset<=500um",
    "section_count": 64, "animal_count": len(by_animal), "by_animal": by_animal,
    "animal_macro": {key: float(np.mean([row[key] for row in by_animal.values()])) for key in metrics},
    "probabilities_calibrated": False, "reflection_accuracy_claim": False, "outline_available": False,
    "raw_log_normalization_error": normalization_error,
    "historical_exposure": "cohort development-only; no joinable historical benchmark donor exclusion list was available; not a new untouched final cohort",
    "torch_version": torch.__version__, "opencv_version": cv2.__version__,
    "output_sha256": {name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() for name in (
        "raw_model_input.npy", "image_geometry.jsonl", "raw_cell_log_probability.npy", "raw_component_log_score.npy", "plane_predictions.npz",
    )},
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({"output": str(OUTPUT), "animal_macro": summary["animal_macro"], "animal_count": len(by_animal)}), flush=True)
