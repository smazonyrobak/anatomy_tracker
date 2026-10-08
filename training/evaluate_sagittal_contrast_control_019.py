"""One TRAIN-derived fixed-gain control on frozen weak sagittal DEV."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(root / "tmp")
os.environ["TORCH_HOME"] = str(root / "cache/torch")
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel


source = root / "data/allen_sagittal_ish_weak_inputs_001_20261008"
baseline = root / "runs/sagittal_weak_transfer_019_001"
checkpoint_path = root / "runs/one_shot_exposure_019/joint_step_18000.pt"
out = root / "runs/sagittal_contrast_control_019_001"
protocol_path = Path(__file__).resolve().parents[1] / "docs/publication/SAGITTAL_CONTRAST_CONTROL_019_PROTOCOL_20261008.md"
source_summary = json.loads((source / "summary.json").read_text())
baseline_summary = json.loads((baseline / "summary.json").read_text())
assert hashlib.sha256((source / "summary.json").read_bytes()).hexdigest() == "811106150f042a5b98b59e51754747192d1731450f014bc9e56343d478a78050"
assert hashlib.sha256((source / "model_input.npy").read_bytes()).hexdigest() == source_summary["output_sha256"]["model_input.npy"]
assert hashlib.sha256((source / "geometry.jsonl").read_bytes()).hexdigest() == source_summary["output_sha256"]["geometry.jsonl"]
assert hashlib.sha256((baseline / "rows.jsonl").read_bytes()).hexdigest() == baseline_summary["rows_sha256"]
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == baseline_summary["input_sha256"]["checkpoint"]
records = [json.loads(line) for line in (source / "geometry.jsonl").read_text().splitlines() if json.loads(line)["split"] == "weak_dev"]
baseline_rows = [json.loads(line) for line in (baseline / "rows.jsonl").read_text().splitlines()]
assert len(records) == len(baseline_rows) == 80
assert [row["section_id"] for row in records] == [row["section_id"] for row in baseline_rows]
images = np.load(source / "model_input.npy", mmap_mode="r")
gain = 4.259364821544654

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
assert checkpoint["step"] == 18000 and not checkpoint["calibrated"]
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
model.load_state_dict(checkpoint["model"], strict=True)
del checkpoint
pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.], [127.5, 127.5]], device="cuda")
chart = pixels / 256
reflected_chart = chart.clone()
reflected_chart[:, 0] = 255 / 256 - chart[:, 0]
uv = torch.stack((chart, reflected_chart)) - .5

out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode():
    for index, record in enumerate(records):
        image = np.concatenate((np.clip(gain * np.asarray(images[record["array_row_index"]]), 0, 1).astype(np.float32),
                                np.zeros((4, 256, 256), dtype=np.float32)))[None]
        prediction = model.predict(torch.from_numpy(image).cuda())
        centre, frame, basis = full_frame_state_to_components(prediction["state"])
        tangent = frame[..., :2] @ basis
        fitted = centre[0, :, None, None, :] + torch.einsum("mij,rpj->mrpi", tangent[0], uv)
        affine = torch.as_tensor(record["model_pixel_to_ccf_ref9_ap_dv_ml_um"], device="cuda", dtype=torch.float32)
        reference = affine[:, 2] + pixels[:, :1] * affine[:, 0] + pixels[:, 1:] * affine[:, 1]
        error = (fitted - reference).norm(dim=-1).mean(-1)
        normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
        angle = torch.rad2deg((frame[0, :, :, 2] @ normal).abs().clamp(0, 1).acos())
        prior = prediction["log_mass"][0, :, None] + torch.stack((
            F.logsigmoid(-prediction["reflection_logit"][0]),
            F.logsigmoid(prediction["reflection_logit"][0])), -1)
        selected_mode, selected_reflection = divmod(int(prior.flatten().argmax()), 2)
        rows.append({"donor_id": record["donor_id"], "specimen_id": record["specimen_id"],
                     "experiment_id": record["experiment_id"], "section_id": record["section_id"],
                     "image_sha256": record["image_sha256"], "selected_mode": selected_mode,
                     "selected_reflection": selected_reflection,
                     "selected_five_point_um": float(error[selected_mode, selected_reflection]),
                     "oracle_best_branch_five_point_um": float(error.min()),
                     "selected_normal_deg": float(angle[selected_mode]),
                     "oracle_best_normal_deg": float(angle.min()),
                     "uncalibrated_mode_mass_within_15deg": float(prediction["log_mass"][0].exp()[angle <= 15].sum()),
                     "all_branch_five_point_um": error.cpu().tolist(),
                     "all_mode_normal_deg": angle.cpu().tolist(),
                     "all_branch_prior_log_mass": prior.cpu().tolist()})
        if (index + 1) % 20 == 0:
            print(f"evaluated fixed-gain control {index + 1}/80", flush=True)

(out / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
names = ("selected_five_point_um", "oracle_best_branch_five_point_um", "selected_normal_deg",
         "oracle_best_normal_deg", "uncalibrated_mode_mass_within_15deg")
donors = sorted({row["donor_id"] for row in rows})
by_donor = {str(donor): {name: float(np.mean([row[name] for row in rows if row["donor_id"] == donor]))
                         for name in names} for donor in donors}
donor_equal = {name: float(np.mean([by_donor[str(donor)][name] for donor in donors])) for name in names}
baseline_mean = baseline_summary["donor_equal"]
improvement = {"selected_normal_deg": baseline_mean["selected_normal_deg"] - donor_equal["selected_normal_deg"],
               "selected_five_point_um": baseline_mean["selected_five_point_um"] - donor_equal["selected_five_point_um"]}
summary = {"version": "sagittal-contrast-control-019-001", "gain": gain, "rows": len(rows),
           "by_donor": by_donor, "donor_equal": donor_equal, "paired_baseline": baseline_mean,
           "improvement": improvement,
           "predeclared_major_contributor_gate": bool(improvement["selected_normal_deg"] >= 20
                                                     and improvement["selected_five_point_um"] >= 2000),
           "reference": "inherited weak Allen affine; not independently adjudicated anatomical truth",
           "input_sha256": {"source_summary": hashlib.sha256((source / "summary.json").read_bytes()).hexdigest(),
                            "source_array": source_summary["output_sha256"]["model_input.npy"],
                            "source_geometry": source_summary["output_sha256"]["geometry.jsonl"],
                            "baseline_rows": baseline_summary["rows_sha256"],
                            "checkpoint": baseline_summary["input_sha256"]["checkpoint"],
                            "protocol": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                            "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
           "rows_sha256": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"donor_equal": donor_equal, "improvement": improvement,
                  "major_contributor_gate": summary["predeclared_major_contributor_gate"]}, indent=2), flush=True)
