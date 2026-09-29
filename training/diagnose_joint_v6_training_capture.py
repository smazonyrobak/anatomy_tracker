"""Bounded CPU fit diagnostic on completed003, not held-out qualification."""

import os
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = str(ROOT / "tmp")
os.environ["TMP"] = str(ROOT / "tmp")

import hashlib
import json
import numpy as np
import torch

from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

RUN = ROOT / "runs/joint_v6_proposal_curriculum_003"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001"
OUTPUT = ROOT / "runs/joint_v6_proposal_curriculum_003_training_capture"
OUTPUT.mkdir(exist_ok=False)
torch.set_num_threads(4)
checkpoint = torch.load(RUN / "joint_model_step_20000.pt", map_location="cpu", weights_only=False)
catalogue = torch.load(RUN / "catalogue.pt", map_location="cpu", weights_only=False)
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cpu", dtype=torch.float32)
model = ArbitraryPlaneJointModelV6(runtime, **checkpoint["experiment"]["model_kwargs"]).eval()
model.load_state_dict(checkpoint["model_state"])
train = torch.load(PREPARED / "training_prepared.pt", map_location="cpu", weights_only=False)
generated = torch.load(RUN / "first_generated_batch.pt", map_location="cpu", weights_only=False)
schedule = np.load(RUN / "generated_schedule.npz")
indices = np.linspace(0, len(train["label"]) - 1, 64).astype(int)
conditions = {
    "frozen_training_fixed64": (train["channels"][indices], train["label"][indices], train["truth_state"][indices], train["weight"][indices].numpy() > 0),
    "generated_training_first8": (generated["input_channels"], torch.as_tensor(schedule["cell_index"][:8]), generated["state"], (np.array(generated["support_mass"]) >= 64) & (np.array(generated["visible_mass"]) >= 64)),
}
states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
normals = np.asarray(catalogue["arrays"]["cell_normal_ap_dv_ml_float64"])[::256]
summary = {"scope": "fit diagnostic on trained observations only; not a validation score or independent animals", "inference_precision": "CPU float32; checkpoint trained with encoder AMP and FP32 proposal head", "frozen_training_indices": indices.tolist(), "probabilities_calibrated": False, "conditions": {}}
with (RUN / "joint_model_step_20000.pt").open("rb") as stream:
    summary["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
for name, (inputs, labels, truth, supervised) in conditions.items():
    raw = []
    with torch.no_grad():
        for start in range(0, len(inputs), 4):
            x = inputs[start:start + 4]
            out = model.pose_model.forward_proposal_only(x[:, :1], x[:, 1:2], x[:, 2].mean((-2, -1)), runtime.expand(len(x)), (96, 96))
            raw.append(out["raw_full_catalogue_cell_log_probability"].numpy())
    raw = np.concatenate(raw)
    normal_lp = np.logaddexp.reduce(raw.reshape(len(inputs), 384, 256), axis=-1)
    _, frame, _ = full_frame_state_to_components(truth)
    angular = np.degrees(np.arccos(np.clip(np.abs(frame[:, :, 2].numpy() @ normals.T), 0, 1)))
    prediction = normal_lp.argmax(-1)
    rows = np.arange(len(inputs))
    rank = (raw > raw[rows, labels, None]).sum(-1) + 1
    metrics = {
        "joint_nll": -raw[rows, labels], "normal_marginal_nll": -normal_lp[rows, labels // 256],
        "normal_marginal_map_angle_deg": angular[rows, prediction],
        "normal_map_within_10deg": angular[rows, prediction] <= 10,
        "joint_hit_at128": rank <= 128,
    }
    summary["conditions"][name] = {"row_count": len(inputs), "supervised_count": int(supervised.sum()), "all_row_mean": {k: float(v.mean()) for k, v in metrics.items()}, "supervised_row_mean": {k: float(v[supervised].mean()) for k, v in metrics.items()}}
    np.savez(OUTPUT / f"{name}.npz", labels=labels, supervised=supervised, normal_log_probability=normal_lp, **metrics)
    print(json.dumps({"condition": name, **summary["conditions"][name]}), flush=True)
summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
