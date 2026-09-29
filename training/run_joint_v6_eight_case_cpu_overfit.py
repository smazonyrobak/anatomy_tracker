"""Eight fixed real synthetic observations: CPU memorization diagnostic only."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = str(ROOT / "tmp")
os.environ["TMP"] = str(ROOT / "tmp")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

RUN = ROOT / "runs/joint_v6_eight_case_cpu_overfit_002"
PACK = ROOT / "data/joint_v6_local_refinement_frozen_001"
BASELINE = ROOT / "runs/joint_v6_proposal_substantive_001"
CURRICULUM = ROOT / "runs/joint_v6_proposal_curriculum_003"
SEED = 2026092908
STEPS = 1000
RETRIEVAL_SHAPE = (96, 96)

RUN.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.manual_seed(SEED)
np.random.seed(SEED)
process = psutil.Process()
prepared = torch.load(PACK / "training.pt", map_location="cpu", weights_only=False)
foreground = prepared["source_tissue_ground_truth_mask"].sum((1, 2, 3))
eligible = (prepared["pose_supervision_weight"] == 1) & (prepared["dense_deformation_supervision_weight"] == 1) & (foreground > 1000)
modes = [row["selected_mode"] for row in prepared["records"]]
selected = []
selected_labels = set()
for index in range(8):
    mode = ("smart-brush-absent", "smart-brush-accurate", "smart-brush-imperfect")[index % 3]
    row = next(row for row in range(len(modes)) if eligible[row] and modes[row] == mode and int(prepared["truth_catalogue_index"][row]) not in selected_labels)
    selected.append(row)
    selected_labels.add(int(prepared["truth_catalogue_index"][row]))
inputs = prepared["channels"][selected].clone()
labels = prepared["truth_catalogue_index"][selected].clone()
truth = prepared["truth_state"][selected].clone()
records = [prepared["records"][row] for row in selected]
foreground = foreground[selected].tolist()
original_indices = prepared["prepared_row_index"][selected].tolist()
del prepared
catalogue = torch.load(BASELINE / "catalogue.pt", map_location="cpu", weights_only=False)
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cpu", dtype=torch.float32)
cell_normals = torch.as_tensor(catalogue["arrays"]["cell_normal_ap_dv_ml_float64"])
_, truth_frames, _ = full_frame_state_to_components(truth)
truth_normals = truth_frames[:, :, 2]
model_kwargs = json.loads((CURRICULUM / "experiment.json").read_text())["model_kwargs"]
model_kwargs["proposal_normal_readout_count"] = None
torch.manual_seed(SEED)
model = ArbitraryPlaneJointModelV6(runtime, **model_kwargs)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
repository = Path(__file__).resolve().parents[1]
source_files = (
    "training/run_joint_v6_eight_case_cpu_overfit.py",
    "training/arbitrary_plane_joint_model_v6.py",
    "training/arbitrary_plane_coarse_proposal_v6.py",
    "training/arbitrary_plane_recurrent_model.py",
)
experiment = {
    "scope": "eight fixed training observations; memorization diagnostic, no generalization or calibration claim",
    "initialization": "fresh complete joint model; no loaded model weights, features or pseudolabels",
    "seed": SEED, "model_kwargs": model_kwargs, "device": "cpu", "threads": 4,
    "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
    "loss": "mean full98304-cell joint NLL; no auxiliary normal loss",
    "batch_size": 4, "schedule": "alternating fixed rows0:4 and4:8",
    "retrieval_shape_h_w": RETRIEVAL_SHAPE, "maximum_steps": STEPS,
    "budget_guard": "at100 stop if elapsed experiment time predicts >300seconds at1000",
    "selection": "first eligible rows cycling three input modes; unique catalogue labels; pose=dense=1,tissue>1000pixels",
    "pack_row_indices": selected, "prepared_row_indices": original_indices,
    "labels": labels.tolist(), "foreground_pixels": foreground, "records": records,
    "pack_receipt": json.loads((PACK / "pack.json").read_text())["partitions"]["training"],
    "catalogue_receipt_sha256": catalogue["receipt_sha256"],
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    "source_sha256": {path: hashlib.sha256((repository / path).read_bytes()).hexdigest() for path in source_files},
}
(RUN / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
print(json.dumps({"selected_pack_rows": selected, "labels": labels.tolist(), "foreground_pixels": foreground, "cpu_threads": 4}), flush=True)
started = time.perf_counter()
status = "completed1000"
with (RUN / "metrics.jsonl").open("w", encoding="utf-8") as trace:
    for step in range(STEPS + 1):
        if step:
            batch = slice(0, 4) if step % 2 else slice(4, 8)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model.pose_model.forward_proposal_only(inputs[batch, :1], inputs[batch, 1:2], inputs[batch, 2].mean((-2, -1)), runtime.expand(4), RETRIEVAL_SHAPE)
            loss = F.nll_loss(output["raw_full_catalogue_cell_log_probability"], labels[batch])
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                raw = torch.cat([model.pose_model.forward_proposal_only(inputs[start:start + 4, :1], inputs[start:start + 4, 1:2], inputs[start:start + 4, 2].mean((-2, -1)), runtime.expand(4), RETRIEVAL_SHAPE)["raw_full_catalogue_cell_log_probability"] for start in (0, 4)])
                nll = F.nll_loss(raw, labels, reduction="none")
                predicted = raw.argmax(1)
                normal_error = torch.rad2deg(torch.acos((cell_normals[predicted] * truth_normals).sum(1).abs().clamp(0, 1)))
                normal_log = torch.logsumexp(raw.reshape(8, 384, 256), dim=2)
                normal_nll = F.nll_loss(normal_log, labels // 256)
            elapsed = time.perf_counter() - started
            metric = {"step": step, "seconds": elapsed, "mean_joint_nll": float(nll.mean()), "normal_nll": float(normal_nll), "mean_normal_error_deg": float(normal_error.mean()), "exact_cell_recall": float((predicted == labels).float().mean()), "per_row_nll": nll.tolist(), "per_row_normal_error_deg": normal_error.tolist(), "predicted_cell": predicted.tolist(), "rss_bytes": process.memory_info().rss, "peak_working_set_bytes": process.memory_info().peak_wset}
            if step:
                metric["last_gradient_norm"] = float(gradient)
                metric["projected1000_seconds"] = elapsed * STEPS / step
            trace.write(json.dumps(metric) + "\n")
            trace.flush()
            print(json.dumps(metric), flush=True)
            if step == 100 and elapsed * 10 > 300:
                status = "stopped100_cpu_budget"
                break
np.save(RUN / "final_raw_log_probability.npy", raw.numpy())
(RUN / "completed.json").write_text(json.dumps({"status": status, "steps": step, "seconds": time.perf_counter() - started, "final_metrics": metric}, indent=2), encoding="utf-8")
print(f"Eight-case CPU diagnostic finished: {status}", flush=True)
