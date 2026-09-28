"""CPU-only normal-capture audit; run only after curriculum003 has exited."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = str(ROOT / "tmp")
os.environ["TMP"] = str(ROOT / "tmp")

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

RUN = ROOT / "runs/joint_v6_proposal_curriculum_003"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001"
STEPS = (0, 20000)
OUTPUT = ROOT / "runs" / f"{RUN.name}_normal_capture_audit"

completed = json.loads((RUN / "completed.json").read_text())
assert completed["steps"] == max(STEPS)
OUTPUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
dev = torch.load(PREPARED / "internal_development_prepared.pt", map_location="cpu", weights_only=False)
catalogue = torch.load(PREPARED / "catalogue.pt", map_location="cpu", weights_only=False)
normal_count = catalogue["counts"]["normal_count"]
offset_count = catalogue["counts"]["offset_count_per_normal"]
roll_count = catalogue["counts"]["roll_count"]
cells_per_normal = offset_count * roll_count
normals = np.asarray(catalogue["arrays"]["cell_normal_ap_dv_ml_float64"])[::cells_per_normal]
_, frame, _ = full_frame_state_to_components(dev["truth_state"])
truth_normal = frame[:, :, 2].numpy()
labels = dev["label"].numpy()
groups = np.array([row["animal_id"] for row in dev["records"]])
modes = np.array([row["selected_mode"] for row in dev["records"]])
identifiable = dev["weight"].numpy() > 0
angular_distance = np.degrees(np.arccos(np.clip(np.abs(truth_normal @ normals.T), 0, 1)))
nearest_normal = angular_distance.argmin(axis=1)
hashes = {}
for path in (RUN / "experiment.json", RUN / "completed.json", PREPARED / "internal_development_prepared.pt", PREPARED / "catalogue.pt"):
    with path.open("rb") as stream:
        hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
summary = {
    "scope": "frozen internal synthetic-group diagnostics, not independent biological animals or calibration",
    "run": str(RUN), "source_sha256": hashes,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "uniform_nll": {"normal": float(np.log(normal_count)), "offset_given_normal": float(np.log(offset_count)), "roll_given_normal_offset": float(np.log(roll_count))},
    "label_quantization": {
        "mean_target_normal_angle_deg": float(angular_distance[np.arange(len(labels)), labels // cells_per_normal].mean()),
        "mean_nearest_normal_angle_deg": float(angular_distance.min(axis=1).mean()),
    },
    "evaluations": [],
}
for step in STEPS:
    path = RUN / f"development_log_probability_step_{step:05d}.npy"
    raw = np.load(path, mmap_mode="r")
    normal_log_probability = np.empty((len(labels), normal_count), dtype=np.float64)
    metrics = {name: np.empty(len(labels)) for name in (
        "joint_nll", "normal_nll", "nearest_geometric_normal_nll", "offset_given_normal_nll",
        "roll_given_normal_offset_nll", "normal_marginal_map_angle_deg", "joint_map_normal_angle_deg",
        "normal_entropy", "normal_map_within_10deg", "normal_posterior_mass_within_10deg",
        "top8_normal_oracle_capture_angle_deg",
    )}
    for start in range(0, len(labels), 32):
        end = min(start + 32, len(labels))
        rows = np.arange(end - start)
        lp = np.asarray(raw[start:end], dtype=np.float64).reshape(-1, normal_count, offset_count, roll_count)
        nd = np.logaddexp.reduce(lp, axis=3)
        nlp = np.logaddexp.reduce(nd, axis=2)
        normal_log_probability[start:end] = nlp
        n = labels[start:end] // cells_per_normal
        d = (labels[start:end] // roll_count) % offset_count
        r = labels[start:end] % roll_count
        angles = angular_distance[start:end]
        normal_map = nlp.argmax(axis=1)
        joint_map = lp.reshape(len(rows), -1).argmax(axis=1) // cells_per_normal
        metrics["joint_nll"][start:end] = -lp[rows, n, d, r]
        metrics["normal_nll"][start:end] = -nlp[rows, n]
        metrics["nearest_geometric_normal_nll"][start:end] = -nlp[rows, nearest_normal[start:end]]
        metrics["offset_given_normal_nll"][start:end] = nlp[rows, n] - nd[rows, n, d]
        metrics["roll_given_normal_offset_nll"][start:end] = nd[rows, n, d] - lp[rows, n, d, r]
        metrics["normal_marginal_map_angle_deg"][start:end] = angles[rows, normal_map]
        metrics["joint_map_normal_angle_deg"][start:end] = angles[rows, joint_map]
        metrics["normal_entropy"][start:end] = -(np.exp(nlp) * nlp).sum(axis=1)
        metrics["normal_map_within_10deg"][start:end] = angles[rows, normal_map] <= 10
        metrics["normal_posterior_mass_within_10deg"][start:end] = (np.exp(nlp) * (angles <= 10)).sum(axis=1)
        top8 = np.argsort(-nlp, axis=1, kind="stable")[:, :8]
        metrics["top8_normal_oracle_capture_angle_deg"][start:end] = np.take_along_axis(angles, top8, axis=1).min(axis=1)
    subsets = {"all": np.ones(len(labels), dtype=bool), "identifiable": identifiable, "censored": ~identifiable}
    subsets.update({mode: identifiable & (modes == mode) for mode in np.unique(modes)})
    by_subset = {}
    for name, selected in subsets.items():
        group_ids = np.unique(groups[selected])
        by_group = {group: {key: float(value[selected & (groups == group)].mean()) for key, value in metrics.items()} for group in group_ids}
        by_subset[name] = {
            "section_count": int(selected.sum()), "synthetic_group_count": len(group_ids),
            "section_mean": {key: float(value[selected].mean()) for key, value in metrics.items()},
            "synthetic_group_macro": {key: float(np.mean([row[key] for row in by_group.values()])) for key in metrics},
            "by_synthetic_group": by_group,
        }
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    evaluation = {"step": step, "raw_probability_file": str(path), "raw_probability_sha256": digest, "by_subset": by_subset}
    summary["evaluations"].append(evaluation)
    np.savez(OUTPUT / f"normal_rows_step_{step:05d}.npz", section_id=[r["section_id"] for r in dev["records"]],
             synthetic_group_id=groups, input_mode=modes, identifiable=identifiable,
             normal_log_probability=normal_log_probability, **metrics)
    print(json.dumps({"step": step, "all": by_subset["all"]["synthetic_group_macro"], "identifiable": by_subset["identifiable"]["synthetic_group_macro"]}), flush=True)
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
