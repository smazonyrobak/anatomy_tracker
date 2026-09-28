"""Inspect the completed proposal experiment using a fixed, unselected case panel."""

import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = r"I:\AnatomyTracker\cache\matplotlib"
os.environ["TEMP"] = r"I:\AnatomyTracker\tmp"
os.environ["TMP"] = os.environ["TEMP"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
import torch

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training import arbitrary_plane_finite_row_binding_v6 as rows
from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_plane, full_frame_state_to_components

run = Path(r"I:\AnatomyTracker\runs\joint_v6_proposal_capacity_002")
cache = Path(r"I:\AnatomyTracker\runs\arbitrary_plane_finite_v6_substantive_data_001\internal_development_cache")
step = 10000
indices = [0, 53, 106, 159, 212, 265, 318, 371, 424, 477, 530, 583]
prepared = torch.load(run / "internal_development_prepared.pt", map_location="cpu", weights_only=False)
catalogue = torch.load(run / "catalogue.pt", map_location="cpu", weights_only=False)
metrics = np.load(run / f"development_rows_step_{step:05d}.npz")
atlas, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas)
geometry = catalogue["support_geometry"]
states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
selected = rows.load_frozen_training_rows_v6(
    cache, indices,
    expected_manifest_receipt_sha256="22d6ae273f13974250d74a2f5b32446a49a1c2c3051d40ad45706300aa1159de",
)["rows"]

fig, axes = plt.subplots(len(indices), 3, figsize=(9, 2.35 * len(indices)))
for axis_row, index, row in zip(axes, indices, selected):
    contract = row["finite_psf_contract"]
    state = torch.stack((prepared["truth_state"][index], states[int(metrics["prediction"][index])]))
    with torch.no_grad():
        planes = render_finite_thickness_plane(
            atlas, state, (96, 96), geometry["origin_ap_dv_ml_um"],
            geometry["voxel_size_ap_dv_ml_um"],
            torch.tensor(contract["axial_offsets_um"]), torch.tensor(contract["axial_weights"]),
        ).numpy()
    axis_row[0].imshow(prepared["channels"][index, 0], cmap="gray", vmin=0, vmax=1)
    axis_row[0].set_title(f"Row {index}: {row['selected_mode']}\n{row['lineage']['animal_id']}", fontsize=7)
    axis_row[1].imshow(planes[0, 0], cmap="gray", vmin=0, vmax=1)
    axis_row[1].set_title("Reference plane, before deformation", fontsize=8)
    axis_row[2].imshow(planes[1, 0], cmap="gray", vmin=0, vmax=1)
    axis_row[2].set_title(
        f"Coarse MAP: normal {metrics['plane_angle_deg'][index]:.1f}°\n"
        f"offset {metrics['normal_offset_error_um'][index]:.0f} µm; rank {metrics['truth_rank'][index]}",
        fontsize=8,
    )
    for ax in axis_row:
        ax.axis("off")
fig.suptitle(
    f"{run.name}: held-out synthetic sample groups\n"
    "Canonical atlas renders; raster reflections and nonlinear deformation are not applied",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.98))
path = run / f"development_fixed_panel_step_{step:05d}.png"
fig.savefig(path, dpi=140)
print(path)

trace = [json.loads(line) for line in (run / "training_trace.jsonl").read_text().splitlines()]
evaluations = [json.loads(path.read_text()) for path in sorted(run.glob("development_metrics_step_*.json"))]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
training_loss = np.array([row["weighted_nll"] for row in trace])
axes[0].plot(np.arange(100, len(trace) + 1, 100), training_loss[:len(trace) // 100 * 100].reshape(-1, 100).mean(1), label="Training")
axes[0].plot([r["step"] for r in evaluations], [r["by_support"]["identifiable"]["animal_macro"]["nll"] for r in evaluations], label="Held-out animals")
axes[0].set_ylabel("Nearest-cell NLL")
axes[0].legend()
for k in (8, 32, 128):
    axes[1].plot([r["step"] for r in evaluations], [r["by_support"]["identifiable"]["animal_macro"][f"hit_at_{k}"] for r in evaluations], label=f"Top {k}")
axes[1].set_ylabel("Held-out nearest-cell recall")
axes[1].legend()
axes[2].plot([r["step"] for r in evaluations], [r["by_support"]["identifiable"]["animal_macro"]["plane_angle_deg"] for r in evaluations])
axes[2].set_ylabel("Held-out MAP normal error (degrees)")
for ax in axes:
    ax.set_xlabel("Optimization step")
fig.tight_layout()
fig.savefig(run / "learning_curves.png", dpi=160)

log_probability = np.load(run / f"development_log_probability_step_{step:05d}.npy", mmap_mode="r")
top = np.argpartition(log_probability, -32, axis=1)[:, -32:]
_, frames, _ = full_frame_state_to_components(states[top])
_, truth_frames, _ = full_frame_state_to_components(prepared["truth_state"])
normal_error = torch.rad2deg(torch.acos((frames[..., :, 2] * truth_frames[:, None, :, 2]).sum(-1).abs().clamp(0, 1)))
best_normal = normal_error.min(1).values.numpy()
animals = np.array([row["animal_id"] for row in prepared["records"]])
valid = prepared["weight"].numpy() > 0
summary = {
    "scope": "descriptive best-of-top32 capture diagnostic; truth-selected, not achieved model accuracy",
    "step": step,
    "animal_macro_best_normal_error_deg": float(np.mean([best_normal[valid & (animals == animal)].mean() for animal in np.unique(animals[valid])])),
    "best_normal_error_deg_by_row": best_normal.tolist(),
}
(run / "top32_capture_diagnostic.json").write_text(json.dumps(summary, indent=2))
print({key: value for key, value in summary.items() if not key.endswith("by_row")})
