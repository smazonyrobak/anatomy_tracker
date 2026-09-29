"""CPU-only frozen 006 diagnostic; run only AFTER root confirms its process exited."""

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TEMP"] = r"I:\AnatomyTracker\tmp"
os.environ["TMP"] = r"I:\AnatomyTracker\tmp"

import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

torch.set_num_threads(4)
root = Path(r"I:\AnatomyTracker\runs")
prepared = root / "joint_v6_proposal_substantive_001"
runs = {
    "normal_objective_005": root / "joint_v6_proposal_normal_objective_005",
    "normal_readout_006": root / "joint_v6_normal_readout_control_006",
}
output = root / "joint_v6_normal_readout_collapse_audit_005_006"
output.mkdir(exist_ok=False)
train = torch.load(prepared / "training_prepared.pt", map_location="cpu", weights_only=False)
dev = torch.load(prepared / "internal_development_prepared.pt", map_location="cpu", weights_only=False)
catalogue = torch.load(prepared / "catalogue.pt", map_location="cpu", weights_only=False)
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cpu", dtype=torch.float32)
eligible = np.flatnonzero(train["weight"].numpy() > 0)
selected = eligible[np.linspace(0, len(eligible) - 1, 8).astype(int)]
inputs = train["channels"][selected].float()
labels = dev["label"].numpy()
groups = np.array([row["animal_id"] for row in dev["records"]])
normal_count = catalogue["counts"]["normal_count"]
cells_per_normal = catalogue["counts"]["offset_count_per_normal"] * catalogue["counts"]["roll_count"]
normal_grid = catalogue["arrays"]["cell_normal_ap_dv_ml_float64"][::cells_per_normal]
_, truth_frame, _ = full_frame_state_to_components(dev["truth_state"].double())
normal_angles = np.rad2deg(np.arccos(np.abs(truth_frame[:, :, 2].numpy() @ normal_grid.T).clip(0, 1)))
result = {
    "scope": "CPU FP32 feature/context-only diagnostic on eight fixed prepared training images; frozen raw development predictions at4000 attempted updates; no new GPU inference or benchmark",
    "selected_training_indices": selected.tolist(),
    "selected_training_records": [train["records"][int(index)] for index in selected],
    "selected_training_labels": train["label"][selected].tolist(),
    "runs": {},
}
for name, run in runs.items():
    config = json.loads((run / "experiment.json").read_text())
    initial = torch.load(run / "joint_model_step_00000.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(run / "joint_model_step_04000.pt", map_location="cpu", weights_only=False)
    trace = [json.loads(line) for line in (run / "training_trace.jsonl").read_text().splitlines() if json.loads(line)["step"] <= 4000]
    model = ArbitraryPlaneJointModelV6(runtime, **config["model_kwargs"]).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    head = model.pose_model.proposal_head_v6
    with torch.no_grad():
        encoded = model.pose_model._proposal_features(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1)), tuple(config["retrieval_shape_h_w"]))
        pooled = F.adaptive_avg_pool2d(encoded, head.spatial_bins_h_w).flatten(1)
        pre_context = head.source_context[0](pooled)
        context = head.source_context[1](pre_context)
        activations = {
            "input_image": inputs[:, :1], "encoded": encoded, "pooled": pooled,
            "source_context_linear": pre_context, "source_context_gelu": context,
            "normal_query": head.normal_query(context), "offset_query": head.offset_query(context),
            "roll_query": head.roll_query(context), "mixture_logits": head.mixture_logit(context),
        }
        if head.normal_readout_count is not None:
            activations["direct_normal_logits"] = head.normal_readout(context)
        activation_summary = {}
        for stage, value in activations.items():
            flat = value.double().flatten(1)
            centered = flat - flat.mean(0, keepdim=True)
            activation_summary[stage] = {
                "shape": list(value.shape), "all_finite": bool(torch.isfinite(value).all()),
                "minimum": float(flat.min()), "maximum": float(flat.max()),
                "mean": float(flat.mean()), "standard_deviation": float(flat.std(unbiased=False)),
                "rms": float(flat.square().mean().sqrt()),
                "across_input_centered_rms": float(centered.square().mean().sqrt()),
                "fraction_abs_below_1e_8": float((flat.abs() < 1e-8).double().mean()),
                "fraction_abs_below_1e_6": float((flat.abs() < 1e-6).double().mean()),
                "fraction_negative": float((flat < 0).double().mean()),
                "fraction_rows_all_negative": float((flat < 0).all(1).double().mean()),
                "fraction_channels_negative_for_all_inputs": float((flat < 0).all(0).double().mean()),
                "fraction_rows_all_abs_below_1e_8": float((flat.abs() < 1e-8).all(1).double().mean()),
            }
        x = pre_context.double()
        gelu_derivative = 0.5 * (1 + torch.erf(x / math.sqrt(2))) + x * torch.exp(-0.5 * x.square()) / math.sqrt(2 * math.pi)
        derivative_summary = {
            "fraction_pre_context_below_minus5": float((x < -5).double().mean()),
            "fraction_pre_context_below_minus10": float((x < -10).double().mean()),
            "maximum_absolute_exact_gelu_derivative": float(gelu_derivative.abs().max()),
            "fraction_absolute_exact_gelu_derivative_below_1e_8": float((gelu_derivative.abs() < 1e-8).double().mean()),
        }
    parameter_changes = {}
    for key, value in checkpoint["model_state"].items():
        if key.startswith(("pose_model.histology_stem.", "pose_model.shared_encoder.", "pose_model.proposal_head_v6.")):
            delta = value.double() - initial["model_state"][key].double()
            parameter_changes[key] = {"delta_l2": float(delta.square().sum().sqrt()), "initial_l2": float(initial["model_state"][key].double().square().sum().sqrt()), "final_l2": float(value.double().square().sum().sqrt()), "final_minimum": float(value.min()), "final_maximum": float(value.max())}
    raw = np.load(run / "development_log_probability_step_04000.npy", mmap_mode="r")
    prediction = raw.argmax(1)
    normal_log = np.empty((len(labels), normal_count), dtype=np.float64)
    ranks = np.empty(len(labels), dtype=np.int64)
    all_finite, normalization_error = True, 0.0
    for start in range(0, len(labels), 32):
        block = np.asarray(raw[start:start + 32], dtype=np.float64)
        row_labels = labels[start:start + len(block)]
        target = block[np.arange(len(block)), row_labels]
        all_finite &= bool(np.isfinite(block).all())
        maximum = block.max(1)
        normalization_error = max(normalization_error, float(np.abs(maximum + np.log(np.exp(block - maximum[:, None]).sum(1))).max()))
        normal_log[start:start + len(block)] = np.logaddexp.reduce(block.reshape(len(block), normal_count, cells_per_normal), axis=2)
        ranks[start:start + len(block)] = (block > target[:, None]).sum(1) + ((block == target[:, None]) & (np.arange(raw.shape[1])[None] < row_labels[:, None])).sum(1) + 1
    normal_prediction = normal_log.argmax(1)
    row_metrics = {
        "joint_nll": -raw[np.arange(len(labels)), labels].astype(np.float64),
        "normal_nll": -normal_log[np.arange(len(labels)), labels // cells_per_normal],
        "joint_map_normal_angle_deg": normal_angles[np.arange(len(labels)), prediction // cells_per_normal],
        "normal_marginal_map_angle_deg": normal_angles[np.arange(len(labels)), normal_prediction],
        "normal_mass_within10deg": (np.exp(normal_log) * (normal_angles < 10)).sum(1),
        **{f"hit_at_{k}": (ranks <= k).astype(float) for k in (32, 128)},
    }
    macro = {key: float(np.mean([value[groups == group].mean() for group in np.unique(groups)])) for key, value in row_metrics.items()}
    gradients = np.array([row["gradient_norm"] if row["gradient_norm"] is not None else np.nan for row in trace])
    hashes = {}
    for filename in ("experiment.json", "experiment_source.py", "joint_model_step_00000.pt", "joint_model_step_04000.pt", "development_log_probability_step_04000.npy", "training_trace.jsonl"):
        with (run / filename).open("rb") as stream:
            hashes[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    result["runs"][name] = {
        "step": 4000, "model_kwargs": config["model_kwargs"],
        "activation_summary": activation_summary, "gelu_derivative_summary": derivative_summary,
        "proposal_parameter_changes": parameter_changes,
        "development_synthetic_group_macro": macro,
        "raw_all_finite": all_finite, "raw_maximum_absolute_log_normalization_error": normalization_error,
        "trace_attempted_steps": len(trace), "trace_applied_steps": sum(row["optimizer_step_applied"] for row in trace),
        "optimizer_step_values": sorted({int(value["step"]) for value in checkpoint["optimizer_state"]["state"].values()}),
        "skipped_step_ids": [row["step"] for row in trace if not row["optimizer_step_applied"]],
        "gradient_norm_quantiles": np.nanquantile(gradients, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "gradient_norm_nonfinite_count": int((~np.isfinite(gradients)).sum()),
        "gradient_norm_median_per100": [float(np.nanmedian(gradients[start:start + 100])) for start in range(0, len(trace), 100)],
        "last100_joint_nll": float(np.mean([row["weighted_nll"] for row in trace[-100:]])),
        "last100_normal_nll": float(np.mean([row["weighted_normal_marginal_nll"] for row in trace[-100:]])),
        "artifact_sha256": hashes,
    }
    np.savez(output / f"{name}_fixed_training_activations.npz", selected_training_indices=selected, **{key: value.numpy() for key, value in activations.items()})
    print(json.dumps({"run": name, "development": macro, "source_context_linear": activation_summary["source_context_linear"], "source_context_gelu": activation_summary["source_context_gelu"], "gelu_derivative": derivative_summary}, indent=2), flush=True)
    del model, initial, checkpoint, raw
(output / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(output / "diagnostic.json", flush=True)
