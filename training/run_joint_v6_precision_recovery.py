"""Continue the exact capacity run schedule from step 3000 with a stable FP32 head."""

import os
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = str(ROOT / "tmp")
os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache" / "torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache" / "cuda")

import hashlib
import json
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_catalogue_runtime_v6 as catalogue_runtime
from training import arbitrary_plane_staged_trainer_v6 as trainer
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

RUN = ROOT / "runs" / "joint_v6_proposal_precision_recovery_004"
PARENT = ROOT / "runs" / "joint_v6_proposal_capacity_002"
PARENT_CHECKPOINT = PARENT / "joint_model_step_03000.pt"
PARENT_SHA256 = "2447989285e9ff386b37a43fd5dc7dd7ce8ca13fba925a3d0c933a9289cc2322"
START_STEP = 3000
PREPARED_SOURCE = ROOT / "runs" / "joint_v6_proposal_substantive_001"
SEED = 2026092801
STEPS = 10_000
BATCH = 16
EVALUATE_EVERY = 1000
LEARNING_RATE = 0.001
RETRIEVAL_SHAPE = (96, 96)
MANIFESTS = {
    "training": "af778499dd77ac13f673ce0778972218519f421b97fa01c4d055e00d1dd16266",
    "internal_development": "22d6ae273f13974250d74a2f5b32446a49a1c2c3051d40ad45706300aa1159de",
}
MODEL_KWARGS = {
    "atlas_channels": 2,
    "feature_channels": 64,
    "hidden_channels": 128,
    "correlation_radius": 2,
    "proposal_channels": 16,
    "proposal_mixture_components": 8,
    "proposal_spatial_bins_h_w": (8, 8),
    "cascade_max_rendered_cells_per_sample": 32,
    "cascade_max_closure_rounds": 4,
    "pose_only_steps": 1,
    "deformation_integration_steps": 3,
}

with PARENT_CHECKPOINT.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == PARENT_SHA256
parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
assert parent["step"] == START_STEP
RUN.mkdir(parents=True, exist_ok=False)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
started = time.perf_counter()
repository = Path(__file__).resolve().parents[1]
source_files = sorted(set(trainer._SOURCE_FILES) | {
    Path(__file__).relative_to(repository).as_posix(),
    "training/arbitrary_plane_allen_atlas_binding_v6.py",
    "training/arbitrary_plane_finite_row_binding_v6.py",
    "training/arbitrary_plane_training_data_v6.py",
})
source = {
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    "file_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in source_files},
}
(RUN / "source_diff.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=repository))
print("Loading completed experiment 001 data only; no model weights", flush=True)
prepared_files = ("experiment.json", "catalogue.pt", "training_prepared.pt", "internal_development_prepared.pt")
prepared_sha256 = {
    name: hashlib.file_digest((PREPARED_SOURCE / name).open("rb"), "sha256").hexdigest()
    for name in prepared_files
}
baseline_config = json.loads((PREPARED_SOURCE / "experiment.json").read_text())
assert baseline_config["cache_manifests"] == MANIFESTS
catalogue = torch.load(PREPARED_SOURCE / "catalogue.pt", map_location="cpu", weights_only=False)
runtime = catalogue_runtime.make_complete_catalogue_runtime_v6(
    catalogue, expected_catalogue_receipt_sha256=baseline_config["catalogue_receipt_sha256"],
    device="cuda", dtype=torch.float32,
)
torch.save(catalogue, RUN / "catalogue.pt")
partitions = {}
for partition in MANIFESTS:
    partitions[partition] = torch.load(
        PREPARED_SOURCE / f"{partition}_prepared.pt", map_location="cpu", weights_only=False,
    )
    torch.save(partitions[partition], RUN / f"{partition}_prepared.pt")
    (RUN / f"{partition}_identities.json").write_text(
        json.dumps(partitions[partition]["records"], indent=2), encoding="utf-8",
    )

for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not ({r[key] for r in partitions["training"]["records"]} & {r[key] for r in partitions["internal_development"]["records"]})

config = {
    "source": source, "seed": SEED, "steps": STEPS, "batch_size": BATCH,
    "learning_rate": LEARNING_RATE, "optimizer": "AdamW", "weight_decay": 0.0001,
    "evaluation_interval": EVALUATE_EVERY, "model_kwargs": MODEL_KWARGS,
    "retrieval_shape_h_w": RETRIEVAL_SHAPE, "cache_manifests": MANIFESTS,
    "atlas_binding": baseline_config["atlas_binding"], "catalogue_receipt_sha256": catalogue["receipt_sha256"],
    "initialization": "continuation_of_own_fresh_capacity_002_lineage",
    "parent_checkpoint": str(PARENT_CHECKPOINT), "parent_sha256": PARENT_SHA256,
    "parent_experiment": parent["experiment"], "start_step": START_STEP, "training_phase": "proposal_only",
    "learned_dependencies": [str(PARENT_CHECKPOINT)],
    "external_or_legacy_learned_dependencies": [], "probabilities_calibrated": False,
    "prepared_data_source": str(PREPARED_SOURCE), "prepared_data_sha256": prepared_sha256,
    "comparison": "whole probability head outside AMP plus centered geometry; exact parent optimizer/scaler/RNG/row schedule continued",
    "scope": "internal synthetic development; no public benchmark, external or final-test data",
    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
}
(RUN / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
model = ArbitraryPlaneJointModelV6(runtime, **MODEL_KWARGS).cuda()
model.load_state_dict(parent["model_state"], strict=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
optimizer.load_state_dict(parent["optimizer_state"])
scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
scaler.load_state_dict(parent["scaler_state"])
torch.set_rng_state(parent["torch_rng"])
torch.cuda.set_rng_state_all(parent["cuda_rng"])
np.random.set_state(parent["numpy_rng"])
random.setstate(parent["python_rng"])
del parent
train = partitions["training"]
dev = partitions["internal_development"]
sampled = np.load(PARENT / "training_row_indices.npy")
np.save(RUN / "training_row_indices.npy", sampled)
dev_animal = np.array([r["animal_id"] for r in dev["records"]])
dev_modes = np.array([r["selected_mode"] for r in dev["records"]])
truth_center, truth_frame, _ = full_frame_state_to_components(dev["truth_state"])
truth_normal = truth_frame[:, :, 2]
cell_states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
support_origin = torch.as_tensor(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=torch.float64)
training_seconds = 0.0
recent_losses = []
print(f"Continuing proposal updates {START_STEP + 1}..{STEPS} with the stable FP32 head", flush=True)

with (RUN / "training_trace.jsonl").open("w", encoding="utf-8") as trace:
    for step in range(START_STEP, STEPS + 1):
        if step > START_STEP:
            tick = time.perf_counter()
            model.train()
            index = sampled[step - 1]
            inputs = train["channels"][index].cuda()
            label = train["label"][index].cuda()
            weight = train["weight"][index].cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model.pose_model.forward_proposal_only(
                    inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1)),
                    runtime.expand(len(index)), RETRIEVAL_SHAPE,
                )
                losses = F.nll_loss(output["raw_full_catalogue_cell_log_probability"], label, reduction="none")
                loss = (losses * weight).sum() / weight.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied = scaler.get_scale() >= previous_scale
            loss_value = float(loss.detach())
            training_seconds += time.perf_counter() - tick
            recent_losses.append(loss_value)
            trace.write(json.dumps({"step": step, "weighted_nll": loss_value, "gradient_norm": float(gradient) if torch.isfinite(gradient) else None, "optimizer_step_applied": applied}) + "\n")
            if step % 100 == 0:
                trace.flush()
                print(json.dumps({"step": step, "mean_last100_nll": float(np.mean(recent_losses)), "training_seconds": training_seconds, "steps_per_training_second": (step - START_STEP) / training_seconds}), flush=True)
                recent_losses.clear()
            del output, inputs, losses, loss

        if step % EVALUATE_EVERY == 0 or step == STEPS:
            model.eval()
            raw = np.lib.format.open_memmap(RUN / f"development_log_probability_step_{step:05d}.npy", mode="w+", dtype=np.float32, shape=(len(dev["label"]), runtime.cell_count))
            with torch.no_grad():
                for start in range(0, len(dev["label"]), BATCH):
                    inputs = dev["channels"][start:start + BATCH].cuda()
                    with torch.autocast("cuda", dtype=torch.float16):
                        output = model.pose_model.forward_proposal_only(
                            inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1)),
                            runtime.expand(len(inputs)), RETRIEVAL_SHAPE,
                        )
                    raw[start:start + len(inputs)] = output["raw_full_catalogue_cell_log_probability"].float().cpu().numpy()
            raw.flush()
            labels = dev["label"].numpy()
            truth_log = raw[np.arange(len(labels)), labels]
            ranks = (raw > truth_log[:, None]).sum(axis=1) + ((raw == truth_log[:, None]) & (np.arange(runtime.cell_count)[None] < labels[:, None])).sum(axis=1) + 1
            predicted = np.asarray(raw.argmax(axis=1))
            center, frame, _ = full_frame_state_to_components(cell_states[predicted])
            normal = frame[:, :, 2]
            dot = (normal * truth_normal).sum(-1)
            sign = torch.where(dot < 0, -1.0, 1.0)
            offset_error = (((center - support_origin) * normal).sum(-1) - sign * ((truth_center - support_origin) * truth_normal).sum(-1)).abs()
            frame_cosine = ((frame * truth_frame).sum((-2, -1)) - 1.0) / 2.0
            antipodal_cosine = ((frame * torch.tensor([-1.0, 1.0, -1.0]) * truth_frame).sum((-2, -1)) - 1.0) / 2.0
            metrics = {
                "nll": -np.asarray(truth_log), "truth_rank": ranks,
                "plane_angle_deg": torch.rad2deg(torch.acos(dot.abs().clamp(0, 1))).numpy(),
                "antipodal_frame_angle_deg": torch.rad2deg(torch.acos(torch.maximum(frame_cosine, antipodal_cosine).clamp(-1, 1))).numpy(),
                "representation_sensitive_frame_angle_deg": torch.rad2deg(torch.acos(frame_cosine.clamp(-1, 1))).numpy(),
                "normal_offset_error_um": offset_error.numpy(),
                **{f"hit_at_{k}": (ranks <= k).astype(float) for k in (1, 8, 32, 128)},
            }
            animal_metrics = {animal: {name: float(values[dev_animal == animal].mean()) for name, values in metrics.items()} for animal in np.unique(dev_animal)}
            macro = {name: float(np.mean([record[name] for record in animal_metrics.values()])) for name in metrics}
            mode_macro = {mode: {name: float(np.mean([values[(dev_animal == animal) & (dev_modes == mode)].mean() for animal in np.unique(dev_animal[dev_modes == mode])])) for name, values in metrics.items()} for mode in np.unique(dev_modes)}
            support_subsets = {"identifiable": dev["weight"].numpy() > 0, "censored_low_support": dev["weight"].numpy() == 0}
            support_macro = {name: {
                "row_count": int(selected.sum()), "animal_count": len(np.unique(dev_animal[selected])),
                "animal_macro": {metric: float(np.mean([values[selected & (dev_animal == animal)].mean() for animal in np.unique(dev_animal[selected])])) for metric, values in metrics.items()} if selected.any() else None,
            } for name, selected in support_subsets.items()}
            evaluation = {"step": step, "animal_count": len(animal_metrics), "animal_macro": macro, "by_input_mode_animal_macro": mode_macro, "by_support": support_macro, "by_animal": animal_metrics, "probabilities_calibrated": False, "geometry_metric_scope": "coarse plane and antipodal frame proxies; not anatomical landmark registration", "elapsed_seconds": time.perf_counter() - started}
            (RUN / f"development_metrics_step_{step:05d}.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
            np.savez(RUN / f"development_rows_step_{step:05d}.npz", label=labels, prediction=predicted, pose_supervision_weight=dev["weight"].numpy(), **metrics)
            checkpoint = {
                "experiment": config, "phase": "proposal_only", "step": step,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "training_row_schedule": "training_row_indices.npy",
                "development_metrics": macro, "development_by_support": support_macro, "probabilities_calibrated": False,
            }
            torch.save(checkpoint, RUN / f"joint_model_step_{step:05d}.pt.tmp")
            os.replace(RUN / f"joint_model_step_{step:05d}.pt.tmp", RUN / f"joint_model_step_{step:05d}.pt")
            print(json.dumps({"development_step": step, "animal_macro": macro, "identifiable_animal_macro": support_macro["identifiable"]["animal_macro"], "elapsed_seconds": evaluation["elapsed_seconds"]}), flush=True)
            del checkpoint, raw, output, inputs

print(f"Finished {STEPS - START_STEP} continuation updates, cumulative step {STEPS}: {RUN}", flush=True)
