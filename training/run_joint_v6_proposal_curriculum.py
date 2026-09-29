"""Mixed atlas-coverage/frozen-row proposal experiment; edit constants, then run.

Fresh complete joint model by default. Only the proposal is optimized here.
Generated IDs are organizational groups from ONE atlas, not biological animals.
This is a data-volume intervention, not a replacement for individual deformation.
"""

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
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training import arbitrary_plane_catalogue_runtime_v6 as catalogue_runtime
from training import arbitrary_plane_staged_trainer_v6 as trainer
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components,
    render_finite_thickness_plane,
)
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

RUN = ROOT / "runs" / "joint_v6_normal_readout_control_006"
PREPARED = ROOT / "runs" / "joint_v6_proposal_substantive_001"
SEED = 2026092805
STEPS = 4_000
SCHEDULE_STEPS = 20_000  # Preserve the exact 005 schedule prefix when stopping early.
BATCH = 16
GENERATED = 8
EVALUATE_EVERY = 1_000
LEARNING_RATE = 0.001
NORMAL_MARGINAL_NLL_WEIGHT = 1.0
RETRIEVAL_SHAPE = (96, 96)
SUPPORT_MASS_THRESHOLD = 64.0
MODES = ("raw", "exact_black", "imperfect_brush")
MODEL_KWARGS = {
    "atlas_channels": 2, "feature_channels": 64, "hidden_channels": 128,
    "correlation_radius": 2, "proposal_channels": 16,
    "proposal_mixture_components": 8, "proposal_spatial_bins_h_w": (8, 8),
    "cascade_max_rendered_cells_per_sample": 32, "cascade_max_closure_rounds": 4,
    "pose_only_steps": 1, "deformation_integration_steps": 3,
    "proposal_normal_readout_count": 384,
}
PREPARED_SHA256 = {
    "catalogue.pt": "9b49d203cc73ce3a66e648bbe5228231eb5cc9c17d5db4669eefe0f08ae22c71",
    "experiment.json": "90bde7ec17104e340f03aa71913d581a5f9373d50862f0699d4acfba1453be10",
    "training_prepared.pt": "27a1084bdc22c98a10808f8059659199aad1b789745392e10c92f827c919d07d",
    "internal_development_prepared.pt": "89e078582aaa365299d9823deb2a005308f360ab8909db7125a4534556d622c4",
}

for name, expected in PREPARED_SHA256.items():
    with (PREPARED / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
catalogue = torch.load(PREPARED / "catalogue.pt", map_location="cpu", weights_only=False)
train = torch.load(PREPARED / "training_prepared.pt", map_location="cpu", weights_only=False)
dev = torch.load(PREPARED / "internal_development_prepared.pt", map_location="cpu", weights_only=False)
baseline = json.loads((PREPARED / "experiment.json").read_text(encoding="utf-8"))
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not ({row[key] for row in train["records"]} & {row[key] for row in dev["records"]})
RUN.mkdir(parents=True, exist_ok=False)
shutil.copyfile(PREPARED / "catalogue.pt", RUN / "catalogue.pt")
repository = Path(__file__).resolve().parents[1]
source_files = sorted(set(trainer._SOURCE_FILES) | {
    Path(__file__).relative_to(repository).as_posix(),
    "training/arbitrary_plane_allen_atlas_binding_v6.py",
    "training/arbitrary_plane_full_frame_primitives.py",
})
source = {
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    "file_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in source_files},
}
(RUN / "source_diff.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=repository))
(RUN / "experiment_source.py").write_bytes(Path(__file__).read_bytes())
for partition, prepared in (("training", train), ("internal_development", dev)):
    (RUN / f"{partition}_identities.json").write_text(json.dumps(prepared["records"], indent=2), encoding="utf-8")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
started = time.perf_counter()
print("Decoding pinned Allen atlas once; reusing authenticated complete catalogue", flush=True)
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
runtime = catalogue_runtime.make_complete_catalogue_runtime_v6(
    catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"],
    device="cuda", dtype=torch.float32,
)
assert runtime.cell_count == 384 * 16 * 16
normal_count = catalogue["counts"]["normal_count"]
cells_per_normal = catalogue["counts"]["offset_count_per_normal"] * catalogue["counts"]["roll_count"]
cell_states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
render_states = cell_states.to(device="cuda", dtype=torch.float32)
psf_weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device="cuda", dtype=torch.float32) / 16
psf_positions = torch.linspace(-0.5, 0.5, 9, device="cuda")
yy, xx = torch.meshgrid(torch.linspace(-1, 1, 96, device="cuda"), torch.linspace(-1, 1, 96, device="cuda"), indexing="ij")

# Independent schedule/noise domains; counters never reset at a catalogue cycle.
rng = np.random.default_rng(SEED + 1)
count = SCHEDULE_STEPS * GENERATED
cell_schedule = np.concatenate([rng.permutation(runtime.cell_count) for _ in range((count + runtime.cell_count - 1) // runtime.cell_count)])[:count]
frozen_count = SCHEDULE_STEPS * (BATCH - GENERATED)
frozen_schedule = np.concatenate([rng.permutation(len(train["label"])) for _ in range((frozen_count + len(train["label"]) - 1) // len(train["label"]))])[:frozen_count].reshape(SCHEDULE_STEPS, BATCH - GENERATED)
schedule = {
    "cell_index": cell_schedule,
    "sample_seed": SEED * 1_000_000 + np.arange(count, dtype=np.int64),
    "mode_index": rng.integers(0, len(MODES), count),
    "horizontal_flip": rng.integers(0, 2, count).astype(bool),
    "thickness_um": rng.uniform(25, 100, count).astype(np.float32),
    "gain": rng.uniform(0.6, 1.4, count).astype(np.float32),
    "gamma": np.exp(rng.uniform(np.log(0.6), np.log(1.6), count)).astype(np.float32),
    "invert_tissue": rng.integers(0, 2, count).astype(bool),
    "noise_std": rng.uniform(0.005, 0.05, count).astype(np.float32),
    "background_mean": rng.uniform(0, 0.8, count).astype(np.float32),
    "background_slope_yx": rng.uniform(-0.15, 0.15, (count, 2)).astype(np.float32),
    "mask_dilate": rng.integers(0, 2, count).astype(bool),
    "mask_radius_px": rng.integers(1, 4, count),
}
prefix = RUN.name + "-training-atlas-organizational"
for field in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    schedule[field] = np.array([f"{prefix}-{field}-{index:07d}" for index in range(count)])
    assert not set(schedule[field]) & {record[field] for record in dev["records"]}
np.savez_compressed(RUN / "generated_schedule.npz", **schedule)
np.save(RUN / "frozen_training_row_indices.npy", frozen_schedule)
schedule_sha256 = {}
for name in ("generated_schedule.npz", "frozen_training_row_indices.npy"):
    with (RUN / name).open("rb") as stream:
        schedule_sha256[name] = hashlib.file_digest(stream, "sha256").hexdigest()

config = {
    "source": source, "seed": SEED, "steps": STEPS, "schedule_horizon_steps": SCHEDULE_STEPS, "batch_size": BATCH,
    "generated_per_batch": GENERATED, "learning_rate": LEARNING_RATE,
    "optimizer": "AdamW", "weight_decay": 0.0001, "optimizer_state": "fresh",
    "evaluation_interval": EVALUATE_EVERY, "model_kwargs": MODEL_KWARGS,
    "retrieval_shape_h_w": RETRIEVAL_SHAPE,
    "prepared_source_directory": str(PREPARED), "prepared_source_sha256": PREPARED_SHA256,
    "cache_manifests": baseline["cache_manifests"], "atlas_binding": baseline["atlas_binding"],
    "catalogue_receipt_sha256": catalogue["receipt_sha256"],
    "initialization": "fresh_random_complete_joint_model", "resume_checkpoint": None,
    "training_phase": "proposal_only", "external_or_legacy_learned_dependencies": [], "probabilities_calibrated": False,
    "objective": "weighted joint-cell NLL + normal_marginal_nll_weight * weighted normal-marginal NLL",
    "normal_marginal_nll_weight": NORMAL_MARGINAL_NLL_WEIGHT,
    "comparison": "first4000 updates of the exact005 20000-step schedule; same existing initial parameters and RNG; add zero-initialized384-normal final-mixture tilt preserving offset/roll conditionals; same joint+normal objective; organizational ID namespace differs; enabled zero tilt may introduce FP32 normalization roundoff",
    "generated_schedule_sha256": schedule_sha256,
    "generator": {
        "cell_sampling": "shuffled complete 98304-cell permutations; no support rejection; exact cell frames, no subcell jitter",
        "new_rendered_observations_planned": STEPS * GENERATED, "scheduled_observations": count,
        "unique_cells_planned": len(np.unique(cell_schedule[:STEPS * GENERATED])),
        "raster_shape_h_w": [96, 96], "physical_fov_y_x_um": [12000, 12000],
        "pixel_size_y_x_um": [125, 125], "psf_family": "boxcar_trapezoid_9",
        "psf_offsets": "linspace(-0.5,0.5,9) * scheduled thickness_um",
        "psf_weights": [0.0625, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0625],
        "modes": MODES, "mask_source": "rendered atlas annotation support > 0; never image threshold or inferred real mask",
        "point_pose_weight": f"finite slab support mass and visible support mass both >= {SUPPORT_MASS_THRESHOLD} native pixels",
        "input_channels": ["image", "four_neighbor_mask_boundary", "brush_availability"],
        "reflection": "horizontal pixel permutation x -> W-1-x; canonical cell target unchanged; no reflected physical-frame relabel",
        "seed_scope": "schedule PCG64(seed+1); each rendered observation has its own CUDA generator sample_seed for image noise",
        "appearance_order": "finite PSF intensity / finite support, then tissue gamma/inversion/gain, then multiply finite support and mix background, add noise, apply optional brush; post-PSF augmentation, not a calibrated acquisition model",
        "group_semantics": "unique organizational synthetic IDs from ONE Allen template; not independent biological subjects or coherent deformed animals",
        "limitations": "extra canonical pose/appearance coverage only; synthetic backgrounds, no new real acquisitions or 3D subject deformation; complex frozen rows retained 50%",
    },
    "scope": "internal synthetic development; no public benchmark, external or final-test data; same 40 development groups",
    "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
}
(RUN / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
model = ArbitraryPlaneJointModelV6(runtime, **MODEL_KWARGS).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
noise_generator = torch.Generator(device="cuda")
dev_animal = np.array([r["animal_id"] for r in dev["records"]])
dev_modes = np.array([r["selected_mode"] for r in dev["records"]])
truth_center, truth_frame, _ = full_frame_state_to_components(dev["truth_state"])
truth_normal = truth_frame[:, :, 2]
support_origin = torch.as_tensor(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=torch.float64)
training_seconds = 0.0
recent_losses = []
recent_normal_losses = []
applied_steps = 0
generated_supervised = 0
generated_censored = 0
print("Starting mixed complete-catalogue rendering/frozen-row proposal optimization", flush=True)

with (RUN / "training_trace.jsonl").open("w", encoding="utf-8") as trace:
    for step in range(STEPS + 1):
        if step:
            tick = time.perf_counter()
            model.train()
            first = (step - 1) * GENERATED
            positions = np.arange(first, first + GENERATED)
            with torch.no_grad():
                generated_labels = torch.as_tensor(cell_schedule[positions], device="cuda")
                thickness = torch.as_tensor(schedule["thickness_um"][positions], device="cuda")
                rendered = render_finite_thickness_plane(
                    atlas, render_states[generated_labels], RETRIEVAL_SHAPE,
                    allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6, allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
                    thickness[:, None] * psf_positions[None], psf_weights,
                )
                finite_support = rendered[:, 1].clamp(0, 1)
                tissue = finite_support > 0
                generated_inputs = torch.empty((GENERATED, 3, 96, 96), device="cuda")
                visible_mass = torch.empty(GENERATED, device="cuda")
                for local, position in enumerate(positions):
                    # Per-observation noise seed does not affect model/dropout RNG.
                    noise_generator.manual_seed(int(schedule["sample_seed"][position]))
                    noise = torch.randn((96, 96), device="cuda", generator=noise_generator)
                    appearance = (rendered[local, 0] / finite_support[local].clamp_min(1e-6)).clamp(0, 1)
                    appearance = appearance.pow(float(schedule["gamma"][position]))
                    if schedule["invert_tissue"][position]:
                        appearance = 1.0 - appearance
                    appearance = (appearance * float(schedule["gain"][position])).clamp(0, 1)
                    slope = schedule["background_slope_yx"][position]
                    background = float(schedule["background_mean"][position]) + float(slope[0]) * yy + float(slope[1]) * xx
                    image = appearance * finite_support[local] + background * (1.0 - finite_support[local])
                    image = (image + float(schedule["noise_std"][position]) * noise).clamp(0, 1)
                    mode = int(schedule["mode_index"][position])
                    mask = tissue[local].clone()
                    if mode == 2:
                        radius = int(schedule["mask_radius_px"][position])
                        if schedule["mask_dilate"][position]:
                            mask = F.max_pool2d(mask[None, None].float(), 2 * radius + 1, 1, radius)[0, 0] > 0
                        else:
                            padded = F.pad(mask[None, None].float(), (radius,) * 4, value=0)
                            mask = -F.max_pool2d(-padded, 2 * radius + 1, 1)[0, 0] > 0
                    visible_mass[local] = (finite_support[local] * mask).sum() if mode else finite_support[local].sum()
                    outline = torch.zeros_like(mask)
                    if mode:
                        image = image * mask  # Exact zero exterior after noise/appearance.
                        eroded = mask.clone()
                        eroded[1:] &= mask[:-1]
                        eroded[:-1] &= mask[1:]
                        eroded[:, 1:] &= mask[:, :-1]
                        eroded[:, :-1] &= mask[:, 1:]
                        eroded[[0, -1], :] = False
                        eroded[:, [0, -1]] = False
                        outline = mask & ~eroded
                    generated_inputs[local] = torch.stack((image, outline.float(), torch.full_like(image, float(mode != 0))))
                    if schedule["horizontal_flip"][position]:
                        generated_inputs[local] = generated_inputs[local].flip(-1)
                support_mass = finite_support.sum((-2, -1))
                generated_weight = ((support_mass >= SUPPORT_MASS_THRESHOLD) & (visible_mass >= SUPPORT_MASS_THRESHOLD)).float()
                support_values = support_mass.cpu().tolist()
                visible_values = visible_mass.cpu().tolist()
                weight_values = generated_weight.cpu().tolist()
                if step == 1:
                    torch.save({"sample_indices": positions, "input_channels": generated_inputs.cpu(), "state": cell_states[cell_schedule[positions]], "support_mass": support_values, "visible_mass": visible_values}, RUN / "first_generated_batch.pt")
                index = frozen_schedule[step - 1]
                inputs = torch.cat((generated_inputs, train["channels"][index].cuda()))
                label = torch.cat((generated_labels, train["label"][index].cuda()))
                weight = torch.cat((generated_weight, train["weight"][index].cuda()))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model.pose_model.forward_proposal_only(
                    inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1)),
                    runtime.expand(BATCH), RETRIEVAL_SHAPE,
                )
                log_probability = output["raw_full_catalogue_cell_log_probability"]
                losses = F.nll_loss(log_probability, label, reduction="none")
                normal_log_probability = torch.logsumexp(log_probability.reshape(BATCH, normal_count, cells_per_normal), dim=-1)
                normal_losses = F.nll_loss(normal_log_probability, label // cells_per_normal, reduction="none")
                joint_loss = (losses * weight).sum() / weight.sum().clamp_min(1.0)
                normal_loss = (normal_losses * weight).sum() / weight.sum().clamp_min(1.0)
                loss = joint_loss + NORMAL_MARGINAL_NLL_WEIGHT * normal_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied = scaler.get_scale() >= previous_scale
            applied_steps += int(applied)
            generated_supervised += int(sum(weight_values))
            generated_censored += GENERATED - int(sum(weight_values))
            loss_value = float(joint_loss.detach())
            normal_loss_value = float(normal_loss.detach())
            subset_nll = [(losses[part] * weight[part]).sum() / weight[part].sum().clamp_min(1.0) for part in (slice(None, GENERATED), slice(GENERATED, None))]
            training_seconds += time.perf_counter() - tick
            recent_losses.append(loss_value)
            recent_normal_losses.append(normal_loss_value)
            trace.write(json.dumps({
                "step": step, "weighted_nll": loss_value,
                "weighted_normal_marginal_nll": normal_loss_value,
                "weighted_objective": float(loss.detach()),
                "generated_weighted_nll": float(subset_nll[0].detach()), "frozen_weighted_nll": float(subset_nll[1].detach()),
                "gradient_norm": float(gradient) if torch.isfinite(gradient) else None,
                "optimizer_step_applied": applied, "generated_sample_indices": positions.tolist(),
                "finite_slab_support_mass_px": support_values, "visible_support_mass_px": visible_values,
                "generated_point_pose_weight": weight_values,
            }) + "\n")
            if step % 100 == 0:
                trace.flush()
                print(json.dumps({"step": step, "mean_last100_nll": float(np.mean(recent_losses)), "mean_last100_normal_nll": float(np.mean(recent_normal_losses)), "training_seconds": training_seconds, "generated_censored": generated_censored, "applied_steps": applied_steps}), flush=True)
                recent_losses.clear()
                recent_normal_losses.clear()
            del output, inputs, losses, loss, subset_nll, rendered, generated_inputs, log_probability, normal_log_probability, normal_losses, joint_loss, normal_loss

        if step % EVALUATE_EVERY == 0 or step == STEPS:
            model.eval()
            raw = np.lib.format.open_memmap(RUN / f"development_log_probability_step_{step:05d}.npy", mode="w+", dtype=np.float32, shape=(len(dev["label"]), runtime.cell_count))
            with torch.no_grad():
                for start in range(0, len(dev["label"]), BATCH):
                    inputs = dev["channels"][start:start + BATCH].cuda()
                    with torch.autocast("cuda", dtype=torch.float16):
                        output = model.pose_model.forward_proposal_only(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1)), runtime.expand(len(inputs)), RETRIEVAL_SHAPE)
                    raw[start:start + len(inputs)] = output["raw_full_catalogue_cell_log_probability"].float().cpu().numpy()
            raw.flush()
            labels = dev["label"].numpy()
            truth_log = raw[np.arange(len(labels)), labels]
            ranks = (raw > truth_log[:, None]).sum(axis=1) + ((raw == truth_log[:, None]) & (np.arange(runtime.cell_count)[None] < labels[:, None])).sum(axis=1) + 1
            predicted = np.asarray(raw.argmax(axis=1))
            normal_lp = np.logaddexp.reduce(np.asarray(raw).reshape(len(labels), normal_count, cells_per_normal), axis=-1)
            normal_prediction = normal_lp.argmax(axis=-1)
            _, normal_frames, _ = full_frame_state_to_components(cell_states[normal_prediction * cells_per_normal])
            marginal_normal_error = torch.rad2deg(torch.acos((normal_frames[:, :, 2] * truth_normal).sum(-1).abs().clamp(0, 1))).numpy()
            center, frame, _ = full_frame_state_to_components(cell_states[predicted])
            normal = frame[:, :, 2]
            dot = (normal * truth_normal).sum(-1)
            sign = torch.where(dot < 0, -1.0, 1.0)
            offset_error = (((center - support_origin) * normal).sum(-1) - sign * ((truth_center - support_origin) * truth_normal).sum(-1)).abs()
            frame_cosine = ((frame * truth_frame).sum((-2, -1)) - 1.0) / 2.0
            antipodal_cosine = ((frame * torch.tensor([-1.0, 1.0, -1.0]) * truth_frame).sum((-2, -1)) - 1.0) / 2.0
            metrics = {
                "nll": -np.asarray(truth_log), "truth_rank": ranks,
                "normal_marginal_nll": -normal_lp[np.arange(len(labels)), labels // cells_per_normal],
                "normal_marginal_map_angle_deg": marginal_normal_error,
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
                "schedule_files": schedule_sha256, "generated_observations": step * GENERATED,
                "generated_supervised": generated_supervised, "generated_censored": generated_censored,
                "optimizer_steps_applied": applied_steps, "development_metrics": macro,
                "development_by_support": support_macro, "probabilities_calibrated": False,
            }
            torch.save(checkpoint, RUN / f"joint_model_step_{step:05d}.pt.tmp")
            os.replace(RUN / f"joint_model_step_{step:05d}.pt.tmp", RUN / f"joint_model_step_{step:05d}.pt")
            print(json.dumps({"development_step": step, "animal_macro": macro, "identifiable_animal_macro": support_macro["identifiable"]["animal_macro"], "elapsed_seconds": evaluation["elapsed_seconds"]}), flush=True)
            del checkpoint, raw, output, inputs

(RUN / "completed.json").write_text(json.dumps({"steps": STEPS, "optimizer_steps_applied": applied_steps, "new_rendered_observations": STEPS * GENERATED, "unique_catalogue_cells_sampled": len(np.unique(cell_schedule[:STEPS * GENERATED])), "generated_supervised": generated_supervised, "generated_censored": generated_censored, "elapsed_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
print(f"Finished {STEPS} mixed proposal updates: {RUN}", flush=True)
