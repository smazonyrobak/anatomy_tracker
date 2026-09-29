"""Conditional truth-near joint learning; edit constants, then run on I:.

This is NOT global-capture qualification. Synthetic truth sets local starts and
known PSF schedules; reflection truth is a loss target, never an input prior.
"""

import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True

import hashlib
import json
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_physical_ouv
from training.arbitrary_plane_joint_loss import physical_frame_landmarks
from training.arbitrary_plane_joint_loss_v6 import _dense_mean, _weighted_mean
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6
from training.arbitrary_plane_joint_uncertainty import full_frame_residual
from training.arbitrary_plane_recurrent_model import compose_antipodal_plane_frame_residual

RUN = ROOT / "runs/joint_v6_local_refinement_001"
PARENT = ROOT / "runs/joint_v6_proposal_curriculum_003/joint_model_step_20000.pt"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001"
PACK = ROOT / "data/joint_v6_local_refinement_frozen_001"
SEED = 2026092906
STEPS, BATCH, EVALUATE_EVERY = 4000, 4, 250
REFINEMENT_STEPS, POSE_ONLY_STEPS = 3, 1
LEARNING_RATE, AMP = 2e-4, False
SMALL_NOISE = (.06, .06, 250., .08, 300., 300., .05, .05, .05)
LARGE_NOISE = (.15, .15, 600., .20, 600., 600., .12, .12, .12)
TRAINABLE_PREFIXES = (
    "pose_model.atlas_stem.", "pose_model.refinement_pair_encoder.",
    "pose_model.recurrent_cell.", "pose_model.recurrent_update.",
    "pose_model.recurrent_log_likelihood.", "deformation_decoder.",
)

pack_manifest = json.loads((PACK / "pack.json").read_text())
for partition, receipt in pack_manifest["partitions"].items():
    with (PACK / f"{partition}.pt").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == receipt["sha256"]
train = torch.load(PACK / "training.pt", map_location="cpu", weights_only=False, mmap=True)
dev = torch.load(PACK / "internal_development.pt", map_location="cpu", weights_only=False, mmap=True)
assert len(train["records"]) == 2048 and len(dev["records"]) == 256
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not ({row[key] for row in train["records"]} & {row[key] for row in dev["records"]})
with PARENT.open("rb") as stream:
    parent_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
parent = torch.load(PARENT, map_location="cpu", weights_only=False, mmap=True)
catalogue = torch.load(PREPARED / "catalogue.pt", map_location="cpu", weights_only=False)
RUN.mkdir(parents=True, exist_ok=False)
repository = Path(__file__).resolve().parents[1]
source_names = sorted(set(parent["experiment"]["source"]["file_sha256"]) | {
    "training/run_joint_v6_local_refinement.py", "training/arbitrary_plane_joint_uncertainty.py",
    "training/arbitrary_plane_joint_loss.py", "training/arbitrary_plane_joint_loss_v6.py",
})
source = {
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    "file_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in source_names},
}
(RUN / "source_diff.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=repository))
(RUN / "experiment_source.py").write_bytes(Path(__file__).read_bytes())
for name, prepared in (("training", train), ("internal_development", dev)):
    (RUN / f"{name}_identities.json").write_text(json.dumps(prepared["records"], indent=2))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
schedule_rng = torch.Generator().manual_seed(SEED + 1)
order = torch.cat([torch.randperm(len(train["records"]), generator=schedule_rng)
                   for _ in range((STEPS * BATCH + len(train["records"]) - 1) // len(train["records"]))])[:STEPS * BATCH].reshape(STEPS, BATCH)
noise = 2 * torch.rand(STEPS, BATCH, 9, generator=schedule_rng) - 1
noise[:500] *= torch.tensor(SMALL_NOISE)
noise[500:] *= torch.tensor(LARGE_NOISE)
dev_noise = (2 * torch.rand(len(dev["records"]), 9, generator=schedule_rng) - 1) * torch.tensor(LARGE_NOISE)
dev_subset = torch.randperm(len(dev["records"]), generator=schedule_rng)[:64].sort().values
torch.save({"training_row_index": order, "training_perturbation": noise,
            "development_perturbation": dev_noise, "development_subset": dev_subset}, RUN / "schedule.pt")
with (RUN / "schedule.pt").open("rb") as stream:
    schedule_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

print("Decoding pinned Allen atlas for conditional local refinement", flush=True)
started = time.perf_counter()
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cuda", dtype=torch.float32)
model_kwargs = {**parent["experiment"]["model_kwargs"], "deformation_integration_steps": 7, "joint_uncertainty_rank": None}
model = ArbitraryPlaneJointModelV6(runtime, **model_kwargs).cuda()
model.load_state_dict(parent["model_state"], strict=True)
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
frozen_reference = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=AMP)
geometry = catalogue["support_geometry"]
origin, spacing = geometry["origin_ap_dv_ml_um"], geometry["voxel_size_ap_dv_ml_um"]
support_origin = geometry["support_origin_ap_dv_ml_um"]
affines = torch.as_tensor(catalogue["arrays"]["representation_to_canonical_raster_affine_float64"][0], device="cuda", dtype=torch.float32)
assert affines.shape == (2, 2, 3)
config = {
    "source": source, "seed": SEED, "steps": STEPS, "batch_size": BATCH,
    "learning_rate": LEARNING_RATE, "amp": AMP, "evaluation_interval": EVALUATE_EVERY,
    "refinement_steps": REFINEMENT_STEPS, "pose_only_steps": POSE_ONLY_STEPS,
    "parent_checkpoint": str(PARENT), "parent_sha256": parent_sha256,
    "parent_step": parent["step"], "model_kwargs": model_kwargs,
    "pack_manifest": pack_manifest, "schedule_sha256": schedule_sha256,
    "catalogue_receipt_sha256": catalogue["receipt_sha256"], "atlas_binding": parent["experiment"]["atlas_binding"],
    "trainable_prefixes": TRAINABLE_PREFIXES,
    "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
    "small_noise_first_500_steps": SMALL_NOISE, "large_noise_after_500_steps": LARGE_NOISE,
    "noise_coordinates": "normal tangent u/v radians, normal offset um, roll radians, in-plane u/v um, log basis u/v, shear",
    "scope": "conditional truth-near/oracle-initialized local development with known synthetic PSF; not global capture or benchmark",
    "development_deformation_feedback": "all rows enabled; truth censor flags affect metric eligibility only",
    "reflection_prior": "uniform over bound identity/horizontal states; labels used only for CE loss",
    "initial_metric_baseline": "perturbed frame, identity pullback, identity reflection (fixed argmax tie of uniform prior)",
    "covariance": "disabled; no calibrated uncertainty claim",
    "grouping": "animal-disjoint organizational synthetic groups from one Allen atlas, not biological subject generalization",
    "initialization": "own randomly initialized lineage; full parent checkpoint, fresh optimizer, no external weights",
}
(RUN / "experiment.json").write_text(json.dumps(config, indent=2))
del parent


def local_pass(prepared, index, perturbation, training):
    """One local forward and geometric objective; no fabricated coarse posterior."""
    image, outline = prepared["image"][index].cuda(), prepared["outline"][index].cuda()
    available = prepared["outline_available"][index].cuda()
    truth = prepared["truth_state"][index].cuda().float()
    pose_weight = prepared["pose_supervision_weight"][index].cuda().float()
    dense_weight = pose_weight * prepared["dense_deformation_supervision_weight"][index].cuda().float()
    pixel_weight = prepared["deformation_weight"][index].cuda().float()
    velocity_truth = prepared["truth_stationary_velocity_yx_px"][index].cuda().float()
    map_truth = prepared["truth_pullback_map_yx_px"][index].cuda().float()
    reflection = prepared["reflection_representation_index"][index].cuda()
    batch, _, height, width = image.shape
    with torch.autocast("cuda", enabled=False):
        initial = compose_antipodal_plane_frame_residual(truth, perturbation.cuda().float(), support_origin)[:, None]
    source_features = model.pose_model.encode_histology(image, outline, available)
    refinement = model.pose_model.refine(
        source_features, atlas, initial, torch.full((batch, 1, 2), -np.log(2), device="cuda"),
        affines[None, None].expand(batch, 1, -1, -1, -1), (height, width), origin, spacing, support_origin,
        prepared["axial_offsets_um"][index].cuda().float(), prepared["axial_weights"][index].cuda().float(),
        REFINEMENT_STEPS, deformation_decoder=model.deformation_decoder, pose_only_steps=POSE_ONLY_STEPS,
        dense_deformation_supervision_weight=dense_weight if training else None,
    )
    output = model._legacy_joint_output(refinement, POSE_ONLY_STEPS)
    pose = output["pose"]
    with torch.autocast("cuda", enabled=False):
        state_sequence = pose["refined_cell_state_sequence"][:, 0, 1:].float()
        residual = full_frame_residual(state_sequence, truth[:, None].expand_as(state_sequence))
        scale = residual.new_tensor((.1, .1, .1, 500., 500., 500., .1, .1, .1))
        pose_sequence = .8 ** torch.arange(REFINEMENT_STEPS - 1, -1, -1, device="cuda")
        pose_error = F.smooth_l1_loss(residual / scale, torch.zeros_like(residual), reduction="none").mean(-1)
        pose_loss = _weighted_mean((pose_error * pose_sequence).sum(-1) / pose_sequence.sum(), pose_weight)
        truth_landmarks = physical_frame_landmarks(truth)
        final_state = pose["final_cell_state"][:, 0].float()
        landmark_error = (physical_frame_landmarks(final_state) - truth_landmarks).norm(dim=-1).mean(-1)
        initial_error = (physical_frame_landmarks(initial[:, 0]) - truth_landmarks).norm(dim=-1).mean(-1)
        landmark_loss = _weighted_mean(F.smooth_l1_loss(landmark_error / 250., torch.zeros_like(landmark_error), reduction="none"), pose_weight)
        reflection_log_probability = pose["final_representation_log_conditional_within_cell"][:, 0].float()
        reflection_loss = _weighted_mean(F.nll_loss(reflection_log_probability, reflection, reduction="none"), pose_weight)
        velocity = output["stationary_velocity_yx_px_sequence"][:, 0].float()
        pullback = output["pullback_map_yx_px_sequence"][:, 0].float()
        support_logits = output["support_logits_sequence"][:, 0].float()
        jacobian = output["forward_jacobian_determinant_sequence"][:, 0].float()
        active = output["deformation_active_sequence"].float()
        sequence = (active * .8 ** torch.arange(len(active) - 1, -1, -1, device="cuda"))[None, :, None, None, None]
        weight = pixel_weight[:, None] * sequence
        vector_weight = weight.expand_as(velocity)
        svf_loss = _dense_mean(F.smooth_l1_loss(velocity, velocity_truth[:, None].expand_as(velocity), beta=.5, reduction="none"), vector_weight, dense_weight)
        map_loss = _dense_mean(F.smooth_l1_loss(pullback, map_truth[:, None].expand_as(pullback), beta=.5, reduction="none"), vector_weight, dense_weight)
        support_loss = _dense_mean(F.binary_cross_entropy_with_logits(support_logits, pixel_weight[:, None].expand_as(support_logits), reduction="none"), sequence, dense_weight)
        topology_loss = _dense_mean(F.relu(.05 - jacobian).square(), weight, dense_weight)
        smoothness_loss = .5 * (
            _dense_mean((velocity[..., 1:, :] - velocity[..., :-1, :]).square(), vector_weight[..., 1:, :] * vector_weight[..., :-1, :], dense_weight)
            + _dense_mean((velocity[..., :, 1:] - velocity[..., :, :-1]).square(), vector_weight[..., :, 1:] * vector_weight[..., :, :-1], dense_weight)
        )
        cycle_loss = sum(_dense_mean(output[f"{direction}_error_yx_sequence"][:, 0].float().square(),
                                    vector_weight * output[f"{direction}_valid_mask_sequence"][:, 0], dense_weight)
                         for direction in ("forward_then_inverse", "inverse_then_forward")) / 2
        total = pose_loss + landmark_loss + .25 * reflection_loss + .5 * svf_loss + .25 * map_loss + .05 * support_loss + .1 * topology_loss + .05 * smoothness_loss + .05 * cycle_loss
    losses = dict(zip(
        ("pose9", "landmark", "reflection", "svf", "map", "support", "topology", "smoothness", "cycle"),
        torch.stack((pose_loss, landmark_loss, reflection_loss, svf_loss, map_loss, support_loss,
                     topology_loss, smoothness_loss, cycle_loss)).detach().cpu().tolist(),
    ))
    if training:
        return total, losses, None, None
    with torch.no_grad():
        mass = pixel_weight.sum((1, 2, 3))
        dense_eligible = (dense_weight > 0) & (mass > 0)
        final_velocity, final_map = velocity[:, -1], pullback[:, -1]
        svf_error = (((final_velocity - velocity_truth).square().sum(1, keepdim=True) * pixel_weight).sum((1, 2, 3)) / mass.clamp_min(1)).sqrt()
        map_error = (((final_map - map_truth).norm(dim=1, keepdim=True) * pixel_weight).sum((1, 2, 3)) / mass.clamp_min(1))
        representation = reflection_log_probability.argmax(-1)
        represented_yx = final_map.clone()
        represented_yx[:, 1] = torch.where(representation[:, None, None] == 1, width - 1 - final_map[:, 1], final_map[:, 1])
        ouv = full_frame_state_to_physical_ouv(final_state)
        ccf = ouv[:, :3, None, None] + ouv[:, 3:6, None, None] * (represented_yx[:, 1:2] / width) + ouv[:, 6:9, None, None] * (represented_yx[:, :1] / height)
        ccf_truth = prepared["target_ccf_coordinates_ap_dv_ml_um_float64"][index].cuda().float()
        ccf_error = (torch.where(pixel_weight > 0, (ccf - ccf_truth).norm(dim=1, keepdim=True), 0) * pixel_weight).sum((1, 2, 3)) / mass.clamp_min(1)
        identity_yx = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda"), indexing="ij")).float()[None]
        initial_map_error = ((identity_yx - map_truth).norm(dim=1, keepdim=True) * pixel_weight).sum((1, 2, 3)) / mass.clamp_min(1)
        initial_ouv = full_frame_state_to_physical_ouv(initial[:, 0])
        initial_ccf = initial_ouv[:, :3, None, None] + initial_ouv[:, 3:6, None, None] * (identity_yx[:, 1:2] / width) + initial_ouv[:, 6:9, None, None] * (identity_yx[:, :1] / height)
        initial_ccf_error = (torch.where(pixel_weight > 0, (initial_ccf - ccf_truth).norm(dim=1, keepdim=True), 0) * pixel_weight).sum((1, 2, 3)) / mass.clamp_min(1)
        truth_ouv = full_frame_state_to_physical_ouv(truth)
        truth_normal = F.normalize(torch.linalg.cross(truth_ouv[:, 3:6], truth_ouv[:, 6:9]), dim=-1)
        initial_normal = F.normalize(torch.linalg.cross(initial_ouv[:, 3:6], initial_ouv[:, 6:9]), dim=-1)
        final_normal = F.normalize(torch.linalg.cross(ouv[:, 3:6], ouv[:, 6:9]), dim=-1)
        normal_angles = [torch.atan2(torch.linalg.cross(normal, truth_normal).norm(dim=-1), (normal * truth_normal).sum(-1).abs()) * (180 / np.pi) for normal in (initial_normal, final_normal)]
        metrics = {
            "initial_landmark_error_um": initial_error.masked_fill(pose_weight <= 0, torch.nan),
            "final_landmark_error_um": landmark_error.masked_fill(pose_weight <= 0, torch.nan),
            "initial_plane_normal_error_deg": normal_angles[0].masked_fill(pose_weight <= 0, torch.nan),
            "final_plane_normal_error_deg": normal_angles[1].masked_fill(pose_weight <= 0, torch.nan),
            "frame_rotation_error_deg": residual[:, -1, :3].norm(dim=-1).mul(180 / np.pi).masked_fill(pose_weight <= 0, torch.nan),
            "reflection_correct": (representation == reflection).float().masked_fill(pose_weight <= 0, torch.nan),
            "svf_vector_rmse_px": svf_error.masked_fill(~dense_eligible, torch.nan),
            "initial_pullback_endpoint_error_px": initial_map_error.masked_fill(~dense_eligible, torch.nan),
            "pullback_endpoint_error_px": map_error.masked_fill(~dense_eligible, torch.nan),
            "initial_joint_ccf_correspondence_error_um": initial_ccf_error.masked_fill(~dense_eligible, torch.nan),
            "joint_ccf_correspondence_error_um": ccf_error.masked_fill(~dense_eligible, torch.nan),
            "minimum_jacobian": jacobian[:, -1].flatten(1).min(-1).values,
        }
        raw = {"initial_state": initial[:, 0].detach().cpu(), "final_state": final_state.detach().cpu(),
               "reflection_log_probability": reflection_log_probability.detach().cpu(),
               "stationary_velocity_yx_px": final_velocity.detach().cpu(), "pullback_map_yx_px": final_map.detach().cpu(),
               "pose_supervision_weight": pose_weight.cpu(), "dense_supervision_weight": dense_weight.cpu()}
    return total, losses, {key: value.cpu() for key, value in metrics.items()}, raw


applied = 0
with (RUN / "training_trace.jsonl").open("w") as trace:
    for step in range(STEPS + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=AMP):
                objective, losses, _, _ = local_pass(train, order[step - 1], noise[step - 1], True)
            assert bool(torch.isfinite(objective))
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied += int(scaler.get_scale() >= previous_scale)
            record = {"step": step, "objective": float(objective.detach()), "losses": losses,
                      "preclip_gradient_norm": float(gradient), "row_index": order[step - 1].tolist(),
                      "optimizer_steps_applied": applied, "elapsed_seconds": time.perf_counter() - started}
            trace.write(json.dumps(record) + "\n")
            trace.flush()
            if step % 100 == 0:
                print(json.dumps(record), flush=True)
            del objective
        if step % EVALUATE_EVERY == 0:
            model.eval()
            selected = torch.arange(len(dev["records"])) if step in (0, STEPS) else dev_subset
            metric_batches, raw_batches = [], []
            with torch.inference_mode(), torch.autocast("cuda", enabled=AMP):
                for index in selected.split(BATCH):
                    _, _, metrics, raw = local_pass(dev, index, dev_noise[index], False)
                    metric_batches.append(metrics)
                    raw_batches.append(raw)
            metrics = {key: torch.cat([batch[key] for batch in metric_batches]) for key in metric_batches[0]}
            raw = {key: torch.cat([batch[key] for batch in raw_batches]) for key in raw_batches[0]}
            animal = np.array([dev["records"][row]["animal_id"] for row in selected.tolist()])
            by_animal = {name: {key: float(values[torch.from_numpy(animal == name)].nanmean()) for key, values in metrics.items()} for name in np.unique(animal)}
            by_animal = {name: {key: value if np.isfinite(value) else None for key, value in record.items()} for name, record in by_animal.items()}
            macro = {key: float(np.mean([record[key] for record in by_animal.values() if record[key] is not None])) for key in metrics}
            evaluation = {"step": step, "scope": config["scope"], "row_count": len(selected),
                          "animal_count": len(by_animal), "animal_macro": macro, "by_animal": by_animal,
                          "eligible_row_count": {key: int(torch.isfinite(value).sum()) for key, value in metrics.items()},
                          "probabilities_calibrated": False, "elapsed_seconds": time.perf_counter() - started}
            (RUN / f"local_development_metrics_step_{step:05d}.json").write_text(json.dumps(evaluation, indent=2))
            torch.save({"pack_row_index": selected, "prepared_row_index": dev["prepared_row_index"][selected],
                        "perturbation": dev_noise[selected], "metrics": metrics, **raw}, RUN / f"local_development_raw_step_{step:05d}.pt")
            checkpoint = {"experiment": config, "phase": "conditional_local_joint", "step": step,
                          "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                          "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                          "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                          "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                          "optimizer_steps_applied": applied, "local_development_metrics": macro,
                          "schedule_sha256": schedule_sha256, "probabilities_calibrated": False}
            torch.save(checkpoint, RUN / f"joint_model_step_{step:05d}.pt.tmp")
            os.replace(RUN / f"joint_model_step_{step:05d}.pt.tmp", RUN / f"joint_model_step_{step:05d}.pt")
            print(json.dumps({"local_development_step": step, "animal_macro": macro}), flush=True)
            del checkpoint, metric_batches, raw_batches, metrics, raw

frozen_unchanged = all(torch.equal(parameter.detach().cpu(), frozen_reference[name])
                       for name, parameter in model.named_parameters() if name in frozen_reference)
assert frozen_unchanged
(RUN / "completed.json").write_text(json.dumps({"steps": STEPS, "optimizer_steps_applied": applied,
    "frozen_parameters_equal_parent": frozen_unchanged, "scope": config["scope"],
    "probabilities_calibrated": False, "elapsed_seconds": time.perf_counter() - started}, indent=2))
print(f"Completed conditional local joint learning, not global qualification: {RUN}", flush=True)
