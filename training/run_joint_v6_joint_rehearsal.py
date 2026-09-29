"""One whole model: conditional local joint learning plus broad proposal replay.

Same local002 protocol, with source/shared/proposal adaptation. Replayed003
observations retain their original IDs; this is not global-capture qualification.
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
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, full_frame_state_to_physical_ouv, render_finite_thickness_plane,
)
from training.arbitrary_plane_joint_loss import physical_frame_landmarks
from training.arbitrary_plane_joint_loss_v6 import _dense_mean, _weighted_mean
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6
from training.arbitrary_plane_joint_uncertainty import full_frame_residual
from training.arbitrary_plane_recurrent_model import compose_antipodal_plane_frame_residual

RUN = ROOT / "runs/joint_v6_joint_rehearsal_003"
REPLAY = ROOT / "runs/joint_v6_proposal_curriculum_003"
PARENT = ROOT / "runs/joint_v6_proposal_curriculum_003/joint_model_step_20000.pt"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001"
PACK = ROOT / "data/joint_v6_local_refinement_frozen_001"
SEED = 2026092906
STEPS, BATCH, EVALUATE_EVERY = 4000, 4, 250
REFINEMENT_STEPS, POSE_ONLY_STEPS = 3, 1
LEARNING_RATE, AMP = 2e-4, False
REHEARSAL_BATCH, GENERATED, REHEARSAL_WEIGHT = 16, 8, 1.0
RETRIEVAL_SHAPE, SUPPORT_MASS_THRESHOLD = (96, 96), 64.0
MODES = ("raw", "exact_black", "imperfect_brush")
SMALL_NOISE = (.06, .06, 250., .08, 300., 300., .05, .05, .05)
LARGE_NOISE = (.15, .15, 600., .20, 600., 600., .12, .12, .12)
TRAINABLE_PREFIXES = (
    "pose_model.atlas_stem.", "pose_model.refinement_pair_encoder.",
    "pose_model.recurrent_cell.", "pose_model.recurrent_update.",
    "pose_model.recurrent_log_likelihood.", "deformation_decoder.",
    "pose_model.coordinate_evidence.",
    "pose_model.histology_stem.", "pose_model.shared_encoder.",
    "pose_model.spatial_residual_blocks.", "pose_model.proposal_head_v6.",
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
prepared_sha256 = parent["experiment"]["prepared_source_sha256"]
for name, expected in prepared_sha256.items():
    with (PREPARED / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
catalogue = torch.load(PREPARED / "catalogue.pt", map_location="cpu", weights_only=False)
proposal_train = torch.load(PREPARED / "training_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
proposal_dev = torch.load(PREPARED / "internal_development_prepared.pt", map_location="cpu", weights_only=False, mmap=True)
assert len(proposal_train["records"]) == 5120 and len(proposal_dev["records"]) == 640
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    training_ids = {row[key] for row in train["records"] + proposal_train["records"]}
    development_ids = {row[key] for row in dev["records"] + proposal_dev["records"]}
    assert not training_ids & development_ids
replay_sha256 = {
    **parent["experiment"]["generated_schedule_sha256"],
    "experiment_source.py": parent["experiment"]["source"]["file_sha256"]["training/run_joint_v6_proposal_curriculum.py"],
}
for name, expected in replay_sha256.items():
    with (REPLAY / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
with np.load(REPLAY / "generated_schedule.npz", allow_pickle=False) as saved_schedule:
    schedule = {name: saved_schedule[name][:STEPS * GENERATED] for name in saved_schedule.files}
cell_schedule = schedule["cell_index"]
frozen_schedule = np.load(REPLAY / "frozen_training_row_indices.npy", allow_pickle=False)[:STEPS]
assert frozen_schedule.shape == (STEPS, REHEARSAL_BATCH - GENERATED)
assert cell_schedule.shape == (STEPS * GENERATED,)
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not set(schedule[key]) & {row[key] for row in proposal_dev["records"]}
RUN.mkdir(parents=True, exist_ok=False)
repository = Path(__file__).resolve().parents[1]
source_names = sorted(set(parent["experiment"]["source"]["file_sha256"]) | {
    "training/run_joint_v6_joint_rehearsal.py", "training/run_joint_v6_local_refinement.py",
    "training/arbitrary_plane_joint_uncertainty.py",
    "training/arbitrary_plane_joint_loss.py", "training/arbitrary_plane_joint_loss_v6.py",
})
source = {
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    "file_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in source_names},
}
(RUN / "source_diff.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=repository))
(RUN / "experiment_source.py").write_bytes(Path(__file__).read_bytes())
for name in replay_sha256:
    shutil.copyfile(REPLAY / name, RUN / f"replay003_{name}")
shutil.copyfile(PACK / "pack.json", RUN / "local_pack_manifest.json")
shutil.copyfile(PREPARED / "catalogue.pt", RUN / "catalogue.pt")
for name, prepared in (("training", train), ("internal_development", dev),
                       ("proposal_training", proposal_train), ("proposal_internal_development", proposal_dev)):
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
model_kwargs = {**parent["experiment"]["model_kwargs"], "deformation_integration_steps": 7,
                "joint_uncertainty_rank": None, "coordinate_evidence_conditioning": True,
                "frame_centre_offset_conditioning": False}
model = ArbitraryPlaneJointModelV6(runtime, **model_kwargs).cuda()
missing, unexpected = model.load_state_dict(parent["model_state"], strict=False)
assert missing == ["pose_model.coordinate_evidence.weight"] and not unexpected
assert torch.count_nonzero(model.pose_model.coordinate_evidence.weight) == 0
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
assert runtime.cell_count == 384 * 16 * 16
cell_states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
render_states = cell_states.to(device="cuda", dtype=torch.float32)
psf_weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device="cuda", dtype=torch.float32) / 16
psf_positions = torch.linspace(-0.5, 0.5, 9, device="cuda")
yy, xx = torch.meshgrid(torch.linspace(-1, 1, 96, device="cuda"), torch.linspace(-1, 1, 96, device="cuda"), indexing="ij")
noise_generator = torch.Generator(device="cuda")
dev_animal = np.array([row["animal_id"] for row in proposal_dev["records"]])
dev_modes = np.array([row["selected_mode"] for row in proposal_dev["records"]])
truth_center, truth_frame, _ = full_frame_state_to_components(proposal_dev["truth_state"])
truth_normal = truth_frame[:, :, 2]
proposal_support_origin = torch.as_tensor(support_origin, dtype=torch.float64)
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
    "matched_control": "local002 whole003 parent, zero-initialized coordinate hook ON, local data/seed/schedule/noise/loss/AdamW/FP32/budget unchanged; source/shared/proposal parameters unfrozen plus broad proposal NLL replay",
    "metadata_conditioning": "disabled; no surgical constraints supplied",
    "proposal_rehearsal": {
        "source_directory": str(REPLAY), "source_sha256": replay_sha256,
        "prepared_source_directory": str(PREPARED), "prepared_source_sha256": prepared_sha256,
        "local_pack_directory": str(PACK), "local_pack_manifest_sha256": hashlib.sha256((PACK / "pack.json").read_bytes()).hexdigest(),
        "cache_manifests": parent["experiment"]["cache_manifests"],
        "batch_size": REHEARSAL_BATCH, "generated_per_batch": GENERATED, "loss_weight": REHEARSAL_WEIGHT,
        "loss": "weighted full98304-cell joint NLL; no auxiliary normal NLL or kernel score",
        "comparison": "joint adaptation-plus-rehearsal intervention, not adaptation-only attribution; compare FP32 proposal endpoints to this run's FP32 step0, not old AMP metrics",
        "gradient": "local backward, then proposal backward, then one shared clip1/AdamW step",
        "schedule_selection": "original003 first4000 steps / first32000 generated observations; original IDs retained",
        "replayed_generated_observations_planned": STEPS * GENERATED, "new_observations": 0,
        "replayed_frozen_presentations_planned": STEPS * (REHEARSAL_BATCH - GENERATED),
        "unique_catalogue_cells_planned": len(np.unique(cell_schedule)),
        "unique_frozen_rows_planned": len(np.unique(frozen_schedule)),
        "generated_mode_counts": {mode: int((schedule["mode_index"] == index).sum()) for index, mode in enumerate(MODES)},
        "frozen_training_rows": 5120, "frozen_pose_anchor_rows": 3072,
        "generator": parent["experiment"]["generator"],
        "generator_note": "inherited generator describes original003 acquisition, not new observations in this continuation",
        "limitations": "replayed single-atlas synthetic observations, not biological generalization, real acquired backgrounds or full new catalogue coverage",
    },
    "new_parent_missing_parameter": "pose_model.coordinate_evidence.weight",
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


applied = generated_supervised = generated_censored = 0
proposal_evaluation = None
with (RUN / "training_trace.jsonl").open("w") as trace:
    for step in range(STEPS + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=AMP):
                objective, losses, _, _ = local_pass(train, order[step - 1], noise[step - 1], True)
            assert bool(torch.isfinite(objective))
            local_objective = float(objective.detach())
            scaler.scale(objective).backward()
            del objective
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
                    appearance_noise = torch.randn((96, 96), device="cuda", generator=noise_generator)
                    appearance = (rendered[local, 0] / finite_support[local].clamp_min(1e-6)).clamp(0, 1)
                    appearance = appearance.pow(float(schedule["gamma"][position]))
                    if schedule["invert_tissue"][position]:
                        appearance = 1.0 - appearance
                    appearance = (appearance * float(schedule["gain"][position])).clamp(0, 1)
                    slope = schedule["background_slope_yx"][position]
                    background = float(schedule["background_mean"][position]) + float(slope[0]) * yy + float(slope[1]) * xx
                    image = appearance * finite_support[local] + background * (1.0 - finite_support[local])
                    image = (image + float(schedule["noise_std"][position]) * appearance_noise).clamp(0, 1)
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
                index = frozen_schedule[step - 1]
                inputs = torch.cat((generated_inputs, proposal_train["channels"][index].cuda()))
                label = torch.cat((generated_labels, proposal_train["label"][index].cuda()))
                weight = torch.cat((generated_weight, proposal_train["weight"][index].cuda()))
            with torch.autocast("cuda", enabled=AMP):
                proposal = model.pose_model.forward_proposal_only(
                    inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1)),
                    runtime.expand(REHEARSAL_BATCH), RETRIEVAL_SHAPE,
                )
                proposal_losses = F.nll_loss(proposal["raw_full_catalogue_cell_log_probability"], label, reduction="none")
                proposal_nll = (proposal_losses * weight).sum() / weight.sum().clamp_min(1.0)
            assert bool(torch.isfinite(proposal_nll))
            scaler.scale(REHEARSAL_WEIGHT * proposal_nll).backward()
            replay_nll = float(proposal_nll.detach())
            replay_subset_nll = [float((proposal_losses[part].detach() * weight[part]).sum() / weight[part].sum().clamp_min(1.0))
                                 for part in (slice(None, GENERATED), slice(GENERATED, None))]
            generated_supervised += int(sum(weight_values))
            generated_censored += GENERATED - int(sum(weight_values))
            del proposal, proposal_losses, proposal_nll, inputs, rendered, generated_inputs
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied += int(scaler.get_scale() >= previous_scale)
            record = {"step": step, "objective": local_objective + REHEARSAL_WEIGHT * replay_nll, "local_objective": local_objective, "losses": losses,
                      "proposal_nll": replay_nll, "generated_nll": replay_subset_nll[0], "frozen_nll": replay_subset_nll[1],
                      "replayed_generated_indices": positions.tolist(), "replayed_frozen_row_indices": index.tolist(),
                      "finite_support_mass_px": support_values, "visible_support_mass_px": visible_values,
                      "generated_pose_weight": weight_values, "generated_supervised": generated_supervised, "generated_censored": generated_censored,
                      "preclip_gradient_norm": float(gradient), "row_index": order[step - 1].tolist(),
                      "optimizer_steps_applied": applied, "elapsed_seconds": time.perf_counter() - started}
            trace.write(json.dumps(record) + "\n")
            trace.flush()
            if step % 100 == 0:
                print(json.dumps(record), flush=True)
        if step in (0, STEPS):
            model.eval()
            raw = np.lib.format.open_memmap(RUN / f"proposal_development_log_probability_step_{step:05d}.npy", mode="w+", dtype=np.float32, shape=(len(proposal_dev["label"]), runtime.cell_count))
            with torch.no_grad():
                for start in range(0, len(proposal_dev["label"]), REHEARSAL_BATCH):
                    inputs = proposal_dev["channels"][start:start + REHEARSAL_BATCH].cuda()
                    with torch.autocast("cuda", enabled=AMP):
                        output = model.pose_model.forward_proposal_only(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1)), runtime.expand(len(inputs)), RETRIEVAL_SHAPE)
                    raw[start:start + len(inputs)] = output["raw_full_catalogue_cell_log_probability"].float().cpu().numpy()
            raw.flush()
            labels = proposal_dev["label"].numpy()
            truth_log = raw[np.arange(len(labels)), labels]
            ranks = (raw > truth_log[:, None]).sum(axis=1) + ((raw == truth_log[:, None]) & (np.arange(runtime.cell_count)[None] < labels[:, None])).sum(axis=1) + 1
            predicted = np.asarray(raw.argmax(axis=1))
            center, frame, _ = full_frame_state_to_components(cell_states[predicted])
            normal = frame[:, :, 2]
            dot = (normal * truth_normal).sum(-1)
            sign = torch.where(dot < 0, -1.0, 1.0)
            offset_error = (((center - proposal_support_origin) * normal).sum(-1) - sign * ((truth_center - proposal_support_origin) * truth_normal).sum(-1)).abs()
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
            support_subsets = {"identifiable": proposal_dev["weight"].numpy() > 0, "censored_low_support": proposal_dev["weight"].numpy() == 0}
            support_macro = {name: {
                "row_count": int(selected.sum()), "animal_count": len(np.unique(dev_animal[selected])),
                "animal_macro": {metric: float(np.mean([values[selected & (dev_animal == animal)].mean() for animal in np.unique(dev_animal[selected])])) for metric, values in metrics.items()} if selected.any() else None,
            } for name, selected in support_subsets.items()}
            proposal_evaluation = {"step": step, "animal_count": len(animal_metrics), "animal_macro": macro, "by_input_mode_animal_macro": mode_macro, "by_support": support_macro, "by_animal": animal_metrics, "probabilities_calibrated": False, "geometry_metric_scope": "coarse plane and antipodal frame proxies; not anatomical landmark registration", "elapsed_seconds": time.perf_counter() - started}
            (RUN / f"proposal_development_metrics_step_{step:05d}.json").write_text(json.dumps(proposal_evaluation, indent=2), encoding="utf-8")
            np.savez(RUN / f"proposal_development_rows_step_{step:05d}.npz", label=labels, prediction=predicted, pose_supervision_weight=proposal_dev["weight"].numpy(), **metrics)
            print(json.dumps({"proposal_development_step": step, "animal_macro": macro, "identifiable_animal_macro": support_macro["identifiable"]["animal_macro"]}), flush=True)
            del raw, output, inputs

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
            modes = np.array([dev["records"][row]["selected_mode"] for row in selected.tolist()])
            subsets = {f"mode:{mode}": modes == mode for mode in np.unique(modes)}
            subsets.update({"pose_identifiable": raw["pose_supervision_weight"].numpy() > 0,
                            "pose_censored": raw["pose_supervision_weight"].numpy() == 0,
                            "dense_identifiable": raw["dense_supervision_weight"].numpy() > 0,
                            "dense_censored": raw["dense_supervision_weight"].numpy() == 0})
            by_subset = {}
            for subset_name, mask in subsets.items():
                subset_macro = {}
                for key, values in metrics.items():
                    finite = mask & np.isfinite(values.numpy())
                    group_means = [float(values.numpy()[finite & (animal == group)].mean()) for group in np.unique(animal[finite])]
                    subset_macro[key] = float(np.mean(group_means)) if group_means else None
                by_subset[subset_name] = {"row_count": int(mask.sum()), "animal_count": len(np.unique(animal[mask])),
                                          "animal_macro": subset_macro}
            evaluation = {"step": step, "scope": config["scope"], "row_count": len(selected),
                          "animal_count": len(by_animal), "animal_macro": macro, "by_animal": by_animal,
                          "by_input_mode_and_censor_subset": by_subset,
                          "eligible_row_count": {key: int(torch.isfinite(value).sum()) for key, value in metrics.items()},
                          "probabilities_calibrated": False, "elapsed_seconds": time.perf_counter() - started}
            (RUN / f"local_development_metrics_step_{step:05d}.json").write_text(json.dumps(evaluation, indent=2))
            torch.save({"pack_row_index": selected, "prepared_row_index": dev["prepared_row_index"][selected],
                        "perturbation": dev_noise[selected], "metrics": metrics, **raw}, RUN / f"local_development_raw_step_{step:05d}.pt")
            checkpoint = {"experiment": config, "phase": "conditional_local_joint_with_global_rehearsal", "step": step,
                          "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                          "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                          "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                          "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                          "optimizer_steps_applied": applied, "local_development_metrics": macro,
                          "schedule_sha256": schedule_sha256, "replay_schedule_sha256": replay_sha256,
                          "proposal_development": proposal_evaluation,
                          "replayed_generated_observations": step * GENERATED, "new_observations": 0,
                          "generated_supervised": generated_supervised, "generated_censored": generated_censored,
                          "probabilities_calibrated": False}
            torch.save(checkpoint, RUN / f"joint_model_step_{step:05d}.pt.tmp")
            os.replace(RUN / f"joint_model_step_{step:05d}.pt.tmp", RUN / f"joint_model_step_{step:05d}.pt")
            print(json.dumps({"local_development_step": step, "animal_macro": macro}), flush=True)
            del checkpoint, metric_batches, raw_batches, metrics, raw

frozen_unchanged = all(torch.equal(parameter.detach().cpu(), frozen_reference[name])
                       for name, parameter in model.named_parameters() if name in frozen_reference)
assert frozen_unchanged
(RUN / "completed.json").write_text(json.dumps({"steps": STEPS, "optimizer_steps_applied": applied,
    "frozen_parameters_equal_parent": frozen_unchanged, "scope": config["scope"],
    "replayed_generated_observations": STEPS * GENERATED, "new_observations": 0,
    "unique_replayed_cells": len(np.unique(cell_schedule)), "unique_replayed_frozen_rows": len(np.unique(frozen_schedule)),
    "generated_supervised": generated_supervised, "generated_censored": generated_censored,
    "probabilities_calibrated": False, "elapsed_seconds": time.perf_counter() - started}, indent=2))
print(f"Completed joint learning with broad proposal replay, not global qualification: {RUN}", flush=True)
