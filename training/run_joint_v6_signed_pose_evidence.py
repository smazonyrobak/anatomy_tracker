"""Conditional native signed-pose evidence control from the whole original4k parent.

No global-capture, biological-calibration or benchmark claim. No legacy SVF loss.
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
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components, full_frame_state_to_components, full_frame_state_to_physical_ouv
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6
from training.arbitrary_plane_recurrent_model import compose_antipodal_plane_frame_residual
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact

READY_AFTER_PREFLIGHT = True
assert READY_AFTER_PREFLIGHT, "Commit module and complete the fixed training-batch compatibility preflight before launch"
RUN = ROOT / "runs/joint_v6_signed_pose_evidence_001"
COMPARATOR = ROOT / "runs/joint_v6_ribbon_local_001"
COMPARATOR_COMPLETION_SHA256 = "a2bee6dc1ffcaafcef3b4ad4a49223653f582f41ab071a3ca25928e7db7a83d7"
COMPARATOR_SCHEDULE_SHA256 = "e8aaa6eb1ff7e70097cbf1eeb1f6b9fc4d2ba4f01f26fc6d91b671c90c8dc0b9"
COMPARATOR_DRIVER_SHA256 = "79902a890786c67384ef0f670d83cd651469c06c4503c6f64ee06314457ceddf"
PREFLIGHT = ROOT / "runs/joint_v6_signed_pose_evidence_preflight_001/preflight.json"
PREFLIGHT_SHA256 = "4e5a2fd3a9697da6a87c435abf35f337bb383e596c46abaa2cc2560e0a39a76e"
DATA = ROOT / "data/joint_v6_coherent_subject_cohort_sections_002"
PARENT = ROOT / "runs/joint_v6_imagekey_retrieval_001/joint_model_step_04000.pt"
PARENT_AUDIT = ROOT / "runs/joint_v6_imagekey_retrieval_001_independent_audit/audit.json"
PARENT_SHA256 = "d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5"
PARENT_AUDIT_SHA256 = "f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b"
COHORT_COMPLETION_SHA256 = "ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2"
SEED, STEPS, BATCH, T, PREFIX = 2026092913, 2000, 4, 3, 1
SMALL = (.06, .06, 250., .08, 300., 300., .05, .05, .05)
LARGE = (.15, .15, 600., .20, 600., 600., .12, .12, .12)
TRAINABLE = ("pose_model.refinement_pair_encoder.", "pose_model.recurrent_cell.", "pose_model.recurrent_update.", "pose_model.recurrent_log_likelihood.", "pose_model.coordinate_evidence.", "pose_model.signed_pose_evidence.", "ribbon_field_head.")
assert len(PREFLIGHT_SHA256) == 64 and hashlib.sha256(PREFLIGHT.read_bytes()).hexdigest() == PREFLIGHT_SHA256
assert hashlib.sha256((COMPARATOR / "completed.json").read_bytes()).hexdigest() == COMPARATOR_COMPLETION_SHA256
assert hashlib.sha256((COMPARATOR / "schedule.pt").read_bytes()).hexdigest() == COMPARATOR_SCHEDULE_SHA256
assert hashlib.sha256((COMPARATOR / "source/training/run_joint_v6_ribbon_local.py").read_bytes()).hexdigest() == COMPARATOR_DRIVER_SHA256
comparator_completion = json.loads((COMPARATOR / "completed.json").read_text())
comparator_schedule = torch.load(COMPARATOR / "schedule.pt", weights_only=False, map_location="cpu")
assert all(len(value) == 64 for value in (PARENT_SHA256, PARENT_AUDIT_SHA256, COHORT_COMPLETION_SHA256))
assert hashlib.sha256(PARENT_AUDIT.read_bytes()).hexdigest() == PARENT_AUDIT_SHA256
parent_audit = json.loads(PARENT_AUDIT.read_text())
assert parent_audit["advance_to_joint_integration_pilot"] is True
with PARENT.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == PARENT_SHA256
completion_bytes = (DATA / "completed.json").read_bytes()
assert hashlib.sha256(completion_bytes).hexdigest() == COHORT_COMPLETION_SHA256
completed = json.loads(completion_bytes)
assert completed["section_count"] == 640 and completed["observation_count"] == 1920
parent = torch.load(PARENT, map_location="cpu", weights_only=False, mmap=True)
assert parent["phase"] == "experimental_image_key_proposal_only" and parent["step"] == 4000
catalogue_path = PARENT.parent / "catalogue.pt"
with catalogue_path.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == parent["experiment"]["prepared_source_sha256"]["catalogue.pt"]
catalogue = torch.load(catalogue_path, map_location="cpu", weights_only=False)
repository = Path(__file__).resolve().parents[1]
RUN.mkdir(parents=True, exist_ok=False)
(RUN / "parent_cohort_completed.json").write_bytes(completion_bytes)
shutil.copyfile(PARENT_AUDIT, RUN / "parent_independent_audit.json")
shutil.copyfile(catalogue_path, RUN / "catalogue.pt")
shutil.copyfile(PREFLIGHT, RUN / "preflight.json")
shutil.copyfile(COMPARATOR / "completed.json", RUN / "comparator_completed.json")
source_names = sorted(set(parent["experiment"]["source"]["file_sha256"]) | {
    "training/run_joint_v6_signed_pose_evidence.py", "training/arbitrary_plane_ribbon_v6.py",
    "training/arbitrary_plane_coherent_subject_v6.py", "training/arbitrary_plane_subject_section_v2.py",
    "training/subject_deformed_slab_multiresolution_bundle_v2.py",
})
source = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(), "file_sha256": {}}
for name in source_names:
    archived = RUN / "source" / name
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / name, archived)
    source["file_sha256"][name] = hashlib.sha256(archived.read_bytes()).hexdigest()

print("Loading completed raw section artifacts once; no intermediate data pack", flush=True)
physical = {key: [] for key in ("state", "plane", "centre", "slab", "director", "offsets", "weights", "reflection")}
observations, channels, visible, eligible, section_index = [], [], [], [], []
for si, record in enumerate(completed["sections"]):
    for name, expected in record["artifact_sha256"].items():
        with (DATA / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
    section = _read_raw_artifact(DATA, record["artifacts"])
    assert not section["legacy_total_2d_svf_pack_compatible"] and section["catalogue_cell_truth"] is None
    assert section["lineage"] == record["lineage"] and not section["reflection_xy"][1]
    fit = section["canonical_anatomy_plane_fit"]["arrays"]
    state = full_frame_state_from_components(*physical_ouv_to_frame(torch.from_numpy(fit["physical_ouv_ap_dv_ml_um_float64"])))
    centre = section["target_centre_ccf_coordinates_ap_dv_ml_um_float64"]
    slab = section["target_psf_ccf_coordinates_ap_dv_ml_um_float64"]
    z, w = section["axial_offsets_um_float64"], section["axial_weights_float64"]
    director = np.einsum("s,shwc->hwc", w * z, slab - centre[None]) / np.sum(w * z * z)
    values = (state, torch.from_numpy(fit["fitted_coordinate_raster_ap_dv_ml_um_float64"]), torch.from_numpy(centre), torch.from_numpy(slab), torch.from_numpy(director), torch.from_numpy(z), torch.from_numpy(w), torch.tensor(int(section["reflection_xy"][0])))
    for key, value in zip(physical, values):
        physical[key].append(value.float() if key != "reflection" else value)
    for item in section["observations"]:
        observations.append({**item["lineage"], "selected_mode": item["selected_mode"], "section_array_index": si, "finite_support_mass_px": item["finite_support_mass_px"], "visible_support_mass_px": item["visible_support_mass_px"], "support_information_eligible": item["support_information_eligible"], "section_artifact_sha256": record["artifact_sha256"]})
        channels.append(torch.from_numpy(item["image_outline_availability_float32"]))
        visible.append(torch.from_numpy(item["visible_finite_support_float32"]))
        eligible.append(item["support_information_eligible"])
        section_index.append(si)
    del section
physical = {key: torch.stack(values) for key, values in physical.items()}
channels, visible = torch.stack(channels), torch.stack(visible)
eligible, section_index = torch.tensor(eligible), torch.tensor(section_index)
split = np.array([row["split"] for row in observations])
subject = np.array([row["subject_id"] for row in observations])
mode = np.array([row["selected_mode"] for row in observations])
train_subjects, dev_subjects = np.unique(subject[split == "train"]), np.unique(subject[split == "development"])
assert len(train_subjects) == 8 and len(dev_subjects) == 4 and not set(train_subjects) & set(dev_subjects)
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not {row[key] for row in observations if row["split"] == "train"} & {row[key] for row in observations if row["split"] == "development"}
train_choices = [torch.from_numpy(np.flatnonzero((subject == name) & eligible.numpy())) for name in train_subjects]
assert all(len(choices) for choices in train_choices)
dev_rows = torch.from_numpy(np.flatnonzero(split == "development"))
assert len(channels) == 1920 and len(dev_rows) == 384
(RUN / "observation_identities.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rng = torch.Generator().manual_seed(SEED + 1)
subject_order = torch.randint(8, (STEPS, BATCH), generator=rng)
order = torch.empty_like(subject_order)
for i, choices in enumerate(train_choices):
    selected = subject_order == i
    order[selected] = choices[torch.randint(len(choices), (int(selected.sum()),), generator=rng)]
noise = 2 * torch.rand(STEPS, BATCH, 9, generator=rng) - 1
noise[:250] *= torch.tensor(SMALL)
noise[250:] *= torch.tensor(LARGE)
dev_noise = (2 * torch.rand(len(physical["state"]), 9, generator=torch.Generator().manual_seed(SEED + 2)) - 1) * torch.tensor(LARGE)
assert all(torch.equal(value, comparator_schedule[key]) for key, value in {"training_subject_index": subject_order, "training_observation_index": order, "training_perturbation": noise, "development_perturbation_by_section": dev_noise, "development_observation_index": dev_rows}.items())
shutil.copyfile(COMPARATOR / "schedule.pt", RUN / "schedule.pt")
schedule_sha256 = hashlib.sha256((RUN / "schedule.pt").read_bytes()).hexdigest()
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cuda", dtype=torch.float32)
model_kwargs = {**parent["experiment"]["model_kwargs"], "pose_only_steps": PREFIX, "coordinate_evidence_conditioning": True, "signed_pose_evidence": True, "ribbon_deformation": True, "joint_uncertainty_rank": None, "frame_centre_offset_conditioning": False}
model = ArbitraryPlaneJointModelV6(runtime, **model_kwargs).cuda()
missing, unexpected = model.load_state_dict(parent["model_state"], strict=False)
assert set(missing) == {"pose_model.coordinate_evidence.weight", "pose_model.signed_pose_evidence.weight", "ribbon_field_head.weight", "ribbon_field_head.bias"} and not unexpected
assert all(torch.count_nonzero(model.state_dict()[key]) == 0 for key in missing)
assert all(torch.equal(model.state_dict()[key].cpu(), value) for key, value in parent["model_state"].items())
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith(TRAINABLE))
frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if not name.startswith(TRAINABLE)}
trainable = [value for value in model.parameters() if value.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=1e-4)
origin, spacing, support_origin = [catalogue["support_geometry"][key] for key in ("origin_ap_dv_ml_um", "voxel_size_ap_dv_ml_um", "support_origin_ap_dv_ml_um")]
height, width = channels.shape[-2:]
yy, xx = torch.meshgrid(torch.arange(height, device="cuda") / height, torch.arange(width, device="cuda") / width, indexing="ij")
reflection_flags = torch.tensor([False, True], device="cuda")
config = {"source": source, "seed": SEED, "steps": STEPS, "batch_size": BATCH, "precision": "FP32; AMP/TF32 disabled", "optimizer": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-4, "clip_norm": 1., "model_kwargs": model_kwargs, "refinement_updates": T, "pose_prefix": PREFIX, "parent_checkpoint": str(PARENT), "parent_sha256": PARENT_SHA256, "parent_independent_audit_sha256": PARENT_AUDIT_SHA256, "cohort_completion_sha256": COHORT_COMPLETION_SHA256, "schedule_sha256": schedule_sha256, "trainable_prefixes": TRAINABLE, "frozen_state_tensor_names": list(frozen), "atlas_binding": parent["experiment"]["atlas_binding"], "catalogue_receipt_sha256": catalogue["receipt_sha256"], "small_noise_first_250": SMALL, "large_noise_remaining_and_development": LARGE, "sampling": "uniform training subject then uniform eligible observation of that subject, with replacement", "loss": "correct reflection branch only: late-iteration .8-normalized centre squared-Euclidean/100um + final PSF-weighted slab squared-Euclidean/100um + .1 canonical fitted-plane fullcanvas gauge/100um + .1 observed world-director squared error/.05 + reflection CE; per-row frozen visible finite support normalized once; final slab/director active only", "director_target": "exact-centre-anchored PSF-weighted LS of exact slab vs physical z; world coordinates", "geometric_initializer": "same perturbed state, zero fields and uniform reflection prior; selected baseline is fixed identity tie; oracle baseline is recorded reflection", "scope": "conditional truth-near known-PSF joint pose+surface pilot, not honest global capture, calibration, benchmark or biological validation", "frozen_global_scope": "exact global retrieval-producing tensors/buffers retained, no gallery rebuild; this says nothing about joint inference quality", "counts": {part: {"observations": int((split == part).sum()), "eligible": int(((split == part) & eligible.numpy()).sum()), "censored": int(((split == part) & ~eligible.numpy()).sum())} for part in ("train", "development")}, "promotion_gate": {"eligible_subject_macro_centre_and_slab_mean_reduction_min": .20, "both_readouts": ["selected", "oracle"], "no_mean_regression_each_mode_and_heldout_subject": True, "all_rows_all_branches_topology_and_orientation_positive": True, "frozen_global_tensors_exact": True}, "probabilities_calibrated": False}
config["comparator"] = {"directory": str(COMPARATOR), "completion_sha256": COMPARATOR_COMPLETION_SHA256, "schedule_sha256": COMPARATOR_SCHEDULE_SHA256, "archived_driver_sha256": COMPARATOR_DRIVER_SHA256, "archived_source_sha256": comparator_completion["experiment"]["source"]["file_sha256"]}
config["preflight_sha256"] = PREFLIGHT_SHA256
config["signed_pose_evidence"] = {"axis_sign_order": ["u+", "u-", "v+", "v-", "d+", "d-"], "normal_tangent_step_rad": float(np.deg2rad(3.)), "normal_offset_step_um": 150., "cost": "channel-normalized source/atlas cosine cost minus current-render cost; six maps, zero bias-free6-to-hidden projection", "extra_probe_renders_per_pose_update": 6, "same_current_canonical_ribbon_and_psf": True, "state_and_ribbon_gradients_attached": True, "extra_encoder_or_solver": False}
config["promotion_gate"]["oracle_normal_reduction_min"] = .20
config["promotion_gate"]["oracle_normal_nonregression_each_mode_and_subject"] = True
config["counts_by_subject_mode"] = {name: {label: {"observations": int(((subject == name) & (mode == label)).sum()), "eligible": int(((subject == name) & (mode == label) & eligible.numpy()).sum()), "censored": int(((subject == name) & (mode == label) & ~eligible.numpy()).sum())} for label in np.unique(mode)} for name in np.unique(subject)}
(RUN / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
del parent


def batch_pass(rows, perturbation):
    ids = section_index[rows]
    data = {key: value[ids].cuda() for key, value in physical.items()}
    inputs, q = channels[rows].cuda(), visible[rows].cuda()
    initial = compose_antipodal_plane_frame_residual(data["state"], perturbation.cuda(), support_origin)
    out = model.refine_ribbon(inputs[:, :1], inputs[:, 1:2], inputs[:, 2, 0, 0].bool(), atlas, initial[:, None], initial.new_full((len(rows), 1, 2), -np.log(2)), reflection_flags, (height, width), origin, spacing, support_origin, data["offsets"], data["weights"], T)
    row, branch = torch.arange(len(rows), device="cuda"), data["reflection"]
    centre = out["observed_centre_ccf_ap_dv_ml_um_sequence"][row, 0, branch, 1:]
    sequence = .8 ** torch.arange(T - 1, -1, -1, device="cuda")
    mass = q.sum((-2, -1)).clamp_min(1)
    centre_error = ((centre - data["centre"][:, None]).square().sum(-1) * q[:, None]).sum((-2, -1)) / mass[:, None] / 100.**2
    centre_loss = (centre_error * sequence).sum(-1) / sequence.sum()
    slab = out["final_observed_ccf_slab_ap_dv_ml_um"][row, 0, branch]
    slab_loss = ((slab - data["slab"]).square().sum(-1) * q[:, None] * data["weights"][:, :, None, None]).sum((1, 2, 3)) / mass / 100.**2
    state = out["final_component_state"][:, 0]
    _, frame, _ = full_frame_state_to_components(state)
    director = frame[..., :, 2][:, :, None, None] + torch.einsum("brij,brjhw->brhwi", frame, out["final_canonical_director_delta_local"][:, 0])
    director = torch.where(reflection_flags[None, :, None, None, None], director.flip(-2), director)
    director_loss = ((director[row, branch] - data["director"]).square().sum(-1) * q).sum((-2, -1)) / mass / .05**2
    ouv = full_frame_state_to_physical_ouv(state[row, branch])
    plane = ouv[:, None, None, :3] + xx[None, :, :, None] * ouv[:, None, None, 3:6] + yy[None, :, :, None] * ouv[:, None, None, 6:9]
    gauge_loss = (plane - data["plane"]).square().sum(-1).mean((-2, -1)) / 100.**2
    ce = F.nll_loss(out["conditional_kept_component_log_probability"][:, 0], branch, reduction="none")
    active = out["deformation_active_sequence"][-1].float()
    row_losses = {"centre": centre_loss, "slab": slab_loss * active, "plane_gauge": .1 * gauge_loss, "world_director": .1 * director_loss * active, "reflection_ce": ce}
    e = eligible[rows].cuda().float()
    losses = {key: (value * e).sum() / e.sum().clamp_min(1) for key, value in row_losses.items()}
    return sum(losses.values()), losses, out, initial, data, q, director


def coordinate_metrics(error, weight, normal):
    distance = error.norm(dim=-1)
    mass = weight.flatten(1).sum(-1)
    good = mass > 0
    along = (error * normal[:, None, None, None]).sum(-1).abs()
    tangent = (distance.square() - along.square()).clamp_min(0).sqrt()
    values, order = distance.flatten(1).sort(dim=-1)
    cumulative = weight.flatten(1).gather(1, order).cumsum(-1)
    position = torch.searchsorted(cumulative.contiguous(), (.95 * mass)[:, None], right=False).clamp_max(values.shape[1] - 1)
    result = {"mean_um": (distance * weight).flatten(1).sum(-1) / mass.clamp_min(1e-12), "point_p95_um": values.gather(1, position)[:, 0], "normal_mean_um": (along * weight).flatten(1).sum(-1) / mass.clamp_min(1e-12), "tangential_mean_um": (tangent * weight).flatten(1).sum(-1) / mass.clamp_min(1e-12)}
    return {key: value.masked_fill(~good, torch.nan) for key, value in result.items()}


def canonical_topology(out, offsets):
    state = out["component_state_sequence"][:, 0]
    _, frame, basis = full_frame_state_to_components(state)
    r = out["canonical_residual_local_um_sequence"][:, 0].flatten(0, 2)
    d = out["canonical_director_delta_local_sequence"][:, 0].flatten(0, 2)
    inverse = torch.linalg.inv(basis.flatten(0, 2))
    z = offsets[:, None, None].expand(-1, 2, T + 1, -1).flatten(0, 2)
    rs, rt = r.diff(dim=-1) * width, r.diff(dim=-2) * height
    ds, dt = d.diff(dim=-1) * width, d.diff(dim=-2) * height
    minimum = r.new_full((len(r),), torch.inf)
    for depth in (z.amin(-1), z.amax(-1)):
        for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)):
            derivatives = torch.stack((rs[..., y:y + height - 1, :] + depth[:, None, None, None] * ds[..., y:y + height - 1, :], rt[..., :, x:x + width - 1] + depth[:, None, None, None] * dt[..., :, x:x + width - 1]), dim=-1).permute(0, 2, 3, 1, 4)
            xy = derivatives @ inverse[:, None, None]
            axial = d[..., y:y + height - 1, x:x + width - 1].permute(0, 2, 3, 1)[..., None]
            determinant = torch.linalg.det(torch.cat((xy, axial), dim=-1) + torch.eye(3, device=r.device))
            minimum = torch.minimum(minimum, determinant.flatten(1).amin(-1))
    return minimum.reshape(state.shape[:3]).flatten(1).amin(-1), torch.linalg.det(frame).flatten(1).amin(-1)


started, applied = time.perf_counter(), 0
with (RUN / "training_trace.jsonl").open("w", encoding="utf-8") as trace:
    for step in range(STEPS + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            objective, losses, out, initial, data, q, director = batch_pass(order[step - 1], noise[step - 1])
            assert bool(torch.isfinite(objective))
            objective.backward()
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step()
            applied += 1
            record = {"step": step, "optimizer_steps_applied": applied, "loss": float(objective.detach()), "losses": {key: float(value.detach()) for key, value in losses.items()}, "preclip_gradient_norm": float(gradient), "observation_index": order[step - 1].tolist(), "subject_index": subject_order[step - 1].tolist(), "elapsed_seconds": time.perf_counter() - started}
            trace.write(json.dumps(record) + "\n")
            if step % 100 == 0:
                trace.flush()
                print(json.dumps(record), flush=True)
            del objective, losses, out, initial, data, q, director
        if step % 500 == 0:
            assert all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in frozen.items())
            checkpoint = {"experiment": config, "phase": "conditional_native_pose_ribbon", "step": step, "optimizer_steps_applied": applied, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "optimizer_state": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "global_frozen_tensors_exact": True, "probabilities_calibrated": False}
            torch.save(checkpoint, RUN / f"joint_model_step_{step:05d}.pt")
            del checkpoint
        if step not in (0, STEPS):
            continue
        model.eval()
        batches, raw_receipts = [], []
        with torch.inference_mode():
            for start in range(0, len(dev_rows), BATCH):
                rows = dev_rows[start:start + BATCH]
                fields = []
                hook = model.ribbon_field_head.register_forward_hook(lambda module, args, output: fields.append(output.detach()))
                _, _, out, initial, data, q, director = batch_pass(rows, dev_noise[section_index[rows]])
                hook.remove()
                row = torch.arange(len(rows), device="cuda")
                selected = out["conditional_kept_component_log_probability"][:, 0].argmax(-1)
                truth_branch = data["reflection"]
                _, truth_frame, _ = full_frame_state_to_components(data["state"])
                normal = truth_frame[..., :, 2]
                ouv = full_frame_state_to_physical_ouv(initial)
                base = ouv[:, None, None, :3] + xx[None, :, :, None] * ouv[:, None, None, 3:6] + yy[None, :, :, None] * ouv[:, None, None, 6:9]
                _, init_frame, _ = full_frame_state_to_components(initial)
                init_slab = base[:, None] + data["offsets"][:, :, None, None, None] * init_frame[:, None, None, None, :, 2]
                init_slab = torch.stack((init_slab, init_slab.flip(-2)), dim=1)
                base = torch.stack((base, base.flip(-2)), dim=1)
                metrics = {}
                for label, centres, slabs in (
                    ("initial_selected", base[:, 0], init_slab[:, 0]),
                    ("initial_oracle", base[row, truth_branch], init_slab[row, truth_branch]),
                    ("selected", out["final_observed_centre_ccf_ap_dv_ml_um"][row, 0, selected], out["final_observed_ccf_slab_ap_dv_ml_um"][row, 0, selected]),
                    ("oracle", out["final_observed_centre_ccf_ap_dv_ml_um"][row, 0, truth_branch], out["final_observed_ccf_slab_ap_dv_ml_um"][row, 0, truth_branch]),
                ):
                    for geometry, error, weight in (("centre", (centres - data["centre"])[:, None], q[:, None]), ("slab", slabs - data["slab"], q[:, None] * data["weights"][:, :, None, None])):
                        metrics.update({f"{label}_{geometry}_{name}": value for name, value in coordinate_metrics(error, weight, normal).items()})
                        metrics[f"{label}_{geometry}_fullcanvas_mean_um"] = error.norm(dim=-1).flatten(1).mean(-1)
                for label, branch in (("selected", selected), ("oracle", truth_branch)):
                    metrics[f"{label}_director_mean_error"] = (((director[row, branch] - data["director"]).norm(dim=-1) * q).sum((-2, -1)) / q.sum((-2, -1)).clamp_min(1e-12)).masked_fill(q.sum((-2, -1)) <= 0, torch.nan)
                    _, frame, _ = full_frame_state_to_components(out["final_component_state"][row, 0, branch])
                    predicted_normal = frame[..., :, 2]
                    metrics[f"{label}_fitted_plane_normal_angle_deg"] = torch.atan2(torch.linalg.cross(predicted_normal, normal).norm(dim=-1), (predicted_normal * normal).sum(-1).abs()) * (180 / np.pi)
                metrics["initial_fitted_plane_normal_angle_deg"] = torch.atan2(torch.linalg.cross(init_frame[..., :, 2], normal).norm(dim=-1), (init_frame[..., :, 2] * normal).sum(-1).abs()) * (180 / np.pi)
                metrics["reflection_correct"] = (selected == truth_branch).float()
                metrics["initial_reflection_correct"] = (truth_branch == 0).float()
                metrics["reflection_nll"] = -out["conditional_kept_component_log_probability"][row, 0, truth_branch]
                metrics["all_components_iterations_minimum_relative_jacobian"], metrics["all_components_iterations_minimum_frame_determinant"] = canonical_topology(out, data["offsets"])
                metrics["all_components_iterations_maximum_derivative_bound"] = out["postlimit_derivative_frobenius_bound_sequence"].flatten(1).amax(-1)
                metrics["all_components_iterations_minimum_rescale"] = out["deformation_rescale_sequence"].flatten(1).amin(-1)
                fields = F.interpolate(torch.stack(fields, dim=1).flatten(0, 1), (height, width), mode="bilinear", align_corners=False).tanh().reshape(len(rows), 2, T, 6, height, width)
                metrics["raw_surface_tanh_cap_fraction"] = (fields[..., :3, :, :].abs() >= .99).float().flatten(1).mean(-1)
                metrics["raw_director_tanh_cap_fraction"] = (fields[..., 3:, :, :].abs() >= .99).float().flatten(1).mean(-1)
                metrics["pose_update_cap_fraction"] = (out["component_pose_update_sequence"].abs() / model.pose_model.update_limits >= .99).float().flatten(1).mean(-1)
                metrics["all_outputs_finite"] = torch.tensor([all(bool(torch.isfinite(value[i]).all()) for key, value in out.items() if torch.is_tensor(value) and value.ndim and value.shape[0] == len(rows) and key != "horizontal_reflection") for i in range(len(rows))], device="cuda")
                metrics = {key: value.cpu() for key, value in metrics.items()}
                batches.append(metrics)
                raw_path = RUN / f"development_step_{step:05d}_batch_{start // BATCH:04d}.pt"
                torch.save({"observation_index": rows, "section_array_index": section_index[rows], "perturbation": dev_noise[section_index[rows]], "initial_state": initial.cpu(), "initial_selected_reflection": torch.zeros(len(rows), dtype=torch.int64), "selected_reflection": selected.cpu(), "oracle_reflection": truth_branch.cpu(), "support_information_eligible": eligible[rows], "metrics": metrics, "prediction": {key: value.cpu() if torch.is_tensor(value) else value for key, value in out.items()}}, raw_path)
                with raw_path.open("rb") as stream:
                    raw_receipts.append({"path": raw_path.name, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
                del out, initial, data, q, director, fields
        metrics = {key: torch.cat([item[key] for item in batches]).double().numpy() for key in batches[0]}
        dev_subject, dev_mode, dev_eligible = subject[dev_rows.numpy()], mode[dev_rows.numpy()], eligible[dev_rows].numpy()
        scopes = {"all": np.ones(len(dev_rows), dtype=bool), "eligible": dev_eligible, "censored": ~dev_eligible}
        scopes.update({f"eligible_mode:{name}": dev_eligible & (dev_mode == name) for name in np.unique(dev_mode)})
        scopes.update({f"eligible_subject:{name}": dev_eligible & (dev_subject == name) for name in dev_subjects})
        summaries = {}
        for name, mask in scopes.items():
            by_subject = {}
            for animal in dev_subjects:
                take = mask & (dev_subject == animal)
                by_subject[animal] = {key: float(values[take & np.isfinite(values)].mean()) if np.any(take & np.isfinite(values)) else None for key, values in metrics.items()}
            macro = {key: float(np.mean([item[key] for item in by_subject.values() if item[key] is not None])) if any(item[key] is not None for item in by_subject.values()) else None for key in metrics}
            summaries[name] = {"rows": int(mask.sum()), "subjects_with_rows": int(len(np.unique(dev_subject[mask]))), "subject_macro": macro, "by_subject": by_subject}
        gate = {}
        for readout in ("selected", "oracle"):
            for geometry in ("centre", "slab"):
                key, baseline = f"{readout}_{geometry}_mean_um", f"initial_{readout}_{geometry}_mean_um"
                overall = summaries["eligible"]["subject_macro"]
                gate[f"{readout}_{geometry}_overall_20pct"] = overall[key] is not None and overall[baseline] is not None and overall[key] <= .8 * overall[baseline]
                for name, summary in summaries.items():
                    if name.startswith(("eligible_mode:", "eligible_subject:")):
                        values = summary["subject_macro"]
                        gate[f"{readout}_{geometry}_nonregression:{name}"] = values[key] is not None and values[baseline] is not None and values[key] <= values[baseline]
        overall = summaries["eligible"]["subject_macro"]
        gate["oracle_normal_overall_20pct"] = overall["oracle_fitted_plane_normal_angle_deg"] <= .8 * overall["initial_fitted_plane_normal_angle_deg"]
        for name, summary in summaries.items():
            if name.startswith(("eligible_mode:", "eligible_subject:")):
                values = summary["subject_macro"]
                gate[f"oracle_normal_nonregression:{name}"] = values["oracle_fitted_plane_normal_angle_deg"] <= values["initial_fitted_plane_normal_angle_deg"]
        gate["all_outputs_finite"] = bool(np.all(metrics["all_outputs_finite"] == 1))
        gate["canonical_orientation_and_no_folds"] = bool(np.all(metrics["all_components_iterations_minimum_relative_jacobian"] > 0) and np.all(metrics["all_components_iterations_minimum_frame_determinant"] > 0) and np.all(metrics["all_components_iterations_maximum_derivative_bound"] <= .35001))
        gate["frozen_global_tensors_exact"] = all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in frozen.items())
        evaluation = {"step": step, "optimizer_steps_applied": applied, "summaries": summaries, "gate_conditions": gate, "conditional_local_gate_passed": bool(all(gate.values())), "raw_prediction_files": raw_receipts, "scope": config["scope"], "probabilities_calibrated": False, "elapsed_seconds": time.perf_counter() - started}
        evaluation["difference_from_no_probe_comparator"] = {key: summaries["eligible"]["subject_macro"][key] - comparator_completion["final_evaluation"]["summaries"]["eligible"]["subject_macro"][key] for key in ("oracle_centre_mean_um", "oracle_slab_mean_um", "oracle_fitted_plane_normal_angle_deg", "selected_centre_mean_um", "selected_slab_mean_um")}
        (RUN / f"development_metrics_step_{step:05d}.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"step": step, "eligible_subject_macro": summaries["eligible"]["subject_macro"], "conditional_local_gate_passed": evaluation["conditional_local_gate_passed"]}), flush=True)
for name, expected in source["file_sha256"].items():
    assert hashlib.sha256((repository / name).read_bytes()).hexdigest() == expected
(RUN / "completed.json").write_text(json.dumps({"optimizer_steps_applied": applied, "source_unchanged": True, "final_evaluation": evaluation, "experiment": config}, indent=2, allow_nan=False), encoding="utf-8")
