"""Fresh image-key retrieval control; no checkpoint import or joint-training claim."""

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
from torch.utils.checkpoint import checkpoint

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, full_frame_state_to_physical_ouv, render_finite_thickness_plane,
)
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

RUN = ROOT / "runs/joint_v6_imagekey_retrieval_001"
REPLAY = ROOT / "runs/joint_v6_proposal_curriculum_003"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001"
SEED, NEGATIVE_SEED = 2026092805, 2026092910
STEPS, BATCH, GENERATED = 4000, 16, 8
SHAPE, DESCRIPTOR_DIM, KEY_CHUNK = (96, 96), 256, 16
TEMPERATURE, GALLERY_THICKNESS_UM = .1, 50.
LEARNING_RATE, SUPPORT_MASS_THRESHOLD = .001, 64.
IGNORE_ANGLE_DEG, IGNORE_CORNER_UM = 10., 1000.
TRAINABLE = ("pose_model.histology_stem.", "pose_model.atlas_stem.",
             "pose_model.shared_encoder.", "pose_model.image_key_descriptor.")

previous = json.loads((REPLAY / "experiment.json").read_text())
source_hashes = {"experiment.json": "43f78e82b3d550cabd3f3a974d47d5d5256d9a9126a309cedb15b223eec70b07",
                 "experiment_source.py": previous["source"]["file_sha256"]["training/run_joint_v6_proposal_curriculum.py"],
                 **previous["generated_schedule_sha256"]}
for directory, expected_hashes in ((PREPARED, previous["prepared_source_sha256"]), (REPLAY, source_hashes)):
    for name, expected in expected_hashes.items():
        with (directory / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
catalogue = torch.load(PREPARED / "catalogue.pt", weights_only=False, map_location="cpu")
train = torch.load(PREPARED / "training_prepared.pt", weights_only=False, map_location="cpu", mmap=True)
dev = torch.load(PREPARED / "internal_development_prepared.pt", weights_only=False, map_location="cpu", mmap=True)
with np.load(REPLAY / "generated_schedule.npz", allow_pickle=False) as saved:
    schedule = {name: saved[name][:STEPS * GENERATED] for name in saved.files}
frozen_schedule = np.load(REPLAY / "frozen_training_row_indices.npy", allow_pickle=False)[:STEPS]
baseline_report = ROOT / "runs/joint_v6_imagekey_baseline_003_step4000/baseline.json"
baseline_sha256 = "5672303902e0b3d108507e40a883b848418c656c20e1bdc48754469009961eeb"
assert hashlib.sha256(baseline_report.read_bytes()).hexdigest() == baseline_sha256
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not ({row[key] for row in train["records"]} | set(schedule[key])) & {row[key] for row in dev["records"]}

RUN.mkdir(parents=True, exist_ok=False)
repository = Path(__file__).resolve().parents[1]
source_names = sorted(set(previous["source"]["file_sha256"]) | {Path(__file__).relative_to(repository).as_posix()})
source = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
          "file_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in source_names}}
(RUN / "source_diff.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=repository))
(RUN / "experiment_source.py").write_bytes(Path(__file__).read_bytes())
for name in source_names:
    target = RUN / "source" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / name, target)
shutil.copyfile(PREPARED / "catalogue.pt", RUN / "catalogue.pt")
shutil.copyfile(baseline_report, RUN / "baseline003_step4000.json")
for name in source_hashes:
    shutil.copyfile(REPLAY / name, RUN / f"replay003_{name}")
for name, prepared in (("training", train), ("internal_development", dev)):
    (RUN / f"{name}_identities.json").write_text(json.dumps(prepared["records"], indent=2))
negative_rng = np.random.default_rng(NEGATIVE_SEED)
global_schedule = negative_rng.integers(0, 98304, (STEPS, 32), dtype=np.int64)
local_rank_schedule = negative_rng.integers(0, 8, (STEPS, BATCH), dtype=np.int64)
np.savez(RUN / "negative_schedule.npz", global_cell_index=global_schedule, local_pool_rank=local_rank_schedule)
negative_sha256 = hashlib.sha256((RUN / "negative_schedule.npz").read_bytes()).hexdigest()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
started = time.perf_counter()
print("Decoding pinned Allen atlas for fresh image-key retrieval", flush=True)
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cuda", dtype=torch.float32)
assert runtime.cell_count == 98304 and runtime.representation_count == 2
model_kwargs = {**previous["model_kwargs"], "image_key_descriptor_dim": DESCRIPTOR_DIM}
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
model = ArbitraryPlaneJointModelV6(runtime, **model_kwargs).cuda()
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith(TRAINABLE))
trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=1e-4)
cell_states = torch.as_tensor(catalogue["arrays"]["cell_states_float64"])
render_states = cell_states.cuda().float()
cell_center, cell_frame, _ = full_frame_state_to_components(cell_states)
cell_normal = cell_frame[:, :, 2]
ouv = full_frame_state_to_physical_ouv(cell_states).reshape(-1, 3, 3)
finite_u, finite_v = ouv[:, 1] * (95 / 96), ouv[:, 2] * (95 / 96)
finite_center = ouv[:, 0] + .5 * (finite_u + finite_v)
key_center, key_u, key_v, key_normal = [value.cuda().float() for value in (finite_center, finite_u, finite_v, cell_normal)]
cell_log_mass = catalogue["tensors"]["cell_log_mass"][0].cuda().float()
representation_log_weight = catalogue["tensors"]["representation_log_weight"][0].cuda().float()
affine = np.asarray(catalogue["arrays"]["representation_to_canonical_raster_affine_float64"])[0]
# Bound affines use centered grid_sample coordinates, not the [0,1] edge chart.
assert np.array_equal(affine, np.array([[[1., 0., 0.], [0., 1., 0.]], [[-1., 0., 0.], [0., 1., 0.]]]))
psf_weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device="cuda", dtype=torch.float32) / 16
psf_positions = torch.linspace(-.5, .5, 9, device="cuda")
yy, xx = torch.meshgrid(torch.linspace(-1, 1, 96, device="cuda"), torch.linspace(-1, 1, 96, device="cuda"), indexing="ij")
noise_generator = torch.Generator(device="cuda")
origin = catalogue["support_geometry"]["origin_ap_dv_ml_um"]
spacing = catalogue["support_geometry"]["voxel_size_ap_dv_ml_um"]
support_origin = torch.as_tensor(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=torch.float64)
truth_center, truth_frame, _ = full_frame_state_to_components(dev["truth_state"])
truth_normal = truth_frame[:, :, 2]
dev_group = np.array([row["animal_id"] for row in dev["records"]])
dev_mode = np.array([row["selected_mode"] for row in dev["records"]])
config = {
    "source": source, "seed": SEED, "negative_seed": NEGATIVE_SEED, "steps": STEPS,
    "batch_size": BATCH, "generated_per_batch": GENERATED, "model_kwargs": model_kwargs,
    "initialization": "fresh whole random model; no checkpoint weights/features/pseudolabels loaded",
    "trained_modules": TRAINABLE, "trainable_parameters": sum(p.numel() for p in trainable),
    "untrained_modules": "original parametric proposal, reranker, recurrent pose, deformation and uncertainty; not deployable",
    "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": 1e-4, "gradient_clip": 5.,
    "precision": "FP32 training; FP16 normalized gallery storage followed by FP32 re-normalization/scoring",
    "comparison": "original003 step4000, identical query schedule/seed/initial original parameters; mechanism, training objective, keys and precision differ; not a single-cause ablation",
    "prepared_source_directory": str(PREPARED), "prepared_source_sha256": previous["prepared_source_sha256"],
    "replay_directory": str(REPLAY), "replay_sha256": source_hashes, "negative_schedule_sha256": negative_sha256,
    "replayed_generated_observations": STEPS * GENERATED, "replayed_frozen_presentations": STEPS * (BATCH - GENERATED),
    "new_query_observations": 0, "generator": previous["generator"], "atlas_binding": previous["atlas_binding"],
    "catalogue_receipt_sha256": catalogue["receipt_sha256"], "temperature": TEMPERATURE,
    "candidate_sampling": "deduplicated union of <=16 batch truths +32 uniform draws +one of 8 nearest admissible cells per query; <=64 cells; no refill",
    "local_pool": "nearest finite four-corner RMS after excluding own truth and cells satisfying BOTH angle<=10deg and RMS<=1000um; min identity/horizontal raster correspondence, never anatomical ML reflection",
    "loss": "weighted sampled cell NLL; logcellmass + logsumexp_r(cosine/0.1 + logrepresentationprior), priors once; ignored near cells omitted per query, own truth always retained",
    "sampling_bias": "no importance correction; sampled classification objective is NOT unbiased full-gallery NLL or calibrated uncertainty",
    "training_keys": "fresh current shared encoder; exact horizontal pixel permutation BEFORE encoding; checkpointed 16-image chunks",
    "gallery": {"shape_h_w": SHAPE, "thickness_um": GALLERY_THICKNESS_UM, "psf": "normalized boxcar-trapezoid9", "representations": "identity,horizontal x->95-x", "cell_count": runtime.cell_count},
    "evaluation": "complete newly rebuilt gallery at steps0/4000, no truth injection/proximity censor; 640 fixed development rows, eligible/all/censored and eligible mode intersections",
    "reflection_readout": "latent representation conditionals saved at truth cell and top128 cells; prepared coarse rows do not carry authenticated reflection labels, so no reflection-accuracy claim",
    "baseline_report": {"path": str(baseline_report), "sha256": baseline_sha256},
    "promotion_gate": {"eligible_group_macro_normal_improvement_deg_min": 5., "eligible_group_macro_top32_plane_capture_improvement_min": .05,
                       "each_eligible_mode_normal_regression_deg_max": 2., "each_eligible_mode_top32_plane_capture_regression_max": .02,
                       "plane_capture": "any topK cell has antipodal normal<=10deg AND sign-aligned normaloffset<=500um against continuous truth"},
    "scope": "organizational groups from one atlas, not biological animals; no benchmarks, constraints, joint adaptation, calibration or trajectory claims",
    "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
}
(RUN / "experiment.json").write_text(json.dumps(config, indent=2))


def encode_keys(images):
    return model.pose_model.descriptor_from_features(model.pose_model._encode_atlas(images))


applied_steps = generated_supervised = generated_censored = 0
training_seconds = 0.
with (RUN / "training_trace.jsonl").open("w") as trace:
    for step in range(STEPS + 1):
        if step:
            tick = time.perf_counter()
            model.train()
            positions = np.arange((step - 1) * GENERATED, step * GENERATED)
            with torch.no_grad():
                generated_labels = torch.as_tensor(schedule["cell_index"][positions], device="cuda")
                thickness = torch.as_tensor(schedule["thickness_um"][positions], device="cuda")
                rendered = render_finite_thickness_plane(atlas, render_states[generated_labels], SHAPE, origin, spacing, thickness[:, None] * psf_positions[None], psf_weights)
                finite_support = rendered[:, 1].clamp(0, 1)
                tissue = finite_support > 0
                generated_inputs = torch.empty((GENERATED, 3, 96, 96), device="cuda")
                visible_mass = torch.empty(GENERATED, device="cuda")
                for local, position in enumerate(positions):
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
                        image = image * mask
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
                index = frozen_schedule[step - 1]
                inputs = torch.cat((generated_inputs, train["channels"][index].cuda()))
                label = torch.cat((generated_labels, train["label"][index].cuda()))
                weight = torch.cat((generated_weight, train["weight"][index].cuda()))
                if step == 1:
                    torch.save({"sample_indices": positions, "input_channels": generated_inputs.cpu(), "state": cell_states[schedule["cell_index"][positions]], "support_mass": support_mass.cpu().tolist(), "visible_mass": visible_mass.cpu().tolist()}, RUN / "first_generated_batch.pt")
                distance2 = (key_center[label, None] - key_center[None]).square().sum(-1)
                same_u = (key_u[label, None] - key_u[None]).square().sum(-1)
                flip_u = (key_u[label, None] + key_u[None]).square().sum(-1)
                distance2 += .25 * (torch.minimum(same_u, flip_u) + (key_v[label, None] - key_v[None]).square().sum(-1))
                ignore = ((key_normal[label] @ key_normal.T).abs() >= np.cos(np.deg2rad(IGNORE_ANGLE_DEG))) & (distance2 <= IGNORE_CORNER_UM ** 2)
                ignore.scatter_(1, label[:, None], True)
                local_pool = distance2.masked_fill(ignore, torch.inf).topk(8, largest=False, sorted=True).indices
                local_ids = local_pool[torch.arange(BATCH, device="cuda"), torch.as_tensor(local_rank_schedule[step - 1], device="cuda")]
                candidate = torch.unique(torch.cat((label, torch.as_tensor(global_schedule[step - 1], device="cuda"), local_ids)), sorted=True)
                target = torch.searchsorted(candidate, label)
                keep = ~ignore[:, candidate]
                keep.scatter_(1, target[:, None], True)
                key_images = render_finite_thickness_plane(atlas, render_states[candidate], SHAPE, origin, spacing, psf_positions * GALLERY_THICKNESS_UM, psf_weights)
                key_images = torch.stack((key_images, key_images.flip(-1)), dim=1).flatten(0, 1)
                del distance2, same_u, flip_u, ignore, local_pool
            optimizer.zero_grad(set_to_none=True)
            query = model.pose_model.descriptor_from_features(model.pose_model.encode_histology(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1))))
            keys = torch.cat([checkpoint(encode_keys, part, use_reentrant=False) for part in key_images.split(KEY_CHUNK)]).reshape(len(candidate), 2, DESCRIPTOR_DIM)
            scores = model.pose_model.image_key_cosine_logits(query, keys, TEMPERATURE)
            scores = torch.logsumexp(scores + representation_log_weight[candidate][None], dim=-1) + cell_log_mass[candidate][None]
            losses = torch.logsumexp(scores.masked_fill(~keep, -torch.inf), dim=-1) - scores.gather(1, target[:, None])[:, 0]
            loss = (losses * weight).sum() / weight.sum().clamp_min(1.)
            assert bool(torch.isfinite(loss))
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
            if weight.sum() > 0:
                optimizer.step()
                applied_steps += 1
            generated_supervised += int(generated_weight.sum())
            generated_censored += GENERATED - int(generated_weight.sum())
            training_seconds += time.perf_counter() - tick
            record = {"step": step, "sampled_nll": float(loss.detach()), "gradient_norm": float(gradient),
                      "optimizer_steps_applied": applied_steps, "candidate_ids": candidate.cpu().tolist(), "local_negative_ids": local_ids.cpu().tolist(),
                      "ignored_candidate_counts": (~keep).sum(-1).cpu().tolist(), "generated_indices": positions.tolist(), "frozen_row_indices": index.tolist(),
                      "finite_support_mass_px": support_mass.cpu().tolist(), "visible_support_mass_px": visible_mass.cpu().tolist(),
                      "generated_point_pose_weight": generated_weight.cpu().tolist(), "training_seconds": training_seconds}
            trace.write(json.dumps(record) + "\n")
            if step % 100 == 0:
                trace.flush()
                print(json.dumps({name: record[name] for name in ("step", "sampled_nll", "gradient_norm", "optimizer_steps_applied", "training_seconds")}), flush=True)
            del keys, query, scores, losses, loss, inputs, key_images, rendered, generated_inputs

        if step in (0, STEPS):
            model.eval()
            endpoint = {"experiment": config, "phase": "experimental_image_key_proposal_only", "step": step,
                        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                        "optimizer_state": optimizer.state_dict(), "optimizer_steps_applied": applied_steps,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "probabilities_calibrated": False}
            checkpoint_path = RUN / f"joint_model_step_{step:05d}.pt"
            torch.save(endpoint, checkpoint_path)
            with checkpoint_path.open("rb") as stream:
                checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            del endpoint
            bank = torch.empty((runtime.cell_count, 2, DESCRIPTOR_DIM), device="cuda", dtype=torch.float16)
            with torch.no_grad():
                for start in range(0, runtime.cell_count, 64):
                    images = render_finite_thickness_plane(atlas, render_states[start:start + 64], SHAPE, origin, spacing, psf_positions * GALLERY_THICKNESS_UM, psf_weights)
                    images = torch.stack((images, images.flip(-1)), dim=1).flatten(0, 1)
                    bank[start:start + 64] = torch.cat([encode_keys(part) for part in images.split(KEY_CHUNK)]).reshape(-1, 2, DESCRIPTOR_DIM).half()
                    if (start + 64) % 16384 == 0:
                        print(json.dumps({"gallery_step": step, "cells_encoded": start + 64, "elapsed_seconds": time.perf_counter() - started}), flush=True)
                bank_path = RUN / f"gallery_descriptors_step_{step:05d}.npy"
                np.save(bank_path, bank.cpu().numpy())
                with bank_path.open("rb") as stream:
                    bank_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
                raw = np.lib.format.open_memmap(RUN / f"development_log_probability_step_{step:05d}.npy", mode="w+", dtype=np.float32, shape=(len(dev["label"]), runtime.cell_count))
                query_bank, top_indices, truth_rep, top_rep = [], [], [], []
                for start in range(0, len(dev["label"]), BATCH):
                    inputs = dev["channels"][start:start + BATCH].cuda()
                    query = model.pose_model.descriptor_from_features(model.pose_model.encode_histology(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1))))
                    representation_score = model.pose_model.image_key_cosine_logits(query, bank, TEMPERATURE) + representation_log_weight[None]
                    scores = torch.logsumexp(representation_score, dim=-1) + cell_log_mass[None]
                    log_probability = scores.log_softmax(dim=-1)
                    raw[start:start + len(inputs)] = log_probability.cpu().numpy()
                    top = torch.argsort(log_probability, dim=-1, descending=True, stable=True)[:, :128]
                    rows = torch.arange(len(inputs), device="cuda")
                    labels_here = dev["label"][start:start + len(inputs)].cuda()
                    query_bank.append(query.cpu())
                    top_indices.append(top.cpu())
                    truth_rep.append(representation_score[rows, labels_here].softmax(-1).cpu())
                    top_rep.append(representation_score[rows[:, None], top].softmax(-1).cpu())
                raw.flush()
            top_indices = torch.cat(top_indices)
            labels = dev["label"].numpy()
            truth_lp = np.asarray(raw[np.arange(len(labels)), labels])
            rank = (raw > truth_lp[:, None]).sum(-1) + ((raw == truth_lp[:, None]) & (np.arange(runtime.cell_count)[None] < labels[:, None])).sum(-1) + 1
            predicted = top_indices[:, 0]
            normals = cell_normal[top_indices]
            dot = (normals * truth_normal[:, None]).sum(-1)
            angles = torch.rad2deg(torch.acos(dot.abs().clamp(0, 1)))
            offsets = (((cell_center[top_indices] - support_origin) * normals).sum(-1) - torch.where(dot < 0, -1., 1.) * ((truth_center - support_origin) * truth_normal).sum(-1)[:, None]).abs()
            frame_cosine = ((cell_frame[predicted] * truth_frame).sum((-2, -1)) - 1.) / 2.
            antipodal_cosine = ((cell_frame[predicted] * torch.tensor([-1., 1., -1.]) * truth_frame).sum((-2, -1)) - 1.) / 2.
            normal_lp = np.logaddexp.reduce(np.asarray(raw).reshape(len(labels), 384, 256), axis=-1)
            marginal_normals = cell_normal[normal_lp.argmax(-1) * 256]
            metrics = {"nll": -truth_lp, "truth_rank": rank, "plane_angle_deg": angles[:, 0].numpy(),
                       "normal_offset_error_um": offsets[:, 0].numpy(),
                       "antipodal_frame_angle_deg": torch.rad2deg(torch.acos(torch.maximum(frame_cosine, antipodal_cosine).clamp(-1, 1))).numpy(),
                       "normal_marginal_nll": -normal_lp[np.arange(len(labels)), labels // 256],
                       "normal_marginal_map_angle_deg": torch.rad2deg(torch.acos((marginal_normals * truth_normal).sum(-1).abs().clamp(0, 1))).numpy(),
                       **{f"hit_at_{k}": (rank <= k).astype(float) for k in (1, 8, 32, 128)},
                       **{f"physical_plane_capture_at_{k}": ((angles[:, :k] <= 10.) & (offsets[:, :k] <= 500.)).any(-1).double().numpy() for k in (32, 128)}}
            subsets = {"all": np.ones(len(labels), dtype=bool), "eligible": dev["weight"].numpy() > 0, "censored": dev["weight"].numpy() == 0,
                       **{f"eligible_mode:{mode}": (dev["weight"].numpy() > 0) & (dev_mode == mode) for mode in np.unique(dev_mode)}}
            summaries = {name: {"rows": int(selected.sum()), "groups": len(np.unique(dev_group[selected])),
                               "group_macro": {metric: float(np.mean([values[selected & (dev_group == group)].mean() for group in np.unique(dev_group[selected])])) for metric, values in metrics.items()} if selected.any() else None}
                         for name, selected in subsets.items()}
            np.savez(RUN / f"development_rows_step_{step:05d}.npz", label=labels, prediction=predicted.numpy(), pose_supervision_weight=dev["weight"].numpy(),
                     query_descriptor=torch.cat(query_bank).numpy(), top128_cell_index=top_indices.numpy(),
                     truth_cell_representation_probability=torch.cat(truth_rep).numpy(), top128_representation_probability=torch.cat(top_rep).numpy(), **metrics)
            receipt = {"step": step, "checkpoint_sha256": checkpoint_sha256, "gallery_sha256": bank_sha256,
                       "catalogue_sha256": previous["prepared_source_sha256"]["catalogue.pt"], "source": source,
                       "gallery": config["gallery"], "temperature": TEMPERATURE, "priors_applied_once": True,
                       "subsets": summaries, "probabilities_calibrated": False, "elapsed_seconds": time.perf_counter() - started}
            (RUN / f"development_metrics_step_{step:05d}.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps({"development_step": step, "eligible": summaries["eligible"], "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)
            del bank, raw, representation_score, scores, log_probability, inputs, images, query

(RUN / "completed.json").write_text(json.dumps({"steps": STEPS, "optimizer_steps_applied": applied_steps,
    "replayed_generated_observations": STEPS * GENERATED, "replayed_frozen_presentations": STEPS * (BATCH - GENERATED),
    "new_query_observations": 0, "generated_supervised": generated_supervised, "generated_censored": generated_censored,
    "training_seconds": training_seconds, "elapsed_seconds": time.perf_counter() - started}, indent=2))
print(f"Finished image-key retrieval control: {RUN}", flush=True)
