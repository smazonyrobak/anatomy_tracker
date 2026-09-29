"""Matched whole-parent optional-outline dropout continuation; source prepared only."""
import os
import sys
from pathlib import Path

ROOT = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True

import copy
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
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components, full_frame_state_to_physical_ouv, render_finite_thickness_plane
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6

RUN = ROOT / "runs/joint_v6_imagekey_outline_dropout_001"
PARENT = ROOT / "runs/joint_v6_imagekey_retrieval_001"
AUDIT = ROOT / "runs/joint_v6_imagekey_retrieval_001_independent_audit/audit.json"
RAW = ROOT / "runs/joint_v6_imagekey_retrieval_001_allen_raw"
PREPARED = ROOT / "runs/joint_v6_proposal_substantive_001"
REPLAY = ROOT / "runs/joint_v6_proposal_curriculum_003"
PARENT_SHA256 = "d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5"
AUDIT_SHA256 = "f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b"
RAW_SUMMARY_SHA256 = "2394eae62ef6c64e04b41947e57ca94dd64a2c456e0bc04a58b8dc2997107b37"
START, STEPS, BATCH, GENERATED = 4000, 2000, 16, 8
NEGATIVE_SEED, DROPOUT_SEED = 2026092914, 2026092915
SHAPE, DESCRIPTOR_DIM, KEY_CHUNK = (96, 96), 256, 16
TEMPERATURE, GALLERY_THICKNESS_UM, SUPPORT_MASS_THRESHOLD = .1, 50., 64.
TRAINABLE = ("pose_model.histology_stem.", "pose_model.atlas_stem.", "pose_model.shared_encoder.", "pose_model.image_key_descriptor.")
repository = Path(__file__).resolve().parents[1]
bindings = {}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


for path, expected in ((AUDIT, AUDIT_SHA256), (PARENT / "joint_model_step_04000.pt", PARENT_SHA256), (RAW / "summary.json", RAW_SUMMARY_SHA256)):
    assert sha(path) == expected
    bindings[str(path)] = expected
audit = json.loads(AUDIT.read_text())
assert audit["integrity_passed"] and audit["performance_gates_passed"] and audit["advance_to_joint_integration_pilot"]
parent = torch.load(PARENT / "joint_model_step_04000.pt", weights_only=False, map_location="cpu", mmap=True)
previous = parent["experiment"]
assert parent["step"] == parent["optimizer_steps_applied"] == START and parent["phase"] == "experimental_image_key_proposal_only"
assert tuple(previous["trained_modules"]) == TRAINABLE and previous["learning_rate"] == .001 and previous["weight_decay"] == 1e-4
for directory, expected_hashes in ((PREPARED, previous["prepared_source_sha256"]), (REPLAY, previous["replay_sha256"]), (PARENT, {name: audit["frozen_run_inventory_sha256"][name] for name in ("development_metrics_step_04000.json", "catalogue.pt")})):
    for name, expected in expected_hashes.items():
        assert sha(directory / name) == expected, name
        bindings[str(directory / name)] = expected
raw_parent = json.loads((RAW / "summary.json").read_text())
assert raw_parent["checkpoint_sha256"] == PARENT_SHA256 and raw_parent["section_count"] == 64 and raw_parent["animal_count"] == 6
for name in ("raw_model_input.npy", "image_geometry.jsonl"):
    assert sha(RAW / name) == raw_parent["output_sha256"][name]
    bindings[str(RAW / name)] = raw_parent["output_sha256"][name]
raw_images = np.load(RAW / "raw_model_input.npy", allow_pickle=False)
raw_records = [json.loads(line) for line in (RAW / "image_geometry.jsonl").read_text().splitlines()]
assert raw_images.shape == (64, 1, 96, 96) and all(row["split"] == "development_validation" for row in raw_records)
catalogue = torch.load(PREPARED / "catalogue.pt", weights_only=False, map_location="cpu")
train = torch.load(PREPARED / "training_prepared.pt", weights_only=False, map_location="cpu", mmap=True)
dev = torch.load(PREPARED / "internal_development_prepared.pt", weights_only=False, map_location="cpu", mmap=True)
with np.load(REPLAY / "generated_schedule.npz", allow_pickle=False) as saved:
    schedule = {name: saved[name][START * GENERATED:(START + STEPS) * GENERATED] for name in saved.files}
frozen_schedule = np.load(REPLAY / "frozen_training_row_indices.npy", allow_pickle=False)[START:START + STEPS]
assert len(schedule["cell_index"]) == STEPS * GENERATED and frozen_schedule.shape == (STEPS, BATCH - GENERATED)
for key in ("animal_id", "specimen_id", "experiment_id", "synthetic_animal_id", "section_id"):
    assert not ({row[key] for row in train["records"]} | set(schedule[key])) & {row[key] for row in dev["records"]}
negative_rng = np.random.default_rng(NEGATIVE_SEED)
global_schedule = negative_rng.integers(0, 98304, (STEPS, 32), dtype=np.int64)
local_rank_schedule = negative_rng.integers(0, 8, (STEPS, BATCH), dtype=np.int64)
dropout_draw = np.random.default_rng(DROPOUT_SEED).random((STEPS, BATCH)) < .5
available_schedule = np.concatenate((schedule["mode_index"].reshape(STEPS, GENERATED) != 0, train["channels"][frozen_schedule, 2, 0, 0].numpy().astype(bool)), axis=1)
dropout_mask = dropout_draw & available_schedule

RUN.mkdir(parents=True, exist_ok=False)
source_names = sorted(set(previous["source"]["file_sha256"]) | {Path(__file__).relative_to(repository).as_posix()})
source = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(), "file_sha256": {}}
for name in source_names:
    target = RUN / "source" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / name, target)
    source["file_sha256"][name] = sha(target)
    if not name.startswith("training/run_"):
        assert source["file_sha256"][name] == previous["source"]["file_sha256"][name], name
np.savez(RUN / "generated_schedule.npz", **schedule)
np.savez(RUN / "shared_schedule.npz", frozen_row_indices=frozen_schedule, global_cell_index=global_schedule, local_pool_rank=local_rank_schedule, dropout_draw=dropout_draw, original_outline_available=available_schedule, arm_B_outline_dropped=dropout_mask)
for name, prepared in (("training", train), ("development", dev)):
    (RUN / f"{name}_identities.json").write_text(json.dumps(prepared["records"], indent=2), encoding="utf-8")
for original, copied in ((PARENT / "catalogue.pt", "catalogue.pt"), (PARENT / "development_metrics_step_04000.json", "parent_synthetic_step4000.json"), (AUDIT, "parent_independent_audit.json"), (RAW / "summary.json", "parent_allen_raw_summary.json"), (RAW / "raw_model_input.npy", "allen_raw_model_input.npy"), (RAW / "image_geometry.jsonl", "allen_raw_image_geometry.jsonl")):
    shutil.copyfile(original, RUN / copied)
shared_schedule_sha256 = {name: sha(RUN / name) for name in ("generated_schedule.npz", "shared_schedule.npz", "training_identities.json", "development_identities.json")}
config = {"source": source, "parent_checkpoint": str(PARENT / "joint_model_step_04000.pt"), "parent_sha256": PARENT_SHA256, "input_sha256": bindings, "shared_schedule_sha256": shared_schedule_sha256,
    "arms": {"A": "unchanged optional channels", "B": "independent p=.5 draw, zero boundary+availability only where original availability is true; image pixels unchanged"},
    "initialization": "both arms strict whole-parent model and independent deep copies of AdamW state; same parent Python/NumPy/Torch/CUDA RNG restored after construction", "parent_step": START, "additional_applied_updates_per_arm": STEPS, "final_cumulative_step": START + STEPS,
    "negative_seed": NEGATIVE_SEED, "dropout_seed": DROPOUT_SEED, "generated_schedule_slice": [32000, 48000], "frozen_schedule_slice": [4000, 6000], "batch_size": BATCH, "generated_per_batch": GENERATED,
    "model_kwargs": previous["model_kwargs"], "trained_modules": TRAINABLE, "optimizer": "AdamW", "learning_rate": .001, "weight_decay": 1e-4, "gradient_clip": 5., "precision": previous["precision"],
    "temperature": TEMPERATURE, "gallery": previous["gallery"], "atlas_binding": previous["atlas_binding"], "catalogue_receipt_sha256": catalogue["receipt_sha256"], "generator": previous["generator"],
    "candidate_sampling": previous["candidate_sampling"], "local_pool": previous["local_pool"], "loss": previous["loss"], "sampling_bias": previous["sampling_bias"],
    "evaluation": "full newly rebuilt98304x2 gallery only at final6000 of each arm; all640 unchanged synthetic rows and64 hash-bound unchanged raw Allen pixels/6 development donors; no train-donor predictions", "dropout_at_evaluation": False,
    "promotion_gate": {"B_vs_A_real_donor_macro_normal_improvement_deg_min": 5., "B_vs_A_real_top32_plane_capture_improvement_min": .10, "B_synthetic_retention_vs": ["A6000", "parent4000"], "synthetic_normal_regression_deg_max": 2., "synthetic_top32_plane_capture_regression_max": .02, "synthetic_subsets": "eligible overall and each populated eligible original input mode"},
    "scope": "metadata-association control; no general appearance-correction, calibrated uncertainty, joint-model qualification, independent final validation or public benchmark claim", "probabilities_calibrated": False}
(RUN / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
started = time.perf_counter()
print("Decoding pinned atlas once for sequential matched arms A then B", flush=True)
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cuda", dtype=torch.float32)
assert runtime.cell_count == 98304 and runtime.representation_count == 2
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
assert np.array_equal(affine, np.array([[[1., 0., 0.], [0., 1., 0.]], [[-1., 0., 0.], [0., 1., 0.]]]))
psf_weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device="cuda", dtype=torch.float32) / 16
psf_positions = torch.linspace(-.5, .5, 9, device="cuda")
yy, xx = torch.meshgrid(torch.linspace(-1, 1, 96, device="cuda"), torch.linspace(-1, 1, 96, device="cuda"), indexing="ij")
noise_generator = torch.Generator(device="cuda")
origin, spacing = [catalogue["support_geometry"][key] for key in ("origin_ap_dv_ml_um", "voxel_size_ap_dv_ml_um")]
support_origin = torch.as_tensor(catalogue["support_geometry"]["support_origin_ap_dv_ml_um"], dtype=torch.float64)
dev_center, dev_frame, _ = full_frame_state_to_components(dev["truth_state"])
raw_center = torch.tensor([row["truth_center_ap_dv_ml_um"] for row in raw_records], dtype=torch.float64)
raw_normal = torch.tensor([row["truth_normal_ap_dv_ml"] for row in raw_records], dtype=torch.float64)
raw_channels = torch.cat((torch.from_numpy(raw_images), torch.zeros(64, 2, 96, 96)), dim=1)


def encode_keys(images):
    return model.pose_model.descriptor_from_features(model.pose_model._encode_atlas(images))


results = {}
for arm in ("A", "B"):
    output = RUN / arm
    output.mkdir()
    model = ArbitraryPlaneJointModelV6(runtime, **previous["model_kwargs"]).cuda()
    model.load_state_dict(parent["model_state"], strict=True)
    assert all(torch.equal(value.cpu(), parent["model_state"][name]) for name, value in model.state_dict().items())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=1e-4)
    optimizer.load_state_dict(copy.deepcopy(parent["optimizer_state"]))
    loaded = optimizer.state_dict()
    assert loaded["param_groups"] == parent["optimizer_state"]["param_groups"] and loaded["state"].keys() == parent["optimizer_state"]["state"].keys()
    for key, state in parent["optimizer_state"]["state"].items():
        assert all(torch.equal(loaded["state"][key][name].cpu(), value) if torch.is_tensor(value) else loaded["state"][key][name] == value for name, value in state.items())
    del loaded
    random.setstate(parent["python_rng"])
    np.random.set_state(parent["numpy_rng"])
    torch.set_rng_state(parent["torch_rng"])
    torch.cuda.set_rng_state_all(parent["cuda_rng"])
    (output / "initialization.json").write_text(json.dumps({"parent_sha256": PARENT_SHA256, "model_tensors_exact": True, "optimizer_state_exact": True, "restored_parent_rng": True, "step": START, "initial_gallery_rebuilt": False}), encoding="utf-8")
    applied = 0
    with (output / "training_trace.jsonl").open("w", encoding="utf-8") as trace:
        for step in range(1, STEPS + 1):
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
                assert np.array_equal(inputs[:, 2, 0, 0].bool().cpu().numpy(), available_schedule[step - 1])
                dropped = torch.as_tensor(dropout_mask[step - 1] if arm == "B" else np.zeros(BATCH, dtype=bool), device="cuda")
                if step == 1:
                    before_dropout = inputs.clone()
                inputs[dropped, 1:] = 0.
                if step == 1:
                    assert torch.equal(before_dropout[:, :1], inputs[:, :1])
                    torch.save({"input_channels_before": before_dropout.cpu(), "input_channels_after": inputs.cpu(), "outline_dropped": dropped.cpu(), "labels": label.cpu(), "weights": weight.cpu(), "global_generated_indices": positions + START * GENERATED, "frozen_row_indices": index}, output / "first_batch.pt")
                    del before_dropout
                distance2 = (key_center[label, None] - key_center[None]).square().sum(-1)
                same_u = (key_u[label, None] - key_u[None]).square().sum(-1)
                flip_u = (key_u[label, None] + key_u[None]).square().sum(-1)
                distance2 += .25 * (torch.minimum(same_u, flip_u) + (key_v[label, None] - key_v[None]).square().sum(-1))
                ignore = ((key_normal[label] @ key_normal.T).abs() >= np.cos(np.deg2rad(10.))) & (distance2 <= 1000.**2)
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
            assert bool(torch.isfinite(loss)) and weight.sum() > 0
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
            optimizer.step()
            applied += 1
            original_modes = [["raw", "exact_black", "imperfect_brush"][int(value)] for value in schedule["mode_index"][positions]] + [train["records"][int(value)]["selected_mode"] for value in index]
            record = {"arm": arm, "step": START + step, "additional_step": step, "optimizer_steps_applied": START + applied, "sampled_nll": float(loss.detach()), "gradient_norm": float(gradient), "candidate_ids": candidate.cpu().tolist(), "local_negative_ids": local_ids.cpu().tolist(), "ignored_candidate_counts": (~keep).sum(-1).cpu().tolist(), "generated_indices": (positions + START * GENERATED).tolist(), "frozen_row_indices": index.tolist(), "original_mode": original_modes, "original_outline_available": available_schedule[step - 1].tolist(), "outline_dropped": dropped.cpu().tolist(), "finite_support_mass_px": support_mass.cpu().tolist(), "visible_support_mass_px": visible_mass.cpu().tolist(), "generated_point_pose_weight": generated_weight.cpu().tolist(), "elapsed_seconds": time.perf_counter() - started}
            trace.write(json.dumps(record) + "\n")
            if step % 100 == 0:
                trace.flush()
                print(json.dumps({key: record[key] for key in ("arm", "step", "sampled_nll", "gradient_norm", "optimizer_steps_applied", "elapsed_seconds")}), flush=True)
            del keys, query, scores, losses, loss, inputs, key_images, rendered, generated_inputs
    assert applied == STEPS and all(int(state["step"]) == START + STEPS for state in optimizer.state_dict()["state"].values())
    assert all(torch.equal(value.cpu(), parent["model_state"][name]) for name, value in model.state_dict().items() if not name.startswith(TRAINABLE))
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    model.eval()
    checkpoint_path = output / "joint_model_step_06000.pt"
    torch.save({"experiment": config, "arm": arm, "phase": "experimental_image_key_proposal_only", "step": START + STEPS, "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()}, "optimizer_state": optimizer.state_dict(), "optimizer_steps_applied": START + applied, "additional_optimizer_steps_applied": applied, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "probabilities_calibrated": False}, checkpoint_path)
    checkpoint_sha256 = sha(checkpoint_path)
    bank = torch.empty((runtime.cell_count, 2, DESCRIPTOR_DIM), device="cuda", dtype=torch.float16)
    with torch.no_grad():
        for start in range(0, runtime.cell_count, 64):
            images = render_finite_thickness_plane(atlas, render_states[start:start + 64], SHAPE, origin, spacing, psf_positions * GALLERY_THICKNESS_UM, psf_weights)
            images = torch.stack((images, images.flip(-1)), dim=1).flatten(0, 1)
            bank[start:start + 64] = torch.cat([encode_keys(part) for part in images.split(KEY_CHUNK)]).reshape(-1, 2, DESCRIPTOR_DIM).half()
            if (start + 64) % 16384 == 0:
                print(json.dumps({"arm": arm, "gallery_cells": start + 64}), flush=True)
    bank_path = output / "gallery_descriptors_step_06000.npy"
    np.save(bank_path, bank.cpu().numpy())
    bank_sha256 = sha(bank_path)
    arm_results = {}
    for dataset, channels, records, truth_center, truth_normal in (("synthetic", dev["channels"], dev["records"], dev_center, dev_frame[:, :, 2]), ("allen_raw", raw_channels, raw_records, raw_center, raw_normal)):
        directory = output / dataset
        directory.mkdir()
        raw = np.lib.format.open_memmap(directory / "raw_cell_log_probability.npy", mode="w+", dtype=np.float32, shape=(len(records), runtime.cell_count))
        components = np.lib.format.open_memmap(directory / "raw_component_log_score.npy", mode="w+", dtype=np.float32, shape=(len(records), runtime.cell_count, 2))
        query_bank, tops, top_rep = [], [], []
        with torch.no_grad():
            for start in range(0, len(records), BATCH):
                inputs = channels[start:start + BATCH].cuda()
                query = model.pose_model.descriptor_from_features(model.pose_model.encode_histology(inputs[:, :1], inputs[:, 1:2], inputs[:, 2].mean((-2, -1))))
                representation_score = model.pose_model.image_key_cosine_logits(query, bank, TEMPERATURE) + representation_log_weight[None]
                score = torch.logsumexp(representation_score, dim=-1) + cell_log_mass[None]
                log_probability = score.log_softmax(dim=-1)
                assert bool(torch.isfinite(representation_score).all() and torch.isfinite(log_probability).all())
                raw[start:start + len(inputs)] = log_probability.cpu().numpy()
                components[start:start + len(inputs)] = (representation_score + cell_log_mass[None, :, None]).cpu().numpy()
                top = torch.argsort(log_probability, dim=-1, descending=True, stable=True)[:, :128]
                query_bank.append(query.cpu().numpy()); tops.append(top.cpu())
                top_rep.append(representation_score[torch.arange(len(inputs), device="cuda")[:, None], top].softmax(-1).cpu().numpy())
        raw.flush(); components.flush()
        normalization_error = max(float(np.abs(np.logaddexp.reduce(np.asarray(raw[start:start + BATCH], dtype=np.float64), axis=1)).max()) for start in range(0, len(records), BATCH))
        assert normalization_error < 2e-5
        top = torch.cat(tops)
        normals = cell_normal[top]
        dot = (normals * truth_normal[:, None]).sum(-1)
        angles = torch.rad2deg(torch.atan2(torch.linalg.cross(normals, truth_normal[:, None]).norm(dim=-1), dot.abs()))
        offsets = (((cell_center[top] - support_origin) * normals).sum(-1) - torch.where(dot < 0, -1., 1.) * ((truth_center - support_origin) * truth_normal).sum(-1)[:, None]).abs()
        metrics = {"plane_angle_deg": angles[:, 0].numpy(), "normal_offset_error_um": offsets[:, 0].numpy(), **{f"physical_plane_capture_at_{k}": ((angles[:, :k] <= 10.) & (offsets[:, :k] <= 500.)).any(-1).double().numpy() for k in (32, 128)}}
        group = np.array([row["animal_id"] for row in records])
        subsets = {"all": np.ones(len(records), dtype=bool)}
        if dataset == "synthetic":
            labels = dev["label"].numpy()
            truth_lp = np.asarray(raw[np.arange(len(labels)), labels])
            rank = (raw > truth_lp[:, None]).sum(-1) + ((raw == truth_lp[:, None]) & (np.arange(runtime.cell_count)[None] < labels[:, None])).sum(-1) + 1
            normal_lp = np.logaddexp.reduce(np.asarray(raw).reshape(len(labels), 384, 256), axis=-1)
            marginal_normal = cell_normal[normal_lp.argmax(-1) * 256]
            metrics.update({"nll": -truth_lp, "truth_rank": rank, "normal_marginal_nll": -normal_lp[np.arange(len(labels)), labels // 256], "normal_marginal_map_angle_deg": torch.rad2deg(torch.atan2(torch.linalg.cross(marginal_normal, truth_normal).norm(dim=-1), (marginal_normal * truth_normal).sum(-1).abs())).numpy(), **{f"hit_at_{k}": (rank <= k).astype(float) for k in (1, 8, 32, 128)}})
            eligible = dev["weight"].numpy() > 0
            modes = np.array([row["selected_mode"] for row in records])
            subsets.update({"eligible": eligible, "censored": ~eligible, **{f"eligible_mode:{mode}": eligible & (modes == mode) for mode in np.unique(modes)}})
        summaries = {}
        for name, selected in subsets.items():
            by_group = {str(identity): {"rows": int((selected & (group == identity)).sum()), **{key: float(values[selected & (group == identity)].mean()) for key, values in metrics.items()}} for identity in np.unique(group[selected])}
            summaries[name] = {"rows": int(selected.sum()), "groups": len(by_group), "by_group": by_group, "group_macro": {key: float(np.mean([row[key] for row in by_group.values()])) for key in metrics} if by_group else None}
        row_arrays = {key: np.array([row[key] for row in records]) for key in ("animal_id", "specimen_id", "experiment_id", "section_id")}
        if dataset == "synthetic":
            row_arrays.update(label=labels, pose_supervision_weight=dev["weight"].numpy(), selected_mode=modes)
        np.savez(directory / "rows.npz", **row_arrays, top128_cell_index=top.numpy(), query_descriptor=np.concatenate(query_bank), top128_representation_probability=np.concatenate(top_rep), reference_center_ap_dv_ml_um=truth_center.numpy(), reference_normal_ap_dv_ml=truth_normal.numpy(), **metrics)
        del raw, components
        receipt = {"arm": arm, "step": START + STEPS, "checkpoint_sha256": checkpoint_sha256, "gallery_sha256": bank_sha256, "subsets": summaries, "probabilities_calibrated": False, "raw_log_normalization_error": normalization_error, "output_sha256": {name: sha(directory / name) for name in ("raw_cell_log_probability.npy", "raw_component_log_score.npy", "rows.npz")}}
        if dataset == "allen_raw":
            receipt["donor_macro"] = summaries["all"]["group_macro"]
        (directory / "summary.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        arm_results[dataset] = receipt
        print(json.dumps({"arm": arm, "dataset": dataset, "macro": summaries["eligible" if dataset == "synthetic" else "all"]["group_macro"]}), flush=True)
    results[arm] = arm_results
    (output / "completed.json").write_text(json.dumps({"arm": arm, "additional_optimizer_steps_applied": applied, "optimizer_steps_applied": START + applied, "frozen_nonretrieval_tensors_exact": True, "endpoints": arm_results}, indent=2), encoding="utf-8")
    del model, optimizer, trainable, bank, query, images, inputs, representation_score, score, log_probability
    torch.cuda.empty_cache()

parent_synthetic = json.loads((PARENT / "development_metrics_step_04000.json").read_text())["subsets"]
a_real, b_real = [results[arm]["allen_raw"]["donor_macro"] for arm in ("A", "B")]
gates = {"B_real_normal_improvement_at_least_5deg": b_real["plane_angle_deg"] <= a_real["plane_angle_deg"] - 5., "B_real_top32_plane_improvement_at_least_point10": b_real["physical_plane_capture_at_32"] >= a_real["physical_plane_capture_at_32"] + .10}
for reference_name, reference in (("A6000", results["A"]["synthetic"]["subsets"]), ("parent4000", parent_synthetic)):
    for name, candidate in results["B"]["synthetic"]["subsets"].items():
        if name == "eligible" or name.startswith("eligible_mode:"):
            assert candidate["rows"] == reference[name]["rows"] and candidate["groups"] == reference[name]["groups"]
            if candidate["rows"]:
                b, a = candidate["group_macro"], reference[name]["group_macro"]
                gates[f"synthetic_normal_retention:{reference_name}:{name}"] = b["plane_angle_deg"] <= a["plane_angle_deg"] + 2.
                gates[f"synthetic_top32_retention:{reference_name}:{name}"] = b["physical_plane_capture_at_32"] >= a["physical_plane_capture_at_32"] - .02
for name, expected in source["file_sha256"].items():
    assert sha(repository / name) == expected
completion = {"experiment": config, "source_unchanged": True, "integrity_passed": True, "additional_applied_updates_each_arm": STEPS, "final_cumulative_updates_each_arm": START + STEPS, "gates": gates, "outline_dropout_gate_passed": bool(all(gates.values())), "results": results, "elapsed_seconds": time.perf_counter() - started}
(RUN / "completed.json").write_text(json.dumps(completion, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"gates": gates, "outline_dropout_gate_passed": completion["outline_dropout_gate_passed"]}), flush=True)
