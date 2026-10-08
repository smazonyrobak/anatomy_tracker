"""One-pass physical-sagittal/coronal/full-sphere pose continuation of joint 019."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

root = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(root / "tmp")
os.environ["TORCH_HOME"] = str(root / "cache/torch")
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components, full_frame_state_to_components
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64, sample_streaming_synthetic_v7_64


sagittal = root / "data/allen_sagittal_ish_expansion_002_train_inputs_20261008"
catalogue_path = root / "data/allen_sagittal_ish_expansion_002_20261008/manifest.json"
parent = root / "runs/one_shot_exposure_019/joint_step_18000.pt"
run = root / "runs/sagittal_mixed_pose_continuation_001"
protocol = Path(__file__).resolve().parents[1] / "docs/publication/SAGITTAL_MIXED_POSE_CONTINUATION_001_PROTOCOL_20261008.md"
seed, updates, side, synthetic_count = 2026100801, 653, 256, 2
gain = 4.259364821544654
assert hashlib.sha256(parent.read_bytes()).hexdigest() == "70edfaa53fcd84a2d20be5a246948e17a7eeb0406e054cf65535e381f5895ab8"
summary_bytes = (sagittal / "summary.json").read_bytes()
summary = json.loads(summary_bytes)
assert hashlib.sha256((sagittal / "model_input.npy").read_bytes()).hexdigest() == summary["output_sha256"]["model_input.npy"]
assert hashlib.sha256((sagittal / "geometry.jsonl").read_bytes()).hexdigest() == summary["output_sha256"]["geometry.jsonl"]
records = [json.loads(line) for line in (sagittal / "geometry.jsonl").read_text().splitlines()]
assert len(records) == updates and len({row["section_id"] for row in records}) == updates
assert len({row["donor_id"] for row in records}) == 40 and all(row["split"] == "train" for row in records)
assert all(row["fixed_gain"] == gain for row in records)
assert [row["array_row_index"] for row in records] == list(range(updates))
catalogue_bytes = catalogue_path.read_bytes()
assert hashlib.sha256(catalogue_bytes).hexdigest() == "b4f77e00325273f937e4327b52d8b3f7db95458c130d4f88f682a8a08323f7bd"
catalogue = json.loads(catalogue_bytes)
weak_dev2 = {10422, 10405, 10355, 10347, 10430, 10248, 10443, 10354}
sealed_holdout = {10410, 10223, 10302, 10376, 10392, 10305, 10173, 10351}
expected_train = {(row["donor_id"], row["specimen_id"], row["experiment_id"], row["section_id"])
                  for row in catalogue["sections"] if row["donor_id"] not in weak_dev2 | sealed_holdout}
actual_train = {(row["donor_id"], row["specimen_id"], row["experiment_id"], row["section_id"])
                for row in records}
assert len(expected_train) == len(actual_train) == updates and expected_train == actual_train
assert summary["source_metadata_manifest_sha256"] == hashlib.sha256(catalogue_bytes).hexdigest()
sagittal_images = np.load(sagittal / "model_input.npy", mmap_mode="r")
assert sagittal_images.shape == (updates, 1, side, side)
affines = torch.tensor(np.asarray([row["model_pixel_to_ccf_ref9_ap_dv_ml_um"] for row in records]), dtype=torch.float64)
ouv = torch.stack((affines[:, :, 2], side * affines[:, :, 0], side * affines[:, :, 1]), 1)
states = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
normal = full_frame_state_to_components(states)[1][..., :, 2]
sagittal_reflection = normal.gather(1, normal.abs().argmax(-1)[:, None])[:, 0] < 0
ouv[sagittal_reflection, 0] += (side - 1) / side * ouv[sagittal_reflection, 1]
ouv[sagittal_reflection, 1] *= -1
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device="cuda")
coronal = load_reserved_real_train()
prior_schedules = sorted([*root.glob("runs/*/real_schedule.npy"), *root.glob("runs/*/new_real_schedule.npy")])
prior_real = {tuple(map(int, pair)) for path in prior_schedules for pair in np.load(path)}
rng = np.random.default_rng(seed)
sagittal_schedule = rng.permutation(updates).astype(np.int32)
coronal_schedule = []
for donor_index in rng.permutation(len(coronal["donors"])):
    available = [section for section in range(len(coronal["donors"][donor_index]["identities"]))
                 if (int(donor_index), section) not in prior_real]
    if available:
        coronal_schedule.append((int(donor_index), int(rng.choice(available))))
    if len(coronal_schedule) == updates:
        break
assert len(coronal_schedule) == updates and len({donor for donor, _ in coronal_schedule}) == updates
assert not set(coronal_schedule) & prior_real
run.mkdir(parents=True, exist_ok=False)
np.save(run / "sagittal_schedule.npy", sagittal_schedule)
np.save(run / "coronal_schedule.npy", np.asarray(coronal_schedule, dtype=np.int32))
config = {
    "version": "sagittal-mixed-pose-continuation-001", "seed": seed, "updates": updates,
    "batch": {"independent_synthetic": synthetic_count, "unique_sagittal_real": 1, "unique_coronal_real": 1},
    "parent": str(parent), "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
    "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    "sagittal_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
    "sagittal_source_sha256": summary["output_sha256"], "sagittal_fixed_gain": gain,
    "sagittal_frozen_catalogue_sha256": hashlib.sha256(catalogue_bytes).hexdigest(),
    "coronal_bindings": coronal["bindings"], "coronal_prior_schedule_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in prior_schedules},
    "synthetic_provenance": context["provenance"],
    "schedule_sha256": {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                        for name in ("sagittal_schedule.npy", "coronal_schedule.npy")},
    "trainable": ["image encoder", "direct pose head"], "frozen": ["atlas encoder", "warp mapper", "recurrent fitter", "fitted scorer"],
    "optimizer": {"name": "AdamW", "encoder_lr": 5e-6, "pose_lr": 1e-5, "weight_decay": 1e-4, "gradient_clip": 5},
    "real_label_role": "weak inherited Allen affine", "calibrated": False, "public_benchmark_used": False,
    "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in ("train_sagittal_mixed_pose_continuation_001.py",
                                   "arbitrary_plane_one_shot_model.py", "arbitrary_plane_full_frame_primitives.py",
                                   "arbitrary_plane_geometry.py", "arbitrary_plane_reserved_real_stream_v8.py",
                                   "arbitrary_plane_streaming_synthetic_v7.py", "arbitrary_plane_streaming_synthetic_v7_64.py")},
}
(run / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), (side - 1) / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum("...ij,...pj->...pi", frame[..., :2] @ basis, chart - .5)


torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
subjects_rng = torch.Generator().manual_seed(seed + 1)
draw_seed = seed + 2
checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
assert checkpoint["step"] == 18000 and not checkpoint["calibrated"]
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().train()
model.load_state_dict(checkpoint["model"], strict=True)
del checkpoint
model.requires_grad_(False)
model.encoder.requires_grad_(True)
model.pose.requires_grad_(True)
optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 5e-6},
                               {"params": model.pose.parameters(), "lr": 1e-5}], weight_decay=1e-4)
pixels = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.], [127.5, 127.5]], device="cuda") / side
flags = torch.tensor([0, 1], device="cuda")


def save(step):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "subjects_rng": subjects_rng.get_state(), "draw_seed": draw_seed,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "step": step, "config": config, "calibrated": False},
               run / f"joint_step_{step:04d}.pt")


save(0)
started = time.perf_counter()
with (run / "training.jsonl").open("w") as log, (run / "draws.jsonl").open("w") as draws:
    for step in range(1, updates + 1):
        accepted, pending = {}, list(range(synthetic_count))
        while pending:
            subjects = torch.randint(len(context["subjects"]), (len(pending),), generator=subjects_rng).tolist()
            sample = sample_streaming_synthetic_v7_64(context, subjects, draw_seed, side=side)
            draw_seed += 1
            for row, identity in enumerate(sample["provenance"]):
                slot = pending[row]
                used = bool(sample["eligible"][row])
                draws.write(json.dumps({**identity, "step": step, "slot": slot, "used": used}) + "\n")
                if used:
                    accepted[slot] = {key: sample[key][row:row + 1] for key in
                                      ("inputs", "state", "reflection", "centre", "valid_mask")}
            pending = [slot for slot in pending if slot not in accepted]
        synthetic = {key: torch.cat([accepted[slot][key] for slot in range(synthetic_count)])
                     for key in accepted[0]}
        donor_index, coronal_index = coronal_schedule[step - 1]
        acquired_coronal = sample_reserved_real_train(coronal, donor_index, [coronal_index], device="cuda")
        acquired_coronal["inputs"] = F.interpolate(acquired_coronal["inputs"], (side, side),
                                                     mode="bilinear", align_corners=False)
        sagittal_index = int(sagittal_schedule[step - 1])
        acquired_sagittal = torch.zeros(1, 5, side, side, device="cuda")
        acquired_sagittal[:, :1] = torch.from_numpy(np.asarray(sagittal_images[sagittal_index]).copy()).to("cuda")
        inputs = torch.cat((synthetic["inputs"], acquired_coronal["inputs"], acquired_sagittal))
        truth = torch.cat((synthetic["state"], acquired_coronal["state"],
                           sagittal_states[sagittal_index:sagittal_index + 1].to("cuda")))
        truth_reflection = torch.cat((synthetic["reflection"], acquired_coronal["reflection"],
                                      sagittal_reflection[sagittal_index:sagittal_index + 1].long().to("cuda")))
        draws.write(json.dumps({**acquired_coronal["identities"][0], "step": step,
                                "slot": synthetic_count, "role": "coronal_weak_train"}) + "\n")
        draws.write(json.dumps({**{key: records[sagittal_index][key] for key in
                                    ("donor_id", "specimen_id", "experiment_id", "section_id", "image_sha256")},
                                "step": step, "slot": synthetic_count + 1,
                                "role": "sagittal_weak_train"}) + "\n")
        prediction = model.predict(inputs)
        branches = prediction["state"][:, :, None].expand(-1, -1, 2, -1)
        branch_flags = flags[None, None].expand(len(inputs), model.modes, 2)
        reference = points(truth, truth_reflection, pixels)
        five = (points(branches, branch_flags, pixels) - reference[:, None, None]).norm(dim=-1).mean(-1)
        indices = torch.multinomial(synthetic["valid_mask"].flatten(1).float(), 128, replacement=True)
        target = synthetic["centre"].reshape(synthetic_count, -1, 3).gather(
            1, indices[..., None].expand(-1, -1, 3))
        chart = torch.stack((indices.remainder(side), indices.div(side, rounding_mode="floor")), -1).float() / side
        dense = (points(branches[:synthetic_count], branch_flags[:synthetic_count], chart[:, None, None])
                 - target[:, None, None]).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(prediction["state"])[1][..., :, 2]
        true_normal = full_frame_state_to_components(truth)[1][..., :, 2]
        normal_penalty = 4000 * (1 - (normal[:synthetic_count] * true_normal[:synthetic_count, None])
                                 .sum(-1).abs().clamp_max(1))
        distance = torch.cat((.75 * dense + .25 * five[:synthetic_count]
                              + normal_penalty[..., None], five[synthetic_count:]), 0).flatten(1)
        prior = (prediction["log_mass"][..., None] + torch.stack((
            F.logsigmoid(-prediction["reflection_logit"]),
            F.logsigmoid(prediction["reflection_logit"])), -1)).flatten(1)
        rank = F.kl_div(prior, (-distance.detach() / 1000).softmax(-1), reduction="none").sum(-1)
        direct = (distance.min(-1).values / 1000
                  - .75 * torch.logsumexp(prior - distance / 1500, -1) + .75 * rank)
        loss = (direct[:synthetic_count].sum() + .5 * direct[synthetic_count:].sum()) / (synthetic_count + 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                                   5., error_if_nonfinite=True)
        optimizer.step()
        log.write(json.dumps({"step": step, "loss": float(loss.detach()),
                              "synthetic_best_um": float(distance[:synthetic_count].min(-1).values.mean().detach()),
                              "coronal_best_um": float(distance[synthetic_count].min().detach()),
                              "sagittal_best_um": float(distance[synthetic_count + 1].min().detach()),
                              "gradient_norm": float(gradient), "elapsed_seconds": time.perf_counter() - started}) + "\n")
        if step in (326, 653):
            save(step)
        if step == 1 or step % 100 == 0 or step == updates:
            log.flush()
            draws.flush()
            print(json.dumps({"batch": step, "batches_total": updates,
                              "loss": float(loss.detach()), "elapsed_seconds": time.perf_counter() - started}), flush=True)

completed = {"updates": updates, "unique_sagittal_train_sections": len(set(sagittal_schedule.tolist())),
             "unique_coronal_train_donors": len({donor for donor, _ in coronal_schedule}),
             "synthetic_accepted": synthetic_count * updates,
             "config_sha256": hashlib.sha256((run / "config.json").read_bytes()).hexdigest(),
             "trace_sha256": hashlib.sha256((run / "training.jsonl").read_bytes()).hexdigest(),
             "draws_sha256": hashlib.sha256((run / "draws.jsonl").read_bytes()).hexdigest(),
             "checkpoint_sha256": {str(step): hashlib.sha256((run / f"joint_step_{step:04d}.pt").read_bytes()).hexdigest()
                                   for step in (0, 326, 653)},
             "calibrated": False, "public_benchmark_used": False}
(run / "completed.json").write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"completed_batches": updates, "output": str(run),
                  "checkpoint_sha256": completed["checkpoint_sha256"]}, indent=2), flush=True)
