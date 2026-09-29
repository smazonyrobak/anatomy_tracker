"""Prepared TRAIN-only oracle cost-direction readout; no learning or qualification."""
from pathlib import Path
import os
import sys

RUN_AFTER_FAILED_NATIVE_AUDIT = True
FAILED_NATIVE_AUDIT = Path("I:/AnatomyTracker/runs/joint_v6_signed_pose_evidence_001_independent_audit/audit.json")
FAILED_NATIVE_AUDIT_SHA256 = "7acdc3ef87f85691b296f67fc828428aa090b1afed88a5ce05853a6e81853ad5"
assert RUN_AFTER_FAILED_NATIVE_AUDIT and FAILED_NATIVE_AUDIT is not None
assert len(FAILED_NATIVE_AUDIT_SHA256) == 64

ROOT = Path("I:/AnatomyTracker")
RUN = ROOT / "runs/joint_v6_train_pose_cost_direction_001"
DATA = ROOT / "data/joint_v6_coherent_subject_cohort_sections_002"
ARCHIVE = ROOT / "runs/joint_v6_ribbon_local_001/source"
PARENT = ROOT / "runs/joint_v6_imagekey_retrieval_001/joint_model_step_04000.pt"
PARENT_SHA = "d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5"
COHORT_SHA = "ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2"
ARCHIVE_CONFIG_SHA = "b0aa101b42b70a5530daac0c3120946bcec4cdb564ef4833482ce1ed45d549c5"
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ARCHIVE))  # Completed code, never the operative model sources.

import hashlib
import json
import shutil
import numpy as np
import torch
import torch.nn.functional as F

assert hashlib.sha256(Path(FAILED_NATIVE_AUDIT).read_bytes()).hexdigest() == FAILED_NATIVE_AUDIT_SHA256
failed_audit = json.loads(Path(FAILED_NATIVE_AUDIT).read_text())
assert failed_audit["integrity_passed"] and not failed_audit["conditional_local_gate_passed"]
archive_config_bytes = (ARCHIVE.parent / "experiment.json").read_bytes()
assert hashlib.sha256(archive_config_bytes).hexdigest() == ARCHIVE_CONFIG_SHA
archive_config = json.loads(archive_config_bytes)
for name, expected in archive_config["source"]["file_sha256"].items():
    assert hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == expected

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components, full_frame_state_to_components, full_frame_state_to_physical_ouv, render_finite_thickness_coordinate_grid
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6
from training.arbitrary_plane_recurrent_model import compose_antipodal_plane_frame_residual
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact

with PARENT.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == PARENT_SHA
completion_bytes = (DATA / "completed.json").read_bytes()
assert hashlib.sha256(completion_bytes).hexdigest() == COHORT_SHA
completed = json.loads(completion_bytes)
parent = torch.load(PARENT, map_location="cpu", weights_only=False, mmap=True)
assert parent["step"] == 4000 and parent["phase"] == "experimental_image_key_proposal_only"
catalogue_path = PARENT.parent / "catalogue.pt"
with catalogue_path.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == parent["experiment"]["prepared_source_sha256"]["catalogue.pt"]
catalogue = torch.load(catalogue_path, map_location="cpu", weights_only=False)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cuda", dtype=torch.float32)
model = ArbitraryPlaneJointModelV6(runtime, **parent["experiment"]["model_kwargs"]).cuda().eval()
model.load_state_dict(parent["model_state"], strict=True)
model.requires_grad_(False)
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
origin, spacing, support_origin = [catalogue["support_geometry"][key] for key in ("origin_ap_dv_ml_um", "voxel_size_ap_dv_ml_um", "support_origin_ap_dv_ml_um")]
train_subjects = sorted({record["lineage"]["subject_id"] for record in completed["sections"] if record["lineage"]["split"] == "train"})
assert len(train_subjects) == 8
RUN.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, RUN / Path(__file__).name)
shutil.copyfile(FAILED_NATIVE_AUDIT, RUN / "failed_native_audit.json")
yy, xx = torch.meshgrid(torch.arange(96, device="cuda", dtype=torch.float64) / 96, torch.arange(96, device="cuda", dtype=torch.float64) / 96, indexing="ij")
axes, start_sizes, probe_sizes = ("normal_u", "normal_v", "normal_offset"), (np.deg2rad(6), np.deg2rad(6), 300.), (np.deg2rad(3), np.deg2rad(3), 150.)
selected, selection_trace, measurements, artifacts = {}, [], [], {}

with torch.inference_mode():
    for record in completed["sections"]:
        lineage = record["lineage"]
        if lineage["split"] != "train" or lineage["subject_id"] in selected:
            continue
        for name, expected in record["artifact_sha256"].items():
            with (DATA / name).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
        section = _read_raw_artifact(DATA, record["artifacts"])
        flags = [bool(item["support_information_eligible"]) for item in section["observations"]]
        selection_trace.append({"lineage": lineage, "artifact_sha256": record["artifact_sha256"], "eligibility": flags})
        if not any(flags):
            continue
        selected[lineage["subject_id"]] = lineage["section_id"]
        assert not section["reflection_xy"][1] and len(flags) == 3
        reflection = bool(section["reflection_xy"][0])
        fit = section["canonical_anatomy_plane_fit"]["arrays"]
        state = full_frame_state_from_components(*physical_ouv_to_frame(torch.from_numpy(fit["physical_ouv_ap_dv_ml_um_float64"]).cuda()))[None]
        _, frame, _ = full_frame_state_to_components(state)
        canonical_centre = torch.from_numpy(section["canonical_anatomy_centre_ccf_ap_dv_ml_um_float64"]).cuda()
        plane = torch.from_numpy(fit["fitted_coordinate_raster_ap_dv_ml_um_float64"]).cuda()
        target_centre = torch.from_numpy(section["target_centre_ccf_coordinates_ap_dv_ml_um_float64"]).cuda()
        target_slab = torch.from_numpy(section["target_psf_ccf_coordinates_ap_dv_ml_um_float64"]).cuda()
        z, w = [torch.from_numpy(section[key]).cuda() for key in ("axial_offsets_um_float64", "axial_weights_float64")]
        w = w / w.sum()
        director = ((w * z)[:, None, None, None] * (target_slab - target_centre)).sum(0) / (w * z.square()).sum()
        if reflection:
            director = director.flip(1)  # Undo raster reflection; never flip physical components.
        residual, removed_affine = project_surface_affine_out(torch.einsum("ij,hwi->jhw", frame[0], canonical_centre - plane)[None])
        delta = torch.einsum("ij,hwi->jhw", frame[0], director - frame[0, :, 2])[None]
        states, state_labels, triplets = [state[0]], ["truth"], []
        for axis in range(3):
            for sign in (-1, 1):
                update = state.new_zeros((1, 9))
                update[0, axis] = sign * start_sizes[axis]
                current = compose_antipodal_plane_frame_residual(state, update, support_origin)
                indices = []
                for probe_sign in (0, -1, 1):
                    update.zero_()
                    update[0, axis] = probe_sign * probe_sizes[axis]
                    indices.append(len(states))
                    states.append(compose_antipodal_plane_frame_residual(current, update, support_origin)[0])
                    state_labels.append(f"{axes[axis]}_{sign:+d}_probe_{probe_sign:+d}")
                triplets.append((axis, sign, indices))
        states = torch.stack(states)
        channels = torch.stack([torch.from_numpy(item["image_outline_availability_float32"]) for item in section["observations"]]).cuda()
        q = torch.stack([torch.from_numpy(item["visible_finite_support_float32"]) for item in section["observations"]]).cuda().double()
        source_features = F.normalize(model.pose_model.encode_histology(channels[:, :1], channels[:, 1:2], channels[:, 2, 0, 0].bool()), dim=1, eps=1e-6)
        q_feature = F.adaptive_avg_pool2d(q[:, None].float(), source_features.shape[-2:])[:, 0]
        cost_maps, centre_errors, slab_errors = [], [], []
        for start in range(0, len(states), 4):
            batch_state = states[start:start + 4]
            _, rotation, _ = full_frame_state_to_components(batch_state)
            ouv = full_frame_state_to_physical_ouv(batch_state)
            base = ouv[:, None, None, :3] + xx[None, :, :, None] * ouv[:, None, None, 3:6] + yy[None, :, :, None] * ouv[:, None, None, 6:9]
            centre = base + torch.einsum("bij,bjhw->bhwi", rotation, residual.expand(len(batch_state), -1, -1, -1))
            direction = rotation[:, None, None, :, 2] + torch.einsum("bij,bjhw->bhwi", rotation, delta.expand(len(batch_state), -1, -1, -1))
            slab = centre[:, None] + z[None, :, None, None, None] * direction[:, None]
            if reflection:
                centre, slab = centre.flip(-2), slab.flip(-2)
            rendered = render_finite_thickness_coordinate_grid(atlas, slab, origin, spacing, w)
            atlas_features = F.normalize(model.pose_model._encode_atlas(rendered), dim=1, eps=1e-6)
            cost_maps.append(1 - torch.einsum("mchw,bchw->mbhw", source_features, atlas_features))
            centre_errors.append((centre - target_centre).norm(dim=-1))
            slab_errors.append(((slab - target_slab).norm(dim=-1) * w[None, :, None, None]).sum(1))
        costs = torch.cat(cost_maps, dim=1)
        centre_error, slab_error = torch.cat(centre_errors), torch.cat(slab_errors)
        cost_means = {"full": costs.mean((-2, -1)), "oracle_visible_support": (costs * q_feature[:, None]).sum((-2, -1)) / q_feature.sum((-2, -1))[:, None]}
        geometry = {}
        for name, error in (("centre", centre_error), ("slab", slab_error)):
            geometry[(name, "full")] = error.mean((-2, -1))[None].expand(3, -1)
            geometry[(name, "oracle_visible_support")] = (error[None] * q[:, None]).sum((-2, -1)) / q.sum((-2, -1))[:, None]
        raw_name = f"section_{len(selected):02d}.npz"
        np.savez_compressed(RUN / raw_name, states=states.cpu().numpy(), canonical_residual_local_um=residual.cpu().numpy(), canonical_director_delta_local=delta.cpu().numpy(), removed_affine=removed_affine.cpu().numpy(), offsets_um=z.cpu().numpy(), psf_weights=w.cpu().numpy(), feature_cost_maps=costs.cpu().numpy(), feature_support_weight=q_feature.cpu().numpy(), centre_distance_um=centre_error.cpu().numpy(), psf_mean_slab_distance_um=slab_error.cpu().numpy(), visible_support=q.cpu().numpy(), reflection=reflection, state_labels=np.array(state_labels))
        artifacts[raw_name] = hashlib.sha256((RUN / raw_name).read_bytes()).hexdigest()
        for mi, item in enumerate(section["observations"]):
            for metric, values in cost_means.items():
                for axis, sign, (current, minus, plus) in triplets:
                    c = values[mi, [current, minus, plus]].cpu().numpy()
                    g = geometry[("slab", metric)][mi, [current, minus, plus]].cpu().numpy()
                    finite = bool(np.isfinite(c).all() and np.isfinite(g).all())
                    cost_sign, geometry_sign = (int(np.sign(c[1] - c[2])), int(np.sign(g[1] - g[2]))) if finite else (None, None)
                    measurements.append({**item["lineage"], "mode": item["selected_mode"], "eligible": flags[mi], "weighting": metric, "axis": axes[axis], "start_sign": sign, "state_indices_current_minus_plus": [current, minus, plus], "cost_current_minus_plus": c.tolist(), "cost_truth": float(values[mi, 0]), "cost_plus_minus_current": [float(c[2] - c[0]), float(c[1] - c[0])], "slab_mean_um_current_minus_plus": g.tolist(), "centre_mean_um_current_minus_plus": geometry[("centre", metric)][mi, [current, minus, plus]].cpu().tolist(), "slab_truth_mean_um": float(geometry[("slab", metric)][mi, 0]), "cost_preferred_probe_sign": cost_sign, "geometry_preferred_probe_sign": geometry_sign, "agreement": finite and cost_sign != 0 and geometry_sign != 0 and cost_sign == geometry_sign, "cost_preferred_geometry_descent": finite and cost_sign != 0 and float(g[2 if cost_sign == 1 else 1]) < float(g[0]), "finite": finite, "raw_artifact": raw_name})
        print(json.dumps({"subject_id": lineage["subject_id"], "section_id": lineage["section_id"], "eligible_modes": flags, "sections_completed": len(selected)}), flush=True)
assert set(selected) == set(train_subjects)
groups = []
for field in ("all", "mode", "axis", "subject_id"):
    for value in (["all"] if field == "all" else sorted({row[field] for row in measurements})):
        for weighting in ("full", "oracle_visible_support"):
            rows = [row for row in measurements if row["weighting"] == weighting and (field == "all" or row[field] == value)]
            for scope in ("all", "eligible", "censored"):
                chosen = [row for row in rows if scope == "all" or row["eligible"] == (scope == "eligible")]
                decisive = [row for row in chosen if row["finite"] and row["cost_preferred_probe_sign"] != 0 and row["geometry_preferred_probe_sign"] != 0]
                groups.append({"group": field, "value": value, "weighting": weighting, "scope": scope, "rows": len(chosen), "finite": sum(row["finite"] for row in chosen), "decisive": len(decisive), "agree": sum(row["agreement"] for row in decisive), "cost_preferred_geometry_descent": sum(row["cost_preferred_geometry_descent"] for row in decisive)})
(RUN / "measurements.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
(RUN / "selection.json").write_text(json.dumps(selection_trace, indent=2), encoding="utf-8")
for name in ("measurements.json", "selection.json", Path(__file__).name):
    artifacts[name] = hashlib.sha256((RUN / name).read_bytes()).hexdigest()
receipt = {"parent_checkpoint_sha256": PARENT_SHA, "cohort_completion_sha256": COHORT_SHA, "completed_archive_config_sha256": ARCHIVE_CONFIG_SHA, "archived_sources": archive_config["source"], "failed_native_audit_sha256": FAILED_NATIVE_AUDIT_SHA256, "selected_train_sections": selected, "rendered_pose_states": 8 * 19, "mode_observations": 24, "direction_comparisons_per_weighting": 24 * 6, "precision": "FP64 physical geometry; FP32 renderer/encoder, no AMP/TF32", "fields": "exact canonical target residual projected once; anchored PSF director; fixed local fields across each pose/probe; no probe-specific limiter", "scope": "TRAIN-only oracle feature-cost direction diagnostic; not learned updates, optimization, calibration, global capture, biological validation, or a qualification gate", "groups": groups, "artifacts_sha256": artifacts}
(RUN / "completed.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
