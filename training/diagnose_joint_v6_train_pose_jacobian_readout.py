"""Guarded oracle-conditioned TRAIN readout discriminator; no training or qualification."""
import os
import sys
from pathlib import Path

READY_AFTER_NORMALIZED_ENDPOINT_AND_ROOT_REVIEW = False
assert READY_AFTER_NORMALIZED_ENDPOINT_AND_ROOT_REVIEW, "Draft: await normalized endpoint and explicit launch decision"
ROOT = Path("I:/AnatomyTracker")
SIGNED = ROOT / "runs/joint_v6_signed_pose_evidence_001"
DIAGNOSTIC = ROOT / "runs/joint_v6_train_pose_cost_direction_001"
DATA = ROOT / "data/joint_v6_coherent_subject_cohort_sections_002"
SUPPLEMENT = ROOT / "tmp/signed_activation_source_supplement"
OUT = ROOT / "runs/joint_v6_train_pose_jacobian_readout_001"
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True
sys.path.insert(0, str(SIGNED / "source"))
sys.path.insert(1, str(SUPPLEMENT))

import hashlib
import json
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F

pins = {
    SIGNED / "completed.json": "7e814be7b2207e54544b67f6180a61808de17ea6bc530b41930092ced90828c1",
    SIGNED / "joint_model_step_02000.pt": "555c129b2cbe5832b37c7642d5ee15e31c1ba57199952901959751918cae6136",
    SIGNED / "catalogue.pt": "9b49d203cc73ce3a66e648bbe5228231eb5cc9c17d5db4669eefe0f08ae22c71",
    ROOT / "runs/joint_v6_signed_pose_evidence_001_independent_audit/audit.json": "7acdc3ef87f85691b296f67fc828428aa090b1afed88a5ce05853a6e81853ad5",
    ROOT / "runs/joint_v6_signed_pose_activation_train_001/result.json": "5762fdcda9d5eed305835b6e77c9f262710600ef97aa286fd1a1e167f091558b",
    DIAGNOSTIC / "completed.json": "556fc13ee2f6dfc1db059c3ce8e5a3b00eca429b2de23c394be19e552cdc5ed5",
    DATA / "completed.json": "ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2",
    SUPPLEMENT / "training/arbitrary_plane_joint_uncertainty.py": "e1989b684a72fd9af4a552066cc4e7d5c33cf2cb293513e628e52946962139e3",
}
hashes = {}


def sha(path):
    with Path(path).open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    hashes[str(path)] = value
    return value


for path, expected in pins.items():
    assert sha(path) == expected
signed = json.loads((SIGNED / "completed.json").read_text())
source = signed["experiment"]["source"]
for name, expected in source["file_sha256"].items():
    assert sha(SIGNED / "source" / name) == expected
diagnostic = json.loads((DIAGNOSTIC / "completed.json").read_text())
for name, expected in diagnostic["artifacts_sha256"].items():
    assert sha(DIAGNOSTIC / name) == expected
cohort = json.loads((DATA / "completed.json").read_text())
measurements = json.loads((DIAGNOSTIC / "measurements.json").read_text())
selected = diagnostic["selected_train_sections"]
assert len(selected) == 8 and diagnostic["mode_observations"] == 24

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_catalogue_runtime_v6 import make_complete_catalogue_runtime_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components, full_frame_state_to_physical_ouv, render_finite_thickness_coordinate_grid
from training.arbitrary_plane_joint_model_v6 import ArbitraryPlaneJointModelV6
from training.arbitrary_plane_recurrent_model import _pair_evidence, compose_antipodal_plane_frame_residual
from training.subject_deformed_slab_multiresolution_bundle_v2 import _read_raw_artifact

loaded_source = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items() if name.startswith("training.") and getattr(module, "__file__", None)}
assert all(Path(path).is_relative_to(SIGNED / "source") or (name == "training.arbitrary_plane_joint_uncertainty" and Path(path) == (SUPPLEMENT / "training/arbitrary_plane_joint_uncertainty.py").resolve()) for name, path in loaded_source.items())
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
checkpoint = torch.load(SIGNED / "joint_model_step_02000.pt", map_location="cpu", weights_only=False, mmap=True)
assert checkpoint["step"] == checkpoint["optimizer_steps_applied"] == 2000 and checkpoint["phase"] == "conditional_native_pose_ribbon"
catalogue = torch.load(SIGNED / "catalogue.pt", map_location="cpu", weights_only=False)
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cuda", dtype=torch.float32)
model = ArbitraryPlaneJointModelV6(runtime, **checkpoint["experiment"]["model_kwargs"]).cuda().eval()
model.load_state_dict(checkpoint["model_state"], strict=True)
model.requires_grad_(False)
pose = model.pose_model
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation
origin, spacing, support_origin = [catalogue["support_geometry"][key] for key in ("origin_ap_dv_ml_um", "voxel_size_ap_dv_ml_um", "support_origin_ap_dv_ml_um")]
support_origin = torch.tensor(support_origin, device="cuda", dtype=torch.float64)
h = torch.tensor([np.pi / 60, np.pi / 60, 150.], device="cuda", dtype=torch.float64)
yy, xx = torch.meshgrid(torch.arange(96, device="cuda", dtype=torch.float64) / 96, torch.arange(96, device="cuda", dtype=torch.float64) / 96, indexing="ij")
mode_indices = torch.arange(3, device="cuda").repeat_interleave(6)
arms, modes = ("learned_capped", "damped_feature_solve"), ("raw", "exact_black", "imperfect_brush")
OUT.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, OUT / Path(__file__).name)
for name, path in loaded_source.items():
    destination = OUT / "source" / Path(*name.split(".")).with_suffix(".py")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
shutil.copyfile(DIAGNOSTIC / "completed.json", OUT / "parent_direction_completed.json")
rows, artifacts = [], {}
started = time.perf_counter()


def render(state):
    _, rotation, _ = full_frame_state_to_components(state)
    ouv = full_frame_state_to_physical_ouv(state)
    plane = ouv[:, None, None, :3] + xx[None, :, :, None] * ouv[:, None, None, 3:6] + yy[None, :, :, None] * ouv[:, None, None, 6:9]
    centre = plane + torch.einsum("bij,bjhw->bhwi", rotation, residual.expand(len(state), -1, -1, -1))
    director = rotation[:, None, None, :, 2] + torch.einsum("bij,bjhw->bhwi", rotation, director_delta.expand(len(state), -1, -1, -1))
    slab = centre[:, None] + z[None, :, None, None, None] * director[:, None]
    if reflection:
        centre, slab = centre.flip(-2), slab.flip(-2)
    image = render_finite_thickness_coordinate_grid(atlas, slab, origin, spacing, w)
    features = pose._encode_atlas(image)
    return features, F.normalize(features, dim=1, eps=1e-6), centre, slab


with torch.inference_mode():
    for si, (subject_id, section_id) in enumerate(selected.items()):
        record = next(item for item in cohort["sections"] if item["lineage"]["section_id"] == section_id)
        assert record["lineage"]["split"] == "train" and record["lineage"]["subject_id"] == subject_id
        for name, expected in record["artifact_sha256"].items():
            assert sha(DATA / name) == expected
        section = _read_raw_artifact(DATA, record["artifacts"])
        records = [item for item in measurements if item["section_id"] == section_id and item["weighting"] == "full"]
        raw_names = {item["raw_artifact"] for item in records}
        assert len(records) == 18 and len(raw_names) == 1
        raw = np.load(DIAGNOSTIC / raw_names.pop(), allow_pickle=False)
        starts = np.array([item["state_indices_current_minus_plus"][0] for item in records if item["mode"] == "raw"])
        assert starts.shape == (6,)
        initial = torch.from_numpy(raw["states"][starts]).cuda().repeat(3, 1)
        truth_state = torch.from_numpy(raw["states"][:1]).cuda()
        residual = torch.from_numpy(raw["canonical_residual_local_um"]).cuda()
        director_delta = torch.from_numpy(raw["canonical_director_delta_local"]).cuda()
        z, w = [torch.from_numpy(raw[key]).cuda() for key in ("offsets_um", "psf_weights")]
        reflection = bool(raw["reflection"])
        assert [item["selected_mode"] for item in section["observations"]] == list(modes)
        assert not section["reflection_xy"][1] and bool(section["reflection_xy"][0]) == reflection
        assert {item["lineage"]["observation_id"] for item in section["observations"]} == {item["observation_id"] for item in records}
        channels = torch.stack([torch.from_numpy(item["image_outline_availability_float32"]) for item in section["observations"]]).cuda()
        source_feature = pose.encode_histology(channels[:, :1], channels[:, 1:2], channels[:, 2].mean((-2, -1)))[mode_indices]
        source_unit = F.normalize(source_feature, dim=1, eps=1e-6)
        target_centre = torch.from_numpy(section["target_centre_ccf_coordinates_ap_dv_ml_um_float64"]).cuda()
        target_slab = torch.from_numpy(section["target_psf_ccf_coordinates_ap_dv_ml_um_float64"]).cuda()
        truth_c, truth_R, _ = full_frame_state_to_components(truth_state)
        truth_n = truth_R[0, :, 2]
        truth_d = ((truth_c[0] - support_origin) * truth_n).sum()
        _, truth_feature, _, _ = render(truth_state)
        truth_cost = (1 - (source_unit * truth_feature).sum(1)).mean((-2, -1))
        data = {key: [] for key in ("state", "cosine_cost", "half_squared_feature_cost", "centre_error_um", "slab_error_um", "normal_error_deg", "signed_offset_error_um", "atlas_epsilon_pixels", "H", "H_eigenvalues", "damped_condition", "gradient", "gn_q_pre_cap", "learned_update_pre_cap", "applied_update", "trust_cap", "zero_information", "nonfinite_solve")}
        for arm in arms:
            state = initial.clone()
            hidden = source_feature.new_zeros(18, pose.recurrent_cell.candidate.out_channels, *source_feature.shape[-2:])
            history = {key: [] for key in data}
            for step in range(4):
                feature, atlas_unit, centre, slab = render(state)
                r = atlas_unit - source_unit
                central_cost = 1 - (source_unit * atlas_unit).sum(1, keepdim=True)
                c, R, _ = full_frame_state_to_components(state)
                n = R[:, :, 2]
                angle = torch.rad2deg(torch.atan2(torch.linalg.cross(n, truth_n[None].expand_as(n)).norm(dim=-1), (n * truth_n).sum(-1).abs()))
                metrics = (state, central_cost.mean((1, 2, 3)), .5 * r.square().sum(1).mean((-2, -1)), (centre - target_centre).norm(dim=-1).mean((-2, -1)), ((slab - target_slab).norm(dim=-1) * w[None, :, None, None]).sum(1).mean((-2, -1)), angle, ((c - support_origin) * n).sum(-1) - torch.where((n * truth_n).sum(-1) < 0, -truth_d, truth_d), (feature.norm(dim=1) <= 1e-6).sum((-2, -1)))
                for key, value in zip(list(data)[:8], metrics): history[key].append(value.cpu().numpy())
                if step == 3:
                    continue
                columns, differences = [], []
                for axis in range(3):
                    probe_features = []
                    for sign in (1., -1.):
                        delta = state.new_zeros(18, 9); delta[:, axis] = sign * h[axis]
                        _, unit, _, _ = render(compose_antipodal_plane_frame_residual(state, delta, support_origin))
                        probe_features.append(unit)
                        differences.append(1 - (source_unit * unit).sum(1, keepdim=True) - central_cost)
                    columns.append((probe_features[0] - probe_features[1]).flatten(1).double() / 2)
                J = torch.stack(columns, dim=-1)
                residual_vector = r.flatten(1).double()
                H = J.transpose(1, 2) @ J / J.shape[1]
                g = (J.transpose(1, 2) @ residual_vector[..., None]).squeeze(-1) / J.shape[1]
                diagonal = H.diagonal(dim1=-2, dim2=-1)
                mean_diagonal = diagonal.mean(-1)
                matrix = H + .1 * torch.diag_embed(diagonal) + (1e-6 * mean_diagonal)[:, None, None] * torch.eye(3, device="cuda", dtype=torch.float64)
                information = torch.isfinite(matrix).all((-2, -1)) & torch.isfinite(g).all(-1) & (mean_diagonal > 0)
                q = state.new_full((18, 3), torch.nan)
                solution, solver_info = torch.linalg.solve_ex(matrix[information], -g[information], check_errors=False)
                q[information] = solution
                solve_failed = torch.zeros(18, device="cuda", dtype=torch.bool)
                solve_failed[information] = (solver_info != 0) | ~torch.isfinite(solution).all(-1)
                q[solve_failed] = torch.nan
                eigen = state.new_full((18, 3), torch.nan)
                damped_eigen = state.new_full((18, 3), torch.nan)
                finite_H = torch.isfinite(H).all((-2, -1))
                eigen[finite_H] = torch.linalg.eigvalsh(H[finite_H])
                finite_matrix = torch.isfinite(matrix).all((-2, -1))
                damped_eigen[finite_matrix] = torch.linalg.eigvalsh(matrix[finite_matrix])
                condition = damped_eigen[:, -1] / damped_eigen[:, 0]
                evidence = pose.refinement_pair_encoder(_pair_evidence(source_feature, feature, pose.correlation_radius))
                fy, fx = torch.meshgrid(torch.arange(evidence.shape[-2], device="cuda", dtype=state.dtype) * 4, torch.arange(evidence.shape[-1], device="cuda", dtype=state.dtype) * 4, indexing="ij")
                raster_xy = 2 * (torch.stack((fx, fy)) + .5) / 96 - 1
                signs = state.new_tensor((1. - 2. * reflection, 1.))
                coordinates = torch.cat(((centre[:, ::4, ::4].permute(0, 3, 1, 2) - support_origin[None, :, None, None]) / 10000., raster_xy[None].expand(18, -1, -1, -1), signs[None, :, None, None].expand(18, -1, *evidence.shape[-2:])), dim=1)
                evidence = evidence + pose.coordinate_evidence(coordinates.to(pose.coordinate_evidence.weight)).to(evidence)
                evidence = evidence + pose.signed_pose_evidence(torch.cat(differences, dim=1).to(pose.signed_pose_evidence.weight)).to(evidence)
                hidden = pose.recurrent_cell(evidence, hidden)
                learned = pose._bounded_update(pose.recurrent_update(hidden.mean((-2, -1)))).to(state)
                proposed = learned[:, :3] / h if arm == "learned_capped" else q
                finite = torch.isfinite(proposed).all(-1)
                step_norm = proposed.norm(dim=-1)
                capped = proposed / step_norm.clamp_min(1)[:, None]
                update = state.new_zeros(18, 9)
                update[:, :3] = torch.where(finite[:, None], capped * h, torch.zeros_like(capped))
                values = (H, eigen, condition, g, q, learned, update, step_norm > 1, mean_diagonal == 0, solve_failed | ~finite_matrix | ~torch.isfinite(g).all(-1))
                for key, value in zip(list(data)[8:], values): history[key].append(value.cpu().numpy())
                state = compose_antipodal_plane_frame_residual(state, update, support_origin)
            for key in data: data[key].append(np.stack(history[key], axis=1))
        arrays = {key: np.stack(value) for key, value in data.items()}
        np.savez_compressed(OUT / f"section_{si:02d}.npz", **arrays, initial_state=initial.cpu().numpy(), truth_state=truth_state.cpu().numpy(), truth_feature_cost=truth_cost.cpu().numpy(), source_epsilon_pixels=(source_feature.norm(dim=1) <= 1e-6).sum((-2, -1)).cpu().numpy(), saved_parent_start_indices=starts, mode_index=mode_indices.cpu().numpy(), canonical_residual_local_um=raw["canonical_residual_local_um"], canonical_director_delta_local=raw["canonical_director_delta_local"], offsets_um=raw["offsets_um"], psf_weights=raw["psf_weights"], reflection=reflection)
        rows.append({"subject_id": subject_id, "section_id": section_id, "section_artifacts": record["artifacts"], "observations": [{**item["lineage"], "mode": item["selected_mode"], "eligible": item["support_information_eligible"]} for item in section["observations"]], "parent_start_labels": raw["state_labels"][starts].tolist(), "artifact": f"section_{si:02d}.npz"})
        print(json.dumps({"section": si + 1, "subject_id": subject_id, "elapsed_seconds": time.perf_counter() - started}), flush=True)
assert len(rows) == 8
assert all(torch.equal(value.cpu(), checkpoint["model_state"][name]) for name, value in model.state_dict().items())
(OUT / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
summary = {"scope": "TRAIN-only oracle-conditioned fixed-field diagnostic, NOT original native rollout or calibrated uncertainty. Nonzero oracle fields bypass the trained pose-only prefix: off-policy comparison.", "arms": arms, "observations": 24, "starts_per_observation": 6, "updates_per_arm": 3, "dimensionless_probe_scales": h.cpu().tolist(), "damping": "H+.1diag(H)+1e-6mean(diag(H))*I", "trust_region": "dimensionless L2<=1 for both arms; pre-cap learned9D retained", "fields": "saved original diagnostic canonical local coefficients, unchanged; one observed spatial reflection; normalized saved finite PSF", "support_mask_weights": False, "optimizer_steps": 0, "weights_unchanged": True, "loaded_archived_sources": loaded_source, "input_sha256": hashes, "source_sha256": sha(Path(__file__)), "output_sha256": {path.name: sha(path) for path in OUT.iterdir() if path.is_file()}, "elapsed_seconds": time.perf_counter() - started}
(OUT / "completed.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
