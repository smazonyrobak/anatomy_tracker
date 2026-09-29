"""One frozen native training batch: zero-head compatibility and backward, no update."""
import os
import sys
from pathlib import Path

ROOT = Path("I:/AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda")
sys.dont_write_bytecode = True

import hashlib
import json
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

RUN = ROOT / "runs/joint_v6_signed_pose_evidence_preflight_001"
NATIVE = ROOT / "runs/joint_v6_ribbon_local_001"
DATA = ROOT / "data/joint_v6_coherent_subject_cohort_sections_002"
PARENT = ROOT / "runs/joint_v6_imagekey_retrieval_001/joint_model_step_04000.pt"
PINS = {
    PARENT: "d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5",
    NATIVE / "completed.json": "a2bee6dc1ffcaafcef3b4ad4a49223653f582f41ab071a3ca25928e7db7a83d7",
    NATIVE / "schedule.pt": "e8aaa6eb1ff7e70097cbf1eeb1f6b9fc4d2ba4f01f26fc6d91b671c90c8dc0b9",
    DATA / "completed.json": "ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2",
}
for path, expected in PINS.items():
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
native = json.loads((NATIVE / "completed.json").read_text())
completed = json.loads((DATA / "completed.json").read_text())
identities_bytes = (NATIVE / "observation_identities.json").read_bytes()
identities = json.loads(identities_bytes)
schedule = torch.load(NATIVE / "schedule.pt", map_location="cpu", weights_only=False)
rows = schedule["training_observation_index"][0]
perturbation = schedule["training_perturbation"][0]
assert rows.shape == (4,) and perturbation.shape == (4, 9)
assert all(identities[i]["split"] == "train" for i in rows.tolist())
parent = torch.load(PARENT, map_location="cpu", weights_only=False, mmap=True)
assert parent["step"] == 4000
catalogue_path = PARENT.parent / "catalogue.pt"
with catalogue_path.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == parent["experiment"]["prepared_source_sha256"]["catalogue.pt"]
catalogue = torch.load(catalogue_path, map_location="cpu", weights_only=False)
repository = Path(__file__).resolve().parents[1]
RUN.mkdir(parents=True, exist_ok=False)
source_names = sorted(set(parent["experiment"]["source"]["file_sha256"]) | {
    "training/preflight_joint_v6_signed_pose_evidence.py", "training/arbitrary_plane_ribbon_v6.py",
    "training/run_joint_v6_ribbon_local.py", "training/subject_deformed_slab_multiresolution_bundle_v2.py",
})
source = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(), "file_sha256": {}}
for name in source_names:
    archived = RUN / "source" / name
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository / name, archived)
    source["file_sha256"][name] = hashlib.sha256(archived.read_bytes()).hexdigest()

sections, artifact_hashes = {}, {}
for i in sorted({identities[row]["section_array_index"] for row in rows.tolist()}):
    record = completed["sections"][i]
    for name, expected in record["artifact_sha256"].items():
        with (DATA / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
        artifact_hashes[name] = expected
    sections[i] = _read_raw_artifact(DATA, record["artifacts"])
physical = {key: [] for key in ("state", "plane", "centre", "slab", "director", "offsets", "weights", "reflection")}
channels, visible = [], []
for row in rows.tolist():
    identity = identities[row]
    section = sections[identity["section_array_index"]]
    item = next(item for item in section["observations"] if item["selected_mode"] == identity["selected_mode"])
    assert item["support_information_eligible"] and all(identity[key] == value for key, value in item["lineage"].items())
    fit = section["canonical_anatomy_plane_fit"]["arrays"]
    state = full_frame_state_from_components(*physical_ouv_to_frame(torch.from_numpy(fit["physical_ouv_ap_dv_ml_um_float64"])))
    centre, slab = section["target_centre_ccf_coordinates_ap_dv_ml_um_float64"], section["target_psf_ccf_coordinates_ap_dv_ml_um_float64"]
    z, w = section["axial_offsets_um_float64"], section["axial_weights_float64"]
    director = np.einsum("s,shwc->hwc", w * z, slab - centre[None]) / np.sum(w * z * z)
    values = (state, torch.from_numpy(fit["fitted_coordinate_raster_ap_dv_ml_um_float64"]), torch.from_numpy(centre), torch.from_numpy(slab), torch.from_numpy(director), torch.from_numpy(z), torch.from_numpy(w), torch.tensor(int(section["reflection_xy"][0])))
    for key, value in zip(physical, values):
        physical[key].append(value.float() if key != "reflection" else value)
    channels.append(torch.from_numpy(item["image_outline_availability_float32"]))
    visible.append(torch.from_numpy(item["visible_finite_support_float32"]))
data = {key: torch.stack(values).cuda() for key, values in physical.items()}
inputs, q = torch.stack(channels).cuda(), torch.stack(visible).cuda()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
del atlas_array, annotation, sections
runtime = make_complete_catalogue_runtime_v6(catalogue, expected_catalogue_receipt_sha256=catalogue["receipt_sha256"], device="cuda", dtype=torch.float32)
kwargs = native["experiment"]["model_kwargs"]
torch.manual_seed(2026092913)
disabled = ArbitraryPlaneJointModelV6(runtime, **kwargs)
disabled_rng = torch.get_rng_state().clone()
torch.manual_seed(2026092913)
enabled = ArbitraryPlaneJointModelV6(runtime, **kwargs, signed_pose_evidence=True)
assert torch.equal(disabled_rng, torch.get_rng_state())
new_key = "pose_model.signed_pose_evidence.weight"
assert set(enabled.state_dict()) - set(disabled.state_dict()) == {new_key}
assert all(torch.equal(value, enabled.state_dict()[name]) for name, value in disabled.state_dict().items())
assert torch.count_nonzero(enabled.state_dict()[new_key]) == 0
trainable = (*native["experiment"]["trainable_prefixes"], "pose_model.signed_pose_evidence.")
for model in (disabled, enabled):
    missing, unexpected = model.load_state_dict(parent["model_state"], strict=False)
    expected = {"pose_model.coordinate_evidence.weight", "ribbon_field_head.weight", "ribbon_field_head.bias"}
    if model is enabled:
        expected.add(new_key)
    assert set(missing) == expected and not unexpected
    assert all(torch.count_nonzero(model.state_dict()[name]) == 0 for name in expected)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable))
origin, spacing, support_origin = [catalogue["support_geometry"][key] for key in ("origin_ap_dv_ml_um", "voxel_size_ap_dv_ml_um", "support_origin_ap_dv_ml_um")]
height, width = inputs.shape[-2:]
initial = compose_antipodal_plane_frame_residual(data["state"], perturbation.cuda(), support_origin).detach()
flags = torch.tensor([False, True], device="cuda")
arguments = (inputs[:, :1], inputs[:, 1:2], inputs[:, 2, 0, 0].bool(), atlas)
tail = (initial.new_full((4, 1, 2), -np.log(2)), flags, (height, width), origin, spacing, support_origin, data["offsets"], data["weights"], 3)
measurements = {}
for label, model in (("disabled", disabled), ("enabled_zero_projection", enabled)):
    model.cuda().eval()
    render_calls = []
    hook = model.pose_model.atlas_stem.register_forward_hook(lambda module, args, output: render_calls.append(1))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        out = model.refine_ribbon(*arguments, initial[:, None], *tail)
    torch.cuda.synchronize()
    measurements[label] = {"forward_seconds": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "atlas_encoder_calls": len(render_calls)}
    assert len(render_calls) == (4 if model is disabled else 22)
    hook.remove()
    tensor_outputs = {key: value.detach().cpu() for key, value in out.items() if torch.is_tensor(value)}
    assert all(bool(torch.isfinite(value).all()) for value in tensor_outputs.values())
    if model is disabled:
        reference = tensor_outputs
        model.cpu()
    else:
        assert set(reference) == set(tensor_outputs)
        assert all(torch.equal(reference[key], value) for key, value in tensor_outputs.items())
    del out, tensor_outputs
    print(json.dumps({label: measurements[label]}), flush=True)

gradient_nodes = {"raw_ribbon": [], "raw_pose_update": [], "signed_cost_differences": []}


def retain_gradient(value, name):
    value.retain_grad()
    gradient_nodes[name].append(value)


handles = [
    enabled.ribbon_field_head.register_forward_hook(lambda module, args, output: retain_gradient(output, "raw_ribbon")),
    enabled.pose_model.recurrent_update.register_forward_hook(lambda module, args, output: retain_gradient(output, "raw_pose_update")),
    enabled.pose_model.signed_pose_evidence.register_forward_pre_hook(lambda module, args: retain_gradient(args[0], "signed_cost_differences")),
]
enabled.train()
initial.requires_grad_(True)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
out = enabled.refine_ribbon(*arguments, initial[:, None], *tail)
row, branch = torch.arange(4, device="cuda"), data["reflection"]
mass = q.sum((-2, -1)).clamp_min(1)
centre = out["observed_centre_ccf_ap_dv_ml_um_sequence"][row, 0, branch, 1:]
sequence = .8 ** torch.arange(2, -1, -1, device="cuda")
centre_error = ((centre - data["centre"][:, None]).square().sum(-1) * q[:, None]).sum((-2, -1)) / mass[:, None] / 100.**2
slab = out["final_observed_ccf_slab_ap_dv_ml_um"][row, 0, branch]
state = out["final_component_state"][:, 0]
_, frame, _ = full_frame_state_to_components(state)
director = frame[..., :, 2][:, :, None, None] + torch.einsum("brij,brjhw->brhwi", frame, out["final_canonical_director_delta_local"][:, 0])
director = torch.where(flags[None, :, None, None, None], director.flip(-2), director)
ouv = full_frame_state_to_physical_ouv(state[row, branch])
yy, xx = torch.meshgrid(torch.arange(height, device="cuda") / height, torch.arange(width, device="cuda") / width, indexing="ij")
plane = ouv[:, None, None, :3] + xx[None, :, :, None] * ouv[:, None, None, 3:6] + yy[None, :, :, None] * ouv[:, None, None, 6:9]
losses = {
    "centre": ((centre_error * sequence).sum(-1) / sequence.sum()).mean(),
    "slab": (((slab - data["slab"]).square().sum(-1) * q[:, None] * data["weights"][:, :, None, None]).sum((1, 2, 3)) / mass / 100.**2).mean(),
    "plane_gauge": .1 * (plane - data["plane"]).square().sum(-1).mean() / 100.**2,
    "world_director": .1 * (((director[row, branch] - data["director"]).square().sum(-1) * q).sum((-2, -1)) / mass / .05**2).mean(),
    "reflection_ce": F.nll_loss(out["conditional_kept_component_log_probability"][:, 0], branch),
}
objective = sum(losses.values())
assert bool(torch.isfinite(objective))
objective.backward()
torch.cuda.synchronize()
measurements["enabled_forward_backward"] = {"seconds": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
for handle in handles:
    handle.remove()
gradient_summary = {}
for name, nodes in gradient_nodes.items():
    assert len(nodes) == 3 and all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in nodes)
    gradient_summary[name] = {"count": len(nodes), "gradient_norms": [float(value.grad.norm()) for value in nodes]}
assert initial.grad is not None and bool(torch.isfinite(initial.grad).all())
projection_grad = enabled.pose_model.signed_pose_evidence.weight.grad
assert projection_grad is not None and bool(torch.isfinite(projection_grad).all()) and float(projection_grad.norm()) > 0
assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in enabled.parameters() if parameter.grad is not None)
assert all(torch.equal(value, enabled.state_dict()[name].detach().cpu()) for name, value in disabled.state_dict().items())
assert torch.count_nonzero(enabled.state_dict()[new_key]) == 0
for name, expected in source["file_sha256"].items():
    assert hashlib.sha256((repository / name).read_bytes()).hexdigest() == expected
result = {
    "passed": True, "optimizer_steps_applied": 0, "source": source,
    "input_sha256": {str(path): value for path, value in PINS.items()}, "section_artifact_sha256": artifact_hashes,
    "native_observation_identities_sha256": hashlib.sha256(identities_bytes).hexdigest(),
    "observation_index": rows.tolist(), "identities": [identities[i] for i in rows.tolist()], "perturbation": perturbation.tolist(),
    "model_kwargs": {**kwargs, "signed_pose_evidence": True}, "probe_steps": {"normal_tangent_rad": np.pi / 60, "normal_offset_um": 150., "order": ["u+", "u-", "v+", "v-", "d+", "d-"]},
    "common_initialization_and_cpu_rng_exact": True, "all_zero_projection_outputs_exact": True, "all_parameters_unchanged": True,
    "losses": {key: float(value.detach()) for key, value in losses.items()}, "gradient_checks": gradient_summary,
    "initial_state_gradient_norm": float(initial.grad.norm()), "new_projection_gradient_norm": float(projection_grad.norm()),
    "measurements": measurements, "device": torch.cuda.get_device_name(), "precision": "FP32; AMP/TF32 disabled",
    "scope": "One first scheduled native B4 TRAIN batch, T3/prefix1; no optimizer, new data, development rows or model-quality claim. Gradients checked at initial state, raw pose/ribbon heads and probe-cost maps; zero projection makes cost-map gradients zero by design. Timing includes instrumentation and is not a throughput benchmark.",
}
(RUN / "preflight.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"passed": True, "measurements": measurements, "new_projection_gradient_norm": result["new_projection_gradient_norm"], "preflight_sha256": hashlib.sha256((RUN / "preflight.json").read_bytes()).hexdigest()}), flush=True)
