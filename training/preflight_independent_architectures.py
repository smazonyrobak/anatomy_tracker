"""Fixed-workload, pre-training resource preflight for cold-start architectures."""

from __future__ import annotations

import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from training.independent_joint_data import (
    OUTLINE_CURRICULUM_CONTRACT,
    OUTLINE_MODE_NAMES,
    OUTLINE_MODE_PROBABILITIES,
)
from training.independent_joint_model import (
    IndependentCandidateScorerExport,
    IndependentJointModel,
)
from training.independent_joint_variants import (
    FactorizedCNNControl,
    RecurrentAttentionVariant,
    install_resource_hooks,
    resource_snapshot,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
CONFIG_SCHEMA_VERSION = 1
FIXED_WORKLOAD = {
    "seed": 4322,
    "source_batch": 3,
    "candidates_per_source": 3,
    "height": 320,
    "width": 464,
    "recurrent_steps": 3,
    "warmup_iterations": 1,
    "measured_iterations": 3,
    "amp": True,
    "onnx_provider": "DmlExecutionProvider",
}
MASK_CURRICULUM = {
    "mode_probabilities": {
        name: float(probability)
        for name, probability in zip(OUTLINE_MODE_NAMES, OUTLINE_MODE_PROBABILITIES)
    },
    "preflight_modes": list(OUTLINE_MODE_NAMES),
    "contract_sha256": hashlib.sha256(
        OUTLINE_CURRICULUM_CONTRACT.encode("utf-8")
    ).hexdigest(),
}
PREFLIGHT_PROTOCOL = {
    "purpose": "resource-and-export-preflight-only",
    "optimizer_steps": 0,
    "cold_start": True,
    "learned_checkpoint_dependencies": [],
}
MODEL_CLASSES = {
    "training.independent_joint_model.IndependentJointModel": IndependentJointModel,
    "training.independent_joint_variants.FactorizedCNNControl": FactorizedCNNControl,
    "training.independent_joint_variants.RecurrentAttentionVariant": RecurrentAttentionVariant,
}


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source_sha256(path: str | Path) -> str:
    source = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_config(path: str | Path) -> dict:
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    contract = config.pop("contract_sha256")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION or config.get("frozen") is not True:
        raise ValueError("architecture preflight config is not frozen schema v1")
    if _canonical_sha256(config) != contract:
        raise ValueError("architecture preflight config hash differs from its frozen payload")
    if config["workload"] != FIXED_WORKLOAD:
        raise ValueError("architecture preflight workload differs from the fixed comparison")
    if config["mask_curriculum"] != MASK_CURRICULUM:
        raise ValueError("architecture preflight changed the frozen outline-mask curriculum")
    if config["training_protocol"] != PREFLIGHT_PROTOCOL:
        raise ValueError("architecture screen config is not preflight-only")
    if config["architecture"]["class"] not in MODEL_CLASSES:
        raise ValueError("architecture class is not in the cold-start preflight allowlist")
    if int(config["architecture"]["recurrent_steps"]) != 3:
        raise ValueError("architecture screen requires exactly three render/update steps")
    for relative_path, expected_hash in config["lineage"]["source_sha256"].items():
        if _source_sha256(REPOSITORY_ROOT / relative_path) != expected_hash:
            raise ValueError(f"source lineage changed: {relative_path}")
    config["contract_sha256"] = contract
    config["config_file_sha256"] = _source_sha256(path)
    return config


def build_model(config: dict) -> IndependentJointModel:
    torch.manual_seed(int(config["workload"]["seed"]))
    model = MODEL_CLASSES[config["architecture"]["class"]](
        **config["architecture"]["kwargs"]
    )
    if getattr(model, "learned_weight_dependencies", ()):
        raise RuntimeError("architecture preflight cannot load learned dependencies")
    if getattr(model, "initialization", None) != "random":
        raise RuntimeError("architecture preflight requires random initialization")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["architecture"]["expected_parameter_count"]):
        raise RuntimeError("architecture parameter count differs from its frozen config")
    if _state_sha256(model) != config["architecture"]["initial_state_sha256"]:
        raise RuntimeError("random initial state differs from its frozen lineage receipt")
    if bool(getattr(model, "uses_recurrent_state", True)) != bool(
        config["architecture"]["uses_recurrent_state"]
    ):
        raise RuntimeError("architecture recurrence semantics differ from its frozen config")
    return model


def _fixed_batch(workload: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch = int(workload["source_batch"])
    candidates = int(workload["candidates_per_source"])
    height = int(workload["height"])
    width = int(workload["width"])
    if batch != 3:
        raise ValueError("fixed preflight uses one accurate, imperfect, and absent outline")
    generator = torch.Generator().manual_seed(int(workload["seed"]))
    y = torch.linspace(-1.0, 1.0, height)
    x = torch.linspace(-1.0, 1.0, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    tissue = ((xx / 0.88).square() + (yy / 0.78).square() <= 1.0).float()
    image = 0.12 + 0.18 * torch.rand(batch, 1, height, width, generator=generator)
    anatomy = (
        0.45
        + 0.20 * torch.cos(xx * 4.0)[None, None]
        + 0.15 * torch.sin(yy * 5.0)[None, None]
    )
    image = image + tissue[None, None] * anatomy
    accurate = tissue.clone()
    imperfect = torch.roll(tissue, shifts=(1, -2), dims=(0, 1))
    imperfect[height // 3 : height // 3 + max(height // 14, 1), : width // 5] = 0.0
    source_mask = torch.stack(
        (accurate, imperfect, torch.zeros_like(tissue)), dim=0
    )[:, None]
    mask_available = torch.tensor((1.0, 1.0, 0.0)).view(batch, 1, 1, 1)
    atlas_base = torch.flip(image * tissue[None, None], dims=(-1,))
    offsets = torch.linspace(-0.04, 0.04, candidates)[None, :, None, None, None]
    candidate_image = (atlas_base[:, None] + offsets).clamp(0.0, 1.0)
    candidate_mask = tissue[None, None, None].expand(
        batch, candidates, 1, height, width
    )
    ap = torch.tensor((-1800.0, -2200.0, -2600.0))[:, None]
    candidate_offset = torch.linspace(-100.0, 100.0, candidates)[None]
    candidate_pose = torch.stack(
        (
            ap + candidate_offset,
            torch.linspace(-2.0, 2.0, candidates)[None].expand(batch, -1),
            torch.linspace(1.5, -1.5, candidates)[None].expand(batch, -1),
        ),
        dim=2,
    )
    source_index = torch.arange(batch).repeat_interleave(candidates)
    middle = candidates // 2
    return {
        "source_image": image.to(device),
        "source_mask": source_mask.to(device),
        "mask_available": mask_available.to(device),
        "candidate_image": candidate_image.flatten(0, 1).to(device),
        "candidate_mask": candidate_mask.flatten(0, 1).to(device),
        "candidate_pose": candidate_pose.flatten(0, 1).to(device),
        "source_index": source_index.to(device),
        "recurrent_image": candidate_image[:, middle].to(device),
        "recurrent_mask": candidate_mask[:, middle].to(device),
    }


def _fixed_forward(
    model: IndependentJointModel,
    batch: dict[str, torch.Tensor],
    recurrent_steps: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    source_features = model.encode_source(
        batch["source_image"], batch["source_mask"], batch["mask_available"]
    )
    initialization = model.pose_head(source_features)
    atlas_available = batch["candidate_image"].new_ones(
        len(batch["candidate_image"]), 1, 1, 1
    )
    ranking = model.score_candidate_from_features(
        batch["candidate_image"],
        batch["candidate_mask"],
        atlas_available,
        batch["candidate_pose"],
        initialization["pose_context"],
        None,
        source_features[-2:],
        batch["source_index"],
    )
    pose = initialization["pose"]
    hidden = None
    recurrent_available = batch["recurrent_image"].new_ones(
        len(batch["recurrent_image"]), 1, 1, 1
    )
    recurrent = []
    for _ in range(recurrent_steps):
        update = model.score_candidate_from_features(
            batch["recurrent_image"],
            batch["recurrent_mask"],
            recurrent_available,
            pose,
            initialization["pose_context"],
            hidden,
            source_features[-2:],
        )
        pose = update["pose"]
        hidden = update["hidden_state"]
        recurrent.append(update)
    dense = model.refine_from_features(
        batch["recurrent_image"],
        batch["recurrent_mask"],
        recurrent_available,
        pose,
        initialization["pose_context"],
        hidden,
        source_features,
    )
    loss = (
        initialization["ap_logits"].mean()
        + initialization["pose_cholesky"].mean()
        + ranking["compatibility_logit"].mean()
        + sum(update["compatibility_logit"].mean() for update in recurrent)
        + dense["pose_delta"].sum() * 1e-3
        + dense["similarity_parameters"].sum()
        + dense["stationary_velocity"].mean()
        + dense["affine_velocity_coefficients"].sum()
        + dense["validity_logits"].mean()
    )
    return {
        "initialization": initialization,
        "ranking": ranking,
        "recurrent": recurrent,
        "dense": dense,
    }, loss


def _synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _onnx_provider_check(
    model: IndependentJointModel,
    workload: dict,
    requested_provider: str,
) -> dict:
    import onnx
    import onnxruntime as ort

    model = model.cpu().eval()
    batch = _fixed_batch(workload, torch.device("cpu"))
    with torch.no_grad():
        features = model.encode_source(
            batch["source_image"], batch["source_mask"], batch["mask_available"]
        )
        initialization = model.pose_head(features)
        state = model.initial_hidden_state(batch["candidate_image"])
        atlas_available = torch.ones(
            len(batch["candidate_image"]), 1, 1, 1
        )
        arguments = (
            batch["candidate_image"],
            batch["candidate_mask"],
            atlas_available,
            batch["candidate_pose"],
            initialization["pose_context"],
            state,
            batch["source_index"],
            *features[-2:],
        )
        expected = IndependentCandidateScorerExport(model)(*arguments)
    input_names = [
        "atlas_image", "atlas_mask", "atlas_mask_available", "current_pose",
        "pose_context", "hidden_state", "source_index", "source_feature_2",
        "source_feature_3",
    ]
    output_names = [
        "pose", "pose_delta", "compatibility_logit", "hidden_state_out"
    ]
    buffer = io.BytesIO()
    torch.onnx.export(
        IndependentCandidateScorerExport(model),
        arguments,
        buffer,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            **{
                name: {0: "candidate_batch"}
                for name in input_names[:7]
                if name != "pose_context"
            },
            "pose_context": {0: "source_batch"},
            "source_feature_2": {0: "source_batch"},
            "source_feature_3": {0: "source_batch"},
            **{name: {0: "candidate_batch"} for name in output_names},
        },
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load_from_string(buffer.getvalue())
    onnx.checker.check_model(graph)
    available = ort.get_available_providers()
    if requested_provider not in available:
        return {
            "requested_provider": requested_provider,
            "available_providers": available,
            "checker_passed": True,
            "runtime_passed": False,
            "graph_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
            "max_abs_error": None,
        }
    options = ort.SessionOptions()
    if requested_provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        buffer.getvalue(), sess_options=options, providers=[requested_provider]
    )
    actual = session.run(
        None,
        {
            name: value.detach().numpy()
            for name, value in zip(input_names, arguments)
        },
    )
    maximum_error = max(
        float((expected_value - torch.from_numpy(actual_value)).abs().max())
        for expected_value, actual_value in zip(expected, actual)
    )
    finite = all(np.isfinite(value).all() for value in actual)
    return {
        "requested_provider": requested_provider,
        "available_providers": available,
        "checker_passed": True,
        "runtime_passed": bool(finite and maximum_error <= 5e-3),
        "graph_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "max_abs_error": maximum_error,
    }


def profile_model(
    model: IndependentJointModel,
    workload: dict,
    *,
    device: str,
    onnx_provider: str,
) -> dict:
    selected_device = torch.device(device)
    model = model.to(selected_device).train()
    batch = _fixed_batch(workload, selected_device)
    amp_enabled = bool(workload["amp"] and selected_device.type == "cuda")
    initial_state = _state_sha256(model)
    statistics_record, handles = install_resource_hooks(model)
    with torch.no_grad(), torch.amp.autocast(
        device_type=selected_device.type, enabled=amp_enabled
    ):
        _fixed_forward(model, batch, int(workload["recurrent_steps"]))
    for handle in handles:
        handle.remove()

    for _ in range(int(workload["warmup_iterations"])):
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=selected_device.type, enabled=amp_enabled):
            _, loss = _fixed_forward(model, batch, int(workload["recurrent_steps"]))
        loss.backward()
    _synchronize(selected_device)
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)

    forward_seconds = []
    backward_seconds = []
    for _ in range(int(workload["measured_iterations"])):
        model.zero_grad(set_to_none=True)
        _synchronize(selected_device)
        start = time.perf_counter()
        with torch.amp.autocast(device_type=selected_device.type, enabled=amp_enabled):
            _, loss = _fixed_forward(model, batch, int(workload["recurrent_steps"]))
        _synchronize(selected_device)
        middle = time.perf_counter()
        loss.backward()
        _synchronize(selected_device)
        end = time.perf_counter()
        forward_seconds.append(middle - start)
        backward_seconds.append(end - middle)

    resources = resource_snapshot(model, statistics_record)
    model.zero_grad(set_to_none=True)
    provider = _onnx_provider_check(model, workload, onnx_provider)
    return {
        "parameter_count": resources["parameters"],
        "mac_proxy": resources["macs"],
        "mac_proxy_scope": "Conv2d+Linear+explicit-local-attention; excludes elementwise and grid sampling",
        "peak_vram_bytes": resources["peak_vram_bytes"],
        "forward_median_seconds": statistics.median(forward_seconds),
        "backward_median_seconds": statistics.median(backward_seconds),
        "device": str(selected_device),
        "amp_enabled": amp_enabled,
        "initial_state_sha256": initial_state,
        "onnx": provider,
    }


def profile_config(path: str | Path, *, device: str | None = None) -> dict:
    config = load_frozen_config(path)
    model = build_model(config)
    profile = profile_model(
        model,
        config["workload"],
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        onnx_provider=config["workload"]["onnx_provider"],
    )
    return {
        "preflight_only": True,
        "config_name": config["name"],
        "config_contract_sha256": config["contract_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "architecture": config["architecture"],
        "workload": config["workload"],
        "mask_curriculum": config["mask_curriculum"],
        "lineage": config["lineage"],
        "profile": profile,
    }


if __name__ == "__main__":
    print(json.dumps(profile_config(sys.argv[1]), indent=2, sort_keys=True))
