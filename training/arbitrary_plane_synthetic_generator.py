"""Complete random-only synthetic observations from finite arbitrary CCF planes.

The stored maps are absolute float32 pixel-centre coordinates in ``(x, y)``
order. ``source_to_fixed_map`` is the pullback used to render source truth;
``fixed_to_source_map`` maps fixed atlas pixels into that source raster.  No
learned model, checkpoint, pseudo-label, or historical generator is loaded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage

from training.arbitrary_plane_rendered_generator import (
    FINITE_RENDER_SCHEMA,
    effective_renderer_sampling_arrays,
    finite_render_receipt,
    verify_finite_arbitrary_plane_render,
)
from training.arbitrary_plane_synthetic_observation import (
    ABSENT_OUTLINE,
    ACCURATE_OUTLINE,
    IMPERFECT_OUTLINE,
    SMART_BRUSH_MODES,
    robust_clean_normalization,
    smart_brush_input,
)
from training.arbitrary_plane_synthetic_ops import (
    ARBITRARY_PLANE_SYNTHETIC_OPS_VERSION,
    bilinear_sample_scalar,
    fixed_source_maps,
    identity_pixel_map,
    nearest_sample_labels,
    remove_tissue_affine_component,
    sample_multiscale_physical_velocity,
    topology_acceptance_metrics,
)


SYNTHETIC_SCHEMA = "anatomy-tracker.arbitrary-plane-synthetic-realization/v1"
SYNTHETIC_ALGORITHM = "provenance-bound-arbitrary-plane-g1-g2-g3/v1"
SYNTHETIC_STRATA = ("ordinary", "tiny-tangent-stress", "low-information-stress")
SEED_ENCODING = "canonical-lowercase-uint64-hex/v1"
_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = {
    "generator": Path(__file__).resolve(),
    "ops": Path(__file__).with_name("arbitrary_plane_synthetic_ops.py"),
    "observation": Path(__file__).with_name("arbitrary_plane_synthetic_observation.py"),
    "finite_renderer": Path(__file__).with_name("arbitrary_plane_rendered_generator.py"),
    "predeclared_config": _ROOT / "publication" / "arbitrary_plane_synthetic_preflight.yaml",
}

_DEFAULT_CONFIG = {
    "schema_version": SYNTHETIC_SCHEMA,
    "algorithm": SYNTHETIC_ALGORITHM,
    "maximum_g1_attempts": 32,
    "maximum_g2_attempts": 32,
    "ordinary_minimum_clean_brain_pixels_floor": 256,
    "ordinary_minimum_clean_brain_fraction": 0.005,
    "g1": {
        "identity_probability": 0.15,
        "correlation_length_over_D": [[0.08, 0.15], [0.20, 0.35]],
        "target_rms_displacement_over_D": [0.005, 0.04],
        "postintegration_rms_relative_tolerance": 0.02,
        "maximum_displacement_over_D": 0.10,
        "analytic_probability": 0.50,
        "analytic_radius_over_D": [0.10, 0.35],
        "analytic_peak_over_D": [0.0, 0.02],
        "similarity_angle_rad": [-0.02, 0.02],
        "similarity_scale": [0.995, 1.005],
        "similarity_translation_over_D": [-0.002, 0.002],
        "maximum_squaring_steps": 12,
        "minimum_jacobian": 0.20,
        "maximum_jacobian": 5.0,
        "maximum_cycle_rms_px": 0.05,
        "maximum_cycle_q99_px": 0.25,
        "maximum_cycle_max_px": 0.50,
        "minimum_tissue_retained_fraction": 0.995,
    },
    "g2": {
        "identity_probability": 0.15,
        "source_family_probabilities": [0.50, 0.25, 0.25],
        "mixture_alpha": [0.25, 0.75],
        "label_mean_range": [0.10, 0.90],
        "polarity_probability": 0.50,
        "gamma": [0.5, 2.0],
        "gain": [0.7, 1.4],
        "offset": [-0.15, 0.15],
        "bias_std": [0.0, 0.30],
        "blur_probability": 0.50,
        "blur_sigma_px": [0.0, 1.5],
        "resolution_probability": 0.25,
        "resolution_factor": [1.0, 3.0],
        "noise_probability": 0.70,
        "noise_std": [0.0, 0.06],
        "background_base": [0.0, 0.25],
        "background_field_std": [0.0, 0.08],
        "background_noise_std": [0.0, 0.03],
        "artifact_probability": 0.35,
        "artifact_fraction": [0.001, 0.01],
        "minimum_q99_q01": 0.10,
        "minimum_std": 0.03,
    },
    "g3": {
        "event_count_probabilities": [0.50, 0.40, 0.10],
        "affected_fraction": [0.005, 0.15],
        "disable_damage_below_pixels": 128,
        "maximum_union_damage_fraction": 0.25,
        "minimum_visible_fraction": 0.70,
        "minimum_visible_pixels": 64,
        "imperfect_iou": [0.70, 0.98],
        "maximum_outline_attempts": 64,
    },
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _uint64(value: int | str) -> int:
    if isinstance(value, str):
        if len(value) != 18 or not value.startswith("0x") or value != value.lower():
            raise ValueError("seed must be canonical 0x plus 16 lowercase hex digits")
        parsed = int(value[2:], 16)
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        parsed = int(value)
    else:
        raise TypeError("seed must be an integer or canonical uint64 hex string")
    if not 0 <= parsed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return parsed


def _seed_hex(value: int | str) -> str:
    return f"0x{_uint64(value):016x}"


def derive_field_seed(
    root_seed: int | str,
    split: str,
    sample_index: int,
    stage: str,
    field: str,
    attempt: int,
) -> int:
    """Canonical isolated RNG domain: SHA256 of six length-delimited fields."""
    if split not in {"train", "development"}:
        raise ValueError("synthetic generation is restricted to train/development")
    if int(sample_index) < 0 or int(attempt) < 0 or not stage or not field:
        raise ValueError("sample index, attempt, stage, and field must be named and nonnegative")
    parts = (
        SYNTHETIC_ALGORITHM,
        _seed_hex(root_seed),
        split,
        str(int(sample_index)),
        stage,
        field,
        str(int(attempt)),
    )
    payload = b"".join(len(part.encode()).to_bytes(4, "big") + part.encode() for part in parts)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _rng(config: dict[str, object], stage: str, field: str, attempt: int = 0) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(
            derive_field_seed(
                config["root_seed"], config["split"], config["sample_index"], stage, field, attempt
            )
        )
    )


def _seed_record(config: dict[str, object], stage: str, fields: list[str], attempt: int) -> dict[str, str]:
    return {
        field: _seed_hex(
            derive_field_seed(
                config["root_seed"], config["split"], config["sample_index"], stage, field, attempt
            )
        )
        for field in fields
    }


def _uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    low, high = map(float, bounds)
    return low if low == high else float(rng.uniform(low, high))


def _log_uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    low, high = map(float, bounds)
    if low <= 0.0:
        raise ValueError("log-uniform bounds must be positive")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _array_bytes(array: np.ndarray) -> tuple[bytes, str]:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype == np.bool_:
        return np.packbits(contiguous.reshape(-1), bitorder="little").tobytes(), "packbits-little"
    return contiguous.tobytes(order="C"), "C-order"


def _array_receipt(array: np.ndarray) -> dict[str, object]:
    value = np.asarray(array)
    body, storage = _array_bytes(value)
    header = _canonical_json({"dtype": value.dtype.str, "shape": list(value.shape), "storage": storage}).encode()
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "nbytes": int(value.nbytes),
        "stored_nbytes": len(body),
        "storage": storage,
        "array_sha256": hashlib.sha256(header + b"\0" + body).hexdigest(),
    }


def _array_receipts(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {name: _array_receipt(value) for name, value in sorted(arrays.items())}


def _merge_config(base: dict[str, object], changes: dict[str, object] | None) -> dict[str, object]:
    merged = copy.deepcopy(base)
    for key, value in (changes or {}).items():
        if key not in merged:
            raise ValueError(f"unknown synthetic config key: {key}")
        if isinstance(merged[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"synthetic config section {key} must be a mapping")
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _inside(pixel_map: np.ndarray) -> np.ndarray:
    height, width = pixel_map.shape[1:]
    return (
        (pixel_map[0] >= 0.0)
        & (pixel_map[0] <= width - 1.0)
        & (pixel_map[1] >= 0.0)
        & (pixel_map[1] <= height - 1.0)
    )


def _common_cell_mask(pixel_map: np.ndarray) -> np.ndarray:
    valid = _inside(pixel_map)
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    return valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]


def _tissue_scale(tissue: np.ndarray, pixel_pitch_um: float) -> tuple[float, float, float]:
    y, x = np.nonzero(tissue)
    if len(x):
        u_span = (float(x.max() - x.min()) + 1.0) * pixel_pitch_um
        v_span = (float(y.max() - y.min()) + 1.0) * pixel_pitch_um
    else:
        u_span = v_span = pixel_pitch_um
    return min(u_span, v_span), u_span, v_span


def _analytic_velocity(
    shape: tuple[int, int], tissue: np.ndarray, D_um: float, pitch_um: float, config: dict[str, object], attempt: int
) -> tuple[np.ndarray, dict[str, object]]:
    choice_rng = _rng(config, "g1", "analytic-choice", attempt)
    parameter_rng = _rng(config, "g1", "analytic-parameters", attempt)
    choices = ("radial-expansion-compression", "anisotropic-local-stretch", "swirl")
    choice = choices[int(choice_rng.integers(3))]
    locations = np.argwhere(tissue)
    center_y, center_x = locations[int(parameter_rng.integers(len(locations)))]
    radius_um = _uniform(parameter_rng, config["g1"]["analytic_radius_over_D"]) * D_um
    peak_um = _uniform(parameter_rng, config["g1"]["analytic_peak_over_D"]) * D_um
    peak_um *= -1.0 if parameter_rng.random() < 0.5 else 1.0
    y, x = np.mgrid[: shape[0], : shape[1]]
    dx = (x - center_x) * pitch_um
    dy = (y - center_y) * pitch_um
    distance = np.sqrt(dx * dx + dy * dy)
    envelope = np.exp(-0.5 * (distance / max(radius_um, pitch_um)) ** 2)
    norm = np.maximum(distance, pitch_um)
    if choice == choices[0]:
        velocity = np.stack((peak_um * envelope * dx / norm, peak_um * envelope * dy / norm))
    elif choice == choices[1]:
        angle = float(parameter_rng.uniform(-np.pi, np.pi))
        axis = np.asarray((math.cos(angle), math.sin(angle)))
        projection = axis[0] * dx + axis[1] * dy
        velocity = peak_um * envelope * axis[:, None, None] * projection[None] / max(radius_um, pitch_um)
    else:
        velocity = np.stack((-peak_um * envelope * dy / norm, peak_um * envelope * dx / norm))
    return velocity.astype(np.float32), {
        "enabled": True,
        "choice": choice,
        "center_xy_px": [float(center_x), float(center_y)],
        "radius_um": radius_um,
        "peak_amplitude_um": peak_um,
    }


def _g1(
    parent: dict[str, object], support: dict[str, object], config: dict[str, object]
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    scalar = np.asarray(parent["raster"]["scalar"], dtype=np.float32)
    fixed_labels = np.asarray(parent["raster"]["annotation"])
    fixed_tissue = np.asarray(parent["raster"]["brain_mask"], dtype=bool)
    height, width = scalar.shape
    ordinary_minimum = max(
        int(config["ordinary_minimum_clean_brain_pixels_floor"]),
        int(math.ceil(float(config["ordinary_minimum_clean_brain_fraction"]) * height * width)),
    )
    marginal_support = int(fixed_tissue.sum()) < ordinary_minimum
    pitch_um = float(parent["geometry"]["reference_aspect_policy"]["pixel_pitch_u_um"])
    if pitch_um != float(parent["geometry"]["reference_aspect_policy"]["pixel_pitch_v_um"]):
        raise ValueError("finite parent must use the isotropic reference pixel-pitch contract")
    D_um, u_span, v_span = _tissue_scale(fixed_tissue, pitch_um)
    g1 = config["g1"]
    fields = [
        "identity", "fine-correlation", "coarse-correlation", "fine-svf-field", "coarse-svf-field", "target-rms",
        "analytic-enable", "analytic-choice", "analytic-parameters", "similarity-angle",
        "similarity-scale", "similarity-translation",
    ]
    logs = []
    for attempt in range(int(config["maximum_g1_attempts"])):
        identity = bool(
            marginal_support
            or _rng(config, "g1", "identity", attempt).random()
            < float(g1["identity_probability"])
        )
        if identity:
            velocity_um = np.zeros((2, height, width), np.float32)
            target_rms_um = 0.0
            analytic = {"enabled": False, "choice": None}
        else:
            fine = _uniform(_rng(config, "g1", "fine-correlation", attempt), g1["correlation_length_over_D"][0])
            coarse = _uniform(_rng(config, "g1", "coarse-correlation", attempt), g1["correlation_length_over_D"][1])
            fine_velocity_um = sample_multiscale_physical_velocity(
                _rng(config, "g1", "fine-svf-field", attempt),
                (height, width),
                correlation_lengths_px=(max(1.0, fine * D_um / pitch_um),),
                rms_amplitudes_um=(1.0,),
            )
            coarse_velocity_um = sample_multiscale_physical_velocity(
                _rng(config, "g1", "coarse-svf-field", attempt),
                (height, width),
                correlation_lengths_px=(max(1.0, coarse * D_um / pitch_um),),
                rms_amplitudes_um=(1.0,),
            )
            velocity_um = fine_velocity_um + coarse_velocity_um
            velocity_px = remove_tissue_affine_component(velocity_um / pitch_um, fixed_tissue)
            velocity_um = velocity_px * pitch_um
            analytic_enabled = bool(
                _rng(config, "g1", "analytic-enable", attempt).random() < float(g1["analytic_probability"])
            )
            if analytic_enabled:
                primitive, analytic = _analytic_velocity((height, width), fixed_tissue, D_um, pitch_um, config, attempt)
                velocity_um += primitive
                velocity_um = remove_tissue_affine_component(velocity_um / pitch_um, fixed_tissue) * pitch_um
            else:
                analytic = {"enabled": False, "choice": None}
            target_rms_um = _log_uniform(
                _rng(config, "g1", "target-rms", attempt), g1["target_rms_displacement_over_D"]
            ) * D_um
            rms = float(np.sqrt(np.mean(np.sum(velocity_um[:, fixed_tissue].astype(np.float64) ** 2, axis=0))))
            velocity_um *= target_rms_um / max(rms, np.finfo(np.float32).eps)
        angle = 0.0 if identity else _uniform(_rng(config, "g1", "similarity-angle", attempt), g1["similarity_angle_rad"])
        scale = 1.0 if identity else _uniform(_rng(config, "g1", "similarity-scale", attempt), g1["similarity_scale"])
        translation = np.zeros(2, np.float32) if identity else (
            _rng(config, "g1", "similarity-translation", attempt).uniform(
                float(g1["similarity_translation_over_D"][0]), float(g1["similarity_translation_over_D"][1]), 2
            ) * D_um / pitch_um
        ).astype(np.float32)
        maps = fixed_source_maps(
            velocity_um,
            np.eye(2, dtype=np.float64) * pitch_um,
            angle_rad=angle,
            scale=scale,
            translation_xy=translation,
        )
        identity_map = identity_pixel_map((height, width))
        initial_postintegration_rms_um = (
            0.0
            if not fixed_tissue.any()
            else float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                maps["local_fixed_to_fixed_map"][:, fixed_tissue]
                                - identity_map[:, fixed_tissue]
                            ).astype(np.float64)
                            ** 2,
                            axis=0,
                        )
                    )
                )
                * pitch_um
            )
        )
        reintegrated_after_rms_rescale = False
        if not identity and initial_postintegration_rms_um > 0.0:
            velocity_um *= target_rms_um / initial_postintegration_rms_um
            maps = fixed_source_maps(
                velocity_um,
                np.eye(2, dtype=np.float64) * pitch_um,
                angle_rad=angle,
                scale=scale,
                translation_xy=translation,
            )
            reintegrated_after_rms_rescale = True
        achieved_postintegration_rms_um = (
            0.0
            if not fixed_tissue.any()
            else float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                maps["local_fixed_to_fixed_map"][:, fixed_tissue]
                                - identity_map[:, fixed_tissue]
                            ).astype(np.float64)
                            ** 2,
                            axis=0,
                        )
                    )
                )
                * pitch_um
            )
        )
        fixed_to_source = maps["fixed_to_source_map"]
        source_to_fixed = maps["source_to_fixed_map"]
        fixed_valid = _inside(fixed_to_source)
        source_valid = _inside(source_to_fixed)
        metrics = topology_acceptance_metrics(
            fixed_to_source,
            source_to_fixed,
            minimum_jacobian=float(g1["minimum_jacobian"]),
            maximum_jacobian=float(g1["maximum_jacobian"]),
            maximum_cycle_rms_px=float(g1["maximum_cycle_rms_px"]),
            maximum_cycle_q99_px=float(g1["maximum_cycle_q99_px"]),
            maximum_cycle_max_px=float(g1["maximum_cycle_max_px"]),
            fixed_cell_mask=_common_cell_mask(fixed_to_source),
            source_cell_mask=_common_cell_mask(source_to_fixed),
            fixed_valid_mask=fixed_valid,
            source_valid_mask=source_valid,
        )
        source_labels = nearest_sample_labels(fixed_labels, source_to_fixed)
        source_tissue = source_labels != 0
        retained = float((fixed_tissue & fixed_valid).sum() / max(int(fixed_tissue.sum()), 1))
        max_displacement_um = float(
            np.sqrt(
                np.sum(
                    (maps["local_fixed_to_fixed_map"] - identity_map).astype(np.float64) ** 2,
                    axis=0,
                )
            ).max()
            * pitch_um
        )
        fov_passed = bool(
            marginal_support
            or retained >= float(g1["minimum_tissue_retained_fraction"])
        )
        max_displacement_passed = max_displacement_um <= float(g1["maximum_displacement_over_D"]) * D_um
        steps_passed = int(maps["integration_steps"]) <= int(g1["maximum_squaring_steps"])
        rms_target_passed = bool(
            (identity and achieved_postintegration_rms_um == 0.0)
            or (
                not identity
                and abs(achieved_postintegration_rms_um - target_rms_um)
                / target_rms_um
                <= float(g1["postintegration_rms_relative_tolerance"])
            )
        )
        accepted = bool(
            metrics["accepted"] and fov_passed and max_displacement_passed
            and steps_passed and rms_target_passed
        )
        entry = {
            "attempt_index": attempt,
            "field_stream_seed_uint64": _seed_record(config, "g1", fields, attempt),
            "identity_path": identity,
            "target_rms_displacement_um": target_rms_um,
            "initial_postintegration_rms_displacement_um": initial_postintegration_rms_um,
            "achieved_postintegration_rms_displacement_um": achieved_postintegration_rms_um,
            "postintegration_rms_rescaled_and_reintegrated_once": reintegrated_after_rms_rescale,
            "postintegration_rms_relative_tolerance": float(g1["postintegration_rms_relative_tolerance"]),
            "postintegration_rms_target_gate_passed": rms_target_passed,
            "analytic_primitive": analytic,
            "similarity": {
                "angle_rad": angle,
                "scale": scale,
                "translation_xy_px": translation.tolist(),
                "reflection": False,
            },
            "integration_steps": int(maps["integration_steps"]),
            "topology_metrics": metrics,
            "clean_tissue_retained_in_valid_fov_fraction": retained,
            "fov_gate_passed": fov_passed,
            "maximum_postintegration_displacement_um": max_displacement_um,
            "maximum_displacement_gate_passed": max_displacement_passed,
            "squaring_steps_gate_passed": steps_passed,
            "accepted": accepted,
        }
        logs.append(entry)
        if accepted:
            effective = effective_renderer_sampling_arrays(
                parent["geometry"],
                tuple(int(value) for value in support["annotation_shape"]),
                origin_ap_dv_ml_um=tuple(support["origin_um"]),
                voxel_size_ap_dv_ml_um=tuple(support["voxel_size_um"]),
            )
            fixed_allen = effective["coordinate_raster_allen_index_float32"]
            source_allen = np.stack(
                [bilinear_sample_scalar(fixed_allen[..., axis], source_to_fixed) for axis in range(3)], axis=-1
            ).astype(np.float32)
            origin = np.asarray(support["origin_um"], np.float32)
            spacing = np.asarray(support["voxel_size_um"], np.float32)
            source_ccf_um = (origin + (source_allen + np.float32(0.5)) * spacing).astype(np.float32)
            source_ccf_um[~source_valid] = 0.0
            u0, v0 = (
                float(parent["geometry"]["sampled_endpoint_bounds_u_um"][0]),
                float(parent["geometry"]["sampled_endpoint_bounds_v_um"][0]),
            )
            fixed_to_source_uv_um = np.stack(
                (u0 + fixed_to_source[0] * pitch_um, v0 + fixed_to_source[1] * pitch_um)
            ).astype(np.float32)
            source_to_fixed_uv_um = np.stack(
                (u0 + source_to_fixed[0] * pitch_um, v0 + source_to_fixed[1] * pitch_um)
            ).astype(np.float32)
            fixed_to_source_uv_um_float64 = np.stack(
                (u0 + fixed_to_source[0].astype(np.float64) * pitch_um,
                 v0 + fixed_to_source[1].astype(np.float64) * pitch_um)
            )
            source_to_fixed_uv_um_float64 = np.stack(
                (u0 + source_to_fixed[0].astype(np.float64) * pitch_um,
                 v0 + source_to_fixed[1].astype(np.float64) * pitch_um)
            )
            arrays = {
                "velocity_uv_um": velocity_um.astype(np.float32),
                "velocity_xy_px": maps["velocity_xy_px"],
                "fixed_to_source_map": fixed_to_source,
                "source_to_fixed_map": source_to_fixed,
                "fixed_to_source_map_uv_um": fixed_to_source_uv_um,
                "source_to_fixed_map_uv_um": source_to_fixed_uv_um,
                "fixed_to_source_map_uv_um_float64_from_effective_map": fixed_to_source_uv_um_float64,
                "source_to_fixed_map_uv_um_float64_from_effective_map": source_to_fixed_uv_um_float64,
                "fixed_map_domain_mask": fixed_valid,
                "source_map_domain_mask": source_valid,
                "fixed_clean_tissue_mask": fixed_tissue.copy(),
                "source_scalar_clean": bilinear_sample_scalar(scalar, source_to_fixed),
                "source_annotation": source_labels,
                "source_clean_tissue_mask": source_tissue,
                "source_ccf_ap_dv_ml_um": source_ccf_um,
            }
            parameters = {
                "coordinate_contract": {
                    "map_order": "absolute pixel-centre (x,y), align_corners=True",
                    "fixed_to_source": "point map A o exp(v)",
                    "source_to_fixed": "pullback exp(-v) o inverse(A)",
                    "scalar_interpolation": "bilinear-zero",
                    "annotation_interpolation": "nearest-integer-zero",
                    "nearest_half_tie_rule": "numpy.rint ties-to-even, matching torch.round",
                    "reflection": False,
                    "physical_velocity_unit": "um",
                },
                "pixel_pitch_um": pitch_um,
                "D_um": D_um,
                "clean_tissue_span_u_v_um": [u_span, v_span],
                "marginal_raster_support_identity_bypass": marginal_support,
                "ordinary_requested_identifiability_threshold_pixels": ordinary_minimum,
                "accepted_attempt_index": attempt,
                "accepted_attempt": entry,
            }
            return arrays, parameters, logs
    raise ValueError("no G1 realization passed every predeclared topology, cycle, displacement, and FOV gate")


def _smooth_field(shape: tuple[int, int], rng: np.random.Generator, sigma: float) -> np.ndarray:
    value = ndimage.gaussian_filter(rng.normal(size=shape), sigma=sigma, mode="reflect")
    value -= value.mean()
    return (value / max(float(value.std()), 1e-7)).astype(np.float32)


def _coarse_field(
    shape: tuple[int, int], rng: np.random.Generator, grid_shape: tuple[int, int] = (4, 4)
) -> np.ndarray:
    coarse = rng.normal(size=grid_shape).astype(np.float32)
    y = np.linspace(0, grid_shape[0] - 1, shape[0], dtype=np.float32)
    x = np.linspace(0, grid_shape[1] - 1, shape[1], dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    field = ndimage.map_coordinates(coarse, (yy, xx), order=3, mode="nearest", prefilter=True)
    field -= field.mean()
    return (field / max(float(field.std()), 1e-7)).astype(np.float32)


def _label_conditioned_appearance(
    normalized: np.ndarray,
    labels: np.ndarray,
    tissue: np.ndarray,
    config: dict[str, object],
    attempt: int,
) -> tuple[np.ndarray, dict[str, object]]:
    region_ids = np.unique(labels[tissue])
    means_rng = _rng(config, "g2", "label-means", attempt)
    means = means_rng.uniform(*config["g2"]["label_mean_range"], len(region_ids)).astype(np.float32)
    anti_shortcut = bool(_rng(config, "g2", "label-anti-shortcut-enable", attempt).random() < 0.5)
    return_fraction = (
        float(_rng(config, "g2", "label-anti-shortcut-parameter", attempt).uniform(0.5, 1.0))
        if anti_shortcut else 0.0
    )
    global_mean = float(normalized[tissue].mean())
    effective_means = means * (1.0 - return_fraction) + global_mean * return_fraction
    standard_deviations = _rng(config, "g2", "label-standard-deviations", attempt).uniform(
        0.0, 0.08, len(region_ids)
    ).astype(np.float32)
    noise = _rng(config, "g2", "label-noise-values", attempt).normal(size=normalized.shape).astype(np.float32)
    rendered = np.zeros(normalized.shape, np.float32)
    for region_id, mean, standard_deviation in zip(region_ids, effective_means, standard_deviations):
        selected = tissue & (labels == region_id)
        rendered[selected] = mean + standard_deviation * noise[selected]
    blur_sigma = float(_rng(config, "g2", "label-postrender-blur", attempt).uniform(0.5, 1.5))
    rendered = ndimage.gaussian_filter(rendered, blur_sigma, mode="nearest").astype(np.float32)
    return np.clip(rendered, 0.0, 1.0), {
        "region_ids": [int(value) for value in region_ids],
        "sampled_region_means": [float(value) for value in means],
        "effective_region_means": [float(value) for value in effective_means],
        "region_standard_deviations": [float(value) for value in standard_deviations],
        "anti_boundary_shortcut_return_to_global_mean": anti_shortcut,
        "return_fraction": return_fraction,
        "global_tissue_mean": global_mean,
        "postrender_blur_sigma_pixels": blur_sigma,
        "ontology_coarsening": None,
    }


def _resize_axis(image: np.ndarray, fy: float, fx: float) -> np.ndarray:
    height, width = image.shape
    low = ndimage.zoom(image, (1.0 / fy, 1.0 / fx), order=1, mode="nearest", prefilter=False)
    y = np.linspace(0, low.shape[0] - 1, height, dtype=np.float32)
    x = np.linspace(0, low.shape[1] - 1, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return ndimage.map_coordinates(low, (yy, xx), order=1, mode="nearest", prefilter=False).astype(np.float32)


def _g2_attempt(
    g1_arrays: dict[str, np.ndarray], config: dict[str, object], attempt: int
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    scalar = g1_arrays["source_scalar_clean"]
    labels = g1_arrays["source_annotation"]
    tissue = g1_arrays["source_clean_tissue_mask"]
    g2 = config["g2"]
    stage_rng = lambda field: _rng(config, "g2", field, attempt)
    normalized = robust_clean_normalization(scalar, tissue)
    source_values = scalar[tissue].astype(np.float64)
    if not len(source_values):
        normalization = {
            "method": "empty-tissue",
            "lower": None,
            "upper": None,
            "tissue_pixel_count": 0,
        }
    elif len(source_values) >= 256:
        normalization = {
            "method": "clean-tissue-q01-q99",
            "lower": float(np.quantile(source_values, 0.01)),
            "upper": float(np.quantile(source_values, 0.99)),
            "tissue_pixel_count": int(len(source_values)),
        }
    else:
        normalization = {
            "method": "clean-tissue-min-max-fallback",
            "lower": float(source_values.min()),
            "upper": float(source_values.max()),
            "tissue_pixel_count": int(len(source_values)),
        }
    identity = bool(
        not tissue.any()
        or stage_rng("identity").random() < float(g2["identity_probability"])
    )
    if identity:
        family, alpha = "identity", 0.0
        appearance = normalized.copy()
        label_parameters = {"present": False}
    else:
        family_index = int(
            stage_rng("source-family").choice(3, p=np.asarray(g2["source_family_probabilities"]))
        )
        family = ("template-derived", "label-conditioned", "template-label-mixture")[family_index]
        alpha = 0.0 if family_index == 0 else (1.0 if family_index == 1 else _uniform(stage_rng("mixture-alpha"), g2["mixture_alpha"]))
        if alpha > 0.0:
            label_image, label_parameters = _label_conditioned_appearance(
                normalized, labels, tissue, config, attempt
            )
            appearance = ((1.0 - alpha) * normalized + alpha * label_image).astype(np.float32)
        else:
            appearance = normalized.copy()
            label_parameters = {"present": False}
    parameters = {
        "identity_bypass": identity,
        "source_family": family,
        "mixture_alpha": alpha,
        "normalization": normalization,
        "label_conditioned": label_parameters,
        "operator_order": [],
    }
    if not identity:
        polarity = bool(stage_rng("polarity").random() < float(g2["polarity_probability"]))
        gamma = _log_uniform(stage_rng("gamma"), g2["gamma"])
        gain = _log_uniform(stage_rng("gain"), g2["gain"])
        offset = _uniform(stage_rng("offset"), g2["offset"])
        bias_std = _uniform(stage_rng("bias"), g2["bias_std"])
        blur = bool(stage_rng("blur-enable").random() < float(g2["blur_probability"]))
        blur_sigma = _uniform(stage_rng("blur-parameter"), g2["blur_sigma_px"]) if blur else 0.0
        resolution = bool(stage_rng("resolution-enable").random() < float(g2["resolution_probability"]))
        factors = stage_rng("resolution-parameters").uniform(*g2["resolution_factor"], 2) if resolution else np.ones(2)
        noise = bool(stage_rng("noise-enable").random() < float(g2["noise_probability"]))
        noise_std = _uniform(stage_rng("noise-parameter"), g2["noise_std"]) if noise else 0.0
        if polarity:
            appearance[tissue] = 1.0 - appearance[tissue]
        appearance[tissue] = np.power(np.clip(appearance[tissue], 0, 1), gamma)
        bias = np.exp(bias_std * _coarse_field(appearance.shape, stage_rng("bias-field"), (4, 4)))
        appearance[tissue] = appearance[tissue] * gain * bias[tissue] + offset
        if blur_sigma:
            appearance = ndimage.gaussian_filter(appearance, blur_sigma, mode="nearest").astype(np.float32)
        if resolution:
            appearance = _resize_axis(appearance, float(factors[0]), float(factors[1]))
        if noise_std:
            noise_values = stage_rng("noise-values").normal(0, noise_std, appearance.shape).astype(np.float32)
            appearance[tissue] += noise_values[tissue]
        parameters.update(
            polarity={"present": polarity}, gamma={"present": True, "value": gamma},
            gain={"present": True, "value": gain}, offset={"present": True, "value": offset},
            multiplicative_bias={"present": bias_std > 0, "standard_deviation": bias_std, "base_grid": [4, 4]},
            gaussian_blur={"present": blur, "sigma_pixels": blur_sigma},
            anisotropic_resolution={"present": resolution, "downsample_factor_y_x": factors.tolist()},
            additive_noise={"present": noise, "standard_deviation": noise_std},
        )
        parameters["operator_order"] = ["polarity", "gamma", "gain-offset-bias", "blur", "anisotropic-resolution", "noise"]
    else:
        parameters.update(
            polarity={"present": False}, gamma={"present": False, "value": None},
            gain={"present": False, "value": None}, offset={"present": False, "value": None},
            multiplicative_bias={"present": False, "standard_deviation": None, "base_grid": [4, 4]},
            gaussian_blur={"present": False, "sigma_pixels": None},
            anisotropic_resolution={"present": False, "downsample_factor_y_x": None},
            additive_noise={"present": False, "standard_deviation": None},
        )
    appearance = np.clip(appearance, 0, 1).astype(np.float32)
    base = _uniform(stage_rng("background-base"), g2["background_base"])
    field_std = _uniform(stage_rng("background-field-parameter"), g2["background_field_std"])
    noise_std_bg = _uniform(stage_rng("background-noise-parameter"), g2["background_noise_std"])
    background = base + field_std * _smooth_field(appearance.shape, stage_rng("background-field"), max(1.0, max(appearance.shape) / 12))
    background += stage_rng("background-noise").normal(0, noise_std_bg, appearance.shape)
    background = np.clip(background, 0, 1).astype(np.float32)
    artifact_enabled = bool(
        not identity
        and stage_rng("artifact-enable").random() < float(g2["artifact_probability"])
    )
    artifact = np.zeros(tissue.shape, bool)
    if artifact_enabled and tissue.any():
        fraction = _log_uniform(stage_rng("artifact-parameter"), g2["artifact_fraction"])
        artifact_field = _smooth_field(tissue.shape, stage_rng("artifact-field"), 1.0)
        threshold = np.quantile(artifact_field[tissue], 1.0 - fraction)
        artifact = tissue & (artifact_field >= threshold)
        value = float(stage_rng("artifact-value").uniform())
        appearance[artifact] = value
    else:
        fraction, value = 0.0, None
    pre_damage = np.where(tissue, appearance, background).astype(np.float32)
    values = appearance[tissue].astype(np.float64)
    spread = float(np.quantile(values, 0.99) - np.quantile(values, 0.01)) if len(values) else 0.0
    std = float(values.std()) if len(values) else 0.0
    information_passed = spread >= float(g2["minimum_q99_q01"]) and std >= float(g2["minimum_std"])
    parameters.update(
        background={"base": base, "smooth_field_standard_deviation": field_std, "pixel_noise_standard_deviation": noise_std_bg},
        appearance_artifact={"present": artifact_enabled, "target_fraction": fraction, "value": value},
        information_metrics={"q99_minus_q01": spread, "standard_deviation": std, "accepted": information_passed},
        field_stream_seed_uint64=_seed_record(
            config, "g2", [
                "identity", "source-family", "mixture-alpha", "label-means", "label-standard-deviations",
                "label-noise-values", "label-anti-shortcut-enable", "label-anti-shortcut-parameter",
                "label-postrender-blur", "polarity", "gamma", "gain", "offset",
                "bias", "bias-field", "blur-enable", "blur-parameter", "resolution-enable", "resolution-parameters",
                "noise-enable", "noise-parameter", "noise-values", "background-base", "background-field-parameter",
                "background-field", "background-noise-parameter", "background-noise", "artifact-enable", "artifact-parameter",
                "artifact-field", "artifact-value",
            ], attempt
        ),
    )
    return {
        "normalized_source_scalar": normalized,
        "pre_damage_tissue_appearance": appearance,
        "acquired_background": background,
        "pre_damage_image": pre_damage,
        "g2_appearance_artifact_mask": artifact,
    }, parameters


def _g2(
    g1_arrays: dict[str, np.ndarray], config: dict[str, object]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rejection_attempts = []
    for attempt in range(int(config["maximum_g2_attempts"])):
        arrays, parameters = _g2_attempt(g1_arrays, config, attempt)
        marginal_support = int(g1_arrays["source_clean_tissue_mask"].sum()) < max(
            int(config["ordinary_minimum_clean_brain_pixels_floor"]),
            int(
                math.ceil(
                    float(config["ordinary_minimum_clean_brain_fraction"])
                    * g1_arrays["source_clean_tissue_mask"].size
                )
            ),
        )
        accepted = bool(
            parameters["information_metrics"]["accepted"]
            or config["synthetic_stratum"] == "low-information-stress"
            or marginal_support
        )
        rejection_attempts.append(
            {
                "attempt_index": attempt,
                "source_family": parameters["source_family"],
                "information_metrics": copy.deepcopy(parameters["information_metrics"]),
                "accepted": accepted,
            }
        )
        if accepted:
            parameters["marginal_raster_support_information_bypass"] = marginal_support
            parameters["accepted_attempt_index"] = attempt
            parameters["rejection_attempts"] = rejection_attempts
            return arrays, parameters
    raise ValueError("G2 realization failed all deterministic information-content rejection attempts")


def _ellipse(shape: tuple[int, int], cx: float, cy: float, rx: float, ry: float, angle: float) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    dx, dy = x - cx, y - cy
    c, s = math.cos(angle), math.sin(angle)
    a, b = c * dx + s * dy, -s * dx + c * dy
    return (a / max(rx, 1.0)) ** 2 + (b / max(ry, 1.0)) ** 2 <= 1.0


def _polygon_mask(shape: tuple[int, int], vertices_xy: np.ndarray) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]
    inside = np.zeros(shape, bool)
    x0, y0 = vertices_xy[-1]
    for x1, y1 in vertices_xy:
        crossing = ((y1 > y) != (y0 > y)) & (
            x < (x0 - x1) * (y - y1) / (y0 - y1 + np.finfo(float).eps) + x1
        )
        inside ^= crossing
        x0, y0 = x1, y1
    return inside


def _polyline_mask(
    shape: tuple[int, int], control_points_xy: np.ndarray, half_width: float
) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]
    minimum_squared = np.full(shape, np.inf)
    for start, end in zip(control_points_xy[:-1], control_points_xy[1:]):
        vx, vy = end - start
        denominator = max(float(vx * vx + vy * vy), np.finfo(float).eps)
        t = np.clip(((x - start[0]) * vx + (y - start[1]) * vy) / denominator, 0.0, 1.0)
        squared = (x - start[0] - t * vx) ** 2 + (y - start[1] - t * vy) ** 2
        minimum_squared = np.minimum(minimum_squared, squared)
    return minimum_squared <= half_width**2


def _damage_event(
    kind: str, tissue: np.ndarray, config: dict[str, object], slot: int, attempt: int
) -> tuple[np.ndarray, str, dict[str, object]]:
    rng = _rng(config, "g3", f"event-{slot}-parameters", attempt)
    yx = np.argwhere(tissue)
    height, width = tissue.shape
    target_fraction = _log_uniform(rng, config["g3"]["affected_fraction"])
    radius = max(1.0, math.sqrt(target_fraction * int(tissue.sum()) / math.pi))
    angle = float(rng.uniform(-np.pi, np.pi))
    geometry: dict[str, object] = {}
    if kind == "boundary-bite-or-missing-cortex":
        boundary = tissue & ~ndimage.binary_erosion(tissue)
        candidates = np.argwhere(boundary)
        cy, cx = candidates[int(rng.integers(len(candidates)))]
        mask = _ellipse(tissue.shape, cx, cy, radius * 1.6, radius, angle) & tissue
        category = "physical_loss"
        geometry = {"primitive": "boundary-centred ellipse", "radius_xy_px": [radius * 1.6, radius]}
    elif kind == "internal-hole":
        cy, cx = yx[int(rng.integers(len(yx)))]
        mask = _ellipse(tissue.shape, cx, cy, radius, radius * float(rng.uniform(0.6, 1.4)), angle) & tissue
        category = "physical_loss"
        geometry = {"primitive": "internal ellipse", "radius_px": radius}
    elif kind == "tear-or-crack":
        cy, cx = yx[int(rng.integers(len(yx)))]
        length = max(3.0, 2.0 * radius)
        width_px = max(1.0, float(rng.uniform(0.01, 0.04)) * min(height, width))
        count = int(rng.integers(2, 6))
        along = np.linspace(-length, length, count)
        perpendicular = rng.normal(0.0, max(0.5, width_px), count)
        control_points = np.stack(
            (
                cx + along * math.cos(angle) - perpendicular * math.sin(angle),
                cy + along * math.sin(angle) + perpendicular * math.cos(angle),
            ),
            axis=-1,
        )
        mask = _polyline_mask(tissue.shape, control_points, width_px / 2) & tissue
        category = "physical_loss"
        geometry = {"primitive": "polyline", "control_points_xy": control_points.tolist(), "width_px": width_px}
    elif kind == "blackout-or-occluding-polygon":
        cy, cx = yx[int(rng.integers(len(yx)))]
        count = int(rng.integers(4, 8))
        vertex_angles = angle + np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        vertex_radius = radius * rng.uniform(0.7, 1.4, count)
        vertices = np.stack((cx + vertex_radius * np.cos(vertex_angles), cy + vertex_radius * np.sin(vertex_angles)), axis=-1)
        mask = _polygon_mask(tissue.shape, vertices) & tissue
        category = "occlusion"
        geometry = {"primitive": "irregular convex polygon", "vertices_xy": vertices.tolist()}
    else:
        cy, cx = yx[int(rng.integers(len(yx)))]
        yy, xx = np.mgrid[:height, :width]
        across = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
        mask = (np.abs(across) <= max(1.0, radius / 2)) & tissue
        category = "appearance_artifact"
        geometry = {"primitive": "fold-like straight strip", "half_width_px": max(1.0, radius / 2)}
    return mask, category, {
        "type": kind,
        "category": category,
        "target_tissue_fraction": target_fraction,
        "changed_pixel_count": int(mask.sum()),
        "center_xy": [float(cx), float(cy)],
        "angle_rad": angle,
        "geometry": geometry,
        "field_stream_seed_uint64": _seed_hex(
            derive_field_seed(
                config["root_seed"], config["split"], config["sample_index"],
                "g3", f"event-{slot}-parameters", attempt,
            )
        ),
    }


def _g3(
    g1_arrays: dict[str, np.ndarray], g2_arrays: dict[str, np.ndarray], config: dict[str, object]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    tissue = g1_arrays["source_clean_tissue_mask"]
    map_valid = g1_arrays["source_map_domain_mask"]
    g3 = config["g3"]
    marginal_support = int(tissue.sum()) < max(
        int(config["ordinary_minimum_clean_brain_pixels_floor"]),
        int(
            math.ceil(
                float(config["ordinary_minimum_clean_brain_fraction"])
                * tissue.size
            )
        ),
    )
    eligible = int(tissue.sum()) >= int(g3["disable_damage_below_pixels"])
    kinds = (
        "boundary-bite-or-missing-cortex", "internal-hole", "tear-or-crack",
        "blackout-or-occluding-polygon", "fold-like-bright-or-doubled-strip",
    )
    rejection_attempts = []
    for damage_attempt in range(32):
        event_count = int(
            _rng(config, "g3", "event-count", damage_attempt).choice(
                3, p=np.asarray(g3["event_count_probabilities"])
            )
        ) if eligible else 0
        physical = np.zeros(tissue.shape, bool)
        occlusion = np.zeros(tissue.shape, bool)
        damage_artifact = np.zeros(tissue.shape, bool)
        events = []
        events_valid = True
        for slot in range(event_count):
            kind = kinds[int(_rng(config, "g3", f"event-{slot}-type", damage_attempt).integers(len(kinds)))]
            mask, category, receipt = _damage_event(kind, tissue, config, slot, damage_attempt)
            mask &= ~(physical | occlusion | damage_artifact)
            receipt["changed_pixel_count"] = int(mask.sum())
            receipt["type_seed_uint64"] = _seed_hex(
                derive_field_seed(
                    config["root_seed"], config["split"], config["sample_index"],
                    "g3", f"event-{slot}-type", damage_attempt,
                )
            )
            events_valid &= bool(mask.any())
            if category == "physical_loss":
                physical |= mask
            elif category == "occlusion":
                occlusion |= mask
            else:
                damage_artifact |= mask
            events.append(receipt)
        occlusion &= ~physical
        appearance_artifact = (
            g2_arrays["g2_appearance_artifact_mask"] | damage_artifact
        ) & ~physical & ~occlusion
        damage = physical | occlusion | appearance_artifact
        footprint = tissue & ~physical
        missing = physical
        artifact = occlusion | appearance_artifact
        visible = tissue & ~(missing | artifact)
        observation_invalid = tissue & damage
        source_valid = map_valid & visible
        fixed_valid = (
            g1_arrays["fixed_map_domain_mask"]
            & g1_arrays["fixed_clean_tissue_mask"]
            & nearest_sample_labels(source_valid.astype(np.uint8), g1_arrays["fixed_to_source_map"]).astype(bool)
        )
        union_fraction = float(damage.sum() / max(int(tissue.sum()), 1))
        minimum_visible = max(
            int(g3["minimum_visible_pixels"]),
            int(math.ceil(float(g3["minimum_visible_fraction"]) * tissue.sum())),
        )
        gates = {
            "partition_exact": bool(
                not np.any(missing & artifact)
                and np.array_equal(visible | missing | artifact, tissue)
            ),
            "union_damage_fraction": union_fraction,
            "maximum_union_damage_fraction_passed": union_fraction <= float(g3["maximum_union_damage_fraction"]),
            "visible_pixel_count": int(visible.sum()),
            "minimum_visible_pixels": minimum_visible,
            "minimum_visible_passed": int(visible.sum()) >= minimum_visible,
            "declared_events_change_pixels": events_valid,
        }
        accepted = marginal_support or config["synthetic_stratum"] != "ordinary" or all(
            gates[key]
            for key in (
                "partition_exact", "maximum_union_damage_fraction_passed",
                "minimum_visible_passed", "declared_events_change_pixels",
            )
        )
        rejection_attempts.append(
            {
                "attempt_index": damage_attempt,
                "event_count_seed_uint64": _seed_hex(
                    derive_field_seed(
                        config["root_seed"], config["split"], config["sample_index"],
                        "g3", "event-count", damage_attempt,
                    )
                ),
                "event_count": event_count,
                "events": copy.deepcopy(events),
                "gates": copy.deepcopy(gates),
                "accepted": bool(accepted),
            }
        )
        if accepted:
            break
    else:
        raise ValueError("G3 realization failed all deterministic damage/visibility rejection attempts")
    image = g2_arrays["pre_damage_image"].copy()
    image[physical] = g2_arrays["acquired_background"][physical]
    image[occlusion] = 0.0
    image[damage_artifact] = np.clip(image[damage_artifact] + 0.45, 0, 1)
    arrays = {
        "damaged_acquired_image": image.astype(np.float32),
        "physical_loss_mask": physical,
        "occlusion_mask": occlusion,
        "appearance_artifact_mask": appearance_artifact,
        "damage_mask": damage,
        "observable_footprint_mask": footprint,
        "observation_invalid_mask": observation_invalid,
        "visible_mask": visible,
        "missing_mask": missing,
        "artifact_mask": artifact,
        "loss_valid_mask": visible,
        "source_valid_correspondence_mask": source_valid,
        "fixed_valid_correspondence_mask": fixed_valid,
    }
    parameters = {
        "damage_eligible": eligible,
        "damage_disabled_reason": None if eligible else "source clean tissue below 128 pixels",
        "marginal_raster_support_visibility_bypass": marginal_support,
        "event_count": event_count,
        "events": events,
        "accepted_attempt_index": damage_attempt,
        "rejection_attempts": rejection_attempts,
        "exclusive_category_precedence": [
            "physical_loss", "occlusion", "appearance_artifact", "visible"
        ],
        "gates": gates,
        "field_stream_seed_uint64": _seed_record(
            config, "g3", ["event-count"] + [f"event-{slot}-{suffix}" for slot in range(2) for suffix in ("type", "parameters")], damage_attempt
        ),
    }
    return arrays, parameters


def _outline(
    g3_arrays: dict[str, np.ndarray], config: dict[str, object]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    mode = config["outline_mode"]
    footprint = g3_arrays["observable_footprint_mask"]
    marginal_support = int(footprint.sum()) < max(
        int(config["ordinary_minimum_clean_brain_pixels_floor"]),
        int(
            math.ceil(
                float(config["ordinary_minimum_clean_brain_fraction"])
                * footprint.size
            )
        ),
    )
    attempts = []
    for attempt in range(int(config["g3"]["maximum_outline_attempts"])):
        result = smart_brush_input(
            g3_arrays["damaged_acquired_image"],
            g3_arrays["observable_footprint_mask"],
            _rng(config, "outline", "perturbation", attempt),
            mode,
            morphology_radius_px=min(5, max(1, int(round(0.02 * min(g3_arrays["damaged_acquired_image"].shape))))),
            jitter_amplitude_px=max(0.25, 0.01 * min(g3_arrays["damaged_acquired_image"].shape)),
        )
        iou = result["outline_quality_iou"]
        accepted = marginal_support or mode != IMPERFECT_OUTLINE or (
            float(config["g3"]["imperfect_iou"][0]) <= float(iou) <= float(config["g3"]["imperfect_iou"][1])
        )
        attempts.append({
            "attempt_index": attempt,
            "field_stream_seed_uint64": _seed_hex(
                derive_field_seed(config["root_seed"], config["split"], config["sample_index"], "outline", "perturbation", attempt)
            ),
            "quality_iou": iou,
            "accepted": accepted,
        })
        if accepted:
            break
    else:
        raise ValueError("imperfect outline did not meet its predeclared IoU gate")
    input_image = result["input_image"].astype(np.float32)
    outline_mask = result["input_outline_mask"].astype(bool)
    black_exterior = bool(not result["outline_available"] or np.all(input_image[~outline_mask] == 0.0))
    if result["outline_available"] and not black_exterior:
        raise ValueError("available outline mode failed exact black-exterior gate")
    if mode == ABSENT_OUTLINE and not np.array_equal(input_image, g3_arrays["damaged_acquired_image"]):
        raise ValueError("absent outline must retain the acquired background exactly")
    arrays = {"model_input_image": input_image, "input_outline_mask": outline_mask}
    parameters = {
        "mode": mode,
        "outline_plan": copy.deepcopy(config["outline_plan"]),
        "outline_available": bool(result["outline_available"]),
        "quality_iou": result["outline_quality_iou"],
        "parameters": result["parameters"],
        "accepted_attempt_index": attempts[-1]["attempt_index"],
        "rejection_attempts": attempts,
        "black_exterior_exact": black_exterior if result["outline_available"] else None,
        "marginal_raster_support_outline_bypass": marginal_support,
        "automatic_perfect_mask_substitution": False,
        "truth_mask_used_as_model_validity_target": False,
    }
    return arrays, parameters


def _finite_identity(parent: dict[str, object]) -> dict[str, object]:
    return {
        key: parent[key]
        for key in (
            "support_index_sha256", "plane_realization_id", "finite_plane_geometry_sha256",
            "rendered_artifacts_sha256", "finite_plane_render_id",
        )
    }


def _stage_identity(name: str, recipe: object, arrays: dict[str, np.ndarray]) -> tuple[str, str]:
    recipe_id = _payload_sha256({"schema": f"anatomy-tracker.{name}-recipe/v1", "recipe": recipe})
    realization_id = _payload_sha256(
        {"schema": f"anatomy-tracker.{name}-realization/v1", "recipe_id": recipe_id, "arrays": _array_receipts(arrays)}
    )
    return recipe_id, realization_id


def _resolved_config(
    parent: dict[str, object], root_seed: int | str | None, sample_index: int | None, synthetic_stratum: str,
    outline_mode: str | None, overrides: dict[str, object] | None,
) -> dict[str, object]:
    if parent["schema_version"] != FINITE_RENDER_SCHEMA:
        raise ValueError("synthetic generator requires a finite arbitrary-plane parent")
    if (outline_mode is not None and outline_mode not in SMART_BRUSH_MODES) or synthetic_stratum not in SYNTHETIC_STRATA:
        raise ValueError("outline mode or synthetic stratum is not declared")
    config = _merge_config(_DEFAULT_CONFIG, overrides)
    config.update(
        split=parent["split"],
        root_seed=_seed_hex(parent["root_seed"] if root_seed is None else root_seed),
        sample_index=int(parent["sample_index"] if sample_index is None else sample_index),
        synthetic_stratum=synthetic_stratum,
        seed_encoding=SEED_ENCODING,
    )
    if config["split"] not in {"train", "development"} or config["sample_index"] < 0:
        raise ValueError("synthetic generation is restricted to nonnegative train/development samples")
    plan_seed = derive_field_seed(
        config["root_seed"], config["split"], config["sample_index"], "outline", "outline-plan", 0
    )
    if outline_mode is None:
        outline_mode = SMART_BRUSH_MODES[int(np.random.Generator(np.random.PCG64(plan_seed)).integers(3))]
        assignment = "isolated equal-probability outline-plan draw"
    else:
        assignment = "explicit paired-counterfactual assignment"
    config["outline_mode"] = outline_mode
    config["outline_plan"] = {
        "assignment": assignment,
        "field_stream_seed_uint64": _seed_hex(plan_seed),
        "development_probabilities": {mode: 1.0 / 3.0 for mode in SMART_BRUSH_MODES},
        "selected_mode": outline_mode,
    }
    return config


def _generate(
    parent: dict[str, object],
    support: dict[str, object],
    config: dict[str, object],
    *,
    finite_parent_generator_source_commit: str | None,
) -> dict[str, object]:
    verify_finite_arbitrary_plane_render(
        parent,
        support,
        generator_source_commit=finite_parent_generator_source_commit,
    )
    if parent["support_index_sha256"] != support["support_index_sha256"]:
        raise ValueError("finite parent and support index do not match")
    brain_pixel_count = int(parent["acceptance_contract"]["brain_pixel_count"])
    requested_threshold = int(
        parent["acceptance_contract"]["minimum_brain_pixels"]
    )
    support_supervision = {
        "continuous_plane_sample_retained": True,
        "pose_redrawn_for_raster_support": False,
        "raster_brain_pixel_count": brain_pixel_count,
        "requested_identifiability_threshold_pixels": requested_threshold,
        "raster_support_meets_requested_identifiability_threshold": bool(
            brain_pixel_count >= requested_threshold
        ),
        "marginal_support_generation_policy": (
            "identity deformation, damage/information/outline gates bypassed only as explicitly "
            "recorded; point and dense loss eligibility is decided by the curriculum row"
        ),
    }
    g1_arrays, g1_parameters, g1_logs = _g1(parent, support, config)
    g2_arrays, g2_parameters = _g2(g1_arrays, config)
    g3_arrays, g3_parameters = _g3(g1_arrays, g2_arrays, config)
    outline_arrays, outline_parameters = _outline(g3_arrays, config)
    arrays = {**g1_arrays, **g2_arrays, **g3_arrays, **outline_arrays}
    g1_recipe, g1_realization = _stage_identity(
        "deformation", {"config": config["g1"], "parameters": g1_parameters, "rejections": g1_logs}, g1_arrays
    )
    g2_recipe, g2_realization = _stage_identity("appearance", {"config": config["g2"], "parameters": g2_parameters}, g2_arrays)
    g3_recipe, g3_realization = _stage_identity("damage", {"config": config["g3"], "parameters": g3_parameters}, g3_arrays)
    outline_recipe, outline_realization = _stage_identity("outline", outline_parameters, outline_arrays)
    parent_identity = _finite_identity(parent)
    provenance = copy.deepcopy(parent["provenance"])
    implementation = {
        "source_path": "training/arbitrary_plane_synthetic_generator.py",
        "loaded_source_sha256": {name: _file_sha256(path) for name, path in _SOURCE_FILES.items()},
        "dependency_contract_versions": {
            "finite_parent": FINITE_RENDER_SCHEMA,
            "ops": ARBITRARY_PLANE_SYNTHETIC_OPS_VERSION,
            "observation": "procedural-appearance-damage-outline-primitives/v1",
        },
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    model_independence = {
        "learned_checkpoint_dependencies": [],
        "previous_model_dependencies": [],
        "pretrained_feature_dependencies": [],
        "initialization": "random seed streams only; no learned initialization",
    }
    array_receipts = _array_receipts(arrays)
    synthetic_artifacts_sha256 = _payload_sha256(array_receipts)
    artifact = {
        "schema_version": SYNTHETIC_SCHEMA,
        "generator_algorithm": SYNTHETIC_ALGORITHM,
        "split": config["split"],
        "root_seed": config["root_seed"],
        "sample_index": config["sample_index"],
        "synthetic_stratum": config["synthetic_stratum"],
        "finite_parent_identity": parent_identity,
        "finite_parent_receipt": finite_render_receipt(parent),
        "finite_parent": copy.deepcopy(parent),
        "support_index_sha256": support["support_index_sha256"],
        "provenance": provenance,
        "provenance_sha256": _payload_sha256(provenance),
        "support_supervision": support_supervision,
        "generator": {
            "implementation": implementation,
            "implementation_sha256": _payload_sha256(implementation),
            "resolved_config": copy.deepcopy(config),
            "resolved_config_sha256": _payload_sha256(config),
            **model_independence,
            "model_independence_sha256": _payload_sha256(model_independence),
        },
        "g1": {
            "parameters": g1_parameters,
            "rejection_attempts": g1_logs,
            "deformation_recipe_id": g1_recipe,
            "deformation_realization_id": g1_realization,
        },
        "g2": {
            "parameters": g2_parameters,
            "appearance_recipe_id": g2_recipe,
            "appearance_realization_id": g2_realization,
        },
        "g3": {
            "parameters": g3_parameters,
            "damage_recipe_id": g3_recipe,
            "damage_realization_id": g3_realization,
        },
        "outline": {
            "parameters": outline_parameters,
            "outline_recipe_id": outline_recipe,
            "outline_realization_id": outline_realization,
        },
        "paired_view_group_id": _payload_sha256(
            {"parent": parent_identity, "g1": g1_realization, "g2": g2_realization, "g3": g3_realization}
        ),
        "arrays": arrays,
        "array_receipts": array_receipts,
        "synthetic_artifacts_sha256": synthetic_artifacts_sha256,
        "development_scope": {
            "status": "small deterministic generator engineering smoke only",
            "benchmark": False,
            "qualification": False,
            "final_test_access": False,
        },
    }
    complete_identity = {
        "schema_version": SYNTHETIC_SCHEMA,
        "generator_algorithm": SYNTHETIC_ALGORITHM,
        "finite_parent_identity": parent_identity,
        "implementation_sha256": artifact["generator"]["implementation_sha256"],
        "resolved_config_sha256": artifact["generator"]["resolved_config_sha256"],
        "provenance_sha256": artifact["provenance_sha256"],
        "deformation_recipe_id": g1_recipe,
        "deformation_realization_id": g1_realization,
        "appearance_recipe_id": g2_recipe,
        "appearance_realization_id": g2_realization,
        "damage_recipe_id": g3_recipe,
        "damage_realization_id": g3_realization,
        "outline_recipe_id": outline_recipe,
        "outline_realization_id": outline_realization,
        "paired_view_group_id": artifact["paired_view_group_id"],
        "synthetic_artifacts_sha256": synthetic_artifacts_sha256,
    }
    artifact["synthetic_realization_id"] = _payload_sha256(complete_identity)
    artifact["synthetic_receipt_sha256"] = _payload_sha256(synthetic_realization_receipt(artifact))
    return artifact


def make_arbitrary_plane_synthetic_realization(
    finite_parent: dict[str, object],
    support_index: dict[str, object],
    *,
    root_seed: int | str | None = None,
    sample_index: int | None = None,
    synthetic_stratum: str = "ordinary",
    outline_mode: str | None = None,
    config_overrides: dict[str, object] | None = None,
    finite_parent_generator_source_commit: str | None = None,
) -> dict[str, object]:
    """Create one complete G1/G2/G3 realization and issue its final identity."""
    config = _resolved_config(
        finite_parent, root_seed, sample_index, synthetic_stratum, outline_mode, config_overrides
    )
    return _generate(
        finite_parent,
        support_index,
        config,
        finite_parent_generator_source_commit=finite_parent_generator_source_commit,
    )


def synthetic_realization_receipt(artifact: dict[str, object]) -> dict[str, object]:
    """Return the JSON-safe receipt; all dense arrays remain in the sidecar."""
    receipt = {
        key: copy.deepcopy(value)
        for key, value in artifact.items()
        if key not in {"finite_parent", "arrays", "synthetic_receipt_sha256"}
    }
    json.dumps(receipt, allow_nan=False)
    return receipt


def replay_arbitrary_plane_synthetic_realization(
    artifact: dict[str, object],
    support_index: dict[str, object],
    *,
    finite_parent_generator_source_commit: str | None = None,
) -> dict[str, object]:
    """Replay all isolated streams from the verified parent and bound config."""
    if "synthetic_realization_id" not in artifact:
        raise ValueError("incomplete synthetic artifact has no synthetic_realization_id")
    return _generate(
        artifact["finite_parent"],
        support_index,
        copy.deepcopy(artifact["generator"]["resolved_config"]),
        finite_parent_generator_source_commit=finite_parent_generator_source_commit,
    )


def verify_arbitrary_plane_synthetic_realization(
    artifact: dict[str, object],
    support_index: dict[str, object],
    *,
    finite_parent_generator_source_commit: str | None = None,
) -> None:
    """Verify provenance, arrays, mask algebra, identity, and exact deterministic replay."""
    required = {
        "schema_version", "generator_algorithm", "finite_parent", "arrays", "array_receipts",
        "synthetic_artifacts_sha256", "synthetic_realization_id", "synthetic_receipt_sha256",
    }
    if not required <= artifact.keys():
        raise ValueError("incomplete synthetic artifact cannot be verified or identified")
    if artifact["schema_version"] != SYNTHETIC_SCHEMA or artifact["generator_algorithm"] != SYNTHETIC_ALGORITHM:
        raise ValueError("synthetic schema or algorithm does not match")
    verify_finite_arbitrary_plane_render(
        artifact["finite_parent"],
        support_index,
        generator_source_commit=finite_parent_generator_source_commit,
    )
    if artifact["finite_parent_identity"] != _finite_identity(artifact["finite_parent"]):
        raise ValueError("finite parent identity does not match")
    if artifact["finite_parent_receipt"] != finite_render_receipt(artifact["finite_parent"]):
        raise ValueError("finite parent receipt does not match")
    if artifact["provenance"] != artifact["finite_parent"]["provenance"]:
        raise ValueError("synthetic provenance must preserve finite-parent subject and atlas IDs exactly")
    if artifact["provenance_sha256"] != _payload_sha256(artifact["provenance"]):
        raise ValueError("synthetic provenance hash does not match")
    model = {
        key: artifact["generator"][key]
        for key in ("learned_checkpoint_dependencies", "previous_model_dependencies", "pretrained_feature_dependencies", "initialization")
    }
    if model != {
        "learned_checkpoint_dependencies": [], "previous_model_dependencies": [], "pretrained_feature_dependencies": [],
        "initialization": "random seed streams only; no learned initialization",
    } or artifact["generator"]["model_independence_sha256"] != _payload_sha256(model):
        raise ValueError("synthetic generator is not random-only and model independent")
    dependency_keys = {
        key for key in artifact["generator"] if key.endswith("_dependencies")
    }
    if dependency_keys != {
        "learned_checkpoint_dependencies", "previous_model_dependencies", "pretrained_feature_dependencies"
    }:
        raise ValueError("synthetic generator must declare exactly three empty dependency lists")
    if artifact["generator"]["resolved_config_sha256"] != _payload_sha256(artifact["generator"]["resolved_config"]):
        raise ValueError("resolved synthetic config hash does not match")
    current_sources = {name: _file_sha256(path) for name, path in _SOURCE_FILES.items()}
    if artifact["generator"]["implementation"]["loaded_source_sha256"] != current_sources:
        raise ValueError("loaded synthetic implementation sources do not match")
    receipts = _array_receipts(artifact["arrays"])
    if artifact["array_receipts"] != receipts or artifact["synthetic_artifacts_sha256"] != _payload_sha256(receipts):
        raise ValueError("synthetic array sidecar receipts do not match")
    arrays = artifact["arrays"]
    if not np.issubdtype(arrays["source_annotation"].dtype, np.integer):
        raise ValueError("source annotation is not integer")
    tissue = arrays["source_clean_tissue_mask"]
    if not np.array_equal(tissue, arrays["source_annotation"] != 0):
        raise ValueError("source annotation/tissue consistency does not match")
    if not np.array_equal(
        arrays["visible_mask"] | arrays["missing_mask"] | arrays["artifact_mask"], tissue
    ) or np.any(arrays["missing_mask"] & arrays["artifact_mask"]):
        raise ValueError("damage supervision partition does not match")
    physical = arrays["physical_loss_mask"]
    occlusion = arrays["occlusion_mask"]
    appearance = arrays["appearance_artifact_mask"]
    if (
        np.any(physical & occlusion)
        or np.any(physical & appearance)
        or np.any(occlusion & appearance)
        or not np.array_equal(arrays["observable_footprint_mask"], tissue & ~physical)
        or not np.array_equal(arrays["observation_invalid_mask"], tissue & (physical | occlusion | appearance))
        or not np.array_equal(
            arrays["source_valid_correspondence_mask"],
            arrays["source_map_domain_mask"] & tissue & ~arrays["observation_invalid_mask"],
        )
    ):
        raise ValueError("exclusive damage or source-valid mask algebra does not match")
    expected_fixed_valid = (
        arrays["fixed_map_domain_mask"]
        & arrays["fixed_clean_tissue_mask"]
        & nearest_sample_labels(
            arrays["source_valid_correspondence_mask"].astype(np.uint8),
            arrays["fixed_to_source_map"],
        ).astype(bool)
    )
    if not np.array_equal(arrays["fixed_valid_correspondence_mask"], expected_fixed_valid):
        raise ValueError("fixed-valid correspondence mask algebra does not match")
    if artifact["outline"]["parameters"]["outline_available"]:
        if np.any(arrays["model_input_image"][~arrays["input_outline_mask"]] != 0.0):
            raise ValueError("available outline does not have exact black exterior")
    elif not np.array_equal(arrays["model_input_image"], arrays["damaged_acquired_image"]):
        raise ValueError("absent outline does not retain acquired background")
    receipt = synthetic_realization_receipt(artifact)
    if artifact["synthetic_receipt_sha256"] != _payload_sha256(receipt):
        raise ValueError("synthetic JSON receipt hash does not match")
    replayed = replay_arbitrary_plane_synthetic_realization(
        artifact,
        support_index,
        finite_parent_generator_source_commit=finite_parent_generator_source_commit,
    )
    if synthetic_realization_receipt(replayed) != receipt:
        raise ValueError("synthetic receipt does not match exact replay")
    if any(not np.array_equal(value, replayed["arrays"][name]) for name, value in arrays.items()):
        raise ValueError("synthetic arrays do not match exact replay")


__all__ = [
    "ABSENT_OUTLINE", "ACCURATE_OUTLINE", "IMPERFECT_OUTLINE", "SEED_ENCODING",
    "SYNTHETIC_ALGORITHM", "SYNTHETIC_SCHEMA", "derive_field_seed",
    "make_arbitrary_plane_synthetic_realization", "replay_arbitrary_plane_synthetic_realization",
    "synthetic_realization_receipt", "verify_arbitrary_plane_synthetic_realization",
]
