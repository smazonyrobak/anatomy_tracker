"""Fixed-canvas observations with parent-first damage and explicit lineage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_observation_v2 as observation_v2
import training.arbitrary_plane_acquisition_window_v3 as acquisition_window
import training.arbitrary_plane_deformation_gauge_v3 as deformation_gauge
import training.arbitrary_plane_legacy_chain_v3 as legacy_chain_v3
import training.arbitrary_plane_section_processing_v2 as section_processing
import training.arbitrary_plane_subject_slab_v2 as subject_slab_v2
from training.arbitrary_plane_section_processing_v2 import (
    section_processing_plan_receipt_v2,
    section_processing_render_receipt_v2,
)


_verify_section_processing_render_with_mapper_v2 = (
    legacy_chain_v3.verify_section_processing_render_v3
)


OBSERVATION_V3_SCHEMA = "anatomy-tracker.arbitrary-plane-observation/v3"
OBSERVATION_V3_ALGORITHM = (
    "parent-damage-fixed-canvas-affine-gauge-and-paired-smart-brush/v3"
)
OBSERVATION_DESCENDANT_V3_SCHEMA = "anatomy-tracker.observation-descendant/v3"
OBSERVATION_PARENT_AUTH_V3_SCHEMA = (
    "anatomy-tracker.observation-parent-authentication/v3"
)
OBSERVATION_PARENT_AUTH_V3_ALGORITHM = (
    "verify-heavy-parent-once-then-live-receipt-bind/v3"
)
MODALITIES = observation_v2.MODALITIES
DESCENDANT_MODES = observation_v2.DESCENDANT_MODES
CANONICAL_TRAINABLE_RAW_BACKGROUND_MODE = "smart-brush-absent"
RAW_BACKGROUND_EQUIVALENT_MODES = ("raw", "smart-brush-absent")
_SOURCE_ROOT = Path(__file__).parent
_SOURCE_FILES = (
    "arbitrary_plane_observation_v3.py",
    "arbitrary_plane_acquisition_window_v3.py",
    "arbitrary_plane_deformation_gauge_v3.py",
    "arbitrary_plane_deformation_primitives.py",
    "arbitrary_plane_geometry_v3.py",
    "arbitrary_plane_legacy_chain_v3.py",
    "arbitrary_plane_acquisition_v2.py",
    "arbitrary_plane_observation_v2.py",
    "arbitrary_plane_section_processing_v2.py",
)
_ARRAY_KEYS = {
    "source_scalar_canvas_float32",
    "source_label_ground_truth_canvas_int64",
    "source_tissue_ground_truth_mask",
    "source_correspondence_domain_mask",
    "source_dense_correspondence_weight_float32",
    "source_dense_correspondence_abstention_mask",
    "processed_mapped_ccf_physical_coordinates_canvas_float64",
    "processed_bilinear_domain_valid_mask",
    "processed_nearest_domain_valid_mask",
    "processed_dense_coordinate_valid_mask",
    "truth_section_pullback_map_yx_px_float64",
    "truth_section_pullback_stationary_velocity_yx_px_float64",
    "truth_section_deformation_valid_mask",
    "parent_sampling_domain_mask",
    "normalized_template_float32",
    "label_conditioned_latent_float32",
    "acquired_background_float32",
    "pre_damage_acquired_image_float32",
    "raw_acquired_image_float32",
    "physical_loss_mask",
    "occlusion_mask",
    "appearance_artifact_mask",
    "damage_union_mask",
    "observable_footprint_mask",
    "observation_invalid_mask",
    "outside_correspondence_domain_mask",
    "valid_correspondence_mask",
    "valid_correspondence_weight_float32",
}
_DESCENDANT_ARRAY_KEYS = {
    "model_input_image_float32",
    "selected_input_mask",
    "brush_mask_error_mask",
}


def _source_hashes() -> dict[str, str]:
    return {
        name: acquisition._normalized_text_sha256(_SOURCE_ROOT / name)
        for name in _SOURCE_FILES
    }


def _array_receipts(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: acquisition._array_receipt(np.asarray(array))
        for name, array in arrays.items()
    }


def _engineering_priors() -> dict[str, object]:
    priors = observation_v2._engineering_priors()
    del priors["crop"]
    priors["appearance_label_anti_shortcut"] = {
        "probability": 0.5,
        "return_fraction_range": [0.5, 1.0],
        "operator": (
            "with probability 0.5 return every sampled visible-region level by a "
            "shared fraction toward the normalized global tissue mean"
        ),
        "ontology_coarsening": None,
    }
    priors["damage"]["geometry_families"]["physical-loss"] = [
        "edge-bite",
        "internal-hole",
        "tear-stripe",
    ]
    priors["damage"]["internal_hole_component_count_range_inclusive"] = [1, 3]
    priors["damage"]["internal_hole_interior_policy"] = (
        "every proposed and retained hole pixel lies in a one-pixel erosion of the "
        "authenticated parent tissue; infeasible events are recorded without redraw"
    )
    priors["smart_brush"]["island_or_gap_count_range_inclusive"] = [0, 2]
    priors["smart_brush"]["island_or_gap_clean_tissue_fraction_range"] = [
        0.001,
        0.02,
    ]
    priors["smart_brush"]["quality_iou_policy"] = (
        "audit statistic only; the v1 [0.70,0.98] interval is explicitly superseded "
        "and never causes acceptance, rejection, clipping, or redraw"
    )
    priors["acquisition_window"] = {
        "source": "separately precomputed and verified acquisition-window v3 plan",
        "parent_conditioning": "none: pose, tissue, support, and labels are unavailable",
        "application_order": "after full-parent damage geometry and before appearance",
        "empty_or_partial_policy": "retain, record, and never reject or redraw",
    }
    return priors


def _rng(
    provenance: dict[str, object],
    stage: str,
    field: str,
    receipts: dict[str, dict[str, object]],
    attempt: int = 0,
) -> np.random.Generator:
    """Use the frozen acquisition-v2 seed tuple for every v3 observation field."""
    seed = acquisition.derive_v2_field_seed(
        provenance["root_seed_uint64"],
        provenance["split"],
        provenance["sample_index"],
        stage,
        field,
        attempt,
    )
    key = f"{stage}/{field}/attempt-{attempt}"
    receipt = {
        "derivation_function": "acquisition.derive_v2_field_seed",
        "derivation_tuple": [
            acquisition.V2_RNG_DOMAIN,
            acquisition.V2_SCHEMA,
            provenance["split"],
            acquisition._root_seed_hex(provenance["root_seed_uint64"]),
            str(provenance["sample_index"]),
            stage,
            field,
            str(attempt),
        ],
        "tuple_encoding": "uint32 big-endian UTF-8 byte-length prefix per component",
        "digest": "BLAKE2b-64",
        "person": "AP-ACQ-V2",
        "seed_endian": "unsigned big-endian uint64",
        "stage": stage,
        "field": field,
        "attempt": int(attempt),
        "seed_uint64": f"0x{seed:016x}",
        "generator": "NumPy PCG64DXSM",
    }
    if key in receipts and receipts[key] != receipt:
        raise ValueError("named RNG stream receipt collision")
    receipts[key] = receipt
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _uniform(
    provenance: dict[str, object],
    stage: str,
    field: str,
    receipts: dict[str, dict[str, object]],
    interval: list[float],
) -> float:
    return float(_rng(provenance, stage, field, receipts).uniform(*interval))


def _byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def _lineage(
    precursor: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
) -> dict[str, object]:
    provenance = precursor["provenance"]
    section = section_processing_plan["provenance"]
    return {
        "split": section["split"],
        "animal_index": section["animal_index"],
        "animal_id": acquisition._json_value(provenance["animal_id"]),
        "specimen_id": acquisition._json_value(provenance["specimen_id"]),
        "experiment_id": acquisition._json_value(provenance["experiment_id"]),
        "synthetic_animal_id": acquisition._json_value(
            subject_slab_render["synthetic_animal_id"]
        ),
        "section_index": section["section_index"],
        "section_id": acquisition._json_value(section["section_id"]),
        "identity_exclusion_from_rng": (
            "animal_id, specimen_id, experiment_id, synthetic_animal_id, section_id, "
            "and artifact IDs are authenticated outputs and never RNG inputs"
        ),
    }


def _upstream_reference(
    processed_render: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    precursor: dict[str, object],
) -> dict[str, object]:
    return {
        "v2_plane_realization_id": precursor["v2_plane_realization_id"],
        "slab_render_id": precursor["slab_render_id"],
        "section_processing_render_id": processed_render[
            "section_processing_render_id"
        ],
        "section_processing_render_receipt_sha256": acquisition._payload_sha256(
            section_processing_render_receipt_v2(processed_render)
        ),
        "section_processing_plan_id": section_processing_plan[
            "section_processing_plan_id"
        ],
        "section_processing_realization_id": section_processing_plan[
            "section_processing_realization_id"
        ],
        "synthetic_section_processing_id": section_processing_plan[
            "synthetic_section_processing_id"
        ],
        "subject_slab_render_id": subject_slab_render["subject_slab_render_id"],
    }


def _observation_parent_authentication_payload_v3(
    processed_render,
    subject_slab_render,
    section_processing_plan,
    prepared_context,
    precursor,
):
    section_receipt = section_processing_render_receipt_v2(processed_render)
    return {
        "schema_version": OBSERVATION_PARENT_AUTH_V3_SCHEMA,
        "algorithm": OBSERVATION_PARENT_AUTH_V3_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "context_reference": {
            name: prepared_context.get(name)
            for name in ("v2_context_sha256", "prepared_context_receipt_sha256")
        },
        "upstream_reference": _upstream_reference(
            processed_render,
            subject_slab_render,
            section_processing_plan,
            precursor,
        ),
        "precursor_receipt_sha256": precursor.get("receipt_sha256"),
        "subject_slab_receipt_sha256": subject_slab_render.get("receipt_sha256"),
        "section_processing_plan_receipt_sha256": section_processing_plan.get(
            "receipt_sha256"
        ),
        "section_processing_render_receipt": section_receipt,
        "section_processing_render_receipt_sha256": acquisition._payload_sha256(
            section_receipt
        ),
        "legacy_chain_adapter_v3": (
            legacy_chain_v3.adapter_receipt_v3(precursor)
            if {
                "geometry",
                "centre_plane_render_id",
                "slab_recipe_id",
                "receipt_sha256",
            }
            <= set(precursor)
            else None
        ),
    }


def observation_parent_authentication_receipt_v3(authentication):
    return {
        key: authentication[key]
        for key in (
            "schema_version",
            "algorithm",
            "implementation_source_sha256",
            "context_reference",
            "upstream_reference",
            "precursor_receipt_sha256",
            "subject_slab_receipt_sha256",
            "section_processing_plan_receipt_sha256",
            "section_processing_render_receipt",
            "section_processing_render_receipt_sha256",
            "legacy_chain_adapter_v3",
            "observation_parent_authentication_id",
        )
    }


def _make_observation_parent_authentication_v3(
    processed_render,
    subject_slab_render,
    section_processing_plan,
    prepared_context,
    precursor,
):
    payload = _observation_parent_authentication_payload_v3(
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
    )
    authentication = {
        **payload,
        "observation_parent_authentication_id": acquisition._payload_sha256(
            {"domain": OBSERVATION_PARENT_AUTH_V3_SCHEMA, **payload}
        ),
    }
    authentication["receipt_sha256"] = acquisition._payload_sha256(
        observation_parent_authentication_receipt_v3(authentication)
    )
    return authentication


def authenticate_observation_parent_v3(
    processed_render,
    subject_slab_render,
    section_processing_plan,
    prepared_context,
    precursor,
    *,
    subject_plan,
    batch_size=None,
):
    _verify_section_processing_render_with_mapper_v2(
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=None,
    )
    return _make_observation_parent_authentication_v3(
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
    )


def verify_observation_parent_authentication_v3(
    authentication,
    processed_render,
    subject_slab_render,
    section_processing_plan,
    prepared_context,
    precursor,
):
    expected = _make_observation_parent_authentication_v3(
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
    )
    if (
        authentication != expected
        or authentication.get("receipt_sha256")
        != acquisition._payload_sha256(
            observation_parent_authentication_receipt_v3(authentication)
        )
    ):
        raise ValueError("observation parent authentication receipt changed")
    return True


def _window_geometry(
    processed_render: dict[str, object],
) -> tuple[str, tuple[float, float], np.ndarray, np.ndarray, np.ndarray]:
    pose = processed_render["pose_anatomy_policy"]["pose_anatomy_reference"]
    fit = pose["centre_plane_fit"]
    design = np.asarray(
        fit["arrays"]["physical_ouv_ap_dv_ml_um_float64"], dtype=np.float64
    ).reshape(3, 3)
    if not np.isfinite(design).all():
        raise ValueError("authenticated parent design O/U/V is not finite")
    raster = processed_render["raster"]
    centre = np.asarray(raster["centre_plane_support_mask"])
    optical = np.asarray(raster["slab_brain_occupancy"])
    height, width = centre.shape
    fov = (
        float(np.linalg.norm(design[1])) * (width - 1.0) / width,
        float(np.linalg.norm(design[2])) * (height - 1.0) / height,
    )
    if (
        centre.dtype != bool
        or centre.shape != optical.shape
        or not np.issubdtype(optical.dtype, np.floating)
        or not np.isfinite(optical).all()
        or np.any(optical < 0.0)
        or min(fov) <= 0.0
    ):
        raise ValueError("authenticated parent design or support rasters are invalid")
    return (
        pose["precursor_reference"]["global_reference_grid_id"],
        (fov[0], fov[1]),
        np.ascontiguousarray(design),
        np.ascontiguousarray(centre),
        np.ascontiguousarray(optical),
    )


def _verify_plan_binding(
    plan: dict[str, object],
    provenance: dict[str, object],
    parent_shape: tuple[int, int],
    sample_index: int,
) -> None:
    acquisition_window.verify_acquisition_window_plan_v3(plan)
    expected = {"split": provenance["split"], "sample_index": sample_index}
    if (
        any(plan["provenance"][name] != value for name, value in expected.items())
        or tuple(plan["parent_shape_h_w"]) != tuple(parent_shape)
    ):
        raise ValueError(
            "precomputed acquisition-window plan split, sample, or parent shape differs"
        )


def _rotated_ellipse(
    shape: tuple[int, int],
    center_yx: tuple[float, float],
    radius_yx: tuple[float, float],
    angle_radians: float,
) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    ry, rx = radius_yx
    dy, dx = y - cy, x - cx
    along = np.cos(angle_radians) * dx + np.sin(angle_radians) * dy
    across = -np.sin(angle_radians) * dx + np.cos(angle_radians) * dy
    return (along / max(rx, 1.0)) ** 2 + (across / max(ry, 1.0)) ** 2 <= 1.0


def _stable_budget_clip(
    mask: np.ndarray, center_yx: tuple[float, float], maximum_pixels: int
) -> np.ndarray:
    if int(mask.sum()) <= maximum_pixels:
        return np.ascontiguousarray(mask, dtype=bool)
    indices = np.argwhere(mask)
    distances = np.sum(
        (indices - np.asarray(center_yx, dtype=np.float64)) ** 2, axis=1
    )
    retained = indices[np.argsort(distances, kind="stable")[:maximum_pixels]]
    clipped = np.zeros(mask.shape, dtype=bool)
    clipped[retained[:, 0], retained[:, 1]] = True
    return clipped


def _internal_hole_mask(
    tissue: np.ndarray,
    remaining: np.ndarray,
    component_count: int,
    component_rngs: list[np.random.Generator],
    radius_fraction_range: list[float],
    maximum_new_pixels: int,
) -> tuple[np.ndarray, dict[str, object]]:
    interior = remaining & scipy.ndimage.binary_erosion(tissue, iterations=1)
    proposed_components: list[np.ndarray] = []
    component_parameters = []
    proposed_union = np.zeros(tissue.shape, dtype=bool)
    low, high = radius_fraction_range
    for component_index in range(component_count):
        rng = component_rngs[component_index]
        candidates = interior & ~proposed_union
        separated = candidates & ~scipy.ndimage.binary_dilation(
            proposed_union, iterations=1
        )
        if separated.any():
            candidates = separated
        yx = np.argwhere(candidates)
        radius_y = max(1.0, tissue.shape[0] * float(rng.uniform(low, high)))
        radius_x = max(1.0, tissue.shape[1] * float(rng.uniform(low, high)))
        angle = float(rng.uniform(-np.pi, np.pi))
        if not len(yx):
            proposed_components.append(np.zeros(tissue.shape, dtype=bool))
            component_parameters.append(
                {
                    "component_index": component_index,
                    "realized": False,
                    "infeasibility_reason": "no-unused-genuine-interior-candidate",
                    "radius_y_x_px": [radius_y, radius_x],
                    "angle_radians": angle,
                    "proposed_pixel_count": 0,
                    "retained_pixel_count": 0,
                }
            )
            continue
        center_y, center_x = yx[int(rng.integers(len(yx)))]
        proposal = _rotated_ellipse(
            tissue.shape,
            (float(center_y), float(center_x)),
            (radius_y, radius_x),
            angle,
        ) & interior
        proposal[center_y, center_x] = True
        proposed_components.append(proposal)
        proposed_union |= proposal
        component_parameters.append(
            {
                "component_index": component_index,
                "realized": True,
                "infeasibility_reason": "none",
                "center_y_x": [int(center_y), int(center_x)],
                "radius_y_x_px": [radius_y, radius_x],
                "angle_radians": angle,
                "proposed_pixel_count": int(proposal.sum()),
                "retained_pixel_count": 0,
            }
        )
    retained = np.zeros(tissue.shape, dtype=bool)
    ordered = []
    for parameters, component in zip(component_parameters, proposed_components):
        if not parameters["realized"]:
            ordered.append(np.empty((0, 2), dtype=np.int64))
            continue
        indices = np.argwhere(component)
        center = np.asarray(parameters["center_y_x"], dtype=np.float64)
        distance = np.sum((indices - center) ** 2, axis=1)
        ordered.append(indices[np.argsort(distance, kind="stable")])
    cursors = np.zeros(len(ordered), dtype=np.int64)
    while int(retained.sum()) < maximum_new_pixels:
        progressed = False
        for component_index, indices in enumerate(ordered):
            while cursors[component_index] < len(indices):
                yy, xx = indices[cursors[component_index]]
                cursors[component_index] += 1
                if not retained[yy, xx]:
                    retained[yy, xx] = True
                    component_parameters[component_index]["retained_pixel_count"] += 1
                    progressed = True
                    break
            if int(retained.sum()) >= maximum_new_pixels:
                break
        if not progressed:
            break
    realized_components = sum(
        int(parameters["retained_pixel_count"] > 0)
        for parameters in component_parameters
    )
    return np.ascontiguousarray(retained), {
        "genuine_interior_definition": (
            "one-pixel binary erosion of authenticated parent tissue, intersected "
            "with remaining disjoint support"
        ),
        "interior_candidate_pixel_count": int(interior.sum()),
        "requested_component_count": int(component_count),
        "realized_component_count": int(realized_components),
        "component_support_limited": realized_components != component_count,
        "components": component_parameters,
        "proposed_pixel_count": int(proposed_union.sum()),
        "retained_pixel_count": int(retained.sum()),
        "budget_clipped": int(proposed_union.sum()) > int(retained.sum()),
        "no_rejection_or_redraw": True,
    }


def _damage_streams(
    provenance: dict[str, object],
    receipts: dict[str, dict[str, object]],
    priors: dict[str, object],
) -> list[dict[str, object]]:
    damage = priors["damage"]
    categories = tuple(damage["event_category_probabilities"])
    probabilities = np.asarray(
        [damage["event_category_probabilities"][name] for name in categories]
    )
    maximum_slots = max(
        int(specification["event_count_range_inclusive"][1])
        for specification in damage["strata"].values()
    )
    component_minimum, component_maximum = damage[
        "internal_hole_component_count_range_inclusive"
    ]
    streams = []
    for event_index in range(maximum_slots):
        category = str(
            _rng(
                provenance,
                "damage",
                f"event-{event_index:02d}-category",
                receipts,
            ).choice(np.asarray(categories), p=probabilities)
        )
        geometry = str(
            _rng(
                provenance,
                "damage",
                f"event-{event_index:02d}-geometry-family",
                receipts,
            ).choice(np.asarray(damage["geometry_families"][category]))
        )
        component_count = int(
            _rng(
                provenance,
                "damage",
                f"event-{event_index:02d}-internal-hole-component-count",
                receipts,
            ).integers(component_minimum, component_maximum + 1)
        )
        streams.append(
            {
                "category": category,
                "geometry": geometry,
                "component_count": component_count,
                "parameters": _rng(
                    provenance,
                    "damage",
                    f"event-{event_index:02d}-parameters",
                    receipts,
                ),
                "component_parameters": [
                    _rng(
                        provenance,
                        "damage",
                        f"event-{event_index:02d}-internal-hole-component-{component_index:02d}",
                        receipts,
                    )
                    for component_index in range(component_maximum)
                ],
            }
        )
    return streams


def _event_mask_v3(
    tissue: np.ndarray,
    excluded: np.ndarray,
    event_index: int,
    radius_fraction_range: list[float],
    maximum_new_pixels: int,
    stream: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    remaining = tissue & ~excluded
    category = str(stream["category"])
    geometry = str(stream["geometry"])
    common = {
        "event_index": int(event_index),
        "category": category,
        "geometry": geometry,
        "maximum_new_pixels": int(maximum_new_pixels),
        "no_rejection_or_redraw": True,
    }
    if maximum_new_pixels < 1 or not remaining.any():
        return np.zeros(tissue.shape, dtype=bool), {
            **common,
            "realized": False,
            "infeasibility_reason": "zero-budget-or-no-remaining-tissue",
            "proposed_pixel_count": 0,
            "retained_pixel_count": 0,
            "budget_clipped": False,
        }
    if category == "physical-loss" and geometry == "internal-hole":
        mask, parameters = _internal_hole_mask(
            tissue,
            remaining,
            int(stream["component_count"]),
            stream["component_parameters"],
            radius_fraction_range,
            maximum_new_pixels,
        )
        centers = [
            component["center_y_x"]
            for component in parameters["components"]
            if "center_y_x" in component
        ]
        return mask, {
            **common,
            "realized": bool(mask.any()),
            "infeasibility_reason": (
                "none" if mask.any() else "no-genuine-interior-support"
            ),
            "center_y_x": centers[0] if centers else [-1, -1],
            **parameters,
        }
    rng = stream["parameters"]
    if category == "physical-loss" and geometry == "edge-bite":
        candidates = remaining & tissue & ~scipy.ndimage.binary_erosion(tissue)
    else:
        candidates = remaining
    yx = np.argwhere(candidates)
    if not len(yx):
        return np.zeros(tissue.shape, dtype=bool), {
            **common,
            "realized": False,
            "infeasibility_reason": "no-eligible-geometry-centre",
            "proposed_pixel_count": 0,
            "retained_pixel_count": 0,
            "budget_clipped": False,
        }
    center_y, center_x = yx[int(rng.integers(len(yx)))]
    low, high = radius_fraction_range
    radius_y = max(1.0, tissue.shape[0] * float(rng.uniform(low, high)))
    radius_x = max(1.0, tissue.shape[1] * float(rng.uniform(low, high)))
    angle = float(rng.uniform(-np.pi, np.pi))
    if geometry in {"ellipse", "edge-bite"}:
        proposed = _rotated_ellipse(
            tissue.shape,
            (float(center_y), float(center_x)),
            (radius_y, radius_x),
            angle,
        )
    else:
        y, x = np.mgrid[: tissue.shape[0], : tissue.shape[1]]
        delta_y, delta_x = y - center_y, x - center_x
        along = np.cos(angle) * delta_x + np.sin(angle) * delta_y
        across = -np.sin(angle) * delta_x + np.cos(angle) * delta_y
        proposed = (
            (np.abs(along) <= 1.8 * max(radius_y, radius_x))
            & (np.abs(across) <= 0.45 * min(radius_y, radius_x))
        )
    proposed &= remaining
    proposed[center_y, center_x] = True
    mask = _stable_budget_clip(
        proposed, (float(center_y), float(center_x)), maximum_new_pixels
    )
    return mask, {
        **common,
        "realized": bool(mask.any()),
        "infeasibility_reason": "none",
        "center_y_x": [int(center_y), int(center_x)],
        "radius_y_x_px": [radius_y, radius_x],
        "angle_radians": angle,
        "proposed_pixel_count": int(proposed.sum()),
        "retained_pixel_count": int(mask.sum()),
        "budget_clipped": int(proposed.sum()) > int(mask.sum()),
    }


def _sample_parent_damage(
    tissue: np.ndarray,
    provenance: dict[str, object],
    rng_receipts: dict[str, dict[str, object]],
    priors: dict[str, object],
    modality: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    damage_prior = priors["damage"]
    probabilities = damage_prior["stratum_probabilities"]
    strata = tuple(probabilities)
    stratum = str(
        _rng(provenance, "damage", "stratum", rng_receipts).choice(
            np.asarray(strata), p=np.asarray([probabilities[name] for name in strata])
        )
    )
    specification = damage_prior["strata"][stratum]
    minimum_events, maximum_events = specification["event_count_range_inclusive"]
    requested = int(
        _rng(provenance, "damage", "event-count", rng_receipts).integers(
            minimum_events, maximum_events + 1
        )
    )
    streams = _damage_streams(provenance, rng_receipts, priors)
    tissue_count = int(tissue.sum())
    if tissue_count < 1:
        raise ValueError("authenticated section-processing parent has no tissue")
    maximum_fraction = float(specification["maximum_damaged_tissue_fraction"])
    budget = int(np.floor(maximum_fraction * tissue_count))
    scheduled = min(requested, budget)
    masks = {
        "physical-loss": np.zeros(tissue.shape, dtype=bool),
        "occlusion": np.zeros(tissue.shape, dtype=bool),
        "appearance-artifact": np.zeros(tissue.shape, dtype=bool),
    }
    events = []
    for event_index in range(scheduled):
        excluded = masks["physical-loss"] | masks["occlusion"] | masks[
            "appearance-artifact"
        ]
        maximum_new_pixels = budget - int(excluded.sum()) - (
            scheduled - event_index - 1
        )
        event, parameters = _event_mask_v3(
            tissue,
            excluded,
            event_index,
            specification["radius_fraction_range"],
            maximum_new_pixels,
            streams[event_index],
        )
        masks[str(streams[event_index]["category"])] |= event
        events.append(parameters)
    if modality == "brightfield-nissl-like":
        occlusion_value = float(
            _rng(provenance, "damage", "occlusion-value", rng_receipts).uniform(
                0.0, 0.12
            )
        )
        artifact_value = float(
            _rng(
                provenance, "damage", "appearance-artifact-value", rng_receipts
            ).uniform(0.88, 1.0)
        )
    else:
        occlusion_value = float(
            _rng(provenance, "damage", "occlusion-value", rng_receipts).uniform(
                0.82, 1.0
            )
        )
        artifact_value = float(
            _rng(
                provenance, "damage", "appearance-artifact-value", rng_receipts
            ).uniform(0.65, 0.95)
        )
    physical = masks["physical-loss"]
    occlusion = masks["occlusion"]
    appearance = masks["appearance-artifact"]
    union = physical | occlusion | appearance
    realized = sum(int(event["realized"]) for event in events)
    arrays = {
        "parent_physical_loss_mask": np.ascontiguousarray(physical),
        "parent_occlusion_mask": np.ascontiguousarray(occlusion),
        "parent_appearance_artifact_mask": np.ascontiguousarray(appearance),
        "parent_damage_union_mask": np.ascontiguousarray(union),
    }
    return arrays, {
        "sampling_domain": (
            "full authenticated section-processing parent before acquisition-window application"
        ),
        "parent_shape_h_w": list(tissue.shape),
        "parent_tissue_pixel_count": tissue_count,
        "stratum": stratum,
        "stratum_probability": float(probabilities[stratum]),
        "requested_event_count": requested,
        "scheduled_event_count": scheduled,
        "realized_event_count": realized,
        "damage_budget_pixels": budget,
        "support_limited": realized != requested,
        "maximum_damaged_tissue_fraction": maximum_fraction,
        "parent_damaged_tissue_fraction": float(union.sum() / tissue_count),
        "events": events,
        "occlusion_value": occlusion_value,
        "appearance_artifact_value": artifact_value,
        "fixed_named_event_stream_slots": len(streams),
        "no_window_or_pose_conditioning": True,
        "no_rejection_or_redraw": True,
        "array_receipts": _array_receipts(arrays),
    }


def _appearance(
    scalar: np.ndarray,
    labels: np.ndarray,
    tissue: np.ndarray,
    modality: str,
    provenance: dict[str, object],
    rng_receipts: dict[str, dict[str, object]],
    priors: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    modality_prior = priors["modalities"][modality]
    normalized = np.zeros(scalar.shape, dtype=np.float32)
    values = scalar[tissue].astype(np.float64)
    if len(values) >= 256:
        lower, upper = np.quantile(values, (0.01, 0.99), method="linear")
        method = "q01-q99"
    elif len(values):
        lower, upper = float(values.min()), float(values.max())
        method = "min-max-small-support"
    else:
        lower = upper = 0.0
        method = "empty-window-no-tissue"
    if len(values) and upper <= lower:
        normalized[tissue] = np.float32(0.5)
    elif len(values):
        normalized[tissue] = np.clip(
            (values - lower) / (upper - lower), 0.0, 1.0
        ).astype(np.float32)
    region_ids = np.unique(labels[tissue])
    sampled_region_levels = _rng(
        provenance, "appearance", "label-region-levels", rng_receipts
    ).uniform(
        *modality_prior["label_level_range"], size=len(region_ids)
    ).astype(np.float32)
    anti_prior = priors["appearance_label_anti_shortcut"]
    anti_shortcut = bool(
        _rng(
            provenance,
            "appearance",
            "label-boundary-anti-shortcut-enable",
            rng_receipts,
        ).random()
        < float(anti_prior["probability"])
    )
    sampled_return_fraction = float(
        _rng(
            provenance,
            "appearance",
            "label-boundary-return-fraction",
            rng_receipts,
        ).uniform(*anti_prior["return_fraction_range"])
    )
    effective_return_fraction = sampled_return_fraction if anti_shortcut else 0.0
    global_tissue_mean = (
        float(normalized[tissue].mean(dtype=np.float64)) if tissue.any() else 0.0
    )
    effective_region_levels = (
        sampled_region_levels * np.float32(1.0 - effective_return_fraction)
        + np.float32(global_tissue_mean * effective_return_fraction)
    ).astype(np.float32)
    label_image = np.zeros(scalar.shape, dtype=np.float32)
    for ordinal, region_id in enumerate(region_ids):
        label_image[tissue & (labels == region_id)] = effective_region_levels[ordinal]
    label_weight = float(modality_prior["label_blend_weight"])
    latent = np.zeros(scalar.shape, dtype=np.float32)
    latent[tissue] = (
        (1.0 - label_weight) * normalized[tissue]
        + label_weight * label_image[tissue]
    )
    background_level = _uniform(
        provenance,
        "appearance",
        "background-level",
        rng_receipts,
        modality_prior["background_level_range"],
    )
    background_texture = _uniform(
        provenance,
        "appearance",
        "background-texture-strength",
        rng_receipts,
        modality_prior["background_texture_range"],
    )
    background = background_level + background_texture * observation_v2._smooth_unit_field(
        scalar.shape,
        _rng(provenance, "appearance", "background-field", rng_receipts),
    )
    background += _rng(
        provenance, "appearance", "background-sensor-noise", rng_receipts
    ).normal(
        0.0, 0.002 if modality == "brightfield-nissl-like" else 0.004, scalar.shape
    )
    background = np.clip(background, 0.0, 1.0).astype(np.float32)
    photon_count = _uniform(
        provenance,
        "appearance",
        "photon-count",
        rng_receipts,
        modality_prior["photon_count_range"],
    )
    read_noise = _uniform(
        provenance,
        "appearance",
        "read-noise-std",
        rng_receipts,
        modality_prior["read_noise_std_range"],
    )
    bias = 1.0 + np.float32(0.08) * observation_v2._smooth_unit_field(
        scalar.shape,
        _rng(
            provenance, "appearance", "tissue-bias-field", rng_receipts
        ),
    )
    if modality == "brightfield-nissl-like":
        base = _uniform(
            provenance,
            "appearance",
            "optical-density-base",
            rng_receipts,
            modality_prior["optical_density_base_range"],
        )
        contrast = _uniform(
            provenance,
            "appearance",
            "optical-density-contrast",
            rng_receipts,
            modality_prior["optical_density_contrast_range"],
        )
        expected = np.clip(
            bias * np.exp(-(base + contrast * (1.0 - latent))), 0.0, 1.0
        )
        forward = {
            "family": "transmitted-light Beer-Lambert-like attenuation",
            "optical_density_base": base,
            "optical_density_contrast": contrast,
        }
    else:
        baseline = _uniform(
            provenance,
            "appearance",
            "emission-baseline",
            rng_receipts,
            modality_prior["emission_baseline_range"],
        )
        gain = _uniform(
            provenance,
            "appearance",
            "emission-gain",
            rng_receipts,
            modality_prior["emission_gain_range"],
        )
        gamma = _uniform(
            provenance,
            "appearance",
            "emission-gamma",
            rng_receipts,
            modality_prior["emission_gamma_range"],
        )
        expected = np.clip(
            baseline + gain * bias * np.power(latent, gamma), 0.0, 1.0
        )
        forward = {
            "family": "additive positive fluorescence emission",
            "emission_baseline": baseline,
            "emission_gain": gain,
            "emission_gamma": gamma,
        }
    shot = _rng(
        provenance, "appearance", "shot-noise", rng_receipts
    ).poisson(np.asarray(expected, dtype=np.float64) * photon_count) / photon_count
    shot += _rng(
        provenance, "appearance", "read-noise", rng_receipts
    ).normal(0.0, read_noise, scalar.shape)
    tissue_image = np.clip(shot, 0.0, 1.0).astype(np.float32)
    pre_damage = np.where(tissue, tissue_image, background).astype(np.float32)
    return {
        "normalized_template_float32": normalized,
        "label_conditioned_latent_float32": latent,
        "acquired_background_float32": background,
        "pre_damage_acquired_image_float32": pre_damage,
    }, {
        "canvas_shape_h_w": list(scalar.shape),
        "synthesis_domain": "fixed acquisition canvas after the one frozen window",
        "normalization": {
            "method": method,
            "lower": float(lower),
            "upper": float(upper),
            "tissue_pixel_count": int(tissue.sum()),
        },
        "label_conditioning": {
            "operator": "equality-defined atlas regions receive random levels; integers are not intensities",
            "template_weight": 1.0 - label_weight,
            "label_weight": label_weight,
            "region_ids": [int(value) for value in region_ids],
            "sampled_region_levels": [float(value) for value in sampled_region_levels],
            "effective_region_levels": [
                float(value) for value in effective_region_levels
            ],
            "anti_boundary_shortcut_return_to_global_mean": anti_shortcut,
            "policy_probability": float(anti_prior["probability"]),
            "sampled_return_fraction": sampled_return_fraction,
            "effective_return_fraction": effective_return_fraction,
            "global_tissue_mean": global_tissue_mean,
            "ontology_coarsening": None,
        },
        "forward_parameters": forward,
        "photon_count": photon_count,
        "read_noise_std": read_noise,
        "empty_canvas_tissue_supported": True,
    }


def _canvas_damage(
    pre_damage: np.ndarray,
    background: np.ndarray,
    tissue: np.ndarray,
    correspondence: np.ndarray,
    dense_weight: np.ndarray,
    transformed: dict[str, np.ndarray],
    parameters: dict[str, object],
) -> dict[str, np.ndarray]:
    parent_domain = np.asarray(transformed["parent_sampling_domain_mask"], dtype=bool)
    physical = (
        np.asarray(transformed["parent_physical_loss_mask"], dtype=bool)
        & parent_domain
    )
    occlusion = (
        np.asarray(transformed["parent_occlusion_mask"], dtype=bool)
        & tissue
        & ~physical
    )
    appearance = (
        np.asarray(transformed["parent_appearance_artifact_mask"], dtype=bool)
        & tissue
        & ~physical
        & ~occlusion
    )
    if np.any(physical & occlusion) or np.any(physical & appearance) or np.any(
        occlusion & appearance
    ):
        raise ValueError("windowed parent damage categories are not disjoint")
    damage = physical | occlusion | appearance
    raw = pre_damage.copy()
    raw[occlusion] = np.float32(parameters["occlusion_value"])
    raw[appearance] = np.float32(parameters["appearance_artifact_value"])
    valid = correspondence & tissue & ~damage
    return {
        "raw_acquired_image_float32": np.ascontiguousarray(raw, dtype=np.float32),
        "physical_loss_mask": np.ascontiguousarray(physical),
        "occlusion_mask": np.ascontiguousarray(occlusion),
        "appearance_artifact_mask": np.ascontiguousarray(appearance),
        "damage_union_mask": np.ascontiguousarray(damage),
        "observable_footprint_mask": np.ascontiguousarray(tissue),
        "observation_invalid_mask": np.ascontiguousarray(damage),
        "outside_correspondence_domain_mask": np.ascontiguousarray(~correspondence),
        "valid_correspondence_mask": np.ascontiguousarray(valid),
        "valid_correspondence_weight_float32": np.ascontiguousarray(
            np.where(valid, dense_weight, np.float32(0)), dtype=np.float32
        ),
    }


def _imperfect_brush_mask_v3(
    footprint: np.ndarray,
    provenance: dict[str, object],
    rng_receipts: dict[str, dict[str, object]],
    priors: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    brush = priors["smart_brush"]
    radius = int(brush["morphology_radius_px"])
    morphology = int(
        _rng(provenance, "brush", "morphology", rng_receipts).choice(
            np.asarray([-radius, -1, 1, radius], dtype=np.int64)
        )
    )
    amplitude = float(brush["jitter_amplitude_px"])
    jitter_x_rng = _rng(provenance, "brush", "jitter-x", rng_receipts)
    jitter_y_rng = _rng(provenance, "brush", "jitter-y", rng_receipts)
    minimum_events, maximum_events = brush[
        "island_or_gap_count_range_inclusive"
    ]
    requested_event_count = int(
        _rng(
            provenance, "brush", "island-or-gap-event-count", rng_receipts
        ).integers(minimum_events, maximum_events + 1)
    )
    slot_streams = []
    for event_index in range(maximum_events):
        kind = str(
            _rng(
                provenance,
                "brush",
                f"island-or-gap-event-{event_index:02d}-kind",
                rng_receipts,
            ).choice(np.asarray(["gap", "island"]))
        )
        parameters_rng = _rng(
            provenance,
            "brush",
            f"island-or-gap-event-{event_index:02d}-parameters",
            rng_receipts,
        )
        low, high = brush["island_or_gap_clean_tissue_fraction_range"]
        slot_streams.append(
            {
                "kind": kind,
                "fraction": float(parameters_rng.uniform(low, high)),
                "aspect_ratio": float(parameters_rng.uniform(0.65, 1.35)),
                "rng": parameters_rng,
            }
        )
    mask = (
        scipy.ndimage.binary_dilation(footprint, iterations=morphology)
        if morphology > 0
        else scipy.ndimage.binary_erosion(footprint, iterations=-morphology)
    )
    y, x = np.mgrid[: footprint.shape[0], : footprint.shape[1]].astype(np.float32)
    dx = amplitude * observation_v2._smooth_unit_field(footprint.shape, jitter_x_rng)
    dy = amplitude * observation_v2._smooth_unit_field(footprint.shape, jitter_y_rng)
    mask = scipy.ndimage.map_coordinates(
        mask.astype(np.uint8),
        (y + dy, x + dx),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool)
    clean_tissue_count = int(footprint.sum())
    events = []
    realized_event_count = 0
    for event_index, stream in enumerate(slot_streams):
        active = event_index < requested_event_count
        kind = str(stream["kind"])
        fraction = float(stream["fraction"])
        target_pixels = (
            max(1, int(round(fraction * clean_tissue_count)))
            if clean_tissue_count
            else 0
        )
        parameters = {
            "event_index": event_index,
            "active": active,
            "kind": kind,
            "sampled_clean_tissue_fraction": fraction,
            "target_pixel_count": target_pixels,
            "aspect_ratio": float(stream["aspect_ratio"]),
            "no_rejection_or_redraw": True,
        }
        if not active:
            events.append(
                {
                    **parameters,
                    "realized": False,
                    "infeasibility_reason": "inactive-fixed-stream-slot",
                    "candidate_center_count": 0,
                    "proposed_pixel_count": 0,
                    "retained_pixel_count": 0,
                }
            )
            continue
        if kind == "gap":
            candidate_mask = mask & ~scipy.ndimage.binary_erosion(mask)
        else:
            extent = max(3, int(np.ceil(np.sqrt(max(target_pixels, 1) / np.pi))))
            candidate_mask = (
                scipy.ndimage.binary_dilation(mask, iterations=extent) & ~mask
            )
        candidates = np.argwhere(candidate_mask)
        if target_pixels < 1 or not len(candidates):
            events.append(
                {
                    **parameters,
                    "realized": False,
                    "infeasibility_reason": (
                        "empty-clean-footprint"
                        if target_pixels < 1
                        else f"no-{kind}-candidate"
                    ),
                    "candidate_center_count": int(len(candidates)),
                    "proposed_pixel_count": 0,
                    "retained_pixel_count": 0,
                }
            )
            continue
        rng = stream["rng"]
        center_y, center_x = candidates[int(rng.integers(len(candidates)))]
        aspect = float(stream["aspect_ratio"])
        base_radius = np.sqrt(max(target_pixels, 1) / np.pi)
        radius_y = max(1.0, base_radius / np.sqrt(aspect))
        radius_x = max(1.0, base_radius * np.sqrt(aspect))
        proposal = observation_v2._ellipse(
            footprint.shape,
            (float(center_y), float(center_x)),
            (radius_y, radius_x),
        )
        proposal &= mask if kind == "gap" else ~mask
        retained = _stable_budget_clip(
            proposal, (float(center_y), float(center_x)), target_pixels
        )
        if kind == "gap":
            mask[retained] = False
        else:
            mask[retained] = True
        realized = bool(retained.any())
        realized_event_count += int(realized)
        events.append(
            {
                **parameters,
                "realized": realized,
                "infeasibility_reason": "none" if realized else "empty-proposal",
                "candidate_center_count": int(len(candidates)),
                "center_y_x": [int(center_y), int(center_x)],
                "radius_y_x_px": [float(radius_y), float(radius_x)],
                "proposed_pixel_count": int(proposal.sum()),
                "retained_pixel_count": int(retained.sum()),
            }
        )
    union = int((mask | footprint).sum())
    quality_iou = float((mask & footprint).sum() / union) if union else 1.0
    empty_selection = not bool(mask.any())
    return np.ascontiguousarray(mask), {
        "operator": (
            "independent morphology, smooth boundary jitter, and 0-2 total named "
            "gap-or-island events"
        ),
        "morphology_px": morphology,
        "requested_island_or_gap_event_count": requested_event_count,
        "realized_island_or_gap_event_count": realized_event_count,
        "support_limited": realized_event_count != requested_event_count,
        "fixed_named_event_stream_slots": maximum_events,
        "events": events,
        "quality_iou": quality_iou,
        "quality_iou_role": "audit-statistic-only",
        "superseded_v1_accepted_iou_interval": [0.70, 0.98],
        "iou_acceptance_rejection_or_redraw": False,
        "empty_selection": empty_selection,
        "selection_failure_tag": (
            "empty-imperfect-brush-selection" if empty_selection else "none"
        ),
        "independent_of_damage_rng": True,
        "used_as_correspondence_truth": False,
        "no_rejection_or_redraw": True,
    }


def _descendant_sampling_policy() -> dict[str, object]:
    return {
        "canonical_trainable_raw_background_mode": (
            CANONICAL_TRAINABLE_RAW_BACKGROUND_MODE
        ),
        "raw_background_equivalent_modes": list(RAW_BACKGROUND_EQUIVALENT_MODES),
        "audit_only_modes": ["raw"],
        "duplicate_equivalent_sampling_rule": (
            "smart-brush-absent is the sole trainable raw-background sample; raw is "
            "a byte-identical audit mirror and is never sampled as an additional input"
        ),
    }


def _apply_verified_window(
    plan: dict[str, object],
    source_arrays: dict[str, np.ndarray],
    array_roles: dict[str, str],
    *,
    source_validity: dict[str, np.ndarray],
    grid_id: str,
    fov: tuple[float, float],
    design_ouv: np.ndarray,
    centre_support: np.ndarray,
    optical_support: np.ndarray,
    upstream_realization_ids: dict[str, str],
    section_processing_receipt: dict[str, object],
    section_processing_receipt_sha256: str,
    lineage: dict[str, object],
) -> dict[str, object]:
    return acquisition_window.apply_acquisition_window_v3(
        plan,
        source_arrays,
        array_roles,
        source_validity=source_validity,
        global_reference_grid_id=grid_id,
        global_reference_fov_uv_um=fov,
        design_quicknii_ouv=design_ouv,
        centre_plane_support_mask=centre_support,
        optical_slab_support_mass=optical_support,
        upstream_realization_ids=upstream_realization_ids,
        section_processing_receipt=section_processing_receipt,
        section_processing_receipt_sha256=section_processing_receipt_sha256,
        lineage=lineage,
        parent_landmarks_yx={},
    )


def _descendant_identity(descendant: dict[str, object]) -> dict[str, object]:
    return {
        key: descendant[key]
        for key in (
            "schema_version",
            "mode",
            "trainable",
            "brush_available",
            "acquired_observation_id",
            "background_policy",
            "parameters",
            "array_receipts",
        )
    }


def _descendant(
    mode: str,
    raw: np.ndarray,
    selected: np.ndarray,
    brush_error: np.ndarray,
    acquired_observation_id: str,
    parameters: dict[str, object],
    *,
    trainable: bool = True,
) -> dict[str, object]:
    available = mode in {"smart-brush-accurate", "smart-brush-imperfect"}
    image = np.zeros(raw.shape, dtype=np.float32) if available else raw.copy()
    if available:
        image[selected] = raw[selected]
    arrays = {
        "model_input_image_float32": np.ascontiguousarray(image, dtype=np.float32),
        "selected_input_mask": np.ascontiguousarray(selected, dtype=bool),
        "brush_mask_error_mask": np.ascontiguousarray(brush_error, dtype=bool),
    }
    descendant = {
        "schema_version": OBSERVATION_DESCENDANT_V3_SCHEMA,
        "mode": mode,
        "trainable": bool(trainable),
        "brush_available": available,
        "acquired_observation_id": acquired_observation_id,
        "background_policy": (
            "exact-positive-float32-black-outside-selected-mask"
            if available
            else "byte-identical-raw-acquired-background"
        ),
        "parameters": parameters,
        "arrays": arrays,
        "array_receipts": _array_receipts(arrays),
    }
    descendant["descendant_id"] = acquisition._payload_sha256(
        _descendant_identity(descendant)
    )
    return descendant


def observation_bundle_receipt_v3(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": artifact["schema_version"],
        "algorithm": artifact["algorithm"],
        "implementation_source_sha256": artifact["implementation_source_sha256"],
        "runtime_dependencies": artifact["runtime_dependencies"],
        "asset_dependencies": artifact["asset_dependencies"],
        "upstream_reference": artifact["upstream_reference"],
        "lineage": artifact["lineage"],
        "rng_provenance": artifact["rng_provenance"],
        "modality": artifact["modality"],
        "engineering_priors": artifact["engineering_priors"],
        "parent_authentication_v3": artifact["parent_authentication_v3"],
        "acquisition_window_realization_receipt": (
            acquisition_window.acquisition_window_realization_receipt_v3(
                artifact["acquisition_window_realization"]
            )
        ),
        "parent_damage_geometry": artifact["parent_damage_geometry"],
        "deformation_pose_gauge": artifact["deformation_pose_gauge"],
        "pose_supervision": artifact["pose_supervision"],
        "parameters": artifact["parameters"],
        "descendant_sampling_policy": artifact["descendant_sampling_policy"],
        "rng_sources": artifact["rng_sources"],
        "array_receipts": artifact["array_receipts"],
        "acquired_observation_id": artifact["acquired_observation_id"],
        "descendant_identity": {
            mode: {
                **_descendant_identity(artifact["descendants"][mode]),
                "descendant_id": artifact["descendants"][mode]["descendant_id"],
            }
            for mode in DESCENDANT_MODES
        },
        "observation_bundle_id": artifact["observation_bundle_id"],
    }


def make_arbitrary_plane_observation_v3(
    processed_render: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    acquisition_window_plan: dict[str, object],
    *,
    subject_plan: dict[str, object] | None,
    root_seed: int | str,
    split: str,
    split_index: int,
    animal_index: int,
    animal_id: str | int,
    section_index: int,
    observation_index: int,
    modality: str,
    batch_size: int | None = None,
    authenticated_parent_v3: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create one v3 observation from a separately frozen acquisition-window plan."""
    root_seed_value = observation_v2._root_seed_uint64(root_seed)
    split_index = observation_v2._nonnegative_integer(split_index, "split_index")
    animal_index = observation_v2._nonnegative_integer(animal_index, "animal_index")
    section_index = observation_v2._nonnegative_integer(section_index, "section_index")
    observation_index = observation_v2._nonnegative_integer(
        observation_index, "observation_index"
    )
    if not isinstance(split, str) or not split or modality not in MODALITIES:
        raise ValueError("split or modality is invalid")
    if authenticated_parent_v3 is None:
        authenticated_parent_v3 = authenticate_observation_parent_v3(
            processed_render,
            subject_slab_render,
            section_processing_plan,
            prepared_context,
            precursor,
            subject_plan=subject_plan,
            batch_size=batch_size,
        )
    else:
        verify_observation_parent_authentication_v3(
            authenticated_parent_v3,
            processed_render,
            subject_slab_render,
            section_processing_plan,
            prepared_context,
            precursor,
        )
    lineage = _lineage(precursor, subject_slab_render, section_processing_plan)
    animal_id = acquisition._json_value(animal_id)
    if (
        split != lineage["split"]
        or animal_index != lineage["animal_index"]
        or section_index != lineage["section_index"]
        or acquisition._payload_sha256({"animal_id": animal_id})
        != acquisition._payload_sha256({"animal_id": lineage["animal_id"]})
        or (
            subject_plan is not None
            and subject_plan["synthetic_animal_id"] != lineage["synthetic_animal_id"]
        )
    ):
        raise ValueError("observation lineage differs from authenticated upstream lineage")
    sample_index = observation_v2._nonnegative_integer(
        precursor["generator"]["resolved_config"]["sample_index"], "sample_index"
    )
    provenance = {
        "root_seed_uint64": f"0x{root_seed_value:016x}",
        "split": split,
        "sample_index": sample_index,
        "split_index": split_index,
        "animal_index": animal_index,
        "section_index": section_index,
        "observation_index": observation_index,
        "rng_input_contract": (
            "exact acquisition.derive_v2_field_seed(root_seed, split, authenticated "
            "precursor sample_index, stage, field, attempt) tuple; split_index, animal_index, "
            "section_index, observation_index, authenticated lineage labels, images, pose, "
            "tissue, support, artifact values, and outcomes are excluded"
        ),
        "rng_tuple_encoding": (
            "uint32 big-endian UTF-8 length prefixes; BLAKE2b-64 person=AP-ACQ-V2; "
            "unsigned big-endian uint64; NumPy PCG64DXSM"
        ),
    }
    (
        scalar,
        labels,
        tissue,
        correspondence,
        mapped,
        bilinear_valid,
        nearest_valid,
        dense_valid,
        dense_weight,
        abstention,
    ) = observation_v2._source_arrays(processed_render)
    _verify_plan_binding(
        acquisition_window_plan, provenance, scalar.shape, sample_index
    )
    grid_id, fov, design_ouv, centre_support, optical_support = _window_geometry(
        processed_render
    )
    priors = _engineering_priors()
    damage_rng: dict[str, dict[str, object]] = {}
    parent_damage, parent_damage_parameters = _sample_parent_damage(
        tissue, provenance, damage_rng, priors, modality
    )
    state = processed_render["state"]
    parent_pullback_map = np.ascontiguousarray(
        state["source_index_yx"], dtype=np.float64
    )
    processed_centres = np.ascontiguousarray(
        state["processed_pixel_centres_yx_um"], dtype=np.float64
    )
    pitch = np.asarray(
        section_processing_plan["resolved_config"]["pixel_pitch_y_x_um"],
        dtype=np.float64,
    )
    if section_processing_plan["resolved_config"]["deformation_mode"] == "identity":
        parent_pullback_velocity = np.zeros(parent_pullback_map.shape, dtype=np.float64)
    else:
        velocity_yx_um = np.asarray(
            section_processing._accepted_field(section_processing_plan)(
                processed_centres, return_gradient=False
            ),
            dtype=np.float64,
        )
        parent_pullback_velocity = np.ascontiguousarray(
            -velocity_yx_um / pitch, dtype=np.float64
        )
    parent_velocity_finite = np.isfinite(parent_pullback_velocity).all(axis=-1)
    deformation_valid = (
        bilinear_valid
        & np.isfinite(parent_pullback_map).all(axis=-1)
        & parent_velocity_finite
    )
    parent_pullback_map = np.where(
        deformation_valid[..., None], parent_pullback_map, 0.0
    ).astype(np.float64)
    parent_pullback_velocity = np.where(
        parent_velocity_finite[..., None], parent_pullback_velocity, 0.0
    ).astype(np.float64)
    physical_loss = parent_damage["parent_physical_loss_mask"]
    surviving_tissue = tissue & ~physical_loss
    surviving_correspondence = correspondence & ~physical_loss
    surviving_dense_valid = dense_valid & ~physical_loss
    surviving_dense_weight = np.where(
        surviving_dense_valid, dense_weight, np.float32(0)
    ).astype(np.float32)
    surviving_abstention = abstention | physical_loss
    surviving_scalar = np.where(physical_loss, np.float32(0), scalar).astype(
        np.float32
    )
    surviving_labels = np.where(physical_loss, 0, labels).astype(np.int64)
    safe_mapped = np.where(
        surviving_dense_valid[..., None], mapped, 0.0
    ).astype(np.float64)
    source_arrays = {
        "source_scalar_float32": surviving_scalar,
        "source_label_int64": surviving_labels,
        "source_tissue_mask": surviving_tissue,
        "source_correspondence_mask": surviving_correspondence,
        "source_dense_weight_float32": surviving_dense_weight,
        "source_abstention_mask": surviving_abstention,
        "source_mapped_ccf_float64": np.ascontiguousarray(safe_mapped),
        "source_bilinear_valid_mask": bilinear_valid,
        "source_nearest_valid_mask": nearest_valid,
        "source_dense_valid_mask": surviving_dense_valid,
        "source_dense_valid_fraction_float32": surviving_dense_valid.astype(
            np.float32
        ),
        "truth_section_pullback_map_yx_px_float64": parent_pullback_map,
        "truth_section_pullback_stationary_velocity_yx_px_float64": (
            parent_pullback_velocity
        ),
        **parent_damage,
    }
    array_roles = {
        "source_scalar_float32": "scalar",
        "source_label_int64": "annotation",
        "source_tissue_mask": "mask",
        "source_correspondence_mask": "mask",
        "source_dense_weight_float32": "scalar",
        "source_abstention_mask": "abstention-mask",
        "source_mapped_ccf_float64": "ccf-coordinate",
        "source_bilinear_valid_mask": "mask",
        "source_nearest_valid_mask": "mask",
        "source_dense_valid_mask": "mask",
        "source_dense_valid_fraction_float32": "scalar",
        "truth_section_pullback_map_yx_px_float64": "absolute-map-yx",
        "truth_section_pullback_stationary_velocity_yx_px_float64": "vector-yx",
        "parent_physical_loss_mask": "mask",
        "parent_occlusion_mask": "mask",
        "parent_appearance_artifact_mask": "mask",
        "parent_damage_union_mask": "mask",
    }
    source_validity = {
        "source_scalar_float32": bilinear_valid,
        "source_label_int64": nearest_valid,
        "source_tissue_mask": nearest_valid,
        "source_correspondence_mask": nearest_valid,
        "source_dense_weight_float32": surviving_dense_valid,
        "source_abstention_mask": nearest_valid,
        "source_mapped_ccf_float64": surviving_dense_valid,
        "source_bilinear_valid_mask": nearest_valid,
        "source_nearest_valid_mask": nearest_valid,
        "source_dense_valid_mask": nearest_valid,
        "source_dense_valid_fraction_float32": bilinear_valid,
        "truth_section_pullback_map_yx_px_float64": deformation_valid,
        "truth_section_pullback_stationary_velocity_yx_px_float64": (
            parent_velocity_finite
        ),
        "parent_physical_loss_mask": nearest_valid,
        "parent_occlusion_mask": nearest_valid,
        "parent_appearance_artifact_mask": nearest_valid,
        "parent_damage_union_mask": nearest_valid,
    }
    upstream_reference = _upstream_reference(
        processed_render, subject_slab_render, section_processing_plan, precursor
    )
    upstream_realization_ids = {
        name: upstream_reference[name]
        for name in acquisition_window.UPSTREAM_REALIZATION_ID_FIELDS
    }
    section_receipt = section_processing_render_receipt_v2(processed_render)
    window_lineage = {
        name: lineage[name] for name in acquisition_window.LINEAGE_FIELDS
    }
    window_realization = _apply_verified_window(
        acquisition_window_plan,
        source_arrays,
        array_roles,
        source_validity=source_validity,
        grid_id=grid_id,
        fov=fov,
        design_ouv=design_ouv,
        centre_support=centre_support,
        optical_support=optical_support,
        upstream_realization_ids=upstream_realization_ids,
        section_processing_receipt=section_receipt,
        section_processing_receipt_sha256=acquisition._payload_sha256(
            section_receipt
        ),
        lineage=window_lineage,
    )
    transformed = window_realization["arrays"]
    transformed_validity = {}
    for name in source_arrays:
        valid = np.asarray(
            transformed[name + acquisition_window.VALIDITY_SUFFIX], dtype=bool
        )
        abstain = np.asarray(
            transformed[name + acquisition_window.ABSTENTION_SUFFIX], dtype=bool
        )
        if not np.array_equal(abstain, ~valid):
            raise ValueError("window validity and abstention outputs disagree")
        transformed_validity[name] = valid & ~abstain
    parent_domain = np.asarray(transformed["parent_sampling_domain_mask"], dtype=bool)
    deformation_valid_canvas = (
        transformed_validity["truth_section_pullback_map_yx_px_float64"]
        & transformed_validity[
            "truth_section_pullback_stationary_velocity_yx_px_float64"
        ]
        & parent_domain
    )
    gauge_artifact = deformation_gauge.gauge_fix_canvas_deformation_v3(
        np.asarray(
            transformed[
                "truth_section_pullback_stationary_velocity_yx_px_float64"
            ],
            dtype=np.float64,
        ),
        np.asarray(
            transformed["truth_section_pullback_map_yx_px_float64"],
            dtype=np.float64,
        ),
        np.asarray(
            window_realization["transform"]["effective_quicknii_ouv"],
            dtype=np.float64,
        ),
        deformation_valid_canvas,
    )
    deformation_pose_gauge = (
        deformation_gauge.deformation_pose_gauge_summary_v3(gauge_artifact)
    )
    pullback_map_canvas = gauge_artifact["arrays"][
        "affine_free_pullback_map_yx_px_float64"
    ]
    pullback_velocity_canvas = gauge_artifact["arrays"][
        "affine_free_stationary_velocity_yx_px_float64"
    ]
    scalar_valid = transformed_validity["source_scalar_float32"]
    label_valid = transformed_validity["source_label_int64"]
    canvas_tissue = (
        np.asarray(transformed["source_tissue_mask"], dtype=bool)
        & transformed_validity["source_tissue_mask"]
        & parent_domain
    )
    mapped_canvas = np.array(
        transformed["source_mapped_ccf_float64"], dtype=np.float64, copy=True, order="C"
    )
    dense_fraction = np.asarray(
        transformed["source_dense_valid_fraction_float32"], dtype=np.float32
    )
    canvas_dense_valid = (
        np.asarray(transformed["source_dense_valid_mask"], dtype=bool)
        & transformed_validity["source_dense_valid_mask"]
        & transformed_validity["source_dense_valid_fraction_float32"]
        & transformed_validity["source_mapped_ccf_float64"]
        & parent_domain
        & np.isfinite(mapped_canvas).all(axis=-1)
        & (dense_fraction >= np.float32(1.0 - 8.0 * np.finfo(np.float32).eps))
    )
    mapped_canvas[~canvas_dense_valid] = 0.0
    canvas_weight = np.array(
        transformed["source_dense_weight_float32"],
        dtype=np.float32,
        copy=True,
        order="C",
    )
    canvas_weight[~transformed_validity["source_dense_weight_float32"]] = np.float32(0)
    canvas_weight[~canvas_dense_valid] = np.float32(0)
    canvas_abstention = (
        np.asarray(transformed["source_abstention_mask"], dtype=bool)
        | ~transformed_validity["source_abstention_mask"]
        | ~canvas_dense_valid
        | ~scalar_valid
    )
    canvas_correspondence = (
        np.asarray(transformed["source_correspondence_mask"], dtype=bool)
        & transformed_validity["source_correspondence_mask"]
        & canvas_tissue
        & canvas_dense_valid
        & (canvas_weight > 0.0)
        & ~canvas_abstention
    )
    appearance_rng: dict[str, dict[str, object]] = {}
    appearance_tissue = canvas_tissue & scalar_valid & label_valid
    appearance_arrays, appearance_parameters = _appearance(
        np.ascontiguousarray(
            np.where(scalar_valid, transformed["source_scalar_float32"], 0),
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            np.where(label_valid, transformed["source_label_int64"], 0),
            dtype=np.int64,
        ),
        np.ascontiguousarray(appearance_tissue),
        modality,
        provenance,
        appearance_rng,
        priors,
    )
    damage_arrays = _canvas_damage(
        appearance_arrays["pre_damage_acquired_image_float32"],
        appearance_arrays["acquired_background_float32"],
        appearance_tissue,
        canvas_correspondence,
        canvas_weight,
        transformed,
        parent_damage_parameters,
    )
    arrays = {
        "source_scalar_canvas_float32": np.ascontiguousarray(
            np.where(scalar_valid, transformed["source_scalar_float32"], 0),
            dtype=np.float32,
        ),
        "source_label_ground_truth_canvas_int64": np.ascontiguousarray(
            np.where(label_valid, transformed["source_label_int64"], 0),
            dtype=np.int64,
        ),
        "source_tissue_ground_truth_mask": np.ascontiguousarray(canvas_tissue),
        "source_correspondence_domain_mask": np.ascontiguousarray(
            canvas_correspondence
        ),
        "source_dense_correspondence_weight_float32": canvas_weight,
        "source_dense_correspondence_abstention_mask": np.ascontiguousarray(
            canvas_abstention
        ),
        "processed_mapped_ccf_physical_coordinates_canvas_float64": mapped_canvas,
        "processed_bilinear_domain_valid_mask": np.ascontiguousarray(
            np.asarray(transformed["source_bilinear_valid_mask"], dtype=bool)
            & transformed_validity["source_bilinear_valid_mask"]
            & scalar_valid
            & parent_domain
        ),
        "processed_nearest_domain_valid_mask": np.ascontiguousarray(
            np.asarray(transformed["source_nearest_valid_mask"], dtype=bool)
            & transformed_validity["source_nearest_valid_mask"]
            & parent_domain
        ),
        "processed_dense_coordinate_valid_mask": np.ascontiguousarray(
            canvas_dense_valid
        ),
        "truth_section_pullback_map_yx_px_float64": np.ascontiguousarray(
            pullback_map_canvas
        ),
        "truth_section_pullback_stationary_velocity_yx_px_float64": (
            np.ascontiguousarray(pullback_velocity_canvas)
        ),
        "truth_section_deformation_valid_mask": np.ascontiguousarray(
            deformation_valid_canvas
        ),
        "parent_sampling_domain_mask": np.ascontiguousarray(parent_domain),
        **appearance_arrays,
        **damage_arrays,
    }
    status = window_realization["retention_audit"]["window_status"]
    empty_canvas_tissue = not bool(canvas_tissue.any())
    abstain_pose = (
        status == "below_0_4_abstention_ood_stress" or empty_canvas_tissue
    )
    pose_supervision = {
        **window_realization["retention_audit"],
        "pose_abstention": abstain_pose,
        "pose_abstention_reason": (
            "empty sampled tissue despite continuous support retention audit"
            if empty_canvas_tissue
            else "retained centre-plane support below 0.4"
            if abstain_pose
            else "none"
        ),
        "empty_canvas_tissue": empty_canvas_tissue,
        "decision_policy": (
            "post-hoc audit only; retain every partial or empty view without rejection or redraw"
        ),
    }
    artifact = {
        "schema_version": OBSERVATION_V3_SCHEMA,
        "algorithm": OBSERVATION_V3_ALGORITHM,
        "implementation_source_sha256": _source_hashes(),
        "runtime_dependencies": {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "asset_dependencies": observation_v2._dependencies(),
        "upstream_reference": upstream_reference,
        "lineage": lineage,
        "rng_provenance": provenance,
        "modality": modality,
        "engineering_priors": priors,
        "parent_authentication_v3": authenticated_parent_v3,
        "acquisition_window_realization": window_realization,
        "parent_damage_geometry": parent_damage_parameters,
        "deformation_pose_gauge": deformation_pose_gauge,
        "pose_supervision": pose_supervision,
        "parameters": {
            "appearance": appearance_parameters,
            "mask_algebra": {
                "damage_geometry": "sampled on full parent then transformed by the one frozen window",
                "physical_loss_application": (
                    "full-parent tissue, scalar, label, correspondence, dense validity, "
                    "dense weight, coordinates, and abstention are modified before windowing; "
                    "retention denominators use untouched pre-damage support"
                ),
                "valid_correspondence": (
                    "windowed source correspondence AND tissue AND strict dense validity "
                    "AND positive source weight AND NOT abstention AND NOT damage"
                ),
                "brush_mask_error_excluded_from_truth": True,
                "deformation_target_gauge": (
                    "project the windowed stationary velocity with the decoder's "
                    "fixed uniform full-canvas affine gauge, factor the removed "
                    "affine flow into the QuickNII frame, and regenerate exp(residual)"
                ),
            },
        },
        "rng_sources": {
            "damage": damage_rng,
            "appearance": appearance_rng,
            "brush": {},
        },
        "arrays": arrays,
        "array_receipts": _array_receipts(arrays),
        "descendant_sampling_policy": _descendant_sampling_policy(),
    }
    artifact["acquired_observation_id"] = acquisition._payload_sha256(
        {
            "domain": "anatomy-tracker.acquired-observation/v3",
            "upstream_reference": artifact["upstream_reference"],
            "lineage": artifact["lineage"],
            "window_realization_id": window_realization[
                "acquisition_window_realization_id"
            ],
            "observation_parent_authentication_id": authenticated_parent_v3[
                "observation_parent_authentication_id"
            ],
            "parent_damage_geometry": parent_damage_parameters,
            "deformation_pose_gauge_receipt_sha256": deformation_pose_gauge[
                "receipt_sha256"
            ],
            "appearance_parameters": appearance_parameters,
            "rng_sources": {
                "damage": damage_rng,
                "appearance": appearance_rng,
            },
            "array_receipts": artifact["array_receipts"],
        }
    )
    raw = arrays["raw_acquired_image_float32"]
    footprint = arrays["observable_footprint_mask"]
    brush_rng: dict[str, dict[str, object]] = {}
    imperfect, imperfect_parameters = _imperfect_brush_mask_v3(
        footprint, provenance, brush_rng, priors
    )
    zeros = np.zeros(footprint.shape, dtype=bool)
    artifact["rng_sources"]["brush"] = brush_rng
    artifact["descendants"] = {
        "raw": _descendant(
            "raw",
            raw,
            zeros,
            zeros,
            artifact["acquired_observation_id"],
            {
                "role": "nontrainable acquired-input audit mirror",
                "equivalent_trainable_mode": "smart-brush-absent",
                "sampling_policy": "never count raw and absent as separate training inputs",
            },
            trainable=False,
        ),
        "smart-brush-accurate": _descendant(
            "smart-brush-accurate",
            raw,
            footprint,
            zeros,
            artifact["acquired_observation_id"],
            {
                "role": "oracle-quality optional smart-brush descendant",
                "selection_source": "windowed observable footprint after physical loss",
                "quality_iou": 1.0,
                "empty_selection": not bool(footprint.any()),
            },
        ),
        "smart-brush-imperfect": _descendant(
            "smart-brush-imperfect",
            raw,
            imperfect,
            imperfect ^ footprint,
            artifact["acquired_observation_id"],
            imperfect_parameters,
        ),
        "smart-brush-absent": _descendant(
            "smart-brush-absent",
            raw,
            zeros,
            zeros,
            artifact["acquired_observation_id"],
            {
                "role": "trainable no-brush descendant; raw background retained exactly",
                "raw_audit_mirror_mode": "raw",
            },
        ),
    }
    artifact["observation_bundle_id"] = acquisition._payload_sha256(
        {
            "domain": "anatomy-tracker.observation-bundle/v3",
            "upstream_reference": artifact["upstream_reference"],
            "lineage": artifact["lineage"],
            "acquired_observation_id": artifact["acquired_observation_id"],
            "descendant_sampling_policy": artifact["descendant_sampling_policy"],
            "descendant_ids": {
                mode: artifact["descendants"][mode]["descendant_id"]
                for mode in DESCENDANT_MODES
            },
            "brush_rng_sources": brush_rng,
        }
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        observation_bundle_receipt_v3(artifact)
    )
    return artifact


def replay_arbitrary_plane_observation_v3(
    artifact: dict[str, object],
    processed_render: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    acquisition_window_plan: dict[str, object],
    **arguments,
) -> dict[str, object]:
    """Replay from the same independently supplied frozen window plan."""
    return make_arbitrary_plane_observation_v3(
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
        acquisition_window_plan,
        **arguments,
    )


def verify_arbitrary_plane_observation_v3(
    artifact: dict[str, object],
    processed_render: dict[str, object],
    subject_slab_render: dict[str, object],
    section_processing_plan: dict[str, object],
    prepared_context: dict[str, object],
    precursor: dict[str, object],
    acquisition_window_plan: dict[str, object],
    **arguments,
) -> None:
    """Verify parent, plan, nested window, live receipts, and byte-exact replay."""
    replay = make_arbitrary_plane_observation_v3(
        processed_render,
        subject_slab_render,
        section_processing_plan,
        prepared_context,
        precursor,
        acquisition_window_plan,
        **arguments,
    )
    if set(artifact) != set(replay):
        raise ValueError("observation v3 has missing or unauthenticated extra fields")
    nested = artifact["acquisition_window_realization"]
    expected_nested = replay["acquisition_window_realization"]
    if (
        set(nested) != set(expected_nested)
        or acquisition_window.acquisition_window_realization_receipt_v3(nested)
        != acquisition_window.acquisition_window_realization_receipt_v3(
            expected_nested
        )
        or nested.get("array_receipts") != _array_receipts(nested.get("arrays", {}))
        or set(nested.get("arrays", {})) != set(expected_nested["arrays"])
        or any(
            not _byte_equal(nested["arrays"][name], expected_nested["arrays"][name])
            for name in expected_nested["arrays"]
        )
    ):
        raise ValueError("nested acquisition-window realization does not replay exactly")
    if (
        artifact.get("array_receipts") != _array_receipts(artifact.get("arrays", {}))
        or artifact.get("receipt_sha256")
        != acquisition._payload_sha256(observation_bundle_receipt_v3(artifact))
        or observation_bundle_receipt_v3(artifact)
        != observation_bundle_receipt_v3(replay)
    ):
        raise ValueError("observation v3 receipt does not replay exactly")
    if set(artifact.get("arrays", {})) != _ARRAY_KEYS or any(
        not _byte_equal(artifact["arrays"][name], replay["arrays"][name])
        for name in _ARRAY_KEYS
    ):
        raise ValueError("observation v3 arrays do not replay byte-exactly")
    if set(artifact.get("descendants", {})) != set(DESCENDANT_MODES):
        raise ValueError("observation v3 descendants are incomplete")
    for mode in DESCENDANT_MODES:
        descendant = artifact["descendants"][mode]
        expected = replay["descendants"][mode]
        if (
            set(descendant.get("arrays", {})) != _DESCENDANT_ARRAY_KEYS
            or descendant.get("array_receipts")
            != _array_receipts(descendant.get("arrays", {}))
            or set(descendant) != set(expected)
            or any(
                not _byte_equal(descendant["arrays"][name], expected["arrays"][name])
                for name in _DESCENDANT_ARRAY_KEYS
            )
        ):
            raise ValueError("observation v3 descendant does not replay byte-exactly")
