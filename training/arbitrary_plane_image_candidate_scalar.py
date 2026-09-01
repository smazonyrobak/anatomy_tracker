"""Scalar adapter for the frozen arbitrary-plane candidate banks."""

import numpy as np

from training.arbitrary_plane_rendered_generator import (
    _render_finite_arbitrary_plane_trusted,
    _validate_prepared_context,
)


def render_candidate_scalar(prepared_render_context, candidate, finite_parent):
    _validate_prepared_context(prepared_render_context)
    if candidate["geometry_storage"] == "truth_parent_geometry":
        geometry = finite_parent["geometry"]
    elif candidate["geometry_storage"] == "candidate":
        geometry = candidate["geometry"]
    else:
        raise ValueError("unknown frozen candidate geometry storage")
    rendered = _render_finite_arbitrary_plane_trusted(
        prepared_render_context["scalar_tensor"],
        prepared_render_context["annotation_tensor"],
        geometry,
    )
    expected_annotation = np.asarray(candidate["rendered_annotation"])
    expected_support = np.asarray(candidate["brain_mask"])
    if (
        rendered["annotation"].dtype != expected_annotation.dtype
        or not np.array_equal(rendered["annotation"], expected_annotation)
        or expected_support.dtype != np.bool_
        or rendered["brain_mask"].dtype != expected_support.dtype
        or not np.array_equal(rendered["brain_mask"], expected_support)
        or rendered["array_receipts"]["annotation"]
        != candidate["render_array_receipts"]["annotation"]
        or rendered["array_receipts"]["brain_mask"]
        != candidate["render_array_receipts"]["brain_mask"]
    ):
        raise ValueError("candidate scalar render did not reproduce its frozen annotation/support")
    if rendered["scalar"].dtype != np.float32:
        raise ValueError("candidate scalar renderer must return float32")
    return rendered


def render_candidate_bank_scalars(prepared_render_context, candidate_bank, finite_parent):
    candidates = candidate_bank["candidates"]
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if (
        len(candidates) != 40
        or len(set(candidate_ids)) != 40
        or candidate_ids != list(candidate_bank["ordered_candidate_ids"])
    ):
        raise ValueError("candidate bank order/count does not match its frozen identity")
    if candidate_bank["truth_parent_geometry"] != finite_parent["geometry"]:
        raise ValueError("candidate bank truth geometry does not match the finite parent")
    rendered = [
        render_candidate_scalar(prepared_render_context, candidate, finite_parent)
        for candidate in candidates
    ]
    return {
        "candidate_ids": candidate_ids,
        "scalar_float32": np.stack([item["scalar"] for item in rendered]),
        "annotation": np.stack([item["annotation"] for item in rendered]),
        "brain_mask": np.stack([item["brain_mask"] for item in rendered]),
        "array_receipts": [item["array_receipts"] for item in rendered],
    }
