"""Conservative subject-space coverage; keep empty/marginal draws, never retry."""

import numpy as np


def subject_support_bounds_um(subject_plan):
    """Enclose the mapped full CCF box using the accepted global speed bound."""
    state = subject_plan["state"]
    speed = np.asarray(state["accepted_component_speed_abs_bound_um"])
    centre, scale = state["frozen_center_um"], state["global_scale"]
    lower = centre + scale * (state["full_ccf_lower_um"] - speed - centre)
    upper = centre + scale * (state["full_ccf_upper_um"] + speed - centre)
    return np.stack((lower, upper))


def sample_subject_planes(subject_plan, rng, count, shape_h_w, axial_offsets_um):
    """Uniform RP2 normals/roll and conditional box-slab offsets; no tissue filtering.

    Caller supplies an authenticated frozen plan, NumPy Generator, H/W > 1,
    and the finite PSF offsets in physical subject micrometres. Returned OUV
    uses x/W and y/H, matching the existing physical raster convention.
    """
    bounds = subject_support_bounds_um(subject_plan)
    centre, half_extent = bounds.mean(0), np.diff(bounds, axis=0)[0] / 2
    normal = rng.normal(size=(count, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    normal *= np.where(
        normal[np.arange(count), np.abs(normal).argmax(1)] < 0, -1, 1
    )[:, None]
    reference = np.eye(3)[np.abs(normal).argmin(1)]
    u0 = reference - (reference * normal).sum(1, keepdims=True) * normal
    u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
    v0 = np.cross(normal, u0)
    roll = rng.uniform(-np.pi, np.pi, count)
    u = np.cos(roll)[:, None] * u0 + np.sin(roll)[:, None] * v0
    v = np.cross(normal, u)
    radius = np.abs(normal) @ half_extent
    offset_interval = np.stack(
        (-radius - np.max(axial_offsets_um), radius - np.min(axial_offsets_um)), 1
    )
    offset = rng.uniform(offset_interval[:, 0], offset_interval[:, 1])
    height, width = shape_h_w
    basis_u = u * (2 * (np.abs(u) @ half_extent) * width / (width - 1))[:, None]
    basis_v = v * (2 * (np.abs(v) @ half_extent) * height / (height - 1))[:, None]
    frame_centre = centre + offset[:, None] * normal
    origin = (frame_centre - (width - 1) / (2 * width) * basis_u
              - (height - 1) / (2 * height) * basis_v)
    return {
        "physical_ouv_ap_dv_ml_um": np.stack((origin, basis_u, basis_v), 1),
        "normal_ap_dv_ml": normal,
        "signed_offset_from_box_centre_um": offset,
        "offset_interval_um": offset_interval,
        "roll_rad": roll,
        "subject_support_bounds_um": bounds,
        "subject_deformation_plan_id": subject_plan["subject_deformation_plan_id"],
        "subject_deformation_realization_id": subject_plan["subject_deformation_realization_id"],
        "synthetic_animal_id": subject_plan["synthetic_animal_id"],
    }
