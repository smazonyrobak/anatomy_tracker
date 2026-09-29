"""Offline exact subject-coordinate targets; no sampler or learned model."""

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_subject_deformation_v2 import _subject_to_ccf_points_from_verified_plan_v2
from training.arbitrary_plane_subject_section_v2 import fit_subject_centre_plane_and_residual_v2


def make_coherent_subject_section_v6(
    subject_plan,
    subject_ouv_ap_dv_ml_um,
    observed_pullback_yx_px,
    reflection_xy,
    axial_offsets_um,
    axial_weights,
    section_identifiers,
    source_identifiers,
    volume_c_ap_dv_ml,
    origin_ap_dv_ml_um,
    voxel_size_ap_dv_ml_um,
    *,
    mapping_batch_size=8192,
):
    """Map one section through a caller-frozen, accepted v2 animal plan.

    OUV is physical AP/DV/ML [3,3] (or flattened9); the absolute observed-raster
    pullback is y/x [2,H,W]. Boolean reflection_xy flags mean horizontal/vertical.
    PSF offsets/positive weights are [S], with an exact zero centre offset.
    Compose observed -> observed pullback -> finite reflection -> subject plane
    + offset*subject normal -> subject-to-CCF. Do not conjugate the map again.

    This CPU NumPy generator is not differentiable through the v2 subject flow.
    Authenticate a loaded plan once at the caller's boundary, or use the frozen
    accepted result of sample_animal_subject_deformation_plan_v2. No plan replay,
    sampling, rejection, mask, crop, damage or appearance augmentation occurs here.
    The returned raw image samples the continuous surface, not a bilinear warp of
    an already-rasterized image. Pullback-domain validity is exported separately.
    Separate full-canvas fits describe unprocessed/unreflected canonical anatomy
    and the total observed map. Neither residual is a 2D stationary velocity.
    """
    for key, expected in (
        ("animal_id", subject_plan["provenance"]["animal_id"]),
        ("split", subject_plan["provenance"]["split"]),
        ("synthetic_animal_id", subject_plan["synthetic_animal_id"]),
    ):
        if section_identifiers[key] != expected:
            raise ValueError(f"section {key} differs from its shared subject plan")
    ouv = np.asarray(subject_ouv_ap_dv_ml_um, dtype=np.float64).reshape(3, 3)
    pullback = np.asarray(observed_pullback_yx_px, dtype=np.float64)
    height, width = pullback.shape[-2:]
    flags = np.asarray(reflection_xy, dtype=bool)
    signs = 1.0 - 2.0 * flags
    xy = np.moveaxis(pullback[::-1], 0, -1)
    canonical_xy = xy * signs + flags * np.asarray((width - 1, height - 1))
    centre_subject = (
        ouv[0] + canonical_xy[..., 0:1] / width * ouv[1]
        + canonical_xy[..., 1:2] / height * ouv[2]
    )
    normal = np.cross(ouv[1], ouv[2])
    normal /= np.linalg.norm(normal)
    offsets = np.asarray(axial_offsets_um, dtype=np.float64)
    weights = np.asarray(axial_weights, dtype=np.float64)
    weights = weights / weights.sum()
    centre_index = int(np.flatnonzero(offsets == 0.0)[0])
    subject_grid = centre_subject[None] + offsets[:, None, None, None] * normal
    y, x = np.meshgrid(np.arange(height) / height, np.arange(width) / width, indexing="ij")
    canonical_subject = ouv[0] + x[..., None] * ouv[1] + y[..., None] * ouv[2]
    mapped = np.ascontiguousarray(_subject_to_ccf_points_from_verified_plan_v2(
        np.concatenate((canonical_subject[None], subject_grid)), subject_plan,
        batch_size=mapping_batch_size,
    ), dtype=np.float64)
    canonical_ccf, ccf_grid = mapped[0], mapped[1:]
    centre_ccf = ccf_grid[centre_index].copy()
    canonical_fit = fit_subject_centre_plane_and_residual_v2(canonical_ccf)
    fit = fit_subject_centre_plane_and_residual_v2(centre_ccf)
    raw_image = render_finite_thickness_coordinate_grid(
        volume_c_ap_dv_ml, torch.from_numpy(ccf_grid)[None],
        origin_ap_dv_ml_um, voxel_size_ap_dv_ml_um, torch.from_numpy(weights),
    )[0]
    return {
        "lineage": dict(section_identifiers),
        "source_identifiers": dict(source_identifiers),
        "subject_reference": {
            "subject_deformation_plan_id": subject_plan["subject_deformation_plan_id"],
            "subject_deformation_realization_id": subject_plan["subject_deformation_realization_id"],
            "subject_plan_receipt_sha256": subject_plan["receipt_sha256"],
            "ccf_context_sha256": subject_plan["provenance"]["ccf_context_sha256"],
            "subject_root_seed_uint64": subject_plan["provenance"]["root_seed_uint64"],
            "animal_index": subject_plan["provenance"]["animal_index"],
            "mapping_direction": "subject physical AP/DV/ML um to CCF physical AP/DV/ML um",
        },
        "subject_ouv_ap_dv_ml_um_float64": ouv.copy(),
        "section_processing_observed_pullback_yx_px_float64": pullback.copy(),
        "reflection_xy": flags.copy(),
        "axial_offsets_um_float64": offsets.copy(),
        "axial_weights_float64": weights,
        "centre_psf_index": centre_index,
        "canonical_anatomy_centre_ccf_ap_dv_ml_um_float64": canonical_ccf,
        "canonical_anatomy_plane_fit": canonical_fit,
        "subject_psf_coordinates_ap_dv_ml_um_float64": subject_grid,
        "target_psf_ccf_coordinates_ap_dv_ml_um_float64": ccf_grid,
        "target_centre_ccf_coordinates_ap_dv_ml_um_float64": centre_ccf,
        "observed_total_map_plane_fit": fit,
        "section_pullback_in_raster_domain": ((xy >= 0) & (xy <= (width - 1, height - 1))).all(-1),
        "raw_rendered_channels": raw_image,
    }
