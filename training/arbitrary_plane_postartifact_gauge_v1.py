"""V9 images and dense truth with a post-artifact central-surface pose gauge."""

import copy
import hashlib
import math
from pathlib import Path

import torch

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
    full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import (
    frame_to_physical_ouv,
    physical_ouv_to_frame,
)
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import (
    sample_one_shot_slide_artifacts_v9,
)


VERSION = 'postartifact_gauge_v1'
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _refit_postartifact_gauge(sample, side, source_supplied=False):
    pre_artifact_state = sample['state'].clone()
    provenance = copy.deepcopy(sample['provenance'])
    fitted_states, fit_applied = [], []
    for row, record in enumerate(provenance):
        mask = sample['valid_mask'][row] & ~sample['fragment_displaced_mask'][row]
        y, x = torch.nonzero(mask, as_tuple=True)
        canonical_x = x.to(torch.float64) / side
        if bool(sample['reflection'][row]):
            canonical_x = (side - 1) / side - canonical_x
        design = torch.stack((torch.ones_like(canonical_x), canonical_x,
                              y.to(torch.float64) / side), -1)
        target = sample['centre'][row][mask].to(torch.float64)
        gram = design.T @ design
        applied = len(design) >= 3 and int(torch.linalg.matrix_rank(gram)) == 3
        condition, residual_mean, residual_median = None, None, None
        normal_change, roundtrip_error, coefficients_list = None, None, None
        if applied:
            coefficients = torch.linalg.solve(gram, design.T @ target)
            weight = torch.ones(len(design), device=design.device, dtype=design.dtype)
            for _ in range(3):
                residual = torch.linalg.vector_norm(target - design @ coefficients, dim=-1)
                median = torch.quantile(residual, .5)
                scale = torch.maximum(1.4826 * torch.quantile((residual - median).abs(), .5),
                                      residual.new_tensor(25.))
                cutoff = 1.345 * scale
                weight = torch.minimum(torch.ones_like(residual),
                                       cutoff / residual.clamp_min(torch.finfo(residual.dtype).eps))
                gram = design.T @ (weight[:, None] * design)
                coefficients = torch.linalg.solve(gram, design.T @ (weight[:, None] * target))
            new_state = full_frame_state_from_components(
                *physical_ouv_to_frame(coefficients[None]))[0].to(pre_artifact_state)
            center, frame, basis = full_frame_state_to_components(new_state[None])
            assert bool(torch.isfinite(new_state).all())
            assert float(torch.linalg.det(frame)[0]) > 0
            assert bool((torch.linalg.diagonal(basis[0]) > 0).all())
            roundtrip = frame_to_physical_ouv(center, frame, basis).reshape(3, 3)
            roundtrip_error = float((roundtrip - coefficients.to(roundtrip)).abs().max())
            assert roundtrip_error < .05
            residual = torch.linalg.vector_norm(target - design @ coefficients, dim=-1)
            condition = float(torch.sqrt(torch.linalg.cond(gram)))
            residual_mean = float(residual.mean())
            residual_median = float(torch.quantile(residual, .5))
            old_normal = full_frame_state_to_components(
                pre_artifact_state[row:row + 1])[1][0, :, 2]
            new_normal = frame[0, :, 2]
            normal_change = math.degrees(math.acos(float(
                (old_normal @ new_normal).abs().clamp(-1, 1))))
            coefficients_list = coefficients.tolist()
        else:
            new_state = pre_artifact_state[row].clone()
        fitted_states.append(new_state)
        fit_applied.append(applied)
        record[VERSION] = {
            'code_sha256': CODE_SHA256,
            'source': ('caller-provided v9 sample; no new draw'
                       if source_supplied else
                       'one exact v9 draw; no retry, resampling, or appearance pairing'),
            'pose_label': 'robust post-artifact central-surface affine fit',
            'fit_mask': 'valid_mask AND NOT fragment_displaced_mask',
            'canonical_chart': 'x/side reflected back with (side-1)/side-x; y/side',
            'fit': 'OLS then three Huber IRLS updates; scale=max(1.4826*MAD(norm residual),25 um); cutoff=1.345*scale',
            'fit_applied': applied,
            'fit_pixels': len(design),
            'weighted_design_condition_number': condition,
            'residual_mean_um': residual_mean,
            'residual_median_um': residual_median,
            'pre_to_post_unsigned_normal_angle_deg': normal_change,
            'ouv_roundtrip_max_abs_um': roundtrip_error,
            'robust_post_artifact_ouv_um': coefficients_list,
            'pre_artifact_state12': pre_artifact_state[row].tolist(),
            'post_artifact_state12': new_state.tolist(),
            'dense_truth': 'unchanged v9 observed-pixel to CCF coordinates',
            'finite_thickness': 'offsets and PSF weights unchanged; refit normal is from the central surface, not a separately fitted post-artifact slab director',
        }
    fit_applied = torch.tensor(fit_applied, device=sample['eligible'].device,
                               dtype=torch.bool)
    assert bool((~sample['eligible'] | fit_applied).all())
    return {**sample, 'state': torch.stack(fitted_states), 'provenance': provenance,
            'pre_artifact_state': pre_artifact_state,
            'post_artifact_pose_fit_applied': fit_applied}


def sample_postartifact_gauge_v1(
        context, subject_indices, seed, side=192, source_sample=None):
    """Refit a supplied v9 sample, or draw v9 once and then refit it."""
    sample = (sample_one_shot_slide_artifacts_v9(context, subject_indices, seed, side)
              if source_sample is None else source_sample)
    return _refit_postartifact_gauge(sample, side, source_sample is not None)
