"""Joint arbitrary-plane correction and tissue mapping for synthetic training."""

import torch
import torch.nn.functional as F

from training.global_plane_matcher_120 import global_plane_match
from training.global_atlas_contrast_090 import rigid_points_090
from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.whole_slice_atlas_feedback_083 import synthetic_match_targets


def joint_forward_121(model, prediction, inputs, mode_index, reflection, atlas, offsets, weights,
                      *, matcher=None, candidate_state=None, match_side=24, map_side=96,
                      support_only=False):
    """Map corrected blind branches; no mask or synthetic target enters inference."""
    match = global_plane_match(model, prediction, mode_index, reflection, offsets, weights,
                               atlas, inputs.shape[-2:], side=match_side,
                               support_only=support_only, candidate_state=candidate_state,
                               matcher=matcher)
    batch, count = mode_index.shape
    local_index = torch.arange(count, device=mode_index.device)[None].expand(batch, -1)
    mapped = model.map({**prediction, 'state': match['state']}, offsets, local_index,
                       reflection, (map_side, map_side), atlas, weights,
                       source_shape=inputs.shape[-2:])
    mapped['mode_index'] = mode_index
    return {'match': match, 'mapped': mapped,
            'unshifted_action': {'state': match['input_state'],
                                 'score': match['input_score']}}


def joint_fit_loss_121(inputs, mapped, atlas, weights, valid_mask, *, side=64):
    """Polarity-invariant tissue LNCC, with atlas coverage measured separately.

    `valid_mask` is synthetic training truth only. Zero common support has unit
    fit loss, never a free perfect match. Combine coverage_penalty in training.
    """
    batch, count, samples, height, width = mapped['coordinates'].shape[:5]
    coordinates = mapped['coordinates'].flatten(0, 1).permute(0, 1, 4, 2, 3)
    coordinates = F.interpolate(coordinates.reshape(batch * count, samples * 3, height, width),
                                (side, side), mode='bilinear', align_corners=False)
    coordinates = coordinates.reshape(batch * count, samples, 3, side, side).permute(0, 1, 3, 4, 2)
    mass = torch.as_tensor(weights, device=inputs.device, dtype=inputs.dtype)
    if mass.ndim == 1:
        mass = mass[None].expand(batch, -1)
    mass = mass[:, None].expand(-1, count, -1).flatten(0, 1)
    rendered = render_finite_thickness_coordinate_grid(
        atlas, coordinates, (0., 0., 0.), (25., 25., 25.), mass)
    support = rendered[:, 1:2].clamp(0, 1)
    source = F.interpolate(inputs[:, :1], (side, side), mode='area')
    source = source[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
    target = rendered[:, :1] / support.clamp_min(1e-4)
    tissue = F.interpolate(valid_mask[:, None].to(source.dtype), (side, side), mode='area')
    tissue = tissue[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
    weight = tissue * support.detach()
    amount = F.avg_pool2d(weight, 9, 1, 4).clamp_min(1e-4)
    mean_source = F.avg_pool2d(weight * source, 9, 1, 4) / amount
    mean_target = F.avg_pool2d(weight * target, 9, 1, 4) / amount
    covariance = F.avg_pool2d(weight * source * target, 9, 1, 4) / amount - mean_source * mean_target
    source_var = (F.avg_pool2d(weight * source.square(), 9, 1, 4) / amount
                  - mean_source.square()).clamp_min(0)
    target_var = (F.avg_pool2d(weight * target.square(), 9, 1, 4) / amount
                  - mean_target.square()).clamp_min(0)
    correlation = covariance / (source_var * target_var + 1e-5).sqrt()
    patch_weight = F.avg_pool2d(weight, 9, 1, 4).detach()
    total = patch_weight.sum((1, 2, 3))
    fit = ((1 - correlation.abs().clamp(0, 1)) * patch_weight).sum((1, 2, 3))
    fit = torch.where(total > 0, fit / total.clamp_min(1e-4), torch.ones_like(fit))
    coverage = (tissue * support).sum((1, 2, 3)) / tissue.sum((1, 2, 3)).clamp_min(1)
    return {'fit_loss': fit.reshape(batch, count),
            'coverage_penalty': (1 - coverage).reshape(batch, count),
            'atlas_coverage': coverage.reshape(batch, count)}


def joint_target_loss_121(mapped, valid_mask, centre_target, truth_state, truth_reflection,
                          site_indices, *, source_shape=(256, 256)):
    """Valid original-pixel sites; CCF target includes warp, rigid target does not."""
    batch, count = mapped['state'].shape[:2]
    sites = site_indices.shape[1]
    source_height, source_width = source_shape
    y = site_indices.div(source_width, rounding_mode='floor')
    x = site_indices.remainder(source_width)
    grid = torch.stack((2 * (x + .5) / source_width - 1,
                        2 * (y + .5) / source_height - 1), -1).to(centre_target.dtype)
    predicted = mapped['centre_surface_ccf_ap_dv_ml_um'].flatten(0, 1).permute(0, 3, 1, 2)
    grid = grid[:, None].expand(-1, count, -1, -1).reshape(batch * count, sites, 1, 2)
    sampled = F.grid_sample(predicted, grid, align_corners=False, padding_mode='border')
    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(batch, count, sites, 3)
    true_map = centre_target.reshape(batch, -1, 3).gather(
        1, site_indices[..., None].expand(-1, -1, 3))
    chart = torch.stack((x / source_width, y / source_height), -1).to(centre_target.dtype)
    true_rigid = rigid_points_090(truth_state, truth_reflection, chart, source_shape)
    predicted_rigid = rigid_points_090(mapped['state'], mapped['reflection'],
                                       chart[:, None], source_shape)
    valid = valid_mask.reshape(batch, -1).gather(1, site_indices).to(centre_target.dtype)
    denominator = valid.sum(-1).clamp_min(1)[:, None]
    map_loss = (((sampled - true_map[:, None]) / 1000).norm(dim=-1) * valid[:, None]
                ).sum(-1) / denominator
    rigid_loss = (((predicted_rigid - true_rigid[:, None]) / 1000).norm(dim=-1)
                  * valid[:, None]).sum(-1) / denominator
    return {'map_loss': map_loss, 'rigid_loss': rigid_loss}


def joint_spatial_ce_121(spatial_head, prediction, state, reflection, atlas, offsets,
                         weights, centre_target, valid_mask, *, source_shape=(256, 256)):
    """Optional 083 correspondence CE for selected candidates, not the full beam."""
    logits = spatial_head(prediction['feature'], state, reflection, atlas, offsets,
                          weights, source_shape=source_shape, match_only=True)
    target = synthetic_match_targets(centre_target, valid_mask, state, reflection)
    result = {}
    for level in ('fine', 'coarse'):
        score = logits[f'{level}_match_logits']
        mask = target[f'{level}_mask']
        cost = F.cross_entropy(score.flatten(0, 1), target[f'{level}_index'].flatten(0, 1),
                               reduction='none')
        result[f'{level}_ce'] = (cost * mask.flatten(0, 1)).sum() / mask.sum().clamp_min(1)
        result[f'{level}_count'] = mask.sum()
    return result
