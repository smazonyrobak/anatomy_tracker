"""Unmodified 120 matcher with optional in-path attention and visual ablations."""

import math

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.global_plane_matcher_120 import global_plane_match


def global_plane_match_128(model, prediction, mode_index, reflection, offsets, weights,
                           atlas, image_shape, side=24, support_only=False,
                           candidate_state=None, matcher=None, return_attention=False,
                           return_support=False, ablation=None, shuffle_seed=0):
    if not return_attention and not return_support and ablation is None:
        return global_plane_match(model, prediction, mode_index, reflection,
            offsets, weights, atlas, image_shape, side, support_only,
            candidate_state, matcher)

    towers = model.global_plane_matcher_120 if matcher is None else matcher
    captured = {}

    def source_hook(module, inputs, output):
        captured['source_output'] = output

    def atlas_hook(module, inputs, output):
        captured['atlas_input'], captured['atlas_output'] = inputs[0], output

    handles = [towers['source'].register_forward_hook(source_hook),
               towers['atlas'].register_forward_hook(atlas_hook)]
    if ablation in ('geometry_only', 'source_features_zero'):
        def zero_source(module, inputs):
            x = inputs[0].clone()
            x[:, :64] = 0
            return (x,)
        handles.append(towers['source'].register_forward_pre_hook(zero_source))
    if ablation in ('geometry_only', 'atlas_intensity_zero', 'atlas_intensity_shuffle'):
        def change_atlas(module, inputs):
            x = inputs[0].clone()
            if ablation == 'atlas_intensity_shuffle':
                generator = torch.Generator().manual_seed(shuffle_seed)
                permutation = torch.randperm(side * side, generator=generator).to(x.device)
                x[:, 0] = x[:, 0].flatten(1)[:, permutation].reshape(-1, side, side)
            else:
                x[:, 0] = 0
            return (x,)
        handles.append(towers['atlas'].register_forward_pre_hook(change_atlas))
    assert ablation in (None, 'geometry_only', 'source_features_zero',
                        'atlas_intensity_zero', 'atlas_intensity_shuffle')
    try:
        result = global_plane_match(model, prediction, mode_index, reflection,
            offsets, weights, atlas, image_shape, side, support_only,
            candidate_state, matcher)
    finally:
        for handle in handles:
            handle.remove()
    if return_attention or return_support:
        batch, count = mode_index.shape
        atlas_input = captured['atlas_input'].reshape(batch, count, 7, side, side)
        result['atlas_support'] = atlas_input[:, :, 1]
    if return_attention:
        query = captured['source_output'].flatten(2).transpose(1, 2)
        key = captured['atlas_output'].flatten(2).transpose(1, 2)
        result['attention_logits'] = (query @ key.transpose(1, 2)
            / math.sqrt(query.shape[-1])).reshape(batch, count, side * side, side * side)
        result['atlas_midplane_ccf'] = (atlas_input[:, :, 4:7]
            * model.center_scale[None, None, :, None, None]
            + model.center_origin[None, None, :, None, None]).permute(0, 1, 3, 4, 2)
    return result


def in_path_correspondence_ce_128(match, state, reflection, target_centre,
                                   valid_mask, offsets, positive_mask=None):
    logits = match['attention_logits']
    batch, count, cells, _ = logits.shape
    side = math.isqrt(cells)
    source_side = valid_mask.shape[-1]
    with torch.no_grad():
        target = F.interpolate(target_centre.permute(0, 3, 1, 2),
            (side, side), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        validity = valid_mask[:, None].float()
        intact = ((F.adaptive_avg_pool2d(validity, (side, side))[:, None, 0] >= .95)
            & (F.interpolate(validity, (side, side), mode='bilinear',
                align_corners=False)[:, None, 0] >= 1 - 1e-6))
        center, frame, basis = full_frame_state_to_components(state.detach())
        delta = target[:, None] - center[:, :, None, None]
        local = torch.einsum('bkhwi,bkij->bkhwj', delta, frame[..., :, :2])
        raster = .5 + torch.linalg.solve(basis[:, :, None, None],
                                         local[..., None]).squeeze(-1)
        key_x = (torch.where(reflection[:, :, None, None].bool(),
            (source_side - 1) / source_side - raster[..., 0], raster[..., 0])
            + .5 / source_side) * side - .5
        key_y = (raster[..., 1] + .5 / source_side) * side - .5
        in_grid = ((key_x >= 0) & (key_x < side - 1)
                   & (key_y >= 0) & (key_y < side - 1))
        grid = torch.stack((2 * key_x / (side - 1) - 1,
                            2 * key_y / (side - 1) - 1), -1)
        support = F.grid_sample(match['atlas_support'].detach().reshape(
            batch * count, 1, side, side), grid.reshape(batch * count, side, side, 2),
            mode='bilinear', padding_mode='zeros', align_corners=True).reshape(
                batch, count, side, side)
        normal = torch.einsum('bkhwi,bki->bkhw', delta, frame[..., :, 2]).abs()
        width = torch.as_tensor(offsets, device=state.device, dtype=state.dtype).abs().amax(-1)
        if width.ndim == 0:
            width = width[None].expand(batch)
        mask = (intact & in_grid & (support >= .8)
                & (normal <= width[:, None, None, None] + 12.5))
        if positive_mask is not None:
            mask &= positive_mask[:, :, None, None]
        x0 = key_x.floor().long().clamp(0, side - 2)
        y0 = key_y.floor().long().clamp(0, side - 2)
        dx, dy = (key_x - x0).clamp(0, 1), (key_y - y0).clamp(0, 1)
        indices = torch.stack((y0 * side + x0, y0 * side + x0 + 1,
            (y0 + 1) * side + x0, (y0 + 1) * side + x0 + 1), -1).flatten(2, 3)
        weights = torch.stack(((1 - dx) * (1 - dy), dx * (1 - dy),
            (1 - dx) * dy, dx * dy), -1).flatten(2, 3)
        valid = mask.flatten(2)
    log_prob = F.log_softmax(logits, -1).gather(-1, indices)
    cross_entropy = -(weights * log_prob).sum(-1)
    pixel_count = valid.sum(-1)
    candidate_loss = (cross_entropy * valid).sum(-1) / pixel_count.clamp_min(1)
    candidate_has_target = pixel_count > 0
    section_loss = ((candidate_loss * candidate_has_target).sum(-1)
                    / candidate_has_target.sum(-1).clamp_min(1))
    return {'loss': section_loss.mean(), 'pixels_per_candidate': pixel_count,
            'candidates_per_section': candidate_has_target.sum(-1)}
