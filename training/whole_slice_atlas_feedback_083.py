"""081 atlas matcher with optional local correspondence logits and synthetic targets."""
import math

import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid,
)
from training.whole_slice_atlas_feedback_081 import WholeSliceAtlasFeedback081


class WholeSliceAtlasFeedback083(WholeSliceAtlasFeedback081):
    def forward(self, image_feature, state, reflection, atlas, offsets, weights,
                source_shape=(256, 256), fit_summary=None, return_match_logits=False,
                match_only=False):
        """Optional logits/support use flattened (depth, dy, dx) class order.

        Depths are -6000:1500:6000 um; dy/dx are -2:2 (fine) or -4:4
        (coarse). ``match_only`` returns logits and atlas support at 32²/16²,
        skipping the spatial, pose, quality and mapper heads.
        """
        batch, candidates = state.shape[:2]
        count, side, margin, depths = batch * candidates, 32, 8, 9
        current = state.reshape(count, 12)
        reflected = reflection.reshape(count).bool()
        centre, frame, basis = full_frame_state_to_components(current)
        edges = frame[..., :, :2] @ basis
        axis = (torch.arange(side + 2 * margin, device=state.device,
                             dtype=state.dtype) + .5 - margin) / side
        yy, xx = torch.meshgrid(axis - .5 / source_shape[0],
                                axis - .5 / source_shape[1], indexing='ij')
        chart = torch.stack((
            torch.where(reflected[:, None, None],
                        (source_shape[1] - 1) / source_shape[1] - xx, xx),
            yy.expand(count, -1, -1)), -1)
        base = centre[:, None, None] + torch.einsum('nij,nhwj->nhwi', edges, chart - .5)
        z = torch.linspace(-6000., 6000., depths, device=state.device, dtype=state.dtype)
        offsets = torch.as_tensor(offsets, device=state.device, dtype=state.dtype)
        weights = torch.as_tensor(weights, device=state.device, dtype=state.dtype)
        if offsets.ndim == 1:
            offsets = offsets[None].expand(batch, -1)
        if weights.ndim == 1:
            weights = weights[None].expand(batch, -1)
        offsets = offsets[:, None].expand(-1, candidates, -1).reshape(count, -1)
        weights = weights[:, None].expand(-1, candidates, -1).reshape(count, -1)
        world = base[:, None, None] + (z[None, :, None] + offsets[:, None, :])[
            ..., None, None, None] * frame[:, None, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, world.reshape(count * depths, offsets.shape[-1],
                                 side + 2 * margin, side + 2 * margin, 3),
            (0., 0., 0.), (25., 25., 25.),
            weights[:, None].expand(-1, depths, -1).reshape(count * depths, -1))
        slab = rendered.reshape(count, depths, 2, side + 2 * margin,
                                side + 2 * margin).permute(0, 2, 1, 3, 4)
        support = slab[:, 1:2].clamp(0, 1)
        atlas_pair = torch.cat((slab[:, :1] / support.clamp_min(1e-4), support), 1)
        key = F.normalize(self.atlas(atlas_pair), dim=1)
        query = F.normalize(self.image(F.adaptive_avg_pool2d(
            image_feature, (side, side))), dim=1)
        query = query[:, None].expand(-1, candidates, -1, -1, -1).reshape(
            count, 16, side, side)

        temperature = self.log_temperature.exp().clamp(2, 20)
        coarse_query = F.normalize(F.avg_pool2d(query, 2), dim=1)
        coarse_key = F.normalize(F.avg_pool3d(key, (1, 2, 2)), dim=1)
        coarse_support = F.avg_pool3d(support, (1, 2, 2))
        match_maps = []
        match_logits = {}
        for name, q, k, tissue, radius, origin in (
            ('fine', query, key, support, 2, margin),
            ('coarse', coarse_query, coarse_key, coarse_support, 4, margin // 2),
        ):
            width = q.shape[-1]
            correlation, visible = [], []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y, x = origin + dy, origin + dx
                    patch = k[:, :, :, y:y + width, x:x + width]
                    correlation.append((q[:, :, None] * patch).sum(1))
                    visible.append(tissue[:, 0, :, y:y + width, x:x + width])
            visible = torch.stack(visible, 2)
            scores = temperature * torch.stack(correlation, 2) - 4 * (1 - visible)
            if return_match_logits or match_only:
                match_logits[f'{name}_match_logits'] = scores.flatten(1, 2).reshape(
                    batch, candidates, -1, width, width)
                match_logits[f'{name}_match_support'] = visible.flatten(1, 2).reshape(
                    batch, candidates, -1, width, width)
            if match_only:
                continue
            probability = scores.flatten(1, 2).softmax(1)
            displacement = torch.linspace(-1, 1, 2 * radius + 1,
                                          device=state.device, dtype=state.dtype)
            dy, dx = torch.meshgrid(displacement, displacement, indexing='ij')
            zz = torch.linspace(-1, 1, depths, device=state.device, dtype=state.dtype)
            positions = torch.stack((
                dx.flatten()[None].expand(depths, -1),
                dy.flatten()[None].expand(depths, -1),
                zz[:, None].expand(-1, (2 * radius + 1) ** 2)), 0)
            moment = (probability[:, None] * positions.reshape(
                1, 3, -1, 1, 1)).sum(2)
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(
                1, keepdim=True) / math.log(depths * (2 * radius + 1) ** 2)
            confidence = scores.flatten(1, 2).amax(1, keepdim=True) / temperature
            matched_support = (probability * visible.flatten(1, 2)).sum(1, keepdim=True)
            match_maps.append(F.interpolate(torch.cat((moment, entropy, confidence,
                matched_support), 1), (side, side), mode='bilinear', align_corners=False))
        if match_only:
            return match_logits
        image_axis = (torch.arange(side, device=state.device,
                                   dtype=state.dtype) + .5) / side - .5
        py, px = torch.meshgrid(image_axis, image_axis, indexing='ij')
        position = torch.stack((px, py))[None].expand(count, -1, -1, -1)
        reflection_map = reflected.to(state.dtype)[:, None, None, None].expand(
            -1, 1, side, side)
        if fit_summary is None:
            fit_summary = state.new_zeros(batch, candidates, 4)
        fit_summary = fit_summary.reshape(count, 4)
        fit_map = fit_summary[:, :, None, None].expand(-1, -1, side, side)
        spatial = self.spatial(torch.cat((
            query, key[:, :, depths // 2, margin:-margin, margin:-margin],
            *match_maps, position,
            reflection_map, fit_map), 1))
        descriptor = torch.cat((
            self.summary(spatial), centre / 10000, frame[..., :, 2],
            current[:, 9:11] - math.log(12000.), current[:, 11:12],
            reflected.to(state.dtype)[:, None], fit_summary), -1)
        limits = current.new_tensor((.35, .35, .35, 3000., 3000., 3000.,
                                     math.log(1.25), math.log(1.25), .2))
        update = self.pose(descriptor).tanh() * limits
        with torch.autocast(device_type=state.device.type, enabled=False):
            corrected = compose_full_frame_state(current, update.to(current.dtype))
        output = {
            'state': corrected.reshape(batch, candidates, 12),
            'update': update.reshape(batch, candidates, 9),
            'quality_logit': self.quality(descriptor).reshape(batch, candidates),
            'spatial_evidence': self.mapper_feature(spatial).reshape(
                batch, candidates, 64, side, side),
        }
        if return_match_logits:
            output.update(match_logits)
        return output


def synthetic_match_targets(centre_ccf_um, valid_mask, state, reflection):
    """Nearest-cell CE labels for exact observed-pixel-to-CCF correspondence.

    Inputs are (B,H,W,3), (B,H,W), (B,K,12), (B,K). Returns fine/coarse
    indices and masks of (B,K,32,32)/(B,K,16,16). Index order is
    ``depth * (2r+1)^2 + (dy+r) * (2r+1) + (dx+r)``. Each CCF query is
    bilinearly sampled at the image block centre; its four contributing valid
    pixels must all be valid. A label is masked if its local chart displacement
    exceeds radius + 0.5 cells on either axis or normal distance exceeds
    6750 um (6000 + half a 1500-um depth step). This quantization-support
    mask is wider than 082's strict radius/6000-um envelope.
    """
    batch, height, width = valid_mask.shape
    candidates = state.shape[1]
    reflected = reflection.bool()
    centre, frame, basis = full_frame_state_to_components(state)
    edges = frame[..., :, :2] @ basis
    result = {}
    for name, side, radius in (('fine', 32, 2), ('coarse', 16, 4)):
        target = F.interpolate(centre_ccf_um.permute(0, 3, 1, 2), (side, side),
                               mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        valid = F.interpolate(valid_mask[:, None].to(centre_ccf_um.dtype),
                              (side, side), mode='bilinear', align_corners=False)[:, 0] == 1
        y = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / height
        x = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / width
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        chart_x = torch.where(reflected[..., None, None], (width - 1) / width - xx, xx)
        chart = torch.stack((chart_x,
            yy.expand(batch, candidates, side, side)), -1)
        base = centre[..., None, None, :] + torch.einsum(
            'bkij,bkhwj->bkhwi', edges, chart - .5)
        residual = target[:, None] - base
        local = torch.einsum('bkhwi,bkij->bkhwj', residual, frame[..., :, :2])
        normal = torch.einsum('bkhwi,bki->bkhw', residual, frame[..., :, 2])
        chart_delta = torch.linalg.solve(
            basis[..., None, None, :, :], local[..., None]).squeeze(-1)
        sign = torch.where(reflected, -1., 1.)
        dx_cell = chart_delta[..., 0] * sign[..., None, None] * side
        dy_cell = chart_delta[..., 1] * side
        displacements = torch.arange(-radius, radius + 1, device=state.device,
                                     dtype=state.dtype)
        dy, dx = torch.meshgrid(displacements, displacements, indexing='ij')
        offsets = torch.stack((dx.flatten(), dy.flatten()), -1)
        offsets = offsets[None, None] * torch.stack((sign, torch.ones_like(sign)), -1)[
            ..., None, :] / side
        physical = torch.einsum('bkij,bknj->bkni', basis, offsets)
        spatial = (local[..., None, :] - physical[:, :, None, None]).square().sum(-1).argmin(-1)
        depth = (normal / 1500 + 4).round().clamp(0, 8).long()
        result[f'{name}_index'] = depth * (2 * radius + 1) ** 2 + spatial
        result[f'{name}_mask'] = (valid[:, None] & (dx_cell.abs() <= radius + .5)
                                  & (dy_cell.abs() <= radius + .5) & (normal.abs() <= 6750))
    return result
