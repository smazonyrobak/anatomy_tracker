"""Rigid pose feedback from the 083 dense atlas correspondence posterior."""

import math

import torch

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, full_frame_state_to_components,
)
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083


class WholeSliceAtlasGeometricFeedback087(WholeSliceAtlasFeedback083):
    def forward(self, image_feature, state, reflection, atlas, offsets, weights,
                source_shape=(256, 256), fit_summary=None, return_match_logits=False,
                match_only=False):
        first_pass = fit_summary is None
        output = super().forward(
            image_feature, state, reflection, atlas, offsets, weights,
            source_shape=source_shape, fit_summary=fit_summary,
            return_match_logits=True, match_only=match_only)
        if match_only:
            return output

        scale, radius = ('coarse', 4) if first_pass else ('fine', 2)
        logits = output[f'{scale}_match_logits'].flatten(0, 1)
        atlas_support = output[f'{scale}_match_support'].flatten(0, 1)
        count, classes, height, width = logits.shape
        probability = logits.softmax(1)
        cells = torch.arange(-radius, radius + 1, device=state.device, dtype=state.dtype)
        dy, dx = torch.meshgrid(cells, cells, indexing='ij')
        depth = torch.linspace(-6., 6., 9, device=state.device, dtype=state.dtype)
        positions = torch.stack((
            dx.flatten()[None].expand(9, -1),
            dy.flatten()[None].expand(9, -1),
            depth[:, None].expand(-1, dx.numel())), -1).reshape(classes, 3)
        moment = torch.einsum('nchw,cd->ndhw', probability, positions)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1) / math.log(classes)
        matched_support = (probability * atlas_support).sum(1)
        weight = (matched_support * (1 - entropy)).reshape(count, -1)

        source_height, source_width = source_shape
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=state.device, dtype=state.dtype) + .5) / height
            - .5 / source_height,
            (torch.arange(width, device=state.device, dtype=state.dtype) + .5) / width
            - .5 / source_width, indexing='ij')
        reflected = reflection.reshape(count).bool()
        chart_x = torch.where(reflected[:, None, None],
                              (source_width - 1) / source_width - xx, xx)
        chart = torch.stack((chart_x - .5, yy.expand(count, -1, -1) - .5), -1)
        _, _, basis = full_frame_state_to_components(state.reshape(count, 12))
        local_xy = torch.einsum('nij,nhwj->nhwi', basis, chart) / 1000
        shift = torch.stack((
            torch.where(reflected[:, None, None], -moment[:, 0], moment[:, 0]) / width,
            moment[:, 1] / height), -1)
        residual_xy = torch.einsum('nij,nhwj->nhwi', basis, shift) / 1000
        residual = torch.cat((residual_xy, moment[:, 2, :, :, None]), -1).reshape(count, -1, 3)

        x, y = local_xy.reshape(count, -1, 2).unbind(-1)
        zero, one = torch.zeros_like(x), torch.ones_like(x)
        jacobian = torch.stack((
            torch.stack((zero, zero, -y, one, zero, zero), -1),
            torch.stack((zero, zero, x, zero, one, zero), -1),
            torch.stack((y, -x, zero, zero, zero, one), -1)), -2)
        normal = torch.einsum('np,npij,npik->njk', weight, jacobian, jacobian)
        right = torch.einsum('np,npij,npi->nj', weight, jacobian, residual)
        normal = normal + .05 * height * width * torch.eye(
            6, device=state.device, dtype=state.dtype)
        step = .5 * torch.linalg.solve(normal, right)
        update = torch.cat((step[:, :3].clamp(-.35, .35),
                            (1000 * step[:, 3:]).clamp(-3000., 3000.),
                            step.new_zeros(count, 3)), -1)
        output['state'] = compose_full_frame_state(
            state.reshape(count, 12), update).reshape_as(state)
        output['update'] = update.reshape(*state.shape[:2], 9)
        if not return_match_logits:
            for name in ('fine', 'coarse'):
                del output[f'{name}_match_logits']
                del output[f'{name}_match_support']
        return output
