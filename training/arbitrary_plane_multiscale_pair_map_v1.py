"""One-checkpoint pose and multiscale image-to-atlas tissue correspondence."""
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_geometry import normalized_raster_to_ccf
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel


class MultiScalePairJointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_model = OneShotJointSliceModel(modes=16, normal_anchor_count=64)
        self.pose_model.requires_grad_(False)
        self.source_stem = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.atlas_stem = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.shared_192 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.shared_96 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.shared_48 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(53, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 2, 3, padding=1),
        )
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)
        with torch.no_grad():
            self.detail[-1].bias[1] = -2
        for side, radius in ((48, 6), (96, 4), (192, 3)):
            yy, xx = torch.meshgrid(torch.arange(-radius, radius + 1),
                                    torch.arange(-radius, radius + 1), indexing='ij')
            self.register_buffer(f'offsets_{side}', torch.stack((xx, yy), -1).reshape(-1, 2).float())

    def _match(self, source, atlas, support, base_flow, side, radius):
        batch, channels, height, width = source.shape
        yy, xx = torch.meshgrid(torch.arange(height, device=source.device, dtype=source.dtype),
                                torch.arange(width, device=source.device, dtype=source.dtype), indexing='ij')
        grid = torch.stack((2 * (xx + base_flow[:, 0]) / (width - 1) - 1,
                            2 * (yy + base_flow[:, 1]) / (height - 1) - 1), -1)
        count = (2 * radius + 1) ** 2
        patches = F.unfold(atlas, 2 * radius + 1, padding=radius).reshape(
            batch, channels * count, height, width)
        shifted = F.grid_sample(patches, grid, padding_mode='zeros', align_corners=True).reshape(
            batch, channels, count, height, width)
        candidate_support = F.unfold(support, 2 * radius + 1, padding=radius).reshape(
            batch, count, height, width)
        candidate_support = F.grid_sample(candidate_support, grid, padding_mode='zeros',
                                          align_corners=True)
        correlation = (F.normalize(source, dim=1, eps=1e-6)[:, :, None] *
                       F.normalize(shifted, dim=1, eps=1e-6)).sum(1)
        logits = (12 * correlation).masked_fill(candidate_support <= .9, -1e4)
        posterior = logits.softmax(1)
        delta = torch.einsum('bkhw,kc->bchw', posterior,
                             getattr(self, f'offsets_{side}').to(posterior))
        flow = base_flow + torch.where((candidate_support > .9).any(1, keepdim=True),
                                       delta, torch.zeros_like(delta))
        return logits, candidate_support, flow, posterior

    def map_at_pose(self, inputs, state, reflection, offsets, weights, atlas):
        batch, _, source_height, source_width = inputs.shape
        side = 192
        center, frame, basis = full_frame_state_to_components(state)
        y = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / source_height
        x = (torch.arange(side, device=state.device, dtype=state.dtype) + .5) / side - .5 / source_width
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        reflected = reflection[:, None, None].bool()
        sx = torch.where(reflected, (source_width - 1) / source_width - xx, xx)
        chart = torch.stack((sx.expand(batch, -1, -1), yy.expand(batch, -1, -1)), -1)
        rigid = normalized_raster_to_ccf(
            center[:, None, None], frame[:, None, None], basis[:, None, None], chart)
        slab = rigid[:, None] + offsets[:, :, None, None, None] * frame[:, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, slab, (0., 0., 0.), (25., 25., 25.), weights)
        support_192 = rendered[:, 1:2]
        atlas_pair = torch.cat((rendered[:, :1] / support_192.clamp_min(1e-4), support_192), 1)
        source_image = F.interpolate(inputs, (side, side), mode='area')
        source_192 = self.shared_192(self.source_stem(source_image))
        atlas_192 = self.shared_192(self.atlas_stem(atlas_pair))
        source_96, atlas_96 = self.shared_96(source_192), self.shared_96(atlas_192)
        source_48, atlas_48 = self.shared_48(source_96), self.shared_48(atlas_96)
        levels = {}
        flow = None
        for size, radius, source_feature, atlas_feature in (
            (48, 6, source_48, atlas_48),
            (96, 4, source_96, atlas_96),
            (192, 3, source_192, atlas_192),
        ):
            base = (source_feature.new_zeros(batch, 2, size, size) if flow is None else
                    2 * F.interpolate(flow, (size, size), mode='bilinear', align_corners=False))
            support = F.interpolate(support_192, (size, size), mode='area')
            logits, candidate_support, flow, posterior = self._match(
                source_feature, atlas_feature, support, base, size, radius)
            levels[size] = {'logits': logits, 'base_flow': base,
                            'flow': flow, 'support': candidate_support}
        yy, xx = torch.meshgrid(torch.arange(side, device=flow.device, dtype=flow.dtype),
                                torch.arange(side, device=flow.device, dtype=flow.dtype), indexing='ij')
        grid = torch.stack((2 * (xx + flow[:, 0]) / (side - 1) - 1,
                            2 * (yy + flow[:, 1]) / (side - 1) - 1), -1)
        matched_atlas = F.grid_sample(atlas_192, grid, padding_mode='zeros', align_corners=True)
        best = posterior.amax(1, keepdim=True)
        selected_support = (posterior * candidate_support).sum(1, keepdim=True)
        evidence = torch.cat((source_192, matched_atlas,
                              (source_192 - matched_atlas).abs(), best,
                              selected_support, source_image[:, :1], flow / side), 1)
        detail = self.detail(evidence)
        canonical_flow = torch.stack((flow[:, 0] * torch.where(reflected, -1., 1.), flow[:, 1]), 1) / side
        inplane = torch.einsum('bij,bjhw->bihw', basis, canonical_flow)
        inplane = inplane * (1500 / inplane.norm(dim=1, keepdim=True).clamp_min(1500))
        local = torch.cat((inplane, 250 * detail[:, :1].tanh()), 1)
        surface = rigid + torch.einsum('bij,bjhw->bhwi', frame, local)
        valid_candidate = (candidate_support > .9).any(1, keepdim=True)
        return {'centre_surface_ccf_ap_dv_ml_um': surface[:, None],
                'rigid_surface_ccf_ap_dv_ml_um': rigid,
                'local_displacement_um': local,
                'correspondence_logit': detail[:, 1:2].masked_fill(~valid_candidate, -20),
                'levels': levels}

    def forward(self, inputs, offsets, weights, atlas, state=None, reflection=None):
        if state is None:
            with torch.no_grad():
                pose = self.pose_model.predict(inputs)
                reflected_log_mass = torch.stack((F.logsigmoid(-pose['reflection_logit']),
                                                  F.logsigmoid(pose['reflection_logit'])), -1)
                choice = (pose['log_mass'][..., None] + reflected_log_mass).flatten(1).argmax(-1)
                state = pose['state'][torch.arange(len(inputs), device=inputs.device), choice // 2]
                reflection = choice % 2
        return self.map_at_pose(inputs, state, reflection, offsets, weights, atlas)
