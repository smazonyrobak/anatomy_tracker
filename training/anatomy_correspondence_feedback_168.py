"""Image-only source/atlas correspondence for blind local pose feedback.

The caller renders the initial atlas plane, composes ``update`` with the plane
state, renders again, and uses only the second comparison's ``score`` for
candidate ranking. No atlas coordinates, support, pose state, or prior enter a
learned path; support only selects valid sites in the final mean.
"""

import torch
import torch.nn.functional as F
from torch import nn

class AnatomyCorrespondenceFeedback(nn.Module):
    def __init__(self, side=48, radius=3, descriptor_channels=32):
        super().__init__()
        self.side, self.radius = side, radius
        self.source_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, padding_mode='replicate'), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'), nn.GELU(),
            nn.Conv2d(32, descriptor_channels, 1),
        )
        self.atlas_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, padding_mode='replicate'), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'), nn.GELU(),
            nn.Conv2d(32, descriptor_channels, 1),
        )
        channels = (2 * radius + 1) ** 2
        self.corrector = nn.Sequential(
            nn.Conv2d(channels + 2, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.GELU(), nn.Linear(128, 9),
        )
        nn.init.zeros_(self.corrector[-1].weight)
        nn.init.zeros_(self.corrector[-1].bias)
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
        dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
        self.register_buffer('offset_cost',
                             (.08 * (dx.square() + dy.square()) / max(radius * radius, 1))
                             .reshape(1, channels, 1, 1))
        self.register_buffer('update_limits', torch.tensor(
            [.3, .3, .3, 1200., 1200., 1200., .16, .16, .12]))

    def encode_source(self, inputs, feature=None):
        # The full five-channel input and backbone feature carry outline/brush
        # information; only the acquired image channel is permitted here.
        image = F.interpolate(inputs[:, :1], (self.side, self.side), mode='area')
        return F.normalize(self.source_encoder(image), dim=1, eps=1e-6)

    def encode_atlas(self, intensity):
        bank = intensity.ndim == 5
        if bank:
            batch, depth = intensity.shape[:2]
            intensity = intensity.flatten(0, 1)
        image = F.interpolate(intensity[:, :1], (self.side, self.side), mode='area')
        descriptor = F.normalize(self.atlas_encoder(image), dim=1, eps=1e-6)
        return descriptor.reshape(batch, depth, *descriptor.shape[1:]) if bank else descriptor

    def compare(self, source_desc, atlas_desc, support, depth_offsets_um=None):
        if atlas_desc.ndim == 4:
            atlas_desc = atlas_desc[:, None]
        if support.ndim == 4:
            support = support[:, None]
        batch, depth, channels, height, width = atlas_desc.shape
        source = source_desc[:, None].expand(-1, depth, -1, -1, -1).reshape(
            batch * depth, channels, height, width)
        work_dtype = torch.float32 if source.dtype in (torch.float16, torch.bfloat16) else source.dtype
        source = F.normalize(source.to(work_dtype), dim=1, eps=1e-6)
        atlas = F.normalize(atlas_desc.reshape(batch * depth, channels, height, width)
                            .to(work_dtype), dim=1, eps=1e-6)
        patches = F.unfold(F.pad(atlas, (self.radius,) * 4, mode='replicate'),
                           2 * self.radius + 1).reshape(batch * depth, channels, -1, height, width)
        correlation = (source[:, :, None] * patches).sum(1).reshape(
            batch, depth, -1, height, width).to(source_desc.dtype)
        if depth_offsets_um is None:
            depth_offsets_um = (torch.arange(depth, device=correlation.device)
                                - (depth - 1) / 2) * 250.
        depth_cost = .08 * torch.as_tensor(depth_offsets_um, device=correlation.device,
                                           dtype=correlation.dtype).abs() / 1000.
        depth_cost = depth_cost.reshape(1, depth, 1, 1, 1) if depth_cost.ndim == 1 else depth_cost[:, :, None, None, None]
        spatial_cost = self.offset_cost.to(correlation)[:, None]
        logits = (correlation - spatial_cost - depth_cost) / .12

        # Peak-over-local-null is zero for constant descriptors or textureless
        # patches; the spatial softmax favors nearby anatomical correspondence.
        centered = correlation - correlation.mean(2, keepdim=True)
        local_prob = ((centered - spatial_cost) / .12).softmax(2)
        match_map = (local_prob * centered).sum(2, keepdim=True)
        valid = F.interpolate(support.detach().flatten(0, 1), (height, width),
                              mode='area').reshape(batch, depth, 1, height, width).clamp(0, 1)
        valid_area = valid.sum((2, 3, 4))
        depth_score = (match_map * valid).sum((2, 3, 4)) / valid_area.clamp_min(1)
        depth_penalty = depth_cost[..., 0, 0, 0]
        depth_logits = ((depth_score - depth_penalty) / .12).masked_fill(valid_area == 0, -1e4)
        null_logits = (-depth_penalty / .12).masked_fill(valid_area == 0, -1e4)
        depth_weight = depth_logits.softmax(1)
        null_weight = null_logits.softmax(1)
        score = (depth_weight * (depth_score - depth_penalty)).sum(1) + (
            null_weight * depth_penalty).sum(1)
        score = torch.where((valid_area > 0).any(1), score, score.new_full(score.shape, -1.))

        # The learned correction sees depth-marginalized image correlation only.
        depth_prob = ((correlation - depth_cost) / .12).softmax(1)
        pooled = (depth_prob * correlation).sum(1)
        pooled = pooled - pooled.mean(1, keepdim=True)
        y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=correlation.device,
                                            dtype=correlation.dtype),
                              torch.linspace(-1, 1, width, device=correlation.device,
                                             dtype=correlation.dtype), indexing='ij')
        xy = torch.stack((x, y))[None].expand(batch, -1, -1, -1)
        update = self.corrector(torch.cat((pooled, xy), 1)).tanh() * self.update_limits.to(correlation)
        return {'correlation': correlation, 'correlation_logits': logits,
                'match_map': match_map, 'depth_score': depth_score,
                'depth_weight': depth_weight, 'score': score, 'update': update}
