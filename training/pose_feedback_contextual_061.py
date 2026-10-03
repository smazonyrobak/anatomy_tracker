"""Candidate-conditioned, bidirectional image/atlas context before 3D matching."""
import torch
import torch.nn.functional as F
from torch import nn

from training.pose_feedback_global_041 import PoseFeedbackGlobal041


class ContextualMatcher061(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_position = nn.Linear(2, 32)
        self.atlas_position = nn.Linear(3, 32)
        self.image_from_atlas = nn.MultiheadAttention(32, 4, batch_first=True)
        self.atlas_from_image = nn.MultiheadAttention(32, 4, batch_first=True)
        self.image_norm = nn.LayerNorm(32)
        self.atlas_norm = nn.LayerNorm(32)
        self.image_gain = nn.Parameter(torch.zeros(()))
        self.atlas_gain = nn.Parameter(torch.zeros(()))

    def forward(self, image, atlas, image_chart):
        n = image.shape[0]
        atlas_coarse = F.adaptive_avg_pool3d(
            atlas.transpose(1, 2).reshape(n, 32, 13, 24, 24), (7, 8, 8))
        atlas_coarse = atlas_coarse.flatten(2).transpose(1, 2)
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, 7, device=atlas.device, dtype=atlas.dtype),
            torch.linspace(-1, 1, 8, device=atlas.device, dtype=atlas.dtype),
            torch.linspace(-1, 1, 8, device=atlas.device, dtype=atlas.dtype), indexing='ij')
        atlas_chart = torch.stack((x, y, z), -1).reshape(1, -1, 3)
        image_tokens = image + self.image_position(2 * image_chart - 1)
        atlas_tokens = atlas_coarse + self.atlas_position(atlas_chart)
        image_context = self.image_from_atlas(
            image_tokens, atlas_tokens, atlas_tokens, need_weights=False)[0]
        atlas_context = self.atlas_from_image(
            atlas_tokens, image_tokens, image_tokens, need_weights=False)[0]
        atlas_context = F.interpolate(
            self.atlas_norm(atlas_context).transpose(1, 2).reshape(n, 32, 7, 8, 8),
            (13, 24, 24), mode='trilinear', align_corners=False)
        return (image + self.image_gain * self.image_norm(image_context),
                atlas + self.atlas_gain * atlas_context.flatten(2).transpose(1, 2))


class PoseFeedbackContextual061(PoseFeedbackGlobal041):
    def __init__(self):
        super().__init__()
        self.contextual = ContextualMatcher061()
        self.quality = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(256, 1))
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)

    def contextualize(self, query, key, query_chart):
        return self.contextual(query, key, query_chart)

    def forward(self, prediction, inputs, state, reflection, atlas, offsets, weights):
        result = super().forward(prediction, inputs, state, reflection, atlas, offsets, weights)
        batch, candidates = state.shape[:2]
        residual = (result['matched_world'] - self._plane_points(
            result['state'], reflection)).norm(dim=-1) / 6000
        spatial = torch.stack((result['logits'].amax(-1) / 30,
                               result['match_entropy'], result['confidence'], residual), 2)
        result['score_delta'] = result['score_delta'] + self.quality(
            spatial.reshape(batch * candidates, 4, 16, 16)).reshape(batch, candidates)
        return result
