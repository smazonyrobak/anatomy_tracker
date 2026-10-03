"""Global image/atlas correspondence update for arbitrary-plane pose candidates."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_joint_model_v7 import ResidualBlock


class PoseFeedbackCorrelation037(nn.Module):
    def __init__(self, use_atlas=True):
        super().__init__()
        self.use_atlas = use_atlas
        self.image_project = nn.Conv2d(64, 32, 1)
        self.atlas_project = nn.Conv2d(64, 32, 1)
        self.head = nn.Sequential(
            nn.Conv2d(161, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            ResidualBlock(128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            ResidualBlock(128), nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(128 * 4 * 4, 10),
        )
        nn.init.dirac_(self.image_project.weight)
        nn.init.dirac_(self.atlas_project.weight)
        nn.init.zeros_(self.image_project.bias)
        nn.init.zeros_(self.atlas_project.bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, prediction, mapped, inputs, state, reflection, atlas_feature):
        batch, count = state.shape[:2]
        source = F.adaptive_avg_pool2d(prediction['feature'], 16)
        source = source[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        image = F.adaptive_avg_pool2d(inputs, 16)
        image = image[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        q = self.image_project(source)
        device, dtype = q.device, q.dtype
        axis = (torch.arange(16, device=device, dtype=dtype) + .5) / 16
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        xy = torch.stack((xx, yy), 0)[None].expand(batch * count, -1, -1, -1)
        target = (F.adaptive_avg_pool2d(atlas_feature.flatten(0, 1), 16)
                  if self.use_atlas else source)
        k = self.atlas_project(target)
        query = F.normalize(q.flatten(2).transpose(1, 2), dim=-1)
        key = F.normalize(k.flatten(2).transpose(1, 2), dim=-1)
        similarity = query @ key.transpose(1, 2)
        attention = (8 * similarity).softmax(-1)
        matched = (attention @ key).transpose(1, 2).reshape(batch * count, 32, 16, 16)
        flow = (attention @ xy.flatten(2).transpose(1, 2)
                - xy.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(batch * count, 2, 16, 16)
        confidence = attention.amax(-1).reshape(batch * count, 1, 16, 16)
        entropy = -(attention * attention.clamp_min(1e-9).log()).sum(-1)
        entropy = (entropy / math.log(256)).reshape(batch * count, 1, 16, 16)
        diagonal = similarity.diagonal(dim1=1, dim2=2).reshape(batch * count, 1, 16, 16)
        if self.use_atlas:
            atlas_pair = F.adaptive_avg_pool2d(mapped['atlas_pair'].flatten(0, 1), 16)
            local_fit = F.adaptive_avg_pool2d(mapped['refinement_feature'][:, :, -6:].flatten(0, 1), 16)
        else:
            atlas_pair = image[:, :2]
            dx = F.pad(image[:, :1, :, 1:] - image[:, :1, :, :-1], (0, 1, 0, 0))
            dy = F.pad(image[:, :1, 1:, :] - image[:, :1, :-1, :], (0, 0, 0, 1))
            local_fit = torch.cat((image, (dx.square() + dy.square() + 1e-8).sqrt()), 1)
        flat = state.flatten(0, 1)
        condition = torch.cat(((flat[:, :3] - flat.new_tensor((6600., 4000., 5700.))) /
                               flat.new_tensor((6600., 4000., 5700.)),
                               flat[:, 3:9], flat[:, 9:11] - math.log(12000.),
                               flat[:, 11:12], reflection.flatten()[:, None].to(dtype)), -1)
        condition = condition[..., None, None].expand(-1, -1, 16, 16)
        evidence = torch.cat((q, k, matched, (q - matched).abs(), flow, confidence,
                              entropy, diagonal, xy, image, atlas_pair, local_fit, condition), 1)
        raw = self.head(evidence).reshape(batch, count, 10)
        limit = raw.new_tensor((1.2, 1.2, 1.2, 5000., 5000., 5000., .35, .35, .25))
        update = raw[..., :9].tanh() * limit
        return compose_full_frame_state(state, update), raw[..., 9], update
