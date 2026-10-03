"""One-pass 2D-to-3D atlas correspondence and robust full-frame pose fitting."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_joint_model_v7 import ResidualBlock
from training.arbitrary_plane_full_frame_primitives import physical_um_to_allen_index_points


class PoseFeedback3D039(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Sequential(
            nn.Conv2d(69, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            ResidualBlock(32), nn.Conv2d(32, 32, 1),
        )
        self.key_2d = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.GELU(),
            ResidualBlock(32), nn.Conv2d(32, 32, 1),
        )
        self.key_3d = nn.Sequential(
            nn.Conv3d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1),
        )
        self.visibility = nn.Conv2d(32, 1, 1)
        self.ranker = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.ranker[-1].weight)
        nn.init.zeros_(self.ranker[-1].bias)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(12.)))

    def contextualize(self, query, key, query_chart):
        return query, key

    def forward(self, prediction, inputs, state, reflection, atlas, offsets, weights):
        batch, candidates = state.shape[:2]
        centre, frame, basis = full_frame_state_to_components(state)
        edges = frame[..., :, :2] @ basis
        device, dtype = inputs.device, inputs.dtype
        atlas_axis = (torch.arange(24, device=device, dtype=dtype) + .5) / 12 - .5
        ay, ax = torch.meshgrid(atlas_axis, atlas_axis, indexing='ij')
        ax = ax[None, None].expand(batch, candidates, -1, -1)
        ay = ay[None, None].expand_as(ax)
        atlas_chart = torch.stack((torch.where(reflection[..., None, None].bool(),
            255 / 256 - ax, ax), ay), -1)
        atlas_base = centre[..., None, None, :] + torch.einsum(
            'bkij,bkhwj->bkhwi', edges, atlas_chart - .5)
        depth = torch.linspace(-6000., 6000., 13, device=device, dtype=dtype)
        normal = frame[..., :, 2]
        points = atlas_base[:, :, None, None] + (
            depth[None, None, :, None] + offsets[:, None, None, :]
        )[..., None, None, None] * normal[:, :, None, None, None, None, :]
        index = physical_um_to_allen_index_points(points, (0., 0., 0.), (25., 25., 25.))
        ap, dv, ml = atlas.shape[-3:]
        grid = torch.stack((index[..., 2] / (ml - 1) * 2 - 1,
                            index[..., 1] / (dv - 1) * 2 - 1,
                            index[..., 0] / (ap - 1) * 2 - 1), -1)
        sampled = F.grid_sample(atlas[None].expand(batch * candidates, -1, -1, -1, -1),
            grid.reshape(batch * candidates, 13 * offsets.shape[1], 24, 24, 3),
            mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.reshape(batch, candidates, 2, 13, offsets.shape[1], 24, 24)
        stack = (sampled * weights[:, None, None, None, :, None, None]).sum(4)
        key_world = atlas_base[:, :, None] + depth[None, None, :, None, None, None] * \
            normal[:, :, None, None, None, :]
        key_world = key_world.expand(-1, -1, -1, 24, 24, -1).reshape(
            batch, candidates, 13 * 24 * 24, 3)
        key = self.key_2d(stack.permute(0, 1, 3, 2, 4, 5).reshape(
            batch * candidates * 13, 2, 24, 24))
        key = key.reshape(batch * candidates, 13, 32, 24, 24).permute(0, 2, 1, 3, 4)
        key = self.key_3d(key).flatten(2).transpose(1, 2)
        source = torch.cat((F.adaptive_avg_pool2d(prediction['feature'], 16),
                            F.adaptive_avg_pool2d(inputs, 16)), 1)
        query = self.query(source)
        visible_logit = self.visibility(query).flatten(2)[:, 0]
        query = query.flatten(2).transpose(1, 2)[:, None].expand(
            -1, candidates, -1, -1).reshape(batch * candidates, 256, 32)
        query_axis = (torch.arange(16, device=device, dtype=dtype) + .5) / 16
        qy, qx = torch.meshgrid(query_axis, query_axis, indexing='ij')
        qchart = torch.stack((qx, qy), -1).reshape(1, 1, 256, 2).expand(
            batch, candidates, -1, -1).clone()
        qchart[..., 0] = torch.where(reflection[..., None].bool(),
                                     255 / 256 - qchart[..., 0], qchart[..., 0])
        query, key = self.contextualize(query, key, qchart.flatten(0, 1))
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        similarity = self.log_temperature.exp().clamp(5, 30) * query @ key.transpose(1, 2)
        query_base = centre[..., None, :] + torch.einsum(
            'bkij,bkqj->bkqi', edges, qchart - .5)
        geometric = torch.cdist(query_base.reshape(batch * candidates, 256, 3),
                                 key_world.reshape(batch * candidates, -1, 3))
        key_support = stack[:, :, 1].reshape(batch * candidates, -1)
        logits = (similarity - .15 * (geometric / 6000).square()
                  - 4 * (1 - key_support[:, None])).reshape(
                      batch, candidates, 256, -1)
        attention = logits.softmax(-1)
        matched_world = attention @ key_world
        max_match = attention.amax(-1)
        entropy = -(attention * attention.clamp_min(1e-9).log()).sum(-1) / math.log(13 * 24 * 24)
        visible = visible_logit[:, None].expand(-1, candidates, -1).sigmoid()
        confidence = visible * (.25 + .75 * (1 - entropy))
        chart = torch.cat((torch.ones_like(qchart[..., :1]), qchart - .5), -1)
        prior = torch.stack((centre, edges[..., :, 0], edges[..., :, 1]), -2)
        ridge = torch.diag(state.new_tensor((2., 1., 1.)))

        def solve(confidence):
            lhs = torch.einsum('bkqi,bkq,bkqj->bkij', chart, confidence, chart) + ridge
            rhs = torch.einsum('bkqi,bkq,bkqj->bkij', chart, confidence, matched_world) \
                  + ridge @ prior
            return torch.linalg.solve(lhs, rhs)

        fitted = solve(confidence)
        residual = (chart @ fitted - matched_world).norm(dim=-1)
        robust = (1500 / residual.clamp_min(1500)).detach()
        fitted = solve(confidence * robust)
        center, u, v = fitted.unbind(-2)
        origin = center - .5 * (u + v)
        refined = full_frame_state_from_components(*physical_ouv_to_frame(
            torch.stack((origin, u, v), -2)))
        residual = (chart @ fitted - matched_world).norm(dim=-1)
        movement = (center - centre).norm(dim=-1)
        score_input = torch.stack((max_match.mean(-1), entropy.mean(-1), visible.mean(-1),
            confidence.mean(-1), (confidence * residual).sum(-1) /
            confidence.sum(-1).clamp_min(1e-3) / 6000,
            movement / 6000, (u - edges[..., :, 0]).norm(dim=-1) / 12000,
            (v - edges[..., :, 1]).norm(dim=-1) / 12000), -1)
        score_delta = self.ranker(score_input)[..., 0]
        return {'state': refined, 'score_delta': score_delta, 'logits': logits,
                'matched_world': matched_world, 'key_world': key_world,
                'key_support': key_support.reshape(batch, candidates, -1),
                'visibility_logit': visible_logit, 'confidence': confidence,
                'query_base': query_base, 'match_entropy': entropy}
