"""Local 2-D/3-D learned-descriptor matches in 240's candidate atlas bank."""

import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.global_atlas_contrast_090 import rigid_points_090
from training.local_3d_descriptor_240 import query_chart_240


def local_3d_learned_match(source_features, atlas_features, support, state, reflection):
    query = source_features[:, :, ::2, ::2].flatten(2).transpose(1, 2)[0]
    chart = query_chart_240(source_features.device, source_features.dtype)
    qy, qx = torch.meshgrid(torch.arange(0, 32, 2, device=query.device),
                            torch.arange(0, 32, 2, device=query.device), indexing='ij')
    qy, qx = qy.flatten(), qx.flatten()
    dy, dx = torch.meshgrid(torch.arange(-6, 7, device=query.device),
                            torch.arange(-6, 7, device=query.device), indexing='ij')
    dy, dx = dy.flatten(), dx.flatten()
    key_y = qy[:, None] + 6 + dy[None]
    key_x = qx[:, None] + 6 + dx[None]
    key_index = key_y * 44 + key_x
    key = atlas_features[0].flatten(2)[:, :, key_index.flatten()]
    key = key.reshape(64, 17, 256, 169).permute(2, 1, 3, 0)
    cosine = torch.einsum('qc,qdkc->qdk', query, key)
    available = (support[0, 0].flatten(1)[:, key_index.flatten()] > .5)
    available = available.reshape(17, 256, 169).permute(1, 0, 2)
    depth = torch.linspace(-4000, 4000, 17, device=query.device, dtype=query.dtype)
    cost = .04 * (dx.square() + dy.square()).to(query.dtype) / 36
    cost = cost[None] + .04 * depth.abs()[:, None] / 2500
    scores = (cosine - cost[None]).flatten(1).masked_fill(~available.flatten(1), -torch.inf)
    matched = scores.argmax(-1)
    supported = available.flatten(1).any(-1)
    best = scores.gather(1, matched[:, None])[:, 0]
    best = torch.where(supported, best, torch.zeros_like(best))
    raw_score = cosine.flatten(1).gather(1, matched[:, None])[:, 0]
    raw_score = torch.where(supported, raw_score, torch.zeros_like(raw_score))
    median = cosine.flatten(1).masked_fill(~available.flatten(1), torch.nan).nanmedian(-1).values
    confidence = (raw_score - torch.nan_to_num(median, nan=0)).clamp_min(0) * raw_score.clamp_min(0)
    confidence = torch.where(supported, confidence, torch.zeros_like(confidence))

    depth_index, local_index = matched.div(169, rounding_mode='floor'), matched.remainder(169)
    pixel_index = key_index[torch.arange(256, device=query.device), local_index]
    axis = (torch.arange(44, device=query.device, dtype=query.dtype) - 5.5) / 32 - .5 / 256
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    bank_chart = torch.stack((xx, yy), -1).reshape(-1, 2)
    plane = rigid_points_090(state, reflection, bank_chart)[0]
    normal = full_frame_state_to_components(state)[1][0, :, 2]
    matched_world = plane[pixel_index] + depth[depth_index, None] * normal
    query_world = rigid_points_090(state, reflection, chart)[0]
    matched_world = torch.where(supported[:, None], matched_world, query_world)
    return {'matched_world': matched_world, 'query_chart': chart,
            'query_world': query_world, 'score': best, 'raw_score': raw_score,
            'confidence': confidence, 'support': supported}
