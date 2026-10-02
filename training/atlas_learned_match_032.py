"""Contextual cross-contrast match selection and differentiable plane fitting."""
import torch
from torch import nn


class AtlasLearnedMatch032(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(269, 128), nn.LayerNorm(128), nn.GELU())
        self.context = nn.Sequential(*(nn.TransformerEncoderLayer(
            128, 4, 256, dropout=0., batch_first=True, norm_first=True) for _ in range(3)))
        self.match = nn.Linear(128, 1)
        self.no_match = nn.Linear(128, 1)
        self.register_buffer('ccf_origin', torch.tensor([6600., 4000., 5700.]))
        self.register_buffer('ccf_scale', torch.tensor([6600., 4000., 5700.]))

    def forward(self, image_feature, atlas_feature, similarity, pixel_yx, position_um,
                axis_u, axis_v, rank):
        batch, queries, matches = similarity.shape
        xy = pixel_yx.flip(-1) / 256 - .5
        token = torch.cat((image_feature, atlas_feature, xy[:, :, None].expand(-1, -1, matches, -1),
                           (position_um - self.ccf_origin) / self.ccf_scale,
                           axis_u, axis_v, similarity[..., None], rank[..., None]), -1)
        feature = self.context(self.project(token).reshape(batch, queries * matches, 128))
        feature = feature.reshape(batch, queries, matches, 128)
        return torch.cat((self.match(feature).squeeze(-1),
                          self.no_match(feature.mean(2))), -1)

    def fit_plane(self, logits, pixel_yx, position_um, prior_coefficients=None):
        batch, queries, matches = position_um.shape[:3]
        probability = logits.softmax(-1)[..., :matches]
        xy = pixel_yx.flip(-1) / 256 - .5
        design = torch.cat((torch.ones_like(xy[..., :1]), xy), -1)
        design = design[:, :, None].expand(-1, -1, matches, -1).reshape(batch, -1, 3)
        weights = probability.reshape(batch, -1)
        target = ((position_um - self.ccf_origin) / self.ccf_scale).reshape(batch, -1, 3)
        gram = torch.einsum('bni,bn,bnj->bij', design, weights, design)
        right = torch.einsum('bni,bn,bnj->bij', design, weights, target)
        ridge = torch.diag(logits.new_tensor([.01, .001, .001]))[None]
        if prior_coefficients is not None:
            right = right + ridge @ prior_coefficients
        coefficients = torch.linalg.solve(gram + ridge, right)
        return torch.cat(((self.ccf_origin + coefficients[:, 0] * self.ccf_scale)[:, None],
                          coefficients[:, 1:] * self.ccf_scale[None, None]), 1)
