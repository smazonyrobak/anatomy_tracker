"""Global-context 3D matching with evidence-gated geometric pose correction."""
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.pose_feedback_3d_039 import PoseFeedback3D039


class GlobalQuery041(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.local = local
        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.norm = nn.LayerNorm(32)

    def forward(self, image):
        local = self.local(image)
        tokens = local.flatten(2).transpose(1, 2)
        global_context = self.attention(tokens, tokens, tokens, need_weights=False)[0]
        return self.norm(tokens + global_context).transpose(1, 2).reshape_as(local)


class GlobalKey041(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.local = local
        self.context = nn.Sequential(
            nn.Conv3d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1),
        )

    def forward(self, atlas):
        local = self.local(atlas)
        coarse = F.adaptive_avg_pool3d(local, (5, 6, 6))
        return local + F.interpolate(self.context(coarse), size=local.shape[-3:],
                                      mode='trilinear', align_corners=False)


class PoseFeedbackGlobal041(PoseFeedback3D039):
    def __init__(self):
        super().__init__()
        self.query = GlobalQuery041(self.query)
        self.key_3d = GlobalKey041(self.key_3d)
        self.gate = nn.Sequential(nn.Linear(326, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -4.)

    def forward(self, prediction, inputs, state, reflection, atlas, offsets, weights):
        result = super().forward(prediction, inputs, state, reflection, atlas, offsets, weights)
        first_centre, first_frame, first_basis = full_frame_state_to_components(state)
        fit_centre, fit_frame, fit_basis = full_frame_state_to_components(result['state'])
        first_edges = first_frame[..., :, :2] @ first_basis
        fit_edges = fit_frame[..., :, :2] @ fit_basis
        fit_residual = (result['matched_world'] - self._plane_points(
            result['state'], reflection)).norm(dim=-1)
        displacement = (result['matched_world'] - result['query_base']).norm(dim=-1)
        visible = result['visibility_logit'].sigmoid()[:, None].expand_as(
            result['confidence'])
        stats = torch.stack((result['match_entropy'].mean(-1),
            result['confidence'].mean(-1), visible.mean(-1),
            (result['confidence'] * fit_residual).sum(-1) /
            result['confidence'].sum(-1).clamp_min(1e-3) / 6000,
            displacement.mean(-1) / 6000,
            (fit_centre - first_centre).norm(dim=-1) / 6000), -1)
        global_image = prediction['global_feature'][:, None].expand(
            -1, state.shape[1], -1)
        gate = self.gate(torch.cat((global_image, stats), -1))[..., 0].sigmoid()
        centre = first_centre + gate[..., None] * (fit_centre - first_centre)
        edges = first_edges + gate[..., None, None] * (fit_edges - first_edges)
        u, v = edges.unbind(-1)
        origin = centre - .5 * (u + v)
        result['raw_fitted_state'] = result['state']
        result['state'] = full_frame_state_from_components(*physical_ouv_to_frame(
            torch.stack((origin, u, v), -2)))
        result['score_delta'] = gate * result['score_delta']
        result['correction_gate'] = gate
        return result

    @staticmethod
    def _plane_points(state, reflection):
        centre, frame, basis = full_frame_state_to_components(state)
        axis = (torch.arange(16, device=state.device, dtype=state.dtype) + .5) / 16
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        chart = torch.stack((x, y), -1).reshape(1, 1, 256, 2).expand(
            state.shape[0], state.shape[1], -1, -1).clone()
        chart[..., 0] = torch.where(reflection[..., None].bool(),
                                     255 / 256 - chart[..., 0], chart[..., 0])
        return centre[..., None, :] + torch.einsum('bkij,bkqj->bkqi',
            frame[..., :, :2] @ basis, chart - .5)
