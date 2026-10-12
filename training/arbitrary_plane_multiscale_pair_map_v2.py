"""Gated multiscale tissue-to-atlas correspondence at an arbitrary rigid plane."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_multiscale_pair_map_v1 import MultiScalePairJointModel


class GatedMultiScalePairJointModel(MultiScalePairJointModel):
    def __init__(self):
        super().__init__()
        self.move_gate_head = nn.Sequential(
            nn.Conv2d(14, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.zeros_(self.move_gate_head[-1].weight)
        nn.init.constant_(self.move_gate_head[-1].bias, -4.)

    def map_at_pose(self, inputs, state, reflection, offsets, weights, atlas):
        result = super().map_at_pose(inputs, state, reflection, offsets, weights, atlas)
        rigid = result['rigid_surface_ccf_ap_dv_ml_um']
        raw_local = result['local_displacement_um']
        _, frame, _ = full_frame_state_to_components(state)
        slab = rigid[:, None] + offsets[:, :, None, None, None] * frame[:, None, None, None, :, 2]
        rendered = render_finite_thickness_coordinate_grid(
            atlas, slab, (0., 0., 0.), (25., 25., 25.), weights)
        support = rendered[:, 1:2]
        atlas_pair = torch.cat((rendered[:, :1] / support.clamp_min(1e-4), support), 1)
        fine = result['levels'][192]
        posterior = fine['logits'].softmax(1)
        peak = posterior.amax(1, keepdim=True)
        entropy = -(posterior * posterior.clamp_min(1e-8).log()).sum(1, keepdim=True)
        entropy = entropy / math.log(posterior.shape[1])
        matched_support = (posterior * fine['support']).sum(1, keepdim=True)
        observed = F.interpolate(inputs, rigid.shape[1:3], mode='area')
        evidence = torch.cat((observed, atlas_pair, raw_local / 1500,
                              peak, entropy, matched_support,
                              result['correspondence_logit'].sigmoid()), 1)
        gate = self.move_gate_head(evidence).sigmoid()
        local = raw_local * gate
        surface = rigid + torch.einsum('bij,bjhw->bhwi', frame, local)
        result.update({
            'raw_centre_surface_ccf_ap_dv_ml_um': result['centre_surface_ccf_ap_dv_ml_um'],
            'raw_local_displacement_um': raw_local,
            'centre_surface_ccf_ap_dv_ml_um': surface[:, None],
            'local_displacement_um': local,
            'move_gate': gate,
        })
        return result
