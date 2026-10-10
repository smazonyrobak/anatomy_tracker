"""Shared versus untied image-conditioned corrections of frozen 132 anchor poses."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state


ARMS = ('shared_visual', 'untied_visual', 'shared_source_zero')
# Conservative 072 full-frame update bounds, fixed before any 132 DEV readout.
UPDATE_LIMITS = (.9, .9, .9, 4000., 4000., 4000., .25, .25, .2)


class AnchorResidual138(nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        self.image = nn.Linear(64 * 8 * 8, 256)
        if arm == 'untied_visual':
            self.first = nn.Parameter(torch.empty(64, 277, 192))
            self.first_bias = nn.Parameter(torch.zeros(64, 192))
            self.last = nn.Parameter(torch.zeros(64, 192, 9))
            self.last_bias = nn.Parameter(torch.zeros(64, 9))
            for weight in self.first:
                nn.init.xavier_uniform_(weight)
        else:
            self.first = nn.Linear(277, 192)
            self.last = nn.Linear(192, 9)
            nn.init.zeros_(self.last.weight)
            nn.init.zeros_(self.last.bias)

    def forward(self, prediction, model):
        states = prediction['state'][:, model.base_modes:]
        visual = F.adaptive_avg_pool2d(prediction['feature'], 8).flatten(1)
        if self.arm == 'shared_source_zero':
            visual = torch.zeros_like(visual)
        visual = F.gelu(self.image(visual))[:, None].expand(-1, 64, -1)
        anchor = model.normal_anchor_frames.flatten(1)[None].expand(len(states), -1, -1)
        geometry = torch.cat(((states[..., :3] - model.center_origin) / model.center_scale,
            states[..., 3:9], states[..., 9:11] - math.log(12000.), states[..., 11:]), -1)
        x = torch.cat((visual, anchor, geometry), -1)
        if self.arm == 'untied_visual':
            hidden = F.gelu(torch.einsum('bni,nij->bnj', x, self.first) + self.first_bias)
            raw = torch.einsum('bni,nij->bnj', hidden, self.last) + self.last_bias
        else:
            raw = self.last(F.gelu(self.first(x)))
        update = raw.tanh() * raw.new_tensor(UPDATE_LIMITS)
        zero = torch.zeros_like(update)
        corrected = states + (compose_full_frame_state(states, update)
            - compose_full_frame_state(states, zero))
        return torch.cat((prediction['state'][:, :model.base_modes], corrected), 1)


def corrected_states(prediction, head, model):
    return head(prediction, model)
