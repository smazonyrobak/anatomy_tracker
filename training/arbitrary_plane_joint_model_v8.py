"""Direct pose plus recurrent pose/deformation fitting; no imported app weights."""
import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    compose_full_frame_state, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_joint_model_v7 import JointSliceModel, anatomical_cost
from training.arbitrary_plane_recurrent_model import local_correlation
from training.arbitrary_plane_ribbon_v6 import compose_curved_ribbon_coordinates


class JointSliceFeedbackModel(JointSliceModel):
    """At each iteration compare with a newly rendered atlas plane, then revise pose.

    The first two iterations forbid deformation so it cannot hide a bad global
    pose. Later iterations update pose and a constrained curved ribbon jointly.
    Candidate quality is learned from anatomical evidence, not a calibrated
    posterior or a guarantee that every visible structure has been matched.
    """
    def __init__(self, modes=8, uncertainty_rank=4):
        super().__init__(modes=modes, uncertainty_rank=uncertainty_rank)
        self.pose_residual = nn.Sequential(nn.Linear(66, 128), nn.GELU(), nn.Linear(128, 9))
        self.fit_quality = nn.Sequential(nn.Linear(66, 128), nn.GELU(), nn.Linear(128, 1))
        self.register_buffer('pose_update_limits', torch.tensor([
            .25, .25, .25, 800., 800., 800., .08, .08, .08,
        ]))
        nn.init.zeros_(self.pose_residual[-1].weight)
        nn.init.zeros_(self.pose_residual[-1].bias)
        nn.init.zeros_(self.fit_quality[-1].weight)
        nn.init.zeros_(self.fit_quality[-1].bias)

    def fit(self, prediction, atlas, offsets, weights, mode_index, reflection,
            steps=4, initial_state=None):
        batch, count = mode_index.shape
        row = torch.arange(batch, device=mode_index.device)[:, None]
        state = prediction['state'][row, mode_index] if initial_state is None else initial_state
        state = state.reshape(-1, 12)
        reflection = reflection.reshape(-1).bool()
        height, width = prediction['appearance'].shape[-2:]
        source = prediction['feature'][:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        hidden = source.new_zeros(len(source), 64, *source.shape[-2:])
        local = source.new_zeros(len(source), 6, height, width)
        z = offsets[:, None].expand(-1, count, -1).flatten(0, 1)
        w = weights[:, None].expand(-1, count, -1).flatten(0, 1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, source.shape[-2], device=source.device),
                                torch.linspace(-1, 1, source.shape[-1], device=source.device), indexing='ij')
        xy = torch.stack((xx, yy))[None].expand(len(source), -1, -1, -1)
        appearance = prediction['appearance'][:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        states, updates = [], []
        for iteration in range(steps + 1):
            geometry = compose_curved_ribbon_coordinates(state, local[:, :3], local[:, 3:], z)
            coordinates = geometry['ccf_coordinates_ap_dv_ml_um']
            observed_coordinates = torch.where(reflection[:, None, None, None, None],
                                               coordinates.flip(-2), coordinates)
            rendered = render_finite_thickness_coordinate_grid(
                atlas, observed_coordinates, (0., 0., 0.), (25., 25., 25.), w)
            mismatch = anatomical_cost(appearance, rendered[:, :1])
            difficulty = (geometry['residual_local_um'] / 200).square().mean((1, 2, 3))
            difficulty = difficulty + (geometry['director_delta_local'] / .1).square().mean((1, 2, 3))
            difficulty = difficulty + (geometry['prelimit_derivative_frobenius_bound'] / .35).square()
            states.append(state)
            if iteration == steps:
                break
            target = self.atlas_encoder(rendered[:, :1])
            evidence = self.pair(torch.cat((source, target, (source - target).abs(),
                                            local_correlation(source, target, 2), xy), 1))
            hidden = self.fitter(evidence, hidden)
            feedback = torch.cat((hidden.mean((-2, -1)), mismatch[:, None], difficulty[:, None]), -1)
            update = torch.tanh(self.pose_residual(feedback)) * self.pose_update_limits
            state = compose_full_frame_state(state, update)
            updates.append(update)
            if iteration >= 1:
                raw = F.interpolate(self.field(hidden), (height, width), mode='bilinear', align_corners=False)
                raw = torch.where(reflection[:, None, None, None], raw.flip(-1), raw)
                local = torch.cat((1000 * raw[:, :3].tanh(), .2 * raw[:, 3:].tanh()), 1)
        feedback = torch.cat((hidden.mean((-2, -1)), mismatch[:, None], difficulty[:, None]), -1)
        covariance = self.uncertainty(hidden.mean((-2, -1))).reshape(len(hidden), 393, 1 + self.uncertainty_rank)
        return {'coordinates': coordinates, 'rendered': rendered, 'geometry': geometry,
                'state': state, 'pose_sequence': torch.stack(states, 1),
                'pose_updates': torch.stack(updates, 1), 'requested_fields': local,
                'mismatch': mismatch.reshape(batch, count),
                'difficulty': difficulty.reshape(batch, count),
                'fit_quality': self.fit_quality(feedback).reshape(batch, count),
                'joint_std': .001 + F.softplus(covariance[..., 0]),
                'joint_factor': covariance[..., 1:], 'calibrated': False}
