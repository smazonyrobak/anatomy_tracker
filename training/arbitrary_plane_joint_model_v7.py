"""Direct probabilistic pose -> atlas extraction -> fitting, in one gradient graph.

No catalogue, pretrained encoder, fixed pose initialization, or detached pose.
Physical coordinates are Allen AP/DV/ML micrometres. Distributions are NOT
calibrated until separately measured on animal-held-out reference alignments.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_recurrent_model import ConvGRUCell, local_correlation
from training.arbitrary_plane_ribbon_v6 import compose_curved_ribbon_coordinates


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.GroupNorm(8, channels), nn.GELU(),
                                    nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.GroupNorm(8, channels))

    def forward(self, x):
        return F.gelu(x + self.layers(x))


class JointSliceModel(nn.Module):
    """Eight continuous pose modes and a shared recurrent anatomical fitter.

    Inputs: image, optional brush, brush availability, mark heatmap, mark
    availability. No thresholding/segmentation/cropping is performed here.
    Context12: AP lower/upper /10000 and available; entry AP/ML /10000,
    radius /10000 and available; probe unit direction3, cone cosine, available.
    Probe direction is NOT interpreted as the section normal. Context is an
    optional learned condition, not a guarantee that hard bounds are satisfied.
    """
    def __init__(self, modes=8, uncertainty_rank=4):
        super().__init__()
        self.modes, self.uncertainty_rank = modes, uncertainty_rank
        self.encoder = nn.ModuleList()
        previous = 5
        for width in (32, 64, 128, 192):
            self.encoder.append(nn.Sequential(nn.Conv2d(previous, width, 5, stride=2, padding=2),
                                             nn.GroupNorm(8, width), nn.GELU(),
                                             ResidualBlock(width), ResidualBlock(width)))
            previous = width
        self.pose = nn.Sequential(nn.Linear(192 * 4 * 4 + 12, 512), nn.GELU(),
                                  nn.Linear(512, modes * 21))
        self.lateral = nn.ModuleList([nn.Conv2d(w, 32, 1) for w in (32, 64, 128, 192)])
        self.image_decoder = nn.Sequential(ResidualBlock(32), nn.Conv2d(32, 1, 3, padding=1))
        self.atlas_encoder = nn.Sequential(nn.Conv2d(1, 32, 5, stride=2, padding=2),
                                           nn.GroupNorm(8, 32), nn.GELU(), ResidualBlock(32))
        self.pair = nn.Sequential(nn.Conv2d(32 * 3 + 25 + 2, 64, 3, padding=1),
                                  nn.GroupNorm(8, 64), nn.GELU(), ResidualBlock(64))
        self.fitter = ConvGRUCell(64, 64)
        self.field = nn.Conv2d(64, 6, 3, padding=1)
        # A shared low-rank latent couples local pose9 and a 6x8x8 field grid.
        # Returned in normalized units: radians/.1, translation/500um,
        # log-span/shear/.1, surface/200um, director/.1.
        self.uncertainty = nn.Linear(64, (9 + 6 * 8 * 8) * (1 + uncertainty_rank))
        self.register_buffer('center_origin', torch.tensor([6600., 4000., 5700.]))
        self.register_buffer('center_scale', torch.tensor([6600., 4000., 5700.]))
        self.register_buffer('euclidean_scale', torch.tensor([6600., 4000., 5700., 1., 1., 1.]))
        self.register_buffer('rotation_grid', torch.linspace(0, math.pi, 257))
        nn.init.normal_(self.pose[-1].weight, std=.001)
        nn.init.zeros_(self.pose[-1].bias)
        nn.init.normal_(self.field.weight, std=1e-4)
        nn.init.zeros_(self.field.bias)
        nn.init.normal_(self.uncertainty.weight, std=.001)
        nn.init.zeros_(self.uncertainty.bias)
        with torch.no_grad():
            # Random independent proper frames prevent identical mixture starts.
            rotations, _ = torch.linalg.qr(torch.randn(modes, 3, 3))
            rotations[:, :, 2] *= torch.linalg.det(rotations)[:, None]
            self.pose[-1].bias.view(modes, 21)[:, 3:9].copy_(
                rotations[:, :, :2].transpose(1, 2).reshape(modes, 6))

    def predict(self, inputs, context=None):
        if context is None:
            context = inputs.new_zeros(len(inputs), 12)
        pyramid, x = [], inputs
        for layer in self.encoder:
            x = layer(x)
            pyramid.append(x)
        z = self.pose(torch.cat((F.adaptive_avg_pool2d(x, 4).flatten(1), context), -1))
        z = z.reshape(-1, self.modes, 21)
        center = self.center_origin + z[..., :3] * self.center_scale
        state = torch.cat((center, z[..., 3:9],
                           math.log(12000.) + z[..., 9:11].clamp(-2.5, 2.),
                           z[..., 11:12]), -1)
        std = (.005 + F.softplus(z[..., 12:18])).clamp_max(4.)
        concentration = (.05 + F.softplus(z[..., 18])).clamp_max(500.)
        feature = self.lateral[-1](pyramid[-1])
        for level in (2, 1, 0):
            feature = self.lateral[level](pyramid[level]) + F.interpolate(
                feature, size=pyramid[level].shape[-2:], mode='bilinear', align_corners=False)
        appearance = torch.sigmoid(F.interpolate(self.image_decoder(feature),
                                                size=inputs.shape[-2:], mode='bilinear', align_corners=False))
        return {'state': state, 'std': std, 'concentration': concentration,
                'log_mass': z[..., 19].log_softmax(-1), 'reflection_logit': z[..., 20],
                'feature': feature, 'appearance': appearance}

    def component_log_prob(self, prediction, target_state, reflection):
        """Density w.r.t. normalized center/log-basis Lebesgue x Haar(SO3) x R2.

        Each finite-frame orientation uses an isotropic matrix-Fisher density;
        the mixture and separate reflection retain multimodal plane ambiguity.
        The Haar normalizer is a differentiable 1-D quadrature, not Euler noise.
        """
        state = prediction['state']
        euclidean = torch.cat((state[..., :3], state[..., 9:]), -1) / self.euclidean_scale
        truth = torch.cat((target_state[..., :3], target_state[..., 9:]), -1) / self.euclidean_scale
        std = prediction['std']
        normal_lp = (-.5 * ((euclidean - truth[:, None]) / std).square()
                     - std.log() - .5 * math.log(2 * math.pi)).sum(-1)
        frame = full_frame_state_to_components(state)[1]
        target_frame = full_frame_state_to_components(target_state)[1]
        trace = (frame * target_frame[:, None]).sum((-2, -1)).clamp(-1., 3.)
        kappa = prediction['concentration']
        angles = self.rotation_grid
        integrand = torch.exp(2 * kappa[..., None] * (angles.cos() - 1)) * (angles / 2).sin().square()
        log_z = ((2 / math.pi) * torch.trapezoid(integrand, angles, dim=-1)).log()
        rotation_lp = kappa * (trace - 3) - log_z
        reflection_lp = -F.binary_cross_entropy_with_logits(
            prediction['reflection_logit'], reflection[:, None].expand_as(kappa).float(), reduction='none')
        return prediction['log_mass'] + normal_lp + rotation_lp + reflection_lp

    def fit(self, prediction, atlas, offsets, weights, mode_index, reflection,
            steps=3, initial_state=None):
        """Fit selected modes; gradients remain connected to their direct means.

        initial_state is ONLY for supervised fitter warm-up, never inference.
        Reflection reverses the observed raster once; all fields stay canonical.
        """
        batch, count = mode_index.shape
        row = torch.arange(batch, device=mode_index.device)[:, None]
        state = prediction['state'][row, mode_index] if initial_state is None else initial_state
        state = state.reshape(-1, 12)
        reflection = reflection.reshape(-1).bool()
        height, width = prediction['appearance'].shape[-2:]
        source = prediction['feature'][:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        shape = source.shape[-2:]
        hidden = source.new_zeros(len(source), 64, *shape)
        local = source.new_zeros(len(source), 6, height, width)
        z = offsets[:, None].expand(-1, count, -1).flatten(0, 1)
        w = weights[:, None].expand(-1, count, -1).flatten(0, 1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, shape[0], device=source.device),
                                torch.linspace(-1, 1, shape[1], device=source.device), indexing='ij')
        xy = torch.stack((xx, yy))[None].expand(len(source), -1, -1, -1)
        for iteration in range(steps + 1):
            geometry = compose_curved_ribbon_coordinates(state, local[:, :3], local[:, 3:], z)
            coordinates = geometry['ccf_coordinates_ap_dv_ml_um']
            coordinates = torch.where(reflection[:, None, None, None, None], coordinates.flip(-2), coordinates)
            rendered = render_finite_thickness_coordinate_grid(atlas, coordinates, (0., 0., 0.), (25., 25., 25.), w)
            if iteration == steps:
                break
            target = self.atlas_encoder(rendered[:, :1])
            evidence = self.pair(torch.cat((source, target, (source - target).abs(),
                                           local_correlation(source, target, 2), xy), 1))
            hidden = self.fitter(evidence, hidden)
            # Absolute fields prevent uncontrolled composition across iterations.
            raw = F.interpolate(self.field(hidden), (height, width), mode='bilinear', align_corners=False)
            raw = torch.where(reflection[:, None, None, None], raw.flip(-1), raw)
            local = torch.cat((1000 * raw[:, :3].tanh(), .2 * raw[:, 3:].tanh()), 1)
        appearance = prediction['appearance'][:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        mismatch = anatomical_cost(appearance, rendered[:, :1])
        # Penalize requested strain too: hitting the limiter must not be free.
        difficulty = (geometry['residual_local_um'] / 200).square().mean((1, 2, 3))
        difficulty = difficulty + (geometry['director_delta_local'] / .1).square().mean((1, 2, 3))
        difficulty = difficulty + (geometry['prelimit_derivative_frobenius_bound'] / .35).square()
        covariance = self.uncertainty(hidden.mean((-2, -1))).reshape(len(hidden), 393, 1 + self.uncertainty_rank)
        return {'coordinates': coordinates, 'rendered': rendered, 'geometry': geometry,
                'state': state, 'requested_fields': local, 'mismatch': mismatch.reshape(batch, count),
                'difficulty': difficulty.reshape(batch, count),
                'joint_std': .001 + F.softplus(covariance[..., 0]),
                'joint_factor': covariance[..., 1:], 'calibrated': False}

    def forward(self, inputs, atlas, offsets, weights, context=None, steps=3):
        prediction = self.predict(inputs, context)
        index = torch.arange(self.modes, device=inputs.device).repeat_interleave(2)[None].expand(len(inputs), -1)
        reflection = torch.arange(2, device=inputs.device).repeat(self.modes)[None].expand(len(inputs), -1)
        fitted = self.fit(prediction, atlas, offsets, weights, index, reflection, steps)
        log_r = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                             F.logsigmoid(prediction['reflection_logit'])), -1)
        prior = (prediction['log_mass'][..., None] + log_r).flatten(1)
        # This score is an uncalibrated energy, not a measured posterior likelihood.
        fitted['component_log_weight'] = (prior - fitted['mismatch'] - .05 * fitted['difficulty']).log_softmax(-1)
        fitted['prediction'] = prediction
        return fitted


def anatomical_cost(source, target):
    """Multi-scale local contrast agreement, with no inferred tissue masking."""
    cost = source.new_zeros(len(source))
    for scale in (1, 2, 4):
        a, b = (F.avg_pool2d(v, scale) if scale > 1 else v for v in (source, target))
        ma, mb = (F.avg_pool2d(v, 7, 1, 3, count_include_pad=False) for v in (a, b))
        va = F.avg_pool2d(a.square(), 7, 1, 3, count_include_pad=False) - ma.square()
        vb = F.avg_pool2d(b.square(), 7, 1, 3, count_include_pad=False) - mb.square()
        cross = F.avg_pool2d(a * b, 7, 1, 3, count_include_pad=False) - ma * mb
        corr = cross / (va.clamp_min(0) * vb.clamp_min(0) + 1e-6).sqrt()
        cost = cost + (1 - corr).mean((1, 2, 3)) / 3
    return cost + .5 * (source - target).square().mean((1, 2, 3))
